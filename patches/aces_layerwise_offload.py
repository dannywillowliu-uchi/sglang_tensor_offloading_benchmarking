import re
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Adapted from skywork AI Infra diffusion optimize
class LayerwiseOffloadManager:
	"""A lightweight layerwise CPU offload manager.

	This utility offloads per-layer parameters/buffers from GPU to CPU, and
	supports async H2D prefetch using a dedicated CUDA stream.

	Typical usage:
	- Construct the manager with the target model and the list-like module
	  attribute that represents transformer blocks (e.g. ``blocks``).
	- Call :meth:`initialize` once to offload weights and prefetch layer 0.
	- During forward, call :meth:`prefetch_layer` for the next layer and
	  :meth:`release_layer` for the finished layer.
	"""

	def __init__(
		self,
		model: torch.nn.Module,
		*,
		layers_attr_str: str,
		num_layers: int,
		enabled: bool,
		pin_cpu_memory: bool = True,
		sharded: bool = False,
	) -> None:
		self.model = model
		self.layers_attr_str = layers_attr_str
		self.num_layers = num_layers
		self.pin_cpu_memory = pin_cpu_memory

		self.enabled = bool(enabled and torch.cuda.is_available())
		if not self.enabled:
			return

		# Sharded offload: each GPU copies 1/N of the layer, then all-gathers
		self.sharded = False
		self._pending_shards: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
		self._shard_info: Dict[int, Dict[torch.dtype, Tuple[int, int]]] = {}
		if sharded:
			if dist.is_initialized() and dist.get_world_size() > 1:
				self.sharded = True
				self.rank = dist.get_rank()
				self.world_size = dist.get_world_size()
			else:
				logger.warning(
					"sharded=True but torch.distributed not initialized or world_size=1; "
					"falling back to non-sharded offload."
				)

		self.device = torch.device("cuda", torch.cuda.current_device())
		self.copy_stream = torch.cuda.Stream()

		self._layer_name_re = re.compile(
			rf"(^|\.){re.escape(layers_attr_str)}\.(\d+)(\.|$)"
		)

		# layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
		# stores the consolidated weight from a same layer, of same dtype
		self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
		# layer_idx -> {name: {dtype, offset, numel, shape}}
		# stores the offset and numel of each weight from a same layer, of same dtype
		self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
		# layer indices that are already in gpu
		self._gpu_layers: Set[int] = set()
		# layer_idx -> torch.cuda.Event for fine-grained sync
		self._prefetch_events: Dict[int, torch.cuda.Event] = {}

		self._named_parameters: Dict[str, torch.nn.Parameter] = {}
		self._named_buffers: Dict[str, torch.Tensor] = {}
		# Store forward hooks for removal
		self._forward_hooks: List[Any] = []

		self._initialize()

	def _match_layer_idx(self, name: str) -> int | None:
		m = self._layer_name_re.search(name)
		if not m:
			return None
		try:
			return int(m.group(2))
		except Exception:
			return None

	@torch.compiler.disable
	def _initialize(self) -> None:
		if not self.enabled:
			return

		self._named_parameters = dict(self.model.named_parameters())
		self._named_buffers = dict(self.model.named_buffers())

		# 1. collect and group tensors by layer and dtype
		layer_groups: Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]] = {}
		all_tensors = chain(self._named_parameters.items(), self._named_buffers.items())
		for name, tensor in all_tensors:
			layer_idx = self._match_layer_idx(name)
			if layer_idx is None or layer_idx >= self.num_layers:
				continue
			layer_groups.setdefault(layer_idx, {}).setdefault(tensor.dtype, []).append(
				(name, tensor)
			)

		# 2. concat and offload (in pinned memory)
		for layer_idx, dtype_to_params in layer_groups.items():
			self._consolidated_cpu_weights[layer_idx] = {}
			self._weight_metadata[layer_idx] = {}

			for dtype, weights in dtype_to_params.items():
				total_numel = sum(t.numel() for _, t in weights)

				# create concatenated CPU buffer (in pinned memory)
				cpu_buffer = torch.empty(
					total_numel, dtype=dtype, pin_memory=self.pin_cpu_memory
				)

				# offload weights to the buffer
				current_offset = 0
				for name, weight in weights:
					numel = weight.numel()
					cpu_buffer[current_offset : current_offset + numel].copy_(
						weight.flatten()
					)
					self._weight_metadata[layer_idx][name] = {
						"dtype": dtype,
						"offset": current_offset,
						"numel": numel,
						"shape": weight.shape,
					}

					weight.data = torch.empty((1,), device=self.device, dtype=dtype)

					current_offset += numel

				self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

				# Pad buffer for even sharding across ranks
				if self.sharded:
					shard_numel = (total_numel + self.world_size - 1) // self.world_size
					padded_numel = shard_numel * self.world_size
					if padded_numel > total_numel:
						padded_buffer = torch.zeros(
							padded_numel, dtype=dtype, pin_memory=self.pin_cpu_memory
						)
						padded_buffer[:total_numel].copy_(cpu_buffer)
						self._consolidated_cpu_weights[layer_idx][dtype] = padded_buffer
					self._shard_info.setdefault(layer_idx, {})[dtype] = (shard_numel, padded_numel)

		# prefetch the first layer for warm-up
		self.prepare_for_next_denoise(non_blocking=False)

		self.register_forward_hooks()
		extra_msg = ""
		if self.sharded:
			extra_msg = f", sharded=True (rank={self.rank}, world_size={self.world_size})"
		logger.info(
			f"LayerwiseOffloadManager initialized, num_layers={self.num_layers}{extra_msg}"
		)

	def prepare_for_next_denoise(self, non_blocking=True):
		self.prefetch_layer(0, non_blocking=non_blocking)
		if not non_blocking and self.copy_stream is not None:
			torch.cuda.current_stream().wait_stream(self.copy_stream)
			if self.sharded:
				self._materialize_sharded_layer(0)

	def get_target_with_name(self, name: str) -> torch.Tensor:
		"""get the target model weight/buffer to be replaced"""
		if name in self._named_parameters:
			target = self._named_parameters[name]
		else:
			target = self._named_buffers[name]
		return target

	@torch.compiler.disable
	def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
		if not self.enabled or self.device is None or self.copy_stream is None:
			return
		if layer_idx < 0 or layer_idx >= self.num_layers:
			return
		if layer_idx in self._gpu_layers:
			return
		if layer_idx not in self._consolidated_cpu_weights:
			return
		self.copy_stream.wait_stream(torch.cuda.current_stream())

		if self.sharded:
			# Sharded path: H2D only local shard, defer all-gather to pre-hook
			gpu_shards: Dict[torch.dtype, torch.Tensor] = {}
			with torch.cuda.stream(self.copy_stream):
				for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
					shard_numel = self._shard_info[layer_idx][dtype][0]
					start = self.rank * shard_numel
					cpu_shard = cpu_buffer[start : start + shard_numel]
					gpu_shard = torch.empty(shard_numel, dtype=dtype, device=self.device)
					gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)
					gpu_shards[dtype] = gpu_shard

			event = torch.cuda.Event()
			event.record(self.copy_stream)
			self._prefetch_events[layer_idx] = event
			self._pending_shards[layer_idx] = gpu_shards
			self._gpu_layers.add(layer_idx)
			return

		# Non-sharded path: create gpu buffer and load from CPU buffer
		gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
		with torch.cuda.stream(self.copy_stream):
			for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
				gpu_buffer = torch.empty(
					cpu_buffer.shape, dtype=dtype, device=self.device
				)
				gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
				gpu_buffers[dtype] = gpu_buffer

		# Record event for fine-grained sync
		event = torch.cuda.Event()
		event.record(self.copy_stream)
		self._prefetch_events[layer_idx] = event

		# restore model's weights by their metadata using gpu buffer
		for name, meta in self._weight_metadata[layer_idx].items():
			dtype = meta["dtype"]
			gpu_buffer = gpu_buffers[dtype]

			# map the parameter's data to the correct slice of the GPU buffer
			target = self.get_target_with_name(name)
			target.data = gpu_buffer[
				meta["offset"] : meta["offset"] + meta["numel"]
			].view(meta["shape"])

		self._gpu_layers.add(layer_idx)

	@torch.compiler.disable
	def release_layer(self, layer_idx: int) -> None:
		if not self.enabled or self.device is None:
			return

		self._prefetch_events.pop(layer_idx, None)
		self._pending_shards.pop(layer_idx, None)

		if layer_idx <= 0:
			return

		if layer_idx not in self._gpu_layers:
			return

		for name, meta in self._weight_metadata.get(layer_idx, {}).items():
			target = self.get_target_with_name(name)
			target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

		self._gpu_layers.discard(layer_idx)

	@torch.compiler.disable
	def _materialize_sharded_layer(self, layer_idx: int) -> None:
		"""All-gather sharded weights and restore model parameters.

		Called on the compute stream after wait_event ensures the H2D shard
		copy is complete. Each rank contributes its 1/N shard, and the
		all-gather reconstructs the full consolidated buffer.
		"""
		if layer_idx not in self._pending_shards:
			return

		gpu_shards = self._pending_shards.pop(layer_idx)
		gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}

		for dtype, gpu_shard in gpu_shards.items():
			padded_numel = self._shard_info[layer_idx][dtype][1]
			gpu_full = torch.empty(padded_numel, dtype=dtype, device=self.device)
			dist.all_gather_into_tensor(gpu_full, gpu_shard)
			gpu_buffers[dtype] = gpu_full
			del gpu_shard

		# Restore weights using existing metadata
		for name, meta in self._weight_metadata[layer_idx].items():
			dtype = meta["dtype"]
			gpu_buffer = gpu_buffers[dtype]
			target = self.get_target_with_name(name)
			target.data = gpu_buffer[
				meta["offset"] : meta["offset"] + meta["numel"]
			].view(meta["shape"])

	@torch.compiler.disable
	def release_all(self) -> None:
		if not self.enabled or self.device is None:
			return
		if self.copy_stream is not None:
			torch.cuda.current_stream().wait_stream(self.copy_stream)

		for layer_idx in list(self._gpu_layers):
			self.release_layer(layer_idx)

	@torch.compiler.disable
	def load_all_layers(self) -> None:
		"""Load all layers from CPU to GPU."""
		if not self.enabled or self.device is None:
			return
		if self.copy_stream is not None:
			torch.cuda.current_stream().wait_stream(self.copy_stream)

		for layer_idx in range(self.num_layers):
			if layer_idx not in self._gpu_layers:
				self.prefetch_layer(layer_idx, non_blocking=False)

		if self.sharded and self.copy_stream is not None:
			torch.cuda.current_stream().wait_stream(self.copy_stream)
			for layer_idx in range(self.num_layers):
				self._materialize_sharded_layer(layer_idx)

	@torch.compiler.disable
	def sync_layer_to_cpu(self, layer_idx: int) -> None:
		"""Sync a layer's weights from GPU back to CPU."""
		if not self.enabled or layer_idx not in self._gpu_layers:
			return
		if layer_idx not in self._consolidated_cpu_weights:
			return

		if self.copy_stream is not None:
			torch.cuda.current_stream().wait_stream(self.copy_stream)

		# Collect current GPU weights and write back to CPU buffer
		for name, meta in self._weight_metadata.get(layer_idx, {}).items():
			target = self.get_target_with_name(name)
			gpu_weight = target.data.flatten().cpu()

			dtype = meta["dtype"]
			cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
			offset = meta["offset"]
			numel = meta["numel"]
			cpu_buffer[offset : offset + numel].copy_(gpu_weight)

	@torch.compiler.disable
	def sync_all_layers_to_cpu(self) -> None:
		"""Sync all loaded layers' weights from GPU back to CPU."""
		if not self.enabled or self.device is None:
			return
		if self.copy_stream is not None:
			torch.cuda.current_stream().wait_stream(self.copy_stream)

		for layer_idx in list(self._gpu_layers):
			self.sync_layer_to_cpu(layer_idx)

	def register_forward_hooks(self) -> None:
		if not self.enabled:
			return

		layers = getattr(self.model, self.layers_attr_str)

		def make_pre_hook(i):
			def hook(module, input):
				if i in self._prefetch_events:
					torch.cuda.current_stream().wait_event(self._prefetch_events[i])
				# Materialize sharded layer after H2D wait
				if self.sharded:
					self._materialize_sharded_layer(i)
				self.prefetch_layer(i + 1, non_blocking=True)

			return hook

		def make_post_hook(i):
			def hook(module, input, output):
				if self.copy_stream is not None:
					torch.cuda.current_stream().wait_stream(self.copy_stream)
				self.release_layer(i)

			return hook

		# register prefetch & release hooks for each layer
		self._forward_hooks.clear()
		for i, layer in enumerate(layers):
			pre_hook_handle = layer.register_forward_pre_hook(make_pre_hook(i))
			post_hook_handle = layer.register_forward_hook(make_post_hook(i))
			self._forward_hooks.extend([pre_hook_handle, post_hook_handle])

	def remove_forward_hooks(self) -> None:
		"""Remove all registered forward hooks."""
		for hook_handle in self._forward_hooks:
			hook_handle.remove()
		self._forward_hooks.clear()


class OffloadableDiTMixin:
	"""
	A mixin that registers forward hooks for a DiT to enable layerwise offload
	"""

	# the list of names of a DiT's layers/blocks
	layer_names: List[str]
	layerwise_offload_managers: list[LayerwiseOffloadManager] | None = None

	def configure_layerwise_offload(self, server_args: ServerArgs):
		self.layerwise_offload_managers = []
		for layer_name in self.layer_names:
			# a manager per layer-list
			module_list = getattr(self, layer_name, None)
			if module_list is None or not isinstance(module_list, torch.nn.ModuleList):
				continue

			num_layers = len(module_list)
			manager = LayerwiseOffloadManager(
				model=self,
				layers_attr_str=layer_name,
				num_layers=num_layers,
				enabled=True,
				pin_cpu_memory=server_args.pin_cpu_memory,
				sharded=getattr(server_args, "dit_offload_sharded", False),
			)
			self.layerwise_offload_managers.append(manager)

		logger.info(
			f"Enabled layerwise offload for {self.__class__.__name__} on modules: {self.layer_names}"
		)

	def prepare_for_next_denoise(self):
		if self.layerwise_offload_managers is None:
			return
		for manager in self.layerwise_offload_managers:
			manager.prepare_for_next_denoise(non_blocking=True)

	def disable_offload(self) -> None:
		"""Disable layerwise offload: load all layers to GPU and remove hooks."""
		if self.layerwise_offload_managers is None:
			return
		for manager in self.layerwise_offload_managers:
			if manager.enabled:
				manager.remove_forward_hooks()
				manager.load_all_layers()

	def enable_offload(self) -> None:
		"""Re-enable layerwise offload: sync weights to CPU, release layers, and restore hooks."""
		if self.layerwise_offload_managers is None:
			return
		for manager in self.layerwise_offload_managers:
			if manager.enabled:
				manager.sync_all_layers_to_cpu()
				for layer_idx in list(manager._gpu_layers):
					if layer_idx > 0:
						manager.release_layer(layer_idx)
				manager.register_forward_hooks()
