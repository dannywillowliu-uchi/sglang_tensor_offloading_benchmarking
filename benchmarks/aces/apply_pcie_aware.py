"""Apply PCIe-aware offload modifications to installed SGLang.

Usage:
    python apply_pcie_aware.py [--revert]

Modifies layerwise_offload.py, server_args.py, and wanvideo.py in-place.
Creates .bak files for revert.
"""
import sys
import os
import shutil

def get_sglang_root():
	import sglang
	return os.path.dirname(os.path.dirname(sglang.__file__))

def apply(root):
	# 1. wanvideo.py: add ffn_attr_names
	wanvideo = os.path.join(root, "sglang/multimodal_gen/runtime/models/dits/wanvideo.py")
	with open(wanvideo) as f:
		content = f.read()
	if "ffn_attr_names" not in content:
		shutil.copy2(wanvideo, wanvideo + ".bak")
		content = content.replace(
			"class WanTransformer3DModel(CachableDiT, OffloadableDiTMixin):\n",
			'class WanTransformer3DModel(CachableDiT, OffloadableDiTMixin):\n\tffn_attr_names = {"blocks": "ffn"}\n'
		)
		with open(wanvideo, "w") as f:
			f.write(content)
		print(f"Patched: {wanvideo}")
	else:
		print(f"Already patched: {wanvideo}")

	# 2. server_args.py: add dit_offload_pcie_aware
	server_args = os.path.join(root, "sglang/multimodal_gen/runtime/server_args.py")
	with open(server_args) as f:
		content = f.read()
	if "dit_offload_pcie_aware" not in content:
		shutil.copy2(server_args, server_args + ".bak")
		# Add field
		content = content.replace(
			"dit_offload_sharded: bool = False",
			"dit_offload_sharded: bool = False\n\tdit_offload_pcie_aware: bool = False"
		)
		# Add argparse entry after dit_offload_sharded argparse block
		# Find the sharded argparse block and add after it
		sharded_help_end = content.find('"Cannot be used with --dit-offload-pcie-aware."')
		if sharded_help_end == -1:
			# Find the end of the sharded argparse entry differently
			idx = content.find("--dit-offload-sharded")
			if idx != -1:
				# Find the closing paren of this add_argument call
				paren_count = 0
				pos = idx
				while pos < len(content):
					if content[pos] == "(":
						paren_count += 1
					elif content[pos] == ")":
						paren_count -= 1
						if paren_count <= 0:
							# Find the end of line
							eol = content.index("\n", pos)
							insert_point = eol + 1
							break
					pos += 1

				pcie_arg = '''        parser.add_argument(
            "--dit-offload-pcie-aware",
            action="store_true",
            default=ServerArgs.dit_offload_pcie_aware,
            help="Use PCIe-aware H2D scheduling for layerwise offload. Moves prefetch to FFN "
            "compute window to avoid contention with NCCL all-to-all on shared PCIe bus.",
        )
'''
				content = content[:insert_point] + pcie_arg + content[insert_point:]
		with open(server_args, "w") as f:
			f.write(content)
		print(f"Patched: {server_args}")
	else:
		print(f"Already patched: {server_args}")

	# 3. layerwise_offload.py: add PCIe-aware scheduling
	offload = os.path.join(root, "sglang/multimodal_gen/runtime/utils/layerwise_offload.py")
	with open(offload) as f:
		content = f.read()
	if "pcie_aware" not in content:
		shutil.copy2(offload, offload + ".bak")

		# Add pcie_aware parameter to __init__
		content = content.replace(
			"sharded: bool = False,\n\t) -> None:",
			"sharded: bool = False,\n\t\tpcie_aware: bool = False,\n\t) -> None:"
		)

		# Add pcie_aware initialization after sharded block
		content = content.replace(
			"self.device = torch.device(\"cuda\", torch.cuda.current_device())\n\t\tself.copy_stream = torch.cuda.Stream()",
			"self.pcie_aware = pcie_aware\n\t\tself.device = torch.device(\"cuda\", torch.cuda.current_device())\n\t\tlow_prio, _ = torch.cuda.Stream.priority_range()\n\t\tself.copy_stream = torch.cuda.Stream(priority=low_prio)\n\n\t\tself.ffn_attr_name = None\n\t\tif self.pcie_aware:\n\t\t\tif hasattr(model, \"ffn_attr_names\") and layers_attr_str in model.ffn_attr_names:\n\t\t\t\tself.ffn_attr_name = model.ffn_attr_names[layers_attr_str]\n\t\t\t\tlogger.info(f\"PCIe-aware scheduling: FFN attr = {self.ffn_attr_name}\")\n\t\t\telse:\n\t\t\t\tlogger.warning(\"pcie_aware=True but no FFN mapping; falling back to block-level.\")"
		)

		# Modify register_forward_hooks to add FFN scheduling
		# Replace the hook registration section
		old_hooks = """	def register_forward_hooks(self) -> None:
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
			self._forward_hooks.extend([pre_hook_handle, post_hook_handle])"""

		new_hooks = """	def register_forward_hooks(self) -> None:
		if not self.enabled:
			return

		layers = getattr(self.model, self.layers_attr_str)
		use_ffn_scheduling = self.pcie_aware and self.ffn_attr_name

		def make_pre_hook(i):
			def hook(module, input):
				if i in self._prefetch_events:
					torch.cuda.current_stream().wait_event(self._prefetch_events[i])
				# Materialize sharded layer after H2D wait
				if self.sharded:
					self._materialize_sharded_layer(i)
				# When PCIe-aware, prefetch moves to FFN pre-hook
				if not use_ffn_scheduling:
					self.prefetch_layer(i + 1, non_blocking=True)

			return hook

		def make_ffn_pre_hook(i):
			def hook(module, input):
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

			if use_ffn_scheduling:
				ffn_submodule = getattr(layer, self.ffn_attr_name, None)
				if ffn_submodule is not None:
					ffn_hook = ffn_submodule.register_forward_pre_hook(make_ffn_pre_hook(i))
					self._forward_hooks.append(ffn_hook)"""

		content = content.replace(old_hooks, new_hooks)

		# Add pcie_aware to OffloadableDiTMixin configure call
		content = content.replace(
			'sharded=getattr(server_args, "dit_offload_sharded", False),',
			'sharded=getattr(server_args, "dit_offload_sharded", False),\n\t\t\t\tpcie_aware=getattr(server_args, "dit_offload_pcie_aware", False),'
		)

		with open(offload, "w") as f:
			f.write(content)
		print(f"Patched: {offload}")
	else:
		print(f"Already patched: {offload}")

	print("\nAll patches applied. Use --revert to undo.")

def revert(root):
	for relpath in [
		"sglang/multimodal_gen/runtime/models/dits/wanvideo.py",
		"sglang/multimodal_gen/runtime/server_args.py",
		"sglang/multimodal_gen/runtime/utils/layerwise_offload.py",
	]:
		filepath = os.path.join(root, relpath)
		bak = filepath + ".bak"
		if os.path.exists(bak):
			shutil.copy2(bak, filepath)
			os.remove(bak)
			print(f"Reverted: {filepath}")
		else:
			print(f"No backup found: {bak}")

if __name__ == "__main__":
	root = get_sglang_root()
	print(f"SGLang root: {root}")
	if "--revert" in sys.argv:
		revert(root)
	else:
		apply(root)
