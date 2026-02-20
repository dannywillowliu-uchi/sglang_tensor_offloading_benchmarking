# Thorough Root Cause Analysis: Why Layerwise Offload is 3s/step Slower on PCIe

**Date:** 2026-02-19
**Hardware:** ACES cluster, 4x H100 PCIe GPUs, 488GB RAM
**Observed delta:** 3.0s/step (23s new vs 20s old, flow_shift=12.0, clean Run 2)

---

## 0. Executive Summary

### Why the PR works on NVLink but not PCIe

The PR's 58% speedup on NVLink comes from **eliminating two blocking transfer spikes** (step 0: 36s, step 19: 31s), NOT from faster steady-state. On NVLink, steady-state is identical: 3.27s vs 3.29s/step.

On PCIe, the new offload **adds** 3s/step of steady-state overhead (23s vs 20s). Over 27 steps, this +81s far exceeds the 27s saved from eliminating the step-19 spike. The overhead exists on PCIe but not NVLink because of **hardware topology**:

| Factor | NVLink (no overhead) | PCIe (3s/step overhead) |
|--------|---------------------|------------------------|
| H2D path | Per-GPU PCIe link (no cross-GPU contention) | Shared PCIe root complex (4 GPUs compete) |
| GPU-GPU path | NVLink (900 GB/s, separate from H2D) | PCIe (shared with H2D) |
| H2D per GPU per layer | 700MB (but no contention) | 700MB (contending with NCCL + other GPUs) |
| FSDP H2D per GPU per layer | 175MB shard (topology-aware) | 175MB shard (topology-aware) |

### Overhead decomposition on PCIe (estimated, pending nsys confirmation)

| Source | Estimated Impact | Hardware-dependent? |
|--------|-----------------|---------------------|
| 1. PCIe contention: 4x H2D + shared bus with NCCL | **~1.5-2.0s** | **YES** -- absent on NVLink |
| 2. No topology-aware transfer scheduling | ~0.5-1.0s | **YES** -- FSDP/NCCL are topology-aware |
| 3. `@torch.compiler.disable` graph breaks (80/step) | ~0.2-0.5s | No -- exists on both, small |
| 4. GPU memory allocation churn | ~0.1-0.3s | No -- exists on both |
| **Total estimated** | **~2.3-3.8s** | |

**Critical uncertainty:** We cannot measure the async H2D bandwidth on the copy_stream from existing data. The 58.3% utilization number comes from the few blocking transfers, not the bulk async copies. The nsys profiles (jobs 1459999-1460000) will reveal actual contention levels.

### Why FSDP handles PCIe better

FSDP + NCCL are designed for shared-bus topologies:
- Each GPU transfers only 175MB from CPU (1/4 shard) vs 700MB -- **4x less root complex pressure**
- NCCL all-gather is **topology-aware** and schedules to minimize PCIe contention
- `reorder_for_compute_comm_overlap=True` lets the compiler optimize FSDP's communication
- LayerwiseOffloadManager's `copy_(non_blocking=True)` has **no topology awareness**

---

## 1. Per-Step Operation Trace: New Offload (Layerwise)

### 1.1 Architecture

- 40 transformer layers per model (`num_layers=40`, wanvideo.py:69)
- `layer_names = ["blocks"]` (wanvideo.py:777)
- `copy_stream = torch.cuda.Stream()` dedicated to H2D (layerwise_offload.py:47)
- `prefetch_size = 1` (default, one layer ahead)
- Each layer: ~700MB at BF16 (14B params / 40 layers * 2 bytes)
- torch.compile with `fullgraph=False` (denoising.py:150)
- torch.compile mode: `max-autotune-no-cudagraphs` (denoising.py:147)
- `reorder_for_compute_comm_overlap = True` (denoising.py:144)

### 1.2 What Happens Each Step (27 total)

For each of 40 layers, this sequence executes:

```
[COMPILED CODE BREAKS] --> pre_hook (Python, @torch.compiler.disable)
    |
    +-- If layer 0: prepare_for_next_req(non_blocking=False)
    |     +-- prefetch_layer(0) --> noop (layer 0 always on GPU)
    |     +-- current_stream.wait_stream(copy_stream)  <-- BLOCKING SYNC
    |
    +-- If layer i in _prefetch_events:
    |     +-- current_stream.wait_event(event_i)  <-- WAIT FOR PREFETCH
    |
    +-- If i % prefetch_size == 0 (every layer when prefetch_size=1):
          +-- prefetch_layer(i+1, non_blocking=True)
                +-- copy_stream.wait_stream(current_stream)  <-- ORDERING DEP
                +-- torch.empty(~700MB, device=GPU)  <-- GPU ALLOC
                +-- gpu_buffer.copy_(cpu_buffer, non_blocking=True)  <-- ASYNC H2D
                +-- event.record(copy_stream)
                +-- .data pointer swap (param -> gpu_buffer slice)

[COMPILED CODE RESUMES] --> layer.forward()  <-- ~575ms compute
    |
    +-- Includes 2x Ulysses all-to-all on COMPUTE stream

[COMPILED CODE BREAKS] --> post_hook (Python, @torch.compiler.disable)
    |
    +-- release_layer(i):
          +-- pop event
          +-- If i > 0: for each weight: target.data = torch.empty((1,), GPU)
          +-- _gpu_layers.discard(i)

[COMPILED CODE RESUMES] --> next layer...
```

### 1.3 Per-Step Operation Count

| Operation | Count | @torch.compiler.disable? | Category |
|-----------|-------|--------------------------|----------|
| Pre-hooks fired | 40 | YES (via prefetch_layer, prepare_for_next_req) | Graph break |
| Post-hooks fired | 40 | YES (via release_layer) | Graph break |
| Graph breaks (compiled -> eager -> compiled) | **80** | -- | Core overhead |
| copy_stream.wait_stream(current_stream) | 40 | -- | Stream sync |
| current_stream.wait_stream(copy_stream) | 1 | -- | Blocking sync |
| current_stream.wait_event(prefetch_event) | 40 | -- | Event sync |
| torch.cuda.Event() + record | 40 | -- | Event mgmt |
| torch.empty(~700MB, GPU) for prefetch | ~40 | -- | GPU alloc |
| torch.empty((1,), GPU) for release stubs | ~39 | -- | GPU alloc |
| .data pointer swaps (to GPU buffer) | ~40 | -- | Pointer mgmt |
| .data pointer swaps (to stub) | ~39 | -- | Pointer mgmt |
| copy_ (H2D, ~700MB, async) | ~40 | -- | PCIe transfer |

### 1.4 Per-Step Operation Trace: Old Offload (FSDP)

```
For each of 40 layers:
    FSDP all-gather:
        +-- Each GPU loads its 1/N shard from CPU pinned memory
        +-- NCCL all-gather collects full weights to GPU
        +-- PIPELINED: all-gather for layer i+1 overlaps with layer i compute
    layer.forward()  <-- same ~575ms compute (no graph break)
    FSDP reshard:
        +-- Keep local shard, release the rest
        +-- Managed by FSDP internal memory pool
```

**Critical difference:** FSDP2 (`fully_shard()`, fsdp_load.py:189,198) is implemented as a torch.compile-compatible primitive. Its hooks do NOT use `@torch.compiler.disable` and do NOT cause graph breaks. The compiler can see through FSDP operations and optimize across layer boundaries.

---

## 2. Overhead Source 1: PCIe Topology Contention (Estimated ~1.5-2.0s, HARDWARE-DEPENDENT)

### 2.0 Why This Is the Primary Factor

The PR's NVLink data proves this is the dominant factor. On NVLink, steady-state delta is 0.02s. On PCIe, it's 3.0s. The only thing that changes is the hardware topology -- the software (including graph breaks) is identical.

On NVLink (DGX/SXM topology):
- Each GPU has a **dedicated PCIe Gen5 link** to the CPU -- no cross-GPU H2D contention
- GPU-GPU communication uses **NVLink** (900 GB/s) -- completely separate physical path
- H2D prefetch runs on PCIe while GPU compute + all-to-all runs on NVLink -- **full overlap, zero contention**

On PCIe (ACES topology):
- All 4 GPUs share the **PCIe root complex** for CPU access
- GPU-GPU communication (Ulysses all-to-all, FSDP all-gather) also goes over PCIe
- H2D prefetch, NCCL all-to-all, and NCCL all-gather all compete for the same bus

New offload per-step per-GPU PCIe load: ~700MB H2D per layer * 40 layers = 28 GB
Old offload (FSDP) per-step per-GPU PCIe load: ~175MB H2D shard per layer * 40 = 7 GB H2D, plus all-gather traffic managed by topology-aware NCCL

The new offload puts **4x more CPU-to-GPU traffic** on the root complex, while FSDP minimizes this with sharding and uses NCCL's topology-aware scheduling for the all-gather.

**We cannot directly measure the async copy bandwidth** from existing profiling data (CPU-side traces only capture launch overhead, not GPU-side completion). The nsys profiles will be definitive.

---

## 3. Overhead Source 2: `@torch.compiler.disable` Graph Breaks (Estimated ~0.2-0.5s, HARDWARE-INDEPENDENT)

### 3.1 The Problem

Every method in LayerwiseOffloadManager that touches CUDA operations is decorated with `@torch.compiler.disable`:
- `_initialize` (line 80)
- `prefetch_layer` (line 157)
- `release_layer` (line 200)
- `release_all` (line 224)
- `load_all_layers` (line 234)
- `sync_layer_to_cpu` (line 246)
- `sync_all_layers_to_cpu` (line 268)

The pre-hooks and post-hooks call these methods, causing the compiled graph to break at every layer boundary. With 40 layers:
- 40 pre-hooks call `prefetch_layer` and optionally `prepare_for_next_req`
- 40 post-hooks call `release_layer`
- = **80 graph breaks per step**

### 3.2 What a Graph Break Costs

When `fullgraph=False` and `@torch.compiler.disable` is hit:
1. The compiled inductor graph must be "broken" -- any pending fused kernels are emitted
2. Control returns to the Python interpreter (eager mode)
3. The hook runs (Python dict lookups, CUDA API calls, etc.)
4. Control returns to the compiled graph (next segment is looked up or compiled)

Each break costs:
- **Kernel launch overhead**: Smaller, unfused kernels instead of large fused ones
- **Python interpreter overhead**: ~1-5ms per hook execution
- **Lost fusion opportunities**: Operations at the end of layer i and start of layer i+1 that could have been fused are now separated by a Python-level hook
- **Inductor optimization loss**: `reorder_for_compute_comm_overlap = True` (line 144) cannot reorder across graph breaks

### 3.3 Why FSDP Doesn't Have This Problem

FSDP2's `fully_shard()` is designed for torch.compile compatibility. Its internal hooks:
- Are registered as compiler-aware primitives, not Python-level hooks
- Do NOT use `@torch.compiler.disable`
- Allow the compiler to see the full graph including FSDP's communication
- Enable `reorder_for_compute_comm_overlap` to overlap FSDP's all-gather with compute

This means the old offload path has **0 graph breaks** from FSDP, while the new offload path has **80**.

### 3.4 Estimate

The NVLink data constrains this estimate. On NVLink, delta is 0.02s with the same 80 graph breaks. This means graph breaks cost at most ~0.02s on NVLink (3.3s/step total). On PCIe (23s/step), graph breaks should cost a similar absolute amount, potentially slightly more due to longer graph segments being interrupted.

**Best estimate: ~0.2-0.5s** (not the dominant factor -- NVLink data proves this)

---

## 4. Overhead Source 3: CUDA Stream Synchronization (Estimated ~0.3-0.5s, HARDWARE-INDEPENDENT)

### 4.1 Blocking Sync at Step Boundary

At layer 0 of each step (layerwise_offload.py:289):
```python
self.prepare_for_next_req(non_blocking=False)
# -> prefetch_layer(0) --> noop (on GPU)
# -> torch.cuda.current_stream().wait_stream(self.copy_stream)  # BLOCKING
```

This forces a full pipeline drain: the compute stream halts until ALL outstanding copy_stream operations complete. Even if the copy_stream is idle (all prefetches done), the `wait_stream` call itself has CUDA API overhead.

### 4.2 Per-Layer copy_stream.wait_stream

Inside every `prefetch_layer` call (line 170):
```python
self.copy_stream.wait_stream(torch.cuda.current_stream())
```

This creates a dependency: the H2D copy for layer i+1 cannot begin until the compute stream reaches the point where `prefetch_layer` was called (in layer i's pre-hook). This is correct for ordering but creates 40 synchronization events per step.

### 4.3 Per-Layer Event Wait

In every pre-hook (line 291):
```python
torch.cuda.current_stream().wait_event(self._prefetch_events[i])
```

This ensures the compute stream doesn't execute layer i's forward pass until the prefetch is done. If prefetch is ahead (the normal case), cost is near-zero. But the event check itself has API overhead.

### 4.4 Contrast with FSDP

FSDP manages stream synchronization internally through PyTorch's distributed runtime. The sync points are:
- Fewer (FSDP can batch synchronization across multiple parameters)
- Compiler-visible (can be reordered by inductor)
- Implemented at the C++ level (no Python overhead)

**Best estimate: ~0.3-0.5s**

---

## 5. Overhead Source 4: GPU Memory Allocation Churn (Estimated ~0.2-0.3s, HARDWARE-INDEPENDENT)

### 5.1 Per-Layer Allocation Pattern

Each step performs:
- **40 large allocations**: `torch.empty(cpu_buffer.shape, dtype=dtype, device=GPU)` (~700MB each) in `prefetch_layer` (line 176)
- **39 tiny allocations**: `torch.empty((1,), device=GPU, dtype=meta["dtype"])` in `release_layer` (line 220)
- **~80 `.data` pointer swaps**: Setting `target.data = new_tensor` (lines 194-196, 220)

Total per step: ~79 GPU memory operations, allocating and deallocating ~28GB of GPU buffers.

### 5.2 CUDA Allocator Overhead

PyTorch's CUDA memory allocator (caching allocator) handles these allocations. For large allocations (~700MB):
- If a cached block exists: ~10us (fast path)
- If a new allocation is needed: ~1-10ms (cudaMalloc)
- If fragmentation requires defragmentation: ~10-100ms

With 40 layers cycling through alloc/dealloc, the caching allocator should hit the fast path most of the time. But 40 releases + 40 allocations per step still amount to non-trivial overhead.

### 5.3 FSDP's Approach

FSDP manages memory through its own internal pool. It pre-allocates buffers for all-gather and reuses them across layers. No per-layer allocation/deallocation cycle.

**Best estimate: ~0.2-0.3s**

---

---

## 6. Why the Delta Widened: 1.8s (flow_shift=3.0) -> 3.0s (flow_shift=12.0)

### 6.1 The Raw Numbers

| Metric | flow_shift=3.0 (profiled) | flow_shift=12.0 (clean) |
|--------|---------------------------|-------------------------|
| New offload steady-state | 21.6s/step | 23.0s/step |
| Old offload steady-state | 19.8s/step | 20.0s/step |
| Delta | 1.8s | 3.0s |

### 6.2 Adjusting for Profiling Overhead

The flow_shift=3.0 data was collected WITH torch.profiler active (adds ~25-35% overhead). Estimated clean flow_shift=3.0 performance:
- New offload: ~21.6 / 1.30 = ~16.6s/step
- Old offload: ~19.8 / 1.30 = ~15.2s/step
- Adjusted delta: **~1.4s**

So the true delta went from ~1.4s to 3.0s -- a **2.1x increase**.

### 6.3 Possible Explanations

**A. Increased compute intensity with flow_shift=12.0:**

The old offload's clean steady-state went from ~15.2s (est.) to 20.0s = +32% more compute. This is significant -- the noise schedule under flow_shift=12.0 produces different sigma values that change the effective computational workload per step.

**B. Compiler overlap optimization:**

With `reorder_for_compute_comm_overlap = True` enabled, torch.compile can reorder FSDP communication to better overlap with compute. With longer compute per step (flow_shift=12.0 = +32%), there's MORE compute to hide communication behind. This makes FSDP's overhead approach zero.

Meanwhile, the new offload's `@torch.compiler.disable` hooks prevent the compiler from performing any such optimization. The overhead stays constant (or increases slightly due to larger graph segments being interrupted).

**C. Combined effect:**

- Old offload benefit: Longer compute -> better FSDP pipelining -> lower effective overhead
- New offload penalty: Longer compute -> more optimization lost per graph break -> higher effective overhead
- Net: delta widens from ~1.4s to 3.0s

### 6.4 This Needs Confirmation

The nsys profiles (jobs 1459999-1460000) will provide definitive data:
- GPU-side timeline showing actual compute vs. transfer overlap
- Per-kernel timing with and without graph breaks
- NCCL vs H2D bandwidth partitioning on PCIe

---

## 7. Corrected Root Cause Summary

### The Key Test: NVLink vs PCIe

The PR's NVLink data is the Rosetta Stone for this analysis. If the overhead were purely software (graph breaks, stream management), it would appear on BOTH platforms. It doesn't:
- NVLink delta: **0.02s/step** (essentially zero)
- PCIe delta: **3.0s/step** (150x larger)

This proves the dominant factor is **hardware topology**, not software overhead.

### What Actually Causes the 3s/step Overhead on PCIe

1. **PCIe topology contention** (~50-65% of overhead): On PCIe, H2D copies and GPU-GPU communication (Ulysses all-to-all) share the same physical bus. The new offload puts 4x more H2D traffic per GPU (700MB/layer vs 175MB shard) on this shared bus, while FSDP minimizes root complex pressure through sharding and uses topology-aware NCCL scheduling. On NVLink, H2D (PCIe) and GPU-GPU (NVLink) are separate paths -- no contention.

2. **No topology-aware transfer scheduling** (~20-30%): FSDP + NCCL are PCIe-topology-aware and coordinate transfers to minimize contention. LayerwiseOffloadManager's `copy_(non_blocking=True)` fires blindly -- no awareness of other GPUs' transfers or NCCL traffic.

3. **Software overhead (graph breaks, stream sync, alloc churn)** (~10-20%): The `@torch.compiler.disable` hooks, manual stream management, and per-layer memory allocation add a small constant overhead (~0.2-0.8s) visible on both platforms.

### Why It Works on NVLink

The PR's speedup on NVLink is NOT from faster steady-state. It's from **eliminating blocking spikes**:
- Old offload step 0: 36s → New: 7.7s (saved 28s)
- Old offload step 19: 31s → New: 3.3s (saved 28s)
- Total spike savings: ~56s out of 149.7s = **37% of runtime**

Steady-state is identical (3.27 vs 3.29s). The layerwise offload overhead is invisible because H2D runs on a separate physical path from GPU-GPU communication.

---

## 8. Actionable Recommendations (Revised)

Based on the corrected root cause, the highest-impact improvements are different from what the PI report suggested:

### 8.1 Add FSDP-style weight sharding to layerwise offload (HIGHEST IMPACT)

Each GPU currently copies the full ~700MB per layer from CPU. With FSDP-style sharding, each GPU copies only 1/N (175MB with 4 GPUs), then all-gathers via NCCL. This:
- Reduces per-GPU root complex H2D pressure by 4x
- Lets NCCL's topology-aware scheduling handle the all-gather efficiently
- Expected improvement: eliminates the dominant PCIe contention factor

**This directly addresses the hardware-dependent bottleneck** that causes the NVLink vs PCIe asymmetry.

### 8.2 Make LayerwiseOffloadManager torch.compile-compatible (MEDIUM IMPACT)

Remove `@torch.compiler.disable` from offload methods and implement as compiler-safe operations (`torch.library.custom_op` or similar). This eliminates graph breaks and allows `reorder_for_compute_comm_overlap` to optimize the transfer schedule.

Expected improvement: ~0.2-0.5s (helps, but NVLink data shows this is not the dominant factor)

### 8.3 Reduce CUDA API call count (MEDIUM IMPACT)

- **Pool GPU buffers**: Instead of allocating and deallocating ~700MB per layer every step, pre-allocate a rotating pool of GPU buffers. Eliminates ~79 torch.empty calls per step.
- **Batch event operations**: Instead of creating/recording/waiting one event per layer, use a smaller number of events for groups of layers.
- Expected improvement: ~0.2-0.5s

### 8.4 Remove blocking sync at step boundaries (LOW-MEDIUM IMPACT)

Change `prepare_for_next_req(non_blocking=False)` at layer 0 to event-based sync. The current implementation forces a full pipeline drain at every step boundary.

Expected improvement: ~0.1-0.3s

### 8.5 Increase prefetch depth (LOW IMPACT)

With `prefetch_size=3-5`, more layers are prefetched ahead, providing more overlap margin. However, since the current pipeline already has sufficient overlap (575ms compute vs 70ms transfer), the benefit is marginal.

Expected improvement: ~0.0-0.1s (mostly helps edge cases with PCIe contention)

---

## 9. What the Pending nsys Data Will Confirm

Jobs 1459999-1460000 (nsys profiling, queued for Feb 19-20) will provide:

| Data | What It Confirms |
|------|-----------------|
| GPU kernel timeline | Whether graph breaks cause visible gaps between layer executions |
| copy_stream vs compute stream overlap | Whether async prefetch is fully overlapped or stalling |
| H2D Memcpy bandwidth per-layer | Actual PCIe bandwidth achieved by async copies |
| NCCL all-to-all timing | Whether Ulysses communication creates PCIe contention |
| Per-hook latency | How long each pre/post hook actually takes (Python + CUDA overhead) |
| FSDP all-gather pipelining | Whether FSDP overlaps layer i+1 all-gather with layer i compute |

**Key predictions to validate:**
1. New offload: H2D copies on copy_stream will show reduced effective bandwidth when concurrent with NCCL all-to-all
2. New offload: PCIe root complex will show higher utilization (more H2D traffic from 4 GPUs simultaneously)
3. Old offload: FSDP all-gather will show efficient pipelining with compute (NCCL topology-aware scheduling)
4. Both: graph breaks will show as small gaps (~2-5ms each), not the dominant cost
5. The difference in per-layer H2D bandwidth between new and old offload will correlate with the 3s delta

---

## 10. Confidence Assessment

| Claim | Confidence | Evidence |
|-------|------------|---------|
| PCIe topology contention is the #1 overhead source | **HIGH** (90%) | NVLink delta=0.02s, PCIe delta=3.0s; only hardware topology changes |
| FSDP handles PCIe better (sharding + NCCL topology awareness) | **HIGH** (85%) | 175MB shard vs 700MB full layer; NCCL is topology-aware by design |
| Graph breaks are a small, secondary factor | **HIGH** (85%) | NVLink delta=0.02s proves graph breaks cost very little |
| Async prefetch achieves good overlap for individual transfers | **MEDIUM** (75%) | Math works out (70ms vs 575ms), but no GPU-side proof yet |
| Actual async bandwidth is lower than blocking bandwidth due to contention | **MEDIUM** (70%) | Hypothesis only; 58.3% blocking BW may not reflect async BW |
| Delta widening (1.4s -> 3.0s) is compute-intensity dependent | **MEDIUM** (65%) | Consistent explanation but multiple possible causes |

**What would change the analysis:**
- If nsys shows copy_stream achieving full bandwidth with no contention, then PCIe contention is NOT the issue and we need a new hypothesis
- If nsys shows large gaps at graph break boundaries, then graph breaks contribute more than NVLink data suggests (possible difference in torch.compile behavior between systems)
- If nsys shows FSDP's all-gather timing is similar to new offload's H2D timing, then the sharding advantage is smaller than estimated

---

## Appendix A: Code Citations

| Reference | File:Line |
|-----------|-----------|
| copy_stream creation | layerwise_offload.py:47 |
| @torch.compiler.disable on _initialize | layerwise_offload.py:80 |
| @torch.compiler.disable on prefetch_layer | layerwise_offload.py:157 |
| @torch.compiler.disable on release_layer | layerwise_offload.py:200 |
| copy_stream.wait_stream in prefetch_layer | layerwise_offload.py:170 |
| torch.empty GPU alloc in prefetch_layer | layerwise_offload.py:176 |
| Async copy_ in prefetch_layer | layerwise_offload.py:179 |
| Event record in prefetch_layer | layerwise_offload.py:183-184 |
| .data swap in prefetch_layer | layerwise_offload.py:194-196 |
| Layer 0 skip in release_layer | layerwise_offload.py:212 |
| Stub alloc in release_layer | layerwise_offload.py:220 |
| prepare_for_next_req blocking sync | layerwise_offload.py:146-147 |
| Pre-hook (prepare + event wait + prefetch) | layerwise_offload.py:286-298 |
| Post-hook (release) | layerwise_offload.py:301-305 |
| FSDP fully_shard with CPUOffloadPolicy | fsdp_load.py:179-180, 189, 198 |
| torch.compile with fullgraph=False | denoising.py:150 |
| reorder_for_compute_comm_overlap | denoising.py:144 |
| Old offload blocking .to() | denoising.py:854, 861 |
| Ulysses all-to-all on compute stream | base_device_communicator.py:142-144 |
| PCIe bandwidth data (58.3% util) | baseline_analysis_pre_benchmarks.md Section 4.2 |

## Appendix B: PCIe Transfer Timeline (One Step, Theoretical)

```
Time (ms)   0    100   200   300   400   500   600   700   ...  23000
            |-----|-----|-----|-----|-----|-----|-----|-----|       |
Compute:    [==layer0 fwd==][==layer1 fwd==][==layer2 fwd==]...[layer39]
            ~575ms each
Copy:        [L1 H2D]       [L2 H2D]       [L3 H2D]
             ~35ms           ~35ms           ~35ms
             (overlapped)    (overlapped)    (overlapped)

At 20 GB/s per-GPU effective bandwidth:
  700MB / 20 GB/s = 35ms transfer
  575ms compute window = 16x more time than needed
  --> Transfer is fully hidden in the overlap
```

This confirms that raw PCIe transfer time is NOT the bottleneck. The overhead comes from the software machinery surrounding the transfers.
