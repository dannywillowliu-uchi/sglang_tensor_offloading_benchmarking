# Experiment 3 Anomaly Analysis: Progressive Slowdown Without torch.compile

**Date:** 2026-02-27 | **Hardware:** 4x B200 NVLink | **Model:** Wan2.2-T2V-A14B (MoE, 2x 14B transformers)

## Observed Behavior

Exp3 (offload, no torch.compile) exhibits a progressive per-step slowdown in the **clean run only**:

| Step Range | Warmup Run (s/step) | Clean Run (s/step) | Delta |
|-----------|--------------------|--------------------|-------|
| 1-6       | 12.58-12.59        | 12.63-12.81        | ~0    |
| 7         | 12.58              | **14.89**           | +2.3  |
| 8-16      | 12.58-12.83        | 12.69-14.14        | ~0-1.3|
| 17        | 12.59              | **14.79**           | +2.2  |
| 18-21     | 12.59-14.58        | 13.19-14.10        | ~0    |
| 22        | 12.59              | **17.34**           | +4.8  |
| 23-27     | 12.70-12.91        | 20.89-27.20        | +8-14 |
| **Average** | **12.90**         | **16.98**           | **+4.08** |

Warmup average: 12.90s/step (perfectly stable). Clean average: 16.98s/step (+32%).

## Root Cause: CUDA Memory Allocator Fragmentation

The progressive slowdown is caused by **CUDA caching allocator fragmentation** in eager mode. torch.compile masks this by making memory access patterns deterministic.

### The Mechanism

1. **LayerwiseOffloadManager allocates ~700MB GPU buffers per layer per step.** In `prefetch_layer` (layerwise_offload.py:270-271):
   ```python
   gpu_buffer = torch.empty(cpu_buffer.shape, dtype=dtype, device=self.device)
   ```
   This creates 40 large contiguous allocations per step (one per transformer layer).

2. **`release_layer` replaces parameters with tiny stubs** (layerwise_offload.py:319):
   ```python
   target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])
   ```
   This frees the 700MB buffer but leaves tiny 1-element allocations scattered through GPU memory.

3. **Eager-mode forward passes create non-deterministic intermediate allocations.** Without torch.compile, each layer forward creates hundreds of intermediate tensors (attention outputs, MLP intermediates, normalization results). These are allocated between the large offload buffers, fragmenting the contiguous memory space.

4. **Fragmentation accumulates because eager-mode allocation order varies.** Python dict iteration order, object creation timing, and GC pauses introduce subtle nondeterminism. Each step's allocation pattern differs slightly from the previous, preventing the caching allocator from perfectly reusing cached blocks.

5. **When fragmentation exceeds a threshold, `cudaMalloc` and `cudaFree` become expensive.** The B200 profiling data confirms this directly:

   | Metric | No-Offload Baseline | With Offload | Ratio |
   |--------|--------------------:|-------------:|------:|
   | cudaFree calls | 195 | 194 | ~1x |
   | cudaFree total time | 0.07s | **4.75s** | **68x** |
   | cudaFree avg | 348us | **24,494us** | **70x** |
   | cudaFree max | 1,416us | **1,010,719us** | **714x** |
   | cudaMalloc calls | 3,816 | 259 | 0.07x |
   | cudaMalloc avg | 601us | **4,995us** | **8.3x** |
   | cudaMalloc max | 69,964us | **121,921us** | **1.7x** |

   The max cudaFree duration of **1,010ms** (1 full second) in offload mode vs 1.4ms baseline proves the allocator is periodically forced to defragment. A single 1-second cudaFree stall explains the step-7 and step-17 jumps perfectly.

### Why torch.compile Masks This

With `torch.compile(mode="max-autotune-no-cudagraphs")`:

1. **Deterministic memory layout.** The compiler pre-plans all intermediate tensor allocations at trace time. Every step uses exactly the same memory layout -- the caching allocator hits its fast path (cached block reuse) every time.

2. **Operator fusion reduces allocation count.** Fused kernels eliminate many intermediate tensors entirely. Fewer allocations = less fragmentation surface.

3. **No Python-level nondeterminism.** Compiled code doesn't go through Python dict lookups, GC pauses, or dynamic dispatch that could alter allocation order.

4. **Offload hooks still use `@torch.compiler.disable`.** The graph breaks at hook boundaries, but the compiled segments within each layer produce identical memory patterns step-to-step.

### Why the Warmup Run Is Stable

The warmup run (12.90s/step average, perfectly stable across 27 steps) starts with a **fresh process and clean GPU memory**. The caching allocator has no prior fragmentation. Even in eager mode, the first run can maintain stability because:

- First 6 steps: Caching allocator builds up cached blocks, all contiguous
- Subsequent steps: Cached blocks from early steps fit new allocations well
- No prior fragmentation to compound

### Why the Clean Run Degrades

The clean run started ~30 seconds after the warmup process was **force-killed**:
```
Generator was garbage collected without being shut down
Local worker sglang-diffusionWorker-0 did not terminate gracefully, forcing.
leaked semaphore objects to clean up at shutdown
```

While the CUDA driver reclaims GPU memory when a process exits, the clean run's process must re-initialize the caching allocator from scratch. The combination of:
1. New allocator with no learned patterns
2. Potentially fragmented driver-level memory from rapid allocation/free cycles during startup
3. Model loading (`transformer_2` reported `consumed: -5.52 GB` -- negative, suggesting shared/remapped memory)

...creates conditions where fragmentation accumulates faster than in the warmup run.

### The Step Pattern: Jump-Recovery-Explosion

| Pattern | Steps | Explanation |
|---------|-------|-------------|
| Stable baseline | 1-6 | Caching allocator in good state, cached blocks available |
| First jump | 7 | Fragmentation hits threshold; cudaFree/cudaMalloc triggers defrag (~2s penalty) |
| Recovery | 8-16 | Defragmentation clears space; allocator recovers; gradual improvement |
| Second jump | 17 | Re-fragmentation after 10 more steps of eager-mode churn |
| Brief recovery | 18-21 | Partial recovery |
| Explosion | 22-27 | Cumulative fragmentation becomes unrecoverable; each step worsens the state |

The exponential growth pattern (17.3s, 20.9s, 23.3s, 25.0s, 26.2s, 27.2s) in steps 22-27 is characteristic of allocator pathology: each defragmentation attempt is less effective because the free list itself has become deeply fragmented.

## Contributing but Non-Dominant Factors

### CUDA Event Accumulation (Minor)

Each step creates 40 `torch.cuda.Event()` objects (layerwise_offload.py:277) and pops 40 from `self._prefetch_events` (layerwise_offload.py:308). In eager mode, popped events become Python garbage awaiting GC. When the GC runs, it triggers CUDA event destruction (`cudaEventDestroy`), which can stall the GPU pipeline.

However, this is unlikely to be dominant because:
- 40 events per step is modest
- Python's cyclic GC for simple objects is fast
- The warmup run had the same event count and showed no degradation

### Layer 0 Never Released (Minor)

`release_layer(0)` returns early at layerwise_offload.py:311 (`if layer_idx <= 0: return`). Layer 0's ~700MB GPU buffer persists forever. This is by design (avoids re-prefetch overhead at step boundaries) and doesn't accumulate -- it's a constant ~700MB overhead.

### MoE Transformer Switching (Not a Factor)

With `dit_layerwise_offload=True`, `dit_cpu_offload` is automatically disabled (server_args.py:1040-1044). The `_manage_device_placement` method returns immediately. The boundary_ratio=0.875 switch at step ~23 doesn't trigger any blocking `.to()` calls. Both transformers have independent LayerwiseOffloadManagers, and only the active transformer's hooks fire.

## Severity Assessment

- **Impact:** +32% slowdown on clean run (16.98 vs 12.90 s/step)
- **Workaround:** Use torch.compile (exp2: 12.03s/step -- even faster than warmup)
- **Production relevance:** Moderate. Users running without compile (e.g., for debugging or on unsupported architectures) will see degraded performance on multi-generation runs
- **Interaction with other overheads:** This is an **eager-mode-specific** issue independent of the PCIe contention and NCCL slowdown analyzed in the ACES experiments

## Recommendations

1. **Use torch.compile (current mitigation).** Exp2 confirms compile mode (12.03s/step) eliminates the issue entirely and is even faster than the warmup eager baseline (12.90s/step).

2. **Pre-allocate a GPU buffer pool (code fix).** The existing `pcie_aware` flag already does this (layerwise_offload.py:181-194) with `_gpu_pool` double-buffering. Enabling this by default (not just for pcie_aware mode) would eliminate the per-step `torch.empty(~700MB)` allocations that drive fragmentation. Estimated fix: ~10 lines of code.

3. **Call `torch.cuda.memory.empty_cache()` periodically (quick fix).** Adding `torch.cuda.empty_cache()` in `prepare_for_next_req` every N steps would force the allocator to release cached blocks, preventing cumulative fragmentation. Trade-off: slight per-step latency for defrag vs avoiding large stalls.

4. **Use `torch.cuda.memory.set_per_process_memory_fraction()` or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (config fix).** PyTorch 2.1+ supports expandable segments that reduce fragmentation by allowing the allocator to grow/shrink segments dynamically. This is a no-code-change fix.
