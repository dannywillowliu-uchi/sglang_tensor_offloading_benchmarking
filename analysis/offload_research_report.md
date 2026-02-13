# SGLang Offload Research Report: Source Code Verification

**Date:** 2026-02-08
**Authors:** Parallel source code analysis agents (A, B, C) + cross-check
**Codebase:** sglang-src (HEAD, shallow clone)
**Prior work:** Obsidian note `Claude/Projects/sglang-offload-research.md`

---

## Executive Summary

This report provides code-verified answers to 6 research questions about SGLang PR #15511's layerwise offloading for Wan2.2 video generation. Three parallel analysis agents examined the source code with line-number citations. Key findings:

1. **All prior claims about offload implementations verified** -- the OLD path uses blocking `.to()`, NEW path uses async H2D prefetch with no D2H copy
2. **Ulysses overlap NOT feasible on ACES H100 PCIe** -- both GPU-GPU and CPU-GPU share PCIe fabric
3. **Step 10 contradiction PARTIALLY resolved** -- math with current code says switch at step 18, not step 10; discrepancy requires live timestep logging
4. **Clean benchmarks needed** -- Tuesday's 3-iteration runs (jobs 1439552-1439555) will provide definitive timing data

---

## Q1: Benchmark Results

### Existing 4-GPU Data (Jobs 1434113/1434114)

| Metric | New Offload (layerwise) | Old Offload (full-model) |
|---|---|---|
| Total time | 12:00 (788.2s denoising) | **11:12 (714.1s denoising)** |
| Step 1 (compile) | ~122s | ~110s |
| Steady state | ~23s/step | ~21s/step |
| Step 10 anomaly | 24.2s (normal) | 29.8s (spike) |
| Peak GPU memory | 22.88 GB | 61.25 GB |
| Large memcpy (>10s) | 44 | 936 |

**Key finding:** Old offload is **7% faster on 4 GPUs** despite having a spike at step 10. New offload's ~2s/step distributed overhead (26 steps x 2s = 52s) exceeds the one-time spike cost.

### Pending: Clean 3-Iteration Benchmarks (Tuesday Feb 10)

| Job | Config | GPUs |
|---|---|---|
| 1439552 | new_offload | 4 GPU |
| 1439553 | old_offload | 4 GPU |
| 1439554 | new_offload | 6 GPU |
| 1439555 | old_offload | 6 GPU |

Each job runs 3 iterations: warmup (compile) -> clean timing -> profiled. Use Run 2 for comparison.

**Action items for Tuesday:**
1. Download logs from ACES
2. Extract per-step timing from Run 2 (no profiling overhead)
3. Compare 4-GPU vs 6-GPU scaling
4. Check if 6-GPU narrows the gap between old and new
5. Extract profiler traces from Run 3

---

## Q2: Ulysses Communication Overlap with Offloading

**Full report:** `analysis/agent_a_ulysses_overlap.md`

### Answer: NOT feasible on ACES H100 PCIe

| Finding | Code Citation |
|---|---|
| LayerwiseOffloadManager creates dedicated `copy_stream` | `layerwise_offload.py:47` |
| H2D transfers run on `copy_stream` with `non_blocking=True` | `layerwise_offload.py:174-179` |
| Ulysses all-to-all runs on current (compute) stream | `base_device_communicator.py:142-144` |
| No NVLink/PCIe topology detection in codebase | Searched `platforms/cuda.py` -- no detection logic |

**Stream-level overlap exists** (different CUDA streams can run concurrently), but **physical PCIe contention prevents benefit on ACES**:

| System | GPU-GPU Path | CPU-GPU Path | Overlap? |
|---|---|---|---|
| H100 SXM (NVLink) | NVLink (900 GB/s) | PCIe (128 GB/s) | YES -- separate paths |
| H100 PCIe (ACES) | PCIe (128 GB/s) | PCIe (128 GB/s) | NO -- shared fabric |

**Implication:** PR #15511's performance claims were likely validated on NVLink systems. On PCIe-only ACES, both Ulysses all-to-all and H2D prefetch compete for the same ~64 GB/s unidirectional PCIe bandwidth.

---

## Q3: Transformer Switch at Step 10

**Full report:** `analysis/agent_b_step10_analysis.md`

### Answer: RESOLVED -- config misresolution caused flow_shift=3.0 instead of 12.0

#### The Switching Mechanism (verified)

**`denoising.py:870`**: `if t_int >= boundary_timestep` -> use high-noise transformer
**`denoising.py:503`**: `boundary_timestep = boundary_ratio * num_train_timesteps` = 0.875 * 1000 = **875**
**`denoising.py:1019`**: `t_int = int(t_host.item())` -- actual timestep value, not step index

#### Root Cause: Wrong `flow_shift` Value in Benchmarks

The benchmark spike at step 10 IS the transformer switch. The discrepancy was caused by `flow_shift=3.0` being used instead of `12.0`.

**Two schedulers exist** (we originally analyzed the wrong one):
- `scheduling_unipc_multistep.py` -- diffusers UniPC (NOT used by Wan2.2)
- `scheduling_flow_unipc_multistep.py` -- Flow UniPC from Wan2.1 official repo (ACTUALLY used, `wan_pipeline.py:50-52`)

Both produce the same result for a given `flow_shift`, but the pipeline creates `FlowUniPCMultistepScheduler(shift=server_args.pipeline_config.flow_shift)`.

**Config misresolution with local model path:**

The benchmarks use `--model-path ./models/wan2.2` (local directory). The registry resolution at `registry.py:249-291`:
1. **Exact match**: `./models/wan2.2` != `Wan-AI/Wan2.2-T2V-A14B-Diffusers` -- FAIL
2. **Partial match**: `"wan2.2"` != `"wan2.2-t2v-a14b-diffusers"` -- FAIL
3. **Detector**: `model_index.json` has `_class_name: "WanPipeline"` -> matches `"wanpipeline"` detector at `registry.py:472`
4. That detector maps to base `WanT2V480PConfig` which has **`flow_shift=3.0`** (line 73)

The correct `Wan2_2_T2V_A14B_Config` (flow_shift=12.0) has NO model detectors -- only an exact HuggingFace path mapping (`registry.py:543`).

#### Switch Point by flow_shift Value

| `flow_shift` | Switch step (0-idx) | Switch step (tqdm) | Config source |
|---|---|---|---|
| **3.0** | **9** | **10** | `WanT2V480PConfig` (base) -- **matches benchmark** |
| 5.0 | 12 | 13 | `WanT2V720PConfig` |
| 8.0 | 15 | 16 | `TurboWanT2V480PConfig` |
| 12.0 | 18 | 19 | `Wan2_2_T2V_A14B_Config` (correct for A14B) |

#### Implications

1. **Benchmark data is valid** for flow_shift=3.0 behavior, but does NOT reflect the intended A14B config
2. **With correct flow_shift=12.0**: the switch happens 9 steps later -> old offload's blocking transfer cost is deferred, and 9 additional steps run with new offload's per-step overhead. This could change the relative performance gap.
3. **Tuesday benchmarks must use correct config**: either `--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers` (HF path) or verify local path resolution

#### Required Actions for Tuesday

1. Add logging: `print(f"flow_shift={scheduler.config.shift}, boundary_timestep={boundary_timestep}")`
2. Add per-step logging: `print(f"Step {i}: t_int={t_int}, transformer={'main' if t_int >= boundary_timestep else 'secondary'}")`
3. Use `--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers` or fix config resolution
4. Compare results with flow_shift=12.0 vs 3.0 to understand impact

---

## Q4: Old vs New Offloading Difference

**Full report:** `analysis/agent_c_offload_deepdive.md`

### Verified Comparison

| Aspect | OLD: dit_cpu_offload | NEW: dit_layerwise_offload | Code Citation |
|---|---|---|---|
| Transfer method | Blocking `.to(device)` | Async H2D on `copy_stream` | denoising.py:854,861 vs layerwise_offload.py:179 |
| D2H copy | `.to("cpu")` on full model | NO D2H (stub replacement) | denoising.py:854 vs layerwise_offload.py:202-205 |
| VRAM | ~28GB (full transformer) | ~700MB (prefetch_size=1) | -- vs layerwise_offload.py:36 |
| FSDP | Uses CPUOffloadPolicy | Disables FSDP inference | fsdp_load.py:179 vs server_args.py:1021 |
| Mutual exclusion | Auto-disabled when layerwise=true | Auto-disables dit_cpu_offload | server_args.py:1026 |
| Cache-DIT | Compatible | **Incompatible** (raises error) | server_args.py:1028-1033 |

All prior claims verified. No corrections needed.

---

## Q5: When Does Offloading Happen?

**Full report:** `analysis/agent_c_offload_deepdive.md`

### Complete Weight Lifecycle (NEW path)

```
INIT:  Collect weights -> Concat to pinned CPU -> Replace GPU .data with stub -> Prefetch layer 0 (blocking)
                                                                                 [layerwise_offload.py:81-133]

DENOISING STEP:
  Pre-hook(i):  wait_event(prefetch[i])          -> Trigger prefetch(i+prefetch_size)
                [LINE 291: sync]                    [LINE 294-297: async H2D]

  Forward(i):   Compute with GPU weights

  Post-hook(i): release_layer(i)                 -> .data = torch.empty((1,))  [NO D2H COPY]
                [LINE 305]                          [LINE 220]

BETWEEN REQUESTS: release_all() -> All layers to stub (except layer 0) -> CPU buffer UNCHANGED
```

### Key Verified Details

| Detail | Verified | Citation |
|---|---|---|
| Dedicated copy_stream for H2D | YES | `layerwise_offload.py:47` |
| Event-based sync (not stream sync) | YES | `layerwise_offload.py:184, 291` |
| NO D2H copy | YES | Docstring at `layerwise_offload.py:202-205` |
| Layer 0 never released | YES | `layerwise_offload.py:212-213` |
| Prefetches `prefetch_size` layers ahead | YES | `layerwise_offload.py:294-297` |

---

## Q6: dit_layerwise_offload vs dit_cpu_offload Implementation

**Full report:** `analysis/agent_c_offload_deepdive.md`

### Key Architectural Differences

**LayerwiseOffloadManager** (`layerwise_offload.py`):
- Manual hook-based control with `register_forward_pre_hook` / `register_forward_hook`
- `.data` pointer swapping (weight.data -> GPU buffer slice, then -> 1-element stub)
- Explicit `copy_stream` + CUDA events for async coordination
- All offload methods decorated `@torch.compiler.disable` (hooks run eager, forward compiled)
- Error handling: idempotent checks (skip if already prefetched/released)

**CPUOffloadPolicy** (`fsdp_load.py`):
- PyTorch FSDP built-in with automatic all-gather/reduce-scatter
- `fully_shard()` wraps modules, FSDP handles H2D/D2H internally
- `reshard_after_forward=True` triggers automatic D2H after each module forward
- Higher-level abstraction, less control over async behavior

---

## Cross-Check Matrix

| Claim | Source Agent | Cross-Check | Status |
|---|---|---|---|
| Separate CUDA streams | Agent A | Agent C's `__init__` analysis | CONSISTENT |
| Step 10 is switch point | Prior analysis | Agent B's math (says step 18) | **RESOLVED** -- benchmark used flow_shift=3.0 (config misresolution), not 12.0 |
| No D2H copy | Agent C | Agent A's overlap model | CONSISTENT |
| Blocking `.to()` in old path | Agent C | Agent B's step spike | CONSISTENT (spike confirms blocking transfer) |
| PCIe contention on ACES | Agent A | Hardware specs | CONSISTENT |
| Mutual exclusion in server_args | Agent C | Agent C's line citations | VERIFIED |

---

## Remaining Uncertainties

1. ~~**Step 10 vs Step 18 switch point**~~ RESOLVED: benchmark used flow_shift=3.0 due to config misresolution (see Q3)
2. **Benchmark validity**: all existing benchmark data used flow_shift=3.0 instead of the intended 12.0. Tuesday re-runs need correct config.
3. **6-GPU scaling** -- does new offload become faster with more GPUs? (Tuesday data)
4. **Profiling overhead impact** -- Run 2 (clean) vs Run 3 (profiled) comparison pending
5. **Actual PCIe bandwidth utilization** -- could measure with `nvidia-smi dmon` during benchmark
6. **Impact of flow_shift on performance gap** -- with flow_shift=12.0, switch at step 18 means 9 more steps with new offload overhead before the old offload's blocking transfer. This may change the 7% gap.

---

## Recommendations

### Immediate (for PI)
1. Share this report with existing 4-GPU benchmark data
2. Note: "old offload 7% faster on 4 GPUs" but uses 3x more VRAM (61GB vs 23GB peak)
3. New offload designed for memory-constrained or high-GPU-count scenarios

### Tuesday Follow-up
1. Add timestep logging to benchmark scripts: `print(f"Step {i}: t_int={t_int}")`
2. Analyze 4-GPU vs 6-GPU scaling trend
3. Determine if 6-GPU narrows performance gap (hypothesis: yes, due to better compute/transfer overlap)

### Future Work
1. Topology-aware prefetch: reduce `prefetch_size` on PCIe systems with sequence parallelism
2. Test with 8 GPUs to match PR #15511's benchmark configuration
3. Profile actual PCIe bandwidth during offload to quantify contention

---

## Appendix: Source Files Analyzed

| File | Lines Read | Key Content |
|---|---|---|
| `utils/layerwise_offload.py` | Full (386 lines) | LayerwiseOffloadManager, OffloadableDiTMixin |
| `pipelines_core/stages/denoising.py` | 485-507, 837-884, 1010-1040 | Boundary ratio, device placement, denoising loop |
| `loader/fsdp_load.py` | 137-198 | FSDP CPUOffloadPolicy setup |
| `server_args.py` | 1013-1033 | Mutual exclusion logic |
| `models/dits/wanvideo.py` | 1-100, 424-461 | WanTransformerBlock forward, Ulysses attention |
| `distributed/device_communicators/base_device_communicator.py` | 140-145 | all_to_all_single |
| `layers/attention/layer.py` | 119-120 | Ulysses attention all-to-all call |
| `models/schedulers/scheduling_unipc_multistep.py` | 334-336, 461-480 | Flow matching schedule, set_shift |
| `configs/pipeline_configs/wan.py` | 208-213 | Wan2_2_T2V_A14B_Config |
| `managers/gpu_worker.py` | 130-137 | LayerwiseOffloadManager attachment |
| `loader/component_loaders/transformer_loader.py` | 95-113 | Transformer loading |
