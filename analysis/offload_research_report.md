# SGLang Offload Research Report: Source Code Verification

**Date:** 2026-02-18 (updated from 2026-02-13)
**Authors:** Parallel source code analysis agents (A, B, C) + cross-check
**Codebase:** sglang-src (HEAD, shallow clone)
**Prior work:** Obsidian note `Claude/Projects/sglang-offload-research.md`

---

## Executive Summary

This report provides code-verified answers to 6 research questions about SGLang PR #15511's layerwise offloading for Wan2.2 video generation. Three parallel analysis agents examined the source code with line-number citations. Key findings:

1. **All prior claims about offload implementations verified** -- the OLD path uses blocking `.to()`, NEW path uses async H2D prefetch with no D2H copy
2. **Ulysses overlap NOT feasible on ACES H100 PCIe** -- both GPU-GPU and CPU-GPU share PCIe fabric
3. **Step 10 contradiction FULLY RESOLVED** -- benchmarks used base `WanT2V480PConfig` (flow_shift=3.0) via config misresolution instead of correct `Wan2_2_T2V_A14B_Config` (flow_shift=12.0). flow_shift=3.0 -> switch at step 9/tqdm 10 (matches benchmarks). flow_shift=12.0 -> switch at step 18/tqdm 19.
4. **NEW BENCHMARK RESULTS (2026-02-18)**: Jobs 1454665-1454666 completed Run 2 (clean timing) with correct flow_shift=12.0. Old offload is **13% faster** (was 7% with wrong flow_shift). Switch confirmed at step 19. Gap widened due to per-step overhead increase in new offload.
5. **Batch 3 submitted** -- Jobs 1459995-1460000: 4-GPU rerun (90 min), 6-GPU (mem fix), nsys profiling

---

## Q1: Benchmark Results

### NEW: Clean 4-GPU Data with Correct flow_shift=12.0 (Jobs 1454665/1454666, Run 2)

| Metric | New Offload (layerwise) | Old Offload (FSDP) | Delta |
|---|---|---|---|
| Total denoising | 662.2s | **585.7s** | Old **13.1% faster** |
| Steady-state per-step | ~23s | ~20s | Old **15% faster** per step |
| First step | 65s | 40s | New 25s slower |
| Step 19 (switch) | 23s (no spike) | **47s (+27s spike)** | New absorbs switch seamlessly |
| Peak VRAM (est.) | ~23 GB | ~61 GB | New uses **62% less VRAM** |

**Key findings:**
1. **Old offload is 13% faster on 4 GPUs** (up from 7% with wrong flow_shift). The per-step gap is actually 15% (3s/step), driven by FSDP sharding each GPU only transferring 1/N of the model vs new offload independently moving the full model on each GPU.
2. **Transformer switch confirmed at step 19** (tqdm 19), validating flow_shift=12.0. Old offload has a 47s spike (27s overhead) due to blocking `.to()` at `denoising.py:854,861`. New offload handles the switch seamlessly.
3. **Per-step gap widened** from 9% (flow_shift=3.0) to 15% (flow_shift=12.0), suggesting the shifted noise schedule disproportionately affects the new offload path.

### Prior 4-GPU Data (Jobs 1434113/1434114) -- flow_shift=3.0, INVALID but instructive

| Metric | New Offload (layerwise) | Old Offload (full-model) |
|---|---|---|
| Total time | 12:00 (788.2s denoising) | **11:12 (714.1s denoising)** |
| Step 1 (compile) | ~122s | ~110s |
| Steady state | ~23s/step | ~21s/step |
| Step 10 anomaly | 24.2s (normal) | 29.8s (spike) |
| Peak GPU memory | 22.88 GB | 61.25 GB |

**Note:** These totals include profiling overhead. Per-step steady-state times are comparable across runs.

### Comparison: flow_shift=3.0 vs 12.0

| Metric | flow_shift=3.0 (wrong) | flow_shift=12.0 (correct) | Change |
|---|---|---|---|
| New steady-state | 21.6s/step | 23s/step | +6.5% (slower) |
| Old steady-state | 19.8s/step | 20s/step | +1.0% (similar) |
| Switch step | tqdm 10 | tqdm 19 | Moved 9 steps later |
| Switch spike (old) | ~8s extra | ~27s extra | 3.4x larger |
| Switch spike (new) | none | none | Consistently seamless |
| Gap | Old 7% faster | Old 13% faster | Gap widened |

### Job History

**Batch 1 -- Jobs 1439552-1439555: CANCELLED** (never ran, Feb 8)

**Batch 2 -- Jobs 1443732-1443737: FAILED** (compute nodes no internet for HF download)
- Fix applied: symlink `$SCRATCH/.../Wan-AI/Wan2.2-T2V-A14B-Diffusers -> ../models/wan2.2`

**Batch 2b -- Jobs 1454665-1454670 (Feb 15):**

| Job | Config | GPUs | Status | Usable Data |
|---|---|---|---|---|
| 1454665 | New offload | 4 | TIMEOUT (45 min) | Run 1 + Run 2 complete |
| 1454666 | Old offload | 4 | TIMEOUT (45 min) | Run 1 + Run 2 complete |
| 1454667 | Pure GPU | 4 | OOM | None (expected) |
| 1454668 | New offload | 6 | SLURM OOM | None |
| 1454669 | Old offload | 6 | SLURM OOM | None |
| 1454670 | Prefetch=3 | 4 | FAILED | None (`--dit-offload-prefetch-size` not in sglang 0.5.8) |

**Batch 3 -- Jobs 1459995-1460000 (Feb 18, submitted):**

| Job | Config | GPUs | Fix Applied |
|---|---|---|---|
| 1459995 | New offload | 4 | `--time=01:30:00` (was 45 min) |
| 1459996 | Old offload | 4 | `--time=01:30:00` |
| 1459997 | New offload | 6 | `--mem=0` (use all node memory) |
| 1459998 | Old offload | 6 | `--mem=0` |
| 1459999 | nsys new offload | 4 | `--time=01:30:00` |
| 1460000 | nsys old offload | 4 | `--time=01:30:00` |

**Remaining action items:**
1. Retrieve Batch 3 results when complete
2. Extract nsys PCIe bandwidth (Memcpy HtoD events)
3. Compare 4-GPU vs 6-GPU scaling
4. Upgrade sglang on ACES for prefetch=3 testing (0.5.8 lacks CLI arg)

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

1. ~~**Step 10 vs Step 18 switch point**~~ RESOLVED: config misresolution. flow_shift=3.0 -> step 10, flow_shift=12.0 -> step 19. Confirmed in Batch 2b results.
2. ~~**Benchmark validity**~~ RESOLVED: Batch 2b jobs used correct HF path -> flow_shift=12.0 confirmed in logs.
3. ~~**Impact of flow_shift on performance gap**~~ ANSWERED: gap widened from 7% to 13%. Per-step gap widened from 9% to 15%.
4. **Pure GPU baseline** -- 4x H100 80GB OOM confirmed (job 1454667). Would need 8 GPUs or smaller model.
5. **Prefetch=3 performance** -- sglang 0.5.8 on ACES lacks `--dit-offload-prefetch-size`. Needs upgrade.
6. **Actual PCIe bandwidth utilization** -- nsys jobs 1459999/1460000 submitted, awaiting results
7. **6-GPU scaling** -- jobs 1459997/1459998 submitted with `--mem=0`, awaiting results
8. **Root cause of per-step gap widening** -- why does flow_shift=12.0 hurt new offload more than old? Possibly memory access pattern change or prefetch timing mismatch.
9. **2-GPU scaling test** -- if PCIe contention confirmed in nsys, test with fewer GPUs

---

## Recommendations

### Immediate (for PI)
1. **Share updated results:** old offload is 13% faster on 4 GPUs with correct flow_shift=12.0, but uses 3x more VRAM (61GB vs 23GB peak)
2. **Key insight:** New offload's advantage is memory efficiency (62% less VRAM), not speed. On PCIe systems without NVLink, the per-step overhead from independent full-model offloading on each GPU is the bottleneck.
3. **Transformer switch analysis:** New offload handles the switch seamlessly (no spike), while old offload has a 47s spike. But this single-step penalty doesn't overcome the accumulated per-step advantage.

### Awaiting Batch 3 Results (Jobs 1459995-1460000)
1. **4-GPU rerun** (1459995-1459996): Should get Run 3 profiled data this time (90 min limit)
2. **6-GPU scaling** (1459997-1459998): Will show if more GPUs narrow the gap (hypothesis: new offload benefits less from more GPUs since it doesn't shard)
3. **nsys profiling** (1459999-1460000): Will give actual PCIe HtoD bandwidth numbers

### Future Work
1. Topology-aware prefetch: reduce `prefetch_size` on PCIe systems with sequence parallelism
2. Upgrade sglang on ACES for prefetch=3 testing
3. Test with 8 GPUs to match PR #15511's benchmark configuration
4. Investigate per-step gap widening with flow_shift=12.0

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
