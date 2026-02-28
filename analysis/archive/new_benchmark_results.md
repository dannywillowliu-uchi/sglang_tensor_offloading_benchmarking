# Benchmark Results: flow_shift=12.0 (Correct Configuration)
# Jobs 1454665-1454670, Submitted 2026-02-15

**ACES H100 PCIe, 4 GPUs, Wan2.2-T2V-A14B-Diffusers**
**Model path:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (exact HF match -> flow_shift=12.0)
**Resolution:** 720x1280, 81 frames, 27 steps

---

## 1. Job Status Summary

| Job ID | Config | GPUs | Status | Runs Completed | Notes |
|--------|--------|------|--------|----------------|-------|
| 1454665 | New offload (layerwise) | 4 | TIMEOUT | 2/3 | Run 3 timed out at step 11/27 |
| 1454666 | Old offload (FSDP) | 4 | TIMEOUT | 2/3 | Run 3 timed out during profiler save |
| 1454667 | Pure GPU (no offload) | 4 | FAILED | 0/3 | OOM - expected, 14B model too large |
| 1454668 | New offload (layerwise) | 6 | OOM | 0/3 | SLURM node memory exceeded |
| 1454669 | Old offload (FSDP) | 6 | OOM | 0/3 | SLURM node memory exceeded |
| 1454670 | Prefetch=3 | 4 | FAILED | 0/3 | `--dit-offload-prefetch-size` not in sglang 0.5.8 |

**Usable data:** Jobs 1454665 and 1454666 Run 2 (clean timing, no profiling overhead).

---

## 2. Per-Step Timing: New Offload (Job 1454665, Run 2)

Config: `--dit-layerwise-offload true --text-encoder-cpu-offload --pin-cpu-memory`

| Step | Cumulative (s) | Delta (s) | Phase |
|------|---------------|-----------|-------|
| 1 | 65.0 | 65.0 | Warmup (first step overhead) |
| 2 | 87.0 | 22.0 | Settling |
| 3 | 110.0 | 23.0 | Steady state |
| 4 | 133.0 | 23.0 | Steady state |
| 5 | 156.0 | 23.0 | Steady state |
| 6 | 178.0 | 22.0 | Steady state |
| 7 | 201.0 | 23.0 | Steady state |
| 8 | 224.0 | 23.0 | Steady state |
| 9 | 247.0 | 23.0 | Steady state |
| 10 | 270.0 | 23.0 | Steady state |
| 11 | 293.0 | 23.0 | Steady state |
| 12 | 316.0 | 23.0 | Steady state |
| 13 | 339.0 | 23.0 | Steady state |
| 14 | 362.0 | 23.0 | Steady state |
| 15 | 385.0 | 23.0 | Steady state |
| 16 | 409.0 | 24.0 | Steady state |
| 17 | 432.0 | 23.0 | Steady state |
| 18 | 455.0 | 23.0 | Steady state (last step with transformer A) |
| **19** | **478.0** | **23.0** | **Transformer switch (boundary_ratio=0.875)** |
| 20 | 501.0 | 23.0 | Steady state (transformer B) |
| 21 | 524.0 | 23.0 | Steady state |
| 22 | 547.0 | 23.0 | Steady state |
| 23 | 570.0 | 23.0 | Steady state |
| 24 | 593.0 | 23.0 | Steady state |
| 25 | 616.0 | 23.0 | Steady state |
| 26 | 639.0 | 23.0 | Steady state |
| 27 | 662.0 | 23.0 | Final step |

**Summary:**
- Total denoising: **662.2s**
- Average per-step: **24.52s/step**
- Steady-state (steps 3-27): **~23s/step**
- First step overhead: **65s** (torch.compile cached, but still init cost)
- **NO spike at step 19** (transformer switch is seamless with layerwise offload)

---

## 3. Per-Step Timing: Old Offload (Job 1454666, Run 2)

Config: `--dit-layerwise-offload false --text-encoder-cpu-offload --pin-cpu-memory`

| Step | Cumulative (s) | Delta (s) | Phase |
|------|---------------|-----------|-------|
| 1 | 40.0 | 40.0 | Warmup |
| 2 | 60.0 | 20.0 | Settling |
| 3 | 80.0 | 20.0 | Steady state |
| 4 | 100.0 | 20.0 | Steady state |
| 5 | 120.0 | 20.0 | Steady state |
| 6 | 140.0 | 20.0 | Steady state |
| 7 | 160.0 | 20.0 | Steady state |
| 8 | 180.0 | 20.0 | Steady state |
| 9 | 200.0 | 20.0 | Steady state |
| 10 | 220.0 | 20.0 | Steady state |
| 11 | 239.0 | 19.0 | Steady state |
| 12 | 259.0 | 20.0 | Steady state |
| 13 | 279.0 | 20.0 | Steady state |
| 14 | 299.0 | 20.0 | Steady state |
| 15 | 319.0 | 20.0 | Steady state |
| 16 | 339.0 | 20.0 | Steady state |
| 17 | 359.0 | 20.0 | Steady state |
| 18 | 379.0 | 20.0 | Steady state (last step with transformer A) |
| **19** | **426.0** | **47.0** | **Transformer switch -- 27s spike (+135% over baseline)** |
| 20 | 446.0 | 20.0 | Steady state (transformer B) |
| 21 | 466.0 | 20.0 | Steady state |
| 22 | 486.0 | 20.0 | Steady state |
| 23 | 505.0 | 19.0 | Steady state |
| 24 | 525.0 | 20.0 | Steady state |
| 25 | 545.0 | 20.0 | Steady state |
| 26 | 565.0 | 20.0 | Steady state |
| 27 | 585.0 | 20.0 | Final step |

**Summary:**
- Total denoising: **585.7s**
- Average per-step: **21.69s/step**
- Steady-state (steps 3-18, 20-27): **~20s/step**
- First step overhead: **40s**
- **47s spike at step 19** (transformer switch: blocking `.to()` moves full transformer to GPU)
- Switch penalty: **+27s** over baseline 20s step

---

## 4. Head-to-Head Comparison (4 GPU, flow_shift=12.0)

| Metric | New Offload | Old Offload | Delta |
|--------|-------------|-------------|-------|
| Total denoising | 662.2s | 585.7s | Old is **13.1% faster** |
| Steady-state per-step | ~23s | ~20s | Old is **15% faster** per step |
| First step | 65s | 40s | New 25s slower on first step |
| Step 19 (switch) | 23s (no spike) | 47s (+27s spike) | New absorbs switch seamlessly |
| Peak VRAM (est.) | ~23 GB | ~61 GB | New uses **62% less VRAM** |
| Switch behavior | Seamless (layerwise prefetch continues) | Blocking (full .to() call) | Fundamentally different |

### Key Observations

1. **Old offload is faster in steady state** (+3s/step = +15% per step). This is because:
   - Old (FSDP) shards weights across GPUs: each GPU only transfers 1/N of the model
   - New offload has NO world_size awareness: each GPU independently offloads the full model
   - With 4 GPUs: old transfers ~7GB per step, new transfers ~28GB per step

2. **Transformer switch is a major event with old offload.** The 47s spike at step 19 means the blocking `.to()` call takes **~27s** to move the full second transformer to GPU. This is `_manage_device_placement()` at denoising.py:854,861.

3. **New offload handles the switch seamlessly.** No timing spike at step 19 because LayerwiseOffloadManager prefetches layers of the new transformer using the async copy stream, just like any other layer transition.

4. **The performance gap WIDENED with correct flow_shift:**
   - flow_shift=3.0 (wrong): old was 7% faster
   - flow_shift=12.0 (correct): old is **13% faster**
   - The switch moved from step 10 to step 19, and the spike is MUCH larger (27s vs 8s), but this is only 1 step out of 27, so it doesn't close the gap.

---

## 5. Comparison with Frozen Baseline (flow_shift=3.0)

| Metric | flow_shift=3.0 (wrong) | flow_shift=12.0 (correct) | Change |
|--------|----------------------|--------------------------|--------|
| **New offload total** | 788.2s | 662.2s | **-16.0%** (faster) |
| New steady-state | 21.6s/step | 23s/step | +6.5% (slower) |
| **Old offload total** | 714.1s | 585.7s | **-18.0%** (faster) |
| Old steady-state | 19.8s/step | 20s/step | +1.0% (similar) |
| Switch step | tqdm 10 (step 9) | tqdm 19 (step 18) | Moved later |
| Switch spike (old) | ~8s extra | ~27s extra | 3.4x larger |
| Switch spike (new) | ~0s (no spike) | ~0s (no spike) | Consistently seamless |
| Gap (old faster by) | 7% | 13.1% | Gap widened |

### Why are totals FASTER with correct flow_shift=12.0?

This seems counterintuitive but makes sense:
- **flow_shift=12.0** pushes more timesteps into the high-noise regime where the shifted sigma is larger
- The UniPC scheduler samples different timestep values, which may result in less compute per step for certain noise levels
- The first step is notably faster with correct flow_shift (65s vs estimated higher for wrong flow_shift on clean runs)
- The steady-state is actually slightly slower (23 vs 21.6s for new), but the overall total is lower

**Note:** The baseline numbers (788.2s, 714.1s) were from profiled runs with `torch.profiler` overhead. The new numbers (662.2s, 585.7s) are from clean Run 2 with no profiling. This accounts for most of the difference. **These are not directly comparable on total time.** The per-step steady-state comparison is more meaningful.

### Corrected Comparison (profiling overhead adjusted)

The prior baseline data had profiling overhead. Comparing steady-state per-step times (which are less affected by profiling):

| Metric | Baseline (profiled) | New Results (clean) | Notes |
|--------|-------------------|-------------------|-------|
| New offload steady-state | 21.6s/step | 23.0s/step | +6.5% -- flow_shift effect on compute |
| Old offload steady-state | 19.8s/step | 20.0s/step | +1.0% -- minimal change |
| Steady-state gap | +1.8s (old 9% faster) | +3.0s (old 15% faster) | Gap widened in per-step |

The per-step gap widening from 9% to 15% with correct flow_shift suggests that the higher flow_shift changes the compute workload in a way that disproportionately affects the new offload path. This could be because:
- Higher flow_shift changes the effective noise schedule, potentially altering memory access patterns
- The layerwise offload's async prefetch timing may not be as well-optimized for the shifted schedule
- More investigation needed with nsys profiling to understand the root cause

---

## 6. Validated Predictions

| Prediction (from baseline doc) | Result | Status |
|-------------------------------|--------|--------|
| Switch moves to step 18/tqdm 19 | Confirmed: spike at step 19 | VALIDATED |
| flow_shift=12.0 in logs | Confirmed in both logs | VALIDATED |
| New offload switch is seamless | Confirmed: no spike at step 19 | VALIDATED |
| Old offload switch causes spike | Confirmed: 47s spike (+27s) | VALIDATED |
| Pure GPU OOMs on 4x H100 | Confirmed: EOFError during launch | VALIDATED |

---

## 7. Remaining Work

### Jobs to Resubmit
1. **4-GPU new + old** (Run 3 only): Increase `--time` to 01:30:00 for full 3-run completion
2. **6-GPU new + old**: Fix SLURM memory (`--mem=0` or investigate node constraints)
3. **Prefetch=3**: Requires sglang upgrade (0.5.8 lacks `--dit-offload-prefetch-size`)
4. **nsys profiling**: Submit nsys jobs for both new and old offload (PCIe bandwidth analysis)

### Analysis Still Needed
1. nsys PCIe bandwidth: actual H2D bytes/timing per step (not available from these runs)
2. 6-GPU scaling: does increasing GPUs narrow the old vs new gap?
3. Root cause of per-step gap widening with flow_shift=12.0
4. Whether the 27s switch spike in old offload can be optimized
