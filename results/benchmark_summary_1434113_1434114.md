# Benchmark Results: Jobs 1434113 (New) vs 1434114 (Old)

## Run Configuration
- **Date**: 2026-02-05
- **Node**: ac055 (both jobs ran on same node)
- **GPUs**: 4x H100 (PCIe)
- **Model**: Wan2.2 (dual 14B transformers)
- **Resolution**: 720x1280, 81 frames, 27 denoising steps

## Flag Comparison

| Config | `--dit-layerwise-offload` | `dit_cpu_offload` (internal) |
|--------|---------------------------|------------------------------|
| New offload (1434113) | `true` | `false` |
| Old offload (1434114) | `false` | `true` |

## Timing Results

| Metric | New Offload | Old Offload | Difference |
|--------|-------------|-------------|------------|
| Total time | 12:00 | 11:12 | -48s (7% faster) |
| Step 1 (torch.compile) | ~122s | ~110s | -12s |
| Step 2-9 avg | ~23s | ~21s | -2s |
| Step 10 (transformer switch) | ~24s | ~29.8s | +5.8s spike |
| Step 11-27 avg | ~23s | ~22s | -1s |

## Profiler Trace Analysis

| Metric | New Offload | Old Offload |
|--------|-------------|-------------|
| Trace size | 235 MB | 455 MB |
| cudaMemcpy > 10s | 44 | 936 |
| cudaMemcpy > 100s | 1 | 79 |

## Key Findings

1. **Old offload is 7% faster on 4 GPUs** despite having a large spike at step 10 (transformer switch).

2. **New offload's per-step overhead accumulates**: ~2s/step extra × 26 steps = 52s total overhead, which exceeds the 16s step 10 spike cost.

3. **Step 10 spike explained**: Wan2.2 uses dual transformers with `boundary_ratio=0.875`. At step 10 (27×0.875≈23.6, so steps 1-23 use high-noise, 24-27 use low-noise when reversed), old offload swaps the entire ~56GB transformer.

4. **Profiling overhead**: SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1 adds cuda.synchronize() calls which may affect timing accuracy.

## Trace Files
- `traces/new_offload_1434113.trace.json.gz` (235 MB, truncated)
- `traces/old_offload_1434114.trace.json.gz` (455 MB, truncated)

## Validity Notes
- Single run (not statistically significant)
- Profiling enabled (adds overhead)
- Same node ensures fair hardware comparison
- torch.compile warmup included in run
