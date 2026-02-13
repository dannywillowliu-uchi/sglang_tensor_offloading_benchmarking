# SGLang Offload Benchmark Results - 4x H100 GPUs

## Run Summary

| Metric | New Offload (PR patch) | Old Offload (baseline) | Pure GPU |
|--------|----------------------|----------------------|----------|
| Job ID | 1417509 | 1417511 | 1417510 |
| Node | ac055 | ac041 | - |
| GPUs | 4x H100 | 4x H100 | 4x H100 |
| `dit_layerwise_offload` | **true** | false | false |
| `dit_cpu_offload` | false | **true** (auto) | false |
| `text_encoder_cpu_offload` | true | true | false |
| `pin_cpu_memory` | true | true | - |
| Denoising time | **788.2s** | **714.1s** | OOM |
| Avg step time | 29.19s | 26.44s | - |
| Decoding time | 26.6s | 26.1s | - |
| Total time | 845.9s | 761.1s | - |
| Peak GPU memory | 22.88 GB | 61.25 GB | 79+ GB (OOM) |
| NSight profile | No (SLURM killed) | No (SLURM killed) | Yes (213MB, with errors) |

## Per-Step Timing Analysis

Actual per-step times (derived from tqdm cumulative timestamps):

| Step | New Offload | Old Offload | Delta |
|------|-------------|-------------|-------|
| 1 | **227s** (compile) | **169s** (compile) | +58s |
| 2 | 21s | 19s | +2s |
| 3 | 22s | 20s | +2s |
| 4 | 21s | 20s | +1s |
| 5 | 21s | 20s | +1s |
| 6 | 22s | 20s | +2s |
| 7 | 21s | 19s | +2s |
| 8 | 22s | 20s | +2s |
| 9 | 21s | 20s | +1s |
| 10 | 22s | **50s** (transformer switch) | -28s |
| 11 | 22s | 20s | +2s |
| 12 | 21s | 20s | +1s |
| 13 | 22s | 19s | +3s |
| 14 | 22s | 20s | +2s |
| 15 | 21s | 20s | +1s |
| 16 | 22s | 20s | +2s |
| 17 | 21s | 20s | +1s |
| 18 | 22s | 19s | +3s |
| 19 | 22s | 20s | +2s |
| 20 | 21s | 20s | +1s |
| 21 | 22s | 20s | +2s |
| 22 | 22s | 20s | +2s |
| 23 | 21s | 19s | +2s |
| 24 | 22s | 20s | +2s |
| 25 | 21s | 20s | +1s |
| 26 | 22s | 20s | +2s |
| 27 | 22s | 19s | +3s |

### Steady-state (steps 2-27, excluding step 10):
- **New offload**: ~21.6s/step
- **Old offload**: ~19.8s/step
- **New is 1.8s/step SLOWER** (9% overhead)

### Breakdown of total time difference:
- Step 1 compile overhead: +58s (new slower)
- Step 10 transformer switch: -28s (old has spike, new doesn't)
- Steady state (25 other steps): +45s (new slower at 1.8s/step)
- **Net: new is 75s slower** (matches observed 788-714 = 74s)

## Configuration Discrepancy vs PR #15511

### PR used 8 GPUs, we used 4 GPUs

PR #15511 benchmarked with:
```
--num-gpus 8 --ulysses-degree 8
```

Our runs used:
```
--num-gpus 4 --ulysses-degree 4
```

PR results (8 GPU): Old=149.69s, New=94.22s → **58% speedup**
Our results (4 GPU): Old=714.1s, New=788.2s → **10% slowdown**

### Key config difference found: `dit_cpu_offload`

Looking at the `server_args` from the logs:

**Our old offload** resolved to:
```json
"dit_cpu_offload": true,
"dit_layerwise_offload": false
```

**Our new offload** resolved to:
```json
"dit_cpu_offload": false,
"dit_layerwise_offload": true
```

The PR's "before" command does NOT explicitly set `--dit-cpu-offload` or `--dit-layerwise-offload`.
SGLang **auto-enables** `dit_cpu_offload` for Wan models by default.
When we set `--dit-layerwise-offload false`, SGLang fell back to `dit_cpu_offload=true`.

This means our comparison IS correct - we're comparing the same two modes the PR compared:
- Before: full-transformer CPU offload (`dit_cpu_offload`)
- After: per-layer async CPU offload (`dit_layerwise_offload`)

## Why Is Layerwise Offload Slower on 4 GPUs?

### 1. Per-layer transfer size is larger with fewer GPUs
With sequence parallelism (ulysses_degree), model weights are sharded across GPUs.
- **8 GPUs**: Each GPU holds 1/8 of weight per layer → small H2D transfers, easier to hide with async prefetch
- **4 GPUs**: Each GPU holds 1/4 of weight per layer → 2x larger H2D transfers per GPU

The async prefetch in the PR is designed to overlap layer N+1's H2D transfer with layer N's compute.
With larger transfers (4 GPU), the transfer time may exceed compute time, breaking the overlap.

### 2. Steady-state overhead: 21.6s vs 19.8s per step
Layerwise offload moves individual layers on/off GPU every step.
Full-transformer offload (`dit_cpu_offload`) keeps the entire transformer on GPU during all denoising steps and only offloads when switching between transformer_1 and transformer_2.

With 4 GPUs and sufficient VRAM (61.25 GB peak for old offload), the full transformer fits in GPU memory, so `dit_cpu_offload` has zero transfer overhead during steady-state inference.

### 3. The one-time spike is cheaper than per-step overhead
Old offload has a single 50s spike at step 10 (transformer switch) but saves ~1.8s on each of 26 other steps (46.8s saved). The spike costs less than the cumulative per-step savings.

### 4. Memory vs Speed tradeoff
- **New offload** (layerwise): 22.88 GB peak → very memory efficient, but slower
- **Old offload** (full): 61.25 GB peak → uses 3x more VRAM, but faster on 4 GPUs

The layerwise offload is designed for memory-constrained scenarios (fewer GPUs, smaller VRAM).
On 4x H100 (80GB each = 320GB total), there's plenty of VRAM, making the old approach viable and faster.

## Memory Analysis

| Component | New Offload | Old Offload |
|-----------|-------------|-------------|
| Peak GPU memory | 22.88 GB | 61.25 GB |
| Free at peak | 56.77 GB | 18.40 GB |
| Max peak (MB) | 23,426 | 62,717 |
| Resident components | text_encoder, vae, transformer | vae only |

The new offload uses ~3x less GPU memory, which is the primary benefit of layerwise offloading.
This tradeoff makes more sense when VRAM is scarce (fewer GPUs, larger models).

## Conclusion

**Our 4-GPU results are valid but not comparable to the PR's 8-GPU benchmark.**

The PR's 58% speedup with layerwise offload is specific to the 8-GPU configuration where:
1. Per-layer transfers are small enough for async prefetch to fully hide latency
2. The overhead of frequent small transfers < the overhead of infrequent large transfers

On 4 GPUs, the dynamics flip: per-layer transfers are too large to hide, making layerwise offload slower than full-transformer offload in absolute time.

**To reproduce the PR's results, we need to run with 8 GPUs and ulysses-degree 8.**

## Issues to Fix for Next Run

1. **GPU count**: Must use 8 GPUs to match PR → FIXED in updated scripts
2. **Time limit**: 20 min insufficient, nsys killed before profile write → FIXED to 30 min
3. **CUDA module**: nsys from CUDA 12.4 incompatible with PyTorch CUDA 12.8 → FIXED to CUDA 12.8.0
4. **Pure GPU OOM**: 4 GPUs not enough → FIXED to 8 GPUs
