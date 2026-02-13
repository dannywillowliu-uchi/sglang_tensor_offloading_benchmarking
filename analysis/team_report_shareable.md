# SGLang Layerwise Offloading Analysis: Wan2.2 Video Generation

**Date:** February 2026
**Hardware:** ACES cluster, 4x/6x H100 PCIe GPUs, 488GB RAM
**Software:** SGLang (PR #15511), Wan2.2-T2V-A14B-Diffusers (dual 14B transformers)

---

## 1. Background

SGLang PR #15511 introduces a new **layerwise offloading** strategy (`dit_layerwise_offload`) for video diffusion models. We evaluated it against the existing **full-model offloading** (`dit_cpu_offload`) to understand the performance tradeoffs on PCIe-based GPU systems.

Wan2.2-T2V-A14B uses two 14B-parameter transformers in a mixture-of-experts (MoE) configuration at the timestep level: one processes high-noise steps, the other handles low-noise steps. During a 27-step denoising run, the model switches from one transformer to the other at a boundary determined by `flow_shift` and `boundary_ratio`.

---

## 2. How the Two Offload Strategies Work

We verified both implementations directly from source code with line-number citations.

### Full-Model Offload (Old)

- Transfers the **entire transformer** (~28GB) between CPU and GPU using blocking `.to(device)` calls
- At the transformer switch point, the active model moves to CPU and the other moves to GPU
- Uses PyTorch FSDP `CPUOffloadPolicy` for automatic weight management
- One large spike in latency at the switch point, but fast steady-state steps

### Layerwise Offload (New, PR #15511)

- Transfers **one layer at a time** (~700MB) using a dedicated CUDA copy stream with async H2D prefetch
- CPU pinned memory buffer is the source of truth -- no GPU-to-CPU (D2H) copies ever occur
- Pre-hooks prefetch the next layer(s) before they're needed; post-hooks release GPU memory immediately
- No spike at the switch point, but adds ~2s overhead per step on our hardware

### Comparison

| Aspect | Full-Model (Old) | Layerwise (New) |
|--------|------------------|-----------------|
| Transfer granularity | Entire transformer (~28GB) | Per-layer (~700MB) |
| Transfer method | Blocking `.to()` | Async H2D on dedicated stream |
| GPU-to-CPU copy | Yes (full model) | None (CPU buffer unchanged) |
| Peak VRAM | ~61 GB | ~23 GB |
| Switch point cost | ~8s blocking spike | No spike |
| Per-step overhead | Minimal | ~2s on PCIe hardware |

---

## 3. PCIe Limitation: Why Layerwise Offload Underperforms on Our Hardware

PR #15511's async prefetch strategy is designed to overlap data transfers with computation. This works on **NVLink systems** (like H100 SXM) where GPU-to-GPU communication (used by Ulysses sequence parallelism) travels over NVLink while CPU-to-GPU transfers use PCIe -- two separate physical paths.

On **ACES H100 PCIe**, both GPU-GPU and CPU-GPU transfers share the same PCIe fabric (~64 GB/s unidirectional). The async prefetch cannot overlap with Ulysses all-to-all communication because they compete for the same bandwidth.

| System | GPU-GPU Path | CPU-GPU Path | Can Overlap? |
|--------|-------------|-------------|-------------|
| H100 SXM (NVLink) | NVLink (900 GB/s) | PCIe (128 GB/s) | Yes |
| H100 PCIe (ACES) | PCIe (128 GB/s) | PCIe (128 GB/s) | No |

This is likely why PR #15511's benchmarks (run on NVLink systems) show better performance than what we observe.

---

## 4. Preliminary Benchmark Results (4 GPUs)

| Metric | Layerwise (New) | Full-Model (Old) |
|--------|----------------|-----------------|
| Total time | 12:00 | 11:12 |
| Denoising time | 788.2s | 714.1s |
| Steady-state per step | ~23s | ~21s |
| Switch step spike | None | ~8s |
| Peak GPU memory | 22.88 GB | 61.25 GB |

**Old offload was ~7% faster on 4 GPUs** despite having a blocking transfer spike. The new offload's per-step overhead (~2s x 26 steps = ~52s total) exceeds the old offload's one-time spike cost.

However, the new offload uses **2.7x less GPU memory** (23 GB vs 61 GB), which is its primary design goal for memory-constrained deployments.

### Caveat: Config Issue Discovered

These benchmarks ran with an incorrect `flow_shift` parameter (3.0 instead of 12.0) due to a SGLang model config resolution bug when using local model paths. This caused the transformer switch to happen at step 10 instead of the correct step 19. We are re-running benchmarks with the correct config. The 7% figure may change.

See Section 5 for details.

---

## 5. Config Resolution Bug Found

We discovered that SGLang's model registry does not correctly resolve local model paths for Wan2.2-A14B.

**The issue:** When using `--model-path ./models/wan2.2`, SGLang's registry (`registry.py`) cannot match the path to the correct `Wan2_2_T2V_A14B_Config` class. It falls through to a generic Wan config (`WanT2V480PConfig`) which sets `flow_shift=3.0` instead of the correct `12.0`.

**Why it matters:** `flow_shift` controls the denoising timestep schedule. With the wrong value, the transformer switch point changes:

| flow_shift | Switch step (out of 27) | Config class |
|-----------|------------------------|-------------|
| 3.0 (incorrect) | Step 10 | WanT2V480PConfig (generic) |
| 12.0 (correct) | Step 19 | Wan2_2_T2V_A14B_Config |

### PR #15511's Reference Numbers (8x GPUs, NVLink)

The PR author's benchmarks confirm the step 19 switch and provide a reference for expected behavior on NVLink hardware:

**Before PR (old `dit_cpu_offload`, 8x GPU NVLink):**

| Step | Time | Notes |
|------|------|-------|
| Step 0 | ~36.0s | Both transformers loaded CPU to GPU |
| Steps 1-17 | ~3.27s each | Steady-state compute |
| Step 18 (19th) | ~31.3s | Full-model swap (blocking) |
| Steps 19-26 | ~3.26s each | Steady-state compute |
| **Total** | **149.7s** | |

**After PR (new `dit_layerwise_offload`, 8x GPU NVLink):**

| Step | Time | Notes |
|------|------|-------|
| Step 0 | ~7.7s | Reduced by NCCL All2All warmup |
| Steps 1-26 | ~3.29s each | Uniform -- no switch spike |
| **Total** | **94.2s** | **58% speedup** |

On NVLink, the PR eliminates both the first-step spike (~36s to ~7.7s via NCCL warmup) and the switch spike (~31s to ~3.3s via layerwise streaming). The CPU-to-GPU transfers overlap completely with GPU compute because they use separate physical paths (PCIe vs NVLink).

**Resolution:** Use the HuggingFace model identifier (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) instead of a local directory path. This matches the exact path registered in SGLang's config registry. All benchmark scripts have been updated.

---

## 6. Next Steps

1. **Re-run benchmarks** with corrected model path (4-GPU and 6-GPU, both offload strategies)
2. **6-GPU scaling comparison** -- does layerwise offload become competitive with more GPUs?
3. **Add timestep logging** to confirm correct config is loaded (`flow_shift=12.0`)
4. **Profile PCIe bandwidth** during offload to quantify contention between Ulysses and H2D transfers

---

## 7. Optimization Insights

Based on the PR's NVLink results and our PCIe analysis, several optimization directions emerge:

### Why NVLink Wins

The PR achieves 58% speedup because CPU-to-GPU layer streaming runs on PCIe while Ulysses all-to-all runs on NVLink -- zero contention. On our PCIe hardware, both share the same bus, turning "zero-cost" overlap into ~2s/step overhead.

### Possible PCIe-Specific Optimizations

1. **Hybrid offload strategy**: Use full-model offload (old approach) when VRAM permits, layerwise only when memory-constrained. The old approach pays one ~8s spike vs ~52s cumulative overhead from layerwise on PCIe.

2. **Quantized weight transfers**: Transfer layers in INT8/FP8 (~350MB instead of ~700MB per layer) and dequantize on GPU. Halves PCIe transfer time per layer, potentially reducing per-step overhead.

3. **Reduced Ulysses degree**: Fewer GPUs in Ulysses parallelism = less all-to-all traffic = more PCIe bandwidth available for offload transfers. Trade-off: slower per-step compute. Worth testing 2-GPU vs 4-GPU Ulysses.

4. **Schedule-aware prefetch timing**: The current implementation prefetches every step uniformly. Since the denoising schedule (via `flow_shift`) determines compute intensity per step, prefetch could be scheduled during compute-heavy steps where GPU is busier and PCIe has more slack.

5. **NCCL warmup adoption**: The PR's All2All warmup logic reduced first-step cost from ~36s to ~7.7s on NVLink. Verify this warmup is active in our runs -- it should help regardless of interconnect.

### What Tuesday's Benchmarks Will Clarify

- Whether the 7% gap widens or narrows with correct `flow_shift=12.0` (switch at step 19 vs 10 changes overhead distribution)
- Whether 6-GPU scaling helps layerwise offload (more compute per step = more time to hide transfers) or hurts it (more Ulysses traffic = more PCIe contention)

---

## 8. Key Takeaways

- Layerwise offloading (PR #15511) trades speed for memory -- 2.7x less VRAM but ~7% slower on 4 PCIe GPUs (pending re-validation with correct config)
- On NVLink, the PR achieves 58% speedup by eliminating both the first-step and switch-step spikes entirely
- The "zero-cost" prefetch claim requires NVLink for compute/transfer overlap; PCIe systems see per-step overhead due to bus contention
- A config resolution bug in SGLang affected our benchmark parameters -- discovered during source code verification
- Primary optimization opportunity for PCIe: hybrid offload strategy or quantized weight transfers to reduce per-step overhead
- Corrected benchmarks are queued for Tuesday re-run
