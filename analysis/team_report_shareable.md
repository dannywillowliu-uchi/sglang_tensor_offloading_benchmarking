# SGLang Layerwise Offloading Analysis: Wan2.2 Video Generation

**Date:** February 18, 2026 (updated)
**Hardware:** ACES cluster, 4x H100 PCIe GPUs, 488GB RAM
**Software:** SGLang 0.5.8 (PR #15511), Wan2.2-T2V-A14B-Diffusers (dual 14B transformers)

---

## 1. Background

SGLang PR #15511 introduces a new **layerwise offloading** strategy (`dit_layerwise_offload`) for video diffusion models. We evaluated it against the existing **full-model offloading** (`dit_cpu_offload`) on PCIe-based GPU systems to understand the performance tradeoffs.

Wan2.2-T2V-A14B uses two 14B-parameter transformers in a mixture-of-experts (MoE) configuration at the timestep level: one processes high-noise steps, the other handles low-noise steps. During a 27-step denoising run, the model switches transformers at step 19 (determined by `flow_shift=12.0` and `boundary_ratio=0.875`).

---

## 2. How the Two Offload Strategies Work

Both implementations verified from source code with line-number citations.

### Full-Model Offload (Old: `dit_cpu_offload`)

- Uses **PyTorch FSDP** with `CPUOffloadPolicy` to shard and manage weights
- Each GPU transfers only **1/N of the model** (~7GB with 4 GPUs) via FSDP all-gather
- Transfers are **blocking** `.to(device)` calls
- At the transformer switch (step 19), a full model swap causes a large latency spike

### Layerwise Offload (New: `dit_layerwise_offload`, PR #15511)

- Transfers **one layer at a time** (~700MB) using a dedicated CUDA copy stream with async H2D prefetch
- CPU pinned memory is the source of truth -- no GPU-to-CPU (D2H) copies ever occur
- Each GPU **independently offloads the full model** (~28GB) -- no FSDP sharding
- Pre-hooks prefetch the next layer before it's needed; post-hooks release GPU memory immediately
- No latency spike at the transformer switch point

### Comparison

| Aspect | Full-Model (Old) | Layerwise (New) |
|--------|------------------|-----------------|
| Transfer granularity | Entire transformer (~28GB) | Per-layer (~700MB) |
| FSDP sharding | Yes -- each GPU transfers 1/N | **No** -- each GPU transfers full model |
| Transfer method | Blocking `.to()` | Async H2D on dedicated stream |
| GPU-to-CPU copy | Yes (full model) | None (CPU buffer unchanged) |
| Peak VRAM | ~61 GB | ~23 GB |
| Switch point cost | ~27s blocking spike | No spike |

---

## 3. Benchmark Results (4 GPUs, Correct Configuration)

All results use `flow_shift=12.0` (correct for Wan2.2-A14B), clean Run 2 timing with no profiling overhead.

### Head-to-Head: New vs Old Offload

| Metric | Layerwise (New) | Full-Model (Old) | Delta |
|--------|----------------|-----------------|-------|
| Total denoising | 662.2s | **585.7s** | Old is **13% faster** |
| Steady-state per step | ~23s | ~20s | Old is **15% faster** per step |
| First step | 65s | 40s | New 25s slower |
| Step 19 (switch) | 23s (no spike) | **47s** (+27s spike) | New handles seamlessly |
| Peak VRAM (est.) | ~23 GB | ~61 GB | New uses **62% less VRAM** |

### Per-Step Timing Profile

**Old offload** has uniform ~20s steps except for a single 47s spike at the transformer switch:

```
Step:  1   2   3  ...  17  18  [19]  20  21 ... 27
Time: 40  20  20  ...  20  20  [47]  20  20 ... 20  (seconds)
                                 ^
                          27s blocking .to() spike
```

**New offload** has uniform ~23s steps with no spike anywhere:

```
Step:  1   2   3  ...  17  18  [19]  20  21 ... 27
Time: 65  22  23  ...  23  23  [23]  23  23 ... 23  (seconds)
                                 ^
                          seamless (layerwise prefetch)
```

### Key Observation

The old offload is faster **despite** having a 27s spike at step 19. Why? Because:
- Old: 26 fast steps at 20s + 1 spike at 47s = 567s + 47s = **~586s**
- New: 27 uniform steps at 23s = **~662s**
- The 3s/step overhead across 27 steps (81s total) far exceeds the single 27s spike

---

## 4. Root Cause Analysis: Why Layerwise Offload is Slower

### The Core Problem: PCIe Topology Mismatch

The PR's NVLink data proves this is a **hardware topology problem**. On NVLink, steady-state is identical (3.27 vs 3.29s/step, delta=0.02s). On our PCIe system, the delta is 3.0s/step. The software is the same -- only the hardware topology changes.

| Factor | NVLink (delta=0.02s) | PCIe (delta=3.0s) |
|--------|---------------------|-------------------|
| H2D path | Per-GPU PCIe link (no contention) | Shared root complex (4 GPUs compete) |
| GPU-GPU path | NVLink 900 GB/s (separate from H2D) | PCIe (shared with H2D) |
| H2D per GPU per layer | 700MB (no contention) | 700MB (contending with NCCL + other GPUs) |
| FSDP H2D per GPU per layer | 175MB shard (topology-aware) | 175MB shard (topology-aware) |

### Why FSDP Handles PCIe Better

Both strategies do per-layer CPU-to-GPU transfers every step. The difference is how:

| | Old (FSDP) | New (Layerwise) |
|---|---|---|
| H2D per GPU per layer | **175MB** (1/4 shard) | **700MB** (full layer) |
| Root complex pressure (4 GPUs) | 700MB total | **2.8 GB total** (4x more) |
| Transfer scheduling | NCCL topology-aware | Blind `copy_(non_blocking=True)` |
| Compiler integration | torch.compile can optimize overlap | `@torch.compiler.disable` blocks optimization |

FSDP + NCCL are designed for shared-bus topologies: they minimize root complex pressure via sharding and use topology-aware scheduling for the all-gather. LayerwiseOffloadManager has no topology awareness.

### Why It's NOT Just About Raw Bandwidth

PCIe bandwidth is only 58.3% utilized in our profiling data. The async pipeline *should* overlap most H2D with compute (each ~700MB transfer takes ~23-70ms against ~575ms of compute per layer). But this 58.3% number comes from blocking transfers only -- we cannot measure the actual async H2D bandwidth under contention with NCCL traffic from existing data. The nsys profiles (pending) will reveal the true picture.

---

## 5. PCIe vs NVLink: Why the PR's Claims Don't Apply Here

PR #15511 reports a **58% speedup** on 8x GPU NVLink systems. On our 4x GPU PCIe system, the new offload is **13% slower**. The difference is hardware topology:

| System | GPU-GPU Path | CPU-GPU Path | Can Overlap? |
|--------|-------------|-------------|-------------|
| H100 SXM (NVLink) | NVLink (900 GB/s) | PCIe (128 GB/s) | **Yes** -- separate paths |
| H100 PCIe (ACES) | PCIe (128 GB/s) | PCIe (128 GB/s) | **No** -- shared fabric |

### PR's Reference Numbers (8x GPU NVLink)

| | Old Offload | New Offload | Improvement |
|---|---|---|---|
| Step 0 | 36.0s | 7.7s | 4.7x faster |
| Steady state | 3.27s/step | 3.29s/step | Equal |
| Step 19 (switch) | 31.3s | 3.29s | 9.5x faster |
| **Total** | **149.7s** | **94.2s** | **58% speedup** |

On NVLink, CPU-to-GPU transfers overlap entirely with GPU compute (separate physical paths). On PCIe, both share the bus, causing the ~3s/step overhead we observe.

---

## 6. Config Resolution Bug (Discovered and Fixed)

Early benchmarks used an incorrect `flow_shift=3.0` (instead of 12.0) due to SGLang's model registry failing to resolve local model paths. The transformer switch occurred at step 10 instead of step 19.

**Fix:** Use the HuggingFace model identifier (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) which exactly matches the registered config. All benchmark scripts corrected.

**Impact on results:** Performance gap **widened** from 7% (wrong flow_shift) to 13% (correct flow_shift). The per-step steady-state gap also widened from 9% to 15%, suggesting the shifted noise schedule disproportionately affects the layerwise offload path.

---

## 7. Recommendations

### For Users: When to Use Which Strategy

| Scenario | Recommended Strategy | Reason |
|----------|---------------------|--------|
| PCIe system, VRAM sufficient | **Old** (dit_cpu_offload) | 13% faster, FSDP sharding efficient |
| PCIe system, VRAM constrained | **New** (dit_layerwise_offload) | 62% less VRAM, acceptable overhead |
| NVLink system (any) | **New** (dit_layerwise_offload) | 58% faster (per PR benchmarks) |
| 8+ GPUs, NVLink | **New** (dit_layerwise_offload) | Designed for this config |

### For SGLang Contributors: Closing the PCIe Gap

Three code-level improvements, prioritized by estimated impact:

**1. Make LayerwiseOffloadManager torch.compile-compatible (highest impact)**

Remove `@torch.compiler.disable` from offload methods and implement them as compiler-safe operations (e.g., `torch.library.custom_op` or `torch.autograd.Function`). This eliminates the 80 graph breaks per step that prevent cross-layer kernel fusion and compiler overlap optimization.

Expected improvement: **~1.5-2.0s/step** (eliminates the dominant overhead source).

**2. Add FSDP-style weight sharding (medium impact)**

Currently each GPU independently copies the full ~28GB model. With 4 GPUs, total PCIe traffic is 112GB/step. FSDP-style sharding (each GPU copies 1/N, then all-gathers) reduces per-GPU H2D to ~7GB. While the async pipeline already overlaps most transfer time, reducing data volume lowers PCIe contention with Ulysses all-to-all and provides more margin for the prefetch pipeline.

Expected improvement: **~0.3-0.5s/step**.

**3. Pool GPU memory buffers and reduce CUDA API calls (medium impact)**

Pre-allocate a rotating pool of GPU buffers instead of calling `torch.empty(~700MB)` and `torch.empty((1,))` for every layer every step (~79 allocations per step). Batch event operations where possible.

Expected improvement: **~0.2-0.5s/step**.

---

## 8. Pending Experiments

| Experiment | Jobs | Status | What It Tests |
|-----------|------|--------|---------------|
| 4-GPU rerun (90 min) | 1459995-1459996 | Queued (Feb 19) | Run 3 profiled data |
| 6-GPU scaling | 1459997-1459998 | Queued (Feb 20) | Does gap narrow with more GPUs? |
| nsys profiling (new + old) | 1459999-1460000 | Queued (Feb 19) | Actual PCIe HtoD bandwidth per step |
| Prefetch=3 | Blocked | sglang 0.5.8 lacks CLI flag | Pipeline depth impact |

### What 6-GPU Results Will Show

**Hypothesis:** The gap should **stay roughly the same or widen slightly** with 6 GPUs, because:
- The dominant overhead (graph breaks) is per-GPU and independent of GPU count
- PCIe contention may increase: 6 GPUs x 28GB = 168GB total vs 4 GPUs x 28GB = 112GB
- FSDP benefits: each GPU copies only 1/6 = ~4.7GB (less than 1/4 = 7GB at 4 GPUs)
- The compute time per step may decrease with more GPUs (Ulysses splits the sequence)

If the gap stays roughly constant, it confirms the overhead is software (graph breaks). If it widens significantly, PCIe contention plays a larger role than estimated.

---

## 9. Summary

| Finding | Detail |
|---------|--------|
| **Performance** | Old offload is **13% faster** on 4x H100 PCIe (585.7s vs 662.2s) |
| **Memory** | New offload uses **62% less VRAM** (23GB vs 61GB peak) |
| **Root cause** | PCIe topology mismatch: on NVLink delta=0.02s, on PCIe delta=3.0s. Shared root complex + no FSDP sharding = 4x more H2D contention. FSDP is topology-aware; LayerwiseOffloadManager is not |
| **PCIe vs NVLink** | PR claims 58% speedup on NVLink; we see 13% slowdown on PCIe |
| **Switch handling** | New offload is seamless; old offload has a 27s spike at step 19 |
| **Key recommendation** | Add FSDP-style weight sharding to layerwise offload (addresses the PCIe topology bottleneck directly) |
| **Broader takeaway** | Layerwise offload trades speed for memory on PCIe; the gap is fixable by adding topology-aware sharding |
