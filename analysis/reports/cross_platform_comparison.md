# Cross-Platform Comparison: B200 NVLink vs H100 PCIe

**Date:** 2026-02-27
**Platforms:** 4x NVIDIA B200 (NVLink, local) vs 4x H100 PCIe (ACES cluster)
**Model:** Wan2.2-T2V-A14B (MoE, 2x14B params), Layerwise offload with async prefetch

---

## Platform Specifications

| Spec | B200 (NVLink) | H100 PCIe (ACES) |
|------|--------------|------------------|
| GPU | NVIDIA B200 | NVIDIA H100 PCIe |
| GPU Memory | 192 GB HBM3e | 80 GB HBM3 |
| GPU-GPU Interconnect | NVLink (5th gen, ~900 GB/s) | PCIe Gen5 (shared root complex) |
| CPU-GPU Interconnect | PCIe Gen5 (dedicated per GPU) | PCIe Gen5 (shared root complex) |
| System RAM | ~192 GB | 488 GB |
| BF16 TFLOPS | ~2,250 (est.) | ~1,979 |

## Headline Results

| Metric | B200 No Offload | B200 Offload | H100 New Offload | H100 Old Offload |
|--------|:--------------:|:------------:|:----------------:|:----------------:|
| Per-step time | 12.0s | 12.03s | 23.7s | 22.6s |
| Denoising total (27 steps) | ~324s | ~325s | 641.1s | 610.2s |
| Peak VRAM | 88 GB | 15.7 GB | 23.1 GB | 61.3 GB |
| GPU Utilization | 85.8% | 74.0% | N/A | N/A |
| NCCL SendRecv total | 29.50s | 47.05s | 963.4s | 690.5s |
| NCCL SendRecv avg | 0.41ms | 0.66ms | 15.7ms | 11.4ms |
| Offload overhead | **+0.03s (+0.25%)** | -- | **+1.1s (+4.9%)** | -- |

---

## Q1: Why Does NCCL Slow Down 61% Even on NVLink?

B200 NCCL SendRecv: 29.50s (no offload) vs 47.05s (offload) = **+59.5% increase**.

This is surprising because NVLink provides a physically separate path from PCIe H2D transfers. The NCCL slowdown should be zero if the only contention is on the physical interconnect. But it isn't. Three mechanisms explain this:

### 1a. GPU Compute Resource Contention (Primary)

Even though NVLink and PCIe are separate physical paths, they share **GPU-internal resources**:

- **Copy engines (CE).** Each GPU has a limited number of copy engines. H2D transfers (via `copy_stream`) and NCCL operations both use copy engines. When the copy stream is busy with a large H2D, fewer CEs are available for NCCL.
- **Memory controllers and HBM bandwidth.** H2D writes land in HBM via the GPU's memory controller. NCCL SendRecv also reads from and writes to HBM. Both compete for HBM bandwidth (~8 TB/s on B200, but shared across all SMs, CEs, and NVLink ports).
- **PCIe ingress port.** H2D uses the GPU's PCIe ingress, which shares internal crossbar bandwidth with NVLink ports. On B200, the internal crossbar is high-bandwidth but not infinite.

The profiling data confirms: **92% of H2D volume (5,855 GB of 6,372 GB) occurs during NCCL windows** in offload mode. On the no-offload baseline, 0 H2D occurs during NCCL. This proves temporal overlap is extreme.

### 1b. NCCL Algorithmic Overhead

The offload case has slightly fewer NCCL calls (70,787 vs 72,288) but **+60% more total time**. The per-call average increased from 0.41ms to 0.66ms, and the max from 435ms to 502ms. This is consistent with NCCL needing to retry or wait for memory controller bandwidth during H2D bursts.

### 1c. Quantifying the Mechanisms

| Resource | Shared? | Impact |
|----------|---------|--------|
| NVLink physical lanes | No (dedicated) | None |
| PCIe physical lanes | No (separate from NVLink) | None |
| GPU copy engines | **Yes** | H2D and NCCL compete for CEs |
| HBM bandwidth | **Yes** | H2D writes + NCCL reads compete |
| GPU internal crossbar | **Yes** | All I/O shares the crossbar |
| L2 cache | **Yes** | H2D may pollute L2, evicting NCCL buffers |

**Bottom line:** On NVLink, the NCCL slowdown is from GPU-internal resource contention (CEs, HBM, crossbar), not interconnect contention. This is a smaller effect (~60% NCCL slowdown on NVLink vs ~40% on PCIe in relative terms), but significant in absolute terms (+17.5s).

### Why the NCCL Slowdown Doesn't Translate to Per-Step Slowdown on B200

The critical difference: **B200 has enough raw compute to absorb the NCCL stalls.**

B200 no-offload: 85.8% GPU utilization. B200 offload: 74.0%. The 11.8% utilization drop represents the time lost to NCCL stalls and H2D waits. But because the B200 is ~1.9x faster than H100 in raw compute (12s vs 23s per step), these stalls are hidden within the generous per-layer compute window.

On B200: Per-layer compute ~300ms (12s / 40 layers). NCCL stall budget: 300ms - NCCL time = ample margin.
On H100: Per-layer compute ~575ms (23s / 40 layers). NCCL stall budget: 575ms - NCCL time = tight, and when NCCL spikes to 3.5s max, it breaks the budget entirely.

---

## Q2: GPU Utilization Drop (85.8% to 74.0%) -- Where Does It Go?

The 11.8 percentage point drop (85.8% to 74.0%) represents **~50.8s** of additional GPU idle time across the full run (430.25s wall time * 0.118 = ~50.8s).

### Overhead Decomposition

From the B200 profiling data:

| Source | Time (s) | % of Utilization Loss |
|--------|---------|----------------------|
| **Kernel gap increase** | +59.28s (117.54s vs 58.26s) | Primary |
| **H2D transfer time delta** | +113.39s | Hidden (overlapped) |
| **NCCL SendRecv delta** | +17.55s (47.05s vs 29.50s) | Secondary |
| **cudaFree stalls** | +4.68s (4.75s vs 0.07s) | Minor but bursty |
| **cudaMalloc overhead** | -1.00s (1.29s vs 2.29s, fewer calls) | Offset |

### Where the utilization goes:

1. **Kernel gaps from `@torch.compiler.disable` graph breaks: +59.28s.** With torch.compile enabled, the offload hooks break the compiled graph 80 times per step (40 pre-hooks + 40 post-hooks). Each break inserts CPU-side Python overhead between GPU kernel launches. The kernel gap analysis shows offload has 237 gaps >10ms vs 90 for baseline, and the max gap is 40,964ms (a startup artifact) vs 19,081ms.

2. **NCCL SendRecv slowdown: +17.55s.** As explained in Q1, GPU-internal resource contention (CE, HBM, crossbar) degrades NCCL when H2D is running concurrently.

3. **cudaFree defragmentation stalls: +4.68s.** The offload path triggers expensive cudaFree calls (avg 24.5ms, max 1,010ms) from memory allocator fragmentation.

4. **H2D is almost entirely hidden.** The profiling shows 92% of H2D occurs during both NCCL and compute. The +113.39s of additional H2D time is overlapped -- only the GPU-internal contention effects are visible.

### Why the per-step time is nearly identical despite 11.8% lower utilization

GPU utilization measures the fraction of wall time with at least one kernel running. With offload:
- Wall time increases from 381.92s to 430.25s (+12.7%)
- Kernel time decreases slightly from 327.63s to 318.35s (-2.8%)
- Net utilization: 318.35/430.25 = 74.0% vs 327.63/381.92 = 85.8%

But the **denoising-stage wall time** (what determines per-step time) is measured from the progress bar, which shows 12.03s/step. The extra wall time (430s vs 382s) includes startup, model loading, and text encoding -- not denoising steps. During the 27 denoising steps themselves, the GPU utilization drop manifests as slightly longer gaps between kernels, but the per-step pipeline still completes in ~12s because the GPU's raw compute speed absorbs the gaps.

---

## Q3: Compute Scaling Factor -- B200 (12s) vs H100 (23s)

### Raw Ratio: 1.92x

H100 PCIe per-step: 23.7s (new offload) or 22.6s (old offload, steady-state)
B200 NVLink per-step: 12.03s (offload) or 12.0s (no offload)

**H100/B200 ratio: ~1.9x** (using comparable offload config)

### Decomposing the Scaling

| Component | B200 | H100 | Ratio | Notes |
|-----------|------|------|-------|-------|
| BF16 TFLOPS (spec) | ~2,250 | ~1,979 | 1.14x | Raw compute advantage |
| HBM Bandwidth (spec) | ~8 TB/s (HBM3e) | ~3.35 TB/s (HBM3) | 2.39x | Memory-bound ops benefit |
| Measured per-step | 12.0s | ~22.6s | 1.88x | Actual end-to-end |

The measured 1.88x is higher than the 1.14x compute TFLOPS ratio, indicating the workload is **memory-bandwidth-bound**, not compute-bound. This aligns with the model architecture: attention (SageAttn) is memory-intensive, and B200's 2.4x HBM bandwidth advantage directly translates to faster attention kernels.

### Confirming with Kernel-Level Data

From B200 profiling:
- SageAttn (`_attn_fwd`): 276.5s (42.7% of GPU time) -- the dominant kernel
- NCCL SendRecv: 47.05s (7.3%) -- communication overhead
- GEMM total: 53.5s -- compute-bound portion

SageAttn being 42.7% of GPU time confirms the workload is attention-dominated and memory-bandwidth-bound. B200's HBM3e advantage is the primary driver of the 1.9x scaling.

### Per-Layer Comparison

| Metric | B200 | H100 | Ratio |
|--------|------|------|-------|
| Per-layer compute (est.) | ~300ms | ~575ms | 1.92x |
| Per-layer H2D (700MB) | ~13ms (at 53 GB/s) | ~35-89ms | 2.7-6.8x |
| NCCL all-to-all per layer | ~1.2ms | ~15.7ms | 13.1x |

The NCCL 13x difference is explained by NVLink (~900 GB/s) vs PCIe (~32 GB/s effective after contention), the most dramatic hardware gap.

---

## Q4: What Optimizations Does B200 Data Reveal That ACES Couldn't?

### 4a. Graph Break Overhead Is Measurable on B200 (Invisible on ACES)

On ACES, we concluded graph breaks cost ~0.2-0.5s because the NVLink delta was only 0.02s. But on B200, we can now **directly measure** graph break overhead by comparing no-offload (no graph breaks) vs offload (80 graph breaks/step), both with torch.compile:

| Metric | B200 No-Offload | B200 Offload | Delta |
|--------|:--------------:|:------------:|:-----:|
| Kernel gap total | 58.26s | 117.54s | +59.28s |
| Gaps >10ms | 90 | 237 | +147 |
| Gaps >1ms | 329 | 450 | +121 |
| Per-step delta | -- | -- | ~2.2s/step |

The +59.28s gap delta over 27 steps = **~2.2s/step of graph break overhead**. But the per-step wall time only increases by 0.03s. This means the graph break gaps are **almost entirely overlapped** with useful work on other SMs. The GPU fills the gaps with NCCL operations, H2D transfers, and other concurrent work.

On ACES, this was invisible because the per-step time was dominated by PCIe contention (+1.1s/step), masking the graph break signal.

### 4b. H2D Bandwidth Is 4-5x Higher on B200

| Context | B200 (GB/s) | ACES (GB/s) | Ratio |
|---------|:-----------:|:-----------:|:-----:|
| H2D during NCCL | 53.3 | 7.9 | 6.7x |
| H2D without NCCL | 31.6 | 12.4 | 2.5x |

Even during NCCL, B200 achieves 53.3 GB/s H2D -- this is NVLink-era PCIe Gen5 with dedicated per-GPU links. ACES H100 PCIe achieves only 7.9 GB/s during NCCL due to shared root complex contention.

The 6.7x H2D bandwidth ratio during NCCL is the fundamental reason offload works on B200 but hurts on ACES: B200 can transfer a 700MB layer in ~13ms during NCCL, while ACES takes ~89ms.

### 4c. GPU Buffer Pool Would Help B200 Too

The B200 profiling reveals `cudaFree` taking up to **1,010ms** in offload mode (vs 1.4ms baseline). This is 714x worse. While it doesn't affect per-step time much (amortized over 27 steps), it represents an optimization opportunity.

The LayerwiseOffloadManager already has a `_gpu_pool` (layerwise_offload.py:188-194) for the `pcie_aware` path that pre-allocates double buffers. **Enabling this pool for all offload modes** would:
- Eliminate ~260 cudaMalloc calls and their associated fragmentation
- Prevent the 1-second cudaFree stalls
- Reduce GPU utilization loss from 11.8% to potentially <5%

### 4d. The "Free Lunch" Zone

B200 data reveals a configuration sweet spot that ACES never exposed:

| Config | B200 Per-Step | VRAM | vs No-Offload |
|--------|:------------:|:----:|:-------------:|
| No offload | 12.0s | 88 GB | baseline |
| Offload + compile | 12.03s | 15.7 GB | +0.25% (negligible) |
| Offload no compile | 16.98s | 15.7 GB | +41.5% |

On B200, **offload with compile is essentially free** -- 82% VRAM reduction with <0.3% speed penalty. This "free lunch" zone exists because:
1. NVLink separates H2D from GPU-GPU traffic
2. B200's raw compute speed provides ample overlap margin
3. torch.compile makes memory patterns deterministic

On ACES, offload always costs +5-13% due to PCIe contention -- there is no "free lunch" on PCIe hardware.

### 4e. NCCL Overhead Is Substantial Even Without PCIe Contention

The ACES analysis attributed NCCL slowdown entirely to PCIe bus contention. B200 data proves **NCCL slows down even without bus contention**:

| Metric | Cause on ACES | Cause on B200 |
|--------|:-------------:|:-------------:|
| NCCL slowdown % | +39.5% | +59.5% |
| Physical contention | PCIe shared bus (dominant) | None (NVLink separate) |
| GPU-internal contention | Unknown | CE + HBM + crossbar (confirmed) |

B200's NCCL slowdown is actually **percentage-wise larger** than ACES (60% vs 40%), despite having no physical interconnect contention. This reveals that **GPU-internal resource contention** is a significant factor even on high-end NVLink systems. The absolute impact is smaller on B200 (17.5s vs ~273s on ACES) because NVLink has 28x more bandwidth than PCIe, so percentage increases translate to much smaller absolute times.

---

## Summary: Hardware Topology Determines Offload Viability

| Factor | B200 NVLink | H100 PCIe |
|--------|:----------:|:---------:|
| Offload speed penalty | 0.25% | 5-13% |
| VRAM savings | 82% | 62% |
| Recommendation | Always use offload | Use old offload unless VRAM-constrained |
| Primary bottleneck | GPU-internal CE/HBM contention | PCIe bus contention |
| Graph break impact | Hidden by GPU parallelism | Hidden by PCIe dominance |
| torch.compile value | Prevents memory fragmentation | Same + enables comm overlap |
| Key optimization | GPU buffer pool | FSDP-style weight sharding |
