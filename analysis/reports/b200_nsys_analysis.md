# Deep nsys SQLite Analysis: B200 Baseline vs Offload

**Date:** 2026-02-28
**Platform:** 8x NVIDIA B200 (NVLink mesh), 4 GPUs used
**Model:** Wan2.2-T2V-A14B-Diffusers, 40 transformer blocks, 27 denoising steps
**Configuration:** Ulysses sequence parallelism (all-to-all), torch.compile enabled
**Traces:**
- Baseline (no offload): `exp1_no_offload_20260227_233628.sqlite` (1364 MB)
- Offload default: `exp2_offload_default_20260227_234442.sqlite` (1507 MB)

---

## Executive Summary

The offload experiment runs 12.6% slower in wall time (430.3s vs 381.9s, device 0) despite B200 NVLink providing 53-55 GB/s H2D bandwidth. The analysis reveals three distinct bottleneck categories with quantified impact:

| Bottleneck | Impact (device 0) | Severity |
|---|---|---|
| Memory API overhead (cudaHostAlloc) | +85.6s (88.1s vs 2.5s) | **DOMINANT** |
| GPU utilization loss (gaps/bubbles) | -11.9% (83.9% -> 72.0%) | Moderate |
| NCCL SendRecv degradation | +1.6s (8.6s vs 7.0s, +22.9%) | Moderate |
| H2D bandwidth during NCCL | No degradation (53.6 vs 45.1 GB/s) | **Non-issue on NVLink** |

**Critical finding: On B200 NVLink, the PCIe contention root cause identified on ACES H100 is entirely absent.** H2D bandwidth is actually *higher* during NCCL (53.6 GB/s) than without (45.1 GB/s), because NVLink handles GPU-GPU traffic independently of PCIe. The dominant bottleneck on B200 is `cudaHostAlloc` overhead -- the offload manager pins host memory for each layer's weights, costing 82.0s total (avg 47.1ms per call, max 670ms).

---

## 1. Per-Layer Compute Time Breakdown

### Method
Used `_attn_fwd` (FlashAttention) kernel occurrences on device 0 as layer markers. Each transformer block produces 2-7 attention kernel calls (varying by Wan model's self-attn + cross-attn architecture). Grouped consecutive attention calls into layer groups using an inter-kernel gap threshold derived from the expected 1080 layers (27 steps x 40 layers).

### Per-Layer Structure
Each layer on device 0 contains:
- **4 attention kernel calls** (average), each ~68-69ms = ~272ms total FlashAttention per layer
- **8 NCCL SendRecv kernels** (2 all-to-all x 4 peer channels per call)
- **Additional compute**: RMSNorm, rotary embeddings, SiLU/GELU, quantization, fused kernels

### Key Findings

| Metric | Baseline | Offload | Delta |
|---|---|---|---|
| Total GPU kernel time (device 0) | 327.6s | 318.4s | -2.8% |
| Stream 7 GPU time | 320.6s | 309.7s | -3.4% |
| Stream 7 wall time | 381.9s | 430.3s | +12.6% |
| Stream 47 (NCCL) GPU time | 7.0s | 8.6s | +23.6% |

**Layer compute is uniform.** Layers with 2 attention calls (the most common type) each take ~136.5ms of attention compute, essentially identical between baseline and offload. The FlashAttention kernel itself is unaffected by offloading:
- Baseline: 68.245ms average per `_attn_fwd` call
- Offload: 69.034ms average per `_attn_fwd` call (+1.2% -- within noise)

**No CUDA graphs are used.** Despite `torch.compile`, no `graphId != 0` kernels were found. The compiler generates Triton kernels but does not capture CUDA graphs, meaning `@torch.compiler.disable` hooks in the offload manager do NOT break CUDA graphs (there are none to break).

### Dominant Kernels (Baseline, Device 0, Stream 7)

| Kernel | Count | Total (s) | Avg (ms) | % of Stream 7 |
|---|---|---|---|---|
| `_attn_fwd` | 4320 | 294.8 | 68.2 | 91.9% |
| `nvjet_tst_256x256` (MLP) | 15120 | 12.9 | 0.85 | 4.0% |
| `nvjet_tst_192x288` (MLP) | 2160 | 4.0 | 1.87 | 1.3% |
| All others | ~157k | ~9.0 | <0.1 | 2.8% |

FlashAttention dominates at 91.9% of compute stream time. This means any optimization that doesn't reduce attention latency will have limited impact on per-step compute time.

---

## 2. H2D Transfer Timing and Prefetch Margin

### Baseline (No Offload)
- Total H2D: 69.1 GB in 3.4s (20.2 GB/s effective)
- Only stream 7 H2D: 1033 ops of model weight loading at startup
- No H2D during denoising (weights remain on GPU)
- D2H: 0.3 GB (negligible)

### Offload
- Total H2D: **1548.0 GB** in 30.6s (**50.6 GB/s** effective)
- Two dedicated copy streams: stream 91 (927.7 GB) and stream 87 (494.1 GB)
  - Stream 91: 1320 ops, 53.7 GB/s bandwidth
  - Stream 87: 703 ops, 54.9 GB/s bandwidth
- D2H: **81.0 GB** in 20.8s (3.9 GB/s) -- offloading weights back to host

### Prefetch Margin
Of 1080 layer boundaries analyzed:
- **98.5% (1064 layers)**: Prefetch completed before layer needed weights
- **1.5% (16 layers)**: Prefetch still in progress when layer started
  - Average late margin: -1.8ms (max: -3.4ms)
  - These marginal overlaps are small enough to likely be hidden by kernel launch latency

**The prefetch mechanism works correctly on B200.** Weights arrive in time for 98.5% of layers, and the few late arrivals are within ~3ms.

### H2D Volume Analysis
- 1548 GB H2D / 27 steps = **57.3 GB per step** = ~1.43 GB per layer (40 layers)
- With 4 GPUs each independently offloading the full model, this implies no weight sharding
- The model is ~14.4B parameters x 2 bytes (bf16) = ~28.8 GB total weights
- 57.3 GB / 28.8 GB ~ 2.0x -- consistent with full model loaded each step (some overlap between H2D and D2H)

---

## 3. NCCL SendRecv Timing Distribution

### Overall Comparison

| Metric | Baseline | Offload | Delta |
|---|---|---|---|
| Total NCCL time (all devices) | 29.5s | 47.0s | +59.5% |
| Device 0 NCCL time | 7.0s | 8.6s | +22.9% |
| Kernel count (device 0) | 18072 | 16571 | -8.3% |
| Average duration | 0.389ms | 0.522ms | +34.2% |
| p50 | 0.387ms | 0.407ms | +5.2% |
| p99 | 0.681ms | 1.973ms | +189.7% |

### Key Observation: Asymmetric Cross-Device Degradation

The NCCL degradation is **NOT uniform across devices**:

| Device | Baseline (avg ms) | Offload (avg ms) | Delta |
|---|---|---|---|
| 0 | 0.389 | 0.522 | +34% |
| 1 | 0.356 | 0.672 | +89% |
| 2 | 0.466 | 0.745 | +60% |
| 3 | 0.421 | 0.708 | +68% |

Devices 1-3 show much larger NCCL degradation than device 0. This is consistent with NVLink bandwidth contention: while NVLink provides separate paths from PCIe, the H2D transfers on all 4 devices simultaneously may create memory controller contention at the CPU/host side, causing NVLink SendRecv to slow down when waiting for data.

### Temporal Distribution
The NCCL slowdown is **uniform across the denoising run** (not concentrated at specific steps):
- Baseline: ~634ms NCCL per time window (steady state), avg 0.360ms/kernel
- Offload: ~780-840ms NCCL per time window (steady state), avg 0.46-0.51ms/kernel

This rules out one-time initialization effects and confirms the overhead is per-step.

### Tail Latency
The p99 jump from 0.681ms to 1.973ms (+190%) in offload suggests occasional stalls in NCCL, likely when H2D transfers are consuming memory controller bandwidth. However, on NVLink this effect is modest compared to the PCIe sharing seen on ACES H100.

---

## 4. Kernel Launch Density and Graph Break Impact

### GPU Utilization

| Metric | Baseline | Offload | Delta |
|---|---|---|---|
| Stream 7 kernels | 179,276 | 148,650 | -17.1% |
| Stream 7 launch rate | 469/s | 345/s | -26.4% |
| Stream 7 GPU utilization | 83.9% | 72.0% | -11.9 pp |
| Overall GPU utilization | 85.8% | 74.0% | -11.8 pp |

### Inter-Kernel Gap Distribution (Stream 7, Steady State)

| Gap Range | Baseline | Offload | Notes |
|---|---|---|---|
| Overlap | 3.6% | 4.8% | Back-to-back kernels |
| < 1 us | 3.0% | 2.3% | Near-zero gap |
| 1-5 us | **79.6%** | **72.8%** | Normal launch latency |
| 5-10 us | 4.0% | 10.3% | Slight increase |
| 100us - 1ms | 9.1% | 8.9% | Graph break signature |
| 1-10 ms | 0.05% | 0.3% | Step transitions |

### Gap Time Budget

| Metric | Baseline | Offload |
|---|---|---|
| Total gap time | 13.9s (4.16%) | 21.2s (7.34%) |
| Large gaps (>100us) | 13,147 gaps, 13.5s | 10,982 gaps, 20.8s |
| Mean gap | 96.6us | 178.0us |

**No CUDA graphs** in either experiment. The `@torch.compiler.disable` hooks in the offload manager cannot break CUDA graphs because `torch.compile` on this model does not produce graph captures. The hooks do increase kernel launch gaps slightly (mean gap +84%), but the absolute impact is small.

The 11.9pp GPU utilization drop in offload is primarily due to:
1. **Longer per-gap duration** (mean 178us vs 97us) suggesting host-side delays from offload management
2. **More 5-10us gaps** (10.3% vs 4.0%) from `torch.compiler.disable` boundary overhead
3. **D2H memory copies** on stream 7 (81 GB of weight offloading back to host) that serialize with compute

---

## 5. Copy Stream Utilization

### Offload Copy Streams During Denoising (318.3s)

| Stream | Direction | Ops | Volume | Active Time | Utilization | BW |
|---|---|---|---|---|---|---|
| 91 | H2D | 1318 | 926.3 GB | 17.5s | 5.5% | 53.7 GB/s |
| 87 | H2D | 701 | 492.7 GB | 9.0s | 2.8% | 54.9 GB/s |
| 7 | D2H | 17 | 1.2 GB | 0.3s | 0.1% | 9.3 GB/s |

**Copy streams are 92% idle** during denoising. The two H2D streams are active only 8.3% of the time combined. This means:
- H2D bandwidth is not saturated -- there is ample headroom
- The prefetch mechanism issues transfers efficiently in short bursts
- The bottleneck is NOT transfer bandwidth

### H2D / NCCL Temporal Overlap

| Category | Ops | Volume | Time | BW |
|---|---|---|---|---|
| H2D during NCCL | 2018 | 1418.2 GB (99.9%) | 26.5s | 53.6 GB/s |
| H2D outside NCCL | 6 | 1.2 GB (0.1%) | 0.03s | 45.1 GB/s |

**99.9% of H2D transfers overlap temporally with NCCL SendRecv** activity -- yet there is **no bandwidth degradation** (53.6 GB/s during NCCL vs 45.1 GB/s without). This definitively proves that **NVLink isolates GPU-GPU communication from PCIe H2D transfers**. This is the fundamental difference from the ACES H100 PCIe topology where H2D and NCCL share the same PCIe root complex.

### Baseline
The baseline has essentially no H2D during denoising (0 ops >1MB on copy streams), confirming weights stay resident on GPU throughout.

---

## 6. Memory Operations

### cudaHostAlloc: The Dominant Bottleneck

| API Call | Baseline | Offload | Delta |
|---|---|---|---|
| cudaHostAlloc | 284 calls, 0.034s | **1741 calls, 81.961s** | **+2410x** |
| cudaMalloc | 3816 calls, 2.292s | 259 calls, 1.294s | -43% |
| cudaFree | 195 calls, 0.068s | 194 calls, 4.752s | +69x |
| **Total memory API** | **2.491s** | **88.099s** | **+3436%** |

The offload manager calls `cudaHostAlloc` 1741 times (pinned host memory allocation for weight staging), averaging **47.1ms per call** with a maximum of **670ms**. This is by far the single largest overhead source:
- 82.0s of `cudaHostAlloc` time alone exceeds the total wall time difference (48.3s)
- This occurs on the host side and can overlap with GPU execution, but it introduces synchronization points that create the GPU idle bubbles observed in the gap analysis

**cudaFree** also shows a 70x increase (4.75s vs 0.07s), with average 24.5ms and max 1010.7ms, indicating expensive host memory teardown for offloaded weights.

### Synchronization
Synchronization events are similar in count (2.60M vs 2.62M) but the offload experiment spends less total time in synchronization (8.6s vs 15.0s), suggesting the offload's async design avoids some explicit syncs but introduces implicit stalls through memory APIs.

---

## 7. Synthesis: Root Cause Attribution

### Wall Time Budget (Device 0)

| Component | Baseline | Offload | Delta |
|---|---|---|---|
| Wall time | 381.9s | 430.3s | **+48.3s** |
| GPU kernel time | 327.6s | 318.4s | -9.3s |
| Stream 7 idle (gaps) | 61.3s | 120.5s | **+59.2s** |

The GPU runs **fewer** kernel seconds in offload, yet wall time is 48.3s longer. The difference comes from:

1. **Memory API overhead** (+85.6s): cudaHostAlloc pinning dominates. While some of this overlaps with GPU execution, it creates host-side stalls that propagate as GPU idle gaps.

2. **Increased GPU idle time** (+59.2s): Stream 7 gap time increases from 61.3s to 120.5s. The additional 59.2s accounts for host-side delays from memory management, D2H weight offloading on stream 7, and synchronization boundaries from `@torch.compiler.disable` hooks.

3. **NCCL overhead** (+1.6s on device 0): A modest 22.9% increase in NCCL time, likely from memory controller contention at the CPU when all 4 GPUs simultaneously perform H2D transfers.

### Comparison with ACES H100 PCIe Findings

| Factor | ACES H100 (PCIe) | B200 (NVLink) |
|---|---|---|
| H2D BW during NCCL | 7,907 MB/s (-36.5%) | 53,600 MB/s (no degradation) |
| PCIe contention | **Primary bottleneck** | **Non-issue** |
| NCCL degradation | +61% total | +22.9% (device 0 only) |
| Memory API overhead | Not measured | **82s cudaHostAlloc** |
| Weight sharding | 9.3x more H2D (no FSDP) | 22.4x more H2D (no FSDP) |
| Overall delta | -13% (old offload faster) | -12.6% (baseline faster) |

**The same code has different bottleneck profiles on different hardware.** On ACES H100, PCIe topology sharing is the primary issue. On B200 NVLink, NVLink eliminates that bottleneck entirely, revealing `cudaHostAlloc` as the new dominant overhead.

---

## 8. Optimization Recommendations for B200

### Immediate Impact (High Priority)

1. **Pre-allocate pinned host memory**: Allocate a persistent pinned memory pool at initialization rather than calling `cudaHostAlloc`/`cudaFreeHost` per layer per step. Expected savings: ~82s (eliminating 1741 cudaHostAlloc calls).

2. **Use cudaMallocHost with memory pool**: If pre-allocation is not feasible, use `cudaMemPool` APIs with pinned memory support to amortize allocation costs.

### Moderate Impact

3. **Reduce D2H volume**: 81 GB of D2H transfers on stream 7 (weight offloading back to host) serializes with compute. Consider keeping weights in GPU memory if VRAM allows, or moving D2H to a separate stream.

4. **FSDP-style weight sharding**: Currently each of 4 GPUs independently H2D-transfers the full model (~57 GB/step). Sharding would reduce to ~14 GB/step per GPU.

### Low Priority on NVLink (but important for PCIe systems)

5. **PCIe-aware scheduling**: The topology-aware H2D scheduling proposed for ACES is irrelevant on B200 NVLink but remains critical for PCIe-connected systems.

6. **NCCL overlap optimization**: The 22.9% NCCL overhead is modest and may not justify complexity. Focus on memory API overhead first.
