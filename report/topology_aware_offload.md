# Hardware Topology Determines CPU Offload Viability in Multi-GPU Video Generation

**Authors:** Danny Liu
**Date:** 2026-02-28
**Status:** Final

---

## 1. Introduction

Layerwise CPU offload is a memory optimization technique that trades GPU VRAM for CPU-GPU transfer overhead: model weights are stored in CPU pinned memory and streamed to the GPU one layer at a time via asynchronous H2D (host-to-device) prefetch. The technique is critical for deploying large video generation models (10B+ parameters) on GPUs with limited VRAM.

The conventional assumption is that async prefetch on a dedicated CUDA copy stream fully hides the H2D transfer latency behind GPU compute. Our study shows this assumption holds on NVLink-connected systems but fails on PCIe-connected systems, where offload incurs a 5-13% speed penalty.

**Central finding:** The viability of layerwise offload depends on the GPU interconnect topology, not just the software implementation. On identical software stacks, offload is effectively free on NVLink (+0.25% overhead) but costly on PCIe (+5-13% overhead). The root cause is that PCIe systems share a single physical bus for both CPU-GPU data transfers and GPU-GPU collective communication, creating bandwidth contention that NVLink systems avoid entirely.

---

## 2. Experimental Setup

### 2.1 Model

Wan2.2-T2V-A14B-Diffusers: a Mixture-of-Experts video diffusion model with 2x14B-parameter transformers, each containing 40 transformer blocks. The model uses flow matching with 27 denoising steps to generate 81 frames at 720x1280 resolution.

### 2.2 Platforms

| Specification | Platform A: H100 PCIe | Platform B: B200 NVLink |
|---|---|---|
| GPU | 4x NVIDIA H100 PCIe (80 GB HBM3) | 4x NVIDIA B200 (192 GB HBM3e) |
| GPU-GPU interconnect | PCIe Gen5 (shared root complex) | NVLink 5th gen (~900 GB/s) |
| CPU-GPU interconnect | PCIe Gen5 (shared root complex) | PCIe Gen5 (dedicated per GPU) |
| System RAM | 488 GB | ~192 GB |
| BF16 TFLOPS | ~1,979 | ~2,250 |
| HBM bandwidth | ~3.35 TB/s | ~8 TB/s |
| Cluster | ACES (Texas A&M HPRC) | Local workstation |

The critical architectural difference: on H100 PCIe, CPU-GPU transfers (H2D) and GPU-GPU communication (NCCL all-to-all) share the same physical PCIe root complex. On B200, GPU-GPU traffic uses NVLink while CPU-GPU traffic uses dedicated PCIe links -- physically separate paths.

### 2.3 Software Stack

- SGLang 0.5.9 with PR #15511 (layerwise offload by BBuf)
- Ulysses sequence parallelism (all-to-all collective communication)
- SageAttention (FlashAttention variant)
- torch.compile with `max-autotune-no-cudagraphs`, `reorder_for_compute_comm_overlap=True`
- Offload config: `prefetch_size=1` (one layer ahead), `copy_stream` for async H2D

### 2.4 Offload Strategies Compared

- **New offload (layerwise):** Each GPU independently prefetches full layer weights (~700 MB per layer at BF16) from CPU pinned memory via async H2D on a dedicated copy stream. Post-hook releases GPU memory. 80 graph breaks per step from `@torch.compiler.disable` hooks.
- **Old offload (FSDP-based):** Each GPU loads its 1/N weight shard from CPU, then NCCL all-gathers the full weights. FSDP2's `fully_shard()` is compiler-compatible (zero graph breaks). Blocking `.to()` transfers serialize H2D before compute.
- **No offload (baseline):** All weights resident on GPU. Available only on B200 (192 GB VRAM sufficient).

### 2.5 Profiling Methodology

Three complementary profiling approaches:

1. **nsys CUPTI traces** (full-run, kernel-level): Captures every H2D memcpy, NCCL kernel, compute kernel, and CUDA API call across the entire run. Analyzed via SQLite queries on the nsys database for temporal overlap detection, bandwidth measurement, and memory API profiling.

2. **torch.profiler traces** (5-step detailed windows): High-resolution GPU timeline decomposition via TraceLens, capturing kernel fusion patterns, communication overlap, and per-operation timing.

3. **End-to-end benchmarks** (3 runs, 27 steps each): Wall-clock timing with progress bar timestamps for per-step measurement.

### 2.6 Experiment Matrix

| Experiment | Platform | Offload | torch.compile | Runs |
|---|---|---|---|---|
| ACES new offload | H100 PCIe | Layerwise | Yes | 3 |
| ACES old offload | H100 PCIe | FSDP | Yes | 3 |
| B200 no offload | B200 NVLink | None | Yes | 3 |
| B200 offload | B200 NVLink | Layerwise | Yes | 3 |
| B200 no compile | B200 NVLink | Layerwise | No | 1 |
| ACES nsys (new) | H100 PCIe | Layerwise | Yes | 1 |
| ACES nsys (old) | H100 PCIe | FSDP | Yes | 1 |
| B200 nsys (baseline) | B200 NVLink | None | Yes | 1 |
| B200 nsys (offload) | B200 NVLink | Layerwise | Yes | 1 |

---

## 3. Results

### 3.1 Headline Performance

| Metric | B200 No Offload | B200 Offload | H100 New Offload | H100 Old Offload |
|---|:---:|:---:|:---:|:---:|
| Per-step time | 12.0s | 12.03s | 23.7s | 22.6s |
| Denoising total (27 steps) | ~324s | ~325s | 641s | 610s |
| Peak VRAM | 88.2 GB | 15.7 GB | 23.1 GB | 61.3 GB |
| VRAM reduction | -- | **82%** | **62%** | -- |
| **Offload overhead** | -- | **+0.25%** | **+4.9%** | -- |

On B200 NVLink, layerwise offload achieves 82% VRAM reduction with negligible speed penalty. On H100 PCIe, the same software incurs a 4.9% penalty, widening to 13% under different noise schedules (flow_shift=12.0).

### 3.2 Profiling Metrics

| Metric | B200 Baseline | B200 Offload | H100 New | H100 Old |
|---|:---:|:---:|:---:|:---:|
| H2D total volume | 276 GB | 6,372 GB | 5,767 GB | 618 GB |
| H2D during NCCL | 0 GB | 5,855 GB (92%) | 5,253 GB (91%) | 0 GB |
| H2D BW during NCCL | N/A | 53.6 GB/s | 7.9 GB/s | N/A |
| H2D BW without NCCL | 21.7 GB/s | 31.6 GB/s | 12.4 GB/s | 11.1 GB/s |
| NCCL SendRecv total | 29.5s | 47.0s | 963.4s | 690.5s |
| NCCL avg latency | 0.41ms | 0.66ms | 15.7ms | 11.4ms |
| NCCL p99 latency | -- | 1.97ms | -- | 0.68ms |
| GPU utilization | 85.8% | 74.0% | N/A | N/A |

### 3.3 The Topology Test

The same offload code, the same model, the same configuration -- only the hardware changes:

| Comparison | NVLink (B200) | PCIe (H100) | Ratio |
|---|:---:|:---:|:---:|
| Offload speed penalty | +0.25% | +4.9 to +13% | 20-52x worse on PCIe |
| H2D BW degradation during NCCL | **None** (+19%) | **-36.5%** | Opposite direction |
| NCCL total time increase | +59.5% | +39.5% | Similar % but vastly different absolute impact |
| NCCL absolute overhead | +17.5s | +273s | 15.6x worse on PCIe |

---

## 4. Analysis: Why Topology Determines Offload Cost

### 4.1 PCIe Root Complex Contention (H100)

On H100 PCIe, all 4 GPUs connect to the CPU through a shared PCIe root complex. Both H2D prefetch transfers and NCCL GPU-GPU communication (Ulysses all-to-all) traverse this same physical bus.

The nsys temporal overlap analysis provides definitive evidence:

- **61% of H2D transfers overlap temporally with NCCL kernels.** These 7,475 transfers (5,253 GB) compete with NCCL for PCIe bandwidth.
- **H2D bandwidth degrades 36.5% during NCCL:** 7,907 MB/s (during NCCL) vs 12,447 MB/s (without NCCL).
- **NCCL latency degrades correspondingly:** average +37.7% (15.7ms vs 11.4ms), maximum +143% (4,145ms vs 1,701ms).
- **The old FSDP offload has zero H2D during NCCL.** Its blocking `.to()` accidentally serializes all transfers, preventing contention.

The new offload fires 40 async H2D copies per step (700 MB each = 28 GB/step per GPU), which share the PCIe bus with Ulysses all-to-all traffic. The FSDP approach transfers only 175 MB per layer per GPU (1/4 shard), reducing root complex pressure by 4x, with the remainder handled via NCCL's topology-aware all-gather.

The TraceLens GPU timeline confirms the impact:

| Timeline Component | Old Offload | New Offload | Delta |
|---|:---:|:---:|:---:|
| Total GPU time | 639.4s | 705.6s | +10.4% |
| Computation | 369.7s | 368.9s | -0.2% (identical) |
| Exposed communication (NCCL) | 207.0s | 273.2s | **+32.0%** |
| GPU idle | 23.9s | 59.8s | **+150%** |

Compute time is identical -- the overhead comes entirely from NCCL communication degradation and GPU idle stalls caused by PCIe contention.

### 4.2 NVLink Isolation (B200)

On B200 with NVLink, H2D transfers use PCIe while NCCL uses NVLink -- physically separate interconnects. The nsys data shows:

- **99.9% of H2D overlaps with NCCL** -- yet there is **no bandwidth degradation.**
- H2D BW is actually **higher** during NCCL: 53.6 GB/s (during NCCL) vs 31.6 GB/s (without). The higher measured bandwidth during NCCL reflects that large-burst prefetch transfers naturally overlap with NCCL windows, and large sequential transfers achieve better PCIe throughput than scattered smaller ones.
- **Prefetch works as intended:** 98.5% of layers have weights ready before compute begins. The 1.5% with marginal overlap (avg -1.8ms late) are within kernel launch latency.

The NCCL degradation that does occur (+59.5% total, +22.9% on device 0) comes from GPU-internal resource contention, not interconnect sharing:

| Resource | Shared between H2D and NCCL? | Impact |
|---|---|---|
| NVLink physical lanes | No | None |
| PCIe physical lanes | No (separate from NVLink) | None |
| GPU copy engines | **Yes** | H2D and NCCL compete for CEs |
| HBM bandwidth | **Yes** | H2D writes + NCCL reads compete for ~8 TB/s |
| GPU internal crossbar | **Yes** | All I/O shares the crossbar |

This GPU-internal contention adds +17.5s of NCCL time -- but B200's faster raw compute (1.9x vs H100) provides ample margin to absorb the stalls within the per-layer compute window (~300ms on B200 vs ~575ms on H100).

### 4.3 What Doesn't Matter

Several hypothesized overhead sources turned out to be non-dominant:

**cudaHostAlloc pinned memory allocation (82s on B200):** The offload manager calls `cudaHostAlloc` 1,741 times totaling 82s of host-side overhead. However, temporal analysis reveals these calls occur entirely during initialization and torch.compile warmup -- **zero calls during denoising.** The 82s is a one-time startup cost, not a per-step overhead.

**Graph breaks from `@torch.compiler.disable` (80 per step):** The NVLink delta of +0.25% with the same 80 graph breaks proves this is not a significant factor. On B200, direct measurement shows graph break gap time increases by +59.3s total, but this is almost entirely overlapped with concurrent work. The per-step wall time impact is ~0.03s.

**CUDA allocator churn (12 cudaMalloc during denoising on H100):** Only 12 `cudaMalloc` calls occur during denoising (0.3s total), too few to explain the 30s+ overhead on PCIe. The caching allocator handles per-layer alloc/dealloc efficiently.

---

## 5. Implications

### 5.1 Offload Strategy Should Be Topology-Aware

No single offload strategy is optimal across hardware:

| Topology | Recommended Strategy | Rationale |
|---|---|---|
| NVLink mesh | Full async layerwise prefetch | H2D and NCCL on separate paths; overhead is negligible |
| PCIe shared root complex | FSDP-sharded offload | Each GPU transfers 1/N from CPU, all-gather via NCCL's topology-aware scheduling; minimizes root complex pressure |
| PCIe with NVSwitch | Likely works (untested) | NVSwitch separates GPU-GPU from PCIe, similar to NVLink |

The optimal implementation would detect the interconnect topology at initialization (NVLink presence, PCIe topology via `nvidia-smi topo -m`) and select the appropriate offload path.

### 5.2 Why FSDP Handles PCIe Better

FSDP's approach is inherently topology-aware for three reasons:

1. **Weight sharding:** Each GPU transfers only 1/N of each layer from CPU (175 MB vs 700 MB with 4 GPUs), reducing per-GPU PCIe pressure by N-fold.
2. **NCCL all-gather scheduling:** NCCL is topology-aware and schedules the all-gather to minimize PCIe contention. It can use ring or tree algorithms that respect the physical topology.
3. **Compiler compatibility:** FSDP2's `fully_shard()` is implemented as a compiler-aware primitive with zero graph breaks, allowing `reorder_for_compute_comm_overlap` to optimize the communication schedule.

### 5.3 Generalization Beyond SGLang

This finding generalizes to any framework that combines CPU offload with multi-GPU collective communication:

- **DeepSpeed ZeRO-Offload** uses FSDP-style sharding and already handles this correctly for most topologies.
- **vLLM, TensorRT-LLM, and other inference frameworks** adding layerwise offload should implement topology detection.
- **Any model with per-layer collectives** (sequence parallelism, tensor parallelism, pipeline parallelism) will face the same topology dependence if combined with async CPU offload.

The key insight is that async prefetch is not free on shared-bus topologies -- it can degrade the collectives it shares the bus with.

### 5.4 Quantified Optimization Targets

For SGLang's layerwise offload on PCIe:

| Optimization | Expected Impact | Complexity |
|---|---|---|
| FSDP-style weight sharding | -4x root complex pressure, eliminates primary bottleneck | High |
| Topology detection + strategy selection | Enables per-platform optimization | Medium |
| Compiler-compatible hooks (remove graph breaks) | ~0.2-0.5s/step | Medium |
| GPU buffer pooling (eliminate per-layer alloc/dealloc) | ~0.2-0.3s/step | Low |
| Pre-allocated pinned memory pool | Eliminates 82s startup cost | Low |

---

## 6. Methodology Notes

### 6.1 nsys SQLite Temporal Overlap Analysis

The key finding (H2D/NCCL bandwidth contention) was established through temporal overlap detection on the nsys SQLite database:

1. Extracted all NCCL kernel windows (start, end timestamps) from the `CUPTI_ACTIVITY_KIND_KERNEL` table, filtering by `ncclDevKernel_SendRecv` demangled name.
2. Extracted all large H2D memcpy events (>1 MB, `copyKind=1`) from `CUPTI_ACTIVITY_KIND_MEMCPY`.
3. For each H2D transfer, checked whether its time range overlapped with any NCCL kernel window.
4. Computed effective bandwidth (bytes/duration) separately for overlapping vs non-overlapping H2D transfers.

This approach avoids the ambiguity of stream-level analysis (H2D and NCCL run on different CUDA streams and can overlap even when not contending) by measuring actual bandwidth degradation during temporal overlap.

### 6.2 cudaHostAlloc Temporal Bucketing

To determine whether `cudaHostAlloc` overhead (82s on B200) affects per-step performance, we bucketed all `cudaHostAlloc` calls by timestamp relative to the first denoising step. All 1,741 calls fell within the initialization/compile window, confirming the overhead is a one-time startup cost.

### 6.3 TraceLens Hierarchical Decomposition

TraceLens decomposes the GPU timeline into mutually exclusive categories (computation, exposed communication, exposed memcpy, idle) by analyzing kernel overlap on the GPU. This reveals how much of each category is "hidden" behind computation vs "exposed" as pipeline bubbles.

### 6.4 Cross-Platform Control

By running identical software on both platforms, we isolate hardware effects from software effects. The same `@torch.compiler.disable` hooks, the same 80 graph breaks per step, the same async prefetch logic -- only the physical interconnect changes. This is why the NVLink/PCIe comparison is the strongest evidence for the topology hypothesis.

---

## Appendix A: Detailed Profiling Data

### A.1 ACES H100 PCIe nsys Overlap (Full Run, 27 Steps)

| Context | H2D Count | Volume (GB) | Bandwidth (MB/s) |
|---|:---:|:---:|:---:|
| New offload during NCCL | 7,475 | 5,253 | 7,907 |
| New offload without NCCL | 4,709 | 514 | 12,447 |
| Old offload during NCCL | 0 | 0 | N/A |
| Old offload without NCCL | 6,314 | 618 | 11,141 |

### A.2 B200 NVLink nsys Overlap (Full Run, 27 Steps)

| Context | H2D Count | Volume (GB) | Bandwidth (GB/s) |
|---|:---:|:---:|:---:|
| Offload during NCCL | 8,331 | 5,855 (92%) | 53.3 |
| Offload without NCCL | 4,829 | 517 (8%) | 31.6 |
| Baseline during NCCL | 0 | 0 | N/A |
| Baseline without NCCL | 4,132 | 276 | 21.7 |

### A.3 ACES TraceLens GPU Timeline

| Component | Old Offload | New Offload | Delta |
|---|:---:|:---:|:---:|
| Total GPU time | 639.4s | 705.6s | +10.4% |
| Computation | 369.7s (57.8%) | 368.9s (52.3%) | -0.2% |
| Exposed communication | 207.0s (32.4%) | 273.2s (38.7%) | +32.0% |
| Exposed memcpy | 38.8s (6.1%) | 3.7s (0.5%) | -90.5% |
| Idle | 23.9s (3.7%) | 59.8s (8.5%) | +150.4% |

### A.4 B200 Per-Configuration Performance

| Configuration | Per-Step | Denoising (27 steps) | Peak VRAM | Overhead vs Baseline |
|---|:---:|:---:|:---:|:---:|
| No offload | 12.0s | 331s | 88.2 GB | -- |
| Offload + torch.compile | 12.03s | 332s | 15.7 GB | +0.25% |
| Offload, no torch.compile | 16.98s | 459s | 15.7 GB | +41.5% |

### A.5 B200 GPU Utilization by Device

| Device | Baseline Util. | Offload Util. | Delta |
|---|:---:|:---:|:---:|
| GPU 0 | 85.8% | 74.0% | -11.8 pp |
| GPU 1 | 85.8% | 74.8% | -11.0 pp |
| GPU 2 | 86.0% | 74.7% | -11.3 pp |
| GPU 3 | 85.8% | 74.7% | -11.1 pp |

### A.6 B200 NCCL Cross-Device Degradation

| Device | Baseline Avg (ms) | Offload Avg (ms) | Delta |
|---|:---:|:---:|:---:|
| 0 | 0.389 | 0.522 | +34% |
| 1 | 0.356 | 0.672 | +89% |
| 2 | 0.466 | 0.745 | +60% |
| 3 | 0.421 | 0.708 | +68% |

The asymmetric degradation across devices is consistent with GPU-internal resource contention (copy engines, memory controllers) rather than NVLink bandwidth limits.
