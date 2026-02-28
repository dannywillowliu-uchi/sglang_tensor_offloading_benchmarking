# SGLang Layerwise Offload: B200 NVLink Profiling & Optimization Report

**Date:** 2026-02-28
**Hardware:** 8x NVIDIA B200 (NVLink mesh), 4 GPUs used
**Model:** Wan2.2-T2V-A14B-Diffusers (MoE, 2x14B transformers, 40 blocks each)
**Software:** SGLang 0.5.9, PR #15511 by BBuf
**Config:** Ulysses sequence parallelism (all-to-all), torch.compile enabled

---

## 1. Executive Summary

Layerwise offload on B200 NVLink achieves **82% VRAM reduction with <0.3% speed penalty** -- essentially a free lunch. This contrasts sharply with H100 PCIe where offload costs 5-13%.

| Config | Per-Step | Denoising (27 steps) | Peak VRAM | Overhead |
|--------|:--------:|:--------------------:|:---------:|:--------:|
| No offload | 12.0s | 331s | 88.2 GB | baseline |
| Offload + compile | 12.03s | 332s | 15.7 GB | **+0.25%** |
| Offload, no compile | 16.98s | 459s | 15.7 GB | +41.5% |

The dominant overhead source differs by hardware topology:
- **B200 NVLink:** `cudaHostAlloc` pinned memory allocation (82s, 1741 calls)
- **H100 PCIe:** PCIe root complex contention (H2D degrades NCCL by 36.5%)

---

## 2. Torch Profiler Trace Analysis (5 profiled steps)

### 2.1 GPU Timeline Breakdown (TraceLens)

**Offload trace (5 steps, 87.7s wall time):**

| Component | Time (ms) | % of Total |
|-----------|----------:|----------:|
| Computation | 86,164 | 98.2% |
| Exposed communication (NCCL) | 1,491 | 1.7% |
| Exposed memcpy | 42 | 0.05% |
| **GPU Busy** | **87,696** | **99.94%** |
| GPU Idle | 50 | 0.06% |

The GPU is busy 99.94% of the time during profiled steps. Offload overhead is almost entirely hidden.

### 2.2 Kernel Time Comparison

| Category | No Offload (s) | Offload (s) | Delta |
|----------|:--------------:|:-----------:|:-----:|
| Attention (`_attn_fwd`) | 102.6 | 80.0 | -22.1% |
| NVJet MLP | 6.1 | 4.9 | -19.8% |
| **Memcpy H2D** | **0.0** | **5.75** | **+inf** |
| NCCL SendRecv | 2.0 | 1.8 | -8.4% |
| Triton kernels | 0.8 | 0.8 | -7.8% |
| GEMM/MatMul | 0.3 | 0.3 | -19.9% |

Note: The lower kernel times in offload reflect shorter profiled wall time (88s vs 112s), not actual per-step speedup.

### 2.3 Top Operations (Offload, TraceLens)

| Operation | Kernel Time | % of GPU | Count |
|-----------|:----------:|:--------:|:-----:|
| `_attn_fwd` (self-attn, 86.4K seq) | 79.6s | 86.6% | 480 |
| `aten::copy_` (H2D offload) | 5.75s | 6.3% | 468 |
| `aten::addmm` (GEMM, 21.6K x 5K) | 2.62s | 2.8% | 2880 |
| `_attn_fwd` (cross-attn, 2K seq) | 1.92s | 2.1% | 480 |
| `aten::addmm` (GEMM, 21.6K x 13.8K) | 1.14s | 1.2% | 480 |

**FlashAttention dominates at 88.7%** of GPU time (self-attn + cross-attn). Any optimization must either reduce attention cost or be perfectly hidden behind it.

### 2.4 H2D Transfer Characteristics

| Metric | Value |
|--------|------:|
| Total H2D transfers | 468 |
| Total H2D volume | ~5.75s kernel time |
| Per-transfer size | 351,394,304 bytes (335 MB) |
| Per-transfer time | 12.29ms avg |
| H2D bandwidth | **~27.3 GB/s** (per-stream) |
| Copy stream utilization | 6.6% of wall time |

Each `aten::copy_` call transfers exactly 351,394,304 bytes (335 MB) of layer weights via pinned H2D. The copy stream is 93.4% idle -- there is massive bandwidth headroom.

### 2.5 Stream Architecture

| Stream | Role | Events | Active Time | Utilization |
|--------|------|-------:|:----------:|:-----------:|
| 7 | Compute (attention, GEMM, Triton) | 25,376 | 86.2s | 98.2% |
| 47 | NCCL SendRecv (Ulysses all-to-all) | 3,840 | 1.8s | 2.1% |
| 87 | H2D copy stream (offload prefetch) | 468 | 5.75s | 6.6% |

---

## 3. nsys Deep Analysis (27 steps, full run)

### 3.1 Wall Time Budget (Device 0)

| Component | No Offload | Offload | Delta |
|-----------|:----------:|:-------:|:-----:|
| Wall time | 381.9s | 430.3s | +48.3s |
| GPU kernel time | 327.6s | 318.4s | -9.3s |
| GPU idle (gaps) | 61.3s | 120.5s | +59.2s |
| GPU utilization | 85.8% | 74.0% | -11.8pp |

### 3.2 Memory API Overhead -- THE Dominant Bottleneck

| API Call | No Offload | Offload | Ratio |
|----------|:----------:|:-------:|:-----:|
| `cudaHostAlloc` | 284 calls, 0.03s | **1741 calls, 82.0s** | **2410x** |
| `cudaFree` | 195 calls, 0.07s | 194 calls, 4.75s | 70x |
| `cudaMalloc` | 3816 calls, 2.29s | 259 calls, 1.29s | 0.56x |
| **Total memory API** | **2.49s** | **88.1s** | **35x** |

`cudaHostAlloc` averages **47.1ms per call** (max 670ms). These are host-side pinned memory allocations for staging layer weights before H2D transfer. While they partially overlap with GPU execution, they create synchronization bubbles visible as the +59.2s GPU idle increase.

### 3.3 H2D / NCCL Overlap (NVLink vs PCIe)

| Metric | B200 NVLink | H100 PCIe |
|--------|:----------:|:---------:|
| H2D during NCCL | 5,855 GB (99.9%) | 5,253 GB (61%) |
| H2D BW during NCCL | **53.6 GB/s** | 7.9 GB/s |
| H2D BW without NCCL | 45.1 GB/s | 12.4 GB/s |
| **BW degradation** | **None (+19%)** | **-36.5%** |

On B200 NVLink, H2D bandwidth is actually *higher* during NCCL because NVLink handles GPU-GPU traffic on a separate physical path from PCIe H2D. On H100 PCIe, they share the same root complex, causing 36.5% degradation.

### 3.4 NCCL Degradation

| Metric | No Offload | Offload | Delta |
|--------|:----------:|:-------:|:-----:|
| Total NCCL time (all GPUs) | 29.5s | 47.1s | +59.5% |
| Device 0 NCCL time | 7.0s | 8.6s | +22.9% |
| Avg SendRecv latency | 0.39ms | 0.52ms | +34.2% |
| p99 latency | 0.68ms | 1.97ms | +189.7% |

NCCL degrades even on NVLink due to **GPU-internal resource contention** (copy engines, HBM controllers, internal crossbar). However, the absolute impact is small (1.6s on device 0) because NVLink provides ~900 GB/s bandwidth.

---

## 4. Root Cause Attribution

### 4.1 B200 NVLink (this study)

| Root Cause | Impact | % of Overhead |
|------------|:------:|:------------:|
| **cudaHostAlloc** pinning overhead | +82s CPU-side | **Dominant** |
| GPU idle gaps from host stalls | +59.2s | Secondary (consequence of above) |
| NCCL SendRecv degradation | +1.6s | Minor |
| Graph break overhead | ~0s visible | Hidden by GPU parallelism |
| PCIe contention | 0s | **Non-issue** |

### 4.2 H100 PCIe (ACES study)

| Root Cause | Impact | % of Overhead |
|------------|:------:|:------------:|
| PCIe root complex contention | +67.3s NCCL | **Dominant** |
| GPU idle from PCIe stalls | +35.9s | Secondary |
| No FSDP weight sharding | 9.3x more H2D volume | Amplifier |
| Graph breaks | ~0.2-0.5s | Minor |

### 4.3 The Key Insight

The **same code has completely different bottleneck profiles** on different hardware. On NVLink, offload is free because H2D and NCCL use separate physical paths. On PCIe, they share the bus, and the lack of weight sharding puts 9.3x more traffic on it.

---

## 5. Optimization Recommendations

### 5.1 Pre-allocate Pinned Host Memory Pool (HIGH IMPACT -- B200)

**Problem:** 1741 `cudaHostAlloc` calls costing 82s total (47ms avg, 670ms max).

**Solution:** Allocate a persistent pinned memory pool at initialization. Reuse across steps.

```python
# In LayerwiseOffloadManager.__init__:
self._pinned_pool = {}
for i, layer in enumerate(self.layers):
    size = sum(p.numel() * p.element_size() for p in layer.parameters())
    self._pinned_pool[i] = torch.empty(size, dtype=torch.uint8, pin_memory=True)
```

**Expected savings:** ~82s (eliminates 1741 cudaHostAlloc calls)
**Applies to:** All hardware

### 5.2 Enable GPU Buffer Pool for All Offload Modes (HIGH IMPACT -- B200)

**Problem:** Per-layer `torch.empty(~700MB, device=GPU)` + `torch.empty((1,), device=GPU)` stubs cause allocator fragmentation. `cudaFree` max latency: **1,010ms**.

**Solution:** The `pcie_aware` path already has `_gpu_pool` double-buffering (layerwise_offload.py:188-194). Enable this for ALL offload modes, not just `pcie_aware`.

**Expected savings:** Eliminates cudaFree stalls (4.75s), prevents allocator fragmentation in eager mode (+41.5% overhead eliminated)

### 5.3 FSDP-Style Weight Sharding (HIGH IMPACT -- PCIe)

**Problem:** Each GPU independently H2D-transfers the full model (~57 GB/step with 4 GPUs). On PCIe, this creates 4x root complex pressure vs FSDP's 1/N sharding.

**Solution:** Each GPU loads only 1/N of each layer from CPU, then NCCL all-gathers the rest. NCCL's topology-aware scheduling minimizes PCIe contention.

**Expected savings:** 4x reduction in per-GPU H2D volume, elimination of PCIe contention
**Applies to:** PCIe systems (primary), NVLink (minor benefit from reduced H2D)

### 5.4 Make Offload Hooks torch.compile-Compatible (MEDIUM IMPACT)

**Problem:** 80 `@torch.compiler.disable` graph breaks per step prevent `reorder_for_compute_comm_overlap` from optimizing across layer boundaries.

**Solution:** Implement offload operations as `torch.library.custom_op` or compiler-safe primitives (like FSDP2 does).

**Expected savings:** ~0.2-0.5s per step. More importantly, enables compiler to optimize the entire pipeline.
**Evidence:** NVLink delta of 0.02s proves this is not dominant, but it compounds with other optimizations.

### 5.5 Move D2H to Separate Stream (LOW IMPACT -- B200)

**Problem:** 81 GB of D2H weight offloading occurs on stream 7 (compute stream), serializing with compute kernels.

**Solution:** Use a dedicated D2H stream, mirroring the H2D copy stream pattern.

**Expected savings:** Reduces GPU idle from D2H serialization.

---

## 6. Cross-Platform Comparison

| Factor | B200 NVLink | H100 PCIe |
|--------|:----------:|:---------:|
| Offload speed penalty | **+0.25%** | +5-13% |
| VRAM savings | 82% (88 -> 15.7 GB) | 62% (61 -> 23 GB) |
| Per-step compute | 12.0s | 22.6s |
| Compute scaling | 1.0x | 0.53x (1.88x slower) |
| H2D BW during NCCL | 53.6 GB/s | 7.9 GB/s |
| NCCL avg latency | 0.52ms | 15.7ms |
| Primary bottleneck | cudaHostAlloc | PCIe contention |
| Key optimization | Pinned memory pool | FSDP weight sharding |
| torch.compile required? | Yes (prevents fragmentation) | Yes (same + enables overlap) |

### The "Free Lunch" Zone

B200 NVLink exposes a sweet spot: offload + compile gives 82% VRAM savings with negligible overhead. This works because:
1. NVLink physically separates H2D from GPU-GPU traffic
2. B200's raw compute speed provides ample overlap margin (~300ms/layer vs ~13ms H2D)
3. torch.compile makes memory patterns deterministic, preventing fragmentation

This zone **does not exist on PCIe hardware** -- there, offload always costs 5-13%.

---

## 7. Methodology

### Data Sources
- **Timing:** `sglang generate` with `--dit-layerwise-offload` flag, 27 denoising steps, 3 runs
- **nsys traces:** CUPTI-level kernel, memcpy, and memory API data. Exported to SQLite (2.7 GB combined)
- **Torch profiler traces:** `--profile --num-profiled-timesteps 5`, exported as Chrome trace JSON (119 MB combined)
- **TraceLens:** Hierarchical GPU timeline decomposition from torch profiler traces

### Trace Files
| File | Size | Content |
|------|:----:|---------|
| `b200_no_offload_5steps.trace.json.gz` | 58 MB | Torch profiler, no offload, 5 steps |
| `b200_offload_5steps.trace.json.gz` | 61 MB | Torch profiler, offload, 5 steps |
| `exp1_no_offload_*.sqlite` | 1.3 GB | nsys full run, no offload |
| `exp2_offload_default_*.sqlite` | 1.4 GB | nsys full run, offload |

### Viewing Traces
Open in Perfetto UI (https://ui.perfetto.dev) or `chrome://tracing`:
- Navigate to denoising steps to see kernel timeline
- Stream 7 = compute, Stream 47 = NCCL, Stream 87 = H2D copy
- Look for gaps between `_attn_fwd` kernels to see offload boundary overhead

---

## Appendix A: Per-Layer Pipeline (One Step)

```
For each of 40 layers:

  [GRAPH BREAK] pre_hook (@torch.compiler.disable)
      |-- prefetch_layer(i+1):
      |     copy_stream.wait_stream(compute_stream)
      |     gpu_buf = torch.empty(~335MB, device=GPU)    <-- cudaMalloc
      |     gpu_buf.copy_(cpu_buf, non_blocking=True)     <-- async H2D on stream 87
      |     event.record(copy_stream)
      |-- compute_stream.wait_event(prefetch_event[i])    <-- ensure weights ready

  [GRAPH RESUMES] layer.forward()
      |-- RMSNorm -> QKV projection (GEMM) -> Attention (_attn_fwd) -> ...
      |-- Ulysses all-to-all (NCCL on stream 47)
      |-- MLP: up_proj (GEMM) -> GELU -> down_proj (GEMM)
      |-- ~300ms total compute per layer on B200

  [GRAPH BREAK] post_hook (@torch.compiler.disable)
      |-- release_layer(i):
            param.data = torch.empty((1,), device=GPU)   <-- stub (frees ~335MB)
```

Per step: 80 graph breaks, 40 H2D transfers (~335MB each = 13.4 GB), 40 GPU alloc + 39 GPU free.
