# B200 Profiling: No-Offload Baseline vs Layerwise Offload

Generated: 2026-02-27 18:36:04
Baseline: no_offload
Offload:  offload_default

## A. H2D Transfer Overhead

| Metric | Baseline | Offload | Delta |
|--------|----------|---------|-------|
| Transfer count | 49072 | 62075 | +13003.00 |
| Total volume | 276.41 GB | 6372.08 GB | +6095.67 GB |
| Total H2D time | 13.95 s | 127.34 s | +113.39 s |
| Effective bandwidth | 19.81 GB/s | 50.04 GB/s | +30.23 GB/s |
| Avg transfer size | 5.63 MB | 102.65 MB | +97.02 MB |
| Max transfer size | 2100.30 MB | 4200.60 MB | +2100.30 MB |
| Avg transfer duration | 0.28 ms | 2.05 ms | +1.77 ms |
| Max transfer duration | 102.19 ms | 143.52 ms | +41.33 ms |

### H2D Size Distribution

**Baseline:**
| Bucket | Count | Total GB | Avg BW (MB/s) |
|--------|-------|----------|---------------|
| 10-100MB | 3376 | 174.35 | 21875.44 |
| 100-500MB | 648 | 93.11 | 21554.31 |
| >500MB | 4 | 8.40 | 24760.71 |
| 1-10MB | 104 | 0.39 | 18607.73 |
| 10KB-1MB | 5668 | 0.16 | 9558.28 |
| <10KB | 39272 | 0.01 | 154.67 |

**Offload:**
| Bucket | Count | Total GB | Avg BW (MB/s) |
|--------|-------|----------|---------------|
| >500MB | 8359 | 5916.58 | 53304.37 |
| 100-500MB | 1808 | 298.57 | 38812.79 |
| 10-100MB | 2941 | 156.61 | 26839.29 |
| 1-10MB | 52 | 0.18 | 34384.16 |
| 10KB-1MB | 6281 | 0.14 | 9039.02 |
| <10KB | 42634 | 0.01 | 107.51 |

### H2D / NCCL Overlap

Key question: Does H2D contend with NCCL on NVLink? (Should be minimal.)

**Baseline:**
| Context | H2D Count | Total GB | Avg BW (MB/s) |
|---------|-----------|----------|---------------|
| during_nccl | 0 | N/A | N/A |
| no_nccl | 4132 | 276.25 | 21745.63 |

**Offload:**
| Context | H2D Count | Total GB | Avg BW (MB/s) |
|---------|-----------|----------|---------------|
| during_nccl | 8331 | 5854.93 | 53303.36 |
| no_nccl | 4829 | 517.01 | 31558.65 |

### H2D / Compute Overlap

Does H2D overlap with layer compute? (Want: most H2D during compute.)

**Baseline:**
| Context | H2D Count | Total GB |
|---------|-----------|----------|
| during_compute | 0 | N/A |
| no_compute | 4132 | 276.25 |

**Offload:**
| Context | H2D Count | Total GB |
|---------|-----------|----------|
| during_compute | 8331 | 5854.93 |
| no_compute | 4829 | 517.01 |

## B. Kernel Gap Overhead (Graph Breaks)

Gaps between consecutive CUDA kernels on GPU 0. Graph breaks from
`@torch.compiler.disable` hooks insert CPU-side Python overhead.

| Metric | Baseline | Offload | Delta |
|--------|----------|---------|-------|
| Total gaps | 166261 | 133708 | -32553.00 |
| Gaps > 100us | 9079 | 8756 | -323.00 |
| Gaps > 1ms | 329 | 450 | +121.00 |
| Gaps > 10ms | 90 | 237 | +147.00 |
| Total gap time | 58.73 s | 117.96 s | +59.23 s |
| Significant gap time (>100us) | 58.26 s | 117.54 s | +59.28 s |
| Avg gap | 353.24 us | 882.20 us | +528.97 us |
| Max gap | 19080.91 ms | 40964.08 ms | +21883.17 ms |

### Largest Kernel Gaps (Offload, GPU 0)

| Gap (ms) | Before Kernel | After Kernel |
|----------|---------------|--------------|
| 40964.08 | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(un | void at::native::vectorized_gather_kernel<(int)16, long>(cha |
| 23261.28 | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(un | void at::native::vectorized_elementwise_kernel<(int)2, at::n |
| 9080.10 | triton_poi_fused_all_to_all_single_clone_permute_view_0 | ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<(unsigned lo |
| 8189.34 | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(un | void at::native::vectorized_elementwise_kernel<(int)4, at::n |
| 4436.94 | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsSto | void at::native::vectorized_elementwise_kernel<(int)4, at::n |
| 3635.86 | void at::native::vectorized_elementwise_kernel<(int)4, at::n | void at::native::vectorized_elementwise_kernel<(int)2, at::n |
| 1488.08 | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsSto | void at::native::vectorized_elementwise_kernel<(int)2, at::n |
| 597.13 | void at::native::vectorized_elementwise_kernel<(int)4, at::n | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsSto |
| 491.98 | triton_poi_fused_all_to_all_single_clone_permute_view_1 | triton_poi_fused__unsafe_view_clone_permute_view_2 |
| 440.51 | triton_poi_fused_arange_0 | void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheKernel<( |

## C. Memory Allocator Overhead (cudaMalloc/cudaFree)

**Baseline:**
| API | Calls | Total Time (s) | Avg (us) | Max (us) |
|-----|-------|----------------|----------|----------|
| cudaMalloc_v3020 | 3816 | 2.29 | 600.66 | 69963.63 |
| cudaFree_v3020 | 195 | 0.07 | 348.22 | 1415.57 |
| cudaFreeHost_v3020 | 240 | 0.02 | 93.27 | 991.79 |

**Offload:**
| API | Calls | Total Time (s) | Avg (us) | Max (us) |
|-----|-------|----------------|----------|----------|
| cudaFree_v3020 | 194 | 4.75 | 24493.70 | 1010719.37 |
| cudaMalloc_v3020 | 259 | 1.29 | 4995.20 | 121920.62 |
| cudaFreeHost_v3020 | 216 | 0.02 | 107.76 | 1488.85 |

## D. NCCL Kernel Timing

**Baseline:**
| Kernel | Count | Total (s) | Avg (ms) | Max (ms) |
|--------|-------|-----------|----------|----------|
| ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<(unsigned lo | 72288 | 29.50 | 0.41 | 435.26 |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(un | 220 | 0.03 | 0.14 | 6.68 |
| ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsSto | 8 | 0.01 | 0.64 | 1.18 |

**Offload:**
| Kernel | Count | Total (s) | Avg (ms) | Max (ms) |
|--------|-------|-----------|----------|----------|
| ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<(unsigned lo | 70787 | 47.05 | 0.66 | 501.60 |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(un | 171 | 0.02 | 0.15 | 5.43 |
| ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsSto | 8 | 0.01 | 1.25 | 4.31 |

## E. GPU Utilization

**Baseline:**
| GPU | Kernels | Kernel Time (s) | Wall Time (s) | Utilization |
|-----|---------|-----------------|---------------|-------------|
| 0 | 197348 | 327.63 | 381.92 | 85.8% |
| 1 | 196169 | 327.83 | 381.91 | 85.8% |
| 2 | 190524 | 328.26 | 381.91 | 86.0% |
| 3 | 191316 | 327.84 | 381.91 | 85.8% |

**Offload:**
| GPU | Kernels | Kernel Time (s) | Wall Time (s) | Utilization |
|-----|---------|-----------------|---------------|-------------|
| 0 | 165221 | 318.35 | 430.25 | 74.0% |
| 1 | 187198 | 333.75 | 446.39 | 74.8% |
| 2 | 188258 | 333.61 | 446.39 | 74.7% |
| 3 | 188893 | 333.46 | 446.39 | 74.7% |

## Summary

### Per-Step Estimates (27 steps)

- Baseline: ~14.15 s/step
- Offload:  ~15.94 s/step
- Delta:    ~+1.79 s/step (+12.7%)

### Overhead Decomposition

- H2D transfer time delta: +113.39 s
- Kernel gap time delta (>100us): +59.28 s
- NCCL total time delta: +17.55 s

### Key Questions Answered

**Q1: H2D/NCCL contention on NVLink?** 92% of H2D occurs during NCCL
  -> Significant contention (unexpected on NVLink -- investigate)

**Q2: Graph break overhead?** Baseline: 58.26s, Offload: 117.54s
  -> Graph breaks add significant overhead

**Q3: H2D overlaps with compute?** 92% of H2D during compute kernels
  -> Good overlap -- prefetch is working

**Q4: Allocator churn?** Baseline: 3816 mallocs, Offload: 259 mallocs
  -> Allocator churn is manageable
