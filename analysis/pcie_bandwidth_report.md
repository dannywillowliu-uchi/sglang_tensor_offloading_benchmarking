# PCIe Bandwidth Analysis Report

**Date:** 2026-02-09
**Data source:** torch.profiler traces from jobs 1434113 (new offload) and 1434114 (old offload)
**Hardware:** ACES cluster, 4x H100 PCIe (Gen5 x16), no NVLink

## Executive Summary

**PCIe is severely underutilized in the old offload path.** The old FSDP-based offloading achieves only **5.38 GB/s average** (8.5% of the 63 GB/s PCIe Gen5 x16 peak) for layer transfers, spending 34.6 seconds on blocking memory copies during inference. The new layerwise offload reduces blocking copy time to 0.41 seconds by using async prefetch, but introduces per-step synchronization overhead that accumulates to ~52 seconds total.

**Key finding: The bottleneck is not raw PCIe bandwidth -- it's how the transfers are scheduled.**

## Data Limitations

The torch.profiler traces contain **CPU-side events only** (categories: `python_function`, `cpu_op`, `user_annotation`). GPU-side events (`cuda_runtime`, `gpu_memcpy`, `kernel`) were configured but the trace gzip files were truncated before reaching them. This means:

1. Blocking copy timing includes actual transfer + synchronization overhead
2. Async copy timing reflects only CPU launch overhead (not actual transfer time)
3. Transfer direction (H2D vs D2H) cannot be determined
4. GPU-side overlap between copies and NCCL cannot be verified

PCIe vs GPU-internal copies were separated using a bandwidth heuristic: copies with effective BW > 200 GB/s are classified as GPU-internal (HBM), those with BW < 76 GB/s and duration > 0.5ms as PCIe transfers.

## Results

### Old Offload (FSDP, blocking `.to()` -- job 1434114)

| Metric | Value |
|--------|-------|
| Large PCIe transfers (>100MB) | 1,275 |
| Total data transferred | 186.3 GB |
| Total blocking time | 34.6 s |
| Weighted avg bandwidth | **5.38 GB/s** |
| Median bandwidth | 7.55 GB/s |
| P10 bandwidth | 2.43 GB/s |
| P90 bandwidth | 19.30 GB/s |
| **PCIe utilization** | **8.5%** |

Typical transfer: 167.8 MB (41,943,040 float32 params = one transformer layer), taking 10-14ms per layer.

**Bandwidth histogram (copies >1MB):**
| Range | Count | Data | Time |
|-------|-------|------|------|
| 0-5 GB/s | 461 | 59.1 GB | 24.0 s |
| 5-20 GB/s | 883 | 116.8 GB | 10.9 s |
| 20-50 GB/s | 160 | 6.8 GB | 0.2 s |
| 50-80 GB/s | 134 | 17.3 GB | 0.3 s |

**Observation:** 75% of PCIe time is spent at 0-5 GB/s. This suggests severe serialization or contention -- the transfers are individually slow, likely due to FSDP's all-gather/shard pattern adding overhead around each `.to()` call.

### New Offload (LayerwiseOffloadManager, async prefetch -- job 1434113)

| Metric | Value |
|--------|-------|
| Large blocking PCIe transfers | 66 (19x fewer than old) |
| Blocking data | 15.2 GB |
| Blocking time | 0.41 s |
| Blocking avg bandwidth | **36.72 GB/s** |
| **PCIe utilization (blocking)** | **58.3%** |
| Async copies (non-blocking) | 2,300 |
| Async data | 3,002 GB |
| Async launch overhead | 0.061 s (27 us/copy) |

**Bandwidth histogram (copies >1MB):**
| Range | Count | Data | Time |
|-------|-------|------|------|
| 0-5 GB/s | 42 | 0.5 GB | 0.4 s |
| 5-20 GB/s | 88 | 5.7 GB | 0.5 s |
| 20-50 GB/s | 154 | 5.0 GB | 0.1 s |
| 50-80 GB/s | 141 | 17.8 GB | 0.3 s |

**Observation:** The few blocking copies in new offload achieve much better bandwidth (58.3% vs 8.5%). The bulk of data (3,002 GB) moves through async copies on the copy_stream, whose actual bandwidth we cannot measure from CPU-side traces.

### Comparative Summary

| Metric | New Offload | Old Offload |
|--------|-------------|-------------|
| Total copy events | 8,325 | 9,818 |
| Blocking / Async | 5,721 / 2,604 | 9,320 / 498 |
| Likely PCIe events | 1,272 | 4,526 |
| PCIe data volume | 29.1 GB | 200.1 GB |
| **PCIe blocking time** | **1.7 s** | **35.8 s** |
| NCCL events | 17,284 | 17,284 |

## Analysis

### Why is old offload's PCIe bandwidth so low?

1. **FSDP overhead**: The old path uses `CPUOffloadPolicy` with FSDP sharding. Each layer transfer involves `all_gather` (collect shards) + `.to(device)` (transfer) + compute + `reduce_scatter` (re-shard). The `.to()` calls are individually small (168MB) and synchronous.

2. **No pipelining**: Every layer's transfer must complete before its computation begins, and the next layer's transfer doesn't start until the current layer's computation completes.

3. **Small transfer size**: At 168MB per transfer, PCIe cannot reach peak throughput. PCIe Gen5 x16 has significant protocol overhead per transaction -- sustained peak requires larger transfers or pipelining.

4. **Possible CUDA synchronization**: The `_manage_device_placement` code issues blocking `.to()` calls, which involve implicit `cudaStreamSynchronize`.

### Why does new offload achieve better per-copy bandwidth but add per-step overhead?

1. **Async prefetch eliminates blocking copy time**: The `copy_stream` runs H2D transfers concurrently with computation. Only the first layer of each step may block (waiting for prefetch to complete).

2. **Per-step overhead source**: The ~2s/step overhead likely comes from `current_stream.wait_stream(copy_stream)` synchronization -- if the prefetch hasn't finished by the time the layer is needed, the compute stream stalls. With 27 steps x 40 layers, each stall adds up.

3. **Better per-copy BW**: The blocking copies that DO occur achieve 36.72 GB/s because the async architecture means they tend to be larger, more contiguous transfers.

### Is PCIe the bottleneck?

**For old offload: Yes.** 34.6s of 714s total (~4.9%) is pure blocking PCIe wait time, and each transfer blocks the GPU from computing.

**For new offload: Partially.** Only 1.7s of blocking PCIe time, but the async prefetch adds synchronization overhead (~2s/step x 26 steps = ~52s). The question is whether the copy_stream can overlap effectively with compute. This requires GPU-side CUDA activity traces to answer definitively.

## Theoretical PCIe Budget

For Wan2.2 with 40 transformer layers at ~168MB each:
- **Per-step data:** 40 layers x 168MB = 6.7 GB (if transferring all layers)
- **At PCIe peak (63 GB/s):** 6.7 GB / 63 = 0.11s per step
- **At observed old offload BW (5.38 GB/s):** 6.7 GB / 5.38 = 1.25s per step
- **Computation time per step:** ~20-23s

So at peak PCIe bandwidth, layer offloading should add only **0.5% overhead per step** (0.11s / 23s). Even with realistic overhead, if we can sustain 30 GB/s, that's 0.22s per step -- still under 1% overhead.

**The problem is not PCIe capacity -- it's scheduling efficiency.**

## Recommendations

### 1. Get proper PCIe traces (Priority: HIGH, Cost: 1 run)
Use nsys instead of torch.profiler to capture CUPTI-level Memcpy events:
```bash
nsys profile --trace=cuda,nvtx,osrt --output=profile python3 -m sglang.multimodal_gen.launch_server ...
```
This gives us: actual H2D/D2H direction, bytes transferred, GPU-side timing, overlap with compute kernels.

### 2. Run bandwidthTest microbenchmark (Priority: MEDIUM, Cost: 0 runs)
Run CUDA samples bandwidthTest on the ACES nodes to establish theoretical peak:
```bash
/usr/local/cuda/extras/demo_suite/bandwidthTest --device=0
```
This tells us the actual achievable PCIe bandwidth independent of SGLang.

### 3. Investigate prefetch depth tuning (Priority: HIGH, Cost: 0 runs)
The `LayerwiseOffloadManager` has `prefetch_size` parameter (default=2). Source code analysis can determine:
- Whether increasing prefetch_size would help overlap more
- What the VRAM cost would be per additional prefetched layer
- Whether the copy_stream utilization can be improved

### 4. Ensure trace files are not truncated (Priority: HIGH)
Allocate more scratch space for traces, or use `profile_memory=False` and `with_stack=False` to reduce trace size:
```python
torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,  # reduces trace size significantly
    profile_memory=False,
)
```

## Files

- `analysis/analyze_pcie_from_copies.py` -- Main analysis script (heuristic PCIe separation)
- `analysis/analyze_copy_events.py` -- Raw copy event analysis
- `analysis/trace_event_survey.py` -- Trace format discovery
- `analysis/analyze_pcie_bandwidth.py` -- Original attempt (found 0 memcpy events)
