# Source Code Research Report: Prefetch Depth, Tunable Parameters, GPU Scaling

**Date:** 2026-02-09
**Source:** SGLang source code at `/Users/dannyliu/research_work/sglang-src/`

---

## 1. Prefetch Depth Analysis (PI Question: "Can we fetch 1+ layers?")

### Answer: Yes, fully configurable via CLI

**CLI Flag:** `--dit-offload-prefetch-size <value>` (server_args.py:650-654)

| Value | Interpretation | Prefetch Layers (40-layer model) | VRAM |
|-------|---------------|----------------------------------|------|
| 0.0 (default) | Ratio | 1 layer | ~670 MB |
| 0.1 | Ratio | 5 layers | ~3.3 GB |
| 0.2 | Ratio | 9 layers | ~5.9 GB |
| 0.3 | Ratio | 13 layers | ~8.5 GB |
| 0.5 | Ratio | 21 layers | ~13.7 GB |
| 3 | Absolute | 3 layers | ~2.0 GB |
| 5 | Absolute | 5 layers | ~3.3 GB |

**Formula:** (layerwise_offload.py:341-346)
- If `value < 1.0`: `prefetch_size = 1 + int(round(value * (num_layers - 1)))`
- If `value >= 1`: `prefetch_size = int(value)`
- Clamped to `[1, num_layers]`

### How Prefetch Works

Batch prefetch pattern (layerwise_offload.py:294-297):
- **Initial:** Layers `[0, prefetch_size)` prefetched before first forward
- **During forward:** Every `prefetch_size` layers, the NEXT batch of `prefetch_size` layers is prefetched async on `copy_stream`
- **Sync:** `wait_event` on compute stream before using each layer (line 291)
- **Release:** After each layer completes, GPU memory freed (except layer 0, which stays resident)

### Key Details
- **Single copy_stream** (layerwise_offload.py:47): No priority tuning exposed
- **Per-layer size (Wan2.2 BF16):** ~670 MB (351M params × 2 bytes)
- **Layer 0 pinned permanently** (line 212-213): Never released, always resident
- **No D2H copies:** CPU pinned buffer is source of truth (line 202-205)

### Recommendation for Benchmarks
Test `--dit-offload-prefetch-size 3` (adds ~1.3 GB VRAM, prefetches 3 layers ahead). This should reduce compute stalls waiting for H2D without significant memory impact.

---

## 2. Tunable Offload Parameters (PI Question: "Different parameters to optimize?")

### Most Impactful Parameters

| # | Parameter | CLI Flag | Default | Impact |
|---|-----------|----------|---------|--------|
| 1 | **Prefetch size** | `--dit-offload-prefetch-size` | 0.0 (1 layer) | VRAM vs speed |
| 2 | **Pin CPU memory** | `--pin-cpu-memory` | True | ~2-3x faster H2D when enabled |
| 3 | **torch.compile** | `--enable-torch-compile` | False | ~1.2-1.5x faster after warmup |
| 4 | **Compile mode** | `SGLANG_TORCH_COMPILE_MODE` env | `max-autotune-no-cudagraphs` | Optimization level |
| 5 | **Autocast** | `--disable-autocast` | Auto | BF16 halves transfer size |
| 6 | **Text encoder offload** | `--text-encoder-cpu-offload` | Auto (True for video) | Frees ~8-10 GB VRAM |
| 7 | **VAE offload** | `--vae-cpu-offload` | Auto (True for video) | Frees VAE VRAM |

### Offload Strategy Matrix

| Strategy | CLI Flags | Per-GPU VRAM | Speed |
|----------|-----------|-------------|-------|
| **Layerwise (new)** | `--dit-layerwise-offload` | ~1-14 GB (depends on prefetch) | Async, good overlap |
| **Full-model (old)** | `--dit-cpu-offload` | ~28 GB (full transformer) | Blocking, slow |
| **FSDP inference** | `--use-fsdp-inference` | Distributed | Low latency (all-gather prefetch) |
| **No offload** | Neither flag | ~28 GB | Fastest |

### Mutual Exclusions (server_args.py:1013-1033)
- Layerwise offload disables: FSDP inference, dit_cpu_offload, cache-dit
- Can combine: layerwise + text_encoder_offload + vae_offload + torch.compile

### Cache-DIT (Incompatible with Layerwise Offload)
Environment variables `SGLANG_CACHE_DIT_*` enable block-skipping optimization. Cannot use with layerwise offload due to weight release conflicts.

---

## 3. GPU Scaling Analysis (PI Question: "Scalable difference across GPU configs?")

### CRITICAL FINDING: New Offload Does NOT Scale with GPUs

**LayerwiseOffloadManager has NO world_size awareness** -- each GPU independently offloads the FULL model regardless of GPU count.

| Metric | 2 GPUs | 4 GPUs | 6 GPUs |
|--------|--------|--------|--------|
| **NEW: Per-GPU H2D** | 17.6 GB | 17.6 GB | 17.6 GB |
| **NEW: Total PCIe traffic** | 35.2 GB | 70.4 GB | 105.6 GB |
| **OLD (FSDP): Per-GPU H2D** | 8.8 GB | 4.4 GB | 2.9 GB |
| **OLD (FSDP): Total PCIe traffic** | 17.6 GB | 17.6 GB | 17.6 GB |
| **Ulysses per-GPU all-to-all** | ~128 MB | ~64 MB | ~43 MB |

### Why This Matters on PCIe (No NVLink)

On ACES H100 PCIe nodes:
1. **CPU-GPU (H2D offload)** and **GPU-GPU (NCCL all-to-all)** share the same PCIe fabric
2. With new offload: N GPUs × 17.6 GB = increasing total PCIe demand
3. PCIe Gen5 x16 peak is 63 GB/s per direction per GPU, but shared fabric limits aggregate
4. More GPUs → more concurrent PCIe streams → higher contention

### Scaling Predictions

**New offload overhead INCREASES with GPU count:**
- 2 GPUs: Moderate PCIe utilization, minimal contention
- 4 GPUs: ~4× the H2D traffic, high contention with NCCL
- 6 GPUs: ~6× the H2D traffic, severe contention

**Old offload overhead DECREASES with GPU count:**
- FSDP shards weights, each GPU transfers only 1/N
- But: NCCL all-gather needed to reconstruct → adds overhead

### Ulysses All-to-All Detail
- Uses **compute stream** (base_device_communicator.py:142-144), NOT a separate NCCL stream
- This means all-to-all blocks compute, which in turn delays copy_stream progress
- On PCIe systems: all-to-all traffic competes with H2D for bandwidth

### Implication for Benchmarks
- **2-GPU run is critical:** If new offload is faster at 2 GPUs (less contention) but slower at 4+ GPUs, this proves PCIe contention is the bottleneck
- **6-GPU run would be most interesting** for the contention question but costs more SUs

---

## Recommendations for Next Steps

### Zero-Cost Actions
1. **Add `--dit-offload-prefetch-size 3` variant** to benchmark matrix (just a CLI flag change)
2. **Consider FSDP inference** (`--use-fsdp-inference`) as a third offload strategy to compare

### Benchmark Priority
1. **4-GPU: new offload vs old offload vs pure GPU** (with correct flow_shift) -- confirms baseline
2. **4-GPU: new offload with prefetch_size=1 vs 3** -- tests if more prefetch helps
3. **2-GPU: new offload vs old offload** -- tests GPU scaling hypothesis
4. **4-GPU: nsys profiling** -- gets proper PCIe bandwidth data

### Source Code Improvement Ideas
- LayerwiseOffloadManager could be made world_size-aware (transfer only the shard each GPU needs)
- Copy stream priority could be elevated for better overlap
- Multiple copy streams could pipeline transfers
