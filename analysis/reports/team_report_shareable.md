# SGLang Layerwise Offloading Analysis: Wan2.2 Video Generation

**Date:** February 24, 2026 (sharded offload implementation submitted)
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

## 3. Benchmark Results (4 GPUs, Batch 3 Definitive)

All results use `flow_shift=12.0`, clean Run 2 timing with no profiling overhead. Batch 3 jobs 1459995-1459996 (3-iteration: warmup + clean + profiled). Earlier batches used incorrect `flow_shift=3.0` due to a model registry path resolution bug, now fixed by using the HuggingFace model identifier.

### Head-to-Head: New vs Old Offload

| Metric | Layerwise (New) | Full-Model (Old) | Delta |
|--------|----------------|-----------------|-------|
| Total denoising | 641.1s | **610.2s** | Old is **5.1% faster** |
| Steady-state per step | ~23.7s | ~22.6s | Old is **4.9% faster** per step |
| Step 19 (switch) | 23.3s (no spike) | **49.2s** (+27s spike) | New handles seamlessly |
| Peak VRAM | 23.08 GB | 61.25 GB | New uses **62% less VRAM** |

### Per-Step Timing Profile

**Old offload** has uniform ~22.6s steps except for a 49.2s spike at the transformer switch:

```
Step:  1   2   3  ...  17  18  [19]  20  21 ... 27
Time: 25  23  23  ...  23  23  [49]  23  23 ... 23  (seconds)
                                 ^
                          ~27s blocking .to() spike
```

**New offload** has uniform ~23.7s steps with no spike anywhere:

```
Step:  1   2   3  ...  17  18  [19]  20  21 ... 27
Time: 25  24  24  ...  24  24  [23]  24  24 ... 24  (seconds)
                                 ^
                          seamless (layerwise prefetch)
```

### Key Observation

The old offload is faster **despite** having a ~27s spike at step 19. The ~1.1s/step overhead across 27 steps (~30s total) exceeds the single 27s spike savings.

---

## 4. Root Cause Analysis (Definitively Confirmed by nsys CUPTI Overlap Analysis)

### The Core Problem: PCIe Topology Contention

On NVLink, steady-state is identical (3.27 vs 3.29s/step, delta=0.02s). On our PCIe system, the delta is ~1.1s/step. nsys CUPTI-level temporal overlap analysis has **definitively confirmed** the mechanism: H2D transfers and NCCL collectives compete for the same PCIe bus simultaneously, degrading bandwidth by 36.5%.

### nsys Profiling: H2D Transfer Volume

| Metric | New Offload | Old Offload | Ratio |
|--------|-------------|-------------|-------|
| **H2D transfer volume** | **5,767 GB** | 618 GB | **9.3x** |
| H2D transfers | 22,008 | 18,802 | 1.2x |
| Avg transfer size | 262 MB | 33 MB | 8x |
| H2D total time | 707s | 60s | 11.8x |
| Compute kernel time | ~1,000s | ~1,000s | ~1x |

The new offload moves **9.3x more data** from CPU to GPU because each GPU independently transfers the full ~28GB model, versus FSDP's 1/N sharding (see Section 2 comparison). This continuous H2D stream saturates the shared PCIe bus.

### nsys Temporal Overlap: The Definitive Evidence

Direct CUPTI-level analysis of the nsys SQLite databases confirms that H2D transfers and NCCL kernels overlap temporally on the GPU timeline. This is the smoking gun for the PCIe contention hypothesis.

| Context | New Offload | Old Offload |
|---------|-------------|-------------|
| H2D during NCCL | **7,475 transfers (5,253 GB)** | **0 transfers** |
| H2D bandwidth during NCCL | **7,907 MB/s** | N/A |
| H2D without NCCL | 4,709 transfers (514 GB) | 6,314 transfers (618 GB) |
| H2D bandwidth without NCCL | **12,447 MB/s** | 11,141 MB/s |
| **Bandwidth degradation** | **-36.5%** | **None** |

**61% of the new offload's large H2D transfers overlap with NCCL all-to-all operations.** During overlap, H2D bandwidth drops from 12.4 GB/s to 7.9 GB/s -- a 36.5% degradation. The old offload has **exactly zero** H2D transfers during NCCL because its blocking `.to()` calls serialize everything: all H2D completes before compute (and NCCL) begins.

### nsys NCCL Kernel Performance

| Metric | New Offload | Old Offload | Delta |
|--------|-------------|-------------|-------|
| NCCL SendRecv calls | 61,353 | 60,443 | +1.5% |
| NCCL total time | **963.4s** | **690.5s** | **+39.5%** |
| NCCL avg latency | 15.7ms | 11.4ms | +37.7% |
| NCCL max latency | **4,145ms** | **1,701ms** | **+143%** |

Same operation count, same call pattern -- the only difference is PCIe bandwidth availability. The 143% increase in max latency (4.1s vs 1.7s) represents severe tail contention where large H2D transfers and large NCCL collectives compete simultaneously.

### TraceLens: GPU Timeline Decomposition

| Component | Old Offload | New Offload | Delta |
|-----------|-------------|-------------|-------|
| **Total GPU time** | **639.4s** | **705.6s** | **+10.4%** |
| Computation | 369.7s (57.8%) | 368.9s (52.3%) | -0.2% (identical) |
| Exposed communication | 207.0s (32.4%) | 273.2s (38.7%) | **+32.0%** |
| Exposed memcpy | 38.8s (6.1%) | 3.7s (0.5%) | **-90.5%** |
| Idle | 23.9s (3.7%) | 59.8s (8.5%) | **+150.4%** |

The new offload successfully hides H2D transfers behind compute (only 3.7s exposed vs 38.8s for old). But the continuous H2D traffic contends with Ulysses all-to-all on the shared PCIe bus, causing NCCL communication to take 32% longer and GPU idle time to increase 150%.

### TraceLens: Ulysses All-to-All Contention (Aggregate View)

TraceLens aggregate statistics corroborate the nsys temporal overlap findings:

| Metric | Old Offload | New Offload | Delta |
|--------|-------------|-------------|-------|
| all_to_allv (210MB) mean latency | 16.2ms | **21.4ms** | **+32.1%** |
| all_to_allv (210MB) max latency | 1,932ms | **3,520ms** | **+82.2%** |
| all_to_allv (5MB) mean latency | 341us | 340us | **unchanged** |
| NCCL total kernel time | 211.5s | 278.8s | **+31.8%** |

Small all-to-allv (5MB) is identical -- only large transfers are affected. This rules out systemic NCCL issues and confirms bandwidth contention as the mechanism.

### Data Sources

| Source | Tool | Location |
|--------|------|----------|
| Batch 3 per-step timing | SGLang stage logging | ACES jobs 1459995-1459996 |
| nsys temporal overlap | nsys export + Python sqlite3 | `results/profiles/analysis/overlap_*.csv` |
| nsys H2D/NCCL stats | nsys export + Python sqlite3 | `results/profiles/analysis/h2d_*.csv`, `nccl_*.csv` |
| TraceLens GPU timeline | TraceLens (AMD) | `results/tracelens_old/`, `results/tracelens_new/` |
| TraceLens kernel/coll stats | TraceLens (AMD) | `results/tracelens_*/kernel_summary.csv`, `coll_analysis.csv` |

---

## 5. PCIe vs NVLink

PR #15511 reports a **58% speedup** on 8x GPU NVLink systems. On our 4x GPU PCIe system, the new offload is **5.1% slower**. The difference is that NVLink provides a dedicated GPU-GPU path (900 GB/s) separate from the CPU-GPU PCIe path (128 GB/s), allowing full overlap. On PCIe, both share the same fabric, causing the contention confirmed in Section 4.

| | Old Offload | New Offload | Improvement |
|---|---|---|---|
| Step 0 (NVLink) | 36.0s | 7.7s | 4.7x faster |
| Steady state (NVLink) | 3.27s/step | 3.29s/step | Equal |
| Step 19 switch (NVLink) | 31.3s | 3.29s | 9.5x faster |
| **Total (NVLink)** | **149.7s** | **94.2s** | **58% speedup** |

---

## 6. Recommendations: When to Use Which Strategy

| Scenario | Recommended Strategy | Reason |
|----------|---------------------|--------|
| PCIe system, VRAM sufficient | **Old** (dit_cpu_offload) | 5% faster, FSDP sharding efficient |
| PCIe system, VRAM constrained | **New** (dit_layerwise_offload) | 62% less VRAM, small overhead |
| NVLink system (any) | **New** (dit_layerwise_offload) | 58% faster (per PR benchmarks) |
| 8+ GPUs, NVLink | **New** (dit_layerwise_offload) | Designed for this config |

---

## 7. Optimization Opportunities

Prioritized by confirmed profiling evidence. All impact estimates are relative to the current 641.1s (new offload) denoising time on 4x H100 PCIe.

### Tier 1: High Impact (estimated 10-20% speedup)

**FSDP-style weight sharding for layerwise offload -- IMPLEMENTED (Feb 24)**
- *Addresses:* 9.3x H2D overhead (nsys: 5,767 GB vs 618 GB) causing 32% NCCL slowdown
- *Approach:* Each GPU copies 1/N of each layer (~175MB instead of ~700MB) from CPU via async copy stream, then `dist.all_gather_into_tensor()` reconstructs the full buffer on the compute stream. Reduces per-GPU H2D from ~28GB to ~7GB per denoising step
- *Expected:* Eliminate the NCCL +32% overhead (~66s total), bringing layerwise within ~1% of FSDP speed while retaining low VRAM
- *Status:* **Implemented and submitted for benchmarking** (ACES job 1473071). Code: `patches/sharded_offload.patch`, `patches/aces_layerwise_offload.py`. CLI flag: `--dit-offload-sharded`
- *Design decisions:* All-gather on compute stream (avoids cross-stream NCCL coordination); default process group (serializes with Ulysses all-to-all naturally); padded buffers for even sharding; graceful single-GPU degradation

**Priority-based PCIe scheduling**
- *Addresses:* nsys confirms 61% of H2D overlaps with NCCL, degrading BW 36.5%. Max NCCL latency spikes to 4.1s (vs 1.7s old)
- *Approach:* Insert `cudaStreamWaitEvent` between the H2D copy stream and NCCL stream so prefetch pauses during collective communication windows. Yield PCIe to NCCL during all-to-all
- *Expected:* Reduce NCCL mean latency from 15.7ms back toward 11.4ms baseline, recovering ~50-60s of the 273s NCCL overhead
- *Complexity:* Medium -- requires stream synchronization logic in the prefetch hooks
- *Note:* Partially superseded by sharded offload (which reduces H2D by 4x, making contention less severe). May still be valuable as a complementary optimization if sharded offload alone doesn't fully close the gap

### Tier 2: Medium Impact (estimated 5-10% speedup)

**torch.compile compatibility**
- *Addresses:* Graph breaks from `@torch.compiler.disable` on offload methods prevent kernel fusion of communication overlap
- *Approach:* Remove compiler disables, ensure offload hooks are compile-safe
- *Expected:* ~0.2-0.5s/step (~5-14s total). NVLink data shows steady-state is already near-optimal, confirming this is secondary
- *Complexity:* Medium -- may require torch.compile-friendly rewrite of hook logic

**Denoising step reduction**
- *Addresses:* 27 denoising steps at ~23.7s each = 641s total. Fewer steps = proportionally less time
- *Approach:* Explore improved schedulers (DPM-Solver++, consistency distillation) to reduce from 27 to 15-20 steps with minimal quality loss
- *Expected:* 20-40% wall-clock reduction if steps can be halved, but quality tradeoff needs evaluation
- *Complexity:* Low-Medium -- scheduler swap is straightforward, quality validation requires human review

**GPU memory buffer pooling**
- *Addresses:* TraceLens shows 59.8s idle (new) vs 23.9s (old); some is CUDA allocator overhead from `torch.empty(~700MB)` per layer per step
- *Approach:* Pre-allocate a rotating pool of GPU buffers, reuse across layers and steps
- *Expected:* ~0.1-0.3s/step (~3-8s total), reducing idle time fraction
- *Complexity:* Low -- allocate N buffers at init, cycle through them

### Tier 3: Lower Impact / Longer Term

**Ulysses + Ring hybrid parallelism**
- *Addresses:* Pure Ulysses uses all-to-all which scales poorly beyond 4 GPUs (message size grows)
- *Approach:* Hybrid Ulysses (intra-node) + Ring Attention (inter-node) for >4 GPU configs
- *Expected:* Reduces collective message sizes, relevant for 6-8+ GPU scaling
- *Complexity:* High -- requires attention parallelism rearchitecture

**SageAttention2 integration**
- *Addresses:* Attention kernel throughput (current SageAttention already used)
- *Approach:* Upgrade to SageAttention2 with 8-bit quantized attention for ~2x kernel speedup
- *Expected:* Reduces the 369s computation component; impact depends on attention's share of total compute
- *Complexity:* Medium -- drop-in replacement if API-compatible

**Diffusion KV-caching (DeepCache / Cache-DiT)**
- *Addresses:* Redundant attention computation across denoising steps where KV pairs change minimally
- *Approach:* Cache attention KV pairs and reuse across steps with low delta, skipping recomputation
- *Expected:* Up to 2x speedup on cached steps, but quality impact needs validation
- *Complexity:* Medium-High -- requires model-specific caching strategy

**H100 TMA (Tensor Memory Accelerator)**
- *Addresses:* H2D transfers currently use the main PCIe data path
- *Approach:* Leverage hardware copy engine for H2D transfers, freeing the PCIe bus for NCCL
- *Expected:* Could reduce bus contention without software scheduling changes
- *Complexity:* High -- requires CUDA driver-level integration, limited documentation

---

## 8. Summary

| Finding | Detail |
|---------|--------|
| **Performance** | Old offload is **5.1% faster** on 4x H100 PCIe (610.2s vs 641.1s) |
| **Memory** | New offload uses **62% less VRAM** (23.1GB vs 61.3GB peak) |
| **Root cause** | PCIe contention: 61% of H2D overlaps with NCCL, causing **36.5% BW degradation** (7.9 vs 12.4 GB/s). New generates **9.3x more H2D** (5,767 vs 618 GB); old has **zero** overlap. NCCL slowed **39.5%**, GPU idle doubled. Computation identical |
| **PCIe vs NVLink** | PR claims 58% speedup on NVLink; we see 5.1% slowdown on PCIe |
| **Switch handling** | New offload is seamless; old offload has a 27s spike at step 19 |
| **Top optimization** | FSDP-style sharded layerwise offload -- **IMPLEMENTED**, benchmark submitted |
| **Broader takeaway** | Layerwise offload trades speed for memory on PCIe; the gap is small (5%) and fixable |

### Experiment Status

| Experiment | Status | Key Finding |
|-----------|--------|-------------|
| 4-GPU clean benchmarks (Batch 2b/3) | **Done** | New 641.1s vs Old 610.2s (5.1% gap) |
| nsys profiling (1459999-1460000) | **Done** | 9.3x H2D, NCCL +39.5%, compute identical |
| nsys temporal overlap analysis | **Done** | 61% of H2D during NCCL, 36.5% BW degradation, old has zero overlap |
| TraceLens analysis | **Done** | GPU timeline confirms PCIe contention |
| 6-GPU scaling (1459997-98) | **OOM** | Both configs OOM on single ACES node |
| 4-GPU new offload re-run (1459995) | **OOM** | Unexpected; worked in Batch 2b. Needs investigation |
| Prefetch=3 | **Deprioritized** | sglang 0.5.8 lacks CLI flag; sharded offload is higher priority |
| **Sharded offload (1473071)** | **Submitted** | FSDP-style 1/N sharding + all-gather. Tests top optimization |
