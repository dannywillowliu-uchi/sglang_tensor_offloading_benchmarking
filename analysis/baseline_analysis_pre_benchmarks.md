# Comprehensive Baseline Analysis: SGLang Layerwise Offloading
# Pre-Benchmark Snapshot (2026-02-13)

**Purpose:** Frozen baseline of all findings BEFORE new benchmark results.
**Jobs:** 1443732-1443737 FAILED (no HF model on compute nodes). Resubmitted as 1454665-1454670 with symlink fix.
Compare against this document when analyzing new results.

---

## 1. Hardware Context

| Spec | Value |
|------|-------|
| Cluster | ACES (TAMU) |
| GPUs | H100 **PCIe** (NOT SXM, NO NVLink) |
| GPU Memory | 80 GB HBM3 per GPU |
| CPU RAM | 488 GB |
| PCIe Gen | Gen5 x16 (~63 GB/s unidirectional peak) |
| GPU Interconnect | PCIe fabric only (no NVLink, no NVSwitch) |
| Implication | GPU-GPU (NCCL) and CPU-GPU (offload) share the same PCIe bus |

---

## 2. Model Configuration

| Parameter | Prior Benchmarks (WRONG) | New Benchmarks (CORRECT) |
|-----------|--------------------------|--------------------------|
| Model path | `./models/wan2.2` (local) | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (HF) |
| Config resolved | `WanT2V480PConfig` (base) | `Wan2_2_T2V_A14B_Config` |
| flow_shift | **3.0** | **12.0** |
| boundary_ratio | 0.875 | 0.875 |
| boundary_timestep | 875 | 875 |
| Transformer switch step | Step 9 (tqdm 10) | Step 18 (tqdm 19) |
| Registry resolution | Falls to detector (registry.py:472) | Exact match (registry.py:542-543) |
| Architecture | Dual 14B transformers (MoE at timestep level) |
| Denoising steps | 27 |
| Resolution | 720x1280, 81 frames |

**Root cause of mismatch:** Local path `./models/wan2.2` fails exact match and partial match in registry. Falls through to `"wanpipeline"` detector which maps to base `WanT2V480PConfig` (flow_shift=3.0). The correct `Wan2_2_T2V_A14B_Config` has no detector -- only an exact HF path mapping.

---

## 3. Existing Benchmark Data (flow_shift=3.0, INVALID for A14B)

### 3.1 Job-Level Summary

| Config | Job ID(s) | GPUs | Node | Profiling | Status |
|--------|-----------|------|------|-----------|--------|
| New offload (layerwise) | 1417509, 1434113 | 4 | ac055 | Yes (torch.profiler) | Complete |
| Old offload (FSDP) | 1417511, 1434114 | 4 | ac041/ac055 | Yes (torch.profiler) | Complete |
| Pure GPU (no offload) | 1417510 | 4 | - | Yes | OOM |
| Original clean runs | 1439552-1439555 | 4+6 | - | - | CANCELLED (never ran) |

### 3.2 End-to-End Timing (Jobs 1417509/1417511, single run, profiled)

| Metric | New Offload | Old Offload | Delta |
|--------|-------------|-------------|-------|
| Denoising time | 788.2s | **714.1s** | +74.1s (+10.4%) |
| Avg step time | 29.19s | 26.44s | +2.75s |
| Decoding time | 26.6s | 26.1s | +0.5s |
| Total time | 845.9s | **761.1s** | +84.8s (+11.1%) |
| Peak GPU memory | **22.88 GB** | 61.25 GB | -38.37 GB (-62.6%) |
| Free at peak | 56.77 GB | 18.40 GB | +38.37 GB |

### 3.3 Per-Step Timing (27 steps)

**Steady-state (steps 2-27, excluding step 10):**
- New offload: ~21.6s/step
- Old offload: ~19.8s/step
- Delta: **+1.8s/step (+9.1%)**

**Full step breakdown:**

| Step | New Offload | Old Offload | Delta | Notes |
|------|-------------|-------------|-------|-------|
| 1 | 227s | 169s | +58s | torch.compile warmup |
| 2-9 avg | ~21.5s | ~19.9s | +1.6s | Steady state (transformer_1) |
| 10 | 22s | **50s** | **-28s** | Transformer switch (old has spike) |
| 11-27 avg | ~21.7s | ~19.8s | +1.9s | Steady state (transformer_2) |

**Time budget decomposition:**

| Component | New Offload | Old Offload | Difference |
|-----------|-------------|-------------|------------|
| Step 1 compile | 227s | 169s | +58s |
| Step 10 spike | 22s | 50s | -28s |
| Other 25 steps | 540s (21.6s x 25) | 495s (19.8s x 25) | +45s |
| Decoding | 26.6s | 26.1s | +0.5s |
| **Total** | **815.6s** | **740.1s** | **+75.5s** |

### 3.4 Profiler Trace Summary (Jobs 1434113/1434114)

| Metric | New Offload | Old Offload |
|--------|-------------|-------------|
| Trace file size | 235 MB | 455 MB |
| Total copy events | 8,325 | 9,818 |
| Blocking / Async copies | 5,721 / 2,604 | 9,320 / 498 |
| Likely PCIe copy events | 1,272 | 4,526 |
| cudaMemcpy > 10us | 44 | 936 |
| cudaMemcpy > 100us | 1 | 79 |
| NCCL events | 17,284 | 17,284 |

**Trace limitation:** GPU-side events (cuda_runtime, gpu_memcpy) were truncated in gzip files. Only CPU-side timing available. Cannot verify actual GPU-side overlap.

---

## 4. PCIe Bandwidth Analysis (from torch.profiler CPU-side events)

### 4.1 Old Offload PCIe Performance

| Metric | Value |
|--------|-------|
| Large PCIe transfers (>100MB) | 1,275 |
| Total data transferred | 186.3 GB |
| Total blocking time | **34.6 s** |
| Weighted avg bandwidth | **5.38 GB/s** (8.5% utilization) |
| Median bandwidth | 7.55 GB/s |
| P10 bandwidth | 2.43 GB/s |
| P90 bandwidth | 19.30 GB/s |
| Typical transfer size | 167.8 MB (1 transformer layer) |

**Bandwidth distribution:**
- 0-5 GB/s: 461 events, 59.1 GB, 24.0s (69% of blocking time)
- 5-20 GB/s: 883 events, 116.8 GB, 10.9s (31% of blocking time)
- 20+ GB/s: 294 events, 24.1 GB, 0.5s (<2% of blocking time)

**Diagnosis:** 75% of PCIe time spent at 0-5 GB/s. FSDP's all-gather/shard pattern + synchronous `.to()` calls cause severe serialization.

### 4.2 New Offload PCIe Performance

| Metric | Value |
|--------|-------|
| Large blocking PCIe transfers | 66 (19x fewer) |
| Blocking data | 15.2 GB |
| Blocking time | **0.41 s** |
| Blocking avg bandwidth | **36.72 GB/s** (58.3% utilization) |
| Async copies (non_blocking) | 2,300 |
| Async data | 3,002 GB |
| Async launch overhead | 0.061s (27 us/copy) |

**Key insight:** Blocking copies achieve 7x better bandwidth (36.72 vs 5.38 GB/s). Bulk data moves via async copy_stream, whose GPU-side bandwidth cannot be measured from CPU traces.

### 4.3 Theoretical PCIe Budget

| Scenario | Per-step H2D | Time at BW | % of 23s compute |
|----------|-------------|------------|-------------------|
| Peak (63 GB/s) | 6.7 GB | 0.11s | 0.5% |
| Good (30 GB/s) | 6.7 GB | 0.22s | 1.0% |
| Old observed (5.38 GB/s) | 6.7 GB | 1.25s | 5.4% |

**Conclusion: PCIe bandwidth is NOT the capacity bottleneck. The problem is transfer scheduling efficiency.**

---

## 5. Source Code Verified Facts

All claims verified against `sglang-src/python/sglang/multimodal_gen/` with line citations.

### 5.1 New Offload (LayerwiseOffloadManager)

| Fact | Citation | Verified |
|------|----------|----------|
| Dedicated `copy_stream` for H2D | layerwise_offload.py:47 | YES |
| Async H2D via `gpu_buffer.copy_(cpu_buffer, non_blocking=True)` | layerwise_offload.py:179 | YES |
| NO D2H copy ever (CPU pinned buffer = source of truth) | layerwise_offload.py:202-205 (docstring) | YES |
| Layer 0 never released | layerwise_offload.py:212-213 | YES |
| Event-based sync (not stream sync) | layerwise_offload.py:184, 291 | YES |
| `@torch.compiler.disable` on all offload methods | layerwise_offload.py:80,157,200 | YES |
| Prefetch_size configurable via CLI | server_args.py:650-654 | YES |
| Per-layer VRAM (Wan2.2 BF16): ~670 MB | Calculated (351M params x 2 bytes) | YES |
| Consolidated pinned CPU buffers per (layer, dtype) | layerwise_offload.py:104-130 | YES |
| Weight lifecycle: alloc GPU buffer -> copy -> replace .data -> release to stub | layerwise_offload.py:176-220 | YES |
| Incompatible with cache-dit | server_args.py:1028-1033 | YES |
| NO world_size awareness (each GPU offloads full model) | layerwise_offload.py (no dist import) | YES |

### 5.2 Old Offload (FSDP CPUOffloadPolicy)

| Fact | Citation | Verified |
|------|----------|----------|
| Blocking `.to(device)` for full transformer swap | denoising.py:854, 861 | YES |
| Uses PyTorch FSDP `CPUOffloadPolicy(pin_memory=True)` | fsdp_load.py:179-180 | YES |
| `fully_shard()` wraps modules with FSDP2 | fsdp_load.py:189, 198 | YES |
| FSDP shards weights, each GPU transfers 1/N | FSDP architecture | YES |
| Full transformer stays on GPU during steady-state steps | denoising.py:837-861 | YES |

### 5.3 Mutual Exclusion

| Rule | Citation |
|------|----------|
| `dit_layerwise_offload=true` -> `dit_cpu_offload=false` | server_args.py:1026 |
| `dit_layerwise_offload=true` -> `use_fsdp_inference=false` | server_args.py:1021 |
| `dit_layerwise_offload=true` + `cache-dit` -> ValueError | server_args.py:1028-1033 |

### 5.4 Ulysses Sequence Parallelism

| Fact | Citation | Verified |
|------|----------|----------|
| All-to-all runs on compute stream (NOT separate NCCL stream) | base_device_communicator.py:142-144 | YES |
| 2 all-to-all calls per transformer layer (input + output) | layer.py:119, 151 | YES |
| Uses `current_stream()` for NCCL ops | pynccl.py:134-143 | YES |
| On PCIe: competes with H2D for bandwidth | Hardware architecture | YES |

### 5.5 Timestep Schedule

| Fact | Citation |
|------|----------|
| Formula: `sigma_shifted = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)` | scheduling_flow_unipc_multistep.py:461-471 |
| Switch condition: `t_int >= boundary_timestep` (denoising.py:870) |
| boundary_timestep = boundary_ratio * num_train_timesteps = 0.875 * 1000 = 875 |
| flow_shift=3.0 -> switch at step 9 (tqdm 10) | Calculated + matches benchmark |
| flow_shift=12.0 -> switch at step 18 (tqdm 19) | Calculated + matches PR data |

---

## 6. GPU Scaling Analysis (Source Code)

### 6.1 Critical Finding: New Offload Does NOT Scale with GPU Count

LayerwiseOffloadManager has no `world_size` awareness. Each GPU independently offloads the FULL model.

| Metric | 2 GPUs | 4 GPUs | 6 GPUs |
|--------|--------|--------|--------|
| NEW: Per-GPU H2D per step | 17.6 GB | 17.6 GB | 17.6 GB |
| NEW: Total PCIe traffic per step | 35.2 GB | 70.4 GB | 105.6 GB |
| OLD (FSDP): Per-GPU H2D per step | 8.8 GB | 4.4 GB | 2.9 GB |
| OLD (FSDP): Total PCIe traffic per step | 17.6 GB | 17.6 GB | 17.6 GB |
| Ulysses per-GPU all-to-all | ~128 MB | ~64 MB | ~43 MB |

**Prediction:** New offload overhead should INCREASE with GPU count (more total PCIe traffic), while old offload overhead should DECREASE (FSDP shards get smaller).

### 6.2 Prefetch Size Impact

| prefetch_size | Layers Ahead | VRAM Cost | Expected Effect |
|---------------|-------------|-----------|-----------------|
| 1 (default) | 1 | ~670 MB | Minimal buffer, likely sync stalls |
| 3 (testing) | 3 | ~2.0 GB | More overlap, fewer stalls |
| 5 | 5 | ~3.3 GB | Significant overlap |
| 0.2 (ratio) | 9 | ~5.9 GB | Aggressive prefetch |

---

## 7. Hypothesized Causes of New Offload's ~2s/step Overhead

Ranked by estimated contribution:

| # | Cause | Mechanism | Estimated Impact |
|---|-------|-----------|-----------------|
| 1 | **Sync stalls** | 80 CUDA stream sync calls/step (40 pre + 40 post hooks) | ~1.0-1.5s |
| 2 | **Eager/compiled transitions** | `@torch.compiler.disable` on hooks, compiled forward | ~0.3-0.5s |
| 3 | **GPU memory alloc churn** | 80 `torch.empty()` calls/step (40 prefetch + 40 release) | ~0.1-0.3s |
| 4 | **PCIe contention with Ulysses** | H2D and all-to-all share bus | ~0.1-0.3s |
| 5 | **Prefetch not finishing in time** | copy_stream blocked by PCIe contention | ~0.0-0.2s |

**Total estimated: ~1.5-2.8s** (matches observed ~1.8s steady-state delta)

---

## 8. PR #15511 Reference Numbers (8 GPUs, NVLink)

For comparison against our PCIe results:

| Metric | Old (8 GPU NVLink) | New (8 GPU NVLink) | Speedup |
|--------|--------------------|--------------------|---------|
| Step 0 | ~36.0s | ~7.7s | 4.7x |
| Steps 1-17 | ~3.27s each | ~3.29s each | ~same |
| Step 18 (switch) | ~31.3s | ~3.29s | 9.5x |
| Steps 19-26 | ~3.26s each | ~3.29s each | ~same |
| Total | 149.7s | **94.2s** | **1.6x (58%)** |

**Why NVLink wins:** CPU-GPU transfers (PCIe) overlap completely with GPU-GPU comm (NVLink) because they use physically separate paths. The PR's "zero-cost" claim is valid on NVLink but not on PCIe.

---

## 9. Pending New Benchmarks

### 9.0 Job History

Jobs 1443732-1443737: FAILED (exit code 1:0, <1 min each). Root cause: compute nodes have no internet, `Wan-AI/Wan2.2-T2V-A14B-Diffusers` not in HF cache.

**Fix:** Created symlink `$SCRATCH/sglang-offload-research/Wan-AI/Wan2.2-T2V-A14B-Diffusers -> ../models/wan2.2` so `os.path.exists()` succeeds locally and registry exact-matches the HF path to `Wan2_2_T2V_A14B_Config` (flow_shift=12.0).

### 9.1 Job Matrix (Resubmitted 2026-02-15)

| Job | Config | Key Flags | Purpose |
|-----|--------|-----------|---------|
| 1454665 | New offload (4-GPU) | `--dit-layerwise-offload true` | Baseline new |
| 1454666 | Old offload (4-GPU) | `--dit-layerwise-offload false` | Baseline old |
| 1454667 | Pure GPU (4-GPU) | No offload flags | Memory ceiling test |
| 1454668 | New offload (6-GPU) | `--dit-layerwise-offload true --num-gpus 6` | GPU scaling |
| 1454669 | Old offload (6-GPU) | `--dit-layerwise-offload false --num-gpus 6` | GPU scaling |
| 1454670 | New + prefetch=3 (4-GPU) | `--dit-offload-prefetch-size 3` | Prefetch depth test |

All use `--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers` (symlinked to local model, correct flow_shift=12.0).
Each runs 3 iterations: warmup -> clean timing -> profiled.

### 9.2 What to Check in New Results

| Check | Method | Expected (if hypothesis correct) |
|-------|--------|----------------------------------|
| flow_shift=12.0 loaded | Log grep for `flow_shift` or `shift` | Should see 12.0 in config |
| Switch at step 18/tqdm 19 | Per-step timing analysis | Old offload spike at step 19, not step 10 |
| New offload still slower on 4 GPU | Compare denoising times | Likely still slower due to PCIe, but gap may change |
| Pure GPU: OOM or not? | Job completion status | Likely OOM (dual 14B > 4x80GB) |
| Prefetch=3 helps | Compare step times vs prefetch=1 | Should reduce sync stalls, maybe 0.5-1s/step improvement |
| nsys shows H2D/compute overlap | nsys timeline view | New: partial overlap; Old: no overlap (blocking) |
| nsys shows PCIe contention | Concurrent memcpy + NCCL on timeline | Both should be visible on same PCIe lanes |

### 9.3 Key Comparisons to Make

**A. Old vs New with correct flow_shift:**
- Does the 7% gap widen, narrow, or reverse?
- With switch at step 18 instead of step 10: old offload's spike costs the same but affects different step distribution

**B. Prefetch=3 vs Prefetch=1:**
- Per-step delta reduction?
- VRAM increase (should be ~1.3 GB more)?
- Does it reduce the sync stall component specifically?

**C. nsys actual PCIe bandwidth:**
- What's the real H2D bandwidth for new offload async copies?
- How much does Ulysses all-to-all traffic actually consume?
- Can we see copy_stream and compute stream overlap on the GPU timeline?

---

## 10. Open Questions for Future Work

1. **2-GPU scaling test:** If new offload is faster at 2 GPUs (less PCIe contention), proves the bottleneck
2. **Sequence length 10k comparison on different hardware:** Compare PCIe vs NVLink systems directly
3. **LayerwiseOffloadManager world_size-aware variant:** Transfer only the shard each GPU needs
4. **Quantized weight transfers:** INT8/FP8 transfers to halve PCIe traffic
5. **Copy stream priority elevation:** Could reduce sync stalls
6. **Multiple copy streams for pipelining:** Better PCIe utilization
7. **Schedule-aware prefetch timing:** Prefetch more aggressively during compute-heavy steps

---

## 11. File Index

| File | Contents |
|------|----------|
| `analysis/baseline_analysis_pre_benchmarks.md` | THIS FILE -- frozen baseline |
| `analysis/offload_research_report.md` | Master research report (living document) |
| `analysis/4gpu_benchmark_results.md` | Detailed 4-GPU benchmark results |
| `analysis/pcie_bandwidth_report.md` | PCIe bandwidth analysis from torch.profiler |
| `analysis/source_code_research_report.md` | Prefetch depth, tunable params, GPU scaling |
| `analysis/agent_a_ulysses_overlap.md` | Agent A: Ulysses overlap analysis (Q2) |
| `analysis/agent_b_step10_analysis.md` | Agent B: Step 10 switch analysis (Q3) |
| `analysis/agent_c_offload_deepdive.md` | Agent C: Offload implementation deep dive (Q4-Q6) |
| `analysis/team_report_shareable.md` | PI-ready summary report |
| `results/benchmark_summary_1434113_1434114.md` | Profiled run comparison |
| `results/traces/` | torch.profiler trace files (truncated gzip) |
| `analysis/analyze_pcie_from_copies.py` | PCIe bandwidth analysis script |
