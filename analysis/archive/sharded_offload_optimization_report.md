---
title: "FSDP-Style Sharded Layerwise Offload: Optimization Report"
subtitle: "Addressing PCIe Contention in SGLang's Wan2.2 Video Generation Pipeline"
author: "Danny Liu"
date: "February 24, 2026"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{float}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{Sharded Offload Optimization}
  - \fancyhead[R]{February 2026}
  - \usepackage{xcolor}
  - \definecolor{codegreen}{rgb}{0,0.5,0}
  - \definecolor{codered}{rgb}{0.7,0,0}
---

# 1. Problem Statement

SGLang PR \#15511 introduces a **layerwise offloading** strategy for video diffusion models that achieves a 58% speedup on NVLink systems by eliminating blocking transfer spikes. However, on our 4x H100 PCIe cluster (ACES), the new offload is **5.1% slower** than the existing FSDP-based offload (641.1s vs 610.2s for 27-step denoising), while using 62% less VRAM (23 GB vs 61 GB).

Our nsys CUPTI-level profiling definitively identified the root cause: **PCIe bus contention** between asynchronous H2D weight prefetching and NCCL collective communication.

# 2. Root Cause: PCIe Contention (Confirmed)

## 2.1 The Mechanism

The layerwise offload transfers the **full model** (~700 MB/layer, 28 GB/step per GPU) from CPU to GPU via async H2D on a dedicated CUDA copy stream. On PCIe systems, this H2D traffic shares the bus with Ulysses all-to-all (NCCL), causing mutual bandwidth degradation.

The old FSDP offload transfers only **1/N of each layer** (~175 MB with 4 GPUs), then uses NCCL all-gather to reconstruct. FSDP's topology-aware scheduling minimizes bus contention.

## 2.2 nsys Evidence

\begin{table}[H]
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{New Offload} & \textbf{Old Offload} & \textbf{Ratio} \\
\midrule
H2D total volume & 5,767 GB & 618 GB & 9.3x \\
H2D during NCCL windows & 5,253 GB & 0 GB & $\infty$ \\
H2D BW during NCCL & 7,907 MB/s & N/A & -- \\
H2D BW without NCCL & 12,447 MB/s & 11,141 MB/s & 1.1x \\
\textbf{BW degradation during NCCL} & \textbf{--36.5\%} & \textbf{None} & -- \\
\bottomrule
\end{tabular}
\caption{nsys temporal overlap analysis showing PCIe contention}
\end{table}

**61% of the new offload's H2D transfers overlap temporally with NCCL operations**, degrading H2D bandwidth by 36.5%. The old offload has **zero** H2D during NCCL because its blocking `.to()` serializes all transfers before compute begins.

## 2.3 Impact on NCCL Performance

\begin{table}[H]
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{New Offload} & \textbf{Old Offload} & \textbf{Delta} \\
\midrule
NCCL total time & 963.4s & 690.5s & +39.5\% \\
NCCL avg latency & 15.7ms & 11.4ms & +37.7\% \\
NCCL max latency & 4,145ms & 1,701ms & +143\% \\
all\_to\_allv (210MB) mean & 21.4ms & 16.2ms & +32.1\% \\
all\_to\_allv (5MB) mean & 340$\mu$s & 341$\mu$s & unchanged \\
\bottomrule
\end{tabular}
\caption{NCCL performance degradation from PCIe contention}
\end{table}

The key observation: **small transfers are unaffected** (340 us both), while large transfers degrade 32--143\%. This rules out software bugs and confirms bandwidth contention as the mechanism.

## 2.4 Why NVLink Doesn't Have This Problem

On NVLink systems, GPU-GPU communication (900 GB/s NVLink) and CPU-GPU transfers (128 GB/s PCIe) use **physically separate paths**. H2D and NCCL never compete. On PCIe, both share the same bus.

\begin{table}[H]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{} & \textbf{NVLink (DGX/SXM)} & \textbf{PCIe (ACES)} \\
\midrule
GPU$\leftrightarrow$GPU path & NVLink (900 GB/s) & PCIe (128 GB/s) \\
CPU$\rightarrow$GPU path & PCIe (128 GB/s) & PCIe (128 GB/s) \\
Paths shared? & No & \textbf{Yes} \\
Steady-state delta & 0.02s/step & \textbf{3.0s/step} \\
\bottomrule
\end{tabular}
\caption{NVLink vs PCIe topology and its effect on performance}
\end{table}

# 3. The Optimization: Sharded Layerwise Offload

## 3.1 Core Idea

Apply FSDP-style weight sharding to the layerwise offload paradigm: each GPU copies only **1/N of each layer** from CPU, then uses `dist.all_gather_into_tensor()` to reconstruct the full weights. This reduces per-GPU PCIe H2D traffic by Nx while retaining the layerwise offload's low VRAM and smooth transformer-switch advantages.

## 3.2 Design

### Transfer Pipeline (per layer, per step)

\begin{enumerate}
\item \textbf{Shard H2D} (copy stream): GPU $r$ copies elements $[r \cdot S, (r+1) \cdot S)$ from the CPU buffer, where $S = \lceil \text{numel} / N \rceil$. Transfer size: $\sim$175 MB (vs 700 MB baseline).
\item \textbf{Wait for H2D} (compute stream): \texttt{wait\_event} ensures the shard is on GPU before proceeding.
\item \textbf{All-gather} (compute stream): \texttt{dist.all\_gather\_into\_tensor(full\_buf, shard)} reconstructs the complete layer weights. Runs on the compute stream, naturally serialized with Ulysses all-to-all.
\item \textbf{Restore weights}: Slice the gathered buffer using existing metadata offsets and swap \texttt{.data} pointers.
\item \textbf{Forward pass}: Standard layer computation with full weights available.
\item \textbf{Release}: Free GPU memory (same as baseline).
\end{enumerate}

### Key Design Decisions

\begin{table}[H]
\centering
\begin{tabular}{p{4cm}p{8cm}}
\toprule
\textbf{Decision} & \textbf{Rationale} \\
\midrule
All-gather on compute stream & Avoids cross-stream NCCL coordination. Serializes naturally with Ulysses all-to-all, preventing deadlock. \\
\addlinespace
Default process group & All ranks execute layers 0--39 in identical order. Dedicated group deferred to v2. \\
\addlinespace
Padded CPU buffers & Ensures even division by world\_size. At most $N-1$ extra elements per buffer. \\
\addlinespace
Graceful single-GPU degradation & Falls back to non-sharded if \texttt{torch.distributed} not initialized or \texttt{world\_size=1}. \\
\addlinespace
Mutually exclusive with pcie\_aware & Both solve PCIe contention differently; combining is redundant. \\
\bottomrule
\end{tabular}
\caption{Design decisions and rationale}
\end{table}

### NCCL Ordering Safety

All-gather calls are safe because:

1. All ranks execute layers 0--39 in **identical order** (same model, same forward pass)
2. Materialize is called in the **pre-hook** (compute stream), after the previous layer's Ulysses all-to-all synchronizes all ranks
3. The default process group **serializes** all-gather with Ulysses all-to-all, preventing deadlock

## 3.3 Implementation

The optimization was implemented in two versions:

1. **Local sglang-src** (newer API): Full patch in `patches/sharded_offload.patch` modifying `layerwise_offload.py` and `server_args.py`
2. **ACES sglang 0.5.8** (production): Direct file replacement in `patches/aces_layerwise_offload.py` adapted to the older API

### Code Changes Summary

\begin{table}[H]
\centering
\begin{tabular}{p{5cm}p{7cm}}
\toprule
\textbf{Component} & \textbf{Change} \\
\midrule
\texttt{\_\_init\_\_} & Added \texttt{sharded} param, distributed init, \texttt{\_pending\_shards} and \texttt{\_shard\_info} dicts \\
\texttt{\_initialize()} & Pad CPU buffers for even sharding, compute shard boundaries \\
\texttt{prefetch\_layer()} & Sharded branch: H2D only local 1/N shard \\
\texttt{\_materialize\_sharded\_layer()} & New method: all-gather + weight restoration \\
Pre-hook & Call materialize after wait\_event when sharded \\
\texttt{release\_layer()} & Pop pending shards \\
\texttt{server\_args.py} & Added \texttt{--dit-offload-sharded} CLI flag + validation \\
\bottomrule
\end{tabular}
\caption{Implementation changes}
\end{table}

### CLI Usage

```bash
python -m sglang.launch_server \
    --dit-layerwise-offload \
    --dit-offload-sharded \
    --ulysses-degree 4 \
    ...
```

# 4. Expected Performance

## 4.1 Quantitative Predictions

\begin{table}[H]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{Current (New Offload)} & \textbf{Sharded (Expected)} \\
\midrule
H2D per GPU per step & 28 GB (700MB $\times$ 40) & 7 GB (175MB $\times$ 40) \\
PCIe root complex pressure & 4$\times$ baseline & 1$\times$ baseline (matches FSDP) \\
All-gather overhead & 0 & $\sim$280ms/step (40 $\times$ 7ms) \\
NCCL contention & +39.5\% slowdown & Eliminated \\
Net per-step time & $\sim$23.7s & $\sim$22.0--22.5s \\
Peak VRAM & 23 GB & $\sim$23 GB (unchanged) \\
Step 19 spike & None & None \\
\bottomrule
\end{tabular}
\caption{Expected performance comparison (4$\times$ H100 PCIe)}
\end{table}

## 4.2 Reasoning

The sharded offload reduces per-GPU H2D by 4x (175 MB/layer vs 700 MB/layer). This should:

- **Eliminate NCCL contention**: 4x less H2D traffic means PCIe bus pressure drops to FSDP-equivalent levels. The 39.5\% NCCL slowdown should largely disappear.
- **Add all-gather overhead**: Each layer requires an all-gather of approximately 700 MB across 4 GPUs (5--8ms per layer, 280ms/step total). This runs on the compute stream, adding to the critical path.
- **Net improvement**: Eliminating the 1.1s/step contention overhead minus 0.28s/step all-gather cost yields approximately **0.8s/step net improvement**.

## 4.3 Comparison with All Strategies

\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{} & \textbf{Old (FSDP)} & \textbf{New (Layerwise)} & \textbf{Sharded (Expected)} \\
\midrule
Per-step time & 22.6s & 23.7s & 22.0--22.5s \\
Total denoising & 610.2s & 641.1s & $\sim$594--608s \\
Step 19 spike & +27s & None & None \\
Peak VRAM & 61 GB & 23 GB & $\sim$23 GB \\
H2D/GPU/step & 7 GB & 28 GB & 7 GB \\
\bottomrule
\end{tabular}
\caption{Head-to-head comparison of all three strategies}
\end{table}

If the sharded offload achieves the expected 22s/step, it would be the **best of all worlds**: faster than FSDP (no switch spike), 62\% less VRAM, and equivalent PCIe efficiency.

# 5. Experiment Status

## 5.1 Current Benchmark

The sharded offload has been submitted as ACES job **1473071** (February 24, 2026). The SLURM script runs 3 iterations:

1. **Warmup** (torch.compile JIT compilation)
2. **Clean timing** (definitive benchmark data)
3. **Profiled** (torch.profiler trace for detailed analysis)

Configuration: 4 H100 PCIe GPUs, Ulysses degree 4, SageAttention, torch.compile, 27 denoising steps, 720x1280x81 frames.

## 5.2 Job History

\begin{table}[H]
\centering
\begin{tabular}{llll}
\toprule
\textbf{Batch} & \textbf{Jobs} & \textbf{Status} & \textbf{Key Result} \\
\midrule
Batch 2b (Feb 15) & 1454665--70 & Partial & 4-GPU clean timing obtained \\
Batch 3 (Feb 18) & 1459995--00 & Mixed & nsys profiles; 6-GPU OOM \\
\textbf{Batch 4 (Feb 24)} & \textbf{1473071} & \textbf{Pending} & \textbf{Sharded offload benchmark} \\
\bottomrule
\end{tabular}
\caption{Experiment history}
\end{table}

## 5.3 Verification Plan

Upon job completion, we will compare:

1. **Per-step timing** (Run 2, clean): target 22.5s/step or less steady-state
2. **Peak VRAM**: should remain at approximately 23 GB
3. **Total denoising time**: target 610s or less (matching or beating FSDP)
4. **Step 19 behavior**: should remain spike-free
5. **NCCL performance** (if profiled): all-to-all latency should return to FSDP-equivalent levels

# 6. Summary

\begin{table}[H]
\centering
\begin{tabular}{p{4cm}p{8.5cm}}
\toprule
\textbf{Aspect} & \textbf{Detail} \\
\midrule
Problem & Layerwise offload 5.1\% slower on PCIe due to H2D/NCCL bus contention \\
Root cause & 9.3$\times$ more H2D traffic, 61\% overlaps with NCCL, 36.5\% BW degradation \\
Optimization & FSDP-style 1/N weight sharding + all-gather in layerwise offload \\
Mechanism & 4$\times$ less H2D per GPU, all-gather serialized with Ulysses on compute stream \\
Expected result & $\sim$22s/step (vs 23.7s current), matching FSDP speed with 62\% less VRAM \\
Status & Implemented, deployed to ACES, benchmark job 1473071 pending \\
\bottomrule
\end{tabular}
\end{table}

# Appendix A: Architecture Diagram

```
              SHARDED LAYERWISE OFFLOAD (per layer)
              =====================================

CPU (pinned):  [======= Full Layer Buffer (~700MB) =======]
                |         |         |         |
                v         v         v         v
               GPU0      GPU1      GPU2      GPU3
              [175MB]   [175MB]   [175MB]   [175MB]
                \         |         |         /
                 \        |         |        /
            dist.all_gather_into_tensor (compute stream)
                 /        |         |        \
                /         |         |         \
              [======= Full Layer (~700MB) =======]
              GPU0      GPU1      GPU2      GPU3


              ORIGINAL LAYERWISE OFFLOAD (per layer)
              ======================================

CPU (pinned):  [======= Full Layer Buffer (~700MB) =======]
                |         |         |         |
                v         v         v         v
               GPU0      GPU1      GPU2      GPU3
              [700MB]   [700MB]   [700MB]   [700MB]
               (each GPU independently copies the full layer)
```

# Appendix B: Files Modified

| File | Purpose |
|------|---------|
| `patches/sharded_offload.patch` | Combined patch for local sglang-src |
| `patches/aces_layerwise_offload.py` | ACES-compatible complete file (sglang 0.5.8) |
| `patches/apply_sharded_patch.sh` | Patch application script |
| `patches/revert_sharded_patch.sh` | Patch revert script |
| `run_sharded_offload.slurm` | ACES SLURM benchmark script |
| `sglang-src/.../layerwise_offload.py` | Core implementation (local) |
| `sglang-src/.../server_args.py` | CLI argument registration (local) |
