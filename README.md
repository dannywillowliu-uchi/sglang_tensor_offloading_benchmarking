# Topology-Aware Offloading for Distributed Transformer Inference

## The Problem

CPU-GPU weight offloading and NCCL distributed communication share the PCIe bus on non-NVLink systems, causing bandwidth contention that degrades inference latency. The same SGLang layerwise offload code shows +0.25% overhead on NVLink but +13% on PCIe.

## Key Results

| Finding | Evidence |
|---------|----------|
| PCIe contention is the root cause | 61% of H2D overlaps with NCCL, -36.5% BW degradation (nsys CUPTI) |
| Topology determines overhead | +0.25% on NVLink vs +13% on PCIe, same software |
| R ratio predicts optimal strategy | R = W/(B_p * T_ffn) -- single scalar, computed from specs |
| Three regimes across 11 models | Video DiT (R<<1), image DiT (R~1-20), LLM decode (R~540) |
| LLM decode R is a hardware constant | R = 2 * B_HBM / B_p ~ 540, independent of model size |
| Sharding hurts when R<1 | Strategy S: 23.9s/step vs unsharded 23.7s (confirmed by theorem) |
| NCCL warmup eliminates step-1 cost | 35.6s -> 22.8s with --warmup flag |

## Start Here

| What you want | Where to look |
|---------------|---------------|
| The paper | [`report/paper.tex`](report/paper.tex) (LaTeX) or [`report/paper_draft.md`](report/paper_draft.md) (markdown) |
| Formal model and theorem | [`analysis/bandwidth_contention_model.md`](analysis/bandwidth_contention_model.md) |
| Proofs (3-case optimality + optimal k*) | [`analysis/formal_proofs.md`](analysis/formal_proofs.md) |
| R ratio for 11 models | [`analysis/r_ratio_table.md`](analysis/r_ratio_table.md) |
| Literature review (30+ papers) | [`analysis/literature_review.md`](analysis/literature_review.md) |
| Root cause analysis (nsys data) | [`analysis/reports/root_cause_analysis.md`](analysis/reports/root_cause_analysis.md) |
| NVLink vs PCIe comparison | [`analysis/reports/cross_platform_comparison.md`](analysis/reports/cross_platform_comparison.md) |
| Earlier technical report | [`report/topology_aware_offload.md`](report/topology_aware_offload.md) |

## Open / Pending

- **Strategy P (PCIe-aware scheduling):** Key experiment. Predicted to close the gap to ~23.0s/step. Job queued on ACES (1539465). Patch at [`benchmarks/aces/apply_pcie_aware.py`](benchmarks/aces/apply_pcie_aware.py).
- **LLM contention validation:** Failed due to SGLang 0.5.8 OffloaderV2 bug with Qwen2. Needs rework.
- **"Phase" offloading:** PI's new direction -- sub-layer phase-based prefetch, faster than no_offload on A30. Needs profiling to explain.

## Repository Structure

```
report/
  paper.tex                  LaTeX paper draft (share this)
  paper_draft.md             Markdown version
  topology_aware_offload.md  Earlier technical report

analysis/
  bandwidth_contention_model.md   Formal model v2
  formal_proofs.md                Theorem 1 + Corollary (optimal k*)
  r_ratio_table.md                R ratio for 11 models
  literature_review.md            30+ papers, gap statement
  reports/                        Definitive empirical analyses
    root_cause_analysis.md          nsys overlap proof (alpha=0.365)
    cross_platform_comparison.md    B200 NVLink vs H100 PCIe
    b200_offload_report.md          B200 profiling details
    b200_nsys_analysis.md           B200 nsys deep dive
  archive/                        Intermediate research artifacts
  scripts/                        Python/shell analysis scripts

benchmarks/
  aces/slurm/                     SLURM scripts (named by GPU config + strategy)
  aces/slurm/experimental/        Experimental scripts (Strategy P, S, NCCL warmup, LLM)
  aces/apply_pcie_aware.py        Python patcher for Strategy P on SGLang 0.5.8
  b200/                           B200 NVLink experiment scripts

results/
  aces/                           ACES logs (benchmark + profiling)
  b200/                           B200 metrics and profiling reports

patches/
  pcie_aware_offload.patch        Strategy P: FFN-phase H2D scheduling
  sharded_offload.patch           Strategy S: FSDP-style weight sharding
```

## Experimental Setup

- **Model:** Wan2.2-T2V-A14B (14B MoE video diffusion, 2x40 layers, ~700MB/layer BF16)
- **Task:** 720x1280 video, 81 frames, 27 denoising steps
- **Platform A:** 4x H100 PCIe, shared root complex, ACES cluster (SGLang 0.5.8)
- **Platform B:** 4x B200 NVLink mesh, local workstation
- **Profiling:** nsys CUPTI (SQLite analysis), torch.profiler + TraceLens, tqdm per-step timing
