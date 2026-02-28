# SGLang Layerwise Offload Research

Profiling study investigating why SGLang's layerwise CPU offload (PR #15511) performs differently across GPU interconnect topologies. Tested on 4x H100 PCIe (ACES cluster) and 4x B200 NVLink.

**Key finding:** Offload viability depends on hardware topology. On NVLink, async prefetch is essentially free (+0.25% overhead, 82% VRAM savings). On PCIe, H2D transfers contend with NCCL collectives on the shared bus, costing 5-13%.

## Reading the Results

- **Main report:** [`report/topology_aware_offload.md`](report/topology_aware_offload.md) -- full analysis with quantitative evidence
- **Definitive analyses:** [`analysis/reports/`](analysis/reports/) -- root cause analysis, cross-platform comparison, B200 deep dives
- **Research trail:** [`analysis/archive/`](analysis/archive/) -- intermediate reports, agent deep dives, benchmark data tables

## Repository Structure

```
report/                  Main deliverable report
analysis/
  reports/               Definitive analysis reports
  archive/               Intermediate research artifacts
  scripts/               Python/shell analysis scripts
benchmarks/
  aces/                  ACES H100 PCIe experiments (SLURM scripts, setup)
  b200/                  B200 NVLink experiments (scripts, analysis tools)
results/
  aces/                  ACES benchmark logs, nsys CSVs, TraceLens output
  b200/                  B200 nsys metrics and profiling reports
patches/                 SGLang code patches (sharded offload, PCIe-aware scheduling)
```

## Model

Wan2.2-T2V-A14B-Diffusers (MoE, 2x14B parameters, 40 transformer blocks per pipeline). 27 denoising steps, 81 frames at 720x1280.

## Profiling Tools

- **nsys** (CUPTI-level): Full-run kernel + memcpy + memory API traces, analyzed via SQLite
- **torch.profiler**: 5-step detailed traces, decomposed with TraceLens
- **End-to-end benchmarks**: 3 runs per configuration, per-step timing from progress bars
