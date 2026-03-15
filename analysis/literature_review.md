# Literature Review: Bandwidth Contention in Distributed Inference with Offloading

**Date:** 2026-03-15
**Purpose:** Survey prior work for NeurIPS submission on topology-aware offloading

---

## Gap Statement

**No existing paper formally models the interference between concurrent H2D offload transfers and NCCL all-to-all on a shared PCIe root complex in distributed inference.** The closest ancestor (Martinasso/Hoefler, SC 2016) models PCIe congestion graphs but predates GPU collectives in ML. Multiple systems papers (FlexGen, MoE-Lightning, HeteGen) provide informal or roofline-style performance models but none derive topology-conditional optimal sharding strategies.

---

## 1. Formal Models of PCIe Bandwidth Contention

### Martinasso, Hoefler et al. (SC 2016) -- CLOSEST ANCESTOR
**"A PCIe Congestion-Aware Performance Model for Densely Populated Accelerator Servers"**
- Models PCIe topology as congestion graph, assigns contention factors per link
- >97% accuracy on 8-GPU systems
- **Direct ancestor for our contention factor alpha**
- Limitation: models P2P GPU transfers, predates H2D-vs-NCCL in ML inference
- Has formal graph-theoretic model with algorithm for computing congestion factors
- [Paper](https://dl.acm.org/doi/10.5555/3014904.3014989)

### PCIe Bandwidth-Aware Scheduling for MIG (HPC Asia 2025)
- Uses linear regression to fit sensitivity alpha_j per job
- **Most recent work explicitly modeling alpha as a measurable hardware parameter**
- Multi-Instance GPU focus, not NCCL contention

### TCCL (ASPLOS 2024)
- Discovers better communication paths for PCIe-only GPU clusters
- Models multi-transfer contention from shared CPU host bridge and NUMA
- Up to 2.07x over NCCL
- **Most directly relevant topology-aware prior work**, but no H2D offload interaction
- [Paper](https://dl.acm.org/doi/10.1145/3620666.3651362)

### Li et al. (IEEE CAL 2019) -- Priority-Based PCIe Scheduling
- Semi-formal model for multi-tenant multi-GPU PCIe demand
- +7.6% throughput. Does not address NCCL vs H2D within single job.

### Orion (EuroSys 2024) -- Interference-aware GPU sharing
- PCIe scheduling flagged as future work. Empirical/systems paper.

### MultiPath Transfer Engine (arXiv 2512.16056, Dec 2024)
- Proposes multipath H2D via NVLink+PCIe, 4.62x bandwidth
- Acknowledges PCIe bottleneck but does not model NCCL contention

---

## 2. Communication-Computation Overlap in Distributed Inference

### TokenWeave (arXiv 2505.11329, MLSys 2026) -- CLOSEST IN SPIRIT
- Token-splitting to overlap compute and all-reduce in vLLM
- -18% latency on Llama-3.3-70B, 8xH100
- **Most similar to our scheduling insight**, but targets all-reduce (TP), not H2D+all-to-all
- Empirical, no topology analysis.

### ISO (arXiv 2409.11155, 2024)
- Intra-Sequence Overlap for LLM TP inference
- -35% latency on RTX 4090. Purely empirical.

### ASPLOS 2023 -- "Overlap Communication with Dependent Computation via Decomposition"
- Formal decomposition model on TPU v4, +14-38% throughput
- Training-focused, no H2D offload interaction

### DeepSpeed-Ulysses (arXiv 2309.14509, 2023) -- KEY CITATION
- Formal analysis: all-to-all volume is O(N) independent of sequence length
- **Most relevant formal result for our NCCL component**
- Training-focused but analysis applies to inference

### Context Parallelism for Million-Token Inference (arXiv 2411.01783, 2024, Meta)
- All-to-all vs ring-attention comparison at scale
- Empirical, no formal topology-conditioned analysis

### DSP: Dynamic Sequence Parallelism (ICML 2025)
- Reduces communication volume by 75% via dynamic dimension switching
- Empirical, not topology-conditioned

---

## 3. Optimal Sharding/Chunking for CPU-GPU Offloading

### FlexGen (ICML 2023, Oral) -- KEY FORMAL PRECEDENT
- Formulates tensor placement as LP to maximize throughput
- **Has formal optimization model**
- Our model is the multi-GPU analog with NCCL contention term
- Single-GPU only, no concurrent NCCL
- [Paper](https://arxiv.org/abs/2303.06865)

### MoE-Lightning (ASPLOS 2025) -- STRONGEST FORMAL MODEL IN AREA
- Hierarchical Roofline Model (HRM) for CPU-GPU-I/O pipelining
- Extends roofline to multi-level memory hierarchy
- **Our model extends HRM by adding the NCCL contention interaction term**
- Does not model NCCL contention; single-node; MoE-specific
- [Paper](https://arxiv.org/abs/2411.11217)

### DeepSpeed Inference (SC 2022)
- ZeRO-Inference: each GPU fetches 1/N of each layer (= our sharded offload)
- No formal model of PCIe contention under NCCL

### HeteGen (MLSys 2024)
- Theoretical formula for optimal CPU/GPU partition ratio
- No NCCL contention model

### SpeedLoader (NeurIPS 2024)
- Sub-batch scheduling to overlap activation offload with weight loading
- Empirical, 1.52-2.35x over FlexGen

### NEO (MLSys 2025, Outstanding Paper HM)
- CPU offloading for online LLM inference
- KV cache and attention offload focus. No formal NCCL contention model.

---

## 4. Topology-Aware Scheduling for Distributed Inference

### Blink (MLSys 2020) -- FOUNDATIONAL
- Topology-aware collectives via spanning trees exploiting NVLink+PCIe
- **Foundational work on NVLink vs PCIe distinction**
- No offload interaction
- [Paper](https://arxiv.org/pdf/1910.04940)

### TCCL (ASPLOS 2024) -- see above

### TACOS (arXiv 2304.05301)
- Topology-Aware Collective Algorithm Synthesizer using ILP
- Provably optimal synthesis. Focused on collective synthesis, not offload.

### GPU-to-GPU Communication (SC 2024, De Sensi/Hoefler)
- Systematic measurement on Alps, Leonardo, LUMI at scale
- Good citation for topology characterization. Empirical, no formal model.

---

## 5. DiT/Diffusion Inference Parallelism -- CLOSEST NEIGHBORS

### ScaleFusion (MLSys 2025) -- MUST DIFFERENTIATE
- Intra- and inter-layer communication scheduling for ST-DiT inference
- Same model class, but no CPU offload
- **Very close neighbor paper**

### xDiT (arXiv 2411.01738, 2024)
- Hybrid SP+PipeFusion for DiTs on both PCIe (L40) and NVLink (A100)
- **Only system comparing PCIe vs NVLink for DiT inference**
- No formal model explaining the gap. No offload.

### PipeFusion (arXiv 2405.14430, 2024)
- Patch-level pipeline parallelism for DiT
- **Best existing comparison of LLM vs DiT parallelism tradeoffs**
- Communication volume analysis but no formal proofs

### DistriFusion (CVPR 2024, Highlight) -- MIT Han Lab
- Displaced patch parallelism with async communication
- Exploits inter-step feature similarity. No bandwidth contention model.

---

## 6. Positioning Strategy

### Our novel contributions:
1. **First formal model of H2D + NCCL contention on shared PCIe root complex** -- extends Martinasso/Hoefler (SC 2016) congestion graph to ML inference setting
2. **Topology-conditional optimality theorem** (Section 4 of our model) -- no direct prior art. Proves PCIe-aware scheduling optimal when T_h2d <= T_ffn, sharded offload optimal when R > 1
3. **Same-software NVLink vs PCIe delta** (+0.25% vs +13%) as empirical validation -- first controlled experiment isolating topology effect on offload overhead
4. **Extends FlexGen's LP** (ICML 2023) to multi-GPU with NCCL contention term
5. **Extends MoE-Lightning's HRM** (ASPLOS 2025) with cross-stream contention

### Key citations for related work section:
- Martinasso/Hoefler SC 2016 (PCIe congestion model ancestor)
- FlexGen ICML 2023 (formal offload optimization ancestor)
- MoE-Lightning ASPLOS 2025 (hierarchical roofline ancestor)
- DeepSpeed-Ulysses 2023 (formal all-to-all volume analysis)
- TCCL ASPLOS 2024 (topology-aware PCIe collectives)
- ScaleFusion MLSys 2025 (closest neighbor -- same model class, no offload)
- xDiT 2024 (PCIe vs NVLink comparison, no formal model)
- Blink MLSys 2020 (foundational topology-aware collectives)

### Differentiation from ScaleFusion (MLSys 2025):
ScaleFusion optimizes communication scheduling for multi-machine DiT inference but does NOT address CPU offloading. Our work addresses the orthogonal problem of CPU-GPU offloading under PCIe contention with NCCL, and the two could be composed.
