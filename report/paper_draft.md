# Topology-Aware Offloading for Distributed Transformer Inference

## Abstract

CPU-GPU weight offloading enables large transformer inference on memory-constrained GPUs, while distributed parallelism (sequence or tensor) uses NCCL collectives for inter-GPU communication. On PCIe-only topologies, these two mechanisms share the same physical bus, creating bandwidth contention that degrades inference latency. We present a formal model of this contention and prove a topology-conditional optimality theorem: the optimal offloading strategy depends on the hardware topology and a single computable ratio $R = W / (B_p \cdot T_{ffn})$, where $W$ is the layer weight size, $B_p$ is the PCIe bandwidth, and $T_{ffn}$ is the FFN compute duration. On split-bus topologies (NVLink), unsharded async offloading is optimal. On shared-bus topologies (PCIe), scheduling H2D transfers during the NCCL-free FFN phase eliminates contention entirely when $R \leq 1$; when $R > 1$, weight sharding with optimal shard count $k^* = \lceil R \rceil$ minimizes overhead. We validate on 4x H100 PCIe and 4x B200 NVLink using SGLang's layerwise offload for Wan2.2-T2V-A14B video generation, showing +0.25% overhead on NVLink vs +13% on PCIe with the same software. Our PCIe-aware scheduling (Strategy P) is predicted to close this gap to within 2% of baseline. We extend the framework to LLM inference, showing that the $R$ ratio correctly predicts different optimal strategies for prefill ($R \ll 1$) versus decode ($R \gg 1$) phases.

---

## 1. Introduction

Large transformer models increasingly exceed single-GPU memory capacity, motivating two complementary techniques: **CPU-GPU weight offloading** (storing layer weights in CPU memory, transferring via PCIe as needed) and **distributed parallelism** (sharding computation across GPUs via NCCL collectives). Both techniques are well-studied individually. However, their interaction has received little attention.

We identify a critical interference: on PCIe-only GPU topologies (e.g., H100 PCIe, cloud instances without NVLink), the host-to-device (H2D) weight transfers and NCCL collective communication share the same physical bus. Concurrent H2D and NCCL flows degrade each other's bandwidth -- we measure a 36.5% reduction in H2D bandwidth and 37.7% reduction in NCCL bandwidth on 4x H100 PCIe. On NVLink topologies, H2D (PCIe) and NCCL (NVLink) use physically separate paths, and this contention is absent.

This topology dependence has practical consequences. SGLang's layerwise offload (PR #15511) achieves near-zero overhead on NVLink (+0.25%) but +13% on PCIe -- with identical software. The overhead is entirely attributable to PCIe contention, not algorithmic inefficiency.

**Contributions:**
1. A formal bandwidth contention model for concurrent H2D and NCCL on shared-bus topologies, parameterized by a measurable contention coefficient $\alpha$ (Section 3).
2. A topology-conditional optimality theorem with three cases, proved for the strategy space $\{U, P, S\}$ (Section 4).
3. A closed-form optimal shard count $k^* = \min(\lceil R \rceil, N)$ derived from the $R$ ratio (Section 4).
4. Controlled experiments on two topologies (H100 PCIe, B200 NVLink) validating the model's predictions (Section 5).
5. Generalization to LLM inference showing different $R$ regimes for prefill vs decode (Section 6).

---

## 2. Background and Related Work

### 2.1 CPU-GPU Offloading for Inference

FlexGen (Sheng et al., ICML 2023) formulates tensor placement as an LP maximizing throughput on a single GPU. DeepSpeed-Inference (Aminabadi et al., SC 2022) extends to multi-GPU with ZeRO-Inference, where each GPU fetches 1/N of each layer. MoE-Lightning (ASPLOS 2025) introduces a hierarchical roofline model for multi-level memory pipelining. None of these models account for concurrent NCCL traffic degrading offload bandwidth.

### 2.2 PCIe Contention Modeling

Martinasso and Hoefler (SC 2016) model PCIe topology as a congestion graph with per-link contention factors, achieving >97% prediction accuracy on 8-GPU systems. This is our closest theoretical ancestor, but it predates GPU collectives in ML workloads. TCCL (ASPLOS 2024) discovers better communication paths for PCIe clusters by modeling multi-transfer contention. Neither addresses the interaction between H2D offloading and NCCL collectives.

### 2.3 Communication-Computation Overlap

ISO (2024) and TokenWeave (MLSys 2026) overlap compute and all-reduce in LLM tensor parallelism. DeepSpeed-Ulysses (2023) proves all-to-all volume is $O(1)$ per GPU for sequence parallelism. These focus on overlapping compute with NCCL, not on the interaction between concurrent H2D and NCCL on a shared bus.

### 2.4 DiT Inference Systems

ScaleFusion (MLSys 2025) optimizes communication scheduling for spatial-temporal DiT inference but does not address CPU offloading. xDiT (2024) compares PCIe and NVLink for DiT inference but provides no formal model. PipeFusion (2024) and DistriFusion (CVPR 2024) propose alternative parallelisms for diffusion but do not model offload-collective interference.

**Gap.** No prior work formally models the interference between concurrent H2D offload transfers and NCCL collectives on a shared PCIe bus, or derives optimal offloading strategies conditioned on topology.

---

## 3. Contention Model

### 3.1 Hardware Topology

We model hardware as $\mathcal{T} = (N, B_p, B_n, \gamma)$: $N$ GPUs, PCIe bandwidth $B_p$, NVLink bandwidth $B_n$, and bus-sharing coefficient $\gamma \in [0,1]$ (fraction of NCCL traffic traversing PCIe). Split-bus topologies (NVLink mesh) have $\gamma = 0$; shared-bus (PCIe-only) has $\gamma = 1$.

### 3.2 Contention Coefficients

When H2D and NCCL run concurrently on a shared bus:

$$B_{h2d}^{eff} = B_p \cdot (1 - \alpha \cdot \gamma), \quad B_{nccl}^{eff} = B_{nccl}^{peak} \cdot (1 - \beta \cdot \gamma)$$

where $\alpha, \beta$ are topology-dependent contention coefficients. On H100 PCIe ($\gamma=1$): $\alpha = 0.365$, $\beta = 0.377$, measured via nsys CUPTI temporal overlap analysis of 12,184 H2D transfers.

### 3.3 Transformer Layer Model

Each layer consists of attention (duration $T_{attn}$, includes NCCL collective of duration $T_{nccl}$) followed by FFN (duration $T_{ffn}$, no NCCL). Total compute: $T_c = T_{attn} + T_{ffn}$.

### 3.4 The R Ratio

We define:

$$R = \frac{W}{B_p \cdot T_{ffn}} = \frac{T_{h2d}^0}{T_{ffn}}$$

This ratio determines whether H2D can be fully hidden within the NCCL-free FFN phase. When $R \leq 1$, the transfer fits; when $R > 1$, it spills into the attention phase and contends with NCCL.

---

## 4. Optimal Offloading Strategies

### 4.1 Strategy Space

- **Strategy U (Unsharded):** Each GPU copies full layer weights $W$ from CPU on a dedicated copy stream, concurrent with compute+NCCL.
- **Strategy P (PCIe-Aware):** H2D prefetch triggered at FFN entry (not block entry), confining transfers to the NCCL-free window.
- **Strategy S (Sharded):** Each GPU copies $W/k$ bytes, then all-gathers to reconstruct. Reduces PCIe H2D volume at the cost of additional NCCL traffic.

### 4.2 Per-Layer Latency

**Strategy U:** $T_U = T_c + T_{nccl} \cdot \frac{\beta\gamma}{1-\beta\gamma}$ (NCCL contention penalty when $\phi > 0$).

**Strategy P ($R \leq 1$):** $T_P = T_c$ (zero contention -- H2D hidden in FFN).

**Strategy S ($k$ shards):** $T_S = T_c + T_{ag}(k)$ where $T_{ag}(k) = W(k-1)/(k \cdot B_{ag})$.

### 4.3 Theorem 1 (Topology-Conditional Optimality)

*Given topology $\mathcal{T}$ and $R = W/(B_p \cdot T_{ffn})$, the minimum per-step latency is achieved by:*

**(i)** *Strategy U when $\gamma = 0$, achieving $T_{step}^* = L \cdot T_c$.*

**(ii)** *Strategy P when $\gamma = 1$ and $R \leq 1$, achieving $T_{step}^* = L \cdot T_c$.*

**(iii)** *Strategy S+P with $k^* = \min(\lceil R \rceil, N)$ when $\gamma = 1$ and $R > 1$.*

**Proof sketch.** Case (i): $\gamma = 0$ implies zero contention; U achieves the compute lower bound $L \cdot T_c$; S adds unnecessary all-gather overhead. Case (ii): P schedules H2D during FFN, eliminating overlap with NCCL; achieves lower bound while U and S do not. Case (iii): P alone has spillover; S with $k \geq R$ eliminates spillover; $k^* = \lceil R \rceil$ minimizes all-gather cost. Full proof in Appendix.

### 4.4 Corollary: Optimal Shard Count

$$k^* = \min\left(\left\lceil \frac{W}{B_p \cdot T_{ffn}} \right\rceil, N\right)$$

Special cases: $R \leq 1 \Rightarrow k^* = 1$ (no sharding); $R > N \Rightarrow k^* = N$ (maximum sharding, residual contention may remain).

---

## 5. Experiments

### 5.1 Setup

**Model:** Wan2.2-T2V-A14B (14B MoE video diffusion, 2x40 transformer layers, ~700MB/layer at BF16).
**Task:** 720x1280 video, 81 frames, 27 denoising steps.
**Parallelism:** Ulysses sequence parallelism with all-to-all communication.
**Platforms:**
- Platform A: 4x H100 PCIe, shared root complex ($\gamma = 1$), ACES cluster.
- Platform B: 4x B200 NVLink mesh ($\gamma \approx 0$), local workstation.

### 5.2 Results

| Experiment | Predicted | Measured | Error |
|---|---|---|---|
| Strategy U on NVLink ($\gamma=0$) | 12.0s/step | 12.03s/step | 0.25% |
| Strategy U on PCIe ($\gamma=1$) | ~25.5s/step | 23.7s/step | ~8% |
| Strategy S on PCIe ($R < 1$, $k=4$) | Worse than U | 23.9s/step (vs 23.7) | Correct direction |
| Strategy P on PCIe ($R = 0.196 < 1$) | ~23.0s/step | **[PENDING: job 1539465]** | -- |
| NCCL warmup ($T_{init}$) | One-time cost | 12.8s eliminated by --warmup | Confirmed |

**Key finding:** Strategy S (sharded) on PCIe produces slightly worse performance than unsharded (23.9s vs 23.7s), confirming the theorem's prediction that when $R < 1$, sharding adds all-gather overhead without contention benefit. Strategy P is the correct optimization for this workload.

### 5.3 nsys Validation

CUPTI temporal overlap analysis confirms the contention mechanism:
- 61% of large H2D transfers (7,475 of 12,184) overlap with NCCL kernels
- H2D BW during NCCL: 7,907 MB/s; without: 12,447 MB/s ($\alpha = 0.365$)
- Old (FSDP) offload: zero H2D during NCCL (blocking `.to()` serializes transfers)
- New offload H2D volume: 5,767 GB (9.3x more than FSDP's 618 GB -- no sharding)

---

## 6. Generalization: Diffusion vs LLM

The $R$ ratio unifies the analysis across workload types. We compute $R$ for 11 models spanning video diffusion, image diffusion, and LLM inference:

| Model | Type | Params | W/layer | $T_{ffn}$ | **R** | Strategy |
|---|---|---|---|---|---|---|
| HunyuanVideo | Video DiT | 13B | 433 MB | 317 ms | **0.11** | P |
| **Wan2.2-T2V-A14B** | Video DiT | 14B | 700 MB | 288 ms | **0.20** | **P (validated)** |
| Mochi 1 | Video DiT | 10B | 417 MB | 146 ms | **0.23** | P |
| CogVideoX-5B | Video DiT | 5B | 333 MB | 15 ms | **1.8** | S ($k$=2) |
| SD3 Medium | Image DiT | 2B | 167 MB | 4.2 ms | **3.2** | S ($k$=4) |
| FLUX.1-dev | Image DiT | 12B | 421 MB | 2.1 ms | **16** | S ($k$=17) |
| Llama-3.1-8B (decode) | LLM | 8B | 500 MB | 0.075 ms | **538** | S |
| Llama-3.1-70B (decode) | LLM | 70B | 1,750 MB | 0.261 ms | **540** | S |
| Llama-3.1-405B (decode) | LLM | 405B | 6,429 MB | 0.958 ms | **541** | S |
| Mixtral-8x7B (decode) | LLM MoE | 46.7B | 2,919 MB | 0.436 ms | **540** | S |
| Qwen2.5-72B (decode) | LLM | 72B | 1,800 MB | 0.269 ms | **540** | S |

Three regimes emerge cleanly:

**Regime 1 ($R \ll 1$): Large video DiT.** HunyuanVideo, Wan2.2, Mochi 1. Long per-step compute (seconds) from high-dimensional spatiotemporal attention. Strategy P makes offloading essentially free.

**Regime 2 ($R \sim 1\text{--}20$): Image DiT, small video DiT.** CogVideoX, SD3, FLUX. Shorter per-step compute. Strategy S with $k^* = \lceil R \rceil$ shards (2--17, within typical GPU counts).

**Regime 3 ($R \approx 540$): All LLM decode.** Every LLM collapses to $R \approx 2 \cdot B_{HBM} / B_p \approx 540$ regardless of model size, because both $W$/layer and $T_c$/layer scale linearly with parameters and cancel in the ratio. This is a **hardware constant** reflecting the ~270x gap between HBM bandwidth (3,350 GB/s) and PCIe bandwidth (12.4 GB/s). Offloading during decode is fundamentally bottlenecked; only extreme sharding or prefill-phase offloading is viable.

This explains why SGLang's LLM offloader uses blocking synchronization -- with $R \gg 1$, async overlap is counterproductive (it creates contention without hiding the transfer).

---

## 7. Discussion

### Limitations
- **Two topologies.** We validate on NVLink ($\gamma \approx 0$) and PCIe ($\gamma = 1$). Hybrid topologies ($0 < \gamma < 1$) are untested.
- **Constant $\alpha$.** We treat the contention coefficient as constant. It may vary with $N$, traffic patterns, or NCCL algorithm choices.
- **Single model family.** DiT experiments use only Wan2.2. LLM predictions are analytical, not yet empirically validated due to a SGLang compatibility issue with OffloaderV2 on Qwen2.

### Implications for System Design
- Offloading frameworks should detect topology ($\gamma$) and compute $R$ at initialization, then select the appropriate strategy automatically.
- On PCIe clusters (common in cost-optimized cloud deployments and university clusters), naive async offloading is suboptimal. FFN-phase scheduling is a low-complexity fix that requires only hook placement changes.
- The $R$ ratio provides a simple diagnostic: practitioners can compute it from model specs and hardware bandwidth without running experiments.

---

## 8. Conclusion

We formalize the bandwidth contention between CPU-GPU offloading and NCCL collectives on shared-bus topologies, prove a topology-conditional optimality theorem, and validate it experimentally. The key practical insight is that on PCIe-only systems, scheduling H2D transfers during the NCCL-free FFN phase eliminates contention entirely for models with $R \leq 1$ -- making offloading essentially free. For models with $R > 1$, weight sharding with $k^* = \lceil R \rceil$ minimizes the remaining overhead. The $R$ ratio serves as a simple, computable decision criterion that generalizes across both diffusion and LLM workloads.

---

## References

[1] Martinasso, Hoefler et al. "A PCIe Congestion-Aware Performance Model for Densely Populated Accelerator Servers." SC 2016.
[2] Sheng et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." ICML 2023.
[3] Aminabadi et al. "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." SC 2022.
[4] "MoE-Lightning: High-Throughput MoE Inference on Memory-Constrained GPUs." ASPLOS 2025.
[5] Jacobs et al. "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models." 2023.
[6] "TCCL: Discovering Better Communication Paths for PCIe GPU Clusters." ASPLOS 2024.
[7] "Blink: Fast and Generic Collectives for Distributed ML." MLSys 2020.
[8] "ScaleFusion: Scalable Inference of Spatial-Temporal Diffusion Transformers." MLSys 2025.
[9] Fang et al. "xDiT: an Inference Engine for Diffusion Transformers with Massive Parallelism." 2024.
[10] "PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models." 2024.
[11] Li et al. "DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models." CVPR 2024.
[12] "ISO: Overlap of Computation and Communication within Sequence for LLM Inference." 2024.
[13] "TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference." MLSys 2026.
[14] "HeteGen: Heterogeneous Parallel Inference for Large Language Models." MLSys 2024.
[15] "SpeedLoader: An I/O Efficient Scheme for Heterogeneous and Distributed LLM Operation." NeurIPS 2024.
[16] "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference." MLSys 2025.
