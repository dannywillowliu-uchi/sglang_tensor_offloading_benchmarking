# Formal Bandwidth Contention Model for Topology-Aware Offloading

**Status:** Working Draft v2 (2026-03-15)
**Goal:** Formalize the relationship between hardware topology, offloading strategy, and inference latency. Theoretical foundation for NeurIPS submission.
**Ancestors:** Martinasso/Hoefler PCIe congestion graph (SC 2016), FlexGen LP (ICML 2023), MoE-Lightning HRM (ASPLOS 2025)

---

## 1. Problem Statement

Consider distributed inference of a transformer model with $L$ homogeneous layers across $N$ GPUs. Memory constraints require offloading layer weights to CPU between uses. During each layer's forward pass, inter-GPU collective communication (NCCL all-to-all for sequence parallelism, or all-reduce for tensor parallelism) runs concurrently on the same or separate physical interconnect as the CPU-GPU (H2D) weight transfers.

**Central question:** Given a hardware topology $\mathcal{T}$, what offloading strategy minimizes per-step inference latency?

**Key insight:** The answer depends on whether H2D and NCCL share a physical bus. On shared-bus topologies (PCIe-only), concurrent H2D and NCCL degrade each other's bandwidth. On split-bus topologies (NVLink+PCIe), they use separate paths and do not contend.

---

## 2. System Model

### 2.1 Hardware Topology

We model the hardware as a tuple $\mathcal{T} = (N, B_p, B_n, \gamma)$ where:

| Symbol | Description | Unit |
|--------|-------------|------|
| $N$ | Number of GPUs | -- |
| $B_p$ | Peak unidirectional PCIe bandwidth per GPU | GB/s |
| $B_n$ | Peak unidirectional NVLink bandwidth per GPU (0 if absent) | GB/s |
| $\gamma$ | Bus sharing coefficient: fraction of NCCL traffic that traverses PCIe | [0, 1] |

**Topology classification:**
- $\gamma = 0$: Split-bus (NVLink for NCCL, PCIe for H2D). E.g., DGX/SXM with NVLink mesh.
- $\gamma = 1$: Shared-bus (all traffic on PCIe). E.g., H100 PCIe with shared root complex.
- $0 < \gamma < 1$: Hybrid (partial NVLink coverage). E.g., partial NVLink topologies.

This generalizes Martinasso et al.'s (SC 2016) per-link congestion factor to a single aggregate coefficient capturing the topology's impact on offload-collective interference.

### 2.2 Transformer Layer Model

Each transformer layer consists of two phases executed sequentially:

1. **Attention phase** (duration $T_{attn}$): Includes QKV projection, attention computation, and NCCL collective communication (all-to-all for Ulysses SP, or all-reduce for TP).
2. **FFN phase** (duration $T_{ffn}$): Feed-forward network computation. No NCCL communication.

Total compute time per layer: $T_c = T_{attn} + T_{ffn}$

The NCCL collective during attention has base duration $T_{nccl}$ (uncontended).

### 2.3 Offloading Model

Layer weights of size $W$ bytes must be transferred from CPU pinned memory to GPU (H2D) before each layer's forward pass. The H2D transfer is issued on a dedicated CUDA copy stream, concurrent with computation on the default stream.

**Base H2D transfer time (uncontended):**
$$T_{h2d}^0 = \frac{W}{B_p}$$

---

## 3. Contention Model

### 3.1 Bandwidth Degradation Under Concurrent Traffic

Following Martinasso et al. (SC 2016), when multiple data flows share a physical link, effective bandwidth degrades. We model this as:

When H2D (copy stream) and NCCL (compute stream) are active simultaneously on a shared PCIe bus, both experience bandwidth degradation:

$$B_p^{eff} = B_p \cdot (1 - \alpha \cdot \gamma)$$

where $\alpha \in [0, 1]$ is the **contention coefficient** -- the fractional bandwidth loss experienced by H2D when NCCL is co-active on the same bus. This coefficient captures the aggregate effect of:
- PCIe root complex arbitration between H2D and NCCL flows
- GPU copy engine contention (limited CE units shared between H2D and NCCL)
- HBM controller contention (H2D writes and NCCL reads/writes compete)
- PCIe protocol overhead under multi-flow conditions

**Empirical measurement:** $\alpha = 0.365$ on 4x H100 PCIe (ACES cluster), measured via nsys CUPTI: H2D bandwidth drops from 12,447 MB/s to 7,907 MB/s during NCCL windows.

**Symmetrically, NCCL experiences degradation:**
$$T_{nccl}^{contended} = \frac{T_{nccl}}{1 - \beta \cdot \gamma}$$

where $\beta$ is the NCCL-side contention coefficient. Empirically, $\beta \approx 0.39$ (NCCL SendRecv avg: 11.4ms uncontended vs 15.7ms contended, +37.7%).

### 3.2 Temporal Overlap Fraction

Not all H2D transfers overlap with NCCL. Define $\phi \in [0, 1]$ as the fraction of H2D transfer time that overlaps with NCCL execution.

In the current SGLang layerwise offload design:
- H2D prefetch for layer $i+1$ runs during layer $i$'s forward pass
- Layer $i$'s forward pass includes NCCL during attention
- Measured: $\phi = 0.61$ (61% of H2D transfers overlap temporally with NCCL)

The effective per-layer contention penalty is:
$$\Delta T_{contention} = \phi \cdot T_{nccl} \cdot \frac{\beta \cdot \gamma}{1 - \beta \cdot \gamma}$$

---

## 4. Strategy Analysis

### 4.1 Strategy U: Unsharded Async Offload (Status Quo)

Each GPU independently copies full layer $W$ from CPU. Prefetch on copy stream, compute+NCCL on default stream.

**Per-layer latency:**
$$T_U = \begin{cases}
T_c + \Delta T_{contention} & \text{if } T_{h2d}^{contended} \leq T_c \\
T_{h2d}^{contended} + \Delta T_{contention} & \text{if } T_{h2d}^{contended} > T_c
\end{cases}$$

where $T_{h2d}^{contended} = W / B_p^{eff} = W / (B_p(1 - \alpha\gamma))$.

**Numerical prediction (H100 PCIe, $\gamma = 1$):**
- $T_{h2d}^{contended} = 700\text{MB} / (12.4 \cdot 0.635) = 700/7.87 = 89\text{ms}$
- $T_c = 575\text{ms}$
- $T_{h2d}^{contended} << T_c$: H2D fits within compute window
- $\Delta T_{contention} = 0.61 \cdot T_{nccl} \cdot 0.39/0.61 = 0.39 \cdot T_{nccl}$
- Per-step: $T_{step}^U = L \cdot (T_c + \Delta T_{contention}) = 40 \cdot (575 + \Delta)\text{ms}$
- Predicted overhead vs no-offload: $40 \cdot \Delta T_{contention}$
- From TraceLens: NCCL delta = +67.3s over 27 steps => +2.49s/step => +62ms/layer
- $T_{step}^U = 40 \cdot (575 + 62) = 25,480\text{ms} = 25.5\text{s}$
- **Measured: 23.7s** (model overpredicts by ~8%, likely because $\phi$ varies per layer)

**Numerical prediction (B200 NVLink, $\gamma \approx 0$):**
- $T_{h2d}^{contended} = 700/12.4 = 56\text{ms}$ (no contention)
- $\Delta T_{contention} \approx 0$ (separate paths)
- $T_{step}^U = 40 \cdot 300 = 12,000\text{ms} = 12.0\text{s}$
- **Measured: 12.03s** (model predicts within 0.25%)

### 4.2 Strategy S: Sharded Offload

Each GPU copies $W/k$ bytes (its shard), then all-gathers to reconstruct the full layer.

**Per-layer latency:**
$$T_S(k) = T_{h2d}^{shard} + T_{allgather}(k) + T_c$$

where:
- $T_{h2d}^{shard} = W/(k \cdot B_p)$ (sharded H2D, overlapped with previous layer's compute)
- $T_{allgather}(k) = \frac{W(k-1)}{k \cdot B_{ag}}$ where $B_{ag} = B_n$ if NVLink, $B_p$ if PCIe

The key advantage: sharding reduces H2D volume by $k$x, reducing PCIe pressure. The all-gather adds NCCL traffic, but:
- On NVLink: all-gather uses NVLink ($B_n >> B_p$), so $T_{allgather}$ is small
- On PCIe: all-gather uses PCIe, but replaces distributed H2D (which was also PCIe) with a single coordinated collective that NCCL can optimize internally

**When is sharding beneficial on PCIe?**

Sharding helps when the reduction in H2D contention exceeds the all-gather cost:

$$T_U - T_S(k) > 0$$
$$\Delta T_{contention}^{unsharded} - \Delta T_{contention}^{sharded} > T_{allgather}(k)$$

Since sharded H2D is $k$x smaller, it completes faster and overlaps less with NCCL, reducing $\phi$. Additionally, the FSDP-style all-gather can be topology-aware (NCCL internally optimizes for the PCIe tree structure).

**Numerical estimate (k=N=4, H100 PCIe):**
- Sharded H2D per layer: 700/4 = 175 MB, takes 175/12.4 = 14ms
- At 14ms, the H2D completes before NCCL starts => $\phi \approx 0$ => $\Delta T_{contention} \approx 0$
- All-gather: 700 * 3/4 / 12.4 = 42ms (PCIe all-gather, but NCCL-optimized)
- Net per-layer: $T_c + 42\text{ms}$ (all-gather is on critical path but no contention)
- Per-step: $40 \cdot (575 + 42) = 24,680\text{ms} = 24.7\text{s}$
- vs unsharded: 25.5s predicted => **-0.8s savings from sharding**
- vs old FSDP: 22.6s measured -- gap narrows from +2.9s to +2.1s

Note: This estimate is conservative. NCCL's topology-aware all-gather may achieve better than raw $B_p$ by pipelining.

### 4.3 Strategy P: PCIe-Aware Scheduled Offload

Schedule H2D transfers exclusively during the FFN phase, when no NCCL communication occurs.

**Constraint:** $T_{h2d}^0 \leq T_{ffn}$ (H2D must complete within FFN window)

**Per-layer latency when constraint is satisfied:**
$$T_P = T_c \quad \text{(zero contention -- H2D fully hidden in FFN)}$$

**Per-layer latency when constraint is violated:**

Excess H2D spills into attention phase:
$$T_{spill} = T_{h2d}^0 - T_{ffn}$$
$$T_P = T_c + T_{spill} \cdot \frac{\alpha \cdot \gamma}{1 - \alpha \cdot \gamma}$$

**Feasibility check (H100 PCIe, Wan2.2):**
- $T_{h2d}^0 = 700/12.4 = 56\text{ms}$
- $T_{ffn} \approx T_c / 2 = 288\text{ms}$ (assuming roughly equal attention/FFN split)
- $56\text{ms} << 288\text{ms}$: **Constraint satisfied with 5x margin**
- Predicted per-step: $40 \cdot 575 = 23,000\text{ms} = 23.0\text{s}$
- vs unsharded: 25.5s => **-2.5s savings**
- vs old FSDP: 22.6s => **within 0.4s** (essentially closes the gap)

This is the strongest optimization for this workload.

### 4.4 Strategy Comparison Summary

| Strategy | Per-step (H100 PCIe) | Per-step (B200 NVLink) | Overhead vs no-offload |
|----------|:---:|:---:|:---:|
| U: Unsharded async | 25.5s (pred) / 23.7s (meas) | 12.0s (pred) / 12.03s (meas) | +4.9% PCIe, +0.25% NVLink |
| S: Sharded (k=4) | ~24.7s (pred) | ~12.0s (pred) | ~+3.5% PCIe, ~0% NVLink |
| P: PCIe-aware | ~23.0s (pred) | ~12.0s (pred) | ~+0.4% PCIe, ~0% NVLink |
| Old FSDP baseline | 22.6s (meas) | 12.0s (meas) | reference |

---

## 5. Topology-Conditional Optimality

### 5.1 Theorem (Informal)

**Given** a transformer with $L$ layers of weight size $W$ and per-layer compute $T_c$ split into attention ($T_{attn}$, with NCCL) and FFN ($T_{ffn}$, no NCCL) phases, executed on $N$ GPUs with topology $\mathcal{T} = (N, B_p, B_n, \gamma)$:

**The optimal offloading strategy is:**

1. **If $\gamma = 0$ (split-bus / NVLink):** Strategy U (unsharded async) is optimal. H2D and NCCL use separate paths, so contention penalty $\Delta T_{contention} = 0$. Adding sharding (Strategy S) introduces unnecessary all-gather overhead. Adding FFN-aware scheduling (Strategy P) provides no benefit since there is no contention to avoid.

2. **If $\gamma = 1$ (shared-bus / PCIe) and $R = T_{h2d}^0 / T_{ffn} \leq 1$:** Strategy P (PCIe-aware scheduling) is optimal. H2D fits within the FFN compute window, achieving zero contention without adding NCCL traffic. The per-layer latency equals the no-offload case $T_c$.

3. **If $\gamma = 1$ (shared-bus / PCIe) and $R > 1$:** Strategy S (sharded offload) is preferred. The H2D volume exceeds the FFN window even at full bandwidth, so PCIe-aware scheduling cannot fully hide the transfer. Sharding reduces H2D volume by $k$x, and the resulting all-gather can be NCCL-optimized.

### 5.2 Proof Sketch

**Case 1 ($\gamma = 0$):** When $\gamma = 0$, $B_p^{eff} = B_p(1 - \alpha \cdot 0) = B_p$. The contention penalty $\Delta T_{contention} = \phi \cdot T_{nccl} \cdot \beta \cdot 0 / (1 - \beta \cdot 0) = 0$. Strategy U achieves $T_U = T_c$ (since $T_{h2d} < T_c$ by assumption for models that benefit from offloading). Any additional communication (all-gather in S) strictly increases latency. QED.

**Case 2 ($\gamma = 1$, $R \leq 1$):** Strategy P schedules H2D during FFN ($T_{h2d}^0 \leq T_{ffn}$), so H2D and NCCL never overlap ($\phi = 0$), giving $T_P = T_c$. Strategy U has $\phi > 0$ (H2D spans both phases), giving $T_U = T_c + \Delta T_{contention} > T_c$. Strategy S has $T_S = T_c + T_{allgather} > T_c$ (all-gather adds serial overhead). Therefore $T_P \leq T_U$ and $T_P \leq T_S$. QED.

**Case 3 ($\gamma = 1$, $R > 1$):** Strategy P has spillover: $T_P = T_c + (T_{h2d}^0 - T_{ffn}) \cdot \alpha/(1-\alpha)$. Strategy S with $k = N$: H2D time is $T_{h2d}^0/N$, which may satisfy $T_{h2d}^0/N \leq T_{ffn}$ (i.e., $R/N \leq 1$). If so, combining S+P (sharded + FFN-aware) achieves $T_{S+P} = T_c + T_{allgather}$. The tradeoff between spillover penalty and all-gather cost determines the optimal $k^*$:

$$k^* = \min\left(k : \frac{W}{k \cdot B_p} \leq T_{ffn}\right) = \left\lceil \frac{W}{B_p \cdot T_{ffn}} \right\rceil = \lceil R \rceil$$

This is the minimum shard count that eliminates PCIe contention.

---

## 6. Generalization: Diffusion vs LLM Inference

### 6.1 Structural Differences

| Property | Diffusion (DiT) | LLM (Autoregressive) |
|----------|:---:|:---:|
| Layers per inference | $L \times S$ (repeated over denoising steps) | $L$ per token |
| Offloading motivation | Iterative reuse; large models (14B+) | Very large models (70B+); memory-constrained |
| Parallelism | Sequence parallel (all-to-all) | Tensor parallel (all-reduce) |
| NCCL per layer | 2x all-to-all (input + output scatter/gather) | 1-2x all-reduce (after QKV proj + MLP) |
| Per-layer compute | Large ($T_c \sim$ 300-575ms) | Small ($T_c \sim$ 1-10ms for single token) |
| $R = T_{h2d}/T_{ffn}$ | $\sim 0.1-0.4$ (favorable) | $\sim 1-100$ (unfavorable) |

### 6.2 Contention Model Applicability

The contention model applies to both workloads with one critical difference:

**For diffusion ($R < 1$):** Strategy P (PCIe-aware scheduling) is optimal. The large per-layer compute window easily hides H2D transfers.

**For LLM ($R >> 1$):** H2D far exceeds compute time per layer. Neither Strategy P nor Strategy U can hide the transfer. Strategy S (sharding) is necessary to reduce H2D volume, but even with $k = N$, $R/N$ may still exceed 1 for small batch sizes.

**LLM-specific insight:** SGLang's LLM offloader (OffloaderV2) uses **blocking NCCL** on the compute stream and a separate alt_stream for prefetch. The `alt_stream.wait_stream(compute_stream)` synchronization point means H2D naturally serializes after NCCL, accidentally avoiding contention -- similar to the old FSDP offload's blocking `.to()`. This suggests LLM offloading on PCIe may not suffer the same contention as diffusion, but at the cost of serialized (non-overlapped) transfers.

### 6.3 Unified Framework

Both workloads can be analyzed with:

$$T_{total} = \sum_{s=1}^{S} \sum_{l=1}^{L} T_{layer}(l, s; \mathcal{T}, \mathcal{S})$$

where $\mathcal{S} \in \{U, S, P\}$ is the offloading strategy. The topology $\mathcal{T}$ determines which strategy minimizes $T_{total}$, and the workload class (diffusion vs LLM) determines the $R$ ratio that selects between strategies.

---

## 7. Connections to Prior Work

### 7.1 Extending Martinasso/Hoefler (SC 2016)

Their PCIe congestion graph assigns per-link contention factors based on concurrent traffic patterns:
$$B_{eff}(i) = B_{peak} / (1 + \sum_j c_{ij} \cdot \mathbb{1}[j \text{ active}])$$

Our model specializes this to the two-flow case (H2D + NCCL) and parameterizes via a single aggregate coefficient $\alpha$. The key extension: we connect the congestion factor to an optimal offloading strategy selection, which they did not address (their work targeted weather simulation, not ML inference).

### 7.2 Extending FlexGen (ICML 2023)

FlexGen formulates tensor placement as an LP:
$$\max_{w_g, c_g} \text{throughput}(w_g, c_g) \quad \text{s.t. memory, I/O constraints}$$

Our model extends this to multi-GPU inference where the LP must additionally account for:
- NCCL collective communication sharing the I/O bus
- Topology-dependent contention coefficients
- Strategy selection (not just placement, but scheduling)

### 7.3 Extending MoE-Lightning HRM (ASPLOS 2025)

Their hierarchical roofline model:
$$T_{layer} = \max(W/B_{pcie}, W/B_{hbm}, \text{FLOPs}/\text{TFLOPS})$$

Our model adds the cross-stream contention interaction term:
$$T_{layer} = \max(W/B_{pcie}^{eff}, \text{FLOPs}/\text{TFLOPS}) + \Delta T_{contention}$$

where $B_{pcie}^{eff}$ depends on concurrent NCCL traffic.

---

## 8. Open Questions and Next Steps

### Theoretical
1. **Formal proof of optimality:** Tighten Theorem 5.1 with a proper minimization proof over the strategy space. Currently informal.
2. **Is $\alpha$ constant?** Model $\alpha$ as a function of $N$, $W$, and traffic pattern. May require queuing theory.
3. **Convexity of $k^*$ optimization:** Is the shard count optimization convex? If so, efficient solvers exist.
4. **Pipeline parallelism extension:** Model point-to-point NCCL communication for PP.

### Empirical
5. **LLM validation experiment:** Run SGLang LLM with OffloaderV2 + tensor parallelism on ACES (4x H100 PCIe). Measure whether contention occurs (hypothesis: it doesn't, due to blocking NCCL).
6. **Vary N:** Test with 2, 4, 6, 8 GPUs to characterize $\alpha(N)$.
7. **PCIe-aware and sharded experiments:** Apply our patches and measure.
8. **Different models:** Test with different layer sizes $W$ to validate the $R$ ratio prediction.

### Documentation
9. **NeurIPS paper draft:** Structure as: Problem + Model + Theory + Experiments + Generalization
10. **SGLang contribution:** Document the offloading analysis for upstream SGLang.
