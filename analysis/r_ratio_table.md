# R Ratio Analysis Across Models

**Date:** 2026-03-21
**Constants:** $B_p$ = 12,400 MB/s (PCIe Gen5), BF16 = 2 bytes/param, $T_{ffn} \approx T_c / 2$

---

## Summary Table

| Model | Type | Params | Layers | W/layer (MB) | T_ffn (ms) | **R** | Strategy |
|-------|------|--------|--------|-------------|-----------|-------|----------|
| HunyuanVideo (720p 129f) | Video DiT | 13B | 60 | 433 | 317 | **0.11** | P |
| Wan2.2-T2V-A14B (4xGPU) | Video DiT | 14B | 40 | 700 | 288 | **0.20** | P |
| Mochi 1 | Video DiT | 10B | 48 | 417 | 146 | **0.23** | P |
| CogVideoX-5B | Video DiT | 5B | 30 | 333 | 15 | **1.8** | S (k=2) |
| SD3 Medium | Image DiT | 2B | 24 | 167 | 4.2 | **3.2** | S (k=4) |
| FLUX.1-dev | Image DiT | 12B | 57 | 421 | 2.1 | **16** | S (k=17) |
| Llama-3.1-8B (decode) | LLM | 8B | 32 | 500 | 0.075 | **538** | S |
| Llama-3.1-70B (decode) | LLM | 70B | 80 | 1,750 | 0.261 | **540** | S |
| Llama-3.1-405B (decode) | LLM | 405B | 126 | 6,429 | 0.958 | **541** | S |
| Mixtral-8x7B (decode) | LLM MoE | 46.7B | 32 | 2,919 | 0.436 | **540** | S |
| Qwen2.5-72B (decode) | LLM | 72B | 80 | 1,800 | 0.269 | **540** | S |

---

## Three Regimes

### Regime 1: R << 1 (Strategy P optimal)
**Large video DiT, multi-GPU inference.**
HunyuanVideo (0.11), Wan2.2 (0.20), Mochi 1 (0.23).

These models have long per-step compute times (seconds to tens of seconds) because they process high-dimensional video latent spaces with full 3D attention. The H2D transfer window is tiny relative to the compute window. Strategy P (schedule H2D during FFN) makes offloading essentially free.

### Regime 2: R ~ 1-20 (Strategy S with moderate k)
**Image DiT, small video DiT.**
CogVideoX (1.8), SD3 Medium (3.2), FLUX.1-dev (16).

Shorter per-step compute (image or low-res video). Strategy S with $k^* = \lceil R \rceil$ shards is needed, but $k$ is tractable (2-17 shards, all within typical GPU counts).

### Regime 3: R ~ 200-540 (Fundamentally bottlenecked)
**All LLM decode, regardless of model size.**
All LLM models collapse to R $\approx$ 540.

This is a **hardware constant**: $R_{decode} \approx 2 \cdot B_{HBM} / B_p = 2 \times 3350 / 12.4 \approx 540$. The ~270x bandwidth gap between HBM and PCIe makes it impossible to hide H2D during single-token decode compute. Offloading is only viable with extreme sharding or for prefill.

---

## Key Insight

LLM decode R is independent of model size because both W/layer and T_c/layer scale linearly with model size -- they cancel in the ratio. The R $\approx$ 540 constant reflects the fundamental HBM-to-PCIe bandwidth ratio, not model architecture.

For diffusion models, R varies because T_c depends on spatial/temporal dimensions (sequence length for attention, frame count for video) which are independent of layer size.

---

## Sources

- Wan2.2: Project measured data (23s/step @4xH100 PCIe)
- FLUX.1-dev: xDiT docs (0.24s/step @1xH100)
- SD3 Medium: HF blog (0.24s/step @A100, est. 0.20s @H100)
- HunyuanVideo: xDiT docs (38s/step @1xH100, 720p 129f)
- CogVideoX-5B: ~0.9s/step @1xH100
- Mochi 1: ~14s/step @1xH100
- LLM decode: Roofline model validated against NVIDIA NIM benchmarks (Llama-3.1-8B ITL=4.56ms)
- H100 HBM3 BW: 3,350 GB/s (NVIDIA specs)
