# My Notes: Understanding What I'm Reporting

This is a personal reference to understand the research before presenting it.

---

## The Big Picture

We're studying a feature someone added to SGLang (PR #15511) that changes how video generation models handle GPU memory. The Wan2.2 model has two huge neural networks (14B parameters each), and only one is active at a time. The question is: what's the best way to shuffle them between CPU and GPU?

---

## The Two Approaches (What They Actually Do)

### Old Way: Full-Model Swap

Think of it like moving an entire house. When it's time to switch transformers:
1. Move the active one from GPU to CPU (blocking -- nothing else happens while this runs)
2. Move the other one from CPU to GPU (also blocking)

This creates a big delay at the switch point (~8 seconds), but the rest of the time it runs fast because everything is already on the GPU.

### New Way: Layer-by-Layer Streaming (PR #15511)

Think of it like streaming a movie. Instead of downloading the whole movie first:
1. Keep all weights on CPU as the "master copy"
2. Before each layer runs, copy just that layer to GPU in the background
3. After each layer runs, throw away the GPU copy (CPU still has it)

No big delay at the switch point, but every single step is a little slower because you're constantly streaming data.

---

## Why the New Way is Slower on Our Hardware

The new approach assumes you can transfer data in the background while the GPU does other work. This ONLY works if the background transfer uses a different physical wire than the computation.

- **NVLink systems** (what the PR authors tested on): GPU-to-GPU talk goes over NVLink, CPU-to-GPU goes over PCIe. Two separate wires = overlap works.
- **Our ACES cluster** (H100 PCIe): EVERYTHING goes over PCIe. One wire = they fight over bandwidth = no real overlap.

So on our hardware, every step pays an extra ~2 seconds for the streaming that can't be hidden. Over 27 steps, that's ~52 seconds of overhead. The old approach only pays ~8 seconds total (one spike). That's why old is 7% faster for us.

BUT: the new approach uses way less GPU memory (23 GB vs 61 GB). If you don't have enough VRAM for the old approach, the new one is your only option.

---

## The Config Bug (What Went Wrong With Our Benchmarks)

### What is flow_shift?

The model uses 27 denoising steps. `flow_shift` controls how those steps are distributed across the noise schedule. A higher flow_shift means more steps are spent on the noisy end (where the model does the heavy lifting).

- **flow_shift=3.0**: More evenly distributed. Transformer switch at step 10.
- **flow_shift=12.0**: Heavy front-loading. Transformer switch at step 19.

For Wan2.2-A14B, the correct value is 12.0.

### What happened

SGLang picks model settings based on the model path you give it. There's a lookup table:

```
"Wan-AI/Wan2.2-T2V-A14B-Diffusers"  -->  Wan2_2_T2V_A14B_Config (flow_shift=12.0)
```

But we gave it `./models/wan2.2` (a local folder). SGLang tried to match this:
1. Exact match against "Wan-AI/Wan2.2-T2V-A14B-Diffusers"? No.
2. Partial match "wan2.2" vs "wan2.2-t2v-a14b-diffusers"? No.
3. Fallback: read the model files, detect it's a "WanPipeline", match to the generic Wan config.

The generic config has `flow_shift=3.0` (meant for the smaller 1.3B model). So our benchmarks were running with the wrong parameter.

### Why it matters for our numbers

With the wrong flow_shift, the transformer switch happened at step 10 instead of step 19. This changes the performance comparison between old and new offload because:
- With switch at step 10: old offload pays its spike early, then runs smooth for 17 more steps
- With switch at step 19: old offload runs smooth for longer, but new offload accumulates more per-step overhead before the old path pays its spike

The 7% number might change. We don't know which direction until we re-run.

### The fix

Change `--model-path ./models/wan2.2` to `--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers` in all scripts. Done.

---

## What I'm Telling the Team

1. **Here's how both offload strategies work** (Section 2 of the shareable report) -- this is solid, verified from source code

2. **The new approach can't overlap transfers on our hardware** (Section 3) -- because we have PCIe not NVLink. This explains why the PR's claims don't match our results.

3. **Preliminary: old offload is ~7% faster on 4 GPUs but uses 3x more memory** (Section 4) -- caveat that we need to re-run

4. **We found a SGLang bug** (Section 5) -- local model paths resolve to wrong config. Already fixed in our scripts. Could potentially report upstream.

5. **Re-running Tuesday** (Section 6) -- corrected benchmarks queued, plus 6-GPU comparison to see if more GPUs help the new approach

---

## Questions I Should Be Ready For

**"Why didn't you catch this sooner?"**
The flow_shift parameter is buried several layers deep in the config resolution code. We only found it by doing a full source code trace to understand why the math didn't match the profiler data. This kind of silent misconfiguration is exactly the kind of thing benchmarking studies uncover.

**"Does this invalidate all our results?"**
The mechanism analysis (how offloading works, PCIe contention, etc.) is all still valid. Only the timing numbers need to be re-run. The architectural conclusions about why layerwise offload underperforms on PCIe are unchanged.

**"Should we report this bug to SGLang?"**
Yes, probably. It affects anyone using a local model directory for Wan2.2-A14B. The fix would be adding a model detector to `registry.py` for the A14B config.

**"What do you expect with the correct config?"**
Hard to predict exactly. With the switch at step 19 instead of 10:
- Old offload gets 9 more "cheap" steps before paying the spike
- New offload accumulates 9 more steps of overhead before the comparison resets
- The gap could widen, narrow, or stay similar. Need the data.
