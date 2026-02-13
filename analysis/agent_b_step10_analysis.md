# Agent B: Step 10 Contradiction Analysis (Q3)

**Date:** 2026-02-08
**Codebase:** sglang-src (shallow clone, HEAD)
**Focus:** Why does the transformer switch appear to happen at step 10?

## Summary

**PARTIALLY RESOLVED.** The mathematical analysis with current source code shows the boundary switch occurs at **step index 18** (0-indexed) with `flow_shift=12.0` and `boundary_ratio=0.875`. However, benchmark profiler data clearly shows a spike at **step 10** (1-indexed in tqdm, likely step index 9). This discrepancy remains unresolved and requires timestep logging during a live run.

---

## The Switching Mechanism

### Boundary Timestep Calculation

**`denoising.py:493-503`**:
```python
boundary_ratio = server_args.pipeline_config.dit_config.boundary_ratio  # 0.875
if boundary_ratio is not None:
    boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps  # 875.0
```

### Switching Condition

**`denoising.py:870`**:
```python
if boundary_timestep is None or t_int >= boundary_timestep:
    current_model = self.transformer       # High-noise expert
else:
    current_model = self.transformer_2     # Low-noise expert
```

### t_int Source

**`denoising.py:1019`**:
```python
t_int = int(t_host.item())  # Actual timestep value (e.g., 999, 875, etc.)
```

`t_int` is the **timestep value**, NOT the step index.

## Wan2.2-T2V-A14B Configuration

**`configs/pipeline_configs/wan.py:208-213`**:
```python
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875
```

## Flow Matching Scheduler Timestep Generation

**`models/schedulers/scheduling_unipc_multistep.py:461-471`**:
```python
elif self.config.use_flow_sigmas:
    alphas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
    sigmas = 1.0 - alphas
    sigmas = np.flip(
        self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas)
    )[:-1].copy()
    timesteps = (sigmas * self.config.num_train_timesteps).copy()
```

The key transformation: `sigma_shifted = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)`

With `flow_shift=12.0`, this heavily compresses the high-noise end and expands the low-noise end.

## Computed 27-Step Timestep Schedule

Using `flow_shift=12.0`, `num_train_timesteps=1000`, `num_inference_steps=27`:

| Step (0-idx) | tqdm Step | Timestep | >= 875? | Transformer |
|---|---|---|---|---|
| 0 | 1 | 999.9 | YES | HIGH-NOISE |
| 1 | 2 | 996.7 | YES | HIGH-NOISE |
| 2 | 3 | 993.3 | YES | HIGH-NOISE |
| 3 | 4 | 989.6 | YES | HIGH-NOISE |
| 4 | 5 | 985.6 | YES | HIGH-NOISE |
| 5 | 6 | 981.3 | YES | HIGH-NOISE |
| 6 | 7 | 976.6 | YES | HIGH-NOISE |
| 7 | 8 | 971.6 | YES | HIGH-NOISE |
| 8 | 9 | 966.0 | YES | HIGH-NOISE |
| 9 | 10 | 959.9 | YES | HIGH-NOISE |
| 10 | 11 | 953.2 | YES | HIGH-NOISE |
| 11 | 12 | 945.7 | YES | HIGH-NOISE |
| 12 | 13 | 937.4 | YES | HIGH-NOISE |
| 13 | 14 | 928.0 | YES | HIGH-NOISE |
| 14 | 15 | 917.5 | YES | HIGH-NOISE |
| 15 | 16 | 905.5 | YES | HIGH-NOISE |
| 16 | 17 | 891.7 | YES | HIGH-NOISE |
| 17 | 18 | 875.7 | YES | HIGH-NOISE |
| **18** | **19** | **857.0** | **NO** | **LOW-NOISE (switch)** |
| 19 | 20 | 834.6 | NO | LOW-NOISE |
| 20 | 21 | 807.5 | NO | LOW-NOISE |
| 21 | 22 | 774.0 | NO | LOW-NOISE |
| 22 | 23 | 731.5 | NO | LOW-NOISE |
| 23 | 24 | 675.8 | NO | LOW-NOISE |
| 24 | 25 | 599.7 | NO | LOW-NOISE |
| 25 | 26 | 489.5 | NO | LOW-NOISE |
| 26 | 27 | 315.6 | NO | LOW-NOISE |

**Math says: Switch at step index 18 (tqdm step 19).** Step 17 has timestep 875.7 (>= 875, HIGH-NOISE), step 18 has timestep 857.0 (< 875, LOW-NOISE).

## The Contradiction

### Benchmark Data Shows Step 10 Spike

From `4gpu_benchmark_results.md` (first run, jobs 1417509/1417511):
- Step 10: OLD offload = **50s** vs normal ~20s (30s spike)
- All other steps (2-9, 11-27): 19-22s

From `benchmark_summary_1434113_1434114.md` (profiled run, jobs 1434113/1434114):
- Step 10: OLD offload = **~29.8s** vs normal ~21s (~8s spike)

### But Math Says Switch at Step 18-19

The computed schedule places the switch between step 17 and 18 (0-indexed), which would be tqdm step 18 to 19. There is NO anomaly at step 10 in the benchmark data for steps 18-19.

## Possible Explanations

### 1. Codebase Version Mismatch
The benchmarks were run on an earlier version of SGLang (Feb 3-5, 2026). The current HEAD may have different parameters. The `flow_shift` or scheduler logic could have changed.

### 2. HuggingFace Model Config Override
The model `Wan-AI/Wan2.2-T2V-A14B-Diffusers` includes a `scheduler_config.json` that may override `num_train_timesteps`, `flow_shift`, or other scheduler parameters. SGLang loads these from HuggingFace and applies `set_shift()` afterward, but the base config could differ.

### 3. Different Scheduler Branch
The scheduler has multiple timestep generation branches (`use_flow_sigmas`, `use_exponential_sigmas`, `use_beta_sigmas`, or default). If `use_flow_sigmas` is not set, a different formula produces different timesteps:

**`scheduling_unipc_multistep.py:373-396`**: The default branch uses:
```python
timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1).round()[::-1][:-1]
```
With this LINEAR schedule: step 10 = `1000 * (1 - 10/27) = 629.6`, still above 875 at step index 3 only.

### 4. The Spike Is NOT the Transformer Switch
The step 10 spike could be caused by something else entirely:
- torch.compile recompilation for a different code path
- Memory allocation pattern change
- CUDA graph capture boundary
- FSDP reshard/gather at a specific point

### 5. Step Numbering Off-by-One
If tqdm reports steps 1-27 but the profiler stage numbering is different, the spike might align differently.

## Recommended Next Steps

1. **Add timestep logging**: Insert `print(f"Step {i}: t_int={t_int}, transformer={'main' if t_int >= boundary_timestep else 'secondary'}")` at `denoising.py:1019`
2. **Check HuggingFace scheduler config**: Download `scheduler_config.json` from `Wan-AI/Wan2.2-T2V-A14B-Diffusers` to verify `use_flow_sigmas`, `num_train_timesteps`
3. **Run with clean 3-iteration scripts**: Tuesday's benchmark (jobs 1439552-1439555) should include this logging
4. **Verify codebase version**: Check which SGLang commit was used for the original benchmarks

## Resolution Status

| Previous Claim | Verification Result |
|---|---|
| "boundary_ratio=0.875 means step 3-4" | WRONG -- assumed linear timesteps, ignored flow_shift |
| "switch at step 23-24" | WRONG -- confused step ratio with timestep ratio |
| "step 10 spike = transformer switch" | UNVERIFIED -- math says step 18, not step 10 |
| "flow_shift=12.0 creates non-linear schedule" | VERIFIED -- confirmed in code |
| "condition: t_int >= boundary_timestep" | VERIFIED -- denoising.py:870 |

## Code Files Analyzed

1. `pipelines_core/stages/denoising.py` -- `_handle_boundary_ratio` (line 485), `_select_and_manage_model` (line 863), denoising loop (line 1012)
2. `models/schedulers/scheduling_unipc_multistep.py` -- Flow matching timestep generation (line 461), `set_shift` (line 334)
3. `configs/pipeline_configs/wan.py` -- `Wan2_2_T2V_A14B_Config` (line 208): flow_shift=12.0, boundary_ratio=0.875
4. `configs/sample/wan.py` -- `Wan2_2_T2V_A14B_SamplingParam` (line 237)
