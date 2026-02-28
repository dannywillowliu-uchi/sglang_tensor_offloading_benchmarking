#!/usr/bin/env python3
"""
Per-layer compute time analysis v2.
Uses NCCL SendRecv bursts as layer delimiters. Each transformer block has 2 all-to-all
operations (before and after attention in Ulysses SP), so every 2 NCCL bursts = 1 layer boundary.
"""
import sqlite3
import json
import statistics

DB_BASELINE = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp1_no_offload_20260227_233628.sqlite"
DB_OFFLOAD = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp2_offload_default_20260227_234442.sqlite"

def get_nccl_bursts(cur, gap_threshold_us=500):
	"""Get NCCL SendRecv bursts on device 0, stream 47."""
	cur.execute("""
		SELECT k.start, k.end
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0 AND k.streamId = 47
		ORDER BY k.start
	""")
	kernels = cur.fetchall()
	if not kernels:
		return []

	bursts = []
	burst_start = kernels[0][0]
	burst_end = kernels[0][1]
	burst_count = 1
	for start, end in kernels[1:]:
		if start < burst_end + gap_threshold_us * 1000:
			burst_end = max(burst_end, end)
			burst_count += 1
		else:
			bursts.append((burst_start, burst_end, burst_count))
			burst_start = start
			burst_end = end
			burst_count = 1
	bursts.append((burst_start, burst_end, burst_count))
	return bursts


def identify_denoising_steps(bursts):
	"""
	Each denoising step has 40 layers. Each layer has 2 NCCL burst groups
	(all-to-all before and after attention). So 80 bursts per step.
	Between steps there's a larger gap (scheduler step, etc).
	"""
	# Look at inter-burst gaps
	gaps = [(bursts[i+1][0] - bursts[i][1], i) for i in range(len(bursts)-1)]

	# Sort by gap size to find step boundaries
	sorted_gaps = sorted(gaps, key=lambda x: -x[0])

	# Find gaps that are significantly larger than the per-layer gap
	# Per-layer gaps should be ~1-10ms (compute time), step boundaries ~20-100ms+
	gap_values = [g[0] for g in gaps]
	median_gap = statistics.median(gap_values)

	# Step boundary = gap > 5x median
	step_threshold = max(median_gap * 5, 10e6)  # at least 10ms
	step_boundaries = sorted([g[1] for g in gaps if g[0] > step_threshold])

	return step_boundaries, step_threshold


def analyze_per_layer(db_path, label):
	print(f"\n{'='*80}")
	print(f"PER-LAYER COMPUTE: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	bursts = get_nccl_bursts(cur, gap_threshold_us=500)
	print(f"Total NCCL bursts: {len(bursts)}")

	# Burst duration stats
	burst_durs = [(b[1] - b[0]) / 1e6 for b in bursts]
	print(f"Burst duration: min={min(burst_durs):.3f}ms max={max(burst_durs):.3f}ms "
		  f"mean={statistics.mean(burst_durs):.3f}ms median={statistics.median(burst_durs):.3f}ms")

	step_boundaries, threshold = identify_denoising_steps(bursts)
	print(f"Step boundary threshold: {threshold/1e6:.1f}ms")
	print(f"Step boundaries found: {len(step_boundaries)}")

	# Build steps
	step_starts = [0] + [b + 1 for b in step_boundaries]
	steps = []
	for i, si in enumerate(step_starts):
		ei = step_boundaries[i] + 1 if i < len(step_boundaries) else len(bursts)
		n_bursts = ei - si
		step_start = bursts[si][0]
		step_end = bursts[ei - 1][1]
		steps.append({
			"step": i,
			"n_bursts": n_bursts,
			"burst_start_idx": si,
			"burst_end_idx": ei,
			"start_ns": step_start,
			"end_ns": step_end,
			"duration_ms": (step_end - step_start) / 1e6,
		})

	print(f"\nSteps: {len(steps)}")
	for s in steps:
		print(f"  Step {s['step']:2d}: {s['n_bursts']:3d} bursts, {s['duration_ms']:.1f}ms")

	# Each layer has 2 NCCL bursts (2 all-to-all per Ulysses block).
	# Between burst pairs, there's attention compute. Between layers, there's FFN compute.
	# Pattern: [pre-attn all2all] [attention] [post-attn all2all] [FFN] [pre-attn all2all] ...

	# Let's analyze a steady-state step
	# Pick steps with ~80 bursts (40 layers * 2 all-to-all)
	steady_steps = [s for s in steps if 70 <= s["n_bursts"] <= 90]
	if not steady_steps:
		# Relax constraint
		steady_steps = [s for s in steps if s["n_bursts"] >= 40]
	print(f"\nSteady-state steps (70-90 bursts): {len(steady_steps)}")
	if not steady_steps:
		print("No suitable steps found!")
		conn.close()
		return None

	# For each steady step, extract per-layer timing
	all_layer_data = []

	for step in steady_steps[1:-1]:  # skip first and last
		si = step["burst_start_idx"]
		ei = step["burst_end_idx"]
		step_bursts = bursts[si:ei]

		# Pair up bursts: (burst 0, burst 1) = layer 0, (burst 2, burst 3) = layer 1, ...
		n_layers = len(step_bursts) // 2
		layer_data = []
		for j in range(n_layers):
			b1 = step_bursts[2 * j]       # pre-attention all-to-all
			b2 = step_bursts[2 * j + 1]   # post-attention all-to-all

			# NCCL durations
			nccl1_ms = (b1[1] - b1[0]) / 1e6
			nccl2_ms = (b2[1] - b2[0]) / 1e6

			# Attention compute = gap between burst 1 end and burst 2 start
			attn_compute_ms = (b2[0] - b1[1]) / 1e6

			# FFN compute = gap from this layer's post-attn to next layer's pre-attn
			if j + 1 < n_layers:
				b_next = step_bursts[2 * (j + 1)]
				ffn_compute_ms = (b_next[0] - b2[1]) / 1e6
			else:
				ffn_compute_ms = 0  # last layer

			layer_data.append({
				"layer": j,
				"nccl1_ms": nccl1_ms,
				"attn_compute_ms": attn_compute_ms,
				"nccl2_ms": nccl2_ms,
				"ffn_compute_ms": ffn_compute_ms,
				"total_ms": nccl1_ms + attn_compute_ms + nccl2_ms + ffn_compute_ms,
			})

		all_layer_data.append(layer_data)

	if not all_layer_data:
		print("No layer data extracted!")
		conn.close()
		return None

	# Average across steps
	n_layers = min(len(ld) for ld in all_layer_data)
	avg_layers = []
	for j in range(n_layers):
		vals = [ld[j] for ld in all_layer_data if j < len(ld)]
		avg = {
			"layer": j,
			"nccl1_ms": statistics.mean([v["nccl1_ms"] for v in vals]),
			"attn_compute_ms": statistics.mean([v["attn_compute_ms"] for v in vals]),
			"nccl2_ms": statistics.mean([v["nccl2_ms"] for v in vals]),
			"ffn_compute_ms": statistics.mean([v["ffn_compute_ms"] for v in vals]),
			"total_ms": statistics.mean([v["total_ms"] for v in vals]),
		}
		avg_layers.append(avg)

	print(f"\nPER-LAYER BREAKDOWN (avg over {len(all_layer_data)} steps, {n_layers} layers):")
	print(f"{'Layer':>6s} {'NCCL1(ms)':>10s} {'Attn(ms)':>10s} {'NCCL2(ms)':>10s} {'FFN(ms)':>10s} {'Total(ms)':>10s}")
	print("-" * 60)
	for l in avg_layers:
		print(f"{l['layer']:6d} {l['nccl1_ms']:10.3f} {l['attn_compute_ms']:10.3f} "
			  f"{l['nccl2_ms']:10.3f} {l['ffn_compute_ms']:10.3f} {l['total_ms']:10.3f}")

	totals = {
		"nccl1": sum(l["nccl1_ms"] for l in avg_layers),
		"attn": sum(l["attn_compute_ms"] for l in avg_layers),
		"nccl2": sum(l["nccl2_ms"] for l in avg_layers),
		"ffn": sum(l["ffn_compute_ms"] for l in avg_layers),
		"total": sum(l["total_ms"] for l in avg_layers),
	}
	print(f"\n{'TOTAL':>6s} {totals['nccl1']:10.3f} {totals['attn']:10.3f} "
		  f"{totals['nccl2']:10.3f} {totals['ffn']:10.3f} {totals['total']:10.3f}")

	# Uniformity check
	attn_times = [l["attn_compute_ms"] for l in avg_layers]
	ffn_times = [l["ffn_compute_ms"] for l in avg_layers[:-1]]  # exclude last
	nccl_times = [l["nccl1_ms"] + l["nccl2_ms"] for l in avg_layers]

	print(f"\nUniformity:")
	print(f"  Attention compute: mean={statistics.mean(attn_times):.3f}ms "
		  f"std={statistics.stdev(attn_times):.3f}ms CV={statistics.stdev(attn_times)/statistics.mean(attn_times):.4f}")
	if ffn_times:
		print(f"  FFN compute: mean={statistics.mean(ffn_times):.3f}ms "
			  f"std={statistics.stdev(ffn_times):.3f}ms CV={statistics.stdev(ffn_times)/statistics.mean(ffn_times):.4f}")
	print(f"  NCCL per layer: mean={statistics.mean(nccl_times):.3f}ms "
		  f"std={statistics.stdev(nccl_times):.3f}ms CV={statistics.stdev(nccl_times)/statistics.mean(nccl_times):.4f}")

	# Per-step total duration comparison
	step_durs = [s["duration_ms"] for s in steady_steps]
	print(f"\nPer-step duration: mean={statistics.mean(step_durs):.1f}ms "
		  f"std={statistics.stdev(step_durs):.1f}ms "
		  f"min={min(step_durs):.1f}ms max={max(step_durs):.1f}ms")

	conn.close()
	return {
		"n_steps": len(steps),
		"n_layers": n_layers,
		"avg_layers": avg_layers,
		"totals": totals,
		"step_durations_ms": [s["duration_ms"] for s in steps],
		"step_n_bursts": [s["n_bursts"] for s in steps],
	}

baseline = analyze_per_layer(DB_BASELINE, "Baseline (no offload)")
offload = analyze_per_layer(DB_OFFLOAD, "Offload default")

# Comparison
if baseline and offload:
	print(f"\n{'='*80}")
	print("COMPARISON: Per-Layer")
	print(f"{'='*80}")

	n = min(baseline["n_layers"], offload["n_layers"])
	print(f"\n{'Layer':>6s} {'Base Attn':>10s} {'Off Attn':>10s} {'Delta':>8s} "
		  f"{'Base FFN':>10s} {'Off FFN':>10s} {'Delta':>8s} "
		  f"{'Base NCCL':>10s} {'Off NCCL':>10s} {'Delta':>8s}")
	print("-" * 100)
	for j in range(n):
		bl = baseline["avg_layers"][j]
		ol = offload["avg_layers"][j]
		ba = bl["attn_compute_ms"]
		oa = ol["attn_compute_ms"]
		bf = bl["ffn_compute_ms"]
		of_ = ol["ffn_compute_ms"]
		bn = bl["nccl1_ms"] + bl["nccl2_ms"]
		on = ol["nccl1_ms"] + ol["nccl2_ms"]
		print(f"{j:6d} {ba:10.3f} {oa:10.3f} {oa-ba:+7.3f}ms "
			  f"{bf:10.3f} {of_:10.3f} {of_-bf:+7.3f}ms "
			  f"{bn:10.3f} {on:10.3f} {on-bn:+7.3f}ms")

	# Overall
	print(f"\nOverall per-step comparison:")
	bt = baseline["totals"]
	ot = offload["totals"]
	print(f"  Attention: {bt['attn']:.1f}ms -> {ot['attn']:.1f}ms ({(ot['attn']-bt['attn'])/bt['attn']*100:+.1f}%)")
	print(f"  FFN:       {bt['ffn']:.1f}ms -> {ot['ffn']:.1f}ms ({(ot['ffn']-bt['ffn'])/bt['ffn']*100:+.1f}%)")
	print(f"  NCCL:      {bt['nccl1']+bt['nccl2']:.1f}ms -> {ot['nccl1']+ot['nccl2']:.1f}ms "
		  f"({((ot['nccl1']+ot['nccl2'])-(bt['nccl1']+bt['nccl2']))/(bt['nccl1']+bt['nccl2'])*100:+.1f}%)")
	print(f"  Total:     {bt['total']:.1f}ms -> {ot['total']:.1f}ms ({(ot['total']-bt['total'])/bt['total']*100:+.1f}%)")
