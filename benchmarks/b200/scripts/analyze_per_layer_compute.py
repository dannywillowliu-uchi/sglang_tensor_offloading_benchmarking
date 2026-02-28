#!/usr/bin/env python3
"""
Per-layer compute time breakdown analysis.
Identifies 40 transformer block boundaries using kernel timestamps on device 0.
Uses NCCL SendRecv as layer delimiters (Ulysses all-to-all between layers).
"""
import sqlite3
import json
import sys

DB_BASELINE = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp1_no_offload_20260227_233628.sqlite"
DB_OFFLOAD = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp2_offload_default_20260227_234442.sqlite"

def analyze_layers(db_path, label):
	print(f"\n{'='*80}")
	print(f"PER-LAYER COMPUTE ANALYSIS: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# First, find the denoising time range.
	# The denoising loop has 27 steps, each with 40 transformer blocks.
	# Each block ends with a NCCL SendRecv (Ulysses all-to-all).
	# So we expect ~27*40 = 1080 NCCL SendRecv groups on device 0.

	# Get all NCCL SendRecv kernels on device 0, ordered by start time
	cur.execute("""
		SELECT k.start, k.end, k.streamId
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	nccl_kernels = cur.fetchall()
	print(f"\nNCCL SendRecv kernels on device 0: {len(nccl_kernels)}")

	# Group NCCL kernels that overlap or are close together (within 50us)
	# Each all-to-all is multiple SendRecv calls
	nccl_groups = []
	if nccl_kernels:
		group_start = nccl_kernels[0][0]
		group_end = nccl_kernels[0][1]
		group_count = 1
		for start, end, sid in nccl_kernels[1:]:
			if start < group_end + 50000:  # within 50us of group end
				group_end = max(group_end, end)
				group_count += 1
			else:
				nccl_groups.append((group_start, group_end, group_count))
				group_start = start
				group_end = end
				group_count = 1
		nccl_groups.append((group_start, group_end, group_count))

	print(f"NCCL SendRecv groups (clustered): {len(nccl_groups)}")

	# Each denoising step has 40 blocks, each with all-to-all at end.
	# Plus possibly other NCCL at start/end of denoising.
	# Let's look at the distribution of group durations and gaps to find step boundaries.

	if len(nccl_groups) < 100:
		print("  Not enough NCCL groups to analyze layers")
		conn.close()
		return None

	# Print group duration stats
	durations_ms = [(g[1] - g[0]) / 1e6 for g in nccl_groups]
	print(f"  NCCL group duration: min={min(durations_ms):.3f}ms, max={max(durations_ms):.3f}ms, "
		  f"mean={sum(durations_ms)/len(durations_ms):.3f}ms")

	# Look at gaps between groups
	gaps_ms = [(nccl_groups[i+1][0] - nccl_groups[i][1]) / 1e6
			   for i in range(len(nccl_groups)-1)]

	# Large gaps indicate step boundaries
	sorted_gaps = sorted(gaps_ms, reverse=True)
	print(f"  Gap between groups: min={min(gaps_ms):.3f}ms, max={max(gaps_ms):.3f}ms, "
		  f"mean={sum(gaps_ms)/len(gaps_ms):.3f}ms, median={sorted(gaps_ms)[len(gaps_ms)//2]:.3f}ms")
	print(f"  Top 30 gaps: {[f'{g:.1f}ms' for g in sorted_gaps[:30]]}")

	# Identify step boundaries: gaps significantly larger than median
	median_gap = sorted(gaps_ms)[len(gaps_ms) // 2]
	step_boundary_threshold = max(median_gap * 3, 5.0)  # at least 5ms or 3x median
	print(f"  Step boundary threshold: {step_boundary_threshold:.1f}ms")

	step_boundaries = [i for i, g in enumerate(gaps_ms) if g > step_boundary_threshold]
	print(f"  Step boundaries found: {len(step_boundaries)}")

	# Now segment into steps
	# Each step should have ~40 NCCL groups (one per layer)
	step_starts = [0] + [b + 1 for b in step_boundaries]
	steps = []
	for i, start_idx in enumerate(step_starts):
		end_idx = step_boundaries[i] + 1 if i < len(step_boundaries) else len(nccl_groups)
		n_layers = end_idx - start_idx
		step_start = nccl_groups[start_idx][0]
		step_end = nccl_groups[end_idx - 1][1]
		steps.append({
			"step": i,
			"n_layers": n_layers,
			"start_ns": step_start,
			"end_ns": step_end,
			"duration_ms": (step_end - step_start) / 1e6,
			"group_start_idx": start_idx,
			"group_end_idx": end_idx,
		})

	print(f"\nSteps found: {len(steps)}")
	for s in steps:
		print(f"  Step {s['step']:2d}: {s['n_layers']:3d} layers, {s['duration_ms']:.1f}ms")

	# Focus on a representative steady-state step (skip first and last)
	# Pick the middle step
	if len(steps) < 3:
		print("Not enough steps to analyze")
		conn.close()
		return None

	representative_steps = [s for s in steps if 2 <= s["step"] <= len(steps) - 2 and s["n_layers"] >= 35]
	if not representative_steps:
		representative_steps = steps[2:-1] if len(steps) > 4 else steps[1:-1]

	print(f"\nRepresentative steps (steady state): {len(representative_steps)}")

	# For the representative step, compute per-layer timing
	# Layer i compute = time between NCCL group i-1 end and NCCL group i start
	# Layer 0 compute = time from step start to first NCCL group start

	all_layer_durations = []  # list of lists, one per step
	all_nccl_durations = []

	for step in representative_steps:
		si = step["group_start_idx"]
		ei = step["group_end_idx"]
		groups = nccl_groups[si:ei]

		layer_durations = []
		nccl_durations = []

		for j in range(len(groups)):
			# NCCL duration for this layer's all-to-all
			nccl_dur = (groups[j][1] - groups[j][0]) / 1e6
			nccl_durations.append(nccl_dur)

			if j == 0:
				# First layer compute: from step start to first NCCL start
				# We'd need to know when the step actually starts, approximate as
				# the gap before this NCCL group
				continue
			else:
				# Layer j compute: from NCCL group j-1 end to NCCL group j start
				compute_dur = (groups[j][0] - groups[j-1][1]) / 1e6
				layer_durations.append(compute_dur)

		all_layer_durations.append(layer_durations)
		all_nccl_durations.append(nccl_durations)

	# Average across steps
	if all_layer_durations:
		n_layers = min(len(ld) for ld in all_layer_durations)
		avg_layer_dur = []
		for j in range(n_layers):
			vals = [ld[j] for ld in all_layer_durations if j < len(ld)]
			avg_layer_dur.append(sum(vals) / len(vals))

		n_nccl = min(len(nd) for nd in all_nccl_durations)
		avg_nccl_dur = []
		for j in range(n_nccl):
			vals = [nd[j] for nd in all_nccl_durations if j < len(nd)]
			avg_nccl_dur.append(sum(vals) / len(vals))

		print(f"\nPER-LAYER COMPUTE TIME (avg over {len(representative_steps)} steps):")
		print(f"{'Layer':>6s} {'Compute(ms)':>12s} {'NCCL(ms)':>10s} {'Total(ms)':>10s}")
		print("-" * 42)
		for j in range(n_layers):
			nccl_ms = avg_nccl_dur[j+1] if j+1 < len(avg_nccl_dur) else 0
			total = avg_layer_dur[j] + nccl_ms
			print(f"{j+1:6d} {avg_layer_dur[j]:12.3f} {nccl_ms:10.3f} {total:10.3f}")

		total_compute = sum(avg_layer_dur)
		total_nccl = sum(avg_nccl_dur)
		print(f"\n  Total compute (layers 1-{n_layers}): {total_compute:.1f}ms")
		print(f"  Total NCCL: {total_nccl:.1f}ms")
		print(f"  Compute min={min(avg_layer_dur):.3f}ms max={max(avg_layer_dur):.3f}ms "
			  f"mean={total_compute/n_layers:.3f}ms std={((sum((d-total_compute/n_layers)**2 for d in avg_layer_dur)/n_layers)**0.5):.3f}ms")
		print(f"  NCCL min={min(avg_nccl_dur):.3f}ms max={max(avg_nccl_dur):.3f}ms "
			  f"mean={total_nccl/n_nccl:.3f}ms")

		# Check uniformity
		mean_compute = total_compute / n_layers
		cv = ((sum((d - mean_compute)**2 for d in avg_layer_dur) / n_layers) ** 0.5) / mean_compute
		print(f"  Coefficient of variation (compute): {cv:.4f} ({'uniform' if cv < 0.1 else 'non-uniform'})")

		result = {
			"n_steps": len(steps),
			"n_representative_steps": len(representative_steps),
			"n_layers_per_step": n_layers,
			"avg_layer_compute_ms": avg_layer_dur,
			"avg_nccl_ms": avg_nccl_dur,
			"total_compute_ms": total_compute,
			"total_nccl_ms": total_nccl,
			"compute_cv": cv,
			"step_durations_ms": [s["duration_ms"] for s in steps],
		}
	else:
		result = None

	# Also get total kernel time on device 0 for reference
	cur.execute("""
		SELECT SUM(end - start) / 1e9
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0
	""")
	total_kernel_s = cur.fetchone()[0]
	print(f"\n  Total GPU kernel time (device 0, all streams): {total_kernel_s:.2f}s")

	# Compute kernel time on stream 7 (main compute stream)
	cur.execute("""
		SELECT SUM(end - start) / 1e9
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND streamId = 7
	""")
	s7_kernel_s = cur.fetchone()[0]
	print(f"  Stream 7 kernel time: {s7_kernel_s:.2f}s")

	# Compute kernel time on stream 47
	cur.execute("""
		SELECT SUM(end - start) / 1e9
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND streamId = 47
	""")
	s47_kernel_s = cur.fetchone()[0]
	print(f"  Stream 47 kernel time: {s47_kernel_s:.2f}s")

	conn.close()
	return result

baseline_result = analyze_layers(DB_BASELINE, "Baseline (no offload)")
offload_result = analyze_layers(DB_OFFLOAD, "Offload default")

# Comparison
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
if baseline_result and offload_result:
	print(f"\n{'Metric':>35s} {'Baseline':>12s} {'Offload':>12s} {'Delta':>10s}")
	print("-" * 72)
	b_compute = baseline_result["total_compute_ms"]
	o_compute = offload_result["total_compute_ms"]
	print(f"{'Total layer compute (ms)':>35s} {b_compute:12.1f} {o_compute:12.1f} {(o_compute-b_compute)/b_compute*100:+9.1f}%")

	b_nccl = baseline_result["total_nccl_ms"]
	o_nccl = offload_result["total_nccl_ms"]
	print(f"{'Total NCCL (ms)':>35s} {b_nccl:12.1f} {o_nccl:12.1f} {(o_nccl-b_nccl)/b_nccl*100:+9.1f}%")

	b_cv = baseline_result["compute_cv"]
	o_cv = offload_result["compute_cv"]
	print(f"{'Compute CV':>35s} {b_cv:12.4f} {o_cv:12.4f}")

	# Per-step comparison
	print(f"\nPer-step durations (ms):")
	b_steps = baseline_result["step_durations_ms"]
	o_steps = offload_result["step_durations_ms"]
	max_steps = max(len(b_steps), len(o_steps))
	for i in range(max_steps):
		b_v = b_steps[i] if i < len(b_steps) else None
		o_v = o_steps[i] if i < len(o_steps) else None
		if b_v and o_v:
			print(f"  Step {i:2d}: baseline={b_v:.1f}ms, offload={o_v:.1f}ms, delta={o_v-b_v:+.1f}ms ({(o_v-b_v)/b_v*100:+.1f}%)")
		elif b_v:
			print(f"  Step {i:2d}: baseline={b_v:.1f}ms")
		elif o_v:
			print(f"  Step {i:2d}: offload={o_v:.1f}ms")
