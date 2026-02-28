#!/usr/bin/env python3
"""
Comprehensive nsys analysis v3.
Correct understanding: 4 bursts per layer (4 GPU: all-to-all before attn, all-to-all after attn = 2 bursts,
but each burst is multiple NCCL kernels). With 40 layers and 27 steps.

Approach: Group NCCL bursts into "layers" (each layer = 4 consecutive bursts),
then group layers into denoising steps.
"""
import sqlite3
import statistics
import json
import os
import sys

DB_BASELINE = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp1_no_offload_20260227_233628.sqlite"
DB_OFFLOAD = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp2_offload_default_20260227_234442.sqlite"
OUTPUT_DIR = "/Users/dannyliu/research_work/b200_workspace/results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}

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


# ===========================================================================
# ANALYSIS 1: Per-layer compute time
# ===========================================================================
def analyze_per_layer(db_path, label):
	print(f"\n{'='*80}")
	print(f"1. PER-LAYER COMPUTE: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	bursts = get_nccl_bursts(cur)
	print(f"Total NCCL bursts: {len(bursts)}")

	# Each layer produces 4 bursts: pre-attn all2all (2 bursts?) or rather
	# let's look at the actual structure more carefully.
	# From the data: ~2160 "layers" each with 4 bursts.
	# 2160 / 40 layers / 27 steps = 2.0 -- so each "layer" in my previous
	# analysis was actually half a layer.
	# Wait: 2160 / 27 = 80. 80 / 40 = 2 bursts per layer per denoising step.
	# So 2 bursts per layer (one all-to-all before attention, one after).

	# Let me re-examine: gaps between bursts.
	# With 2 all-to-all per layer: intra-layer gap (attention compute ~7-8ms)
	# and inter-layer gap (FFN + misc compute ~8-9ms)
	# With 40 layers per step: 80 bursts per step, separated by step boundary

	gaps_ms = [(bursts[i+1][0] - bursts[i][1]) / 1e6 for i in range(len(bursts)-1)]

	# With the threshold creating 2160 "steps" each with 4 bursts:
	# The issue is the gap between attention compute (~7ms) and inter-layer gap
	# are similar, making it hard to distinguish with a single threshold.

	# Instead, let's use a hierarchical approach:
	# 1. Find denoising step boundaries (very large gaps, ~100ms+)
	# 2. Within each step, pair bursts as layer boundaries

	# Find the biggest gaps to identify the 27 denoising step boundaries
	# (or 26 gaps between 27 steps)
	gap_data = [(g, i) for i, g in enumerate(gaps_ms)]
	gap_data.sort(key=lambda x: -x[0])

	# The top ~30 gaps should include the 26 step boundaries
	# plus maybe some outliers from initialization
	# Let's see the distribution
	print(f"\nGap distribution:")
	print(f"  Top 35 gaps: {[f'{g[0]:.1f}ms' for g in gap_data[:35]]}")

	# Find a clear separation between inter-layer and inter-step gaps
	# Inter-layer gaps should be ~7-16ms, inter-step gaps should be much larger
	# Look for a gap in the gap distribution itself
	top_gap_values = sorted([g[0] for g in gap_data[:100]], reverse=True)

	# For baseline: expect 26 step boundaries with gaps of ~50-200ms
	# and ~2 init gaps with very large gaps (>500ms)
	# All other gaps are intra-step (~7-16ms)

	# Use a threshold that separates step boundaries from layer boundaries
	# The typical approach: find the gap between the Nth largest and (N+1)th largest
	# where N is approximately 26-30
	n_denoising_steps = 27

	# Try: median of top 35 gaps should be a step boundary
	# Then threshold = mean of 30th and 31st largest gaps
	if len(gap_data) >= 40:
		boundary_gap = gap_data[n_denoising_steps + 5][0]  # 32nd largest gap
		intra_gap = gap_data[n_denoising_steps + 10][0]  # 37th largest
		threshold = (boundary_gap + intra_gap) / 2
		print(f"  Step boundary detection: gap[{n_denoising_steps+5}]={boundary_gap:.1f}ms, "
			  f"gap[{n_denoising_steps+10}]={intra_gap:.1f}ms, threshold={threshold:.1f}ms")
	else:
		threshold = 50.0

	step_boundary_indices = sorted([g[1] for g in gap_data if g[0] > threshold])
	print(f"  Step boundaries: {len(step_boundary_indices)}")

	# Build denoising steps
	step_starts = [0] + [b + 1 for b in step_boundary_indices]
	denoising_steps = []
	for i, si in enumerate(step_starts):
		ei = step_boundary_indices[i] + 1 if i < len(step_boundary_indices) else len(bursts)
		n = ei - si
		if n < 4:
			continue  # skip init fragments
		step_start_ns = bursts[si][0]
		step_end_ns = bursts[ei - 1][1]
		denoising_steps.append({
			"step": len(denoising_steps),
			"n_bursts": n,
			"burst_start": si,
			"burst_end": ei,
			"start_ns": step_start_ns,
			"end_ns": step_end_ns,
			"duration_ms": (step_end_ns - step_start_ns) / 1e6,
		})

	print(f"\nDenoising steps found: {len(denoising_steps)}")
	for s in denoising_steps:
		print(f"  Step {s['step']:2d}: {s['n_bursts']:3d} bursts, {s['duration_ms']:.1f}ms")

	# Within each step, pair bursts into layers
	# Each layer should have 2 bursts: pre-attn all-to-all and post-attn all-to-all
	steady_steps = denoising_steps[2:-1] if len(denoising_steps) > 4 else denoising_steps[1:-1]

	all_layer_data = []
	for step in steady_steps:
		si = step["burst_start"]
		ei = step["burst_end"]
		step_bursts = bursts[si:ei]

		# Within a step, alternate between intra-layer gaps (attention compute)
		# and inter-layer gaps (FFN compute)
		# With 80 bursts: 40 layers * 2 bursts
		# Gap pattern: attn, ffn, attn, ffn, ... (alternating)

		n_layers = len(step_bursts) // 2
		layers = []
		for j in range(n_layers):
			b1 = step_bursts[2 * j]       # pre-attention all-to-all
			b2 = step_bursts[2 * j + 1]   # post-attention all-to-all

			nccl1_ms = (b1[1] - b1[0]) / 1e6
			nccl2_ms = (b2[1] - b2[0]) / 1e6
			attn_compute_ms = (b2[0] - b1[1]) / 1e6  # attention between the two all-to-alls

			# FFN = gap from post-attn to next pre-attn
			if j + 1 < n_layers:
				b_next = step_bursts[2 * (j + 1)]
				ffn_ms = (b_next[0] - b2[1]) / 1e6
			else:
				ffn_ms = 0

			layers.append({
				"layer": j,
				"nccl1_ms": nccl1_ms,
				"attn_ms": attn_compute_ms,
				"nccl2_ms": nccl2_ms,
				"ffn_ms": ffn_ms,
				"total_ms": nccl1_ms + attn_compute_ms + nccl2_ms + ffn_ms,
			})
		all_layer_data.append(layers)

	if not all_layer_data:
		print("No layer data!")
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
			"attn_ms": statistics.mean([v["attn_ms"] for v in vals]),
			"nccl2_ms": statistics.mean([v["nccl2_ms"] for v in vals]),
			"ffn_ms": statistics.mean([v["ffn_ms"] for v in vals]),
			"total_ms": statistics.mean([v["total_ms"] for v in vals]),
		}
		# Also get stdev
		avg["attn_std"] = statistics.stdev([v["attn_ms"] for v in vals]) if len(vals) > 1 else 0
		avg["ffn_std"] = statistics.stdev([v["ffn_ms"] for v in vals]) if len(vals) > 1 else 0
		avg["nccl_std"] = statistics.stdev([v["nccl1_ms"] + v["nccl2_ms"] for v in vals]) if len(vals) > 1 else 0
		avg_layers.append(avg)

	print(f"\nPER-LAYER BREAKDOWN (avg over {len(all_layer_data)} steps, {n_layers} layers):")
	print(f"{'Layer':>6s} {'NCCL1':>8s} {'Attn':>8s} {'NCCL2':>8s} {'FFN':>8s} {'Total':>8s}")
	print("-" * 50)
	for l in avg_layers:
		print(f"{l['layer']:6d} {l['nccl1_ms']:8.3f} {l['attn_ms']:8.3f} "
			  f"{l['nccl2_ms']:8.3f} {l['ffn_ms']:8.3f} {l['total_ms']:8.3f}")

	totals = {k: sum(l[k] for l in avg_layers) for k in ["nccl1_ms", "attn_ms", "nccl2_ms", "ffn_ms", "total_ms"]}
	print(f"\n{'TOTAL':>6s} {totals['nccl1_ms']:8.1f} {totals['attn_ms']:8.1f} "
		  f"{totals['nccl2_ms']:8.1f} {totals['ffn_ms']:8.1f} {totals['total_ms']:8.1f}")

	# Uniformity
	attn_vals = [l["attn_ms"] for l in avg_layers]
	ffn_vals = [l["ffn_ms"] for l in avg_layers[:-1]]
	nccl_vals = [l["nccl1_ms"] + l["nccl2_ms"] for l in avg_layers]

	print(f"\nUniformity across layers:")
	print(f"  Attention: {statistics.mean(attn_vals):.3f} +/- {statistics.stdev(attn_vals):.3f}ms "
		  f"(CV={statistics.stdev(attn_vals)/statistics.mean(attn_vals):.4f})")
	if ffn_vals:
		print(f"  FFN:       {statistics.mean(ffn_vals):.3f} +/- {statistics.stdev(ffn_vals):.3f}ms "
			  f"(CV={statistics.stdev(ffn_vals)/statistics.mean(ffn_vals):.4f})")
	print(f"  NCCL:      {statistics.mean(nccl_vals):.3f} +/- {statistics.stdev(nccl_vals):.3f}ms "
		  f"(CV={statistics.stdev(nccl_vals)/statistics.mean(nccl_vals):.4f})")

	# Step durations
	step_durs = [s["duration_ms"] for s in denoising_steps if s["n_bursts"] > 40]
	if step_durs:
		print(f"\nDenoising step duration: {statistics.mean(step_durs):.1f} +/- {statistics.stdev(step_durs):.1f}ms")

	conn.close()
	return {
		"avg_layers": avg_layers,
		"totals": totals,
		"step_durations_ms": [s["duration_ms"] for s in denoising_steps],
		"n_steps": len(denoising_steps),
		"n_layers": n_layers,
	}


# ===========================================================================
# ANALYSIS 2: H2D transfer timing and prefetch margin (offload only)
# ===========================================================================
def analyze_h2d_prefetch(db_path, label):
	print(f"\n{'='*80}")
	print(f"2. H2D TRANSFER & PREFETCH MARGIN: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# H2D on copy streams (91 and 87 for offload)
	cur.execute("""
		SELECT streamId, COUNT(*), SUM(bytes)/1e9 as gb, SUM(end-start)/1e9 as total_s,
			   AVG(bytes)/1e6 as avg_mb, AVG(end-start)/1e6 as avg_ms,
			   AVG(bytes*1.0/(end-start)) as avg_bw_gbs
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1  -- H2D
		AND bytes > 1000000  -- >1MB (significant transfers)
		GROUP BY streamId
		HAVING SUM(bytes) > 1e9
		ORDER BY gb DESC
	""")
	print("\nH2D streams with >1GB total (device 0):")
	h2d_streams = []
	for r in cur.fetchall():
		print(f"  stream={r[0]:3d}: {r[1]} ops, {r[2]:.1f}GB, {r[3]:.3f}s total, "
			  f"avg={r[4]:.1f}MB, {r[5]:.3f}ms, BW={r[6]:.1f}GB/s")
		h2d_streams.append(r[0])

	if not h2d_streams:
		print("  No significant H2D streams (this is the baseline, no offload)")
		conn.close()
		return None

	# Get all significant H2D transfers on these streams
	stream_list = ",".join(str(s) for s in h2d_streams)
	cur.execute(f"""
		SELECT start, end, bytes, streamId
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1
		AND bytes > 1000000
		AND streamId IN ({stream_list})
		ORDER BY start
	""")
	h2d_transfers = cur.fetchall()
	print(f"\nTotal significant H2D transfers: {len(h2d_transfers)}")

	# Group H2D transfers into per-layer groups
	# Each layer prefetch should be a cluster of H2D transfers
	h2d_groups = []
	if h2d_transfers:
		group = [h2d_transfers[0]]
		for t in h2d_transfers[1:]:
			if t[0] < group[-1][1] + 1000000:  # within 1ms
				group.append(t)
			else:
				g_start = group[0][0]
				g_end = max(x[1] for x in group)
				g_bytes = sum(x[2] for x in group)
				h2d_groups.append((g_start, g_end, g_bytes, len(group)))
				group = [t]
		g_start = group[0][0]
		g_end = max(x[1] for x in group)
		g_bytes = sum(x[2] for x in group)
		h2d_groups.append((g_start, g_end, g_bytes, len(group)))

	print(f"H2D groups: {len(h2d_groups)}")
	if h2d_groups:
		group_durs = [(g[1]-g[0])/1e6 for g in h2d_groups]
		group_bytes = [g[2]/1e6 for g in h2d_groups]
		print(f"  Duration: min={min(group_durs):.3f}ms max={max(group_durs):.3f}ms "
			  f"mean={statistics.mean(group_durs):.3f}ms")
		print(f"  Size: min={min(group_bytes):.1f}MB max={max(group_bytes):.1f}MB "
			  f"mean={statistics.mean(group_bytes):.1f}MB")

	# Now correlate with NCCL bursts (layer boundaries)
	bursts = get_nccl_bursts(cur)

	# Find gaps between NCCL bursts and nearest H2D group
	# For each layer boundary (NCCL burst pair), find the H2D group that
	# completes just before/during the layer's compute
	margins = []
	overlaps = 0
	for i in range(0, len(bursts) - 1, 2):
		layer_start = bursts[i][0]  # start of pre-attn all-to-all
		# Find H2D group that ends closest to (but before) layer_start
		best_margin = None
		for g in h2d_groups:
			h2d_end = g[1]
			if h2d_end <= layer_start:
				margin_ms = (layer_start - h2d_end) / 1e6
				if best_margin is None or margin_ms < best_margin:
					best_margin = margin_ms
			elif g[0] < layer_start:
				# H2D overlaps with layer start
				overlaps += 1
				best_margin = -((layer_start - g[1]) / 1e6)  # negative = overlap
				break
		if best_margin is not None:
			margins.append(best_margin)

	if margins:
		positive_margins = [m for m in margins if m >= 0]
		negative_margins = [m for m in margins if m < 0]
		print(f"\nPrefetch margin (H2D end to layer start):")
		print(f"  Total layers analyzed: {len(margins)}")
		print(f"  Prefetch completed before layer: {len(positive_margins)} ({len(positive_margins)/len(margins)*100:.1f}%)")
		print(f"  Prefetch overlapped with layer: {len(negative_margins)} ({len(negative_margins)/len(margins)*100:.1f}%)")
		if positive_margins:
			print(f"  Positive margins: min={min(positive_margins):.3f}ms max={max(positive_margins):.3f}ms "
				  f"mean={statistics.mean(positive_margins):.3f}ms")
		if negative_margins:
			print(f"  Negative margins (overlap): min={min(negative_margins):.3f}ms max={max(negative_margins):.3f}ms "
				  f"mean={statistics.mean(negative_margins):.3f}ms")

	# D2H analysis (offload back to host)
	cur.execute("""
		SELECT COUNT(*), SUM(bytes)/1e9 as gb, SUM(end-start)/1e9 as total_s,
			   AVG(bytes)/1e6 as avg_mb, AVG(end-start)/1e6 as avg_ms,
			   AVG(bytes*1.0/(end-start)) as avg_bw
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 2  -- D2H
		AND bytes > 1000000
	""")
	r = cur.fetchone()
	print(f"\nD2H transfers (>1MB, device 0): {r[0]} ops, {r[1]:.1f}GB, {r[2]:.3f}s total, "
		  f"avg={r[3]:.1f}MB, {r[4]:.3f}ms, BW={r[5]:.1f}GB/s")

	# Total H2D volume and bandwidth
	cur.execute("""
		SELECT SUM(bytes)/1e9, SUM(end-start)/1e9
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1
	""")
	h2d_gb, h2d_s = cur.fetchone()
	print(f"\nTotal H2D (device 0): {h2d_gb:.1f}GB in {h2d_s:.3f}s ({h2d_gb/h2d_s:.1f}GB/s effective)")

	conn.close()
	return {
		"h2d_groups": len(h2d_groups),
		"margins": margins[:50] if margins else [],
		"h2d_total_gb": h2d_gb,
		"h2d_total_s": h2d_s,
	}


# ===========================================================================
# ANALYSIS 3: NCCL SendRecv comparison
# ===========================================================================
def analyze_nccl(db_path, label):
	print(f"\n{'='*80}")
	print(f"3. NCCL SENDRECV ANALYSIS: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# All NCCL SendRecv on all devices
	cur.execute("""
		SELECT k.deviceId, COUNT(*) as cnt,
			   SUM(k.end - k.start)/1e9 as total_s,
			   AVG(k.end - k.start)/1e6 as avg_ms,
			   MIN(k.end - k.start)/1e6 as min_ms,
			   MAX(k.end - k.start)/1e6 as max_ms
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		GROUP BY k.deviceId
		ORDER BY k.deviceId
	""")
	print("\nNCCL SendRecv by device:")
	for r in cur.fetchall():
		print(f"  dev={r[0]}: {r[1]} kernels, {r[2]:.3f}s total, "
			  f"avg={r[3]:.3f}ms, min={r[4]:.3f}ms, max={r[5]:.3f}ms")

	# Duration distribution on device 0
	cur.execute("""
		SELECT (k.end - k.start)/1e6 as dur_ms
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	durs = [r[0] for r in cur.fetchall()]

	# Histogram
	buckets = {"<0.05ms": 0, "0.05-0.1ms": 0, "0.1-0.5ms": 0, "0.5-1ms": 0,
			   "1-5ms": 0, "5-10ms": 0, "10-50ms": 0, "50-100ms": 0, ">100ms": 0}
	for d in durs:
		if d < 0.05: buckets["<0.05ms"] += 1
		elif d < 0.1: buckets["0.05-0.1ms"] += 1
		elif d < 0.5: buckets["0.1-0.5ms"] += 1
		elif d < 1.0: buckets["0.5-1ms"] += 1
		elif d < 5.0: buckets["1-5ms"] += 1
		elif d < 10.0: buckets["5-10ms"] += 1
		elif d < 50.0: buckets["10-50ms"] += 1
		elif d < 100.0: buckets["50-100ms"] += 1
		else: buckets[">100ms"] += 1
	print(f"\nDuration distribution (device 0):")
	for k, v in buckets.items():
		pct = v / len(durs) * 100
		bar = "#" * int(pct)
		print(f"  {k:>12s}: {v:6d} ({pct:5.1f}%) {bar}")

	# Per-layer NCCL timing using bursts
	bursts = get_nccl_bursts(cur)

	# Identify denoising region and per-step NCCL
	gaps_ms = [(bursts[i+1][0] - bursts[i][1]) / 1e6 for i in range(len(bursts)-1)]
	gap_data = sorted(enumerate(gaps_ms), key=lambda x: -x[1])

	# Find step boundaries (top N gaps)
	step_boundary_indices = sorted([g[0] for g in gap_data[:40] if g[1] > 30])

	# Sum NCCL time per denoising step
	step_starts_list = [0] + [b + 1 for b in step_boundary_indices]
	nccl_per_step = []
	for i, si in enumerate(step_starts_list):
		ei = step_boundary_indices[i] + 1 if i < len(step_boundary_indices) else len(bursts)
		n_bursts = ei - si
		if n_bursts < 20:
			continue
		total_nccl = sum((bursts[j][1] - bursts[j][0]) / 1e6 for j in range(si, ei))
		nccl_per_step.append(total_nccl)

	if nccl_per_step:
		print(f"\nNCCL time per denoising step: "
			  f"mean={statistics.mean(nccl_per_step):.1f}ms "
			  f"std={statistics.stdev(nccl_per_step):.1f}ms "
			  f"min={min(nccl_per_step):.1f}ms max={max(nccl_per_step):.1f}ms")

	# Check if NCCL slowness is uniform or concentrated
	# Look at per-burst durations within a step
	if len(step_starts_list) > 3:
		# Pick a middle step
		mid = len(step_starts_list) // 2
		si = step_starts_list[mid]
		ei = step_boundary_indices[mid] + 1 if mid < len(step_boundary_indices) else len(bursts)
		step_burst_durs = [(bursts[j][1] - bursts[j][0]) / 1e6 for j in range(si, ei)]
		print(f"\nBurst durations in step {mid} ({len(step_burst_durs)} bursts):")
		print(f"  mean={statistics.mean(step_burst_durs):.3f}ms "
			  f"std={statistics.stdev(step_burst_durs):.3f}ms "
			  f"min={min(step_burst_durs):.3f}ms max={max(step_burst_durs):.3f}ms")

	conn.close()
	return {
		"total_nccl_s": sum(durs) / 1000,
		"n_kernels": len(durs),
		"avg_dur_ms": statistics.mean(durs),
		"nccl_per_step": nccl_per_step,
		"duration_distribution": buckets,
	}


# ===========================================================================
# ANALYSIS 4: Kernel launch density (graph break impact)
# ===========================================================================
def analyze_kernel_density(db_path, label):
	print(f"\n{'='*80}")
	print(f"4. KERNEL LAUNCH DENSITY: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Total kernels and time range on device 0
	cur.execute("""
		SELECT COUNT(*), MIN(start), MAX(end),
			   SUM(end - start)/1e9 as total_gpu_s
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0
	""")
	r = cur.fetchone()
	n_kernels = r[0]
	wall_s = (r[2] - r[1]) / 1e9
	gpu_s = r[3]
	print(f"Device 0: {n_kernels} kernels, {wall_s:.2f}s wall, {gpu_s:.2f}s GPU time")
	print(f"  Kernel launch rate: {n_kernels/wall_s:.0f} kernels/s")
	print(f"  GPU utilization: {gpu_s/wall_s*100:.1f}%")

	# Per-stream breakdown
	cur.execute("""
		SELECT streamId, COUNT(*) as cnt,
			   SUM(end-start)/1e9 as gpu_s,
			   (MAX(end) - MIN(start))/1e9 as wall_s
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0
		GROUP BY streamId
		ORDER BY gpu_s DESC
	""")
	print(f"\nPer-stream kernel stats (device 0):")
	for r in cur.fetchall():
		rate = r[1] / r[3] if r[3] > 0 else 0
		print(f"  stream={r[0]:3d}: {r[1]:8d} kernels, {r[2]:.2f}s GPU, {r[3]:.2f}s wall, "
			  f"{rate:.0f} kernels/s")

	# Kernel launch gaps on stream 7 (main compute)
	# A compiled graph would launch many kernels with near-zero gaps
	# Graph breaks show up as gaps in the kernel launch stream
	cur.execute("""
		SELECT start, end
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND streamId = 7
		ORDER BY start
	""")
	s7_kernels = cur.fetchall()
	print(f"\nStream 7: {len(s7_kernels)} kernels")

	# Compute inter-kernel gaps
	gaps_us = [(s7_kernels[i+1][0] - s7_kernels[i][1]) / 1e3
			   for i in range(len(s7_kernels)-1)]

	# Filter to steady state (skip first 10% and last 5%)
	n = len(gaps_us)
	steady_gaps = gaps_us[n//10:n*95//100]

	buckets = {"<1us": 0, "1-5us": 0, "5-10us": 0, "10-50us": 0,
			   "50-100us": 0, "100-500us": 0, "0.5-1ms": 0, "1-10ms": 0, ">10ms": 0}
	for g in steady_gaps:
		if g < 0.001: buckets["<1us"] += 1
		elif g < 0.005: buckets["1-5us"] += 1
		elif g < 0.01: buckets["5-10us"] += 1
		elif g < 0.05: buckets["10-50us"] += 1
		elif g < 0.1: buckets["50-100us"] += 1
		elif g < 0.5: buckets["100-500us"] += 1
		elif g < 1.0: buckets["0.5-1ms"] += 1
		elif g < 10.0: buckets["1-10ms"] += 1
		else: buckets[">10ms"] += 1

	print(f"\nInter-kernel gap distribution (stream 7, steady state):")
	for k, v in buckets.items():
		pct = v / len(steady_gaps) * 100
		bar = "#" * int(pct / 2)
		print(f"  {k:>12s}: {v:6d} ({pct:5.1f}%) {bar}")

	# Mean gap
	mean_gap = statistics.mean(steady_gaps)
	median_gap = statistics.median(steady_gaps)
	print(f"\n  Mean gap: {mean_gap:.3f}us, Median gap: {median_gap:.3f}us")

	# Total gap time
	total_gap_ms = sum(steady_gaps) / 1000
	total_compute_ms = sum((s7_kernels[i][1] - s7_kernels[i][0]) / 1e6
						   for i in range(len(s7_kernels)//10, len(s7_kernels)*95//100))
	print(f"  Total gap time: {total_gap_ms:.1f}ms ({total_gap_ms/(total_gap_ms+total_compute_ms)*100:.1f}% of wall time)")

	# Large gaps (>100us) - potential graph breaks
	large_gaps = [g for g in steady_gaps if g > 100]
	print(f"  Large gaps (>100us): {len(large_gaps)} ({len(large_gaps)/len(steady_gaps)*100:.2f}%)")
	if large_gaps:
		print(f"    Total time in large gaps: {sum(large_gaps)/1000:.1f}ms")

	# Graph launches
	cur.execute("""
		SELECT COUNT(*), COUNT(DISTINCT graphId)
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND graphId != 0
	""")
	r = cur.fetchone()
	print(f"\n  CUDA graph kernels: {r[0]}, unique graphs: {r[1]}")

	conn.close()
	return {
		"n_kernels_s7": len(s7_kernels),
		"kernel_rate": n_kernels / wall_s,
		"gpu_utilization": gpu_s / wall_s,
		"gap_distribution": buckets,
		"mean_gap_us": mean_gap,
		"median_gap_us": median_gap,
		"large_gap_count": len(large_gaps),
		"large_gap_total_ms": sum(large_gaps) / 1000 if large_gaps else 0,
	}


# ===========================================================================
# ANALYSIS 5: Copy stream utilization
# ===========================================================================
def analyze_copy_stream(db_path, label):
	print(f"\n{'='*80}")
	print(f"5. COPY STREAM UTILIZATION: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Get time range of denoising (using NCCL as proxy)
	cur.execute("""
		SELECT MIN(k.start), MAX(k.end)
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
	""")
	denoise_start, denoise_end = cur.fetchone()
	denoise_dur_s = (denoise_end - denoise_start) / 1e9
	print(f"Denoising region: {denoise_dur_s:.2f}s")

	# H2D memcpy activity during denoising
	cur.execute(f"""
		SELECT streamId, COUNT(*) as cnt,
			   SUM(bytes)/1e9 as gb,
			   SUM(end-start)/1e9 as active_s,
			   AVG(bytes*1.0/(end-start)) as avg_bw
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1
		AND start >= {denoise_start} AND end <= {denoise_end}
		AND bytes > 1000000
		GROUP BY streamId
		HAVING SUM(bytes) > 1e8
		ORDER BY gb DESC
	""")
	print(f"\nH2D activity during denoising (device 0, >1MB ops):")
	total_h2d_active = 0
	for r in cur.fetchall():
		utilization = r[3] / denoise_dur_s * 100
		total_h2d_active += r[3]
		print(f"  stream={r[0]:3d}: {r[1]} ops, {r[2]:.1f}GB, active={r[3]:.3f}s "
			  f"({utilization:.1f}%), BW={r[4]:.1f}GB/s")

	print(f"\n  Total copy stream active: {total_h2d_active:.3f}s ({total_h2d_active/denoise_dur_s*100:.1f}% of denoising)")
	print(f"  Copy stream idle: {denoise_dur_s - total_h2d_active:.3f}s ({(1-total_h2d_active/denoise_dur_s)*100:.1f}%)")

	# D2H activity during denoising
	cur.execute(f"""
		SELECT streamId, COUNT(*) as cnt,
			   SUM(bytes)/1e9 as gb,
			   SUM(end-start)/1e9 as active_s
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 2
		AND start >= {denoise_start} AND end <= {denoise_end}
		AND bytes > 1000000
		GROUP BY streamId
		HAVING SUM(bytes) > 1e8
		ORDER BY gb DESC
	""")
	print(f"\nD2H activity during denoising (device 0, >1MB ops):")
	for r in cur.fetchall():
		print(f"  stream={r[0]:3d}: {r[1]} ops, {r[2]:.1f}GB, active={r[3]:.3f}s")

	# H2D overlap with NCCL (temporal overlap)
	cur.execute(f"""
		SELECT m.start, m.end, m.bytes
		FROM CUPTI_ACTIVITY_KIND_MEMCPY m
		WHERE m.deviceId = 0 AND m.copyKind = 1
		AND m.bytes > 1000000
		AND m.start >= {denoise_start} AND m.end <= {denoise_end}
		ORDER BY m.start
	""")
	h2d_ops = cur.fetchall()

	cur.execute(f"""
		SELECT k.start, k.end
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		AND k.start >= {denoise_start} AND k.end <= {denoise_end}
		ORDER BY k.start
	""")
	nccl_ops = cur.fetchall()

	# Calculate overlap
	h2d_during_nccl_bytes = 0
	h2d_during_nccl_time = 0
	h2d_not_during_nccl_bytes = 0
	h2d_not_during_nccl_time = 0

	for h_start, h_end, h_bytes in h2d_ops:
		overlaps = False
		for n_start, n_end in nccl_ops:
			if h_start < n_end and h_end > n_start:
				overlaps = True
				break
		dur = h_end - h_start
		if overlaps:
			h2d_during_nccl_bytes += h_bytes
			h2d_during_nccl_time += dur
		else:
			h2d_not_during_nccl_bytes += h_bytes
			h2d_not_during_nccl_time += dur

	if h2d_ops:
		total_h2d_bytes = h2d_during_nccl_bytes + h2d_not_during_nccl_bytes
		total_h2d_time = h2d_during_nccl_time + h2d_not_during_nccl_time
		print(f"\nH2D overlap with NCCL (temporal):")
		print(f"  During NCCL: {h2d_during_nccl_bytes/1e9:.1f}GB, "
			  f"{h2d_during_nccl_time/1e9:.3f}s ({h2d_during_nccl_bytes/total_h2d_bytes*100:.1f}%)")
		print(f"  Not during NCCL: {h2d_not_during_nccl_bytes/1e9:.1f}GB, "
			  f"{h2d_not_during_nccl_time/1e9:.3f}s ({h2d_not_during_nccl_bytes/total_h2d_bytes*100:.1f}%)")
		if h2d_during_nccl_time > 0:
			bw_during = h2d_during_nccl_bytes / h2d_during_nccl_time
			print(f"  BW during NCCL: {bw_during:.1f} GB/s")
		if h2d_not_during_nccl_time > 0:
			bw_without = h2d_not_during_nccl_bytes / h2d_not_during_nccl_time
			print(f"  BW without NCCL: {bw_without:.1f} GB/s")

	conn.close()
	return {
		"denoise_dur_s": denoise_dur_s,
		"copy_active_s": total_h2d_active,
		"copy_utilization": total_h2d_active / denoise_dur_s,
	}


# ===========================================================================
# ANALYSIS 6: Memory operations (cudaMalloc/cudaFree)
# ===========================================================================
def analyze_memory_ops(db_path, label):
	print(f"\n{'='*80}")
	print(f"6. MEMORY OPERATIONS: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Find cudaMalloc and cudaFree in runtime API
	cur.execute("""
		SELECT s.value, COUNT(*) as cnt,
			   SUM(r.end - r.start)/1e9 as total_s,
			   AVG(r.end - r.start)/1e6 as avg_ms,
			   MAX(r.end - r.start)/1e6 as max_ms
		FROM CUPTI_ACTIVITY_KIND_RUNTIME r
		JOIN StringIds s ON r.nameId = s.id
		WHERE s.value LIKE '%Malloc%' OR s.value LIKE '%Free%'
		OR s.value LIKE '%Alloc%'
		GROUP BY s.value
		ORDER BY total_s DESC
	""")
	print(f"\nMemory-related CUDA runtime calls:")
	for r in cur.fetchall():
		print(f"  {r[0]:40s}: {r[1]:8d} calls, {r[2]:.3f}s total, "
			  f"avg={r[3]:.3f}ms, max={r[4]:.3f}ms")

	# Memset operations on GPU
	cur.execute("""
		SELECT deviceId, COUNT(*), SUM(bytes)/1e9 as gb, SUM(end-start)/1e9 as total_s
		FROM CUPTI_ACTIVITY_KIND_MEMSET
		WHERE deviceId = 0
		GROUP BY deviceId
	""")
	for r in cur.fetchall():
		print(f"\nMemset on device {r[0]}: {r[1]} ops, {r[2]:.1f}GB, {r[3]:.3f}s")

	# Memory pool events
	cur.execute("SELECT COUNT(*) FROM CUDA_GPU_MEMORY_POOL_EVENTS")
	print(f"Memory pool events: {cur.fetchone()[0]}")

	# Memory usage events
	cur.execute("""
		SELECT COUNT(*), MAX(bytes)/1e9 as peak_gb
		FROM CUDA_GPU_MEMORY_USAGE_EVENTS
	""")
	r = cur.fetchone()
	# Check column names
	cur.execute("PRAGMA table_info(CUDA_GPU_MEMORY_USAGE_EVENTS)")
	cols = [c[1] for c in cur.fetchall()]
	print(f"Memory usage events: {r[0]}")
	print(f"  Columns: {cols}")

	# Sample memory usage
	cur.execute("SELECT * FROM CUDA_GPU_MEMORY_USAGE_EVENTS LIMIT 5")
	for r in cur.fetchall():
		print(f"  {r}")

	conn.close()
	return {}


# ===========================================================================
# RUN ALL ANALYSES
# ===========================================================================
print("=" * 80)
print("COMPREHENSIVE NSYS ANALYSIS: B200 Baseline vs Offload")
print("=" * 80)

# Analysis 1: Per-layer compute
baseline_layers = analyze_per_layer(DB_BASELINE, "Baseline (no offload)")
offload_layers = analyze_per_layer(DB_OFFLOAD, "Offload default")
results["per_layer"] = {"baseline": baseline_layers, "offload": offload_layers}

# Comparison
if baseline_layers and offload_layers:
	print(f"\n{'='*80}")
	print("COMPARISON: Per-Layer Compute")
	print(f"{'='*80}")
	bt = baseline_layers["totals"]
	ot = offload_layers["totals"]
	print(f"  Attention: {bt['attn_ms']:.1f}ms -> {ot['attn_ms']:.1f}ms ({(ot['attn_ms']-bt['attn_ms'])/bt['attn_ms']*100:+.1f}%)")
	print(f"  FFN:       {bt['ffn_ms']:.1f}ms -> {ot['ffn_ms']:.1f}ms ({(ot['ffn_ms']-bt['ffn_ms'])/bt['ffn_ms']*100:+.1f}%)")
	nccl_b = bt['nccl1_ms'] + bt['nccl2_ms']
	nccl_o = ot['nccl1_ms'] + ot['nccl2_ms']
	print(f"  NCCL:      {nccl_b:.1f}ms -> {nccl_o:.1f}ms ({(nccl_o-nccl_b)/nccl_b*100:+.1f}%)")
	print(f"  Total/step:{bt['total_ms']:.1f}ms -> {ot['total_ms']:.1f}ms ({(ot['total_ms']-bt['total_ms'])/bt['total_ms']*100:+.1f}%)")

# Analysis 2: H2D prefetch
baseline_h2d = analyze_h2d_prefetch(DB_BASELINE, "Baseline (no offload)")
offload_h2d = analyze_h2d_prefetch(DB_OFFLOAD, "Offload default")

# Analysis 3: NCCL
baseline_nccl = analyze_nccl(DB_BASELINE, "Baseline (no offload)")
offload_nccl = analyze_nccl(DB_OFFLOAD, "Offload default")

if baseline_nccl and offload_nccl:
	print(f"\n{'='*80}")
	print("COMPARISON: NCCL SendRecv")
	print(f"{'='*80}")
	print(f"  Total NCCL: {baseline_nccl['total_nccl_s']:.3f}s -> {offload_nccl['total_nccl_s']:.3f}s "
		  f"({(offload_nccl['total_nccl_s']-baseline_nccl['total_nccl_s'])/baseline_nccl['total_nccl_s']*100:+.1f}%)")
	print(f"  Avg duration: {baseline_nccl['avg_dur_ms']:.3f}ms -> {offload_nccl['avg_dur_ms']:.3f}ms")
	print(f"  Kernel count: {baseline_nccl['n_kernels']} -> {offload_nccl['n_kernels']}")
	if baseline_nccl['nccl_per_step'] and offload_nccl['nccl_per_step']:
		b_avg = statistics.mean(baseline_nccl['nccl_per_step'])
		o_avg = statistics.mean(offload_nccl['nccl_per_step'])
		print(f"  NCCL per step: {b_avg:.1f}ms -> {o_avg:.1f}ms ({(o_avg-b_avg)/b_avg*100:+.1f}%)")

# Analysis 4: Kernel density
baseline_density = analyze_kernel_density(DB_BASELINE, "Baseline (no offload)")
offload_density = analyze_kernel_density(DB_OFFLOAD, "Offload default")

if baseline_density and offload_density:
	print(f"\n{'='*80}")
	print("COMPARISON: Kernel Launch Density")
	print(f"{'='*80}")
	print(f"  Stream 7 kernels: {baseline_density['n_kernels_s7']} -> {offload_density['n_kernels_s7']}")
	print(f"  Overall rate: {baseline_density['kernel_rate']:.0f} -> {offload_density['kernel_rate']:.0f} kernels/s")
	print(f"  GPU utilization: {baseline_density['gpu_utilization']*100:.1f}% -> {offload_density['gpu_utilization']*100:.1f}%")
	print(f"  Mean gap: {baseline_density['mean_gap_us']:.3f}us -> {offload_density['mean_gap_us']:.3f}us")
	print(f"  Large gaps (>100us): {baseline_density['large_gap_count']} -> {offload_density['large_gap_count']}")
	print(f"  Large gap time: {baseline_density['large_gap_total_ms']:.1f}ms -> {offload_density['large_gap_total_ms']:.1f}ms")

# Analysis 5: Copy stream
baseline_copy = analyze_copy_stream(DB_BASELINE, "Baseline (no offload)")
offload_copy = analyze_copy_stream(DB_OFFLOAD, "Offload default")

# Analysis 6: Memory ops
baseline_mem = analyze_memory_ops(DB_BASELINE, "Baseline (no offload)")
offload_mem = analyze_memory_ops(DB_OFFLOAD, "Offload default")

print("\n\nALL ANALYSES COMPLETE.")
