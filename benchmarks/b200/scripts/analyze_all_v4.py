#!/usr/bin/env python3
"""
Comprehensive nsys analysis v4.
Better approach: Use ALL kernels on device 0, stream 7 to identify per-layer compute regions
delimited by NCCL SendRecv kernels on stream 47.

Key insight from the data:
- ~8641 NCCL bursts in baseline, ~8288 in offload
- 27 steps * 40 layers * 2 all-to-all = 2160 all-to-all calls
- 8641 / 2160 = ~4 bursts per all-to-all (each with 2 kernels in the burst)
- So the 500us gap threshold groups them too aggressively

Let me instead use individual NCCL kernel times (not bursts) and look at
the inter-NCCL-group gaps using a larger time window to identify layers.

Actually, simpler approach: use the _attn_fwd kernel (FlashAttention) as
layer markers since there's exactly 1 per layer per denoising step.
Baseline: 4320 = 27*40*4 (wait, 4320/27/40 = 4 per layer -- multi-head on 4 GPUs?
No, it's per device. 4320/27 = 160 per step. 160/40 = 4 per layer.
That means 4 attention kernel calls per layer. With Ulysses SP on 4 GPUs,
the sequence is split 4 ways, so 4 heads groups? Let's check.

Actually with Ulysses: the input is all-to-all'ed (split seq, gather heads),
attention computed, then all-to-all'ed back (gather seq, split heads).
Each attention call is ONE _attn_fwd per layer. 4320/27 = 160 per step.
160/40 = 4 per layer. This could be: Q/K/V self-attention + cross-attention,
or multi-query with 4 calls. Let's just use them as markers.
"""
import sqlite3
import statistics
import os

DB_BASELINE = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp1_no_offload_20260227_233628.sqlite"
DB_OFFLOAD = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp2_offload_default_20260227_234442.sqlite"
OUTPUT_DIR = "/Users/dannyliu/research_work/b200_workspace/results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_all_events_device0(cur):
	"""Get all kernels and memcpy on device 0 sorted by start time."""
	# Get all NCCL SendRecv as events
	cur.execute("""
		SELECT k.start, k.end, 'nccl' as type, k.streamId
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	nccl_events = cur.fetchall()

	# Get all attention kernels
	cur.execute("""
		SELECT k.start, k.end, 'attn' as type, k.streamId
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%_attn_fwd%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	attn_events = cur.fetchall()

	return nccl_events, attn_events


def analyze_per_layer_v4(db_path, label):
	"""Per-layer compute using attention kernels as primary markers."""
	print(f"\n{'='*80}")
	print(f"1. PER-LAYER COMPUTE: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Get attention kernels on device 0
	cur.execute("""
		SELECT k.start, k.end
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%_attn_fwd%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	attn_kernels = [(r[0], r[1]) for r in cur.fetchall()]
	print(f"Attention kernels on device 0: {len(attn_kernels)}")

	# Get NCCL SendRecv kernels on device 0
	cur.execute("""
		SELECT k.start, k.end
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	nccl_kernels = [(r[0], r[1]) for r in cur.fetchall()]
	print(f"NCCL SendRecv kernels on device 0: {len(nccl_kernels)}")

	# Determine attention calls per layer
	# Check gaps between attention calls
	attn_gaps = [(attn_kernels[i+1][0] - attn_kernels[i][1]) / 1e6
				 for i in range(len(attn_kernels)-1)]

	# Large gaps = between layers, small gaps = within a layer
	sorted_gaps = sorted(attn_gaps)
	# With 4320 attn kernels and 1080 layers (27*40), there should be
	# 4320 - 1080 = 3240 intra-layer gaps and 1080 inter-layer gaps
	# (plus 26 inter-step gaps)
	# So the 3240th gap should be near the intra/inter boundary
	n_intra = len(attn_kernels) - 27 * 40  # 4320 - 1080 = 3240 intra-layer gaps
	if n_intra > 0 and n_intra < len(sorted_gaps):
		boundary = (sorted_gaps[n_intra - 1] + sorted_gaps[n_intra]) / 2
		print(f"Intra-layer/inter-layer gap boundary: {boundary:.3f}ms")
		print(f"  Intra-layer gap max: {sorted_gaps[n_intra-1]:.3f}ms")
		print(f"  Inter-layer gap min: {sorted_gaps[n_intra]:.3f}ms")
		attn_per_layer = n_intra // (27 * 40 - 1) + 1 if 27 * 40 > 1 else len(attn_kernels)
	else:
		boundary = 1.0  # fallback: 1ms
		attn_per_layer = 4  # typical for Wan model

	print(f"Attention calls per layer: ~{len(attn_kernels) / (27 * 40):.1f}")

	# Group attention kernels into layers
	# A "layer" is a group of consecutive attention calls with small gaps between them
	layer_groups = []
	current_group = [attn_kernels[0]]
	for i in range(1, len(attn_kernels)):
		gap_ms = (attn_kernels[i][0] - attn_kernels[i-1][1]) / 1e6
		if gap_ms < boundary:
			current_group.append(attn_kernels[i])
		else:
			layer_groups.append(current_group)
			current_group = [attn_kernels[i]]
	layer_groups.append(current_group)

	print(f"Attention layer groups: {len(layer_groups)}")
	attn_per = [len(g) for g in layer_groups]
	print(f"  Attention calls per group: min={min(attn_per)}, max={max(attn_per)}, "
		  f"mode={max(set(attn_per), key=attn_per.count)}")

	# Inter-group (inter-layer) gaps
	inter_layer_gaps = [(layer_groups[i+1][0][0] - layer_groups[i][-1][1]) / 1e6
						for i in range(len(layer_groups)-1)]

	# Find denoising step boundaries: largest inter-layer gaps
	sorted_inter = sorted(enumerate(inter_layer_gaps), key=lambda x: -x[1])
	# The top 26 should be step boundaries (27 steps - 1)
	# But also skip warmup/init. Let's look at the gap distribution.
	print(f"\nInter-layer gap distribution:")
	print(f"  Top 30: {[f'{g[1]:.1f}ms' for g in sorted_inter[:30]]}")

	# Step boundaries = gaps > 2x the typical inter-layer gap
	typical_inter = statistics.median(inter_layer_gaps)
	step_threshold = typical_inter * 2
	step_boundary_indices = sorted([g[0] for g in sorted_inter if g[1] > step_threshold])
	print(f"  Typical inter-layer gap: {typical_inter:.3f}ms")
	print(f"  Step threshold: {step_threshold:.3f}ms")
	print(f"  Step boundaries found: {len(step_boundary_indices)}")

	# Build denoising steps
	step_starts = [0] + [b + 1 for b in step_boundary_indices]
	denoising_steps = []
	for i, si in enumerate(step_starts):
		ei = step_boundary_indices[i] + 1 if i < len(step_boundary_indices) else len(layer_groups)
		n_layers = ei - si
		if n_layers < 10:
			continue
		step_start_ns = layer_groups[si][0][0]
		step_end_ns = layer_groups[ei-1][-1][1]
		denoising_steps.append({
			"step_idx": len(denoising_steps),
			"n_layers": n_layers,
			"group_start": si,
			"group_end": ei,
			"start_ns": step_start_ns,
			"end_ns": step_end_ns,
			"duration_ms": (step_end_ns - step_start_ns) / 1e6,
		})

	print(f"\nDenoising steps: {len(denoising_steps)}")
	for s in denoising_steps:
		print(f"  Step {s['step_idx']:2d}: {s['n_layers']:3d} layers, {s['duration_ms']:.1f}ms")

	# Per-layer timing within steady state steps
	# Skip first 2 and last 1 steps
	steady_steps = [s for s in denoising_steps if 35 <= s["n_layers"] <= 45]
	if not steady_steps:
		steady_steps = denoising_steps[2:-1] if len(denoising_steps) > 3 else denoising_steps

	print(f"\nSteady state steps (35-45 layers): {len(steady_steps)}")

	all_layer_data = []
	for step in steady_steps:
		si = step["group_start"]
		ei = step["group_end"]
		step_layers = layer_groups[si:ei]

		for j, layer_attn in enumerate(step_layers):
			# Layer compute starts at first attn kernel, ends at last attn kernel
			layer_start = layer_attn[0][0]
			layer_end = layer_attn[-1][1]
			attn_time = sum((a[1] - a[0]) for a in layer_attn) / 1e6

			# Total layer time: from this layer start to next layer start
			if j + 1 < len(step_layers):
				next_start = step_layers[j+1][0][0]
				total_time = (next_start - layer_start) / 1e6
				non_attn_time = total_time - attn_time
			else:
				total_time = (layer_end - layer_start) / 1e6
				non_attn_time = 0

			# Count NCCL kernels during this layer's time window
			nccl_in_layer = sum(1 for n in nccl_kernels
							   if n[0] >= layer_start and n[1] <= (next_start if j + 1 < len(step_layers) else layer_end))
			nccl_time = sum(max(0, min(n[1], (next_start if j + 1 < len(step_layers) else layer_end)) -
							   max(n[0], layer_start))
						   for n in nccl_kernels
						   if n[0] < (next_start if j + 1 < len(step_layers) else layer_end) and n[1] > layer_start) / 1e6

			all_layer_data.append({
				"step": step["step_idx"],
				"layer": j,
				"attn_ms": attn_time,
				"nccl_ms": nccl_time,
				"other_ms": total_time - attn_time - nccl_time if j + 1 < len(step_layers) else 0,
				"total_ms": total_time,
				"n_attn": len(layer_attn),
				"n_nccl": nccl_in_layer,
			})

	if not all_layer_data:
		print("No layer data!")
		conn.close()
		return None

	# Average across steps, per layer position
	n_layers_per_step = max(s["n_layers"] for s in steady_steps)
	avg_by_position = {}
	for d in all_layer_data:
		pos = d["layer"]
		if pos not in avg_by_position:
			avg_by_position[pos] = []
		avg_by_position[pos].append(d)

	print(f"\nPER-LAYER BREAKDOWN (avg over {len(steady_steps)} steps):")
	print(f"{'Layer':>6s} {'Attn(ms)':>10s} {'NCCL(ms)':>10s} {'Other(ms)':>10s} {'Total(ms)':>10s} {'#Attn':>6s} {'#NCCL':>6s}")
	print("-" * 66)
	avg_layers = []
	for pos in sorted(avg_by_position.keys()):
		entries = avg_by_position[pos]
		avg = {
			"layer": pos,
			"attn_ms": statistics.mean([e["attn_ms"] for e in entries]),
			"nccl_ms": statistics.mean([e["nccl_ms"] for e in entries]),
			"other_ms": statistics.mean([e["other_ms"] for e in entries]),
			"total_ms": statistics.mean([e["total_ms"] for e in entries]),
			"n_attn": statistics.mean([e["n_attn"] for e in entries]),
			"n_nccl": statistics.mean([e["n_nccl"] for e in entries]),
		}
		avg_layers.append(avg)
		if pos < 42:  # print first 42
			print(f"{pos:6d} {avg['attn_ms']:10.3f} {avg['nccl_ms']:10.3f} "
				  f"{avg['other_ms']:10.3f} {avg['total_ms']:10.3f} {avg['n_attn']:6.0f} {avg['n_nccl']:6.0f}")

	# Totals
	totals = {k: sum(l[k] for l in avg_layers if l["layer"] < n_layers_per_step - 1)
			  for k in ["attn_ms", "nccl_ms", "other_ms", "total_ms"]}
	print(f"\n{'TOTAL':>6s} {totals['attn_ms']:10.1f} {totals['nccl_ms']:10.1f} "
		  f"{totals['other_ms']:10.1f} {totals['total_ms']:10.1f}")

	# Uniformity
	attn_vals = [l["attn_ms"] for l in avg_layers if l["layer"] < n_layers_per_step - 1]
	nccl_vals = [l["nccl_ms"] for l in avg_layers if l["layer"] < n_layers_per_step - 1]
	other_vals = [l["other_ms"] for l in avg_layers if l["layer"] < n_layers_per_step - 1]

	if len(attn_vals) > 1:
		print(f"\nUniformity:")
		print(f"  Attention: {statistics.mean(attn_vals):.3f} +/- {statistics.stdev(attn_vals):.3f}ms "
			  f"(CV={statistics.stdev(attn_vals)/statistics.mean(attn_vals):.4f})")
		if nccl_vals and statistics.mean(nccl_vals) > 0:
			print(f"  NCCL:      {statistics.mean(nccl_vals):.3f} +/- {statistics.stdev(nccl_vals):.3f}ms "
				  f"(CV={statistics.stdev(nccl_vals)/statistics.mean(nccl_vals):.4f})")
		if other_vals and statistics.mean(other_vals) > 0:
			print(f"  Other:     {statistics.mean(other_vals):.3f} +/- {statistics.stdev(other_vals):.3f}ms "
				  f"(CV={statistics.stdev(other_vals)/statistics.mean(other_vals):.4f})")

	# Per-step durations
	step_durs = [s["duration_ms"] for s in denoising_steps]
	if len(step_durs) > 1:
		print(f"\nStep durations: mean={statistics.mean(step_durs):.1f}ms "
			  f"std={statistics.stdev(step_durs):.1f}ms")

	conn.close()
	return {
		"n_steps": len(denoising_steps),
		"n_layers": n_layers_per_step,
		"avg_layers": avg_layers,
		"totals": totals,
		"step_durations_ms": step_durs,
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

	# H2D summary
	cur.execute("""
		SELECT streamId, COUNT(*), SUM(bytes)/1e9 as gb, SUM(end-start)/1e9 as total_s,
			   AVG(bytes)/1e6 as avg_mb, AVG(end-start)/1e6 as avg_ms,
			   AVG(bytes*1.0/(end-start)) as avg_bw_gbs
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1 AND bytes > 1000000
		GROUP BY streamId
		HAVING SUM(bytes) > 1e9
		ORDER BY gb DESC
	""")
	print("\nH2D streams with >1GB (device 0, >1MB ops):")
	copy_streams = []
	for r in cur.fetchall():
		print(f"  stream={r[0]:3d}: {r[1]} ops, {r[2]:.1f}GB, {r[3]:.3f}s, "
			  f"avg={r[4]:.1f}MB/{r[5]:.3f}ms, BW={r[6]:.1f}GB/s")
		copy_streams.append(r[0])

	if not copy_streams:
		print("  No significant H2D streams found")
		# Check baseline H2D
		cur.execute("""
			SELECT SUM(bytes)/1e9, COUNT(*)
			FROM CUPTI_ACTIVITY_KIND_MEMCPY
			WHERE deviceId = 0 AND copyKind = 1
		""")
		r = cur.fetchone()
		print(f"  Total H2D: {r[0]:.1f}GB across {r[1]} ops")
		conn.close()
		return {"is_offload": False, "total_h2d_gb": r[0]}

	# D2H summary
	cur.execute("""
		SELECT streamId, COUNT(*), SUM(bytes)/1e9, SUM(end-start)/1e9,
			   AVG(bytes)/1e6, AVG(bytes*1.0/(end-start))
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 2 AND bytes > 1000000
		GROUP BY streamId
		HAVING SUM(bytes) > 1e8
		ORDER BY SUM(bytes) DESC
	""")
	print("\nD2H streams (>1MB ops, >100MB total):")
	for r in cur.fetchall():
		print(f"  stream={r[0]:3d}: {r[1]} ops, {r[2]:.1f}GB, {r[3]:.3f}s, "
			  f"avg={r[4]:.1f}MB, BW={r[5]:.1f}GB/s")

	# Get attention kernels to identify layer boundaries
	cur.execute("""
		SELECT k.start, k.end
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%_attn_fwd%'
		AND k.deviceId = 0
		ORDER BY k.start
	""")
	attn_kernels = cur.fetchall()

	# Group attention into layers (same as analysis 1)
	attn_gaps = [(attn_kernels[i+1][0] - attn_kernels[i][1]) / 1e6
				 for i in range(len(attn_kernels)-1)]
	n_intra = len(attn_kernels) - 27 * 40
	if n_intra > 0:
		sorted_gaps = sorted(attn_gaps)
		boundary = (sorted_gaps[n_intra - 1] + sorted_gaps[n_intra]) / 2
	else:
		boundary = 1.0

	layer_groups = []
	current_group = [attn_kernels[0]]
	for i in range(1, len(attn_kernels)):
		gap_ms = (attn_kernels[i][0] - attn_kernels[i-1][1]) / 1e6
		if gap_ms < boundary:
			current_group.append(attn_kernels[i])
		else:
			layer_groups.append(current_group)
			current_group = [attn_kernels[i]]
	layer_groups.append(current_group)

	# Get H2D transfers on copy streams during denoising
	stream_list = ",".join(str(s) for s in copy_streams)
	cur.execute(f"""
		SELECT start, end, bytes, streamId
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1
		AND bytes > 1000000
		AND streamId IN ({stream_list})
		ORDER BY start
	""")
	h2d_transfers = cur.fetchall()

	# Group H2D into per-layer prefetch groups (transfers within 2ms)
	h2d_groups = []
	if h2d_transfers:
		group = [h2d_transfers[0]]
		for t in h2d_transfers[1:]:
			if t[0] < group[-1][1] + 2000000:  # within 2ms
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

	print(f"\nH2D prefetch groups: {len(h2d_groups)}")
	if h2d_groups:
		group_durs = [(g[1]-g[0])/1e6 for g in h2d_groups]
		group_sizes = [g[2]/1e6 for g in h2d_groups]
		print(f"  Duration: mean={statistics.mean(group_durs):.3f}ms "
			  f"std={statistics.stdev(group_durs) if len(group_durs)>1 else 0:.3f}ms")
		print(f"  Size: mean={statistics.mean(group_sizes):.1f}MB "
			  f"std={statistics.stdev(group_sizes) if len(group_sizes)>1 else 0:.1f}MB")

	# Match H2D groups to layers and compute prefetch margin
	# For each layer, find the H2D group that should prefetch its weights
	# Prefetch margin = H2D group end - layer start (negative = H2D completed before layer)
	margins = []
	for i, layer_attn in enumerate(layer_groups):
		layer_start = layer_attn[0][0]
		# Find the closest H2D group that ends before or overlaps with layer start
		best = None
		for g in h2d_groups:
			if g[1] <= layer_start + 5000000:  # within 5ms after layer start
				margin_ms = (layer_start - g[1]) / 1e6  # positive = prefetch done early
				if best is None or abs(margin_ms) < abs(best):
					best = margin_ms
		if best is not None:
			margins.append(best)

	if margins:
		positive = [m for m in margins if m >= 0]
		negative = [m for m in margins if m < 0]
		print(f"\nPrefetch margin analysis ({len(margins)} layers matched):")
		print(f"  Prefetch completed early: {len(positive)} ({len(positive)/len(margins)*100:.1f}%)")
		print(f"  Prefetch late/ongoing: {len(negative)} ({len(negative)/len(margins)*100:.1f}%)")
		if positive:
			print(f"  Early margins: mean={statistics.mean(positive):.3f}ms "
				  f"min={min(positive):.3f}ms max={max(positive):.3f}ms")
		if negative:
			print(f"  Late margins: mean={statistics.mean(negative):.3f}ms "
				  f"min={min(negative):.3f}ms max={max(negative):.3f}ms")

	# Total volumes
	cur.execute("""
		SELECT SUM(bytes)/1e9, SUM(end-start)/1e9, COUNT(*)
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1
	""")
	h2d_r = cur.fetchone()
	cur.execute("""
		SELECT SUM(bytes)/1e9, SUM(end-start)/1e9, COUNT(*)
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 2
	""")
	d2h_r = cur.fetchone()
	print(f"\nTotal transfers (device 0):")
	print(f"  H2D: {h2d_r[0]:.1f}GB, {h2d_r[1]:.3f}s, {h2d_r[2]} ops "
		  f"(effective BW: {h2d_r[0]/h2d_r[1]:.1f}GB/s)")
	print(f"  D2H: {d2h_r[0]:.1f}GB, {d2h_r[1]:.3f}s, {d2h_r[2]} ops "
		  f"(effective BW: {d2h_r[0]/d2h_r[1]:.1f}GB/s)")

	conn.close()
	return {
		"is_offload": True,
		"h2d_total_gb": h2d_r[0],
		"d2h_total_gb": d2h_r[0],
		"margins": margins,
		"h2d_groups": len(h2d_groups),
	}


# ===========================================================================
# ANALYSIS 3: NCCL comparison
# ===========================================================================
def analyze_nccl(db_path, label):
	print(f"\n{'='*80}")
	print(f"3. NCCL SENDRECV: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Overall stats
	cur.execute("""
		SELECT COUNT(*) as cnt,
			   SUM(k.end - k.start)/1e9 as total_s,
			   AVG(k.end - k.start)/1e6 as avg_ms,
			   MIN(k.end - k.start)/1e6 as min_ms,
			   MAX(k.end - k.start)/1e6 as max_ms,
			   SUM(CASE WHEN k.deviceId = 0 THEN k.end - k.start ELSE 0 END)/1e9 as dev0_s
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
	""")
	r = cur.fetchone()
	print(f"All devices: {r[0]} kernels, {r[1]:.3f}s total, avg={r[2]:.3f}ms, "
		  f"min={r[3]:.3f}ms, max={r[4]:.3f}ms")
	print(f"Device 0 only: {r[5]:.3f}s")

	# Duration histogram for device 0
	cur.execute("""
		SELECT (k.end - k.start)/1e6 as dur_ms
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
	""")
	durs = [r[0] for r in cur.fetchall()]

	percentiles = [50, 75, 90, 95, 99]
	sorted_durs = sorted(durs)
	print(f"\nDevice 0 duration percentiles:")
	for p in percentiles:
		idx = int(len(sorted_durs) * p / 100)
		print(f"  p{p}: {sorted_durs[idx]:.3f}ms")

	# Check if slowdown is uniform by comparing per-device stats
	cur.execute("""
		SELECT k.deviceId, COUNT(*),
			   SUM(k.end - k.start)/1e9,
			   AVG(k.end - k.start)/1e6
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		GROUP BY k.deviceId
	""")
	print(f"\nPer-device NCCL:")
	for r in cur.fetchall():
		print(f"  dev={r[0]}: {r[1]} kernels, {r[2]:.3f}s, avg={r[3]:.3f}ms")

	# Check temporal distribution - is slowdown concentrated or uniform?
	# Split the denoising into 10 equal time windows
	cur.execute("""
		SELECT MIN(k.start), MAX(k.end)
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
	""")
	t_start, t_end = cur.fetchone()
	window = (t_end - t_start) / 10

	print(f"\nTemporal distribution (10 windows, device 0):")
	for w in range(10):
		w_start = t_start + int(w * window)
		w_end = t_start + int((w + 1) * window)
		cur.execute(f"""
			SELECT COUNT(*), SUM(k.end - k.start)/1e6, AVG(k.end - k.start)/1e6
			FROM CUPTI_ACTIVITY_KIND_KERNEL k
			JOIN StringIds s ON k.demangledName = s.id
			WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
			AND k.deviceId = 0
			AND k.start >= {w_start} AND k.start < {w_end}
		""")
		r = cur.fetchone()
		if r[0] > 0:
			print(f"  Window {w}: {r[0]:5d} kernels, {r[1]:.1f}ms total, avg={r[2]:.3f}ms")

	conn.close()
	return {
		"total_s": sum(durs) / 1000,
		"n_kernels": len(durs),
		"avg_ms": statistics.mean(durs),
		"p50_ms": sorted_durs[len(sorted_durs)//2],
		"p99_ms": sorted_durs[int(len(sorted_durs)*0.99)],
	}


# ===========================================================================
# ANALYSIS 4: Kernel launch density
# ===========================================================================
def analyze_kernel_density(db_path, label):
	print(f"\n{'='*80}")
	print(f"4. KERNEL LAUNCH DENSITY: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Stream 7 stats
	cur.execute("""
		SELECT COUNT(*), MIN(start), MAX(end), SUM(end-start)/1e9
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND streamId = 7
	""")
	r = cur.fetchone()
	n_s7 = r[0]
	wall_s7 = (r[2] - r[1]) / 1e9
	gpu_s7 = r[3]
	print(f"Stream 7: {n_s7} kernels, {wall_s7:.2f}s wall, {gpu_s7:.2f}s GPU")
	print(f"  Launch rate: {n_s7/wall_s7:.0f}/s, GPU util: {gpu_s7/wall_s7*100:.1f}%")

	# Stream 47 stats
	cur.execute("""
		SELECT COUNT(*), MIN(start), MAX(end), SUM(end-start)/1e9
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND streamId = 47
	""")
	r = cur.fetchone()
	if r[0] > 0:
		n_s47 = r[0]
		wall_s47 = (r[2] - r[1]) / 1e9
		gpu_s47 = r[3]
		print(f"Stream 47: {n_s47} kernels, {wall_s47:.2f}s wall, {gpu_s47:.2f}s GPU")

	# Overall device 0
	cur.execute("""
		SELECT COUNT(*), MIN(start), MAX(end), SUM(end-start)/1e9
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0
	""")
	r = cur.fetchone()
	n_all = r[0]
	wall_all = (r[2] - r[1]) / 1e9
	gpu_all = r[3]
	print(f"All streams: {n_all} kernels, {wall_all:.2f}s wall, {gpu_all:.2f}s GPU")
	print(f"  Launch rate: {n_all/wall_all:.0f}/s, GPU util: {gpu_all/wall_all*100:.1f}%")

	# Inter-kernel gap analysis (stream 7)
	cur.execute("""
		SELECT start, end
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND streamId = 7
		ORDER BY start
	""")
	s7_kernels = cur.fetchall()

	# Skip first/last 10%
	n = len(s7_kernels)
	si = n // 10
	ei = n * 9 // 10
	gaps_us = [(s7_kernels[i+1][0] - s7_kernels[i][1]) / 1e3
			   for i in range(si, ei - 1)]

	# Buckets
	buckets = {}
	bucket_ranges = [
		("overlap", -1e10, 0),
		("<1us", 0, 1), ("1-5us", 1, 5), ("5-10us", 5, 10),
		("10-50us", 10, 50), ("50-100us", 50, 100),
		("100us-1ms", 100, 1000), ("1-10ms", 1000, 10000), (">10ms", 10000, 1e10),
	]
	for name, lo, hi in bucket_ranges:
		buckets[name] = len([g for g in gaps_us if lo <= g < hi])

	print(f"\nStream 7 inter-kernel gap distribution (steady state):")
	for name, _, _ in bucket_ranges:
		c = buckets[name]
		pct = c / len(gaps_us) * 100
		bar = "#" * max(1, int(pct / 2)) if c > 0 else ""
		print(f"  {name:>12s}: {c:6d} ({pct:5.1f}%) {bar}")

	mean_gap = statistics.mean(gaps_us)
	median_gap = statistics.median(gaps_us)
	print(f"\n  Mean: {mean_gap:.3f}us, Median: {median_gap:.3f}us")

	# Total gap time and % of wall time
	total_gap_us = sum(max(0, g) for g in gaps_us)
	total_compute_us = sum((s7_kernels[i][1] - s7_kernels[i][0]) / 1e3
						   for i in range(si, ei))
	print(f"  Total gap: {total_gap_us/1e6:.3f}s, Total compute: {total_compute_us/1e6:.3f}s")
	print(f"  Gap fraction: {total_gap_us/(total_gap_us+total_compute_us)*100:.2f}%")

	# Large gaps (>100us) - graph break signature
	large_gaps = [g for g in gaps_us if g > 100]
	print(f"  Large gaps (>100us): {len(large_gaps)}, total={sum(large_gaps)/1e6:.3f}s")

	# CUDA graphs
	cur.execute("""
		SELECT COUNT(*), COUNT(DISTINCT graphId)
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND graphId != 0
	""")
	r = cur.fetchone()
	print(f"\n  CUDA graph kernels: {r[0]}, unique graphs: {r[1]}")

	conn.close()
	return {
		"n_kernels_s7": n_s7,
		"kernel_rate_s7": n_s7 / wall_s7,
		"gpu_util_s7": gpu_s7 / wall_s7,
		"mean_gap_us": mean_gap,
		"median_gap_us": median_gap,
		"gap_fraction": total_gap_us / (total_gap_us + total_compute_us),
		"large_gap_count": len(large_gaps),
		"large_gap_total_s": sum(large_gaps) / 1e6,
		"gap_buckets": buckets,
	}


# ===========================================================================
# ANALYSIS 5: Copy stream utilization
# ===========================================================================
def analyze_copy_stream(db_path, label):
	print(f"\n{'='*80}")
	print(f"5. COPY STREAM: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Denoising time range
	cur.execute("""
		SELECT MIN(k.start), MAX(k.end)
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%_attn_fwd%'
		AND k.deviceId = 0
	""")
	denoise_start, denoise_end = cur.fetchone()
	denoise_s = (denoise_end - denoise_start) / 1e9
	print(f"Denoising region (attn-based): {denoise_s:.2f}s")

	# All memcpy during denoising on device 0
	kind_map = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
	for kind_id, kind_name in kind_map.items():
		cur.execute(f"""
			SELECT streamId, COUNT(*), SUM(bytes)/1e9, SUM(end-start)/1e9,
				   AVG(bytes*1.0/(end-start))
			FROM CUPTI_ACTIVITY_KIND_MEMCPY
			WHERE deviceId = 0 AND copyKind = {kind_id}
			AND start >= {denoise_start} AND end <= {denoise_end}
			AND bytes > 100000
			GROUP BY streamId
			HAVING SUM(bytes) > 1e8
			ORDER BY SUM(bytes) DESC
		""")
		rows = cur.fetchall()
		if rows:
			print(f"\n{kind_name} during denoising (>100KB ops, >100MB total per stream):")
			for r in rows:
				util = r[3] / denoise_s * 100
				print(f"  stream={r[0]:3d}: {r[1]:5d} ops, {r[2]:8.1f}GB, active={r[3]:.3f}s "
					  f"({util:.1f}%), BW={r[4]:.1f}GB/s")

	# H2D/NCCL temporal overlap analysis
	cur.execute(f"""
		SELECT start, end, bytes
		FROM CUPTI_ACTIVITY_KIND_MEMCPY
		WHERE deviceId = 0 AND copyKind = 1 AND bytes > 1000000
		AND start >= {denoise_start} AND end <= {denoise_end}
		ORDER BY start
	""")
	h2d_ops = cur.fetchall()

	cur.execute(f"""
		SELECT start, end
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		AND k.start >= {denoise_start} AND k.end <= {denoise_end}
		ORDER BY k.start
	""")
	nccl_ops = cur.fetchall()

	if h2d_ops and nccl_ops:
		# Build NCCL active intervals (merge overlapping)
		nccl_intervals = []
		for n_start, n_end in nccl_ops:
			if nccl_intervals and n_start <= nccl_intervals[-1][1]:
				nccl_intervals[-1] = (nccl_intervals[-1][0], max(nccl_intervals[-1][1], n_end))
			else:
				nccl_intervals.append((n_start, n_end))

		h2d_during = {"bytes": 0, "time_ns": 0, "count": 0}
		h2d_outside = {"bytes": 0, "time_ns": 0, "count": 0}

		for h_start, h_end, h_bytes in h2d_ops:
			overlap = False
			for n_start, n_end in nccl_intervals:
				if h_start < n_end and h_end > n_start:
					overlap = True
					break
			if overlap:
				h2d_during["bytes"] += h_bytes
				h2d_during["time_ns"] += (h_end - h_start)
				h2d_during["count"] += 1
			else:
				h2d_outside["bytes"] += h_bytes
				h2d_outside["time_ns"] += (h_end - h_start)
				h2d_outside["count"] += 1

		total_bytes = h2d_during["bytes"] + h2d_outside["bytes"]
		print(f"\nH2D / NCCL temporal overlap:")
		print(f"  During NCCL: {h2d_during['count']} ops, {h2d_during['bytes']/1e9:.1f}GB "
			  f"({h2d_during['bytes']/total_bytes*100:.1f}%), {h2d_during['time_ns']/1e9:.3f}s")
		if h2d_during["time_ns"] > 0:
			print(f"    BW during NCCL: {h2d_during['bytes']/h2d_during['time_ns']:.1f}GB/s")
		print(f"  Outside NCCL: {h2d_outside['count']} ops, {h2d_outside['bytes']/1e9:.1f}GB "
			  f"({h2d_outside['bytes']/total_bytes*100:.1f}%), {h2d_outside['time_ns']/1e9:.3f}s")
		if h2d_outside["time_ns"] > 0:
			print(f"    BW outside NCCL: {h2d_outside['bytes']/h2d_outside['time_ns']:.1f}GB/s")
	else:
		print(f"\nH2D ops during denoising: {len(h2d_ops)}, NCCL ops: {len(nccl_ops)}")

	conn.close()
	return {"denoise_s": denoise_s}


# ===========================================================================
# ANALYSIS 6: Memory operations
# ===========================================================================
def analyze_memory_ops(db_path, label):
	print(f"\n{'='*80}")
	print(f"6. MEMORY OPERATIONS: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# cudaMalloc/cudaFree/cudaMemAlloc
	cur.execute("""
		SELECT s.value, COUNT(*), SUM(r.end - r.start)/1e9, AVG(r.end - r.start)/1e6,
			   MAX(r.end - r.start)/1e6
		FROM CUPTI_ACTIVITY_KIND_RUNTIME r
		JOIN StringIds s ON r.nameId = s.id
		WHERE s.value LIKE '%Malloc%' OR s.value LIKE '%Free%'
		OR s.value LIKE '%Alloc%' OR s.value LIKE '%MemPool%'
		GROUP BY s.value
		ORDER BY SUM(r.end - r.start) DESC
	""")
	print("\nMemory-related CUDA runtime calls:")
	total_mem_time = 0
	for r in cur.fetchall():
		total_mem_time += r[2]
		print(f"  {r[0]:50s}: {r[1]:6d} calls, {r[2]:.3f}s, "
			  f"avg={r[3]:.3f}ms, max={r[4]:.3f}ms")
	print(f"  TOTAL memory API time: {total_mem_time:.3f}s")

	# Synchronization analysis
	cur.execute("""
		SELECT COUNT(*), SUM(end-start)/1e9
		FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
	""")
	r = cur.fetchone()
	print(f"\nSynchronization events: {r[0]}, total time: {r[1]:.3f}s")

	# Memset
	cur.execute("""
		SELECT COUNT(*), SUM(bytes)/1e9, SUM(end-start)/1e9
		FROM CUPTI_ACTIVITY_KIND_MEMSET
		WHERE deviceId = 0
	""")
	r = cur.fetchone()
	print(f"Memset (device 0): {r[0]} ops, {r[1]:.1f}GB, {r[2]:.3f}s")

	conn.close()
	return {"total_mem_api_s": total_mem_time}


# ===========================================================================
# RUN ALL
# ===========================================================================
print("=" * 80)
print("COMPREHENSIVE NSYS ANALYSIS: B200 Baseline vs Offload")
print("=" * 80)

# 1. Per-layer compute
baseline_layers = analyze_per_layer_v4(DB_BASELINE, "Baseline (no offload)")
offload_layers = analyze_per_layer_v4(DB_OFFLOAD, "Offload default")

if baseline_layers and offload_layers:
	print(f"\n{'='*80}")
	print("COMPARISON: Per-Layer Compute")
	print(f"{'='*80}")
	bt = baseline_layers["totals"]
	ot = offload_layers["totals"]
	for k in ["attn_ms", "nccl_ms", "other_ms", "total_ms"]:
		name = k.replace("_ms", "").upper()
		if bt[k] > 0:
			print(f"  {name:>8s}: {bt[k]:8.1f}ms -> {ot[k]:8.1f}ms ({(ot[k]-bt[k])/bt[k]*100:+.1f}%)")

# 2. H2D
baseline_h2d = analyze_h2d_prefetch(DB_BASELINE, "Baseline (no offload)")
offload_h2d = analyze_h2d_prefetch(DB_OFFLOAD, "Offload default")

# 3. NCCL
baseline_nccl = analyze_nccl(DB_BASELINE, "Baseline (no offload)")
offload_nccl = analyze_nccl(DB_OFFLOAD, "Offload default")

print(f"\n{'='*80}")
print("COMPARISON: NCCL SendRecv")
print(f"{'='*80}")
print(f"  Total: {baseline_nccl['total_s']:.3f}s -> {offload_nccl['total_s']:.3f}s "
	  f"({(offload_nccl['total_s']-baseline_nccl['total_s'])/baseline_nccl['total_s']*100:+.1f}%)")
print(f"  Avg: {baseline_nccl['avg_ms']:.3f}ms -> {offload_nccl['avg_ms']:.3f}ms")
print(f"  p50: {baseline_nccl['p50_ms']:.3f}ms -> {offload_nccl['p50_ms']:.3f}ms")
print(f"  p99: {baseline_nccl['p99_ms']:.3f}ms -> {offload_nccl['p99_ms']:.3f}ms")

# 4. Kernel density
baseline_density = analyze_kernel_density(DB_BASELINE, "Baseline (no offload)")
offload_density = analyze_kernel_density(DB_OFFLOAD, "Offload default")

print(f"\n{'='*80}")
print("COMPARISON: Kernel Launch Density")
print(f"{'='*80}")
print(f"  Kernels (s7): {baseline_density['n_kernels_s7']} -> {offload_density['n_kernels_s7']}")
print(f"  Rate: {baseline_density['kernel_rate_s7']:.0f}/s -> {offload_density['kernel_rate_s7']:.0f}/s")
print(f"  GPU util: {baseline_density['gpu_util_s7']*100:.1f}% -> {offload_density['gpu_util_s7']*100:.1f}%")
print(f"  Mean gap: {baseline_density['mean_gap_us']:.1f}us -> {offload_density['mean_gap_us']:.1f}us")
print(f"  Gap fraction: {baseline_density['gap_fraction']*100:.2f}% -> {offload_density['gap_fraction']*100:.2f}%")
print(f"  Large gaps: {baseline_density['large_gap_count']} ({baseline_density['large_gap_total_s']:.3f}s) -> "
	  f"{offload_density['large_gap_count']} ({offload_density['large_gap_total_s']:.3f}s)")

# 5. Copy stream
baseline_copy = analyze_copy_stream(DB_BASELINE, "Baseline (no offload)")
offload_copy = analyze_copy_stream(DB_OFFLOAD, "Offload default")

# 6. Memory ops
baseline_mem = analyze_memory_ops(DB_BASELINE, "Baseline (no offload)")
offload_mem = analyze_memory_ops(DB_OFFLOAD, "Offload default")

print(f"\n{'='*80}")
print("COMPARISON: Memory Operations")
print(f"{'='*80}")
print(f"  Memory API time: {baseline_mem['total_mem_api_s']:.3f}s -> {offload_mem['total_mem_api_s']:.3f}s")

print("\n\n=== ALL ANALYSES COMPLETE ===")
