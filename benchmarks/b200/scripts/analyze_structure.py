#!/usr/bin/env python3
"""
Understand the denoising step/layer structure by looking at NVTX markers and kernel patterns.
With torch.compile, kernels might be fused differently.
"""
import sqlite3

DB_BASELINE = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp1_no_offload_20260227_233628.sqlite"
DB_OFFLOAD = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp2_offload_default_20260227_234442.sqlite"

def explore_nvtx(db_path, label):
	print(f"\n{'='*80}")
	print(f"NVTX MARKERS: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Check NVTX text patterns - look for denoising/step/layer markers
	cur.execute("""
		SELECT text, COUNT(*) as cnt
		FROM NVTX_EVENTS
		WHERE text IS NOT NULL
		GROUP BY text
		ORDER BY cnt DESC
		LIMIT 50
	""")
	print("\nTop 50 NVTX text markers:")
	for r in cur.fetchall():
		print(f"  {r[1]:6d}x  {r[0][:120]}")

	# Check textId-based markers
	cur.execute("""
		SELECT s.value, COUNT(*) as cnt
		FROM NVTX_EVENTS n
		JOIN StringIds s ON n.textId = s.id
		WHERE n.textId IS NOT NULL
		GROUP BY s.value
		ORDER BY cnt DESC
		LIMIT 50
	""")
	print("\nTop 50 NVTX textId markers:")
	for r in cur.fetchall():
		print(f"  {r[1]:6d}x  {r[0][:120]}")

	# Look for 'step' or 'denoising' or 'block' or 'layer' related markers
	cur.execute("""
		SELECT text, textId, COUNT(*) as cnt
		FROM NVTX_EVENTS
		WHERE text LIKE '%step%' OR text LIKE '%denois%' OR text LIKE '%block%'
		OR text LIKE '%layer%' OR text LIKE '%forward%' OR text LIKE '%backward%'
		OR text LIKE '%iteration%' OR text LIKE '%loop%'
		GROUP BY text, textId
		ORDER BY cnt DESC
		LIMIT 30
	""")
	print("\nStep/layer related NVTX:")
	for r in cur.fetchall():
		print(f"  {r[2]:6d}x  text={r[0]}, textId={r[1]}")

	# Also check textId-based
	cur.execute("""
		SELECT s.value, n.textId, COUNT(*) as cnt
		FROM NVTX_EVENTS n
		JOIN StringIds s ON n.textId = s.id
		WHERE s.value LIKE '%step%' OR s.value LIKE '%denois%' OR s.value LIKE '%block%'
		OR s.value LIKE '%layer%' OR s.value LIKE '%forward%' OR s.value LIKE '%iteration%'
		GROUP BY s.value, n.textId
		ORDER BY cnt DESC
		LIMIT 30
	""")
	print("\nStep/layer related NVTX (textId):")
	for r in cur.fetchall():
		print(f"  {r[2]:6d}x  {r[0][:120]}")

	conn.close()

def explore_kernel_patterns(db_path, label):
	print(f"\n{'='*80}")
	print(f"KERNEL PATTERNS: {label}")
	print(f"{'='*80}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()

	# Get unique kernel names and their counts on device 0, stream 7
	cur.execute("""
		SELECT s.value, COUNT(*) as cnt, SUM(k.end - k.start)/1e9 as total_s,
			   AVG(k.end - k.start)/1e6 as avg_ms
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE k.deviceId = 0 AND k.streamId = 7
		GROUP BY s.value
		ORDER BY total_s DESC
		LIMIT 30
	""")
	print(f"\nTop 30 kernels on device 0, stream 7 (by total time):")
	print(f"{'Count':>8s} {'Total(s)':>10s} {'Avg(ms)':>10s}  Kernel")
	for r in cur.fetchall():
		print(f"{r[1]:8d} {r[2]:10.3f} {r[3]:10.3f}  {r[0][:100]}")

	# Get unique kernel names on stream 47
	cur.execute("""
		SELECT s.value, COUNT(*) as cnt, SUM(k.end - k.start)/1e9 as total_s,
			   AVG(k.end - k.start)/1e6 as avg_ms
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE k.deviceId = 0 AND k.streamId = 47
		GROUP BY s.value
		ORDER BY total_s DESC
		LIMIT 15
	""")
	print(f"\nTop 15 kernels on device 0, stream 47:")
	for r in cur.fetchall():
		print(f"{r[1]:8d} {r[2]:10.3f} {r[3]:10.3f}  {r[0][:100]}")

	# Look at graph launches (graphId != 0)
	cur.execute("""
		SELECT COUNT(*), COUNT(DISTINCT graphId)
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND graphId != 0
	""")
	r = cur.fetchone()
	print(f"\nGraph-launched kernels on device 0: {r[0]} kernels, {r[1]} unique graphs")

	cur.execute("""
		SELECT graphId, COUNT(*) as cnt, SUM(end - start)/1e9 as total_s,
			   MIN(start) as first_start
		FROM CUPTI_ACTIVITY_KIND_KERNEL
		WHERE deviceId = 0 AND graphId != 0
		GROUP BY graphId
		ORDER BY first_start
		LIMIT 10
	""")
	print("Top 10 graphs (by first appearance):")
	for r in cur.fetchall():
		print(f"  graphId={r[0]}: {r[1]} kernels, {r[2]:.3f}s total")

	# NCCL SendRecv on device 0 -- look at individual kernel durations
	cur.execute("""
		SELECT (k.end - k.start)/1e6 as dur_ms, k.streamId
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		ORDER BY k.start
		LIMIT 200
	""")
	nccl_durs = cur.fetchall()
	print(f"\nFirst 200 NCCL SendRecv durations on dev 0:")
	durs = [r[0] for r in nccl_durs]
	# Group by similar duration ranges
	buckets = {}
	for d in durs:
		if d < 0.1:
			b = "<0.1ms"
		elif d < 0.5:
			b = "0.1-0.5ms"
		elif d < 1.0:
			b = "0.5-1.0ms"
		elif d < 5.0:
			b = "1-5ms"
		elif d < 10.0:
			b = "5-10ms"
		else:
			b = ">10ms"
		buckets[b] = buckets.get(b, 0) + 1
	print(f"  Distribution: {buckets}")

	# How many NCCL kernels per denoising step?
	# With Ulysses SP on 4 GPUs, each all-to-all = ncclSendRecv to 3 peers (6 kernels: 3 send + 3 recv)
	# Per layer: 2 all-to-all (before and after attention) = 12 NCCL kernels
	# Per step: 40 layers * 12 = 480 NCCL kernels per step
	# 27 steps: 27 * 480 = 12960 expected
	# Actual: 18072 baseline, 17696 offload -- more, probably some from other operations

	# Let's look at the timeline of ALL kernels on device 0, stream 7 around one denoising step
	# First, find the denoising region by looking at the dense NCCL activity
	cur.execute("""
		SELECT MIN(k.start), MAX(k.end)
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
	""")
	nccl_start, nccl_end = cur.fetchone()
	print(f"\n  NCCL SendRecv time range: {(nccl_end - nccl_start)/1e9:.2f}s")

	# Count NCCL in the first 1s of NCCL activity
	cur.execute(f"""
		SELECT COUNT(*)
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		AND k.start >= {nccl_start} AND k.start < {nccl_start + int(1e9)}
	""")
	print(f"  NCCL SendRecv in first 1s: {cur.fetchone()[0]}")

	# Get NCCL times for first 2 seconds to see periodicity
	cur.execute(f"""
		SELECT k.start - {nccl_start} as rel_start, k.end - {nccl_start} as rel_end,
			   (k.end - k.start)/1e3 as dur_us
		FROM CUPTI_ACTIVITY_KIND_KERNEL k
		JOIN StringIds s ON k.demangledName = s.id
		WHERE s.value LIKE '%ncclDevKernel_SendRecv%'
		AND k.deviceId = 0
		AND k.start >= {nccl_start} AND k.start < {nccl_start + int(2e9)}
		ORDER BY k.start
	""")
	nccl_first_2s = cur.fetchall()
	print(f"\n  NCCL SendRecv in first 2s: {len(nccl_first_2s)} kernels")

	# Group into bursts (NCCL kernels within 100us of each other)
	bursts = []
	if nccl_first_2s:
		burst_start = nccl_first_2s[0][0]
		burst_end = nccl_first_2s[0][1]
		burst_count = 1
		for rel_s, rel_e, dur in nccl_first_2s[1:]:
			if rel_s < burst_end + 100000:  # 100us
				burst_end = max(burst_end, rel_e)
				burst_count += 1
			else:
				bursts.append((burst_start/1e6, burst_end/1e6, burst_count, (burst_end-burst_start)/1e6))
				burst_start = rel_s
				burst_end = rel_e
				burst_count = 1
		bursts.append((burst_start/1e6, burst_end/1e6, burst_count, (burst_end-burst_start)/1e6))

	print(f"  NCCL bursts in first 2s: {len(bursts)}")
	for i, (s, e, c, d) in enumerate(bursts[:60]):
		print(f"    burst {i:3d}: t={s:10.3f}ms  dur={d:6.3f}ms  {c} kernels")

	conn.close()

explore_nvtx(DB_BASELINE, "Baseline")
explore_nvtx(DB_OFFLOAD, "Offload")
explore_kernel_patterns(DB_BASELINE, "Baseline")
explore_kernel_patterns(DB_OFFLOAD, "Offload")
