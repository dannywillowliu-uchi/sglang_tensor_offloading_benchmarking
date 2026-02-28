#!/usr/bin/env python3
"""analyze_nsys.py -- Extract key profiling metrics from nsys SQLite exports.

Compares no-offload baseline vs layerwise offload to quantify:
  A. H2D transfer overhead (total time, bandwidth, overlap with compute)
  B. Graph break overhead (gaps between CUDA kernels)
  C. Memory allocator overhead (cudaMalloc calls from torch.empty)
  D. Stream synchronization overhead (copy_stream idle time)
  E. Hook dispatch overhead (Python function call gaps)

Usage:
    # Export nsys-rep to SQLite first:
    nsys export --type=sqlite -o trace.sqlite trace.nsys-rep

    # Then analyze:
    python3 b200/analyze_nsys.py <nsys_dir>

    # Or compare two specific traces:
    python3 b200/analyze_nsys.py --baseline <baseline.sqlite> --offload <offload.sqlite>
"""

import argparse
import csv
import glob
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def log(msg):
	print(msg, flush=True)


# ============================================================================
# SQL Queries
# ============================================================================

# --- A. H2D Transfer Overhead ---

Q_H2D_SUMMARY = """
SELECT
	COUNT(*) as num_transfers,
	SUM(bytes) / 1e9 as total_gb,
	SUM(end - start) / 1e9 as total_time_s,
	CASE WHEN SUM(end - start) > 0
		THEN (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9)
		ELSE 0 END as effective_bw_gbps,
	AVG(bytes) / 1e6 as avg_size_mb,
	MAX(bytes) / 1e6 as max_size_mb,
	AVG((end - start) / 1e6) as avg_dur_ms,
	MAX((end - start) / 1e6) as max_dur_ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE copyKind = 1;
"""

Q_H2D_SIZE_BUCKETS = """
SELECT
	CASE
		WHEN bytes > 500000000 THEN '>500MB'
		WHEN bytes > 100000000 THEN '100-500MB'
		WHEN bytes > 10000000 THEN '10-100MB'
		WHEN bytes > 1000000 THEN '1-10MB'
		WHEN bytes > 10000 THEN '10KB-1MB'
		ELSE '<10KB'
	END as size_bucket,
	COUNT(*) as count,
	SUM(bytes) / 1e9 as total_gb,
	SUM(end - start) / 1e9 as total_time_s,
	AVG((bytes / 1e6) / NULLIF((end - start) / 1e9, 0)) as avg_bw_mbps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE copyKind = 1
GROUP BY size_bucket
ORDER BY total_gb DESC;
"""

# H2D during vs outside NCCL (overlap detection)
Q_H2D_NCCL_OVERLAP = """
WITH nccl_ids AS (
	SELECT id FROM StringIds WHERE value LIKE '%ncclDevKernel%'
),
nccl_windows AS (
	SELECT k.start, k.end, k.deviceId
	FROM CUPTI_ACTIVITY_KIND_KERNEL k
	WHERE k.demangledName IN (SELECT id FROM nccl_ids)
),
h2d AS (
	SELECT start, end, bytes, deviceId,
		CASE WHEN (end - start) > 0
			THEN (bytes / 1e6) / ((end - start) / 1e9)
			ELSE 0 END as bw_mbps
	FROM CUPTI_ACTIVITY_KIND_MEMCPY
	WHERE copyKind = 1 AND bytes > 1000000
)
SELECT
	'during_nccl' as context,
	COUNT(*) as h2d_count,
	SUM(h.bytes) / 1e9 as total_gb,
	AVG(h.bw_mbps) as avg_bw_mbps
FROM h2d h
WHERE EXISTS (
	SELECT 1 FROM nccl_windows n
	WHERE n.deviceId = h.deviceId AND n.start < h.end AND n.end > h.start
)
UNION ALL
SELECT
	'no_nccl' as context,
	COUNT(*) as h2d_count,
	SUM(h.bytes) / 1e9 as total_gb,
	AVG(h.bw_mbps) as avg_bw_mbps
FROM h2d h
WHERE NOT EXISTS (
	SELECT 1 FROM nccl_windows n
	WHERE n.deviceId = h.deviceId AND n.start < h.end AND n.end > h.start
);
"""

# --- B. Graph Break / Kernel Gap Overhead ---

# Gaps between consecutive CUDA kernels on same device (>0.1ms = significant)
Q_KERNEL_GAPS = """
WITH ordered AS (
	SELECT start, end, demangledName,
		ROW_NUMBER() OVER (ORDER BY start) as rn
	FROM CUPTI_ACTIVITY_KIND_KERNEL
	WHERE deviceId = 0
),
gaps AS (
	SELECT
		(b.start - a.end) as gap_ns,
		a.demangledName as before_id,
		b.demangledName as after_id
	FROM ordered a
	JOIN ordered b ON b.rn = a.rn + 1
	WHERE b.start > a.end
)
SELECT
	0 as deviceId,
	COUNT(*) as total_gaps,
	SUM(CASE WHEN gap_ns > 100000 THEN 1 ELSE 0 END) as gaps_over_100us,
	SUM(CASE WHEN gap_ns > 1000000 THEN 1 ELSE 0 END) as gaps_over_1ms,
	SUM(CASE WHEN gap_ns > 10000000 THEN 1 ELSE 0 END) as gaps_over_10ms,
	SUM(gap_ns) / 1e9 as total_gap_s,
	SUM(CASE WHEN gap_ns > 100000 THEN gap_ns ELSE 0 END) / 1e9 as significant_gap_s,
	AVG(gap_ns) / 1e3 as avg_gap_us,
	MAX(gap_ns) / 1e6 as max_gap_ms
FROM gaps;
"""

# Top 20 largest gaps with surrounding kernel names
Q_LARGEST_GAPS = """
WITH ordered AS (
	SELECT start, end, demangledName,
		ROW_NUMBER() OVER (ORDER BY start) as rn
	FROM CUPTI_ACTIVITY_KIND_KERNEL
	WHERE deviceId = 0
),
gaps AS (
	SELECT
		(b.start - a.end) / 1e6 as gap_ms,
		a.demangledName as before_id,
		b.demangledName as after_id
	FROM ordered a
	JOIN ordered b ON b.rn = a.rn + 1
	WHERE (b.start - a.end) > 1000000
)
SELECT
	g.gap_ms,
	SUBSTR(s1.value, 1, 80) as before_kernel,
	SUBSTR(s2.value, 1, 80) as after_kernel
FROM gaps g
JOIN StringIds s1 ON g.before_id = s1.id
JOIN StringIds s2 ON g.after_id = s2.id
ORDER BY g.gap_ms DESC
LIMIT 20;
"""

# --- C. Memory Allocator Overhead ---

# cudaMalloc/cudaFree calls from CUDA runtime API
Q_CUDA_MALLOC = """
SELECT
	SUBSTR(s.value, 1, 40) as api_name,
	COUNT(*) as call_count,
	SUM(end - start) / 1e9 as total_time_s,
	AVG((end - start) / 1e3) as avg_dur_us,
	MAX((end - start) / 1e3) as max_dur_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE s.value LIKE 'cudaMalloc%' OR s.value LIKE 'cudaFree%'
GROUP BY api_name
ORDER BY total_time_s DESC;
"""

# --- D. NCCL Kernel Timing ---

Q_NCCL_KERNELS = """
SELECT
	SUBSTR(s.value, 1, 80) as kernel_name,
	COUNT(*) as count,
	SUM(k.end - k.start) / 1e9 as total_time_s,
	AVG((k.end - k.start) / 1e6) as avg_dur_ms,
	MAX((k.end - k.start) / 1e6) as max_dur_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%nccl%'
GROUP BY kernel_name
ORDER BY total_time_s DESC;
"""

# --- E. Overall GPU Utilization ---

Q_GPU_UTILIZATION = """
SELECT
	deviceId,
	COUNT(*) as num_kernels,
	SUM(end - start) / 1e9 as total_kernel_time_s,
	(MAX(end) - MIN(start)) / 1e9 as wall_time_s,
	CAST(SUM(end - start) AS REAL) / NULLIF(MAX(end) - MIN(start), 0) as utilization
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY deviceId
ORDER BY deviceId;
"""

# --- F. Per-Step Timing (using H2D clusters as step boundaries) ---
# Large H2D transfers (~700MB) mark layer prefetches; clusters = denoising steps
Q_H2D_TIMELINE = """
SELECT
	deviceId,
	(start / 1000000000) as time_bucket_s,
	COUNT(*) as transfers,
	SUM(bytes) / 1e6 as total_mb,
	AVG(CASE WHEN (end - start) > 0
		THEN (bytes / 1e6) / ((end - start) / 1e9)
		ELSE 0 END) as avg_bw_mbps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE copyKind = 1 AND bytes > 100000
GROUP BY deviceId, time_bucket_s
ORDER BY deviceId, time_bucket_s;
"""

# --- G. H2D overlap with compute (not just NCCL) ---
Q_H2D_COMPUTE_OVERLAP = """
WITH compute_kernels AS (
	SELECT start, end, deviceId
	FROM CUPTI_ACTIVITY_KIND_KERNEL k
	JOIN StringIds s ON k.demangledName = s.id
	WHERE s.value NOT LIKE '%nccl%' AND s.value NOT LIKE '%Memcpy%'
),
h2d AS (
	SELECT start, end, bytes, deviceId
	FROM CUPTI_ACTIVITY_KIND_MEMCPY
	WHERE copyKind = 1 AND bytes > 1000000
)
SELECT
	'during_compute' as context,
	COUNT(*) as h2d_count,
	SUM(h.bytes) / 1e9 as total_gb
FROM h2d h
WHERE EXISTS (
	SELECT 1 FROM compute_kernels c
	WHERE c.deviceId = h.deviceId AND c.start < h.end AND c.end > h.start
)
UNION ALL
SELECT
	'no_compute' as context,
	COUNT(*) as h2d_count,
	SUM(h.bytes) / 1e9 as total_gb
FROM h2d h
WHERE NOT EXISTS (
	SELECT 1 FROM compute_kernels c
	WHERE c.deviceId = h.deviceId AND c.start < h.end AND c.end > h.start
);
"""


# ============================================================================
# Analysis Engine
# ============================================================================

@dataclass
class TraceMetrics:
	"""Collected metrics from a single nsys trace."""
	name: str
	h2d_summary: dict = field(default_factory=dict)
	h2d_buckets: list = field(default_factory=list)
	h2d_nccl_overlap: list = field(default_factory=list)
	h2d_compute_overlap: list = field(default_factory=list)
	kernel_gaps: dict = field(default_factory=dict)
	largest_gaps: list = field(default_factory=list)
	cuda_malloc: list = field(default_factory=list)
	nccl_kernels: list = field(default_factory=list)
	gpu_utilization: list = field(default_factory=list)
	h2d_timeline: list = field(default_factory=list)


def create_indices(conn):
	"""Create indices for fast overlap queries."""
	indices = [
		("idx_memcpy_copykind", "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind, deviceId, start, end"),
		("idx_kernel_device_time", "CUPTI_ACTIVITY_KIND_KERNEL", "deviceId, start, end"),
		("idx_kernel_name", "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"),
		("idx_stringids_value", "StringIds", "value"),
	]
	for name, table, cols in indices:
		try:
			conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols})")
		except Exception:
			pass
	conn.commit()


def run_query(conn, query, label=""):
	"""Execute query and return list of dicts."""
	t0 = time.time()
	conn.row_factory = sqlite3.Row
	cursor = conn.execute(query)
	rows = [dict(row) for row in cursor.fetchall()]
	elapsed = time.time() - t0
	if label:
		log(f"    {label}: {len(rows)} rows [{elapsed:.1f}s]")
	return rows


def has_table(conn, table_name):
	"""Check if a table exists in the database."""
	result = conn.execute(
		"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
		(table_name,)
	).fetchone()
	return result[0] > 0


def analyze_trace(db_path, name):
	"""Run all queries on a single trace and return TraceMetrics."""
	log(f"\n  Analyzing: {name} ({os.path.basename(db_path)})")
	conn = sqlite3.connect(db_path)
	create_indices(conn)

	metrics = TraceMetrics(name=name)

	# H2D summary
	rows = run_query(conn, Q_H2D_SUMMARY, "H2D summary")
	if rows:
		metrics.h2d_summary = rows[0]

	# H2D size buckets
	metrics.h2d_buckets = run_query(conn, Q_H2D_SIZE_BUCKETS, "H2D buckets")

	# H2D / NCCL overlap
	metrics.h2d_nccl_overlap = run_query(conn, Q_H2D_NCCL_OVERLAP, "H2D/NCCL overlap")

	# H2D / compute overlap
	metrics.h2d_compute_overlap = run_query(conn, Q_H2D_COMPUTE_OVERLAP, "H2D/compute overlap")

	# Kernel gaps
	rows = run_query(conn, Q_KERNEL_GAPS, "Kernel gaps")
	if rows:
		metrics.kernel_gaps = rows[0]

	# Largest gaps
	metrics.largest_gaps = run_query(conn, Q_LARGEST_GAPS, "Largest gaps")

	# cudaMalloc
	if has_table(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
		metrics.cuda_malloc = run_query(conn, Q_CUDA_MALLOC, "cudaMalloc/Free")
	else:
		log("    cudaMalloc: CUPTI_ACTIVITY_KIND_RUNTIME table not found (need -t osrt)")

	# NCCL kernels
	metrics.nccl_kernels = run_query(conn, Q_NCCL_KERNELS, "NCCL kernels")

	# GPU utilization
	metrics.gpu_utilization = run_query(conn, Q_GPU_UTILIZATION, "GPU utilization")

	# H2D timeline
	metrics.h2d_timeline = run_query(conn, Q_H2D_TIMELINE, "H2D timeline")

	conn.close()
	return metrics


# ============================================================================
# Report Generation
# ============================================================================

def fmt(val, suffix="", precision=2):
	"""Format a numeric value nicely."""
	if val is None:
		return "N/A"
	if isinstance(val, float):
		return f"{val:.{precision}f}{suffix}"
	return f"{val}{suffix}"


def compare_report(baseline: TraceMetrics, offload: TraceMetrics, output_dir: str):
	"""Generate a comparison report between baseline (no offload) and offload traces."""

	lines = []
	def w(s=""):
		lines.append(s)

	w("# B200 Profiling: No-Offload Baseline vs Layerwise Offload")
	w(f"")
	w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
	w(f"Baseline: {baseline.name}")
	w(f"Offload:  {offload.name}")
	w()

	# --- A. H2D Transfer Overhead ---
	w("## A. H2D Transfer Overhead")
	w()
	b, o = baseline.h2d_summary, offload.h2d_summary
	w("| Metric | Baseline | Offload | Delta |")
	w("|--------|----------|---------|-------|")
	for key, label, suffix in [
		("num_transfers", "Transfer count", ""),
		("total_gb", "Total volume", " GB"),
		("total_time_s", "Total H2D time", " s"),
		("effective_bw_gbps", "Effective bandwidth", " GB/s"),
		("avg_size_mb", "Avg transfer size", " MB"),
		("max_size_mb", "Max transfer size", " MB"),
		("avg_dur_ms", "Avg transfer duration", " ms"),
		("max_dur_ms", "Max transfer duration", " ms"),
	]:
		bv = b.get(key)
		ov = o.get(key)
		if bv is not None and ov is not None and isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
			delta = ov - bv
			delta_str = f"{delta:+.2f}{suffix}"
		else:
			delta_str = "N/A"
		w(f"| {label} | {fmt(bv, suffix)} | {fmt(ov, suffix)} | {delta_str} |")
	w()

	# H2D size distribution
	w("### H2D Size Distribution")
	w()
	for label, metrics in [("Baseline", baseline), ("Offload", offload)]:
		w(f"**{label}:**")
		w("| Bucket | Count | Total GB | Avg BW (MB/s) |")
		w("|--------|-------|----------|---------------|")
		for row in metrics.h2d_buckets:
			w(f"| {row['size_bucket']} | {row['count']} | {fmt(row['total_gb'])} | {fmt(row.get('avg_bw_mbps'))} |")
		w()

	# --- H2D / NCCL overlap ---
	w("### H2D / NCCL Overlap")
	w()
	w("Key question: Does H2D contend with NCCL on NVLink? (Should be minimal.)")
	w()
	for label, metrics in [("Baseline", baseline), ("Offload", offload)]:
		w(f"**{label}:**")
		w("| Context | H2D Count | Total GB | Avg BW (MB/s) |")
		w("|---------|-----------|----------|---------------|")
		for row in metrics.h2d_nccl_overlap:
			w(f"| {row['context']} | {row['h2d_count']} | {fmt(row.get('total_gb'))} | {fmt(row.get('avg_bw_mbps'))} |")
		w()

	# --- H2D / compute overlap ---
	w("### H2D / Compute Overlap")
	w()
	w("Does H2D overlap with layer compute? (Want: most H2D during compute.)")
	w()
	for label, metrics in [("Baseline", baseline), ("Offload", offload)]:
		w(f"**{label}:**")
		w("| Context | H2D Count | Total GB |")
		w("|---------|-----------|----------|")
		for row in metrics.h2d_compute_overlap:
			w(f"| {row['context']} | {row['h2d_count']} | {fmt(row.get('total_gb'))} |")
		w()

	# --- B. Graph Break / Kernel Gap Overhead ---
	w("## B. Kernel Gap Overhead (Graph Breaks)")
	w()
	w("Gaps between consecutive CUDA kernels on GPU 0. Graph breaks from")
	w("`@torch.compiler.disable` hooks insert CPU-side Python overhead.")
	w()
	w("| Metric | Baseline | Offload | Delta |")
	w("|--------|----------|---------|-------|")
	bg, og = baseline.kernel_gaps, offload.kernel_gaps
	for key, label, suffix in [
		("total_gaps", "Total gaps", ""),
		("gaps_over_100us", "Gaps > 100us", ""),
		("gaps_over_1ms", "Gaps > 1ms", ""),
		("gaps_over_10ms", "Gaps > 10ms", ""),
		("total_gap_s", "Total gap time", " s"),
		("significant_gap_s", "Significant gap time (>100us)", " s"),
		("avg_gap_us", "Avg gap", " us"),
		("max_gap_ms", "Max gap", " ms"),
	]:
		bv = bg.get(key)
		ov = og.get(key)
		if bv is not None and ov is not None and isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
			delta = ov - bv
			delta_str = f"{delta:+.2f}{suffix}"
		else:
			delta_str = "N/A"
		w(f"| {label} | {fmt(bv, suffix)} | {fmt(ov, suffix)} | {delta_str} |")
	w()

	# Largest gaps
	w("### Largest Kernel Gaps (Offload, GPU 0)")
	w()
	w("| Gap (ms) | Before Kernel | After Kernel |")
	w("|----------|---------------|--------------|")
	for row in offload.largest_gaps[:10]:
		w(f"| {fmt(row['gap_ms'])} | {row['before_kernel'][:60]} | {row['after_kernel'][:60]} |")
	w()

	# --- C. Memory Allocator Overhead ---
	w("## C. Memory Allocator Overhead (cudaMalloc/cudaFree)")
	w()
	for label, metrics in [("Baseline", baseline), ("Offload", offload)]:
		w(f"**{label}:**")
		if not metrics.cuda_malloc:
			w("(no data -- CUPTI_ACTIVITY_KIND_RUNTIME table not available)")
		else:
			w("| API | Calls | Total Time (s) | Avg (us) | Max (us) |")
			w("|-----|-------|----------------|----------|----------|")
			for row in metrics.cuda_malloc:
				w(f"| {row['api_name']} | {row['call_count']} | {fmt(row['total_time_s'])} | {fmt(row['avg_dur_us'])} | {fmt(row['max_dur_us'])} |")
		w()

	# --- D. NCCL Kernel Timing ---
	w("## D. NCCL Kernel Timing")
	w()
	for label, metrics in [("Baseline", baseline), ("Offload", offload)]:
		w(f"**{label}:**")
		w("| Kernel | Count | Total (s) | Avg (ms) | Max (ms) |")
		w("|--------|-------|-----------|----------|----------|")
		for row in metrics.nccl_kernels[:5]:
			w(f"| {row['kernel_name'][:60]} | {row['count']} | {fmt(row['total_time_s'])} | {fmt(row['avg_dur_ms'])} | {fmt(row['max_dur_ms'])} |")
		w()

	# --- E. GPU Utilization ---
	w("## E. GPU Utilization")
	w()
	for label, metrics in [("Baseline", baseline), ("Offload", offload)]:
		w(f"**{label}:**")
		w("| GPU | Kernels | Kernel Time (s) | Wall Time (s) | Utilization |")
		w("|-----|---------|-----------------|---------------|-------------|")
		for row in metrics.gpu_utilization:
			util_pct = f"{row['utilization']*100:.1f}%" if row.get("utilization") else "N/A"
			w(f"| {row['deviceId']} | {row['num_kernels']} | {fmt(row['total_kernel_time_s'])} | {fmt(row['wall_time_s'])} | {util_pct} |")
		w()

	# --- Summary ---
	w("## Summary")
	w()

	# Calculate key deltas
	b_step = None
	o_step = None
	if baseline.gpu_utilization and offload.gpu_utilization:
		b_wall = baseline.gpu_utilization[0].get("wall_time_s", 0)
		o_wall = offload.gpu_utilization[0].get("wall_time_s", 0)
		if b_wall and o_wall:
			# Rough per-step: wall_time / 27 steps
			b_step = b_wall / 27
			o_step = o_wall / 27

	w("### Per-Step Estimates (27 steps)")
	w()
	if b_step and o_step:
		w(f"- Baseline: ~{b_step:.2f} s/step")
		w(f"- Offload:  ~{o_step:.2f} s/step")
		w(f"- Delta:    ~{o_step - b_step:+.2f} s/step ({(o_step - b_step) / b_step * 100:+.1f}%)")
	else:
		w("(Could not compute -- check GPU utilization data)")
	w()

	w("### Overhead Decomposition")
	w()
	h2d_delta = (offload.h2d_summary.get("total_time_s", 0) or 0) - (baseline.h2d_summary.get("total_time_s", 0) or 0)
	gap_delta = (offload.kernel_gaps.get("significant_gap_s", 0) or 0) - (baseline.kernel_gaps.get("significant_gap_s", 0) or 0)
	w(f"- H2D transfer time delta: {h2d_delta:+.2f} s")
	w(f"- Kernel gap time delta (>100us): {gap_delta:+.2f} s")

	# NCCL delta
	b_nccl = sum(r.get("total_time_s", 0) or 0 for r in baseline.nccl_kernels)
	o_nccl = sum(r.get("total_time_s", 0) or 0 for r in offload.nccl_kernels)
	w(f"- NCCL total time delta: {o_nccl - b_nccl:+.2f} s")
	w()

	w("### Key Questions Answered")
	w()

	# Q1: Does H2D contend with NCCL on NVLink?
	offload_during = next((r for r in offload.h2d_nccl_overlap if r["context"] == "during_nccl"), None)
	offload_outside = next((r for r in offload.h2d_nccl_overlap if r["context"] == "no_nccl"), None)
	if offload_during and offload_outside:
		during_gb = offload_during.get("total_gb") or 0
		total_gb = during_gb + (offload_outside.get("total_gb") or 0)
		pct = (during_gb / total_gb * 100) if total_gb > 0 else 0
		w(f"**Q1: H2D/NCCL contention on NVLink?** {pct:.0f}% of H2D occurs during NCCL")
		if pct < 10:
			w("  -> Minimal contention (NVLink isolates paths as expected)")
		else:
			w("  -> Significant contention (unexpected on NVLink -- investigate)")
	w()

	# Q2: Do graph breaks dominate?
	if offload.kernel_gaps and baseline.kernel_gaps:
		offload_sig = offload.kernel_gaps.get("significant_gap_s", 0) or 0
		baseline_sig = baseline.kernel_gaps.get("significant_gap_s", 0) or 0
		w(f"**Q2: Graph break overhead?** Baseline: {baseline_sig:.2f}s, Offload: {offload_sig:.2f}s")
		if offload_sig - baseline_sig > 5:
			w("  -> Graph breaks add significant overhead")
		else:
			w("  -> Graph breaks are NOT the dominant overhead source")
	w()

	# Q3: Does H2D overlap with compute?
	offload_compute = next((r for r in offload.h2d_compute_overlap if r["context"] == "during_compute"), None)
	offload_no_compute = next((r for r in offload.h2d_compute_overlap if r["context"] == "no_compute"), None)
	if offload_compute and offload_no_compute:
		compute_gb = offload_compute.get("total_gb") or 0
		total_gb = compute_gb + (offload_no_compute.get("total_gb") or 0)
		pct = (compute_gb / total_gb * 100) if total_gb > 0 else 0
		w(f"**Q3: H2D overlaps with compute?** {pct:.0f}% of H2D during compute kernels")
		if pct > 80:
			w("  -> Good overlap -- prefetch is working")
		else:
			w("  -> Poor overlap -- prefetch not hiding H2D latency")
	w()

	# Q4: Allocator churn
	offload_mallocs = sum(r.get("call_count", 0) for r in offload.cuda_malloc if "Malloc" in r.get("api_name", ""))
	baseline_mallocs = sum(r.get("call_count", 0) for r in baseline.cuda_malloc if "Malloc" in r.get("api_name", ""))
	if offload_mallocs or baseline_mallocs:
		w(f"**Q4: Allocator churn?** Baseline: {baseline_mallocs} mallocs, Offload: {offload_mallocs} mallocs")
		if offload_mallocs > baseline_mallocs * 2:
			w("  -> Significant allocator churn from torch.empty() calls")
		else:
			w("  -> Allocator churn is manageable")
	w()

	# Write report
	report_path = os.path.join(output_dir, "profiling_report.md")
	with open(report_path, "w") as f:
		f.write("\n".join(lines))
	log(f"\n  Report written to: {report_path}")

	# Also write raw metrics as JSON for programmatic access
	json_path = os.path.join(output_dir, "metrics.json")
	metrics_dict = {}
	for label, m in [("baseline", baseline), ("offload", offload)]:
		metrics_dict[label] = {
			"name": m.name,
			"h2d_summary": m.h2d_summary,
			"h2d_buckets": m.h2d_buckets,
			"h2d_nccl_overlap": m.h2d_nccl_overlap,
			"h2d_compute_overlap": m.h2d_compute_overlap,
			"kernel_gaps": m.kernel_gaps,
			"cuda_malloc": m.cuda_malloc,
			"nccl_kernels": m.nccl_kernels,
			"gpu_utilization": m.gpu_utilization,
		}
	with open(json_path, "w") as f:
		json.dump(metrics_dict, f, indent=2, default=str)
	log(f"  Metrics JSON: {json_path}")

	return lines


def single_report(metrics: TraceMetrics, output_dir: str):
	"""Generate a report for a single trace (no comparison)."""
	lines = []
	def w(s=""):
		lines.append(s)

	w(f"# B200 Profiling: {metrics.name}")
	w(f"")
	w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
	w()

	w("## H2D Transfer Summary")
	w()
	for key, label in [
		("num_transfers", "Transfers"),
		("total_gb", "Total GB"),
		("total_time_s", "Total Time (s)"),
		("effective_bw_gbps", "Effective BW (GB/s)"),
		("avg_size_mb", "Avg Size (MB)"),
		("avg_dur_ms", "Avg Duration (ms)"),
	]:
		w(f"- {label}: {fmt(metrics.h2d_summary.get(key))}")
	w()

	w("## Kernel Gaps (GPU 0)")
	w()
	for key, label in [
		("total_gaps", "Total gaps"),
		("gaps_over_1ms", "Gaps > 1ms"),
		("total_gap_s", "Total gap time (s)"),
		("significant_gap_s", "Significant gap time (s)"),
		("max_gap_ms", "Max gap (ms)"),
	]:
		w(f"- {label}: {fmt(metrics.kernel_gaps.get(key))}")
	w()

	w("## GPU Utilization")
	w()
	for row in metrics.gpu_utilization:
		util = f"{row['utilization']*100:.1f}%" if row.get("utilization") else "N/A"
		w(f"- GPU {row['deviceId']}: {util} ({fmt(row.get('total_kernel_time_s'))}s kernel / {fmt(row.get('wall_time_s'))}s wall)")
	w()

	report_path = os.path.join(output_dir, f"report_{metrics.name}.md")
	with open(report_path, "w") as f:
		f.write("\n".join(lines))
	log(f"  Report: {report_path}")


# ============================================================================
# CLI
# ============================================================================

def find_sqlite_files(nsys_dir):
	"""Find and auto-export nsys-rep files to SQLite, return dict of name -> path."""
	files = {}

	# Look for existing .sqlite files
	for f in sorted(glob.glob(os.path.join(nsys_dir, "*.sqlite"))):
		name = os.path.basename(f).replace(".sqlite", "")
		files[name] = f

	# Look for .nsys-rep files that haven't been exported yet
	for f in sorted(glob.glob(os.path.join(nsys_dir, "*.nsys-rep"))):
		sqlite_path = f.replace(".nsys-rep", ".sqlite")
		name = os.path.basename(f).replace(".nsys-rep", "")
		if name not in files and not os.path.exists(sqlite_path):
			log(f"  Exporting {os.path.basename(f)} to SQLite...")
			ret = os.system(f'nsys export --type=sqlite -o "{sqlite_path}" "{f}"')
			if ret == 0 and os.path.exists(sqlite_path):
				files[name] = sqlite_path
			else:
				log(f"  WARNING: nsys export failed for {f}")
		elif os.path.exists(sqlite_path):
			files[name] = sqlite_path

	return files


def main():
	parser = argparse.ArgumentParser(description="Analyze nsys traces for B200 profiling")
	parser.add_argument("nsys_dir", nargs="?", help="Directory containing nsys-rep or sqlite files")
	parser.add_argument("--baseline", help="Baseline (no offload) SQLite file")
	parser.add_argument("--offload", help="Offload SQLite file")
	parser.add_argument("--pcie-aware", help="PCIe-aware offload SQLite file (optional)")
	parser.add_argument("--output", "-o", help="Output directory (default: <nsys_dir>/analysis)")
	args = parser.parse_args()

	log("=" * 64)
	log("  B200 nsys Profiling Analysis")
	log("=" * 64)

	# Mode 1: Explicit baseline + offload
	if args.baseline and args.offload:
		output_dir = args.output or os.path.dirname(args.offload)
		os.makedirs(output_dir, exist_ok=True)

		baseline = analyze_trace(args.baseline, "no_offload")
		offload = analyze_trace(args.offload, "offload_default")
		compare_report(baseline, offload, output_dir)

		if args.pcie_aware:
			pcie = analyze_trace(args.pcie_aware, "offload_pcie_aware")
			compare_report(baseline, pcie, output_dir)
		return

	# Mode 2: Auto-discover from nsys_dir
	if not args.nsys_dir:
		parser.print_help()
		sys.exit(1)

	nsys_dir = args.nsys_dir
	output_dir = args.output or os.path.join(nsys_dir, "analysis")
	os.makedirs(output_dir, exist_ok=True)

	log(f"\n  Scanning: {nsys_dir}")
	files = find_sqlite_files(nsys_dir)

	if not files:
		log("  ERROR: No nsys-rep or sqlite files found")
		sys.exit(1)

	log(f"  Found {len(files)} trace(s):")
	for name, path in files.items():
		log(f"    {name}: {path}")

	# Analyze all traces
	metrics = {}
	for name, path in files.items():
		metrics[name] = analyze_trace(path, name)

	# Auto-detect baseline and offload for comparison
	baseline_key = None
	offload_key = None
	pcie_key = None

	for key in metrics:
		if "no_offload" in key or "exp1" in key or "baseline" in key:
			baseline_key = key
		elif "pcie_aware" in key or "exp4" in key:
			pcie_key = key
		elif "offload" in key and "no_compile" not in key and ("exp2" in key or "default" in key):
			offload_key = key

	# Fall back: first two traces
	keys = list(metrics.keys())
	if not baseline_key and len(keys) >= 1:
		baseline_key = keys[0]
	if not offload_key and len(keys) >= 2:
		offload_key = keys[1]

	# Generate comparison report if we have both
	if baseline_key and offload_key:
		log(f"\n  Comparing: {baseline_key} vs {offload_key}")
		compare_report(metrics[baseline_key], metrics[offload_key], output_dir)

	# Compare pcie-aware vs baseline if available
	if baseline_key and pcie_key:
		pcie_output = os.path.join(output_dir, "pcie_aware")
		os.makedirs(pcie_output, exist_ok=True)
		log(f"\n  Comparing: {baseline_key} vs {pcie_key}")
		compare_report(metrics[baseline_key], metrics[pcie_key], pcie_output)

	# Generate individual reports for all traces
	for name, m in metrics.items():
		single_report(m, output_dir)

	log(f"\n{'=' * 64}")
	log(f"  Analysis Complete")
	log(f"{'=' * 64}")
	log(f"  Output: {output_dir}/")
	log(f"  Key file: profiling_report.md")


if __name__ == "__main__":
	main()
