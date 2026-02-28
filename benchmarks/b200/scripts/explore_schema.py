#!/usr/bin/env python3
"""Explore nsys SQLite schema for both trace files."""
import sqlite3
import os

DB_BASELINE = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp1_no_offload_20260227_233628.sqlite"
DB_OFFLOAD = "/Users/dannyliu/research_work/b200_workspace/results/nsys/exp2_offload_default_20260227_234442.sqlite"

def explore_db(path, label):
	print(f"\n{'='*80}")
	print(f"DATABASE: {label}")
	print(f"Path: {path}")
	print(f"Size: {os.path.getsize(path) / 1e6:.1f} MB")
	print(f"{'='*80}")

	conn = sqlite3.connect(path)
	cur = conn.cursor()

	# List all tables
	cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
	tables = [r[0] for r in cur.fetchall()]
	print(f"\nTables ({len(tables)}):")
	for t in tables:
		cur.execute(f"SELECT COUNT(*) FROM [{t}]")
		count = cur.fetchone()[0]
		print(f"  {t}: {count} rows")

	# Key tables schema
	key_tables = [
		"CUPTI_ACTIVITY_KIND_KERNEL",
		"CUPTI_ACTIVITY_KIND_MEMCPY",
		"CUPTI_ACTIVITY_KIND_RUNTIME",
		"CUPTI_ACTIVITY_KIND_MEMSET",
		"StringIds",
		"NVTX_EVENTS",
	]
	for t in key_tables:
		if t in tables:
			cur.execute(f"PRAGMA table_info([{t}])")
			cols = cur.fetchall()
			print(f"\n  Schema: {t}")
			for c in cols:
				print(f"    {c[1]:30s} {c[2]:15s} {'PK' if c[5] else ''}")

	# Sample kernels
	if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
		print(f"\n  Sample kernels (first 10):")
		cur.execute("""
			SELECT k.start, k.end, k.deviceId, k.streamId, s.value
			FROM CUPTI_ACTIVITY_KIND_KERNEL k
			JOIN StringIds s ON k.demangledName = s.id
			ORDER BY k.start
			LIMIT 10
		""")
		for r in cur.fetchall():
			dur_us = (r[1] - r[0]) / 1000
			print(f"    dev={r[2]} stream={r[3]} dur={dur_us:.1f}us  {r[4][:100]}")

	# Sample memcpy
	if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
		print(f"\n  Sample memcpy (first 10):")
		cur.execute("""
			SELECT start, end, bytes, copyKind, deviceId, streamId
			FROM CUPTI_ACTIVITY_KIND_MEMCPY
			ORDER BY start
			LIMIT 10
		""")
		for r in cur.fetchall():
			dur_us = (r[1] - r[0]) / 1000
			bw_gbs = r[2] / (r[1] - r[0]) if (r[1] - r[0]) > 0 else 0
			kind_str = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}.get(r[3], f"kind={r[3]}")
			print(f"    dev={r[4]} stream={r[5]} dur={dur_us:.1f}us bytes={r[2]} bw={bw_gbs:.2f}GB/s {kind_str}")

	# NCCL kernel sampling
	if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
		print(f"\n  NCCL kernel count:")
		cur.execute("""
			SELECT COUNT(*)
			FROM CUPTI_ACTIVITY_KIND_KERNEL k
			JOIN StringIds s ON k.demangledName = s.id
			WHERE s.value LIKE '%nccl%'
		""")
		print(f"    {cur.fetchone()[0]}")

		print(f"\n  NCCL kernel types (top 5):")
		cur.execute("""
			SELECT s.value, COUNT(*), SUM(k.end - k.start)/1e9 as total_s
			FROM CUPTI_ACTIVITY_KIND_KERNEL k
			JOIN StringIds s ON k.demangledName = s.id
			WHERE s.value LIKE '%nccl%'
			GROUP BY s.value
			ORDER BY total_s DESC
			LIMIT 5
		""")
		for r in cur.fetchall():
			print(f"    count={r[1]:6d} total={r[2]:.3f}s  {r[0][:100]}")

	# Check for NVTX markers
	if "NVTX_EVENTS" in tables:
		print(f"\n  NVTX event count:")
		cur.execute("SELECT COUNT(*) FROM NVTX_EVENTS")
		print(f"    {cur.fetchone()[0]}")
		print(f"\n  Sample NVTX (first 10):")
		cur.execute("SELECT * FROM NVTX_EVENTS LIMIT 10")
		cols = [d[0] for d in cur.description]
		print(f"    Columns: {cols}")
		for r in cur.fetchall():
			print(f"    {r}")

	# Unique devices
	if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
		cur.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")
		devices = [r[0] for r in cur.fetchall()]
		print(f"\n  Unique devices (kernels): {devices}")

	# Unique streams on device 0
	if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
		cur.execute("""
			SELECT DISTINCT streamId, COUNT(*) as cnt
			FROM CUPTI_ACTIVITY_KIND_KERNEL
			WHERE deviceId = 0
			GROUP BY streamId
			ORDER BY cnt DESC
		""")
		print(f"\n  Streams on device 0 (kernel count):")
		for r in cur.fetchall():
			print(f"    stream={r[0]:3d}: {r[1]} kernels")

	# Memcpy streams on device 0
	if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
		cur.execute("""
			SELECT DISTINCT streamId, copyKind, COUNT(*) as cnt, SUM(bytes)/1e9 as gb
			FROM CUPTI_ACTIVITY_KIND_MEMCPY
			WHERE deviceId = 0
			GROUP BY streamId, copyKind
			ORDER BY gb DESC
		""")
		kind_map = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
		print(f"\n  Memcpy streams on device 0:")
		for r in cur.fetchall():
			kind_str = kind_map.get(r[1], f"kind={r[1]}")
			print(f"    stream={r[0]:3d} {kind_str}: {r[2]} ops, {r[3]:.2f} GB")

	# Time range
	if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
		cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=0")
		mn, mx = cur.fetchone()
		print(f"\n  Time range (device 0): {(mx-mn)/1e9:.2f}s")

	conn.close()

explore_db(DB_BASELINE, "Baseline (no offload)")
explore_db(DB_OFFLOAD, "Offload default")
