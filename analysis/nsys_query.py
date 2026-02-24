#!/usr/bin/env python3
"""nsys_query.py -- Run SQL queries on nsys-exported SQLite databases.

Replaces sqlite3 CLI usage since it's not available on ACES compute nodes.
Uses Python's built-in sqlite3 module (always available).

NOTE: nsys SQLite export stores kernel/memcpy names as integer foreign keys
into a StringIds table. All queries must JOIN with StringIds to resolve names.

Usage:
    python3 analysis/nsys_query.py <new_db> <old_db> <output_dir>
"""

import csv
import os
import sqlite3
import sys
import time


def log(msg):
	"""Print with immediate flush for SLURM log visibility."""
	print(msg, flush=True)


def create_indices(db_path):
	"""Create indices for fast overlap queries on large nsys databases."""
	label = os.path.basename(db_path)
	log(f"  Creating indices on {label}...")
	t0 = time.time()
	conn = sqlite3.connect(db_path)
	indices = [
		("idx_memcpy_copykind", "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind, deviceId, start, end"),
		("idx_memcpy_copykind_bytes", "CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind, bytes"),
		("idx_kernel_name", "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"),
		("idx_kernel_device_time", "CUPTI_ACTIVITY_KIND_KERNEL", "deviceId, start, end"),
		("idx_stringids_value", "StringIds", "value"),
	]
	for name, table, cols in indices:
		try:
			conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols})")
		except Exception as e:
			log(f"    Warning: index {name}: {e}")
	conn.commit()
	conn.close()
	log(f"  Indices created in {time.time() - t0:.1f}s")


def run_query(db_path, query, output_path, label):
	"""Execute a SQL query and write results as CSV."""
	log(f"\n--- {label} ---")
	t0 = time.time()
	conn = sqlite3.connect(db_path)
	conn.row_factory = sqlite3.Row
	cursor = conn.execute(query)
	rows = cursor.fetchall()
	elapsed = time.time() - t0

	if not rows:
		log(f"  (no results) [{elapsed:.1f}s]")
		conn.close()
		return

	# Write CSV
	keys = rows[0].keys()
	with open(output_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(keys)
		for row in rows:
			writer.writerow(list(row))

	# Print to stdout
	log(",".join(keys))
	for row in rows:
		log(",".join(str(v) for v in row))
	log(f"  -> Saved to {os.path.basename(output_path)} [{elapsed:.1f}s]")
	conn.close()


# ============================================================================
# SQL Queries -- all JOIN with StringIds for name resolution
# ============================================================================

H2D_QUERY = """
SELECT
    'H2D' as direction,
    COUNT(*) as num_transfers,
    SUM(bytes) as total_bytes,
    SUM(bytes) / 1e9 as total_gb,
    SUM(end - start) / 1e9 as total_time_s,
    (SUM(bytes) / 1e9) / (SUM(end - start) / 1e9) as avg_bw_gbps,
    AVG(bytes / 1e6) as avg_size_mb,
    MIN(bytes / 1e6) as min_size_mb,
    MAX(bytes / 1e6) as max_size_mb,
    AVG((end - start) / 1e6) as avg_dur_ms,
    MIN((end - start) / 1e6) as min_dur_ms,
    MAX((end - start) / 1e6) as max_dur_ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE copyKind = 1;
"""

H2D_BUCKETS_QUERY = """
SELECT
    CASE
        WHEN bytes > 100000000 THEN '>100MB'
        WHEN bytes > 1000000 THEN '1-100MB'
        WHEN bytes > 10000 THEN '10KB-1MB'
        ELSE '<10KB'
    END as size_bucket,
    COUNT(*) as count,
    SUM(bytes) / 1e9 as total_gb,
    AVG((bytes / 1e6) / ((end - start) / 1e9)) as avg_bw_mbps,
    SUM(end - start) / 1e9 as total_time_s
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE copyKind = 1
GROUP BY size_bucket
ORDER BY total_gb DESC;
"""

# NCCL query: JOIN with StringIds to resolve kernel names
NCCL_QUERY = """
SELECT
    SUBSTR(s.value, 1, 80) as kernel_name,
    COUNT(*) as count,
    SUM(k.end - k.start) / 1e9 as total_time_s,
    AVG((k.end - k.start) / 1e6) as avg_dur_ms,
    MIN((k.end - k.start) / 1e6) as min_dur_ms,
    MAX((k.end - k.start) / 1e6) as max_dur_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE s.value LIKE '%nccl%'
GROUP BY kernel_name
ORDER BY total_time_s DESC;
"""

# Overlap query: find NCCL kernel string IDs first, then check temporal overlap
# Uses a subquery to resolve NCCL names via StringIds
OVERLAP_QUERY = """
WITH nccl_string_ids AS (
    SELECT id FROM StringIds WHERE value LIKE '%ncclDevKernel%'
),
nccl_windows AS (
    SELECT k.start, k.end, k.deviceId
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE k.demangledName IN (SELECT id FROM nccl_string_ids)
),
h2d_events AS (
    SELECT start, end, bytes, deviceId,
           (bytes / 1e6) / ((end - start) / 1e9) as bw_mbps
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE copyKind = 1 AND bytes > 1000000
)
SELECT
    'during_nccl' as context,
    COUNT(*) as h2d_count,
    SUM(h.bytes) / 1e9 as total_h2d_gb,
    AVG(h.bw_mbps) as avg_h2d_bw_mbps,
    MIN(h.bw_mbps) as min_h2d_bw_mbps,
    MAX(h.bw_mbps) as max_h2d_bw_mbps
FROM h2d_events h
WHERE EXISTS (
    SELECT 1 FROM nccl_windows n
    WHERE n.deviceId = h.deviceId
    AND n.start < h.end
    AND n.end > h.start
)
UNION ALL
SELECT
    'no_nccl' as context,
    COUNT(*) as h2d_count,
    SUM(h.bytes) / 1e9 as total_h2d_gb,
    AVG(h.bw_mbps) as avg_h2d_bw_mbps,
    MIN(h.bw_mbps) as min_h2d_bw_mbps,
    MAX(h.bw_mbps) as max_h2d_bw_mbps
FROM h2d_events h
WHERE NOT EXISTS (
    SELECT 1 FROM nccl_windows n
    WHERE n.deviceId = h.deviceId
    AND n.start < h.end
    AND n.end > h.start
);
"""

H2D_TIMELINE_QUERY = """
SELECT
    deviceId,
    (start / 1000000000) as time_bucket_s,
    COUNT(*) as transfers,
    SUM(bytes) / 1e6 as total_mb,
    AVG((bytes / 1e6) / ((end - start) / 1e9)) as avg_bw_mbps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE copyKind = 1 AND bytes > 100000
GROUP BY deviceId, time_bucket_s
ORDER BY deviceId, time_bucket_s;
"""

# Idle gaps: uses index on (deviceId, start, end) for efficient window function
IDLE_GAPS_QUERY = """
WITH ordered_kernels AS (
    SELECT
        deviceId,
        start,
        end,
        ROW_NUMBER() OVER (PARTITION BY deviceId ORDER BY start) as rn
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
gaps AS (
    SELECT
        a.deviceId,
        (b.start - a.end) / 1e6 as gap_ms,
        a.end as gap_start,
        b.start as gap_end
    FROM ordered_kernels a
    JOIN ordered_kernels b ON a.deviceId = b.deviceId AND b.rn = a.rn + 1
    WHERE (b.start - a.end) > 1000000
)
SELECT
    deviceId,
    COUNT(*) as num_gaps,
    SUM(gap_ms) as total_gap_ms,
    AVG(gap_ms) as avg_gap_ms,
    MIN(gap_ms) as min_gap_ms,
    MAX(gap_ms) as max_gap_ms
FROM gaps
GROUP BY deviceId
ORDER BY deviceId;
"""

# Large gaps with kernel names resolved via StringIds
LARGE_GAPS_QUERY = """
WITH ordered_kernels AS (
    SELECT
        deviceId,
        start,
        end,
        demangledName,
        ROW_NUMBER() OVER (PARTITION BY deviceId ORDER BY start) as rn
    FROM CUPTI_ACTIVITY_KIND_KERNEL
),
gaps AS (
    SELECT
        a.deviceId,
        (b.start - a.end) / 1e6 as gap_ms,
        a.demangledName as before_name_id,
        b.demangledName as after_name_id
    FROM ordered_kernels a
    JOIN ordered_kernels b ON a.deviceId = b.deviceId AND b.rn = a.rn + 1
    WHERE (b.start - a.end) > 100000000
)
SELECT
    g.deviceId,
    g.gap_ms,
    SUBSTR(s1.value, 1, 60) as before_kernel,
    SUBSTR(s2.value, 1, 60) as after_kernel
FROM gaps g
JOIN StringIds s1 ON g.before_name_id = s1.id
JOIN StringIds s2 ON g.after_name_id = s2.id
WHERE g.deviceId = 0
ORDER BY g.gap_ms DESC
LIMIT 20;
"""


def main():
	if len(sys.argv) != 4:
		log(f"Usage: {sys.argv[0]} <new_db> <old_db> <output_dir>")
		sys.exit(1)

	new_db = sys.argv[1]
	old_db = sys.argv[2]
	output_dir = sys.argv[3]

	for db in [new_db, old_db]:
		if not os.path.exists(db):
			log(f"ERROR: Database not found: {db}")
			sys.exit(1)

	os.makedirs(output_dir, exist_ok=True)

	# Verify StringIds table exists and has NCCL entries
	log("Verifying database schema...")
	for db in [new_db, old_db]:
		conn = sqlite3.connect(db)
		nccl_count = conn.execute(
			"SELECT COUNT(*) FROM StringIds WHERE value LIKE '%ncclDevKernel%'"
		).fetchone()[0]
		kernel_count = conn.execute(
			"SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
			"WHERE k.demangledName IN (SELECT id FROM StringIds WHERE value LIKE '%ncclDevKernel%')"
		).fetchone()[0]
		log(f"  {os.path.basename(db)}: {nccl_count} NCCL string IDs, {kernel_count} NCCL kernel events")
		conn.close()

	# Create indices for fast overlap/range queries
	log("\nCreating indices (one-time, speeds up overlap queries)...")
	for db in [new_db, old_db]:
		create_indices(db)
	log("")

	queries = [
		("QUERY 1: H2D MEMCPY BANDWIDTH", H2D_QUERY, "h2d"),
		("QUERY 1b: H2D BY SIZE BUCKET", H2D_BUCKETS_QUERY, "h2d_buckets"),
		("QUERY 2: NCCL KERNEL TIMING", NCCL_QUERY, "nccl"),
		(
			"QUERY 3: H2D BANDWIDTH DURING vs WITHOUT NCCL\n"
			"  (KEY question: does NCCL degrade H2D bandwidth?)",
			OVERLAP_QUERY,
			"overlap",
		),
		("QUERY 4: H2D BANDWIDTH TIMELINE (per-second)", H2D_TIMELINE_QUERY, "h2d_timeline"),
		("QUERY 5: GPU IDLE GAPS (>1ms between kernels)", IDLE_GAPS_QUERY, "idle_gaps"),
		("QUERY 5b: LARGEST GPU IDLE GAPS (>100ms, device 0)", LARGE_GAPS_QUERY, "large_gaps"),
	]

	total_t0 = time.time()
	for title, query, prefix in queries:
		log(f"\n{'=' * 64}")
		log(f"  {title}")
		log(f"{'=' * 64}")
		run_query(new_db, query, os.path.join(output_dir, f"{prefix}_new.csv"), "New Offload")
		run_query(old_db, query, os.path.join(output_dir, f"{prefix}_old.csv"), "Old Offload")

	log(f"\n{'=' * 64}")
	log("  ANALYSIS COMPLETE")
	log(f"{'=' * 64}")
	log(f"\n  Total query time: {time.time() - total_t0:.1f}s")
	log(f"  Output directory: {output_dir}/")
	log("  Key files:")
	log("    overlap_new.csv / overlap_old.csv  -- H2D BW during vs without NCCL")
	log("    h2d_new.csv / h2d_old.csv          -- Overall H2D statistics")
	log("    nccl_new.csv / nccl_old.csv        -- NCCL kernel timing")
	log("    idle_gaps_new.csv / idle_gaps_old.csv -- GPU idle gaps")


if __name__ == "__main__":
	main()
