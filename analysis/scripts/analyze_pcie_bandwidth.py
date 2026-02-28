"""Analyze PCIe bandwidth from torch.profiler traces (streaming parser).

Handles truncated gzip files and multi-GB decompressed traces by
parsing events line-by-line without loading the full JSON into memory.
"""

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def stream_parse_events(trace_path, event_filter=None):
	"""Parse trace events line-by-line from potentially truncated gzip.

	Chrome Trace JSON format has events as objects in a JSON array.
	We extract individual event objects using regex matching.
	"""
	print(f"Streaming parse: {trace_path.name} ({trace_path.stat().st_size / 1e6:.0f} MB)...")

	# Pattern to match complete JSON objects containing Memcpy or nccl
	memcpy_events = []
	nccl_events = []
	device_props = []
	total_lines = 0
	matched_lines = 0

	with gzip.open(trace_path, "rt", errors="replace") as f:
		buffer = ""
		brace_depth = 0
		in_event = False
		event_start = 0

		try:
			for line in f:
				total_lines += 1

				# Quick filter: skip lines that can't contain what we need
				line_lower = line.lower()

				# Extract device properties from early in the file
				if '"deviceproperties"' in line_lower and not device_props:
					# Will get from header parsing
					pass

				# Look for memcpy or nccl events
				has_memcpy = "memcpy" in line_lower or "Memcpy" in line
				has_nccl = "nccl" in line_lower
				has_useful = has_memcpy or has_nccl

				buffer += line

				# Try to extract complete JSON objects from buffer
				# when we have potentially useful events
				if has_useful or (len(buffer) > 10000):
					# Try to find complete JSON objects in buffer
					# Look for patterns like {"cat":..., "name":..., ...}
					pos = 0
					while pos < len(buffer):
						# Find start of object
						start = buffer.find("{", pos)
						if start < 0:
							break

						# Find matching closing brace
						depth = 0
						i = start
						found_end = False
						while i < len(buffer):
							if buffer[i] == "{":
								depth += 1
							elif buffer[i] == "}":
								depth -= 1
								if depth == 0:
									found_end = True
									break
							i += 1

						if not found_end:
							break  # incomplete object, keep in buffer

						obj_str = buffer[start:i+1]
						pos = i + 1

						# Only parse if it's relevant
						obj_lower = obj_str.lower()
						if "memcpy" not in obj_lower and "nccl" not in obj_lower:
							continue

						try:
							ev = json.loads(obj_str)
						except json.JSONDecodeError:
							continue

						name = ev.get("name", "")
						dur = ev.get("dur")

						if dur is None or dur <= 0:
							continue

						args = ev.get("args", {})
						matched_lines += 1

						if "memcpy" in name.lower() or "Memcpy" in name:
							memcpy_events.append({
								"name": name,
								"cat": ev.get("cat", ""),
								"ts": ev.get("ts", 0),
								"dur": dur,
								"bytes": args.get("bytes", args.get("Bytes", 0)),
								"src_kind": args.get("src_kind", args.get("Src Kind", "")),
								"dst_kind": args.get("dst_kind", args.get("Dst Kind", "")),
								"stream": args.get("stream", ""),
							})
						elif "nccl" in name.lower():
							nccl_events.append({
								"name": name,
								"cat": ev.get("cat", ""),
								"ts": ev.get("ts", 0),
								"dur": dur,
								"bytes": args.get("bytes", args.get("Bytes", 0)),
							})

					# Keep only unprocessed remainder in buffer
					buffer = buffer[pos:] if pos > 0 else ""

				# Prevent buffer from growing too large
				if len(buffer) > 100000:
					buffer = buffer[-10000:]

				# Progress every 1M lines
				if total_lines % 1_000_000 == 0:
					print(f"  ... {total_lines/1e6:.0f}M lines, {len(memcpy_events)} memcpy, {len(nccl_events)} nccl")

		except EOFError:
			print(f"  Hit truncation at line {total_lines:,}")

	print(f"  Parsed {total_lines:,} lines")
	print(f"  Memcpy events: {len(memcpy_events):,}")
	print(f"  NCCL events: {len(nccl_events):,}")

	return memcpy_events, nccl_events


def classify_direction(ev):
	"""Classify memcpy direction from event name or src/dst kind."""
	name = ev["name"]

	if "HtoD" in name or "htod" in name.lower():
		return "H2D"
	elif "DtoH" in name or "dtoh" in name.lower():
		return "D2H"
	elif "DtoD" in name or "dtod" in name.lower():
		return "D2D"

	src = str(ev.get("src_kind", "")).lower()
	dst = str(ev.get("dst_kind", "")).lower()

	if "host" in src or "pageable" in src or "pinned" in src:
		if "device" in dst:
			return "H2D"
	if "device" in src:
		if "host" in dst or "pageable" in dst or "pinned" in dst:
			return "D2H"
		if "device" in dst:
			return "D2D"

	return "unknown"


def compute_bandwidth_stats(memcpy_events):
	"""Compute bandwidth statistics per direction."""
	by_direction = defaultdict(list)

	for ev in memcpy_events:
		direction = classify_direction(ev)
		nbytes = ev.get("bytes", 0)
		dur_us = ev["dur"]

		if nbytes > 0 and dur_us > 0:
			bw_gbps = (nbytes / 1e9) / (dur_us / 1e6)  # GB/s
			by_direction[direction].append({
				"bytes": nbytes,
				"dur_us": dur_us,
				"bw_gbps": bw_gbps,
				"ts": ev["ts"],
				"name": ev["name"],
				"stream": ev.get("stream", ""),
			})

	return by_direction


def print_bandwidth_report(by_direction, label):
	"""Print bandwidth analysis report."""
	# H100 PCIe Gen5 x16 theoretical peaks
	PCIE_GEN5_X16_PEAK = 63.0  # GB/s per direction (unidirectional)

	print(f"\n{'='*70}")
	print(f"PCIe Bandwidth Analysis: {label}")
	print(f"{'='*70}")

	for direction in ["H2D", "D2H", "D2D", "unknown"]:
		transfers = by_direction.get(direction, [])
		if not transfers:
			continue

		bytes_list = [t["bytes"] for t in transfers]
		bw_list = [t["bw_gbps"] for t in transfers]
		dur_list = [t["dur_us"] for t in transfers]

		total_bytes = sum(bytes_list)
		total_dur = sum(dur_list)
		avg_bw = (total_bytes / 1e9) / (total_dur / 1e6) if total_dur > 0 else 0

		print(f"\n--- {direction} Transfers ---")
		print(f"  Count:         {len(transfers):,}")
		print(f"  Total data:    {total_bytes / 1e9:.2f} GB")
		print(f"  Total time:    {total_dur / 1e6:.3f} s")
		print(f"  Weighted avg BW: {avg_bw:.2f} GB/s")

		if bw_list:
			bw_sorted = sorted(bw_list)
			n = len(bw_sorted)
			print(f"  Median BW:     {bw_sorted[n//2]:.2f} GB/s")
			print(f"  P10 BW:        {bw_sorted[int(n*0.1)]:.2f} GB/s")
			print(f"  P90 BW:        {bw_sorted[int(n*0.9)]:.2f} GB/s")
			print(f"  Max BW:        {max(bw_list):.2f} GB/s")
			print(f"  Min BW:        {min(bw_list):.2f} GB/s")

		if direction in ["H2D", "D2H"]:
			print(f"  PCIe peak:     {PCIE_GEN5_X16_PEAK:.1f} GB/s (Gen5 x16)")
			print(f"  Utilization:   {avg_bw / PCIE_GEN5_X16_PEAK * 100:.1f}%")

		# Size distribution
		size_buckets = [
			("<1KB", 0, 1024),
			("1KB-1MB", 1024, 1e6),
			("1MB-100MB", 1e6, 1e8),
			("100MB-1GB", 1e8, 1e9),
			(">1GB", 1e9, float("inf")),
		]
		print(f"\n  Size distribution:")
		for bucket_name, lo, hi in size_buckets:
			bucket_xfers = [t for t in transfers if lo <= t["bytes"] < hi]
			if bucket_xfers:
				bucket_bws = [t["bw_gbps"] for t in bucket_xfers]
				bucket_bytes = sum(t["bytes"] for t in bucket_xfers)
				avg = sum(bucket_bws) / len(bucket_bws)
				print(f"    {bucket_name:>12}: {len(bucket_xfers):>6} xfers, "
					  f"{bucket_bytes/1e9:.2f} GB total, avg BW {avg:.2f} GB/s")

	# Large H2D transfers (likely layer offload)
	h2d = by_direction.get("H2D", [])
	large_h2d = [t for t in h2d if t["bytes"] > 100e6]
	if large_h2d:
		print(f"\n--- Large H2D Transfers (>100MB = likely layer offload) ---")
		print(f"  Count: {len(large_h2d)}")
		bws = [t["bw_gbps"] for t in large_h2d]
		sizes = [t["bytes"] / 1e6 for t in large_h2d]
		print(f"  Avg size:      {sum(sizes)/len(sizes):.1f} MB")
		print(f"  Avg BW:        {sum(bws)/len(bws):.2f} GB/s")
		print(f"  Min BW:        {min(bws):.2f} GB/s")
		print(f"  Max BW:        {max(bws):.2f} GB/s")
		print(f"  Utilization:   {sum(bws)/len(bws)/PCIE_GEN5_X16_PEAK*100:.1f}%")

		# Show individual large transfers chronologically
		large_h2d_sorted = sorted(large_h2d, key=lambda x: x["ts"])
		print(f"\n  Chronological large H2D transfers:")
		for i, t in enumerate(large_h2d_sorted[:30]):  # show first 30
			print(f"    [{i:>3}] {t['bytes']/1e6:>7.1f} MB, "
				  f"{t['dur_us']/1e6:>6.3f} s, "
				  f"{t['bw_gbps']:>6.2f} GB/s, "
				  f"stream={t['stream']}")


def analyze_contention(by_direction, nccl_events):
	"""Check if H2D transfers overlap with NCCL ops (PCIe contention)."""
	h2d = by_direction.get("H2D", [])
	large_h2d = [t for t in h2d if t["bytes"] > 10e6]

	if not large_h2d or not nccl_events:
		print(f"\n--- Contention Analysis: Insufficient data "
			  f"({len(large_h2d)} large H2D, {len(nccl_events)} NCCL) ---")
		return

	print(f"\n{'='*70}")
	print("Contention Analysis: H2D Offload vs NCCL (Ulysses All-to-All)")
	print(f"{'='*70}")

	PCIE_GEN5_X16_PEAK = 63.0

	# Sort by timestamp
	h2d_sorted = sorted(large_h2d, key=lambda x: x["ts"])
	nccl_sorted = sorted(nccl_events, key=lambda x: x["ts"])

	overlapping = 0
	non_overlapping = 0
	overlap_bws = []
	clean_bws = []

	for h in h2d_sorted:
		h_start = h["ts"]
		h_end = h["ts"] + h["dur_us"]

		has_overlap = False
		for n in nccl_sorted:
			n_start = n["ts"]
			n_end = n["ts"] + n["dur"]

			if n_start < h_end and n_end > h_start:
				has_overlap = True
				break

		if has_overlap:
			overlapping += 1
			overlap_bws.append(h["bw_gbps"])
		else:
			non_overlapping += 1
			clean_bws.append(h["bw_gbps"])

	print(f"  Large H2D transfers (>10MB): {len(large_h2d)}")
	print(f"  Overlapping with NCCL:       {overlapping}")
	print(f"  Clean (no overlap):          {non_overlapping}")

	if overlap_bws:
		avg_o = sum(overlap_bws) / len(overlap_bws)
		print(f"\n  During NCCL overlap:")
		print(f"    Avg BW:      {avg_o:.2f} GB/s ({avg_o/PCIE_GEN5_X16_PEAK*100:.1f}% of peak)")
	if clean_bws:
		avg_c = sum(clean_bws) / len(clean_bws)
		print(f"\n  Without NCCL overlap:")
		print(f"    Avg BW:      {avg_c:.2f} GB/s ({avg_c/PCIE_GEN5_X16_PEAK*100:.1f}% of peak)")
	if overlap_bws and clean_bws:
		avg_o = sum(overlap_bws) / len(overlap_bws)
		avg_c = sum(clean_bws) / len(clean_bws)
		if avg_c > 0:
			degradation = (1 - avg_o / avg_c) * 100
			print(f"\n  >>> BW degradation from NCCL contention: {degradation:.1f}%")
			if degradation > 20:
				print(f"  >>> SIGNIFICANT contention -- PCIe fabric shared between "
					  f"CPU-GPU offload and GPU-GPU NCCL")
			elif degradation > 5:
				print(f"  >>> MODERATE contention")
			else:
				print(f"  >>> MINIMAL contention (transfers may be serialized)")

	# NCCL bandwidth stats
	nccl_with_bytes = [n for n in nccl_events if n.get("bytes", 0) > 0]
	if nccl_with_bytes:
		print(f"\n--- NCCL Traffic Summary ---")
		total_nccl_bytes = sum(n["bytes"] for n in nccl_with_bytes)
		total_nccl_dur = sum(n["dur"] for n in nccl_with_bytes)
		print(f"  Total NCCL data:   {total_nccl_bytes / 1e9:.2f} GB")
		print(f"  Total NCCL time:   {total_nccl_dur / 1e6:.3f} s")
		if total_nccl_dur > 0:
			nccl_bw = (total_nccl_bytes / 1e9) / (total_nccl_dur / 1e6)
			print(f"  Avg NCCL BW:       {nccl_bw:.2f} GB/s")


def main():
	traces_dir = Path("results/traces")

	for trace_file in sorted(traces_dir.glob("*.trace.json.gz")):
		label = trace_file.stem.replace(".trace.json", "")
		memcpy_events, nccl_events = stream_parse_events(trace_file)

		if not memcpy_events:
			print(f"  WARNING: No memcpy events found -- trying alternate event names")
			continue

		by_direction = compute_bandwidth_stats(memcpy_events)
		print_bandwidth_report(by_direction, label)
		analyze_contention(by_direction, nccl_events)
		print()


if __name__ == "__main__":
	main()
