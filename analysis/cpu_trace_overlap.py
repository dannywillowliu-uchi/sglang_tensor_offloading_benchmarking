"""CPU-side trace overlap analysis for new vs old offload.

Stream-parses torch.profiler Chrome Trace JSON (.trace.json.gz) to extract:
1. aten::copy_ events (CPU-side H2D timing proxy)
2. NCCL-related CPU events
3. CPU-side temporal overlap between large copies and NCCL
4. Blocking vs async copy patterns
5. Per-denoising-step copy patterns

IMPORTANT CAVEATS:
- GPU events are TRUNCATED in these traces (documented limitation)
- CPU timestamps for async copies reflect LAUNCH overhead, not transfer time
- Blocking copies show actual transfer + sync time
- This analysis is INDICATIVE, not definitive. Use nsys for ground truth.

Runtime: ~2-5 minutes per trace (streaming parser on ~400MB gzipped files)

Usage:
    python analysis/cpu_trace_overlap.py [--traces-dir results/traces]
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

DTYPE_SIZES = {
	"float": 4, "float32": 4, "float64": 8, "double": 8,
	"half": 2, "float16": 2, "bfloat16": 2, "c10::BFloat16": 2,
	"c10::Half": 2, "int": 4, "int32": 4, "int64": 8, "long": 8,
	"int16": 2, "short": 2, "int8": 1, "byte": 1,
	"bool": 1, "uint8": 1, "signed char": 1,
}

CHUNK_SIZE = 1024 * 1024
MAX_EVENTS = 15_000_000


def stream_events(path):
	"""Stream-parse events from Chrome Trace JSON (possibly truncated gzip)."""
	count = 0
	buf = ""
	found_array = False

	try:
		with gzip.open(path, "rt", encoding="utf-8") as f:
			while True:
				chunk = f.read(CHUNK_SIZE)
				if not chunk:
					break
				buf += chunk
				if not found_array:
					idx = buf.find('"traceEvents"')
					if idx < 0:
						buf = buf[-100:]
						continue
					bracket_idx = buf.find("[", idx)
					if bracket_idx < 0:
						continue
					buf = buf[bracket_idx + 1:]
					found_array = True

				while True:
					obj_start = buf.find("{")
					if obj_start < 0:
						break
					depth = 0
					in_string = False
					escape = False
					obj_end = -1
					for i in range(obj_start, len(buf)):
						c = buf[i]
						if escape:
							escape = False
							continue
						if c == "\\":
							if in_string:
								escape = True
							continue
						if c == '"':
							in_string = not in_string
							continue
						if in_string:
							continue
						if c == "{":
							depth += 1
						elif c == "}":
							depth -= 1
							if depth == 0:
								obj_end = i + 1
								break
					if obj_end < 0:
						break
					obj_str = buf[obj_start:obj_end]
					buf = buf[obj_end:]
					try:
						event = json.loads(obj_str)
						if isinstance(event, dict) and "ph" in event:
							count += 1
							yield event
							if count >= MAX_EVENTS:
								return
					except json.JSONDecodeError:
						pass
				if len(buf) > 10 * CHUNK_SIZE:
					buf = buf[-CHUNK_SIZE:]
	except EOFError:
		pass

	print(f"  Parsed {count:,} events")


def estimate_bytes(dims, dtype_str):
	if not dims or not isinstance(dims, list):
		return 0
	try:
		numel = 1
		for d in dims:
			numel *= d
		return numel * DTYPE_SIZES.get(dtype_str.lower() if dtype_str else "float", 4)
	except (TypeError, ValueError):
		return 0


def section(title):
	print(f"\n{'=' * 72}")
	print(f"  {title}")
	print(f"{'=' * 72}")


def subsection(title):
	print(f"\n--- {title} ---")


def analyze_trace(trace_path):
	label = Path(trace_path).stem.replace(".trace.json", "")
	section(f"CPU-Side Trace Analysis: {label}")
	print(f"  File: {trace_path}")

	copy_events = []
	nccl_events = []
	denoising_events = []
	total = 0

	print("  Parsing events...")
	for ev in stream_events(trace_path):
		total += 1
		name = ev.get("name", "")
		dur = ev.get("dur", 0)
		args = ev.get("args", {})
		ts = ev.get("ts", 0)

		if total % 3_000_000 == 0:
			sys.stdout.write(f"\r  ... {total / 1e6:.0f}M events, "
							 f"{len(copy_events)} copies, {len(nccl_events)} nccl")
			sys.stdout.flush()

		# Collect aten::copy_ events
		if name == "aten::copy_" and dur > 0:
			concrete = args.get("Concrete Inputs", [])
			non_blocking = len(concrete) >= 3 and concrete[2] in ("True", True, "1")
			input_dims = args.get("Input Dims", [])
			input_types = args.get("Input type", [])
			src_dims = input_dims[0] if len(input_dims) > 0 else []
			src_type = input_types[0] if len(input_types) > 0 else "float"
			est_bytes = estimate_bytes(src_dims, src_type)

			copy_events.append({
				"ts": ts,
				"dur": dur,
				"end": ts + dur,
				"non_blocking": non_blocking,
				"est_bytes": est_bytes,
				"dtype": src_type,
			})

		# Collect NCCL events
		elif ("nccl" in name.lower() or "record_param_comms" in name.lower()
			  or "all_to_all" in name.lower()) and dur > 0:
			nccl_events.append({
				"ts": ts,
				"dur": dur,
				"end": ts + dur,
				"name": name,
			})

		# Collect denoising step markers
		elif "DenoisingStage" in name or "denoising_step" in name.lower():
			denoising_events.append({
				"ts": ts,
				"dur": dur,
				"name": name,
			})

	print(f"\r  Total: {total:,} events")
	print(f"  aten::copy_: {len(copy_events):,}")
	print(f"  NCCL-related: {len(nccl_events):,}")
	print(f"  Denoising markers: {len(denoising_events):,}")

	if not copy_events:
		print("  WARNING: No copy events found. Trace may be too truncated.")
		return label, {}

	# ======================================================================
	# Analysis 1: Copy Classification
	# ======================================================================
	subsection("1. Copy Event Classification")

	blocking = [e for e in copy_events if not e["non_blocking"]]
	async_c = [e for e in copy_events if e["non_blocking"]]

	# Size buckets
	large = [e for e in copy_events if e["est_bytes"] > 100e6]
	medium = [e for e in copy_events if 1e6 <= e["est_bytes"] <= 100e6]
	small = [e for e in copy_events if e["est_bytes"] < 1e6]

	print(f"  Blocking: {len(blocking)}  Async: {len(async_c)}")
	print(f"  >100MB: {len(large)}  1-100MB: {len(medium)}  <1MB: {len(small)}")

	for group_name, group in [("Blocking", blocking), ("Async (non_blocking=True)", async_c)]:
		if not group:
			continue
		total_bytes = sum(e["est_bytes"] for e in group)
		total_dur = sum(e["dur"] for e in group)
		large_g = [e for e in group if e["est_bytes"] > 100e6]
		print(f"\n  {group_name}: {len(group)} events")
		print(f"    Total data: {total_bytes/1e9:.2f} GB")
		print(f"    Total CPU time: {total_dur/1e6:.3f} s")
		print(f"    Large (>100MB): {len(large_g)}")
		if large_g:
			large_bytes = sum(e["est_bytes"] for e in large_g)
			large_dur = sum(e["dur"] for e in large_g)
			print(f"    Large data: {large_bytes/1e9:.2f} GB, CPU time: {large_dur/1e6:.3f} s")
			if not group[0]["non_blocking"]:
				# Blocking -- can estimate bandwidth
				bw = (large_bytes / 1e9) / (large_dur / 1e6) if large_dur > 0 else 0
				print(f"    Effective BW: {bw:.2f} GB/s (CPU-side, includes sync overhead)")

	# ======================================================================
	# Analysis 2: CPU-side temporal overlap between copies and NCCL
	# ======================================================================
	if nccl_events and copy_events:
		subsection("2. CPU-Side Temporal Overlap: Copies vs NCCL")
		print("  CAVEAT: CPU timestamps show launch overlap, not GPU-side overlap.")
		print("  Async copies: CPU time = launch overhead only (~few us).")
		print("  NCCL events: CPU time includes launch + possible wait.")
		print()

		# Focus on large copies (>100MB)
		large_copies = sorted([e for e in copy_events if e["est_bytes"] > 100e6],
							  key=lambda x: x["ts"])
		nccl_sorted = sorted(nccl_events, key=lambda x: x["ts"])

		overlapping = []
		clean = []

		for c in large_copies:
			has_overlap = any(
				n["ts"] < c["end"] and n["end"] > c["ts"]
				for n in nccl_sorted
			)
			if has_overlap:
				overlapping.append(c)
			else:
				clean.append(c)

		print(f"  Large copies (>100MB): {len(large_copies)}")
		print(f"    Overlapping with NCCL (CPU timeline): {len(overlapping)}")
		print(f"    Clean (no overlap): {len(clean)}")

		# Bandwidth comparison for blocking copies only
		overlap_blocking = [e for e in overlapping if not e["non_blocking"] and e["est_bytes"] > 0]
		clean_blocking = [e for e in clean if not e["non_blocking"] and e["est_bytes"] > 0]

		if overlap_blocking:
			ob_bytes = sum(e["est_bytes"] for e in overlap_blocking)
			ob_dur = sum(e["dur"] for e in overlap_blocking)
			ob_bw = (ob_bytes / 1e9) / (ob_dur / 1e6) if ob_dur > 0 else 0
			print(f"\n  Blocking copies during NCCL: {len(overlap_blocking)}")
			print(f"    Effective BW: {ob_bw:.2f} GB/s")

		if clean_blocking:
			cb_bytes = sum(e["est_bytes"] for e in clean_blocking)
			cb_dur = sum(e["dur"] for e in clean_blocking)
			cb_bw = (cb_bytes / 1e9) / (cb_dur / 1e6) if cb_dur > 0 else 0
			print(f"  Blocking copies without NCCL: {len(clean_blocking)}")
			print(f"    Effective BW: {cb_bw:.2f} GB/s")

		if overlap_blocking and clean_blocking:
			degradation = (1 - ob_bw / cb_bw) * 100 if cb_bw > 0 else 0
			print(f"\n  Bandwidth degradation during NCCL: {degradation:+.1f}%")
			if abs(degradation) < 5:
				print("  -> Minimal degradation at CPU level (expected for async)")
			else:
				print(f"  -> {abs(degradation):.0f}% degradation suggests real contention")

	# ======================================================================
	# Analysis 3: Copy patterns over time
	# ======================================================================
	if copy_events:
		subsection("3. Copy Patterns Over Time")

		# Bucket copies into 10s windows
		if copy_events:
			min_ts = min(e["ts"] for e in copy_events)
			max_ts = max(e["ts"] for e in copy_events)
			duration_s = (max_ts - min_ts) / 1e6

			BUCKET_SIZE = 10_000_000  # 10s in us
			buckets = defaultdict(lambda: {"count": 0, "bytes": 0, "blocking": 0, "async": 0})

			for e in copy_events:
				bucket_idx = (e["ts"] - min_ts) // BUCKET_SIZE
				b = buckets[bucket_idx]
				b["count"] += 1
				b["bytes"] += e["est_bytes"]
				if e["non_blocking"]:
					b["async"] += 1
				else:
					b["blocking"] += 1

			print(f"  Time span: {duration_s:.0f}s")
			print(f"  Showing 10s time windows with copy activity:")
			print(f"  {'Time (s)':<12s} {'Count':>6s} {'GB':>8s} {'Block':>6s} {'Async':>6s}")
			print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")

			for idx in sorted(buckets.keys()):
				b = buckets[idx]
				t = idx * 10
				if b["count"] > 0:
					print(f"  {t:>6.0f}-{t+10:<5.0f} {b['count']:>6d} {b['bytes']/1e9:>8.2f} "
						  f"{b['blocking']:>6d} {b['async']:>6d}")

	# ======================================================================
	# Analysis 4: Large copy duration distribution
	# ======================================================================
	if large:
		subsection("4. Large Copy Duration Distribution (>100MB)")

		blocking_large = sorted([e for e in large if not e["non_blocking"]],
								key=lambda x: x["dur"])
		async_large = sorted([e for e in large if e["non_blocking"]],
							 key=lambda x: x["dur"])

		for name, group in [("Blocking", blocking_large), ("Async", async_large)]:
			if not group:
				continue
			durs = [e["dur"] for e in group]
			n = len(durs)
			print(f"\n  {name} large copies ({n} events):")
			print(f"    Min:    {durs[0]/1e3:.1f} ms")
			if n > 4:
				print(f"    P10:    {durs[int(n*0.1)]/1e3:.1f} ms")
			print(f"    Median: {durs[n//2]/1e3:.1f} ms")
			if n > 4:
				print(f"    P90:    {durs[int(n*0.9)]/1e3:.1f} ms")
			print(f"    Max:    {durs[-1]/1e3:.1f} ms")

			if not group[0]["non_blocking"]:
				# Effective bandwidth for blocking
				bws = [(e["est_bytes"] / 1e9) / (e["dur"] / 1e6)
					   for e in group if e["dur"] > 0]
				if bws:
					bws.sort()
					print(f"    BW min:    {bws[0]:.2f} GB/s")
					print(f"    BW median: {bws[len(bws)//2]:.2f} GB/s")
					print(f"    BW max:    {bws[-1]:.2f} GB/s")

	results = {
		"label": label,
		"total_events": total,
		"copy_count": len(copy_events),
		"nccl_count": len(nccl_events),
		"blocking_count": len(blocking),
		"async_count": len(async_c),
		"large_count": len(large),
	}
	return label, results


def main():
	parser = argparse.ArgumentParser(description="CPU-side trace overlap analysis")
	parser.add_argument("--traces-dir", default="results/traces",
						help="Directory containing .trace.json.gz files")
	parser.add_argument("--files", nargs="*",
						help="Specific trace files to analyze (default: all in traces-dir)")
	args = parser.parse_args()

	print("=" * 72)
	print("  CPU-SIDE TRACE OVERLAP ANALYSIS")
	print("  CAVEAT: GPU events truncated. CPU timestamps only.")
	print("  For definitive answers, use nsys_export_and_analyze.sh")
	print("=" * 72)

	traces_dir = Path(args.traces_dir)
	if args.files:
		trace_files = [Path(f) for f in args.files]
	else:
		trace_files = sorted(traces_dir.glob("*.trace.json.gz"))
		# Prefer the latest batch
		latest = [f for f in trace_files if "1459995" in f.name or "1459996" in f.name]
		if latest:
			trace_files = latest

	if not trace_files:
		print(f"  No trace files found in {traces_dir}")
		return

	print(f"  Traces to analyze: {[f.name for f in trace_files]}")

	all_results = {}
	for tf in trace_files:
		label, results = analyze_trace(str(tf))
		all_results[label] = results

	if len(all_results) >= 2:
		section("COMPARATIVE SUMMARY")
		for label, r in all_results.items():
			if not r:
				continue
			print(f"\n  {label}:")
			print(f"    Events: {r.get('total_events', 0):,}")
			print(f"    Copies: {r.get('copy_count', 0):,} "
				  f"({r.get('blocking_count', 0)}B / {r.get('async_count', 0)}A)")
			print(f"    Large (>100MB): {r.get('large_count', 0)}")
			print(f"    NCCL: {r.get('nccl_count', 0)}")

	print(f"\n{'=' * 72}")
	print("  REMINDER: These are CPU-side timestamps only.")
	print("  Async copies: duration = launch overhead (not transfer time)")
	print("  Blocking copies: duration = transfer + sync (more meaningful)")
	print("  For GPU-side truth: run nsys_export_and_analyze.sh on ACES")
	print(f"{'=' * 72}")


if __name__ == "__main__":
	main()
