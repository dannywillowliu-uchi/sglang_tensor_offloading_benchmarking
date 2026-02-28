"""Analyze CPU-side copy events from torch.profiler traces.

The traces only contain CPU events (GPU events were truncated).
We can still extract:
1. aten::copy_ timing (blocking copies = actual transfer time)
2. Tensor dimensions -> estimated bytes
3. non_blocking flag -> whether copy was async
4. Timing patterns correlating with denoising steps
"""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Dtype sizes in bytes
DTYPE_SIZES = {
	"float": 4, "float32": 4, "float64": 8, "double": 8,
	"half": 2, "float16": 2, "bfloat16": 2,
	"int": 4, "int32": 4, "int64": 8, "long": 8,
	"int16": 2, "short": 2, "int8": 1, "byte": 1,
	"bool": 1, "uint8": 1,
}

MAX_EVENTS = 10_000_000  # Parse up to 10M events


def stream_events(path: str, max_events=MAX_EVENTS):
	"""Stream-parse events from Chrome Trace JSON (possibly truncated gzip)."""
	event_count = 0
	buf = ""
	found_array = False
	CHUNK_SIZE = 1024 * 1024

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
							event_count += 1
							yield event
							if event_count >= max_events:
								return
					except json.JSONDecodeError:
						pass

				if len(buf) > 10 * CHUNK_SIZE:
					buf = buf[-CHUNK_SIZE:]

	except EOFError:
		pass

	print(f"  Parsed {event_count:,} events")


def estimate_bytes(dims, dtype_str):
	"""Estimate bytes from tensor dimensions and dtype."""
	if not dims or not isinstance(dims, list):
		return 0
	try:
		numel = 1
		for d in dims:
			numel *= d
		dtype_lower = dtype_str.lower() if dtype_str else "float"
		byte_size = DTYPE_SIZES.get(dtype_lower, 4)
		return numel * byte_size
	except (TypeError, ValueError):
		return 0


def analyze_trace(trace_path):
	"""Extract copy event statistics from a trace."""
	label = Path(trace_path).stem.replace(".trace.json", "")
	print(f"\n{'='*70}")
	print(f"Analyzing: {label}")
	print(f"{'='*70}")

	copy_events = []  # aten::copy_
	to_copy_events = []  # aten::_to_copy
	to_events = []  # aten::to
	nccl_events = []
	step_annotations = []  # denoising step markers
	total_events = 0

	for ev in stream_events(trace_path):
		total_events += 1
		name = ev.get("name", "")
		cat = ev.get("cat", "")
		dur = ev.get("dur", 0)
		args = ev.get("args", {})

		if name == "aten::copy_" and dur > 0:
			# Extract non_blocking flag
			concrete = args.get("Concrete Inputs", [])
			non_blocking = False
			if len(concrete) >= 3:
				non_blocking = concrete[2] in ("True", True, "1")

			# Extract dimensions for size estimation
			input_dims = args.get("Input Dims", [])
			input_types = args.get("Input type", [])

			# First dim list is typically the source tensor
			src_dims = input_dims[0] if len(input_dims) > 0 else []
			src_type = input_types[0] if len(input_types) > 0 else "float"

			est_bytes = estimate_bytes(src_dims, src_type)

			copy_events.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
				"non_blocking": non_blocking,
				"dims": src_dims,
				"dtype": src_type,
				"est_bytes": est_bytes,
				"args": args,
			})

		elif name == "aten::_to_copy" and dur > 0:
			concrete = args.get("Concrete Inputs", [])
			input_dims = args.get("Input Dims", [])
			input_types = args.get("Input type", [])

			src_dims = input_dims[0] if len(input_dims) > 0 else []
			src_type = input_types[0] if len(input_types) > 0 else "float"
			est_bytes = estimate_bytes(src_dims, src_type)

			to_copy_events.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
				"est_bytes": est_bytes,
				"concrete": concrete,
				"dims": src_dims,
				"dtype": src_type,
			})

		elif name == "aten::to" and dur > 0:
			to_events.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
			})

		elif "nccl" in name.lower() and dur > 0:
			nccl_events.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
				"name": name,
			})

		elif "denoising_step_" in name and cat == "user_annotation":
			step_annotations.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
				"name": name,
			})

		if total_events % 2_000_000 == 0:
			print(f"  ... {total_events/1e6:.0f}M events")

	print(f"\n  Total events: {total_events:,}")
	print(f"  aten::copy_: {len(copy_events):,}")
	print(f"  aten::_to_copy: {len(to_copy_events):,}")
	print(f"  aten::to: {len(to_events):,}")
	print(f"  NCCL: {len(nccl_events):,}")
	print(f"  Step annotations: {len(step_annotations):,}")

	# --- Copy event analysis ---
	if copy_events:
		print(f"\n--- aten::copy_ Analysis ---")

		# Split by blocking vs non-blocking
		blocking = [e for e in copy_events if not e["non_blocking"]]
		async_copies = [e for e in copy_events if e["non_blocking"]]

		print(f"  Blocking copies:     {len(blocking):,}")
		print(f"  Non-blocking copies: {len(async_copies):,}")

		for group_name, group in [("Blocking", blocking), ("Non-blocking", async_copies)]:
			if not group:
				continue

			print(f"\n  [{group_name} copies]")

			# Size distribution
			large = [e for e in group if e["est_bytes"] > 100e6]
			medium = [e for e in group if 1e6 <= e["est_bytes"] <= 100e6]
			small = [e for e in group if 0 < e["est_bytes"] < 1e6]
			unknown = [e for e in group if e["est_bytes"] == 0]

			print(f"    >100MB: {len(large):,}  (likely layer offload)")
			print(f"    1MB-100MB: {len(medium):,}")
			print(f"    <1MB: {len(small):,}")
			print(f"    Unknown size: {len(unknown):,}")

			# Timing stats
			durs = [e["dur"] for e in group]
			total_dur = sum(durs)
			print(f"    Total CPU time: {total_dur/1e6:.3f} s")
			print(f"    Avg duration: {total_dur/len(durs):.0f} us")
			if durs:
				durs_sorted = sorted(durs)
				n = len(durs_sorted)
				print(f"    Median duration: {durs_sorted[n//2]:.0f} us")
				print(f"    P90 duration: {durs_sorted[int(n*0.9)]:.0f} us")
				print(f"    P99 duration: {durs_sorted[int(n*0.99)]:.0f} us")
				print(f"    Max duration: {max(durs):.0f} us ({max(durs)/1e6:.3f} s)")

			# For large blocking copies, estimate bandwidth
			# (CPU-side duration of blocking copy ≈ actual transfer time)
			if large and group_name == "Blocking":
				print(f"\n    --- Large blocking copies (>100MB) = likely layer transfers ---")
				bws = []
				for e in large:
					bw = (e["est_bytes"] / 1e9) / (e["dur"] / 1e6)
					bws.append(bw)
				print(f"    Count: {len(large)}")
				total_bytes = sum(e["est_bytes"] for e in large)
				total_time = sum(e["dur"] for e in large)
				weighted_bw = (total_bytes / 1e9) / (total_time / 1e6) if total_time > 0 else 0
				print(f"    Total data: {total_bytes/1e9:.2f} GB")
				print(f"    Total time: {total_time/1e6:.3f} s")
				print(f"    Weighted avg BW: {weighted_bw:.2f} GB/s")
				if bws:
					bws_sorted = sorted(bws)
					n = len(bws_sorted)
					print(f"    Median BW: {bws_sorted[n//2]:.2f} GB/s")
					print(f"    P10 BW: {bws_sorted[int(n*0.1)]:.2f} GB/s")
					print(f"    P90 BW: {bws_sorted[int(n*0.9)]:.2f} GB/s")
					print(f"    PCIe Gen5 x16 peak: 63.0 GB/s")
					print(f"    Est. utilization: {weighted_bw/63.0*100:.1f}%")

				# Show individual large copies
				print(f"\n    Chronological large blocking copies (first 20):")
				large_sorted = sorted(large, key=lambda x: x["ts"])
				for i, e in enumerate(large_sorted[:20]):
					bw = (e["est_bytes"] / 1e9) / (e["dur"] / 1e6) if e["dur"] > 0 else 0
					print(f"      [{i:>3}] {e['est_bytes']/1e6:>8.1f} MB, "
						  f"{e['dur']/1e6:>6.3f} s, "
						  f"{bw:>6.2f} GB/s, "
						  f"dims={e['dims'][:3]}...")

			# For non-blocking copies: CPU duration = just launch overhead
			if large and group_name == "Non-blocking":
				print(f"\n    --- Large non-blocking copies (>100MB) ---")
				print(f"    NOTE: CPU duration for async copies = launch overhead only,")
				print(f"          NOT actual transfer time. Cannot measure BW from these.")
				print(f"    Count: {len(large)}")
				total_bytes = sum(e["est_bytes"] for e in large)
				total_time = sum(e["dur"] for e in large)
				print(f"    Total data: {total_bytes/1e9:.2f} GB")
				print(f"    Total launch overhead: {total_time/1e6:.3f} s")
				avg_launch = total_time / len(large) if large else 0
				print(f"    Avg launch overhead: {avg_launch:.0f} us per copy")

	# --- Step-level timing correlation ---
	if step_annotations and copy_events:
		print(f"\n--- Copy Events per Denoising Step ---")
		steps_sorted = sorted(step_annotations, key=lambda x: x["ts"])

		for step in steps_sorted[:30]:  # first 30 steps
			step_start = step["ts"]
			step_end = step["ts"] + step["dur"]
			step_name = step["name"]

			# Count copies within this step
			copies_in_step = [
				e for e in copy_events
				if step_start <= e["ts"] < step_end
			]
			blocking_in_step = [e for e in copies_in_step if not e["non_blocking"]]
			async_in_step = [e for e in copies_in_step if e["non_blocking"]]

			total_copy_dur = sum(e["dur"] for e in copies_in_step)
			total_copy_bytes = sum(e["est_bytes"] for e in copies_in_step)

			print(f"  {step_name}: {step['dur']/1e6:.3f}s total, "
				  f"{len(copies_in_step)} copies "
				  f"({len(blocking_in_step)}B/{len(async_in_step)}A), "
				  f"{total_copy_bytes/1e6:.0f}MB, "
				  f"copy_time={total_copy_dur/1e6:.3f}s")

	# --- NCCL vs copy overlap analysis ---
	if nccl_events and copy_events:
		large_copies = [e for e in copy_events if e["est_bytes"] > 10e6]
		if large_copies and nccl_events:
			print(f"\n--- NCCL vs Large Copy Overlap ---")
			nccl_sorted = sorted(nccl_events, key=lambda x: x["ts"])
			overlapping = 0
			non_overlapping = 0

			for c in large_copies:
				c_start = c["ts"]
				c_end = c["ts"] + c["dur"]
				has_overlap = any(
					n["ts"] < c_end and (n["ts"] + n["dur"]) > c_start
					for n in nccl_sorted
				)
				if has_overlap:
					overlapping += 1
				else:
					non_overlapping += 1

			print(f"  Large copies (>10MB): {len(large_copies)}")
			print(f"  Overlapping with NCCL: {overlapping}")
			print(f"  Clean (no overlap): {non_overlapping}")

	return {
		"label": label,
		"copy_events": len(copy_events),
		"to_copy_events": len(to_copy_events),
		"nccl_events": len(nccl_events),
		"step_annotations": len(step_annotations),
		"blocking_copies": len([e for e in copy_events if not e["non_blocking"]]),
		"async_copies": len([e for e in copy_events if e["non_blocking"]]),
	}


def main():
	traces_dir = Path("results/traces")
	results = []

	for trace_file in sorted(traces_dir.glob("*.trace.json.gz")):
		result = analyze_trace(str(trace_file))
		results.append(result)

	if len(results) == 2:
		print(f"\n{'='*70}")
		print("COMPARISON SUMMARY")
		print(f"{'='*70}")
		for r in results:
			print(f"\n  {r['label']}:")
			print(f"    Total copy events: {r['copy_events']:,}")
			print(f"    Blocking: {r['blocking_copies']:,}")
			print(f"    Async: {r['async_copies']:,}")
			print(f"    NCCL: {r['nccl_events']:,}")


if __name__ == "__main__":
	main()
