"""Identify actual PCIe transfers from aten::copy_ events.

Key insight: GPU-internal copies (same device, HBM->HBM) return almost
instantly from the CPU perspective, showing "bandwidth" of 1000+ GB/s.
Real PCIe transfers (CPU->GPU or GPU->CPU) are limited to ~63 GB/s
(Gen5 x16 peak) and show proportionally longer CPU-side duration.

Heuristic to separate them:
- Effective BW > 200 GB/s -> GPU-internal (HBM-to-HBM)
- Effective BW < 100 GB/s and dur > 1ms -> likely PCIe transfer
- Everything else -> ambiguous
"""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

DTYPE_SIZES = {
	"float": 4, "float32": 4, "float64": 8, "double": 8,
	"half": 2, "float16": 2, "bfloat16": 2,
	"int": 4, "int32": 4, "int64": 8, "long": 8,
	"int16": 2, "short": 2, "int8": 1, "byte": 1,
	"bool": 1, "uint8": 1,
}

MAX_EVENTS = 10_000_000
PCIE_GEN5_X16_PEAK = 63.0  # GB/s unidirectional


def stream_events(path, max_events=MAX_EVENTS):
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


def estimate_bytes(dims, dtype_str):
	if not dims or not isinstance(dims, list):
		return 0
	try:
		numel = 1
		for d in dims:
			numel *= d
		byte_size = DTYPE_SIZES.get(dtype_str.lower() if dtype_str else "float", 4)
		return numel * byte_size
	except (TypeError, ValueError):
		return 0


def classify_copy(est_bytes, dur_us):
	"""Classify a copy as PCIe or GPU-internal based on effective bandwidth."""
	if est_bytes <= 0 or dur_us <= 0:
		return "unknown"
	bw_gbps = (est_bytes / 1e9) / (dur_us / 1e6)

	if bw_gbps > 200:
		return "gpu_internal"  # HBM bandwidth range
	elif bw_gbps <= PCIE_GEN5_X16_PEAK * 1.2 and dur_us >= 500:
		return "likely_pcie"  # Within PCIe range and takes >0.5ms
	elif bw_gbps <= 100:
		return "possibly_pcie"  # Low BW but short duration
	else:
		return "ambiguous"


def analyze_trace(trace_path):
	label = Path(trace_path).stem.replace(".trace.json", "")
	print(f"\n{'='*70}")
	print(f"PCIe Transfer Analysis: {label}")
	print(f"{'='*70}")

	copy_events = []
	nccl_events = []
	total = 0

	for ev in stream_events(trace_path):
		total += 1
		name = ev.get("name", "")
		dur = ev.get("dur", 0)
		args = ev.get("args", {})

		if name == "aten::copy_" and dur > 0:
			concrete = args.get("Concrete Inputs", [])
			non_blocking = len(concrete) >= 3 and concrete[2] in ("True", True, "1")
			input_dims = args.get("Input Dims", [])
			input_types = args.get("Input type", [])
			src_dims = input_dims[0] if len(input_dims) > 0 else []
			src_type = input_types[0] if len(input_types) > 0 else "float"
			est_bytes = estimate_bytes(src_dims, src_type)

			if est_bytes > 0:
				bw_gbps = (est_bytes / 1e9) / (dur / 1e6)
				classification = classify_copy(est_bytes, dur)
			else:
				bw_gbps = 0
				classification = "unknown"

			copy_events.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
				"non_blocking": non_blocking,
				"est_bytes": est_bytes,
				"bw_gbps": bw_gbps,
				"classification": classification,
				"dims": src_dims,
				"dtype": src_type,
			})

		elif "nccl" in name.lower() and dur > 0:
			nccl_events.append({
				"ts": ev.get("ts", 0),
				"dur": dur,
				"name": name,
			})

		if total % 2_000_000 == 0:
			print(f"  ... {total/1e6:.0f}M events")

	print(f"\n  Total events: {total:,}")
	print(f"  aten::copy_: {len(copy_events):,}")
	print(f"  NCCL: {len(nccl_events):,}")

	# Separate by classification
	by_class = defaultdict(list)
	for e in copy_events:
		by_class[e["classification"]].append(e)

	print(f"\n--- Copy Classification ---")
	for cls in ["likely_pcie", "possibly_pcie", "ambiguous", "gpu_internal", "unknown"]:
		events = by_class.get(cls, [])
		if not events:
			continue
		total_bytes = sum(e["est_bytes"] for e in events)
		total_dur = sum(e["dur"] for e in events)
		blocking = [e for e in events if not e["non_blocking"]]
		async_c = [e for e in events if e["non_blocking"]]
		print(f"\n  {cls}: {len(events)} events ({len(blocking)}B/{len(async_c)}A)")
		print(f"    Total data: {total_bytes/1e9:.2f} GB")
		print(f"    Total CPU time: {total_dur/1e6:.3f} s")

	# --- Focus on likely PCIe transfers ---
	pcie_events = by_class.get("likely_pcie", []) + by_class.get("possibly_pcie", [])

	if pcie_events:
		print(f"\n{'='*70}")
		print(f"LIKELY PCIe TRANSFERS (BW <= {PCIE_GEN5_X16_PEAK * 1.2:.0f} GB/s)")
		print(f"{'='*70}")

		blocking_pcie = [e for e in pcie_events if not e["non_blocking"]]
		async_pcie = [e for e in pcie_events if e["non_blocking"]]

		for group_name, group in [("Blocking", blocking_pcie), ("Async", async_pcie)]:
			if not group:
				continue

			print(f"\n--- {group_name} PCIe Transfers ---")

			# Size buckets
			large = [e for e in group if e["est_bytes"] > 100e6]
			medium = [e for e in group if 1e6 <= e["est_bytes"] <= 100e6]
			small = [e for e in group if e["est_bytes"] < 1e6]

			print(f"  >100MB: {len(large)}")
			print(f"  1MB-100MB: {len(medium)}")
			print(f"  <1MB: {len(small)}")

			# Bandwidth statistics for large transfers
			if large:
				bws = [e["bw_gbps"] for e in large]
				total_bytes = sum(e["est_bytes"] for e in large)
				total_dur = sum(e["dur"] for e in large)
				weighted_bw = (total_bytes / 1e9) / (total_dur / 1e6) if total_dur > 0 else 0

				print(f"\n  Large transfer ({group_name}) bandwidth:")
				print(f"    Count: {len(large)}")
				print(f"    Total data: {total_bytes/1e9:.2f} GB")
				print(f"    Total time: {total_dur/1e6:.3f} s")
				print(f"    Weighted avg BW: {weighted_bw:.2f} GB/s")

				bws_sorted = sorted(bws)
				n = len(bws_sorted)
				print(f"    Min BW:    {bws_sorted[0]:.2f} GB/s")
				print(f"    P10 BW:    {bws_sorted[max(0,int(n*0.1))]:.2f} GB/s")
				print(f"    Median BW: {bws_sorted[n//2]:.2f} GB/s")
				print(f"    P90 BW:    {bws_sorted[min(n-1,int(n*0.9))]:.2f} GB/s")
				print(f"    Max BW:    {bws_sorted[-1]:.2f} GB/s")
				print(f"    PCIe peak: {PCIE_GEN5_X16_PEAK:.1f} GB/s (Gen5 x16)")
				print(f"    Utilization: {weighted_bw/PCIE_GEN5_X16_PEAK*100:.1f}%")

				# Duration distribution
				dur_sorted = sorted([e["dur"] for e in large])
				print(f"\n    Duration distribution:")
				print(f"      Min:    {dur_sorted[0]/1e3:.1f} ms")
				print(f"      Median: {dur_sorted[n//2]/1e3:.1f} ms")
				print(f"      Max:    {dur_sorted[-1]/1e3:.1f} ms")

				# Show individual large transfers
				print(f"\n    First 25 large {group_name} PCIe transfers (chronological):")
				large_sorted = sorted(large, key=lambda x: x["ts"])
				for i, e in enumerate(large_sorted[:25]):
					print(f"      [{i:>3}] {e['est_bytes']/1e6:>8.1f} MB, "
						  f"{e['dur']/1e3:>8.1f} ms, "
						  f"{e['bw_gbps']:>6.2f} GB/s, "
						  f"{'async' if e['non_blocking'] else 'block'}, "
						  f"dims={e['dims'][:2]}...")

			# Medium transfers
			if medium:
				bws = [e["bw_gbps"] for e in medium]
				total_bytes = sum(e["est_bytes"] for e in medium)
				total_dur = sum(e["dur"] for e in medium)
				weighted_bw = (total_bytes / 1e9) / (total_dur / 1e6) if total_dur > 0 else 0
				print(f"\n  Medium transfer ({group_name}) bandwidth:")
				print(f"    Count: {len(medium)}")
				print(f"    Total data: {total_bytes/1e6:.0f} MB")
				print(f"    Weighted avg BW: {weighted_bw:.2f} GB/s")

	# --- Bandwidth histogram ---
	all_with_bw = [e for e in copy_events if e["bw_gbps"] > 0 and e["est_bytes"] > 1e6]
	if all_with_bw:
		print(f"\n{'='*70}")
		print(f"BANDWIDTH HISTOGRAM (copies >1MB)")
		print(f"{'='*70}")

		bw_buckets = [
			("0-5 GB/s (low PCIe)", 0, 5),
			("5-20 GB/s (mid PCIe)", 5, 20),
			("20-50 GB/s (high PCIe)", 20, 50),
			("50-80 GB/s (near peak PCIe)", 50, 80),
			("80-200 GB/s (ambiguous)", 80, 200),
			("200-1000 GB/s (HBM low)", 200, 1000),
			("1000-3000 GB/s (HBM high)", 1000, 3000),
			(">3000 GB/s (HBM peak)", 3000, float("inf")),
		]

		for bname, lo, hi in bw_buckets:
			bucket = [e for e in all_with_bw if lo <= e["bw_gbps"] < hi]
			if bucket:
				total_b = sum(e["est_bytes"] for e in bucket)
				total_d = sum(e["dur"] for e in bucket)
				blocking = sum(1 for e in bucket if not e["non_blocking"])
				async_c = sum(1 for e in bucket if e["non_blocking"])
				bar = "#" * min(60, len(bucket))
				print(f"  {bname:35s} | {len(bucket):>5} ({blocking}B/{async_c}A) "
					  f"| {total_b/1e9:>6.2f} GB | {total_d/1e6:>6.3f}s | {bar}")

	# --- Contention analysis: PCIe transfers vs NCCL ---
	pcie_large = [e for e in pcie_events if e["est_bytes"] > 10e6]
	if pcie_large and nccl_events:
		print(f"\n{'='*70}")
		print("PCIe CONTENTION: Offload Transfers vs NCCL (Ulysses)")
		print(f"{'='*70}")
		print(f"  NOTE: This uses CPU-side timestamps. Without CUDA activity,")
		print(f"  we see CPU launch overlap, not actual GPU-side overlap.")

		nccl_sorted = sorted(nccl_events, key=lambda x: x["ts"])
		overlapping = 0
		non_overlapping = 0
		overlap_bws = []
		clean_bws = []

		for c in pcie_large:
			c_start = c["ts"]
			c_end = c["ts"] + c["dur"]
			has_overlap = any(
				n["ts"] < c_end and (n["ts"] + n["dur"]) > c_start
				for n in nccl_sorted
			)
			if has_overlap:
				overlapping += 1
				if c["bw_gbps"] > 0:
					overlap_bws.append(c["bw_gbps"])
			else:
				non_overlapping += 1
				if c["bw_gbps"] > 0:
					clean_bws.append(c["bw_gbps"])

		print(f"  PCIe transfers (>10MB): {len(pcie_large)}")
		print(f"  Overlapping with NCCL (CPU timeline): {overlapping}")
		print(f"  Clean (no overlap): {non_overlapping}")

		if overlap_bws and clean_bws:
			avg_o = sum(overlap_bws) / len(overlap_bws)
			avg_c = sum(clean_bws) / len(clean_bws)
			print(f"\n  During NCCL overlap:  avg BW = {avg_o:.2f} GB/s")
			print(f"  Without NCCL overlap: avg BW = {avg_c:.2f} GB/s")
			if avg_c > 0:
				degradation = (1 - avg_o / avg_c) * 100
				print(f"  Degradation: {degradation:.1f}%")

	return label, copy_events, pcie_events


def main():
	traces_dir = Path("results/traces")
	all_results = {}

	for trace_file in sorted(traces_dir.glob("*.trace.json.gz")):
		label, copy_events, pcie_events = analyze_trace(str(trace_file))
		all_results[label] = (copy_events, pcie_events)

	if len(all_results) == 2:
		labels = list(all_results.keys())
		print(f"\n{'='*70}")
		print("COMPARATIVE SUMMARY")
		print(f"{'='*70}")

		for label in labels:
			copies, pcie = all_results[label]
			blocking = [e for e in copies if not e["non_blocking"]]
			async_c = [e for e in copies if e["non_blocking"]]

			pcie_blocking = [e for e in pcie if not e["non_blocking"]]
			pcie_async = [e for e in pcie if e["non_blocking"]]

			total_pcie_bytes = sum(e["est_bytes"] for e in pcie)
			total_pcie_dur = sum(e["dur"] for e in pcie)
			total_blocking_bytes = sum(e["est_bytes"] for e in blocking)
			total_blocking_dur = sum(e["dur"] for e in blocking)

			print(f"\n  {label}:")
			print(f"    All copies: {len(copies):,} ({len(blocking)}B / {len(async_c)}A)")
			print(f"    Likely PCIe: {len(pcie):,} ({len(pcie_blocking)}B / {len(pcie_async)}A)")
			print(f"    PCIe data: {total_pcie_bytes/1e9:.2f} GB, time: {total_pcie_dur/1e6:.3f}s")
			print(f"    All blocking data: {total_blocking_bytes/1e9:.2f} GB, time: {total_blocking_dur/1e6:.3f}s")

		print(f"\n{'='*70}")
		print("KEY CONCLUSIONS")
		print(f"{'='*70}")
		print("""
  1. TRACES ARE INCOMPLETE: Only CPU-side events captured.
     GPU events (with actual Memcpy HtoD/DtoH timing) were truncated.
     The profiler DID configure ProfilerActivity.CUDA but the trace
     gzip file was truncated before reaching the GPU event section.

  2. PCIe BANDWIDTH ESTIMATION IS APPROXIMATE:
     - For blocking copies: CPU duration includes actual transfer + sync overhead
     - For async copies: CPU duration is just launch overhead (cannot measure BW)
     - Cannot distinguish H2D from D2H direction
     - Heuristic separation of PCIe vs GPU-internal based on effective BW

  3. TO GET DEFINITIVE PCIe BANDWIDTH DATA:
     a) Use nsys profiling (captures CUPTI events with explicit direction + bytes)
     b) Or ensure torch.profiler traces are not truncated (need more disk space)
     c) Or run cuda-samples bandwidthTest for theoretical peak measurement
""")


if __name__ == "__main__":
	main()
