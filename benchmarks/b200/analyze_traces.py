"""Analyze B200 torch profiler traces: no-offload vs offload."""
import gzip
import json
import sys
from collections import defaultdict

def load_trace(path):
	with gzip.open(path, "rt") as f:
		return json.load(f)

def categorize_kernel(name):
	nl = name.lower()
	if "nccl" in nl:
		return "NCCL"
	elif "memcpy" in nl or "memset" in nl:
		if "htod" in nl or "HtoD" in name:
			return "Memcpy H2D"
		elif "dtoh" in nl or "DtoH" in name:
			return "Memcpy D2H"
		elif "dtod" in nl or "DtoD" in name:
			return "Memcpy D2D"
		return "Memcpy/Memset"
	elif "_attn_fwd" in nl or "flash" in nl or "sage" in nl:
		return "Attention"
	elif "gemm" in nl or "cutlass" in nl or "cublas" in nl:
		return "GEMM/MatMul"
	elif "triton" in nl:
		return "Triton"
	elif "nvjet" in nl:
		return "NVJet (MLP)"
	elif "elementwise" in nl or "vectorized" in nl:
		return "Elementwise"
	elif "reduce" in nl or "norm" in nl:
		return "Reduction/Norm"
	else:
		return "Other"

def analyze_trace(data, label):
	events = data.get("traceEvents", data) if isinstance(data, dict) else data

	# Separate GPU vs CPU events
	gpu_events = []
	cpu_events = []
	gpu_categories = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime"}

	# Track by category
	kernel_stats = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0, "min_us": float("inf")})
	category_stats = defaultdict(lambda: {"count": 0, "total_us": 0})

	# GPU timeline
	gpu_stream_events = defaultdict(list)  # stream -> [(start, end, name)]

	# Module call analysis
	module_stats = defaultdict(lambda: {"count": 0, "total_us": 0})

	for ev in events:
		if not isinstance(ev, dict):
			continue
		ph = ev.get("ph", "")
		cat = ev.get("cat", "")
		dur = ev.get("dur", 0)
		name = ev.get("name", "")

		if ph != "X" or dur <= 0:
			continue

		ts = ev.get("ts", 0)

		if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
			gpu_events.append(ev)
			category = categorize_kernel(name)
			category_stats[category]["count"] += 1
			category_stats[category]["total_us"] += dur

			# Track top kernels
			kernel_stats[name]["count"] += 1
			kernel_stats[name]["total_us"] += dur
			kernel_stats[name]["max_us"] = max(kernel_stats[name]["max_us"], dur)
			kernel_stats[name]["min_us"] = min(kernel_stats[name]["min_us"], dur)

			# Track per-stream
			args = ev.get("args", {})
			stream = args.get("stream", ev.get("tid", 0))
			gpu_stream_events[stream].append((ts, ts + dur, name))

		elif cat in ("cpu_op", "user_annotation", "python_function"):
			cpu_events.append(ev)
			if "module.py" in name or "_call_impl" in name:
				# Extract module name from args
				mod_name = ev.get("args", {}).get("name", name)
				module_stats[mod_name]["count"] += 1
				module_stats[mod_name]["total_us"] += dur

	# Print results
	print(f"\n{'='*80}")
	print(f"  {label}")
	print(f"{'='*80}")

	total_gpu_us = sum(s["total_us"] for s in category_stats.values())
	print(f"\nTotal GPU kernel time: {total_gpu_us/1e6:.2f}s")
	print(f"Total GPU events: {sum(s['count'] for s in category_stats.values())}")

	print(f"\n--- GPU Kernel Categories ---")
	print(f"{'Category':<20} {'Count':>8} {'Total (s)':>10} {'Avg (ms)':>10} {'% GPU':>8}")
	print("-" * 60)
	for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]["total_us"]):
		pct = 100 * stats["total_us"] / total_gpu_us if total_gpu_us > 0 else 0
		avg_ms = stats["total_us"] / stats["count"] / 1000
		print(f"{cat:<20} {stats['count']:>8} {stats['total_us']/1e6:>10.2f} {avg_ms:>10.3f} {pct:>7.1f}%")

	print(f"\n--- Top 15 Kernels by Total Time ---")
	print(f"{'Kernel':<60} {'Count':>6} {'Total(s)':>8} {'Avg(ms)':>8} {'Max(ms)':>8}")
	print("-" * 95)
	top_kernels = sorted(kernel_stats.items(), key=lambda x: -x[1]["total_us"])[:15]
	for name, stats in top_kernels:
		short = name[:58] if len(name) > 58 else name
		avg_ms = stats["total_us"] / stats["count"] / 1000
		max_ms = stats["max_us"] / 1000
		print(f"{short:<60} {stats['count']:>6} {stats['total_us']/1e6:>8.2f} {avg_ms:>8.3f} {max_ms:>8.1f}")

	# GPU utilization per stream
	print(f"\n--- Per-Stream Summary ---")
	for stream_id in sorted(gpu_stream_events.keys()):
		evts = gpu_stream_events[stream_id]
		if len(evts) < 10:
			continue
		evts.sort()
		total_active = sum(e - s for s, e, _ in evts)
		wall = evts[-1][1] - evts[0][0] if evts else 0
		util = 100 * total_active / wall if wall > 0 else 0
		print(f"  Stream {stream_id}: {len(evts)} events, {total_active/1e6:.2f}s active, {wall/1e6:.2f}s wall, {util:.1f}% util")

	return category_stats, kernel_stats, gpu_stream_events


def compare(stats1, stats2, label1, label2):
	print(f"\n{'='*80}")
	print(f"  COMPARISON: {label1} vs {label2}")
	print(f"{'='*80}")

	all_cats = set(list(stats1.keys()) + list(stats2.keys()))

	print(f"\n{'Category':<20} {label1+' (s)':>12} {label2+' (s)':>12} {'Delta':>10} {'Delta%':>8}")
	print("-" * 65)
	for cat in sorted(all_cats, key=lambda c: -(stats1.get(c, {"total_us": 0})["total_us"] + stats2.get(c, {"total_us": 0})["total_us"])):
		t1 = stats1.get(cat, {"total_us": 0, "count": 0})["total_us"] / 1e6
		t2 = stats2.get(cat, {"total_us": 0, "count": 0})["total_us"] / 1e6
		delta = t2 - t1
		pct = 100 * delta / t1 if t1 > 0 else float("inf")
		print(f"{cat:<20} {t1:>12.2f} {t2:>12.2f} {delta:>+10.2f} {pct:>+7.1f}%")


if __name__ == "__main__":
	traces_dir = "/Users/dannyliu/research_work/b200_workspace/results/traces"

	print("Loading no-offload trace...")
	data1 = load_trace(f"{traces_dir}/b200_no_offload_5steps.trace.json.gz")
	print("Loading offload trace...")
	data2 = load_trace(f"{traces_dir}/b200_offload_5steps.trace.json.gz")

	cat1, kern1, streams1 = analyze_trace(data1, "B200 No Offload (5 steps)")
	cat2, kern2, streams2 = analyze_trace(data2, "B200 Offload (5 steps)")

	compare(cat1, cat2, "NoOffload", "Offload")
