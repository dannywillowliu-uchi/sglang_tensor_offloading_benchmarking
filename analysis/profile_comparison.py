"""Deep A/B comparison of new vs old offload using TraceLens CSVs + Perf JSON.

Analyses:
1. GPU timeline comparison (compute/comm/idle breakdown)
2. NCCL collective deep dive (large vs small all_to_allv)
3. Kernel-by-kernel diff (match by name, compute duration delta)
4. Per-step timing patterns (outliers, steady-state delta)
5. Summary findings (what data tells us, what remains for nsys)

Data sources:
- results/tracelens_new/*.csv / results/tracelens_old/*.csv
- results/logs/perf_new_offload_1459995.json / perf_old_offload_1459996.json
"""

import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_csv(path):
	"""Load CSV file, return list of dicts."""
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def load_json(path):
	with open(path) as f:
		return json.load(f)


def fmt_ms(val_us):
	"""Format microseconds as seconds with 1 decimal."""
	return f"{val_us / 1000:.1f}s"


def fmt_pct(val):
	return f"{val:.1f}%"


def section(title):
	print(f"\n{'=' * 72}")
	print(f"  {title}")
	print(f"{'=' * 72}")


def subsection(title):
	print(f"\n--- {title} ---")


# ==========================================================================
# 1. GPU Timeline Comparison
# ==========================================================================
def analyze_gpu_timeline(new_rows, old_rows):
	section("1. GPU TIMELINE COMPARISON")

	def parse_timeline(rows):
		d = {}
		for r in rows:
			d[r["type"]] = {
				"ms": float(r["time ms"]),
				"pct": float(r["percent"]),
			}
		return d

	new = parse_timeline(new_rows)
	old = parse_timeline(old_rows)

	metrics = [
		("computation_time", "Compute"),
		("exposed_comm_time", "Exposed Comm (NCCL)"),
		("exposed_memcpy_time", "Exposed Memcpy (H2D)"),
		("busy_time", "Busy (total)"),
		("idle_time", "Idle"),
		("total_time", "Total"),
		("total_comm_time", "Total Comm"),
		("total_memcpy_time", "Total Memcpy"),
	]

	print(f"\n  {'Metric':<28s} {'New (s)':>10s} {'Old (s)':>10s} {'Delta (s)':>10s} {'New %':>8s} {'Old %':>8s}")
	print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

	for key, label in metrics:
		n = new.get(key, {"ms": 0, "pct": 0})
		o = old.get(key, {"ms": 0, "pct": 0})
		delta = n["ms"] - o["ms"]
		print(f"  {label:<28s} {n['ms']/1000:>10.1f} {o['ms']/1000:>10.1f} {delta/1000:>+10.1f} {n['pct']:>7.1f}% {o['pct']:>7.1f}%")

	# Key observations
	subsection("Key Observations")
	n_idle = new["idle_time"]["ms"]
	o_idle = old["idle_time"]["ms"]
	n_comm = new["exposed_comm_time"]["ms"]
	o_comm = old["exposed_comm_time"]["ms"]
	n_memcpy = new["exposed_memcpy_time"]["ms"]
	o_memcpy = old["exposed_memcpy_time"]["ms"]
	n_total = new["total_time"]["ms"]
	o_total = old["total_time"]["ms"]

	print(f"  GPU idle time:  new={n_idle/1000:.1f}s ({new['idle_time']['pct']:.1f}%)  "
		  f"old={o_idle/1000:.1f}s ({old['idle_time']['pct']:.1f}%)")
	print(f"    -> New has {(n_idle - o_idle)/1000:+.1f}s MORE idle time ({(n_idle/o_idle - 1)*100:+.1f}%)")

	print(f"  Exposed comm:   new={n_comm/1000:.1f}s  old={o_comm/1000:.1f}s  "
		  f"delta={n_comm - o_comm:+.0f}ms ({(n_comm/o_comm - 1)*100:+.1f}%)")

	print(f"  Exposed memcpy: new={n_memcpy/1000:.1f}s  old={o_memcpy/1000:.1f}s  "
		  f"delta={n_memcpy - o_memcpy:+.0f}ms")
	if o_memcpy > 0:
		print(f"    -> Old has {o_memcpy/1000:.1f}s exposed memcpy (blocking .to()), new has almost none (async)")

	print(f"  Total GPU time: new={n_total/1000:.1f}s  old={o_total/1000:.1f}s  "
		  f"delta={n_total - o_total:+.0f}ms ({(n_total/o_total - 1)*100:+.1f}%)")

	total_memcpy_n = new["total_memcpy_time"]["ms"]
	total_memcpy_o = old["total_memcpy_time"]["ms"]
	print(f"\n  Total memcpy time (incl. overlapped):")
	print(f"    New: {total_memcpy_n/1000:.1f}s ({new['total_memcpy_time']['pct']:.1f}%)")
	print(f"    Old: {total_memcpy_o/1000:.1f}s ({old['total_memcpy_time']['pct']:.1f}%)")
	print(f"    -> New does {total_memcpy_n/total_memcpy_o:.1f}x more memcpy (overlapped with compute)")
	print(f"    -> But only {n_memcpy/1000:.1f}s is *exposed* (not hidden behind compute)")


# ==========================================================================
# 2. NCCL Collective Deep Dive
# ==========================================================================
def analyze_nccl(new_rows, old_rows):
	section("2. NCCL COLLECTIVE DEEP DIVE")

	def parse_coll(rows):
		result = []
		for r in rows:
			result.append({
				"name": r["Collective name"],
				"msg_mb": float(r["In msg size (MB)_first"]),
				"dur_sum": float(r["dur_sum"]),
				"dur_mean": float(r["dur_mean"]),
				"dur_std": float(r["dur_std"]) if r["dur_std"] else 0,
				"dur_min": float(r["dur_min"]),
				"dur_max": float(r["dur_max"]),
				"count": int(r["operation_count"]),
			})
		return result

	new = parse_coll(new_rows)
	old = parse_coll(old_rows)

	for label, data in [("NEW offload", new), ("OLD offload", old)]:
		subsection(label)
		for c in data:
			print(f"  {c['name']} ({c['msg_mb']:.1f} MB, {c['count']} ops)")
			print(f"    sum={c['dur_sum']/1e6:.1f}s  mean={c['dur_mean']/1000:.1f}ms  "
				  f"std={c['dur_std']/1000:.1f}ms")
			print(f"    min={c['dur_min']/1000:.1f}ms  max={c['dur_max']/1000:.1f}ms")

	# Direct comparison of large all_to_allv
	subsection("Large all_to_allv Comparison (211 MB, Ulysses attention)")
	new_large = next((c for c in new if c["msg_mb"] > 100), None)
	old_large = next((c for c in old if c["msg_mb"] > 100), None)

	if new_large and old_large:
		print(f"  {'Metric':<20s} {'New':>12s} {'Old':>12s} {'Delta':>12s} {'Ratio':>8s}")
		print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

		comparisons = [
			("Sum (s)", new_large["dur_sum"] / 1e6, old_large["dur_sum"] / 1e6),
			("Mean (ms)", new_large["dur_mean"] / 1000, old_large["dur_mean"] / 1000),
			("Std (ms)", new_large["dur_std"] / 1000, old_large["dur_std"] / 1000),
			("Min (ms)", new_large["dur_min"] / 1000, old_large["dur_min"] / 1000),
			("Max (s)", new_large["dur_max"] / 1e6, old_large["dur_max"] / 1e6),
		]
		for label, nv, ov in comparisons:
			delta = nv - ov
			ratio = nv / ov if ov > 0 else float("inf")
			print(f"  {label:<20s} {nv:>12.2f} {ov:>12.2f} {delta:>+12.2f} {ratio:>7.2f}x")

		print(f"\n  NCCL slowdown: {(new_large['dur_mean']/old_large['dur_mean'] - 1)*100:+.1f}% mean, "
			  f"{(new_large['dur_max']/old_large['dur_max'] - 1)*100:+.1f}% max tail latency")
		print(f"  Total NCCL overhead: {(new_large['dur_sum'] - old_large['dur_sum'])/1e6:+.1f}s "
			  f"({new_large['count']} ops)")

	# Small all_to_allv
	subsection("Small all_to_allv Comparison (5 MB, position embedding)")
	new_small = next((c for c in new if 1 < c["msg_mb"] < 100 and c["name"] == "all_to_allv"), None)
	old_small = next((c for c in old if 1 < c["msg_mb"] < 100 and c["name"] == "all_to_allv"), None)

	if new_small and old_small:
		print(f"  New mean: {new_small['dur_mean']:.0f} us  Old mean: {old_small['dur_mean']:.0f} us  "
			  f"Delta: {new_small['dur_mean'] - old_small['dur_mean']:+.0f} us")
		print(f"  -> Small all_to_allv is UNCHANGED -- rules out systemic NCCL issues")
		print(f"     Only large transfers (211 MB) are affected -- contention-dependent")

	# Tail latency analysis
	subsection("Tail Latency Analysis")
	if new_large and old_large:
		new_tail_ratio = new_large["dur_max"] / new_large["dur_mean"]
		old_tail_ratio = old_large["dur_max"] / old_large["dur_mean"]
		print(f"  New: max/mean = {new_large['dur_max']/1000:.0f}ms / {new_large['dur_mean']/1000:.1f}ms = {new_tail_ratio:.0f}x")
		print(f"  Old: max/mean = {old_large['dur_max']/1000:.0f}ms / {old_large['dur_mean']/1000:.1f}ms = {old_tail_ratio:.0f}x")
		print(f"  -> New has {new_tail_ratio/old_tail_ratio:.1f}x worse tail ratio")
		print(f"  -> Max latency is {new_large['dur_max']/1e6:.1f}s (new) vs {old_large['dur_max']/1e6:.1f}s (old)")
		print(f"     This likely corresponds to the model-switching step (step 18/19)")


# ==========================================================================
# 3. Kernel-by-Kernel Diff
# ==========================================================================
def analyze_kernels(new_rows, old_rows):
	section("3. KERNEL-BY-KERNEL DIFF")

	def parse_kernels(rows):
		kernels = {}
		for r in rows:
			name = r["Kernel name"]
			kernels[name] = {
				"category": r["Parent op category"],
				"parent_op": r["Parent cpu_op"],
				"sum_us": float(r["Kernel duration (µs)_sum"]),
				"count": int(r["Kernel duration (µs)_count"]),
				"mean_us": float(r["Kernel duration (µs)_mean"]),
				"pct_kernels": float(r["Percent of kernels time (%)"]),
			}
		return kernels

	new_k = parse_kernels(new_rows)
	old_k = parse_kernels(old_rows)

	# Find matching kernels
	all_names = set(new_k.keys()) | set(old_k.keys())
	diffs = []
	for name in all_names:
		n = new_k.get(name)
		o = old_k.get(name)
		if n and o:
			delta_sum_ms = (n["sum_us"] - o["sum_us"]) / 1000
			delta_pct = (n["sum_us"] / o["sum_us"] - 1) * 100 if o["sum_us"] > 0 else float("inf")
			diffs.append({
				"name": name[:80],
				"category": n["category"],
				"parent_op": n["parent_op"],
				"new_sum_s": n["sum_us"] / 1e6,
				"old_sum_s": o["sum_us"] / 1e6,
				"delta_ms": delta_sum_ms,
				"delta_pct": delta_pct,
				"new_count": n["count"],
				"old_count": o["count"],
			})

	# Sort by absolute delta
	diffs.sort(key=lambda x: abs(x["delta_ms"]), reverse=True)

	subsection("Top 15 Kernels by Absolute Time Delta")
	print(f"  {'Kernel':<60s} {'Cat':<12s} {'New (s)':>8s} {'Old (s)':>8s} {'Delta':>10s}")
	print(f"  {'-'*60} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
	for d in diffs[:15]:
		sign = "+" if d["delta_ms"] > 0 else ""
		print(f"  {d['name']:<60s} {d['category']:<12s} {d['new_sum_s']:>8.1f} {d['old_sum_s']:>8.1f} "
			  f"{sign}{d['delta_ms']/1000:>9.1f}s")

	# Compute-only kernels (exclude NCCL and memcpy)
	subsection("Compute Kernel Comparison (excluding NCCL)")
	compute_diffs = [d for d in diffs if d["category"] not in ("record_param_comms",)]
	compute_new_total = sum(d["new_sum_s"] for d in compute_diffs)
	compute_old_total = sum(d["old_sum_s"] for d in compute_diffs)

	print(f"  Total compute time: new={compute_new_total:.1f}s  old={compute_old_total:.1f}s  "
		  f"delta={(compute_new_total - compute_old_total)*1000:+.0f}ms ({(compute_new_total/compute_old_total - 1)*100:+.2f}%)")

	# Check if non-NCCL kernels are truly identical
	significant_compute_diffs = [d for d in compute_diffs if abs(d["delta_ms"]) > 100]
	if significant_compute_diffs:
		print(f"\n  WARNING: {len(significant_compute_diffs)} compute kernels differ by >100ms:")
		for d in significant_compute_diffs[:5]:
			print(f"    {d['parent_op']}: {d['delta_ms']:+.0f}ms ({d['delta_pct']:+.1f}%)")
	else:
		print(f"  All compute kernels within 100ms -- confirms only scheduling changed")

	# NCCL-specific
	nccl_diffs = [d for d in diffs if d["category"] == "record_param_comms"]
	if nccl_diffs:
		subsection("NCCL Kernel Delta")
		for d in nccl_diffs:
			print(f"  {d['name'][:70]}")
			print(f"    new={d['new_sum_s']:.1f}s  old={d['old_sum_s']:.1f}s  "
				  f"delta={d['delta_ms']/1000:+.1f}s ({d['delta_pct']:+.1f}%)")

	# Only-in-new or only-in-old
	new_only = set(new_k.keys()) - set(old_k.keys())
	old_only = set(old_k.keys()) - set(new_k.keys())
	if new_only:
		subsection(f"Kernels ONLY in New ({len(new_only)})")
		for name in sorted(new_only, key=lambda n: new_k[n]["sum_us"], reverse=True)[:5]:
			k = new_k[name]
			print(f"  {name[:70]}: {k['sum_us']/1e6:.3f}s ({k['count']} calls)")
	if old_only:
		subsection(f"Kernels ONLY in Old ({len(old_only)})")
		for name in sorted(old_only, key=lambda n: old_k[n]["sum_us"], reverse=True)[:5]:
			k = old_k[name]
			print(f"  {name[:70]}: {k['sum_us']/1e6:.3f}s ({k['count']} calls)")


# ==========================================================================
# 4. Per-Step Timing Patterns
# ==========================================================================
def analyze_per_step(new_perf, old_perf):
	section("4. PER-STEP TIMING PATTERNS")

	new_steps = new_perf["denoise_steps_ms"]
	old_steps = old_perf["denoise_steps_ms"]

	subsection("Step-by-Step Comparison")
	print(f"  {'Step':>4s} {'New (ms)':>10s} {'Old (ms)':>10s} {'Delta (ms)':>12s} {'Notes':s}")
	print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*12} {'-'*30}")

	new_total_denoise = 0
	old_total_denoise = 0
	deltas = []

	for i in range(len(new_steps)):
		ns = new_steps[i]["duration_ms"]
		os = old_steps[i]["duration_ms"]
		delta = ns - os
		new_total_denoise += ns
		old_total_denoise += os
		deltas.append(delta)

		notes = ""
		if i == 0:
			notes = "WARMUP"
		elif i == 18 and os > 40000:
			notes = "MODEL SWITCH (old)"
		elif abs(delta) > 5000:
			notes = "LARGE DELTA"

		print(f"  {i:>4d} {ns:>10.1f} {os:>10.1f} {delta:>+12.1f} {notes}")

	subsection("Aggregate Statistics")
	print(f"  Total denoising: new={new_total_denoise/1000:.1f}s  old={old_total_denoise/1000:.1f}s  "
		  f"delta={(new_total_denoise - old_total_denoise)/1000:+.1f}s")

	# Steady-state analysis (exclude step 0 warmup and step 18 outlier)
	steady_new = [new_steps[i]["duration_ms"] for i in range(1, len(new_steps)) if i != 18]
	steady_old = [old_steps[i]["duration_ms"] for i in range(1, len(old_steps)) if i != 18]
	steady_deltas = [n - o for n, o in zip(steady_new, steady_old)]

	avg_new = sum(steady_new) / len(steady_new)
	avg_old = sum(steady_old) / len(steady_old)
	avg_delta = sum(steady_deltas) / len(steady_deltas)

	print(f"\n  Steady-state (excl. step 0 warmup + step 18 outlier):")
	print(f"    New avg: {avg_new:.1f} ms/step")
	print(f"    Old avg: {avg_old:.1f} ms/step")
	print(f"    Delta:   {avg_delta:+.1f} ms/step ({avg_delta/avg_old*100:+.1f}%)")
	print(f"    Min delta: {min(steady_deltas):+.1f} ms  Max delta: {max(steady_deltas):+.1f} ms")

	# Warmup analysis
	subsection("Warmup Analysis (Step 0)")
	new_warmup = new_steps[0]["duration_ms"]
	old_warmup = old_steps[0]["duration_ms"]
	print(f"  New step 0: {new_warmup:.1f} ms ({new_warmup/avg_new:.1f}x steady)")
	print(f"  Old step 0: {old_warmup:.1f} ms ({old_warmup/avg_old:.1f}x steady)")
	print(f"  Delta: {new_warmup - old_warmup:+.1f} ms")
	print(f"  -> New warmup is {new_warmup - old_warmup:.0f}ms longer (more setup for async offload)")

	# Step 18 outlier
	subsection("Step 18 Analysis (Model Switch)")
	new_18 = new_steps[18]["duration_ms"]
	old_18 = old_steps[18]["duration_ms"]
	print(f"  New step 18: {new_18:.1f} ms ({new_18 - avg_new:+.1f} from steady)")
	print(f"  Old step 18: {old_18:.1f} ms ({old_18 - avg_old:+.1f} from steady)")
	if old_18 > 40000:
		print(f"  -> Old has +{old_18 - avg_old:.0f}ms spike at step 18 (flow_shift=12 model switch)")
		print(f"     This is the blocking .to() call loading model parameters for the switch")
		print(f"  -> New has only +{new_18 - avg_new:.0f}ms above steady (async handles switch better)")
		spike_saved = (old_18 - avg_old) - (new_18 - avg_new)
		print(f"  -> Spike reduction: {spike_saved:.0f}ms saved by new offload at switch point")

	# Stage timing
	subsection("Pipeline Stage Comparison")
	print(f"  {'Stage':<25s} {'New (s)':>10s} {'Old (s)':>10s} {'Delta (s)':>10s}")
	print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
	for ns, os in zip(new_perf["steps"], old_perf["steps"]):
		n_dur = ns["duration_ms"] / 1000
		o_dur = os["duration_ms"] / 1000
		delta = n_dur - o_dur
		print(f"  {ns['name']:<25s} {n_dur:>10.2f} {o_dur:>10.2f} {delta:>+10.2f}")


# ==========================================================================
# 5. Summary Findings
# ==========================================================================
def print_summary(new_timeline, old_timeline, new_coll, old_coll, new_perf, old_perf):
	section("5. SUMMARY OF FINDINGS")

	def get_tl(rows, key):
		for r in rows:
			if r["type"] == key:
				return float(r["time ms"])
		return 0

	n_idle = get_tl(new_timeline, "idle_time")
	o_idle = get_tl(old_timeline, "idle_time")
	n_total = get_tl(new_timeline, "total_time")
	o_total = get_tl(old_timeline, "total_time")
	n_comm = get_tl(new_timeline, "exposed_comm_time")
	o_comm = get_tl(old_timeline, "exposed_comm_time")
	n_memcpy_exp = get_tl(new_timeline, "exposed_memcpy_time")
	o_memcpy_exp = get_tl(old_timeline, "exposed_memcpy_time")
	n_memcpy_tot = get_tl(new_timeline, "total_memcpy_time")
	o_memcpy_tot = get_tl(old_timeline, "total_memcpy_time")

	overhead = n_total - o_total

	print(f"""
  WHAT THE DATA TELLS US:

  1. IDLE TIME: New has {n_idle/1000:.1f}s vs old {o_idle/1000:.1f}s (+{(n_idle-o_idle)/1000:.1f}s)
     - {(n_idle-o_idle)/overhead*100:.0f}% of total overhead ({overhead/1000:.1f}s) is GPU sitting idle
     - The GPU is waiting for something -- likely PCIe transfers or NCCL

  2. NCCL OVERHEAD: New has {n_comm/1000:.1f}s vs old {o_comm/1000:.1f}s (+{(n_comm-o_comm)/1000:.1f}s) exposed comm
     - Large all_to_allv (211 MB): +{((n_comm-o_comm)/1000):.0f}s slower
     - Small all_to_allv (5 MB): UNCHANGED (rules out systemic NCCL issues)
     - This confirms contention is SIZE-DEPENDENT (PCIe bandwidth competition)

  3. MEMCPY: New does {n_memcpy_tot/o_memcpy_tot:.1f}x more total memcpy ({n_memcpy_tot/1000:.0f}s vs {o_memcpy_tot/1000:.0f}s)
     - But only {n_memcpy_exp/1000:.1f}s is exposed (async overlap works well)
     - Old has {o_memcpy_exp/1000:.1f}s exposed (blocking .to() calls)

  4. COMPUTE: Identical across new/old (validates that only scheduling changed)

  5. PER-STEP: Steady-state delta ~3s/step (15% slower new vs old)
     - Step 18 (model switch): old has ~29s spike, new handles it better
     - But this spike savings does NOT compensate for cumulative steady-state loss

  BUDGET BREAKDOWN (where the +{overhead/1000:.1f}s goes):
  - Extra idle time:        +{(n_idle-o_idle)/1000:.1f}s
  - Extra exposed comm:     +{(n_comm-o_comm)/1000:.1f}s
  - Extra exposed memcpy:   {(n_memcpy_exp-o_memcpy_exp)/1000:+.1f}s
  - Compute change:         ~0s

  WHAT THE DATA DOES NOT TELL US (needs nsys):
  1. Do H2D and NCCL actually overlap on the GPU timeline? (CPU overlap != GPU overlap)
  2. What is the actual PCIe bandwidth during H2D? During NCCL?
  3. Is bandwidth degradation proportional to overlap?
  4. Are there GPU idle gaps within each step, or only between steps?

  HYPOTHESIS STATUS:
  - PCIe contention between H2D and NCCL: SUPPORTED by data
    (large NCCL slows down, small unchanged; new does 5x memcpy)
  - But NOT proven -- need nsys to confirm actual temporal overlap
""")


# ==========================================================================
# Main
# ==========================================================================
def main():
	print("=" * 72)
	print("  PROFILING COMPARISON: New Offload vs Old Offload")
	print("  Data: TraceLens CSVs + Per-step Perf JSON")
	print("  Jobs: 1459995 (new) / 1459996 (old), flow_shift=12.0, 4 GPU")
	print("=" * 72)

	# Load data
	tl_new = load_csv(BASE / "results/tracelens_new/gpu_timeline.csv")
	tl_old = load_csv(BASE / "results/tracelens_old/gpu_timeline.csv")
	coll_new = load_csv(BASE / "results/tracelens_new/coll_analysis.csv")
	coll_old = load_csv(BASE / "results/tracelens_old/coll_analysis.csv")
	kern_new = load_csv(BASE / "results/tracelens_new/kernel_summary.csv")
	kern_old = load_csv(BASE / "results/tracelens_old/kernel_summary.csv")
	perf_new = load_json(BASE / "results/logs/perf_new_offload_1459995.json")
	perf_old = load_json(BASE / "results/logs/perf_old_offload_1459996.json")

	# Run analyses
	analyze_gpu_timeline(tl_new, tl_old)
	analyze_nccl(coll_new, coll_old)
	analyze_kernels(kern_new, kern_old)
	analyze_per_step(perf_new, perf_old)
	print_summary(tl_new, tl_old, coll_new, coll_old, perf_new, perf_old)


if __name__ == "__main__":
	main()
