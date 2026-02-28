"""Survey Chrome Trace event names to find memory transfer events.

The trace format is: { "schemaVersion":..., "traceEvents": [ {event}, {event}, ... ] }
We stream-parse by reading chunks, finding the traceEvents array, then parsing
individual event objects with a bracket-counting approach.
"""

import gzip
import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path


# Keywords to flag as potential memory transfer events
TRANSFER_KEYWORDS = [
	"copy", "transfer", "memcpy", "h2d", "d2h", "htod", "dtoh",
	"move", "offload", "to_device", "cuda_runtime", "pin_memory",
	"pageable", "pinned", "dma", "pcie", "to(",
]

# Categories of interest
INTEREST_CATS = [
	"cuda_runtime", "gpu_memcpy", "kernel", "cuda_driver",
	"gpu_memset", "memcpy", "runtime",
]

MAX_EVENTS = 5_000_000
MAX_ATEN_COPY_SAMPLES = 10
MAX_TRANSFER_SAMPLES_PER_NAME = 3


def matches_transfer(name: str) -> bool:
	name_lower = name.lower()
	return any(kw in name_lower for kw in TRANSFER_KEYWORDS)


def matches_interest_cat(cat: str) -> bool:
	cat_lower = cat.lower()
	return any(c.lower() in cat_lower for c in INTEREST_CATS)


def stream_events(path: str):
	"""Stream-parse events from a Chrome Trace JSON file (possibly truncated gzip).

	Strategy: read chunks, find start of traceEvents array, then use a simple
	state machine to extract individual JSON objects.
	"""
	event_count = 0
	buf = ""
	found_array = False
	CHUNK_SIZE = 1024 * 1024  # 1 MB

	try:
		with gzip.open(path, "rt", encoding="utf-8") as f:
			while True:
				chunk = f.read(CHUNK_SIZE)
				if not chunk:
					break
				buf += chunk

				# Find the start of traceEvents array
				if not found_array:
					idx = buf.find('"traceEvents"')
					if idx < 0:
						# Keep only last 100 chars in case key spans chunks
						buf = buf[-100:]
						continue
					# Find the opening bracket
					bracket_idx = buf.find("[", idx)
					if bracket_idx < 0:
						continue
					buf = buf[bracket_idx + 1:]  # skip past the '['
					found_array = True

				# Now parse events from buf
				# Events are separated by commas, each is a {...} object
				while True:
					# Find start of next object
					obj_start = buf.find("{")
					if obj_start < 0:
						break

					# Count braces to find matching close
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
						# Incomplete object, need more data
						break

					# Extract and parse
					obj_str = buf[obj_start:obj_end]
					buf = buf[obj_end:]

					try:
						event = json.loads(obj_str)
						if isinstance(event, dict) and "ph" in event:
							event_count += 1
							yield event
							if event_count >= MAX_EVENTS:
								print(f"  [Reached {MAX_EVENTS} event limit]")
								return
					except json.JSONDecodeError:
						pass

				# Keep only unprocessed remainder (could be partial object)
				# If buf is very large and we haven't found an object, something's wrong
				if len(buf) > 10 * CHUNK_SIZE:
					print(f"  [Warning: buffer grew to {len(buf)}, trimming]")
					buf = buf[-CHUNK_SIZE:]

	except EOFError:
		print(f"  [Hit EOFError after {event_count} events - truncated gzip]")
	except Exception as e:
		print(f"  [Error after {event_count} events: {type(e).__name__}: {e}]")

	print(f"  [Finished: {event_count} events parsed]")


def survey_trace(trace_path: str):
	print(f"\n{'='*80}")
	print(f"TRACE: {trace_path}")
	print(f"{'='*80}")

	all_names = Counter()
	all_cats = Counter()
	cat_by_ph = defaultdict(Counter)
	transfer_events = defaultdict(list)  # name -> [sample events]
	interest_cat_events = defaultdict(list)  # cat -> [sample events]
	aten_copy_events = []
	aten_to_events = []
	total_events = 0

	# Track (cat, name) pairs for transfer-like events
	transfer_cat_name_pairs = Counter()

	# Track events with bytes/size in args
	bytes_in_args_samples = []

	for event in stream_events(trace_path):
		total_events += 1
		name = event.get("name", "")
		cat = event.get("cat", "")
		ph = event.get("ph", "")

		all_names[name] += 1
		if cat:
			all_cats[cat] += 1
		if ph:
			cat_by_ph[ph][cat if cat else "<no cat>"] += 1

		# Check for transfer-related event names
		if matches_transfer(name):
			transfer_cat_name_pairs[(cat, name)] += 1
			if len(transfer_events[name]) < MAX_TRANSFER_SAMPLES_PER_NAME:
				transfer_events[name].append(event)

		# Check for interesting categories
		if cat and matches_interest_cat(cat):
			if len(interest_cat_events[cat]) < MAX_TRANSFER_SAMPLES_PER_NAME:
				interest_cat_events[cat].append(event)

		# Collect aten::copy_ events
		if name == "aten::copy_" and len(aten_copy_events) < MAX_ATEN_COPY_SAMPLES:
			aten_copy_events.append(event)

		# Collect aten::to events
		if name == "aten::to" and len(aten_to_events) < 5:
			aten_to_events.append(event)

		# Check for bytes/size in args
		args = event.get("args", {})
		if isinstance(args, dict) and len(bytes_in_args_samples) < 10:
			for k in args:
				kl = k.lower()
				if "byte" in kl or "size" in kl or "memcpy" in kl:
					bytes_in_args_samples.append(event)
					break

		if total_events % 500000 == 0:
			print(f"  ... {total_events:,} events processed")

	# --- Print results ---
	print(f"\nTotal events parsed: {total_events:,}")
	print(f"Unique event names: {len(all_names):,}")
	print(f"Unique categories: {len(all_cats):,}")

	# Top 80 event names by count
	print(f"\n--- TOP 80 EVENT NAMES (by count) ---")
	for name, count in all_names.most_common(80):
		print(f"  {count:>8,}  {name}")

	# All categories
	print(f"\n--- ALL CATEGORIES (by count) ---")
	for cat, count in all_cats.most_common():
		print(f"  {count:>8,}  {cat}")

	# Phase breakdown
	print(f"\n--- EVENT PHASES ---")
	for ph, cats in sorted(cat_by_ph.items()):
		total_ph = sum(cats.values())
		print(f"  ph={ph!r}: {total_ph:,} events")
		for c, cnt in cats.most_common(10):
			print(f"    {cnt:>8,}  cat={c!r}")

	# Transfer-related events
	print(f"\n--- TRANSFER-RELATED EVENT NAMES ---")
	if transfer_cat_name_pairs:
		for (cat, name), count in sorted(transfer_cat_name_pairs.items(), key=lambda x: -x[1]):
			print(f"  {count:>8,}  name={name!r}  cat={cat!r}")
	else:
		print("  (none found)")

	# Sample transfer events
	print(f"\n--- SAMPLE TRANSFER EVENTS (full JSON) ---")
	if transfer_events:
		for name, samples in sorted(transfer_events.items()):
			print(f"\n  Event name: {name!r} ({transfer_cat_name_pairs.get(('cpu_op', name), 0) + transfer_cat_name_pairs.get(('', name), 0)} total)")
			for i, ev in enumerate(samples):
				print(f"    Sample {i+1}: {json.dumps(ev, indent=6)}")
	else:
		print("  (none found)")

	# Interest category events
	print(f"\n--- EVENTS WITH INTERESTING CATEGORIES ---")
	if interest_cat_events:
		for cat, samples in sorted(interest_cat_events.items()):
			total_cat = all_cats.get(cat, 0)
			print(f"\n  Category: {cat!r} ({total_cat:,} total events)")
			for i, ev in enumerate(samples):
				print(f"    Sample {i+1}: {json.dumps(ev, indent=6)}")
	else:
		print("  (none found)")

	# aten::copy_ events
	print(f"\n--- aten::copy_ EVENTS ({len(aten_copy_events)} samples, {all_names.get('aten::copy_', 0):,} total) ---")
	if aten_copy_events:
		for i, ev in enumerate(aten_copy_events):
			print(f"  Sample {i+1}: {json.dumps(ev, indent=4)}")
	else:
		print("  (none found)")

	# aten::to events
	print(f"\n--- aten::to EVENTS ({len(aten_to_events)} samples, {all_names.get('aten::to', 0):,} total) ---")
	if aten_to_events:
		for i, ev in enumerate(aten_to_events):
			print(f"  Sample {i+1}: {json.dumps(ev, indent=4)}")
	else:
		print("  (none found)")

	# Events with bytes/size in args
	print(f"\n--- EVENTS WITH 'bytes'/'size' IN ARGS ---")
	if bytes_in_args_samples:
		for i, ev in enumerate(bytes_in_args_samples):
			print(f"  Sample {i+1}: {json.dumps(ev, indent=4)}")
	else:
		print("  (none found)")

	return all_names, all_cats


def main():
	traces = [
		"/Users/dannyliu/research_work/results/traces/new_offload_1434113.trace.json.gz",
		"/Users/dannyliu/research_work/results/traces/old_offload_1434114.trace.json.gz",
	]

	all_results = {}
	for trace_path in traces:
		p = Path(trace_path)
		if not p.exists():
			print(f"SKIP: {trace_path} not found")
			continue
		names, cats = survey_trace(trace_path)
		all_results[p.name] = (names, cats)

	# Cross-file comparison
	if len(all_results) == 2:
		print(f"\n{'='*80}")
		print("CROSS-FILE COMPARISON")
		print(f"{'='*80}")
		files = list(all_results.keys())
		names_a, _ = all_results[files[0]]
		names_b, _ = all_results[files[1]]

		only_a = set(names_a.keys()) - set(names_b.keys())
		only_b = set(names_b.keys()) - set(names_a.keys())

		# Transfer-related unique to each
		print(f"\n  Transfer-related events unique to each file:")
		for n in sorted(only_a):
			if matches_transfer(n):
				print(f"    {files[0]} only: {names_a[n]:>6,}  {n}")
		for n in sorted(only_b):
			if matches_transfer(n):
				print(f"    {files[1]} only: {names_b[n]:>6,}  {n}")

		if only_a:
			print(f"\n  ALL events ONLY in {files[0]} ({len(only_a)}):")
			for n in sorted(only_a)[:50]:
				print(f"    {names_a[n]:>6,}  {n}")
		if only_b:
			print(f"\n  ALL events ONLY in {files[1]} ({len(only_b)}):")
			for n in sorted(only_b)[:50]:
				print(f"    {names_b[n]:>6,}  {n}")

		# Compare counts for key events
		print(f"\n  COUNT COMPARISON for key events:")
		key_events = ["aten::copy_", "aten::to", "aten::empty", "cudaMemcpyAsync",
		              "Memcpy HtoD", "Memcpy DtoH", "cudaMemcpy",
		              "nccl:all_reduce", "nccl:all_to_all"]
		for name in key_events:
			ca = names_a.get(name, 0)
			cb = names_b.get(name, 0)
			if ca > 0 or cb > 0:
				print(f"    {name:40s}  {files[0]}: {ca:>8,}  {files[1]}: {cb:>8,}")


if __name__ == "__main__":
	main()
