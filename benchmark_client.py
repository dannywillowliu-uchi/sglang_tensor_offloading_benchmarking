#!/usr/bin/env python3
"""
Benchmark client for SGLang Wan2.2 video generation.
Sends requests to the SGLang server and measures timing.
Includes GPU memory and utilization sampling.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import requests


def get_gpu_stats():
	"""Query nvidia-smi for GPU utilization and memory usage."""
	try:
		result = subprocess.run(
			[
				"nvidia-smi",
				"--query-gpu=index,utilization.gpu,memory.used,memory.total",
				"--format=csv,noheader,nounits",
			],
			capture_output=True,
			text=True,
			timeout=5,
		)
		if result.returncode != 0:
			return None

		gpus = []
		for line in result.stdout.strip().split("\n"):
			parts = [p.strip() for p in line.split(",")]
			if len(parts) >= 4:
				gpus.append({
					"index": int(parts[0]),
					"utilization_pct": float(parts[1]),
					"memory_used_mb": float(parts[2]),
					"memory_total_mb": float(parts[3]),
				})
		return gpus
	except Exception:
		return None


class GPUSampler:
	"""Background thread to sample GPU stats during inference."""

	def __init__(self, interval: float = 0.5):
		self.interval = interval
		self.samples = []
		self._stop = False
		self._thread = None

	def start(self):
		"""Start sampling in background thread."""
		self._stop = False
		self.samples = []
		self._thread = threading.Thread(target=self._sample_loop, daemon=True)
		self._thread.start()

	def stop(self):
		"""Stop sampling and return collected samples."""
		self._stop = True
		if self._thread:
			self._thread.join(timeout=2)
		return self.samples

	def _sample_loop(self):
		"""Sampling loop running in background."""
		while not self._stop:
			stats = get_gpu_stats()
			if stats:
				self.samples.append({
					"timestamp": time.perf_counter(),
					"gpus": stats,
				})
			time.sleep(self.interval)

	def get_summary(self):
		"""Calculate summary statistics from samples."""
		if not self.samples:
			return {}

		# Aggregate across all samples and GPUs
		all_utils = []
		all_mem_used = []
		all_mem_total = []

		for sample in self.samples:
			for gpu in sample["gpus"]:
				all_utils.append(gpu["utilization_pct"])
				all_mem_used.append(gpu["memory_used_mb"])
				all_mem_total.append(gpu["memory_total_mb"])

		return {
			"num_samples": len(self.samples),
			"avg_gpu_utilization_pct": sum(all_utils) / len(all_utils) if all_utils else 0,
			"max_gpu_utilization_pct": max(all_utils) if all_utils else 0,
			"min_gpu_utilization_pct": min(all_utils) if all_utils else 0,
			"avg_memory_used_mb": sum(all_mem_used) / len(all_mem_used) if all_mem_used else 0,
			"peak_memory_used_mb": max(all_mem_used) if all_mem_used else 0,
			"memory_total_mb": all_mem_total[0] if all_mem_total else 0,
		}


def wait_for_server(base_url: str, timeout: int = 300):
	"""Wait for the SGLang server to be ready."""
	print(f"Waiting for server at {base_url}...")
	start = time.time()

	while time.time() - start < timeout:
		try:
			resp = requests.get(f"{base_url}/health", timeout=5)
			if resp.status_code == 200:
				print("Server is ready!")
				return True
		except requests.exceptions.RequestException:
			pass
		time.sleep(5)

	raise TimeoutError(f"Server not ready after {timeout}s")


def generate_video(
	base_url: str,
	prompt: str,
	width: int = 1280,
	height: int = 720,
	num_frames: int = 81,
	num_inference_steps: int = 27,
	gpu_sampler: GPUSampler = None,
):
	"""Send video generation request and return timing info."""
	url = f"{base_url}/v1/videos"

	payload = {
		"prompt": prompt,
		"size": f"{width}x{height}",
		"num_frames": num_frames,
		"num_inference_steps": num_inference_steps,
	}

	# Start GPU sampling if provided
	if gpu_sampler:
		gpu_sampler.start()

	start = time.perf_counter()
	resp = requests.post(url, json=payload, timeout=600)
	elapsed = time.perf_counter() - start

	# Stop GPU sampling
	gpu_stats = None
	if gpu_sampler:
		gpu_sampler.stop()
		gpu_stats = gpu_sampler.get_summary()

	if resp.status_code != 200:
		raise RuntimeError(f"Generation failed: {resp.status_code} - {resp.text}")

	return {
		"elapsed_seconds": elapsed,
		"time_per_step": elapsed / num_inference_steps,
		"gpu_stats": gpu_stats,
		"response": resp.json(),
	}


def run_benchmark(
	base_url: str,
	prompt: str = "A serene lake with mountains in the background at sunset, cinematic quality",
	width: int = 1280,
	height: int = 720,
	num_frames: int = 81,
	num_inference_steps: int = 27,
	num_warmup: int = 1,
	num_runs: int = 3,
	output_dir: str = "./results",
	config_name: str = "benchmark",
):
	"""Run video generation benchmark."""

	# Get initial GPU state
	initial_gpu_stats = get_gpu_stats()

	results = {
		"config_name": config_name,
		"base_url": base_url,
		"prompt": prompt,
		"width": width,
		"height": height,
		"num_frames": num_frames,
		"num_inference_steps": num_inference_steps,
		"num_warmup": num_warmup,
		"num_runs": num_runs,
		"timestamp": datetime.now().isoformat(),
		"initial_gpu_stats": initial_gpu_stats,
		"warmup_times": [],
		"warmup_gpu_stats": [],
		"run_times": [],
		"run_gpu_stats": [],
	}

	print("=" * 60)
	print(f"BENCHMARK: {config_name}")
	print("=" * 60)
	print(f"Server: {base_url}")
	print(f"Prompt: {prompt}")
	print(f"Resolution: {width}x{height}")
	print(f"Frames: {num_frames}, Steps: {num_inference_steps}")
	print(f"Warmup: {num_warmup}, Runs: {num_runs}")
	if initial_gpu_stats:
		print(f"GPUs detected: {len(initial_gpu_stats)}")
		for gpu in initial_gpu_stats:
			print(f"  GPU {gpu['index']}: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB")
	print("-" * 60)

	# Wait for server
	wait_for_server(base_url)

	# Create GPU sampler
	gpu_sampler = GPUSampler(interval=0.5)

	# Warmup runs
	for i in range(num_warmup):
		print(f"Warmup run {i+1}/{num_warmup}...")
		result = generate_video(
			base_url, prompt, width, height, num_frames, num_inference_steps,
			gpu_sampler=gpu_sampler
		)
		results["warmup_times"].append(result["elapsed_seconds"])
		results["warmup_gpu_stats"].append(result["gpu_stats"])
		print(f"  Time: {result['elapsed_seconds']:.2f}s")
		if result["gpu_stats"]:
			print(f"  Avg GPU util: {result['gpu_stats']['avg_gpu_utilization_pct']:.1f}%")
			print(f"  Peak memory: {result['gpu_stats']['peak_memory_used_mb']:.0f} MB")

	# Benchmark runs
	for i in range(num_runs):
		print(f"Benchmark run {i+1}/{num_runs}...")
		result = generate_video(
			base_url, prompt, width, height, num_frames, num_inference_steps,
			gpu_sampler=gpu_sampler
		)
		results["run_times"].append(result["elapsed_seconds"])
		results["run_gpu_stats"].append(result["gpu_stats"])
		print(f"  Time: {result['elapsed_seconds']:.2f}s")
		if result["gpu_stats"]:
			print(f"  Avg GPU util: {result['gpu_stats']['avg_gpu_utilization_pct']:.1f}%")
			print(f"  Peak memory: {result['gpu_stats']['peak_memory_used_mb']:.0f} MB")

	# Calculate statistics
	run_times = results["run_times"]
	results["avg_time"] = sum(run_times) / len(run_times)
	results["min_time"] = min(run_times)
	results["max_time"] = max(run_times)
	results["avg_time_per_step"] = results["avg_time"] / num_inference_steps

	# Aggregate GPU stats across runs
	if results["run_gpu_stats"] and results["run_gpu_stats"][0]:
		gpu_utils = [s["avg_gpu_utilization_pct"] for s in results["run_gpu_stats"] if s]
		peak_mems = [s["peak_memory_used_mb"] for s in results["run_gpu_stats"] if s]
		results["overall_avg_gpu_utilization_pct"] = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
		results["overall_peak_memory_mb"] = max(peak_mems) if peak_mems else 0

	print("-" * 60)
	print("Results:")
	print(f"  Avg time: {results['avg_time']:.2f}s")
	print(f"  Min time: {results['min_time']:.2f}s")
	print(f"  Max time: {results['max_time']:.2f}s")
	print(f"  Avg time per step: {results['avg_time_per_step']:.2f}s")
	if "overall_avg_gpu_utilization_pct" in results:
		print(f"  Avg GPU utilization: {results['overall_avg_gpu_utilization_pct']:.1f}%")
		print(f"  Peak GPU memory: {results['overall_peak_memory_mb']:.0f} MB")

	# Save results
	os.makedirs(output_dir, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_file = os.path.join(output_dir, f"{config_name}_{timestamp}.json")

	with open(output_file, "w") as f:
		json.dump(results, f, indent=2)

	print(f"\nResults saved to: {output_file}")
	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmark SGLang video generation")
	parser.add_argument("--base-url", default="http://localhost:30010", help="SGLang server URL")
	parser.add_argument("--prompt", default="A serene lake with mountains in the background at sunset, cinematic quality")
	parser.add_argument("--width", type=int, default=1280)
	parser.add_argument("--height", type=int, default=720)
	parser.add_argument("--num-frames", type=int, default=81)
	parser.add_argument("--num-inference-steps", type=int, default=27)
	parser.add_argument("--num-warmup", type=int, default=1)
	parser.add_argument("--num-runs", type=int, default=3)
	parser.add_argument("--output-dir", default="./results")
	parser.add_argument("--config-name", default="benchmark", help="Name for this config")

	args = parser.parse_args()
	run_benchmark(
		base_url=args.base_url,
		prompt=args.prompt,
		width=args.width,
		height=args.height,
		num_frames=args.num_frames,
		num_inference_steps=args.num_inference_steps,
		num_warmup=args.num_warmup,
		num_runs=args.num_runs,
		output_dir=args.output_dir,
		config_name=args.config_name,
	)
