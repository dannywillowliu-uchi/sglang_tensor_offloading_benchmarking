#!/usr/bin/env python3
"""
Benchmark client for SGLang Wan2.2 video generation.
Sends requests to the SGLang server and measures timing.
"""

import argparse
import json
import os
import time
from datetime import datetime

import requests


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
):
	"""Send video generation request and return timing info."""
	url = f"{base_url}/v1/videos"

	payload = {
		"prompt": prompt,
		"size": f"{width}x{height}",
		"num_frames": num_frames,
		"num_inference_steps": num_inference_steps,
	}

	start = time.perf_counter()
	resp = requests.post(url, json=payload, timeout=600)
	elapsed = time.perf_counter() - start

	if resp.status_code != 200:
		raise RuntimeError(f"Generation failed: {resp.status_code} - {resp.text}")

	return {
		"elapsed_seconds": elapsed,
		"time_per_step": elapsed / num_inference_steps,
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
		"warmup_times": [],
		"run_times": [],
	}

	print("=" * 60)
	print(f"BENCHMARK: {config_name}")
	print("=" * 60)
	print(f"Server: {base_url}")
	print(f"Prompt: {prompt}")
	print(f"Resolution: {width}x{height}")
	print(f"Frames: {num_frames}, Steps: {num_inference_steps}")
	print(f"Warmup: {num_warmup}, Runs: {num_runs}")
	print("-" * 60)

	# Wait for server
	wait_for_server(base_url)

	# Warmup runs
	for i in range(num_warmup):
		print(f"Warmup run {i+1}/{num_warmup}...")
		result = generate_video(
			base_url, prompt, width, height, num_frames, num_inference_steps
		)
		results["warmup_times"].append(result["elapsed_seconds"])
		print(f"  Time: {result['elapsed_seconds']:.2f}s")

	# Benchmark runs
	for i in range(num_runs):
		print(f"Benchmark run {i+1}/{num_runs}...")
		result = generate_video(
			base_url, prompt, width, height, num_frames, num_inference_steps
		)
		results["run_times"].append(result["elapsed_seconds"])
		print(f"  Time: {result['elapsed_seconds']:.2f}s")

	# Calculate statistics
	run_times = results["run_times"]
	results["avg_time"] = sum(run_times) / len(run_times)
	results["min_time"] = min(run_times)
	results["max_time"] = max(run_times)
	results["avg_time_per_step"] = results["avg_time"] / num_inference_steps

	print("-" * 60)
	print("Results:")
	print(f"  Avg time: {results['avg_time']:.2f}s")
	print(f"  Min time: {results['min_time']:.2f}s")
	print(f"  Max time: {results['max_time']:.2f}s")
	print(f"  Avg time per step: {results['avg_time_per_step']:.2f}s")

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
