#!/usr/bin/env python3
"""
Benchmark script for SGLang Wan2.2 video generation.
Measures inference time, per-step latency, and peak VRAM usage.
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import sglang as sgl
from sglang import VideoGenerationInput, set_default_backend, GenerationConfig


def get_gpu_memory_usage():
	"""Get current and peak GPU memory usage in GB."""
	if not torch.cuda.is_available():
		return 0, 0

	current = torch.cuda.memory_allocated() / 1024**3
	peak = torch.cuda.max_memory_allocated() / 1024**3
	return current, peak


def reset_peak_memory():
	"""Reset peak memory stats."""
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()


def run_benchmark(
	model_path: str,
	prompt: str = "A serene lake with mountains in the background at sunset",
	num_frames: int = 81,
	num_inference_steps: int = 27,
	num_warmup: int = 1,
	num_runs: int = 3,
	output_dir: str = "./results",
):
	"""Run video generation benchmark."""

	results = {
		"model_path": model_path,
		"prompt": prompt,
		"num_frames": num_frames,
		"num_inference_steps": num_inference_steps,
		"num_warmup": num_warmup,
		"num_runs": num_runs,
		"timestamp": datetime.now().isoformat(),
		"warmup_times": [],
		"run_times": [],
		"peak_vram_gb": [],
	}

	# Get GPU info
	if torch.cuda.is_available():
		results["gpu_name"] = torch.cuda.get_device_name(0)
		results["gpu_count"] = torch.cuda.device_count()
		results["cuda_version"] = torch.version.cuda

	print(f"Starting benchmark with {num_warmup} warmup + {num_runs} runs")
	print(f"Model: {model_path}")
	print(f"Prompt: {prompt}")
	print(f"Frames: {num_frames}, Steps: {num_inference_steps}")
	print("-" * 60)

	# Warmup runs
	for i in range(num_warmup):
		print(f"Warmup run {i+1}/{num_warmup}...")
		reset_peak_memory()

		start = time.perf_counter()
		output = sgl.generate(
			VideoGenerationInput(
				prompt=prompt,
				num_frames=num_frames,
			),
			GenerationConfig(num_inference_steps=num_inference_steps),
		)
		elapsed = time.perf_counter() - start

		_, peak = get_gpu_memory_usage()
		results["warmup_times"].append(elapsed)
		print(f"  Time: {elapsed:.2f}s, Peak VRAM: {peak:.2f} GB")

	# Benchmark runs
	for i in range(num_runs):
		print(f"Benchmark run {i+1}/{num_runs}...")
		reset_peak_memory()

		start = time.perf_counter()
		output = sgl.generate(
			VideoGenerationInput(
				prompt=prompt,
				num_frames=num_frames,
			),
			GenerationConfig(num_inference_steps=num_inference_steps),
		)
		elapsed = time.perf_counter() - start

		_, peak = get_gpu_memory_usage()
		results["run_times"].append(elapsed)
		results["peak_vram_gb"].append(peak)
		print(f"  Time: {elapsed:.2f}s, Peak VRAM: {peak:.2f} GB")

	# Calculate statistics
	run_times = results["run_times"]
	results["avg_time"] = sum(run_times) / len(run_times)
	results["min_time"] = min(run_times)
	results["max_time"] = max(run_times)
	results["avg_time_per_step"] = results["avg_time"] / num_inference_steps
	results["avg_peak_vram_gb"] = sum(results["peak_vram_gb"]) / len(results["peak_vram_gb"])

	print("-" * 60)
	print(f"Results:")
	print(f"  Avg time: {results['avg_time']:.2f}s")
	print(f"  Min time: {results['min_time']:.2f}s")
	print(f"  Max time: {results['max_time']:.2f}s")
	print(f"  Avg time per step: {results['avg_time_per_step']:.2f}s")
	print(f"  Avg peak VRAM: {results['avg_peak_vram_gb']:.2f} GB")

	# Save results
	os.makedirs(output_dir, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_file = os.path.join(output_dir, f"benchmark_{timestamp}.json")

	with open(output_file, "w") as f:
		json.dump(results, f, indent=2)

	print(f"\nResults saved to: {output_file}")
	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Benchmark SGLang Wan2.2 video generation")
	parser.add_argument("--model-path", default="./models/wan2.2", help="Path to model")
	parser.add_argument("--prompt", default="A serene lake with mountains in the background at sunset")
	parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to generate")
	parser.add_argument("--num-inference-steps", type=int, default=27, help="Number of denoising steps")
	parser.add_argument("--num-warmup", type=int, default=1, help="Number of warmup runs")
	parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
	parser.add_argument("--output-dir", default="./results", help="Output directory for results")

	args = parser.parse_args()
	run_benchmark(
		model_path=args.model_path,
		prompt=args.prompt,
		num_frames=args.num_frames,
		num_inference_steps=args.num_inference_steps,
		num_warmup=args.num_warmup,
		num_runs=args.num_runs,
		output_dir=args.output_dir,
	)
