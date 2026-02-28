#!/bin/bash
# run_experiments.sh -- B200 profiling experiments for Wan2.2 layerwise offload
#
# Experiments:
#   1. No offload baseline (4 GPU) -- speed of light
#   2. Layerwise offload default (4 GPU) -- measure overhead
#   3. Layerwise offload no torch.compile (4 GPU) -- isolate graph breaks
#   4. Layerwise offload pcie-aware (4 GPU) -- FFN-scheduled prefetch
#
# Each experiment: warmup run, clean timing run, nsys profiled run
#
# Usage:
#   bash b200/run_experiments.sh              # run all experiments
#   bash b200/run_experiments.sh 1            # run only experiment 1
#   bash b200/run_experiments.sh 2 3          # run experiments 2 and 3

set -euo pipefail

WORK_DIR="${WORK_DIR:-$(pwd)/b200_workspace}"
RESULTS_DIR="$WORK_DIR/results"
MODEL_PATH="$WORK_DIR/model"
VENV_DIR="$WORK_DIR/venv"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR/nsys" "$RESULTS_DIR/logs" "$RESULTS_DIR/videos"

# Activate venv
source "$VENV_DIR/bin/activate"

# Use CUDA 13.1 for sm_100a (Blackwell) JIT compilation (flashinfer, triton, etc.)
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=========================================="
echo "  B200 Profiling Experiments"
echo "=========================================="
echo "Timestamp:  $TIMESTAMP"
echo "Model:      $MODEL_PATH"
echo "Results:    $RESULTS_DIR"
echo "Python:     $(python3 --version 2>&1)"
echo ""

# ---------- Common flags ----------
# 4 GPUs, 27 steps, 81 frames @ 720x1280 (matches ACES config)
COMMON_FLAGS=(
	--model-path "$MODEL_PATH"
	--num-gpus 4
	--ulysses-degree 4
	--attention-backend sage_attn
	--prompt "A cat walks on the grass, realistic"
	--num-frames 81
	--height 720
	--width 1280
	--num-inference-steps 27
	--guidance-scale 3.5
	--guidance-scale-2 4.0
)

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---------- Helpers ----------
run_warmup() {
	local name="$1"
	shift
	echo ""
	echo "  [WARMUP] $name -- torch.compile JIT"
	echo "  Start: $(date)"
	time sglang generate "$@" \
		--output-path "$RESULTS_DIR/videos/${name}_warmup.mp4" \
		2>&1 | tee "$RESULTS_DIR/logs/${name}_warmup.log"
	echo "  End: $(date)"
}

run_clean() {
	local name="$1"
	shift
	echo ""
	echo "  [CLEAN] $name -- timing with stage logging"
	echo "  Start: $(date)"
	export SGLANG_DIFFUSION_STAGE_LOGGING=1
	export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
	time sglang generate "$@" \
		--output-path "$RESULTS_DIR/videos/${name}_clean.mp4" \
		2>&1 | tee "$RESULTS_DIR/logs/${name}_clean.log"
	unset SGLANG_DIFFUSION_STAGE_LOGGING SGLANG_DIFFUSION_SYNC_STAGE_PROFILING
	echo "  End: $(date)"
}

run_nsys() {
	local name="$1"
	shift
	local nsys_out="$RESULTS_DIR/nsys/${name}_${TIMESTAMP}"
	echo ""
	echo "  [NSYS] $name -- capturing 2 denoising steps"
	echo "  Output: $nsys_out.nsys-rep"
	echo "  Start: $(date)"

	# nsys profile: capture CUDA + NVTX + OS runtime, track memory usage
	# Use delay + duration to skip torch.compile warmup and capture only steady-state
	# Warmup typically takes 2-5 min; we'll use cudaProfilerApi if available,
	# otherwise fall back to capturing the full run and filtering in analysis
	nsys profile \
		-t cuda,nvtx,osrt \
		--cuda-memory-usage=true \
		-o "$nsys_out" \
		sglang generate "$@" \
			--output-path "$RESULTS_DIR/videos/${name}_nsys.mp4" \
		2>&1 | tee "$RESULTS_DIR/logs/${name}_nsys.log"

	echo "  End: $(date)"
	echo "  nsys trace: $nsys_out.nsys-rep"
}

# ---------- Experiment definitions ----------

experiment_1() {
	local name="exp1_no_offload"
	echo ""
	echo "=========================================="
	echo "  Experiment 1: No Offload Baseline (4 GPU)"
	echo "  Purpose: Pure compute cost -- speed of light"
	echo "=========================================="

	# Explicitly disable all offloading -- SGLang auto-enables layerwise offload for Wan
	local flags=(
		"${COMMON_FLAGS[@]}"
		--enable-torch-compile
		--dit-layerwise-offload false
		--dit-cpu-offload false
		--text-encoder-cpu-offload false
		--image-encoder-cpu-offload false
		--vae-cpu-offload false
	)

	run_warmup "$name" "${flags[@]}"
	run_clean "$name" "${flags[@]}"
	run_nsys "$name" "${flags[@]}"
}

experiment_2() {
	local name="exp2_offload_default"
	echo ""
	echo "=========================================="
	echo "  Experiment 2: Layerwise Offload Default (4 GPU)"
	echo "  Purpose: Measure offload overhead vs baseline"
	echo "=========================================="

	local flags=(
		"${COMMON_FLAGS[@]}"
		--enable-torch-compile
		--text-encoder-cpu-offload
		--pin-cpu-memory
		--dit-layerwise-offload true
	)

	run_warmup "$name" "${flags[@]}"
	run_clean "$name" "${flags[@]}"
	run_nsys "$name" "${flags[@]}"
}

experiment_3() {
	local name="exp3_offload_no_compile"
	echo ""
	echo "=========================================="
	echo "  Experiment 3: Layerwise Offload No Compile (4 GPU)"
	echo "  Purpose: Isolate graph break overhead"
	echo "=========================================="

	local flags=(
		"${COMMON_FLAGS[@]}"
		--text-encoder-cpu-offload
		--pin-cpu-memory
		--dit-layerwise-offload true
	)
	# No --enable-torch-compile

	run_warmup "$name" "${flags[@]}"
	run_clean "$name" "${flags[@]}"
	# Skip nsys for this -- only need timing comparison
}

experiment_4() {
	local name="exp4_offload_pcie_aware"
	echo ""
	echo "=========================================="
	echo "  Experiment 4: Layerwise Offload PCIe-Aware (4 GPU)"
	echo "  Purpose: Test FFN-scheduled prefetch + double-buffer pool"
	echo "=========================================="

	# Check if --dit-offload-pcie-aware exists in this SGLang version
	if ! sglang generate --help 2>&1 | grep -q "dit-offload-pcie-aware"; then
		echo "  SKIPPED: --dit-offload-pcie-aware not available in SGLang $(python3 -c 'import sglang; print(sglang.__version__)')"
		echo "  This flag requires a development build with pcie-aware offload patches."
		return 0
	fi

	local flags=(
		"${COMMON_FLAGS[@]}"
		--enable-torch-compile
		--text-encoder-cpu-offload
		--pin-cpu-memory
		--dit-layerwise-offload true
		--dit-offload-pcie-aware
	)

	run_warmup "$name" "${flags[@]}"
	run_clean "$name" "${flags[@]}"
	run_nsys "$name" "${flags[@]}"
}

# ---------- Execution ----------

# Parse which experiments to run (default: all)
if [ $# -eq 0 ]; then
	EXPERIMENTS_TO_RUN=(1 2 3 4)
else
	EXPERIMENTS_TO_RUN=("$@")
fi

for exp_num in "${EXPERIMENTS_TO_RUN[@]}"; do
	case "$exp_num" in
		1) experiment_1 ;;
		2) experiment_2 ;;
		3) experiment_3 ;;
		4) experiment_4 ;;
		*) echo "Unknown experiment: $exp_num (valid: 1-4)"; exit 1 ;;
	esac
done

echo ""
echo "=========================================="
echo "  All Requested Experiments Complete"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Results:"
echo "  Logs:   $RESULTS_DIR/logs/"
echo "  nsys:   $RESULTS_DIR/nsys/"
echo "  Videos: $RESULTS_DIR/videos/"
echo ""
echo "Next: python3 b200/analyze_nsys.py $RESULTS_DIR/nsys/"
