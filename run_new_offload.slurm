#!/bin/bash
#SBATCH --job-name=wan2-new-offload
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=01:30:00
#SBATCH --output=results/new_offload_%j.log

# NEW OFFLOAD: PR #15511 "after" config (layerwise offload with async H2D prefetch)
# Uses --text-encoder-cpu-offload --pin-cpu-memory --dit-layerwise-offload true
# 3 iterations: warmup (torch.compile), clean timing, profiled run

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
module load GCC/12.3.0

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "=========================================="
echo "NEW OFFLOAD BENCHMARK (3 iterations)"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""
echo "Config: --text-encoder-cpu-offload --pin-cpu-memory --dit-layerwise-offload true"
echo "        --num-gpus 4 --ulysses-degree 4 --attention-backend sage_attn --enable-torch-compile"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Common generation flags
COMMON_FLAGS=(
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --dit-layerwise-offload true
    --num-gpus 4
    --ulysses-degree 4
    --attention-backend sage_attn
    --enable-torch-compile
    --prompt "A cat walks on the grass, realistic"
    --num-frames 81
    --height 720
    --width 1280
    --num-inference-steps 27
    --guidance-scale 3.5
    --guidance-scale-2 4.0
)

export PYTHONUNBUFFERED=1

# ==========================================
# RUN 1: Warmup (includes torch.compile JIT)
# ==========================================
echo "=========================================="
echo "RUN 1/3: Warmup (torch.compile JIT)"
echo "=========================================="
echo "Start: $(date)"

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/new_offload_${SLURM_JOB_ID}_run1.mp4

echo "End: $(date)"
echo ""

# ==========================================
# RUN 2: Clean timing (no profiling overhead)
# ==========================================
echo "=========================================="
echo "RUN 2/3: Clean timing (per-step logging, no profiler)"
echo "=========================================="
echo "Start: $(date)"

# Enable per-step logging for timing breakdown (minimal overhead, no profiler)
export SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/new_offload_${SLURM_JOB_ID}_run2.mp4

unset SGLANG_DIFFUSION_STAGE_LOGGING
unset SGLANG_DIFFUSION_SYNC_STAGE_PROFILING

echo "End: $(date)"
echo ""

# ==========================================
# RUN 3: Profiled run (for detailed analysis)
# ==========================================
echo "=========================================="
echo "RUN 3/3: Profiled run (torch.profiler)"
echo "=========================================="
echo "Start: $(date)"

# Enable sync profiling for accurate per-step timing
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1

time sglang generate "${COMMON_FLAGS[@]}" \
    --profile \
    --profile-all-stages \
    --num-profiled-timesteps -1 \
    --perf-dump-path ./results/perf_new_offload_${SLURM_JOB_ID}.json \
    --output-path ./results/new_offload_${SLURM_JOB_ID}_run3.mp4

echo "End: $(date)"
echo ""

echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Run 1 (warmup): results/new_offload_${SLURM_JOB_ID}_run1.mp4"
echo "  - Run 2 (clean):  results/new_offload_${SLURM_JOB_ID}_run2.mp4  <-- USE THIS TIMING"
echo "  - Run 3 (profiled): results/new_offload_${SLURM_JOB_ID}_run3.mp4"
echo "  - Profiler traces: logs/*.trace.json.gz"
echo ""
echo "NOTE: Use Run 2 timing for fair comparison (no profiling overhead)"
