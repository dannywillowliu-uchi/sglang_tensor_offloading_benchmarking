#!/bin/bash
#SBATCH --job-name=wan2-nccl-warmup-test
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=01:30:00
#SBATCH --output=results/nccl_warmup_test_%j.log

# NCCL WARMUP TEST: Compare step-1 latency with and without --warmup flag
# The --warmup flag clones the first request and runs it as a throwaway before
# the real inference, warming up NCCL All2All paths, torch.compile, and CUDA pools.
#
# Design:
#   Run A: WITHOUT --warmup (current behavior) -- baseline step-1 timing
#   Run B: WITH --warmup -- step-1 should be faster (NCCL already warm)
#   Both runs use torch.compile disk cache from Run 0 (cold start).
#
# What to compare:
#   - Step 1 latency in Run A vs Run B (the warmup effect)
#   - Steady-state steps should be identical (warmup only affects first step)

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
echo "NCCL WARMUP TEST"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""
echo "Purpose: Test whether --warmup flag reduces step-1 NCCL All2All init overhead"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Common generation flags (same as standard layerwise offload benchmark)
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
export SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

# ==========================================
# RUN 0: Cold start (populates torch.compile cache)
# ==========================================
echo "=========================================="
echo "RUN 0: Cold start (torch.compile JIT -- populates disk cache)"
echo "=========================================="
echo "Start: $(date)"

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/warmup_test_${SLURM_JOB_ID}_cold.mp4

echo "End: $(date)"
echo ""

# ==========================================
# RUN A: WITHOUT --warmup (baseline)
# ==========================================
echo "=========================================="
echo "RUN A: WITHOUT --warmup (baseline step-1 timing)"
echo "=========================================="
echo "Start: $(date)"

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/warmup_test_${SLURM_JOB_ID}_no_warmup.mp4

echo "End: $(date)"
echo ""

# ==========================================
# RUN B: WITH --warmup (NCCL pre-warmed)
# ==========================================
echo "=========================================="
echo "RUN B: WITH --warmup (NCCL All2All pre-warmed)"
echo "=========================================="
echo "Start: $(date)"
echo "NOTE: --warmup runs inference twice internally. First run is throwaway warmup."
echo "      Look for 'Warmed-up request processed in X.XX seconds' in output."
echo ""

time sglang generate "${COMMON_FLAGS[@]}" \
    --warmup \
    --output-path ./results/warmup_test_${SLURM_JOB_ID}_with_warmup.mp4

echo "End: $(date)"
echo ""

unset SGLANG_DIFFUSION_STAGE_LOGGING
unset SGLANG_DIFFUSION_SYNC_STAGE_PROFILING

echo "=========================================="
echo "NCCL warmup test complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Run 0 (cold):       results/warmup_test_${SLURM_JOB_ID}_cold.mp4"
echo "  - Run A (no warmup):  results/warmup_test_${SLURM_JOB_ID}_no_warmup.mp4"
echo "  - Run B (warmup):     results/warmup_test_${SLURM_JOB_ID}_with_warmup.mp4"
echo ""
echo "Analysis:"
echo "  1. Compare step 1 time: Run A vs Run B (tqdm progress bars)"
echo "  2. Run B should show 'Warmed-up request processed in X.XX seconds'"
echo "  3. Steady-state steps (2-27) should be similar in both runs"
echo "  4. Total wall time: Run B includes hidden warmup inference (~same as Run A total)"
echo ""
echo "Expected result on PCIe:"
echo "  Run A step 1: ~36s (NCCL All2All cold init)"
echo "  Run B step 1: ~23-25s (NCCL already warm, close to steady state)"
