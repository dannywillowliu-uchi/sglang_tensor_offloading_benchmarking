#!/bin/bash
#SBATCH --job-name=wan2-new-offload
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=00:30:00
#SBATCH --output=results/new_offload_%j.log

# NEW OFFLOAD: PR #15511 "after" config (layerwise offload with async H2D prefetch)
# Uses --text-encoder-cpu-offload --pin-cpu-memory --dit-layerwise-offload true
# Expected: Fast first step (async prefetch), overlapped compute+transfer, ~58% speedup

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0  # Matches PyTorch CUDA 12.8, provides nsys 2024.6
module load GCC/12.3.0  # Required for FlashInfer JIT compilation

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "=========================================="
echo "NEW OFFLOAD BENCHMARK + NSIGHT PROFILING (PR #15511 'after')"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""
echo "Command: sglang generate (matching PR exactly)"
echo "Config: --text-encoder-cpu-offload --pin-cpu-memory --dit-layerwise-offload true"
echo "        --num-gpus 4 --ulysses-degree 4 --attention-backend sage_attn --enable-torch-compile"
echo "Expected: Fast first step (async prefetch), ~58% speedup over old offload"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Run sglang generate - NEW OFFLOAD (matches PR's "after" config exactly)
# This is the PR version WITH --dit-layerwise-offload true
# With NSight Systems profiling for GPU timeline visualization
echo "Running sglang generate with NSight profiling (new offload - PR 'after' config)..."
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1

time nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --kill=none \
    --output=./results/new_offload_profile_${SLURM_JOB_ID} \
    sglang generate \
    --model-path ./models/wan2.2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --dit-layerwise-offload true \
    --num-gpus 4 \
    --ulysses-degree 4 \
    --attention-backend sage_attn \
    --enable-torch-compile \
    --prompt "A cat walks on the grass, realistic" \
    --num-frames 81 \
    --height 720 \
    --width 1280 \
    --num-inference-steps 27 \
    --guidance-scale 3.5 \
    --guidance-scale-2 4.0 \
    --output-path ./results/new_offload_${SLURM_JOB_ID}.mp4

echo ""
echo "Benchmark complete!"
echo "Timestamp: $(date)"
