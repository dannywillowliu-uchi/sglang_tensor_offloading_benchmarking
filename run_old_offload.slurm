#!/bin/bash
#SBATCH --job-name=wan2-old-offload
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=00:30:00
#SBATCH --output=results/old_offload_%j.log

# OLD OFFLOAD: PR #15511 "before" config (main branch behavior)
# Uses --text-encoder-cpu-offload --pin-cpu-memory WITHOUT --dit-layerwise-offload
# Expected: Slow first step (~35s), slow 19th step due to dual transformer switch

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
echo "OLD OFFLOAD BENCHMARK (PR #15511 'before')"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""
echo "Command: sglang generate (matching PR exactly)"
echo "Config: --text-encoder-cpu-offload --pin-cpu-memory"
echo "        --num-gpus 4 --ulysses-degree 4 --attention-backend sage_attn --enable-torch-compile"
echo "Expected: Slow first step, slow 19th step (dual transformer switch)"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Run sglang generate - OLD OFFLOAD (matches PR's "before" config exactly)
# This is the main branch behavior WITHOUT --dit-layerwise-offload
# With NSight Systems profiling for GPU timeline visualization
echo "Running sglang generate with NSight profiling (old offload - PR 'before' config)..."
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1

time nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --kill=none \
    --output=./results/old_offload_profile_${SLURM_JOB_ID} \
    sglang generate \
    --model-path ./models/wan2.2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --dit-layerwise-offload false \
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
    --output-path ./results/old_offload_${SLURM_JOB_ID}.mp4

echo ""
echo "Benchmark complete!"
echo "Timestamp: $(date)"
