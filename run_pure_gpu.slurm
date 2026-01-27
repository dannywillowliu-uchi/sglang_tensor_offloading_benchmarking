#!/bin/bash
#SBATCH --job-name=wan2-pure-gpu
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=00:20:00
#SBATCH --output=results/pure_gpu_%j.log

# PURE GPU: No offloading at all - everything stays on GPU
# This tests whether the 14B model can fit entirely in 8x H100 VRAM
# Expected: May OOM or work with full GPU utilization

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.4.0
module load GCC/12.3.0  # Required for FlashInfer JIT compilation

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "=========================================="
echo "PURE GPU BENCHMARK (No Offloading)"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""
echo "Command: sglang generate (matching PR setup)"
echo "Config: All offloading disabled"
echo "        --num-gpus 4 --ulysses-degree 4 --attention-backend sage_attn --enable-torch-compile"
echo "Expected: May OOM (14B model is large) or work with high VRAM usage"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Run sglang generate - PURE GPU (no offloading)
# Explicitly disable all offloading flags
# With NSight Systems profiling for GPU timeline visualization
echo "Running sglang generate with NSight profiling (pure GPU - no offloading)..."
echo ""

# Force unbuffered output
export PYTHONUNBUFFERED=1

time nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --kill=none \
    --output=./results/pure_gpu_profile_${SLURM_JOB_ID} \
    sglang generate \
    --model-path ./models/wan2.2 \
    --dit-cpu-offload false \
    --dit-layerwise-offload false \
    --text-encoder-cpu-offload false \
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
    --output-path ./results/pure_gpu_${SLURM_JOB_ID}.mp4

echo ""
echo "Benchmark complete!"
echo "Timestamp: $(date)"
