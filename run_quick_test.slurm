#!/bin/bash
#SBATCH --job-name=wan2-quicktest
#SBATCH --account=155924932299
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=244G
#SBATCH --time=00:15:00
#SBATCH --output=results/quicktest_%j.log

# Quick test: 1 H100 with offloading to verify setup works

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.4.0

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "=========================================="
echo "QUICK TEST (1 H100, with offloading)"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Start SGLang server with offloading (1 GPU)
echo "Starting SGLang server (1 GPU, with offloading)..."
sglang serve \
    --model-path ./models/wan2.2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --dit-layerwise-offload true \
    --num-gpus 1 \
    --port 30010 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
sleep 60

# Quick single inference test
echo ""
echo "Running quick test (1 inference)..."
python scripts/benchmark_client.py \
    --base-url http://localhost:30010 \
    --num-warmup 0 \
    --num-runs 1 \
    --config-name "quicktest_${SLURM_JOB_ID}" \
    --output-dir ./results

# Stop server
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "Quick test complete!"
echo "Timestamp: $(date)"
