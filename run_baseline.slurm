#!/bin/bash
#SBATCH --job-name=wan2-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=00:15:00
#SBATCH --output=results/baseline_%j.log

# Baseline benchmark: No layerwise offloading

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
echo "BASELINE BENCHMARK (No Layerwise Offload)"
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

# Start SGLang server in background - BASELINE (no layerwise offload)
echo "Starting SGLang server (baseline)..."
sglang serve \
    --model-path ./models/wan2.2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --num-gpus 4 \
    --ulysses-degree 2 \
    --ring-degree 2 \
    --port 30010 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
sleep 120

# Run benchmark client
echo ""
echo "Running benchmark..."
python scripts/benchmark_client.py \
    --base-url http://localhost:30010 \
    --num-warmup 1 \
    --num-runs 1 \
    --config-name "baseline_${SLURM_JOB_ID}" \
    --output-dir ./results

# Stop server
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "Benchmark complete!"
echo "Timestamp: $(date)"
