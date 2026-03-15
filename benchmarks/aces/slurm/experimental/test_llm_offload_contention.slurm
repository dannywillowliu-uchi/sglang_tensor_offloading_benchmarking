#!/bin/bash
#SBATCH --job-name=llm-offload-contention
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=01:00:00
#SBATCH --output=results/llm_offload_contention_%j.log

# LLM OFFLOAD CONTENTION TEST
# Tests whether PCIe bandwidth contention (H2D vs NCCL all-reduce) also
# occurs in LLM inference with OffloaderV2, not just diffusion models.
#
# SGLang LLM OffloaderV2 uses async H2D on alt_stream, while NCCL all-reduce
# runs on the communicator's own stream. Both are separate from the compute
# stream, so they CAN overlap and contend on the shared PCIe root complex.
#
# Design:
#   Test A: TP=4, no offloading (baseline per-token latency)
#   Test B: TP=4, offloading with prefetch_step=1 (minimal overlap)
#   Test C: TP=4, offloading with prefetch_step=2 (more overlap)
#
# Hypothesis: Offloading will degrade per-token latency on PCIe due to
# H2D + NCCL all-reduce contention, similar to diffusion findings.

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
echo "LLM OFFLOAD CONTENTION TEST"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

nvidia-smi
echo ""

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
import sglang
print(f'SGLang version: {sglang.__version__}')
"
echo ""

# Model -- pre-download via login node:
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen2.5-7b
MODEL="./models/qwen2.5-7b"
BATCH=1
INPUT_LEN=512
OUTPUT_LEN=64

export PYTHONUNBUFFERED=1

# ==========================================
# TEST A: Baseline (no offloading)
# ==========================================
echo "=========================================="
echo "TEST A: Baseline (TP=4, no offload)"
echo "=========================================="
echo "Start: $(date)"

python -m sglang.bench_one_batch \
    --model-path $MODEL \
    --tensor-parallel-size 4 \
    --batch-size $BATCH \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --run-name baseline_tp4

echo "End: $(date)"
echo ""

# ==========================================
# TEST B: Offloading with prefetch_step=1
# ==========================================
echo "=========================================="
echo "TEST B: Offloading (TP=4, group=2, prefetch=1)"
echo "=========================================="
echo "Start: $(date)"

python -m sglang.bench_one_batch \
    --model-path $MODEL \
    --tensor-parallel-size 4 \
    --offload-group-size 2 \
    --offload-num-in-group 1 \
    --offload-prefetch-step 1 \
    --offload-mode cpu \
    --batch-size $BATCH \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --run-name offload_tp4_pf1

echo "End: $(date)"
echo ""

# ==========================================
# TEST C: Offloading with prefetch_step=2
# ==========================================
echo "=========================================="
echo "TEST C: Offloading (TP=4, group=2, prefetch=2)"
echo "=========================================="
echo "Start: $(date)"

python -m sglang.bench_one_batch \
    --model-path $MODEL \
    --tensor-parallel-size 4 \
    --offload-group-size 2 \
    --offload-num-in-group 1 \
    --offload-prefetch-step 2 \
    --offload-mode cpu \
    --batch-size $BATCH \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --run-name offload_tp4_pf2

echo "End: $(date)"
echo ""

# ==========================================
# TEST D: TP=1 offloading (no NCCL, isolate pure offload cost)
# ==========================================
echo "=========================================="
echo "TEST D: Single GPU offloading (TP=1, no NCCL)"
echo "=========================================="
echo "Start: $(date)"

python -m sglang.bench_one_batch \
    --model-path $MODEL \
    --tensor-parallel-size 1 \
    --offload-group-size 2 \
    --offload-num-in-group 1 \
    --offload-prefetch-step 1 \
    --offload-mode cpu \
    --batch-size $BATCH \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --run-name offload_tp1_pf1

echo "End: $(date)"
echo ""

echo "=========================================="
echo "LLM offload contention test complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Analysis:"
echo "  1. Compare per-token latency: Test A vs B vs C"
echo "  2. If B/C > A: contention exists in LLM inference too"
echo "  3. Test D isolates pure offload cost (no NCCL to contend with)"
echo "  4. Delta between B and D estimates NCCL contention component"
echo ""
echo "Expected (if contention hypothesis holds):"
echo "  Test A (baseline): fastest per-token latency"
echo "  Test D (offload, no NCCL): moderate overhead from H2D only"
echo "  Test B (offload + NCCL): higher overhead (H2D + NCCL contend)"
echo "  Test C (more prefetch): similar or worse than B"
