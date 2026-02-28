#!/bin/bash
#SBATCH --job-name=wan2-nsys-old
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=01:30:00
#SBATCH --output=results/nsys_old_offload_%j.log

# NSYS PROFILING: OLD FSDP offload
# Captures CUPTI-level events: Memcpy HtoD/DtoH with bytes, GPU kernels, NCCL
# 2 iterations: warmup + nsys-profiled run

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
echo "NSYS PROFILING - OLD OFFLOAD (FSDP)"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# GPU info
nvidia-smi
echo ""

# Verify nsys
which nsys
nsys --version
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

COMMON_FLAGS=(
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --dit-layerwise-offload false
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

mkdir -p ./results/profiles

# ==========================================
# RUN 1: Warmup (torch.compile JIT, no profiling)
# ==========================================
echo "=========================================="
echo "RUN 1/2: Warmup (torch.compile JIT)"
echo "=========================================="
echo "Start: $(date)"

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/nsys_old_offload_${SLURM_JOB_ID}_warmup.mp4

echo "End: $(date)"
echo ""

# ==========================================
# RUN 2: nsys-profiled run (CUPTI events)
# ==========================================
echo "=========================================="
echo "RUN 2/2: nsys profiled run"
echo "=========================================="
echo "Start: $(date)"

export SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

time nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --kill=none \
    --force-overwrite=true \
    --output=./results/profiles/old_offload_nsys_${SLURM_JOB_ID} \
    sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/nsys_old_offload_${SLURM_JOB_ID}_profiled.mp4

echo "End: $(date)"
echo ""

echo "=========================================="
echo "nsys profiling complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Warmup: results/nsys_old_offload_${SLURM_JOB_ID}_warmup.mp4"
echo "  - Profiled: results/nsys_old_offload_${SLURM_JOB_ID}_profiled.mp4"
echo "  - nsys profile: results/profiles/old_offload_nsys_${SLURM_JOB_ID}.nsys-rep"
echo ""
echo "To analyze: nsys stats results/profiles/old_offload_nsys_${SLURM_JOB_ID}.nsys-rep"
ls -lh ./results/profiles/
