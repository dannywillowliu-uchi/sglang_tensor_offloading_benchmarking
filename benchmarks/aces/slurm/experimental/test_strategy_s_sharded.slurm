#!/bin/bash
#SBATCH --job-name=wan2-strategy-s
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=01:30:00
#SBATCH --output=results/strategy_s_%j.log

# STRATEGY S: Sharded Layerwise Offload
# Each GPU copies 1/N of each layer, then all-gathers to reconstruct.
# Model predicts ~24.7s/step (vs 23.7s unsharded, 22.6s old FSDP).
#
# This script applies a patch to the SGLang install, runs the benchmark,
# then reverts. The patch modifies:
#   - layerwise_offload.py: sharded H2D, all-gather materialization, padding
#   - server_args.py: --dit-offload-sharded flag + --dit-offload-pcie-aware (both added)
#   - wanvideo.py: ffn_attr_names mapping

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
module load GCC/12.3.0

eval "$(conda shell.bash hook)"
conda activate ./venv

SGLANG_ROOT=$(python -c "import sglang; import os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
echo "SGLang install root: $SGLANG_ROOT"

echo "=========================================="
echo "STRATEGY S: Sharded Offload Benchmark"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

nvidia-smi
echo ""

# No patch needed -- --dit-offload-sharded already installed in ACES SGLang 0.5.8
python -c "
from sglang.multimodal_gen.runtime.server_args import ServerArgs
assert hasattr(ServerArgs, 'dit_offload_sharded'), 'missing dit_offload_sharded'
print('Verified: dit_offload_sharded available')
"

# Common generation flags (note: --dit-offload-sharded, not --dit-offload-pcie-aware)
COMMON_FLAGS=(
    --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --dit-layerwise-offload true
    --dit-offload-sharded
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
# RUN 1: Warmup (torch.compile JIT)
# ==========================================
echo "=========================================="
echo "RUN 1/2: Warmup (torch.compile JIT)"
echo "=========================================="
echo "Start: $(date)"

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/strategy_s_${SLURM_JOB_ID}_run1.mp4

echo "End: $(date)"
echo ""

# ==========================================
# RUN 2: Clean timing
# ==========================================
echo "=========================================="
echo "RUN 2/2: Clean timing (per-step logging)"
echo "=========================================="
echo "Start: $(date)"

export SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

time sglang generate "${COMMON_FLAGS[@]}" \
    --output-path ./results/strategy_s_${SLURM_JOB_ID}_run2.mp4

echo "End: $(date)"
echo ""

# No patch revert needed -- sharded is a built-in flag

echo "=========================================="
echo "Strategy S benchmark complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Run 1 (warmup): results/strategy_s_${SLURM_JOB_ID}_run1.mp4"
echo "  - Run 2 (clean):  results/strategy_s_${SLURM_JOB_ID}_run2.mp4  <-- USE THIS"
echo ""
echo "Compare Run 2 per-step time against:"
echo "  - Unsharded (Strategy U): 23.7s/step"
echo "  - Old FSDP baseline:      22.6s/step"
echo "  - Model prediction (S):   ~24.7s/step"
