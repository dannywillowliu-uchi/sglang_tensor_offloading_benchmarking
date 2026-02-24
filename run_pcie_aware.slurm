#!/bin/bash
#SBATCH --job-name=wan2-pcie-aware
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=488G
#SBATCH --time=01:30:00
#SBATCH --output=results/pcie_aware_%j.log

# PCIE-AWARE OFFLOAD 4-GPU: Layerwise offload with PCIe-aware H2D scheduling
# Identical to run_new_offload.slurm + --dit-offload-pcie-aware
# Compare against: new_offload_1459995 (layerwise baseline) and old_offload_1459996 (FSDP baseline)
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

SGLANG_PKG="./venv/lib/python3.11/site-packages/sglang"
PATCH_DIR="./patches"

echo "=========================================="
echo "PCIE-AWARE OFFLOAD BENCHMARK - 4 GPU (3 iterations)"
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

# ==========================================
# APPLY PATCH
# ==========================================
echo "=========================================="
echo "Applying PCIe-aware offload patch..."
echo "=========================================="

if [ ! -f "${PATCH_DIR}/pcie_aware_offload.patch" ]; then
	echo "ERROR: Patch file not found at ${PATCH_DIR}/pcie_aware_offload.patch"
	exit 1
fi

bash "${PATCH_DIR}/apply_patch.sh" "${SGLANG_PKG}"
echo ""

# Verify patch applied
python -c "
from sglang.multimodal_gen.runtime.utils.layerwise_offload import LayerwiseOffloadManager
import inspect
sig = inspect.signature(LayerwiseOffloadManager.__init__)
assert 'pcie_aware' in sig.parameters, 'Patch NOT applied: pcie_aware param missing'
print('Patch verified: pcie_aware parameter present in LayerwiseOffloadManager')
"
echo ""

echo "Config: --text-encoder-cpu-offload --pin-cpu-memory --dit-layerwise-offload true"
echo "        --dit-offload-pcie-aware --num-gpus 4 --ulysses-degree 4"
echo "        --attention-backend sage_attn --enable-torch-compile"
echo ""

# Common generation flags
COMMON_FLAGS=(
	--model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
	--text-encoder-cpu-offload
	--pin-cpu-memory
	--dit-layerwise-offload true
	--dit-offload-pcie-aware
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
	--output-path ./results/pcie_aware_${SLURM_JOB_ID}_run1.mp4

echo "End: $(date)"
echo ""

# ==========================================
# RUN 2: Clean timing (per-step logging)
# ==========================================
echo "=========================================="
echo "RUN 2/3: Clean timing (per-step logging, no profiler)"
echo "=========================================="
echo "Start: $(date)"

export SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

time sglang generate "${COMMON_FLAGS[@]}" \
	--output-path ./results/pcie_aware_${SLURM_JOB_ID}_run2.mp4

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

export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1

time sglang generate "${COMMON_FLAGS[@]}" \
	--profile \
	--profile-all-stages \
	--num-profiled-timesteps -1 \
	--perf-dump-path ./results/perf_pcie_aware_${SLURM_JOB_ID}.json \
	--output-path ./results/pcie_aware_${SLURM_JOB_ID}_run3.mp4

echo "End: $(date)"
echo ""

# ==========================================
# REVERT PATCH
# ==========================================
echo "=========================================="
echo "Reverting patch..."
echo "=========================================="
bash "${PATCH_DIR}/revert_patch.sh" "${SGLANG_PKG}"
echo ""

echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Run 1 (warmup):   results/pcie_aware_${SLURM_JOB_ID}_run1.mp4"
echo "  - Run 2 (clean):    results/pcie_aware_${SLURM_JOB_ID}_run2.mp4  <-- USE THIS TIMING"
echo "  - Run 3 (profiled): results/pcie_aware_${SLURM_JOB_ID}_run3.mp4"
echo "  - Profiler data:    results/perf_pcie_aware_${SLURM_JOB_ID}.json"
echo ""
echo "=== COMPARISON BASELINES ==="
echo "  Layerwise (no PCIe-aware): results/logs/new_offload_1459995.log"
echo "  FSDP baseline:             results/logs/old_offload_1459996.log"
echo ""
echo "  Target: per-step <=22.0s (was 23.74s), e2e denoising <=600s (was 641s)"
echo ""
echo "NOTE: Use Run 2 timing for fair comparison (no profiling overhead)"
