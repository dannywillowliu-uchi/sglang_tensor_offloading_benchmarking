#!/bin/bash
#SBATCH --job-name=nsys-demo
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=results/nsys_demo_%j.log

# NSIGHT DEMO: Quick validation that nsys profiling works correctly
# Uses a tiny PyTorch workload (no model download needed)
# Verifies: nsys launches, traces CUDA kernels, writes .nsys-rep file

set -e

cd $SCRATCH/sglang-offload-research

# Load modules - matching our benchmark setup
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0  # Matches PyTorch CUDA 12.8, provides nsys 2024.6
module load GCC/12.3.0

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "=========================================="
echo "NSIGHT SYSTEMS DEMO - Dependency Validation"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Verify nsys version
echo "NSight Systems version:"
nsys --version
echo ""

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version (PyTorch): {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

export PYTHONUNBUFFERED=1

# Create a small PyTorch workload that exercises GPU compute + memory transfers
cat > /tmp/nsys_demo_workload.py << 'PYEOF'
import torch
import torch.nn as nn
import time

print("=== NSight Demo Workload ===")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Simple model: a few linear layers
model = nn.Sequential(
	nn.Linear(4096, 4096),
	nn.ReLU(),
	nn.Linear(4096, 4096),
	nn.ReLU(),
	nn.Linear(4096, 1024),
).cuda().to(torch.bfloat16)

# Input data
x = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)

# Warmup
print("Warmup...")
for _ in range(5):
	_ = model(x)
torch.cuda.synchronize()

# Timed run with CPU<->GPU transfers to simulate offloading
print("Running profiled workload (10 iterations with H2D/D2H transfers)...")
start = time.time()
for i in range(10):
	# Simulate H2D transfer (CPU -> GPU)
	cpu_tensor = torch.randn(512, 4096, dtype=torch.bfloat16, pin_memory=True)
	gpu_tensor = cpu_tensor.cuda(non_blocking=True)

	# GPU compute
	out = model(gpu_tensor)

	# Simulate D2H transfer (GPU -> CPU)
	cpu_out = out.cpu()

	if i % 5 == 0:
		torch.cuda.synchronize()
		print(f"  Step {i}: OK")

torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Done! {elapsed:.3f}s for 10 iterations")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
PYEOF

echo "Running nsys profile on demo workload..."
echo ""

time nsys profile \
	--trace=cuda,nvtx,osrt \
	--cuda-memory-usage=true \
	--trace-fork-before-exec=true \
	--kill=none \
	--output=./results/nsys_demo_profile_${SLURM_JOB_ID} \
	python /tmp/nsys_demo_workload.py

echo ""
echo "=========================================="
echo "Checking output files..."
ls -lh ./results/nsys_demo_profile_${SLURM_JOB_ID}* 2>/dev/null && echo "SUCCESS: NSight profile generated!" || echo "FAIL: No profile file found"
echo ""
echo "Demo complete!"
echo "Timestamp: $(date)"
