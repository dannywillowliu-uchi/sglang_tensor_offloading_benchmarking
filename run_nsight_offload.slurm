#!/bin/bash
#SBATCH --job-name=wan2-nsight-offload
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=00:15:00
#SBATCH --output=results/nsight_offload_%j.log

# NSight profiling: WITH layerwise offloading

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
echo "NSIGHT PROFILING - WITH OFFLOADING"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# GPU info
nvidia-smi
echo ""

# Create output directory for profiles
mkdir -p ./results/profiles

# Verify nsys is available
which nsys
nsys --version
echo ""

# Create a one-shot generation script
cat > /tmp/generate_once.py << 'PYSCRIPT'
import requests
import time
import sys

base_url = "http://localhost:30010"

# Wait for server
print("Waiting for server...")
for i in range(60):
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            print("Server ready!")
            break
    except:
        pass
    time.sleep(5)
else:
    print("Server not ready after 5 minutes")
    sys.exit(1)

# Generate one video
print("Generating video...")
payload = {
    "prompt": "A serene lake with mountains in the background at sunset",
    "size": "1280x720",
    "num_frames": 81,
    "num_inference_steps": 27,
}
start = time.perf_counter()
resp = requests.post(f"{base_url}/v1/videos", json=payload, timeout=600)
elapsed = time.perf_counter() - start
print(f"Generation complete in {elapsed:.2f}s")
print(f"Status: {resp.status_code}")
PYSCRIPT

# Start profiled server WITH offloading
echo "Starting NSight profiling (with offloading)..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --output=./results/profiles/offload_profile_${SLURM_JOB_ID} \
    --force-overwrite=true \
    --duration=600 \
    sglang serve \
        --model-path ./models/wan2.2 \
        --text-encoder-cpu-offload \
        --pin-cpu-memory \
        --dit-layerwise-offload true \
        --num-gpus 4 \
        --ulysses-degree 2 \
        --ring-degree 2 \
        --port 30010 &

NSYS_PID=$!
echo "NSight profile PID: $NSYS_PID"

# Wait for server startup
sleep 60

# Run generation
python /tmp/generate_once.py

# Give nsys time to capture
sleep 30

# Stop profiling
kill -SIGINT $NSYS_PID 2>/dev/null || true
wait $NSYS_PID 2>/dev/null || true

echo ""
echo "Profiling complete!"
echo "Profile saved to: ./results/profiles/offload_profile_${SLURM_JOB_ID}.nsys-rep"
ls -lh ./results/profiles/
echo "Timestamp: $(date)"
