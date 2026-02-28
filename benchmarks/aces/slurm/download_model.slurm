#!/bin/bash
#SBATCH --job-name=wan2-download
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/download_%j.log

# Download Wan2.2 model from HuggingFace

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.4.0

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "Starting model download..."
echo "Timestamp: $(date)"

# Download Wan2.2-T2V-A14B-Diffusers model
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --local-dir ./models/wan2.2 \
    --local-dir-use-symlinks False

echo "Download complete!"
echo "Timestamp: $(date)"
ls -lh ./models/wan2.2/
