#!/bin/bash
# Setup script for SGLang offloading research on ACES

set -e

cd $SCRATCH/sglang-offload-research

# Load modules
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.4.0

# Initialize conda
eval "$(conda shell.bash hook)"

# Create conda environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating conda environment..."
    conda create -p ./venv python=3.11 -y
fi

# Activate environment
conda activate ./venv

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install SGLang with diffusion support
# First try the released version with diffusion extras
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# Install nvtx for custom profiling markers
pip install nvtx

# Install huggingface-cli for model download
pip install huggingface_hub[cli]

echo "Environment setup complete!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
