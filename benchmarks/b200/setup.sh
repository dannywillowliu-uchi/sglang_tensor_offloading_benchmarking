#!/bin/bash
# setup.sh -- Environment setup for B200 profiling node
#
# Hardware: 8x NVIDIA B200 (183GB VRAM), NV18 NVLink mesh
# CUDA: 13.1 available (12.5/12.8/13.0 also available)
# Goal: Install SGLang 0.5.9 + deps, download Wan2.2 model
#
# Usage: bash b200/setup.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-$(pwd)/b200_workspace}"
VENV_DIR="$WORK_DIR/venv"
MODEL_ID="Wan-AI/Wan2.2-T2V-A14B-Diffusers"

echo "=========================================="
echo "  B200 Profiling Environment Setup"
echo "=========================================="
echo "Work dir: $WORK_DIR"
echo "Timestamp: $(date)"
echo ""

mkdir -p "$WORK_DIR"

# ---------- System check ----------
echo "--- System Check ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "CUDA toolkit:"
nvcc --version 2>/dev/null | tail -1 || echo "  nvcc not found (will use pip torch)"
echo ""
echo "nsys version:"
nsys --version 2>/dev/null || echo "  nsys not found -- install nsight-systems"
echo ""

# ---------- Python environment ----------
echo "--- Setting up Python environment ---"

if [ ! -d "$VENV_DIR" ]; then
	python3 -m venv "$VENV_DIR"
	echo "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ---------- PyTorch with CUDA 12.8 (Blackwell / sm_100) ----------
echo ""
echo "--- Installing PyTorch (CUDA 12.8) ---"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---------- SGLang 0.5.9 ----------
echo ""
echo "--- Installing SGLang ---"
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu128/torch2.7/flashinfer-python

# ---------- Extra dependencies ----------
echo ""
echo "--- Installing extra dependencies ---"
pip install nvtx huggingface_hub[cli]

# sage_attn: try pip, fall back to source
pip install sageattention 2>/dev/null || {
	echo "  sageattention pip install failed, trying from source..."
	pip install git+https://github.com/thu-ml/SageAttention.git 2>/dev/null || \
		echo "  WARNING: sageattention install failed -- will use default attention"
}

# ---------- Download model ----------
echo ""
echo "--- Downloading model: $MODEL_ID ---"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$WORK_DIR/model')
print('Model downloaded successfully')
"

# ---------- Verification ----------
echo ""
echo "=========================================="
echo "  Verification"
echo "=========================================="

python3 -c "
import torch
print(f'Python:    {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'  GPU {i}: {props.name}, {mem / 1e9:.1f} GB, sm_{props.major}{props.minor}')
print()

try:
    import sglang
    print(f'SGLang:    {sglang.__version__}')
except Exception as e:
    print(f'SGLang:    import failed: {e}')

try:
    from sageattention import sageattn
    print(f'SageAttn:  available')
except Exception:
    print(f'SageAttn:  not available')
"

echo ""
echo "--- Quick sanity: sglang generate --help ---"
sglang generate --help 2>&1 | head -5
echo "..."
# Verify offload flags exist
sglang generate --help 2>&1 | grep -E "dit-layerwise-offload|dit-offload-pcie-aware|dit-offload-prefetch" || {
	echo "WARNING: offload flags not found in sglang generate --help"
}

echo ""
echo "=========================================="
echo "  Setup Complete"
echo "=========================================="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Model at:      $WORK_DIR/model"
echo ""
echo "Next: bash b200/run_experiments.sh"
