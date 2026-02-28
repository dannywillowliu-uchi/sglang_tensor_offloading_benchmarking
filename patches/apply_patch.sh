#!/usr/bin/env bash
# apply_patch.sh - Apply PCIe-aware offload patch to SGLang on ACES
# Usage: ./apply_patch.sh [SGLANG_ROOT]
#
# SGLANG_ROOT defaults to ~/.local/lib/python3.12/site-packages/sglang

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/pcie_aware_offload.patch"
SGLANG_ROOT="${1:-${HOME}/.local/lib/python3.12/site-packages/sglang}"
BACKUP_DIR="${SCRIPT_DIR}/.backups"

# Files to patch (relative to SGLANG_ROOT parent, matching git diff paths)
FILES=(
	"multimodal_gen/runtime/utils/layerwise_offload.py"
	"multimodal_gen/runtime/models/dits/wanvideo.py"
	"multimodal_gen/runtime/server_args.py"
)

echo "=== PCIe-Aware Offload Patch ==="
echo "SGLang root: ${SGLANG_ROOT}"
echo "Patch file:  ${PATCH_FILE}"

# Verify SGLang installation exists
if [ ! -d "${SGLANG_ROOT}/multimodal_gen" ]; then
	echo "ERROR: SGLang multimodal_gen not found at ${SGLANG_ROOT}"
	echo "Pass the correct path: ./apply_patch.sh /path/to/sglang"
	exit 1
fi

# Backup originals
echo ""
echo "Backing up original files..."
mkdir -p "${BACKUP_DIR}"
for f in "${FILES[@]}"; do
	src="${SGLANG_ROOT}/${f}"
	if [ -f "${src}" ]; then
		dst="${BACKUP_DIR}/${f}"
		mkdir -p "$(dirname "${dst}")"
		cp "${src}" "${dst}"
		echo "  backed up: ${f}"
	else
		echo "  WARNING: ${src} not found, skipping backup"
	fi
done

# Apply patch
# The patch has git diff paths like a/python/sglang/multimodal_gen/...
# Strip 3 components (a/python/sglang/) so paths are relative to SGLANG_ROOT
echo ""
echo "Applying patch..."
cd "${SGLANG_ROOT}"
patch -p3 --forward --verbose < "${PATCH_FILE}"

echo ""
echo "=== Patch applied successfully ==="
echo "Backups stored in: ${BACKUP_DIR}"
echo ""
echo "Usage: add --dit-offload-pcie-aware to your sglang-diffusion launch command"
echo "Revert: ./revert_patch.sh [SGLANG_ROOT]"
