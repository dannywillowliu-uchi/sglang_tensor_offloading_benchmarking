#!/usr/bin/env bash
# revert_patch.sh - Restore original SGLang files on ACES
# Usage: ./revert_patch.sh [SGLANG_ROOT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SGLANG_ROOT="${1:-${HOME}/.local/lib/python3.12/site-packages/sglang}"
BACKUP_DIR="${SCRIPT_DIR}/.backups"

FILES=(
	"multimodal_gen/runtime/utils/layerwise_offload.py"
	"multimodal_gen/runtime/models/dits/wanvideo.py"
	"multimodal_gen/runtime/server_args.py"
)

echo "=== Reverting PCIe-Aware Offload Patch ==="
echo "SGLang root: ${SGLANG_ROOT}"
echo "Backup dir:  ${BACKUP_DIR}"

if [ ! -d "${BACKUP_DIR}" ]; then
	echo "ERROR: No backups found at ${BACKUP_DIR}"
	echo "Was the patch applied with apply_patch.sh?"
	exit 1
fi

echo ""
echo "Restoring original files..."
for f in "${FILES[@]}"; do
	src="${BACKUP_DIR}/${f}"
	dst="${SGLANG_ROOT}/${f}"
	if [ -f "${src}" ]; then
		cp "${src}" "${dst}"
		echo "  restored: ${f}"
	else
		echo "  WARNING: backup not found for ${f}, skipping"
	fi
done

echo ""
echo "=== Revert complete ==="
echo "Backups retained in: ${BACKUP_DIR}"
echo "To clean up backups: rm -rf ${BACKUP_DIR}"
