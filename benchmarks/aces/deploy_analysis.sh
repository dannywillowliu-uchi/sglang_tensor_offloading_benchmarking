#!/bin/bash
# deploy_analysis.sh -- Sync analysis scripts to ACES and submit nsys analysis job
#
# Usage: bash deploy_analysis.sh [--submit]
#   Without --submit: just sync files
#   With --submit: sync files AND submit the SLURM job

set -euo pipefail

ACES_SSH="${ACES_SSH:-aces}"
ACES_SCRATCH="/scratch/user/u.dl344013/sglang-offload-research"

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
	SUBMIT=true
fi

echo "=========================================="
echo "  Deploying Analysis Scripts to ACES"
echo "=========================================="
echo "  Target: aces:${ACES_SCRATCH}"
echo ""

# Files to sync
FILES=(
	"analysis/profile_comparison.py"
	"analysis/nsys_export_and_analyze.sh"
	"analysis/nsys_query.py"
	"analysis/cpu_trace_overlap.py"
	"run_nsys_analysis.slurm"
)

echo "--- Syncing files ---"
for f in "${FILES[@]}"; do
	if [ -f "$f" ]; then
		echo "  $f"
		# Ensure remote directory exists
		remote_dir="${ACES_SCRATCH}/$(dirname "$f")"
		ssh "${ACES_SSH}" "mkdir -p ${remote_dir}" 2>/dev/null
		scp "$f" "${ACES_SSH}:${ACES_SCRATCH}/$f"
	else
		echo "  WARNING: $f not found, skipping"
	fi
done

# Ensure output directories exist
echo ""
echo "--- Creating remote directories ---"
ssh "${ACES_SSH}" "mkdir -p ${ACES_SCRATCH}/results/profiles/analysis"

# Verify nsys-rep files exist on ACES
echo ""
echo "--- Checking nsys profiles on ACES ---"
ssh "${ACES_SSH}" "ls -lh ${ACES_SCRATCH}/results/profiles/*.nsys-rep 2>/dev/null || echo 'WARNING: No .nsys-rep files found!'"

echo ""
echo "--- Sync complete ---"

if $SUBMIT; then
	echo ""
	echo "--- Submitting SLURM job ---"
	JOB_ID=$(ssh "${ACES_SSH}" "cd ${ACES_SCRATCH} && sbatch run_nsys_analysis.slurm" | grep -o '[0-9]*')
	echo "  Submitted job: ${JOB_ID}"
	echo "  Monitor: ssh ${ACES_SSH} 'squeue -u \$USER'"
	echo "  Log: ${ACES_SCRATCH}/results/nsys_analysis_${JOB_ID}.log"
	echo ""
	echo "  To fetch results after completion:"
	echo "    scp -r ${ACES_SSH}:${ACES_SCRATCH}/results/profiles/analysis/ results/profiles/analysis/"
else
	echo ""
	echo "  To submit the job:"
	echo "    ssh ${ACES_SSH} 'cd ${ACES_SCRATCH} && sbatch run_nsys_analysis.slurm'"
	echo ""
	echo "  Or re-run with: bash deploy_analysis.sh --submit"
fi

echo ""
echo "  After job completes, fetch results:"
echo "    scp -r ${ACES_SSH}:${ACES_SCRATCH}/results/profiles/analysis/ results/profiles/analysis/"
echo ""
echo "  Then run locally:"
echo "    python3 analysis/profile_comparison.py"
