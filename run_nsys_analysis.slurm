#!/bin/bash
#SBATCH --job-name=nsys-analysis
#SBATCH --account=155924932299
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/nsys_analysis_%j.log

# NSYS ANALYSIS: Export nsys-rep to SQLite and run SQL queries
# This is a POST-PROCESSING job -- no GPU compute, just nsys export + sqlite3
# Needs 1 GPU allocated only because nsys export may require CUDA context
# Low resource request: 1 GPU, 64GB RAM, 30 min

set -e
export PYTHONUNBUFFERED=1

cd $SCRATCH/sglang-offload-research

module purge
module load CUDA/12.8.0
module load Miniforge3/25.3.0-3

echo "=========================================="
echo "NSYS PROFILE ANALYSIS"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Check nsys
if command -v nsys &>/dev/null; then
	echo "nsys found: $(nsys --version 2>&1 | head -1)"
else
	echo "nsys not in PATH, trying module..."
	module load nsight-systems 2>/dev/null || module load nsight 2>/dev/null || true
	if ! command -v nsys &>/dev/null; then
		echo "ERROR: Cannot find nsys. Available modules:"
		module avail 2>&1 | grep -i nsight || echo "  (none found)"
		echo ""
		echo "Falling back to direct sqlite export if .sqlite files exist..."
	fi
fi

echo ""

# Check Python (using built-in sqlite3 module instead of sqlite3 CLI)
echo "python3 found: $(python3 --version 2>&1)"
python3 -c "import sqlite3; print(f'  sqlite3 module: {sqlite3.sqlite_version}')"
echo ""

# Run the analysis script
bash analysis/nsys_export_and_analyze.sh

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Results in: results/profiles/analysis/"
ls -la results/profiles/analysis/*.csv 2>/dev/null || echo "(no output files)"
