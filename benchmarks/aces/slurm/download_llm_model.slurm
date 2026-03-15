#!/bin/bash
#SBATCH --job-name=download-llm
#SBATCH --account=155924932299
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/download_llm_%j.log

# Download an LLM for offload contention testing
# NOTE: This must run on a node with internet access.
# If compute nodes lack internet, run this manually on a login node instead:
#   cd $SCRATCH/sglang-offload-research
#   source venv/bin/activate
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen2.5-7b

set -e

cd $SCRATCH/sglang-offload-research

module purge
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate ./venv

echo "Downloading Qwen2.5-7B-Instruct..."
echo "Timestamp: $(date)"

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/qwen2.5-7b

echo "Download complete: $(date)"
echo "Size:"
du -sh models/qwen2.5-7b
