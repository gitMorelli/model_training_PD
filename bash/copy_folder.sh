#!/bin/bash
#SBATCH --job-name=stage_data
#SBATCH --partition=shortq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/others/stage_data_%j.log

set -euo pipefail   # fail loudly instead of silently continuing

SRC=/home/a_morelli/datasets/shards
DST=/mnt/beegfs02/scratch/a_morelli/model_training/shards

rsync -aP --partial "$SRC/" "$DST/"

# verification pass: a clean run copies nothing new
echo "=== verification pass ==="
rsync -aP --partial "$SRC/" "$DST/"

echo "=== file counts ==="
echo "src: $(find "$SRC" -type f | wc -l)"
echo "dst: $(find "$DST" -type f | wc -l)"