#!/bin/bash
#SBATCH --job-name=shard_feature_extraction
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/shard_feature_extraction/%x_%A_%a.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/shard_feature_extraction/%x_%A_%a.err
#SBATCH --array=0-15%10
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=64G                      # Job memory request
#SBATCH --time=06:00:00                # Time limit hrs:min:sec
#SBATCH --partition=shortq

# --- Environment Setup ---
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate yolo_env
ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"
ROOT="/mnt/beegfs02/scratch/a_morelli/model_training/shards/PD/final_png_whitebg_21_07_26"
OUT="/home/a_morelli/models/model_training_logs/PD/feature_extraction/from_shards"
SPLIT=${SPLIT:-train}

# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1  
#export MALLOC_ARENA_MAX=2
#export LD_PRELOAD=/home/a_morelli/.conda/envs/torch_gpu/lib/libjemalloc.so

# --- Execution ---
# You can run the script from any location using its full path
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$SCRIPT_DIR/logs"
trap 'rm -rf "$TMPDIR"' EXIT

# 1. Configuration
HOME_DIR="/home/a_morelli/vscode_projects/model_training"
# --- Hyperparameters ---

cd $HOME_DIR

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} cpus=${SLURM_CPUS_PER_TASK} split=${SPLIT}"

# --- Execution ---
$ENV_PYTHON -m src.scripts.feature_extraction_from_shard \
    --root "$ROOT" \
    --split "$SPLIT" \
    --out "$OUT" \
    --scratch "$TMPDIR" \
    --workers "$SLURM_CPUS_PER_TASK"