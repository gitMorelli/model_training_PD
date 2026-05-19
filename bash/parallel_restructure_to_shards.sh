#!/bin/bash
#SBATCH --job-name=par_restructure_to_shards
#--array=0-4
#SBATCH --nodes=1                     # 1 Node per array task
#SBATCH --cpus-per-task=32           
#SBATCH --mem=64G                     # Request enough RAM for 16 parallel processes
#SBATCH --time=02:00:00               # Estimated time for 500 images
#SBATCH --partition=shortq
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/parallel_restructure/job_%A_%a.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/parallel_restructure/job_%A_%a.err

# 1. Load necessary modules (this varies by cluster)
# module load python/3.10
# module load cuda/12.1
#module load nvidia/cuda/12.2.2-535.104.05

# --- Environment Setup ---
# Project Root
PROJECT_ROOT="/home/a_morelli/vscode_projects/model_training"
ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"

#export SLURM_ARRAY_COUNT=5
# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

# 1. Move into the project root directory
cd $PROJECT_ROOT

echo "Starting array task $SLURM_ARRAY_TASK_ID on $(hostname) with $SLURM_CPUS_PER_TASK CPUs"
# 2. Run using the -m flag (No .py extension, use dots for path)
$ENV_PYTHON -m src.scripts.parallel_restructure_to_shards