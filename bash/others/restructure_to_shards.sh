#!/bin/bash
#SBATCH --job-name=restructure_to_shards
#SBATCH --nodes=1                     # 1 Node per array task
#SBATCH --cpus-per-task=4           
#SBATCH --mem=16G                     # Request enough RAM for 16 parallel processes
#SBATCH --time=00:30:00               # Estimated time for 500 images
#SBATCH --partition=shortq
#--partition=gpgpuq,visuq
#--gres=gpu:1
#--gres=gpu:v100:1 
#--partition=visuq
#--gres=gpu:t4:1
#--gres=gpu:a100:1
#--gres=gpu:p40:1
#--nodelist=gpu04
#--gres=shard:p40:2
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/restructure.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/restructure.err

# 1. Load necessary modules (this varies by cluster)
# module load python/3.10
# module load cuda/12.1
#module load nvidia/cuda/12.2.2-535.104.05

# --- Environment Setup ---
# Project Root
PROJECT_ROOT="/home/a_morelli/vscode_projects/model_training"
ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"

# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

# 1. Move into the project root directory
cd $PROJECT_ROOT

# 2. Run using the -m flag (No .py extension, use dots for path)
$ENV_PYTHON -m src.scripts.restructure_to_shards