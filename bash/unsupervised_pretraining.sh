#!/bin/bash
#SBATCH --job-name=unsupervised
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/unsupervised.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/unsupervised.err
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=48G                      # Job memory request
#SBATCH --time=24:00:00                # Time limit hrs:min:sec
#SBATCH --partition=shortq
#--partition=gpgpuq
#SBATCH --partition=gpgpuq,visuq
#SBATCH --gres=gpu:1
#--gres=gpu:a100:1
#SBATCH --exclude=gpu01
#,gpu02,gpu04

# --- Environment Setup ---
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate yolo_env
ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"

# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1  
#export MALLOC_ARENA_MAX=2
#export LD_PRELOAD=/home/a_morelli/.conda/envs/torch_gpu/lib/libjemalloc.so

# --- Execution ---
# You can run the script from any location using its full path

# 1. Configuration
HOME_DIR="/home/a_morelli/vscode_projects/model_training"
# --- Hyperparameters ---

cd $HOME_DIR
# --- Execution ---
$ENV_PYTHON -m src.scripts.pretraining \
    --config $HOME_DIR/src/utils/pd_pretrain/configs/dae.yaml \
    --num_workers 15

