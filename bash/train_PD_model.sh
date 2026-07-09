#!/bin/bash
#SBATCH --job-name=train_PD
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/train_PD.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/train_PD.err
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=32G                      # Job memory request
#SBATCH --time=03:00:00                # Time limit hrs:min:sec
#--partition=shortq
#--partition=gpgpuq
#SBATCH --partition=gpgpuq,visuq
#--gres=gpu:1
#SBATCH --gres=gpu:h100:1
#--gres=gpu:v100:1
#--gres=gpu:p40:1
#--nodelist=gpu05
#--gres=gpu:t4:1
#--exclude=gpu05,gpu06

# --- Environment Setup ---
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate yolo_env
ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"

# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE

# --- Execution ---
# You can run the script from any location using its full path

# 1. Configuration
HOME_DIR="/home/a_morelli/vscode_projects/model_training"
# --- Hyperparameters ---

cd $HOME_DIR
# --- Execution ---
$ENV_PYTHON -m src.scripts.train_PD_model \
    --num_workers 8