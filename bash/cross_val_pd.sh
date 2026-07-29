#!/bin/bash
#SBATCH --job-name=cross_val_PD
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/evaluation/cross_val_PD.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/evaluation/cross_val_PD.err
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=28              # Number of CPU cores per task
#SBATCH --mem=64G                      # Job memory request
#SBATCH --time=24:00:00                # Time limit hrs:min:sec
#--partition=shortq
#SBATCH --partition=visuq,gpgpuq
#--gres=gpu:1
#--gres=gpu:h100:1
#SBATCH --gres=gpu:v100:1
#--gres=gpu:p40:1
#--nodelist=gpu04
#--gres=gpu:t4:1

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
$ENV_PYTHON -m src.debug.cross_val_pd \
    --num_workers 26