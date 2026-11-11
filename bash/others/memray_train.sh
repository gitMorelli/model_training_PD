#!/bin/bash
#SBATCH --job-name=memray_PD
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/memray_PD.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/memray_PD.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # num_workers=0 -> no need for 22 cores
#SBATCH --mem=32G                      # leak accrues in one process; 32G is enough for 500 batches
#SBATCH --time=01:00:00
#SBATCH --partition=gpgpuq,visuq
#SBATCH --gres=gpu:v100:1

ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

HOME_DIR="/home/a_morelli/vscode_projects/model_training"
OUT_BIN="$HOME_DIR/results/leak.bin"

cd $HOME_DIR
rm -f "$OUT_BIN"    # memray refuses to overwrite an existing file

$ENV_PYTHON -m memray run -o "$OUT_BIN" -m src.scripts.train_PD_model \
    --num_workers 0

# generate the reports right in the job, so you just open the HTML afterwards
$ENV_PYTHON -m memray flamegraph --leaks -o "$HOME_DIR/results/leak_flamegraph.html" "$OUT_BIN"
$ENV_PYTHON -m memray table --leaks "$OUT_BIN" | head -60