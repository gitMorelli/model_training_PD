#!/bin/bash
#SBATCH --job-name=memray_leak
#SBATCH --output=/home/a_morelli/vscode_projects/model_training/results/memray_leak.out
#SBATCH --error=/home/a_morelli/vscode_projects/model_training/results/memray_leak.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpgpuq,visuq
# no --gres: the dataloader-only test needs no GPU
# (if gpgpuq/visuq refuse GPU-less jobs on your cluster, re-add the gres line
#  or point --partition at a CPU partition like shortq)

ENV_PYTHON="/home/a_morelli/.conda/envs/torch_gpu/bin/python"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2        # keep conditions matched to rung-1 run

HOME_DIR="/home/a_morelli/vscode_projects/model_training"
OUT_BIN="$HOME_DIR/results/leak.bin"

cd $HOME_DIR
rm -f "$OUT_BIN"    # memray refuses to overwrite an existing file

$ENV_PYTHON -m memray run -o "$OUT_BIN" -m src.scripts.train_PD_model \
    --num_workers 0

# generate the reports right in the job
$ENV_PYTHON -m memray flamegraph --leaks -o "$HOME_DIR/results/leak_flamegraph.html" "$OUT_BIN"
$ENV_PYTHON -m memray table --leaks "$OUT_BIN" | head -60