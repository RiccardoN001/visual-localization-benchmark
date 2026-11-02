#!/bin/bash
# One task of a job array = one (dataset, retriever, matcher, k) combination.
# Submitted by hpc/submit.py, which writes the plan file and passes account, QoS, walltime, array
# range and log path on the command line (command-line options override the defaults below).

# Leonardo Booster defaults: 1 GPU + 8 cores is a quarter of a node (4x A100, 32 cores), so SLURM
# packs up to four tasks per node.
#SBATCH --job-name=loc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# No `set -u`: the `module` shell function reads unset variables.
set -eo pipefail
: "${LOC_JOB_PLAN:?LOC_JOB_PLAN is not set (it is exported by hpc/submit.py)}"

# Line N+1 of the plan is the combination of array task N.
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((IDX + 1))p" "$LOC_JOB_PLAN")"
if [ -z "$LINE" ]; then
    echo "line $((IDX + 1)) missing in $LOC_JOB_PLAN" >&2
    exit 1
fi
IFS=$'\t' read -r DATASET RETRIEVER MATCHER K EXTRA <<< "$LINE"
echo "[$DATASET] $RETRIEVER + $MATCHER (k=$K) task=$IDX job=${SLURM_JOB_ID:-?} host=$(hostname)"

source hpc/env.sh

# OmniGlue needs TensorFlow and an older torch (cuDNN 8), so it has its own virtual environment.
# Without memory growth TensorFlow preallocates the whole GPU and torch runs out of memory.
VENV="${LOC_VENV:-envs/loc}"
if [ "$MATCHER" = "omniglue" ]; then
    VENV="${LOC_VENV_OMNIGLUE:-envs/omniglue}"
    export TF_FORCE_GPU_ALLOW_GROWTH=true
fi
source "$VENV/bin/activate"
echo "venv: $VENV"

# Compute nodes have no internet access: everything is read from the caches filled by
# hpc/prefetch_models.py on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOC_OFFLINE_HUB=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [[ "$EXTRA" == *"--name _warmup_"* ]]; then
    echo "warm-up: building the $RETRIEVER cache of $DATASET"
fi

# EXTRA holds the remaining localize.py options and is split into words on purpose.
# shellcheck disable=SC2086
CMD=(python -m loc.localize --dataset "$DATASET" --retriever "$RETRIEVER" --matcher "$MATCHER" --k "$K" $EXTRA)
echo "${CMD[*]}"
"${CMD[@]}"
