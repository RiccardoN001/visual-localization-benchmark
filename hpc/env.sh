# Environment shared by the login node (model download) and the compute nodes (jobs).
# Usage, from the repository root:  source hpc/env.sh
#
# Model caches live inside the repository, so that the jobs find exactly what prefetch_models.py
# downloaded. On CINECA Leonardo the repository should be cloned under $WORK, not $HOME: $HOME has
# a 50 GB quota and virtual environments are not allowed there.

export LOC_REPO="${LOC_REPO:-$(pwd)}"
export TORCH_HOME="$LOC_REPO/cache/torch"
export HF_HOME="$LOC_REPO/cache/hf"
export XFORMERS_DISABLED=1
export OPENCV_IO_ENABLE_OPENEXR=1

# Leonardo-specific: software modules loaded before activating a virtual environment.
# Adapt this list (or leave it empty) on another cluster.
LOC_MODULES="${LOC_MODULES:-profile/deeplrn cuda/12.1}"
if command -v module > /dev/null 2>&1 && [ -n "$LOC_MODULES" ]; then
    module purge
    for m in $LOC_MODULES; do module load "$m"; done
fi
