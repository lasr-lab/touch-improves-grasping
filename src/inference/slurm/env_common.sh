#!/bin/bash
# Common environment setup for digit360-robotlearning-benchmarking SLURM jobs
# on the JUPITER booster partition (GH200 nodes). Source this from an sbatch
# script, don't execute it directly:
#
#   source "$(dirname "$0")/env_common.sh"

REPO_ROOT=/e/project1/m3/vanjani1/digit360-robotlearning-benchmarking

# --- conda env ---
# Base conda is broken by a stale ~/.local/lib/python3.13/site-packages/tqdm
# that shadows the real one; PYTHONNOUSERSITE keeps user site-packages out.
export PYTHONNOUSERSITE=1
source /e/project1/m3/vanjani1/conda/etc/profile.d/conda.sh
conda activate digit360

# --- CUDA toolkit (nvcc / CUDA_HOME) ---
# xlstm JIT-compiles a CUDA kernel at import time and needs CUDA_HOME even to
# import; harmless to load for policies that don't use it (act_tactile).
module load CUDA/13
export CUDA_HOME="$EBROOTCUDA"

# --- Hugging Face cache / offline mode ---
# Compute nodes may not have outbound internet during a job; force everything
# to resolve from the local cache populated ahead of time on a login/dev node.
export HF_HOME=/e/project1/m3/vanjani1/cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE

# --- Weights & Biases (offline; synced later with `wandb sync`) ---
mkdir -p "$REPO_ROOT"/wandb/{runs,cache,artifacts,config}
export WANDB_DIR="$REPO_ROOT/wandb/runs"
export WANDB_CACHE_DIR="$REPO_ROOT/wandb/cache"
export WANDB_ARTIFACT_DIR="$REPO_ROOT/wandb/artifacts"
export WANDB_CONFIG_DIR="$REPO_ROOT/wandb/config"
export WANDB_MODE=offline

echo "===== ENV CHECK ====="
which python
python - <<'PY'
import torch
print("torch", torch.__version__, "| torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available(), "| device count", torch.cuda.device_count())
PY
nvidia-smi || true
echo "======================"
