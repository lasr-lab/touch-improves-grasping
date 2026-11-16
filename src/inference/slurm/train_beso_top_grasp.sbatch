#!/bin/bash
#SBATCH -p booster
#SBATCH -A m3
#SBATCH --job-name=digit360_beso_top_grasp
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Trains the (non-tactile) BESO policy via lerobot's standard training loop.
# Submit from the repo root: sbatch slurm/train_beso_top_grasp.sbatch

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source slurm/env_common.sh

DATASET_ROOT="/e/project1/m3/vanjani1/tactile-dataset/top_grasp"
DATASET_REPO_ID="local/digit360-tactile-top_grasp"

RUN_NAME="beso_top_grasp_$(date +%Y%m%d_%H%M%S)"

srun accelerate launch \
  --num_processes 4 \
  --multi_gpu \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29500 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  --num_machines 1 \
  -m lerobot.scripts.lerobot_train \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.video_backend=pyav \
  --output_dir="./outputs/train/${RUN_NAME}" \
  --job_name="$RUN_NAME" \
  --resume=false \
  --policy.type=beso \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=16 \
  --wandb.enable=false
