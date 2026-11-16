#!/bin/bash
#SBATCH -p booster
#SBATCH -A m3
#SBATCH --job-name=digit360_act_tactile
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=05:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Trains the tactile ACT policy (Digit-360 / sparsh-x D360 encoder) via
# lerobot's standard training loop. Submit from the repo root:
#   sbatch slurm/train_act_tactile.sbatch
#
# NOTE: policy.tactile_encoder_type defaults to "d360", which pulls pretrained
# weights from facebook/sparsh-x-img on first use. Pre-fetch them on a login
# node while online so they land in $HF_HOME/hub before this offline job runs:
#   huggingface-cli download facebook/sparsh-x-img

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source slurm/env_common.sh

# TODO: point these at a real LeRobot-format dataset (needs meta/info.json).
# /e/project1/m3/vanjani1/tactile-dataset is currently empty -- populate it
# first or the job will fail trying to reach the (offline) HF Hub.
DATASET_ROOT="/e/project1/m3/vanjani1/tactile-dataset"
DATASET_REPO_ID="local/digit360-tactile"

RUN_NAME="act_tactile_$(date +%Y%m%d_%H%M%S)"

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
  --policy.type=act_tactile \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.tactile_encoder_type=d360 \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=16 \
  --wandb.enable=true \
  --wandb.entity='TODO_SET_ME' \
  --wandb.project='digit360-benchmarking' \
  --wandb.disable_artifact=true
