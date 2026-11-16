#!/bin/bash
# Submits all FM training jobs (tactile and non-tactile, one per grasp dataset).
# Run from anywhere: bash slurm/submit_train_fm.sh

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

for script in \
  slurm/train_fm_top_grasp.sbatch \
  slurm/train_fm_side_grasp.sbatch \
  slurm/train_fm_combined_grasp.sbatch \
  slurm/train_fm_tactile_top_grasp.sbatch \
  slurm/train_fm_tactile_side_grasp.sbatch \
  slurm/train_fm_tactile_combined_grasp.sbatch \
; do
  echo "Submitting $script"
  sbatch "$script"
done
