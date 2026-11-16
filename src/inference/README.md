# inference

Policy training and deployment package for tactile robot learning. Provides implementations of BESO, ACT, and Flow Matching policies — each with optional tactile (Digit 360) variants — integrated with [LeRobot](https://github.com/huggingface/lerobot) for training and real-robot deployment.

## Policies

| Policy | Tactile Variant | Description |
|--------|----------------|-------------|
| `beso` | `beso_tactile` | Diffusion-based policy (BESO extracted from LeRobot) |
| `act_tactile` | — | Action chunking transformer with tactile input |
| `fm` | `fm_tactile` | Flow matching policy |

Each policy variant has its own configuration, model, and processor module under `src/policy/`.

## Structure

```
src/
├── deploy.py              # Real-robot deployment loop (loads policy, runs inference, records bags)
├── bag_recorder.py        # Synchronized ROS 2 bag recorder with topic discovery
├── policy/
│   ├── beso/              # Standard BESO policy
│   ├── beso_tactile/      # BESO with Digit 360 tactile input
│   ├── act_tactile/       # ACT with tactile input
│   ├── fm/                # Flow matching policy
│   ├── fm_tactile/        # Flow matching with tactile input
│   └── common/            # Shared backbones and tactile encoders
├── robot_description/     # Robot kinematics and configuration
└── scripts/               # Open-loop evaluation and saliency analysis
slurm/                     # SLURM batch scripts for cluster training
```

## Installation

```bash
# Install this package
pip install -e .

# Install LeRobot (required)
# Follow https://github.com/huggingface/lerobot

# For tactile policies with Sparsh Digit 360 encoder:
git clone https://github.com/facebookresearch/sparsh-multisensory-touch.git
cd sparsh-multisensory-touch
pip install -e .

# Additional dependencies for some policies:
pip install xlstm mamba-ssm causal_conv1d lightning
```

See [`INTEGRATION.md`](INTEGRATION.md) for LeRobot registration details.

## Deployment

```bash
# Single run
python3 src/deploy.py

# Continuous deployment loop
bash deploy.sh
```

The deploy script loads a trained policy checkpoint, connects to the robot, runs inference in a loop, and records synchronized ROS 2 bags of each episode.

## Training

Training is managed through LeRobot with SLURM scripts provided in `slurm/`:

```bash
# Example: train BESO with tactile input
sbatch slurm/train_beso_tactile.sbatch

# Grasp-specific variants
sbatch slurm/train_beso_tactile_side_grasp.sbatch
sbatch slurm/train_beso_tactile_top_grasp.sbatch
sbatch slurm/train_beso_tactile_combined_grasp.sbatch
```

## Tactile Representations

Digit 360 images are processed using the Sparsh-X encoder:
- Crop to zoom into the fisheye image
- Resize to 224×224×3
- Tokenize 16×16 patches into 768-dimensional embeddings via linear projection
- Pretrained weights: [facebook/sparsh-x-img](https://huggingface.co/facebook/sparsh-x-img)
- Original Digit 360 paper: [arXiv:2411.02479](https://arxiv.org/pdf/2411.02479)

## Dependencies

- `torch`, `torchvision`
- `numpy`, `einops`, `tqdm`
- `transformers`, `huggingface_hub`
- `lerobot` (required, installed separately)

## License

MIT
