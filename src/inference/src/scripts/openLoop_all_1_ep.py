import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

NUM_WORKERS = 8
VIDEO_BACKEND = os.environ.get("LEROBOT_VIDEO_BACKEND", "torchcodec")


log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration — edit these variables directly
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = "/home/hk-project-p0024638/uhtfz/dev/outputs/train/25hz_2/20-42-22_xvla/checkpoints/050000/pretrained_model"
DATASET_PATH = "/hkfs/work/workspace/scratch/uhtfz-digit/data_pan_25hz/whiteboard_200hz_25hz_right_wrist"
REPO_ID = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPISODE_INDEX = 0  # which single episode to evaluate

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Main evaluation script
# -----------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # -----------------------------------------------------------------
    # 1. Reproducibility
    # -----------------------------------------------------------------
    set_seed_everywhere(SEED)
    os.makedirs("plots", exist_ok=True)
    log.info("Plots will be saved in %s", os.path.abspath("plots"))

    # -----------------------------------------------------------------
    # 2. Load BESO policy from pretrained checkpoint
    # -----------------------------------------------------------------
    log.info(f"Loading policy from {CHECKPOINT_PATH}")
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=CHECKPOINT_PATH)
    if hasattr(config, "compile_model"):
        config.compile_model = False
    policy_cls = get_policy_class(config.type)
    log.info(f"Detected policy type '{config.type}' -> {policy_cls.__name__}")
    agent = policy_cls(config)

    state_dict = load_file(os.path.join(CHECKPOINT_PATH, "model.safetensors"), device=DEVICE)
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    agent = agent.to(DEVICE)
    agent.eval()
    log.info(f"{policy_cls.__name__} loaded successfully")

    # -----------------------------------------------------------------
    # 2b. Load preprocessor (normalizes inputs) and postprocessor
    #     (unnormalizes predicted actions) from the checkpoint
    # -----------------------------------------------------------------
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=CHECKPOINT_PATH
    )
    log.info("Preprocessor and postprocessor loaded from checkpoint")

    # -----------------------------------------------------------------
    # 3. Load LeRobot Dataset
    # -----------------------------------------------------------------
    log.info(f"Loading LeRobot dataset from {DATASET_PATH}")
    dataset = LeRobotDataset(root=DATASET_PATH, repo_id=REPO_ID, video_backend=VIDEO_BACKEND)
    log.info(f"Found {dataset.num_episodes} episodes in dataset")

    # Storage for metrics across all episodes
    all_episode_metrics = []

    torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 4. Process each episode
    # -----------------------------------------------------------------
    if EPISODE_INDEX < 0 or EPISODE_INDEX >= dataset.num_episodes:
        raise ValueError(
            f"EPISODE_INDEX={EPISODE_INDEX} out of range (dataset has {dataset.num_episodes} episodes)"
        )
    for episode_idx in [EPISODE_INDEX]:
        log.info(f"\n{'='*80}")
        log.info(f"Processing episode {episode_idx + 1}/{dataset.num_episodes}")
        log.info(f"{'='*80}")

        try:
            from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
            to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
            num_samples = to_idx - from_idx

            if num_samples == 0:
                log.warning(f"Skipping episode {episode_idx}: no valid samples")
                continue

            log.info(f"Processing {num_samples} samples from this episode")

            predicted_actions = []
            ground_truth_actions = []

            # Reset the action queue at the start of each episode
            agent.reset()

            episode_subset = Subset(dataset, list(range(from_idx, to_idx)))
            episode_loader = DataLoader(
                episode_subset,
                batch_size=1,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(DEVICE == "cuda"),
                persistent_workers=False,
                collate_fn=lambda batch: batch[0],
            )

            for item in tqdm(episode_loader, desc=f"Episode {episode_idx + 1}", total=num_samples):
                # Build observation dict from dataset item
                # (no manual unsqueeze/to(device) — preprocessor handles that)
                obs_dict = {}
                for key in item:
                    if key.startswith("observation.") or key == "task":
                        obs_dict[key] = item[key]

                # Normalize observations via the saved preprocessor
                obs_dict = preprocessor(obs_dict)

                with torch.no_grad():
                    pred_action = agent.select_action(obs_dict)  # [1, action_dim]

                # Unnormalize predicted action back to original scale
                pred_action = postprocessor(pred_action)

                predicted_actions.append(pred_action)

                gt_action = item["action"]
                ground_truth_actions.append(gt_action.unsqueeze(0))

            predicted_actions = torch.cat(predicted_actions, dim=0).detach().float().cpu().numpy()
            ground_truth_actions = torch.cat(ground_truth_actions, dim=0).detach().float().cpu().numpy()

            action_dim = predicted_actions.shape[1]

            # Determine joint dims and gripper dim
            # Convention: last dimension is gripper
            joint_dims = action_dim - 1
            pred_joints = predicted_actions[:, :joint_dims]
            gt_joints = ground_truth_actions[:, :joint_dims]
            pred_gripper = predicted_actions[:, joint_dims:]
            gt_gripper = ground_truth_actions[:, joint_dims:]

            overall_mse = np.mean((pred_joints - gt_joints) ** 2)
            per_dim_mse = np.mean((pred_joints - gt_joints) ** 2, axis=0)

            gripper_pred_flat = pred_gripper.flatten()
            gripper_gt_flat = gt_gripper.flatten()
            pred_discrete = np.where(gripper_pred_flat > 0, 1, -1)
            if gripper_gt_flat.min() >= 0:
                gt_discrete = np.where(gripper_gt_flat > 0.5, 1, -1)
            else:
                gt_discrete = np.where(gripper_gt_flat > 0, 1, -1)

            gripper_accuracy = (pred_discrete == gt_discrete).mean()
            gripper_mse = np.mean((gripper_pred_flat - gripper_gt_flat) ** 2)

            episode_metrics = {
                "episode_name": f"episode_{episode_idx}",
                "num_samples": num_samples,
                "overall_mse": overall_mse,
                "per_dim_mse": per_dim_mse,
                "gripper_accuracy": gripper_accuracy,
                "gripper_mse": gripper_mse,
                "predictions": pred_joints,
                "ground_truth": gt_joints,
                "gripper_predictions": gripper_pred_flat,
                "gripper_ground_truth": gripper_gt_flat,
            }
            all_episode_metrics.append(episode_metrics)

            log.info(f"Episode {episode_idx} - Overall MSE: {overall_mse:.6f}")
            log.info(f"Episode {episode_idx} - Gripper Accuracy: {gripper_accuracy*100:.2f}%")

        except Exception as e:
            log.error(f"Error processing episode {episode_idx}: {str(e)}", exc_info=True)
            continue

    # -----------------------------------------------------------------
    # 5. Compute aggregate statistics across all episodes
    # -----------------------------------------------------------------
    if not all_episode_metrics:
        log.error("No episodes were successfully processed!")
        return

    log.info(f"\n{'='*80}")
    log.info(f"AGGREGATE STATISTICS ACROSS {len(all_episode_metrics)} EPISODES")
    log.info(f"{'='*80}")

    overall_mses = [m["overall_mse"] for m in all_episode_metrics]
    gripper_accuracies = [m["gripper_accuracy"] for m in all_episode_metrics]
    gripper_mses = [m["gripper_mse"] for m in all_episode_metrics]

    num_dims = all_episode_metrics[0]["per_dim_mse"].shape[0]
    per_dim_mses_all = np.array([m["per_dim_mse"] for m in all_episode_metrics])

    log.info(f"\nOverall MSE: {np.mean(overall_mses):.6f} +/- {np.std(overall_mses):.6f}")
    log.info(f"Gripper Accuracy: {np.mean(gripper_accuracies)*100:.2f}% +/- {np.std(gripper_accuracies)*100:.2f}%")
    log.info(f"Gripper MSE: {np.mean(gripper_mses):.6f} +/- {np.std(gripper_mses):.6f}")

    log.info("\nPer-dimension MSE (mean +/- std):")
    for dim in range(num_dims):
        dim_mses = per_dim_mses_all[:, dim]
        log.info(f"  Dimension {dim}: {np.mean(dim_mses):.6f} +/- {np.std(dim_mses):.6f}")

    # -----------------------------------------------------------------
    # 6. Save aggregate results
    # -----------------------------------------------------------------
    save_path = f"plots/{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    os.makedirs(save_path)
    results_path = f"{save_path}/all_predictions.pt"
    torch.save(
        {
            "episode_metrics": all_episode_metrics,
            "aggregate_stats": {
                "overall_mse_mean": np.mean(overall_mses),
                "overall_mse_std": np.std(overall_mses),
                "per_dim_mse_mean": np.mean(per_dim_mses_all, axis=0),
                "per_dim_mse_std": np.std(per_dim_mses_all, axis=0),
                "gripper_accuracy_mean": np.mean(gripper_accuracies),
                "gripper_accuracy_std": np.std(gripper_accuracies),
                "gripper_mse_mean": np.mean(gripper_mses),
                "gripper_mse_std": np.std(gripper_mses),
            },
        },
        results_path,
    )
    log.info(f"Saved aggregate results to {results_path}")

    # -----------------------------------------------------------------
    # 7. Create aggregate visualizations
    # -----------------------------------------------------------------

    # Plot 1: MSE distribution across episodes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].bar(range(len(overall_mses)), overall_mses, alpha=0.7, color="steelblue")
    axes[0, 0].axhline(
        y=np.mean(overall_mses), color="r", linestyle="--", linewidth=2,
        label=f"Mean: {np.mean(overall_mses):.6f}",
    )
    axes[0, 0].set_title("Overall MSE per Episode")
    axes[0, 0].set_xlabel("Episode Index")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(
        range(len(gripper_accuracies)),
        [acc * 100 for acc in gripper_accuracies],
        alpha=0.7, color="green",
    )
    axes[0, 1].axhline(
        y=np.mean(gripper_accuracies) * 100, color="r", linestyle="--", linewidth=2,
        label=f"Mean: {np.mean(gripper_accuracies)*100:.2f}%",
    )
    axes[0, 1].set_title("Gripper Accuracy per Episode")
    axes[0, 1].set_xlabel("Episode Index")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].boxplot(
        [per_dim_mses_all[:, i] for i in range(num_dims)],
        tick_labels=[f"Dim {i}" for i in range(num_dims)],
    )
    axes[1, 0].set_title("MSE Distribution per Dimension (across all episodes)")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, alpha=0.3)

    dim_means = np.mean(per_dim_mses_all, axis=0)
    dim_stds = np.std(per_dim_mses_all, axis=0)
    axes[1, 1].bar(
        range(num_dims), dim_means, yerr=dim_stds, alpha=0.7, color="coral", capsize=5
    )
    axes[1, 1].set_title("Mean MSE per Dimension with Std Dev")
    axes[1, 1].set_xlabel("Dimension")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_xticks(range(num_dims))
    axes[1, 1].set_xticklabels([f"Dim {i}" for i in range(num_dims)])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/aggregate_metrics_summary.png", dpi=150)
    plt.close()
    log.info("Saved aggregate_metrics_summary.png")

    # Plot 2: Concatenated predictions from a sample of episodes (first 5)
    sample_episodes = min(5, len(all_episode_metrics))
    fig, axes = plt.subplots(num_dims + 1, 1, figsize=(20, 3 * (num_dims + 1)))
    if num_dims + 1 == 1:
        axes = [axes]

    for dim in range(num_dims):
        offset = 0
        for ep_idx in range(sample_episodes):
            ep = all_episode_metrics[ep_idx]
            n_samples = ep["predictions"].shape[0]
            time_range = range(offset, offset + n_samples)

            axes[dim].plot(
                time_range, ep["ground_truth"][:, dim],
                linewidth=1.5, alpha=0.7, color="blue",
            )
            axes[dim].plot(
                time_range, ep["predictions"][:, dim],
                linewidth=1.5, alpha=0.7, linestyle="--", color="red",
            )

            if ep_idx < sample_episodes - 1:
                axes[dim].axvline(
                    x=offset + n_samples, color="gray", linestyle=":", linewidth=2, alpha=0.5,
                )

            offset += n_samples

        axes[dim].set_title(f"Dimension {dim} - Concatenated (First {sample_episodes} Episodes)")
        axes[dim].set_xlabel("Time Step")
        axes[dim].set_ylabel("Joint Position")
        axes[dim].grid(True, alpha=0.3)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=2, label="Ground Truth"),
            Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Predicted"),
        ]
        axes[dim].legend(handles=legend_elements)

    # Add Gripper Plot
    dim = num_dims
    offset = 0
    for ep_idx in range(sample_episodes):
        ep = all_episode_metrics[ep_idx]
        n_samples = ep["gripper_predictions"].shape[0]
        time_range = range(offset, offset + n_samples)

        axes[dim].plot(
            time_range, ep["gripper_ground_truth"],
            linewidth=1.5, alpha=0.7, color="blue",
        )
        axes[dim].plot(
            time_range, ep["gripper_predictions"],
            linewidth=1.5, alpha=0.7, linestyle="--", color="red",
        )

        if ep_idx < sample_episodes - 1:
            axes[dim].axvline(
                x=offset + n_samples, color="gray", linestyle=":", linewidth=2, alpha=0.5,
            )

        offset += n_samples

    axes[dim].set_title(f"Gripper - Concatenated (First {sample_episodes} Episodes)")
    axes[dim].set_xlabel("Time Step")
    axes[dim].set_ylabel("Gripper State")
    axes[dim].grid(True, alpha=0.3)
    axes[dim].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(f"{save_path}/concatenated_predictions.png", dpi=150)
    plt.close()
    log.info("Saved concatenated_predictions.png")

    # Plot 3: Average prediction curves
    max_length = max([m["predictions"].shape[0] for m in all_episode_metrics])

    aligned_predictions = np.full((len(all_episode_metrics), max_length, num_dims), np.nan)
    aligned_ground_truth = np.full((len(all_episode_metrics), max_length, num_dims), np.nan)
    aligned_gripper_predictions = np.full((len(all_episode_metrics), max_length), np.nan)
    aligned_gripper_ground_truth = np.full((len(all_episode_metrics), max_length), np.nan)

    for ep_idx, ep in enumerate(all_episode_metrics):
        n = ep["predictions"].shape[0]
        aligned_predictions[ep_idx, :n, :] = ep["predictions"]
        aligned_ground_truth[ep_idx, :n, :] = ep["ground_truth"]
        aligned_gripper_predictions[ep_idx, :n] = ep["gripper_predictions"]
        aligned_gripper_ground_truth[ep_idx, :n] = ep["gripper_ground_truth"]

    fig, axes = plt.subplots(num_dims + 1, 1, figsize=(15, 3 * (num_dims + 1)))
    if num_dims + 1 == 1:
        axes = [axes]

    for dim in range(num_dims):
        pred_mean = np.nanmean(aligned_predictions[:, :, dim], axis=0)
        pred_std = np.nanstd(aligned_predictions[:, :, dim], axis=0)
        gt_mean = np.nanmean(aligned_ground_truth[:, :, dim], axis=0)
        gt_std = np.nanstd(aligned_ground_truth[:, :, dim], axis=0)

        time_steps = np.arange(max_length)

        axes[dim].plot(time_steps, gt_mean, color="blue", linewidth=2, label="GT Mean")
        axes[dim].fill_between(time_steps, gt_mean - gt_std, gt_mean + gt_std, color="blue", alpha=0.2, label="GT Std")
        axes[dim].plot(time_steps, pred_mean, color="red", linewidth=2, linestyle="--", label="Pred Mean")
        axes[dim].fill_between(time_steps, pred_mean - pred_std, pred_mean + pred_std, color="red", alpha=0.2, label="Pred Std")

        axes[dim].set_title(f"Dimension {dim} - Average Trajectory (across {len(all_episode_metrics)} episodes)")
        axes[dim].set_xlabel("Time Step")
        axes[dim].set_ylabel("Joint Position")
        axes[dim].legend()
        axes[dim].grid(True, alpha=0.3)

    # Add Gripper Plot
    gripper_pred_mean = np.nanmean(aligned_gripper_predictions, axis=0)
    gripper_pred_std = np.nanstd(aligned_gripper_predictions, axis=0)
    gripper_gt_mean = np.nanmean(aligned_gripper_ground_truth, axis=0)
    gripper_gt_std = np.nanstd(aligned_gripper_ground_truth, axis=0)

    axes[num_dims].plot(time_steps, gripper_gt_mean, color="blue", linewidth=2, label="GT Mean")
    axes[num_dims].fill_between(time_steps, gripper_gt_mean - gripper_gt_std, gripper_gt_mean + gripper_gt_std, color="blue", alpha=0.2, label="GT Std")
    axes[num_dims].plot(time_steps, gripper_pred_mean, color="red", linewidth=2, linestyle="--", label="Pred Mean")
    axes[num_dims].fill_between(time_steps, gripper_pred_mean - gripper_pred_std, gripper_pred_mean + gripper_pred_std, color="red", alpha=0.2, label="Pred Std")

    axes[num_dims].set_title(f"Gripper - Average Trajectory (across {len(all_episode_metrics)} episodes)")
    axes[num_dims].set_xlabel("Time Step")
    axes[num_dims].set_ylabel("Gripper State")
    axes[num_dims].legend()
    axes[num_dims].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/average_trajectories.png", dpi=150)
    plt.close()
    log.info("Saved average_trajectories.png")

    # Plot 4: Statistical summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = [
        ["Metric", "Mean", "Std Dev", "Min", "Max"],
        ["Overall MSE", f"{np.mean(overall_mses):.6f}", f"{np.std(overall_mses):.6f}",
         f"{np.min(overall_mses):.6f}", f"{np.max(overall_mses):.6f}"],
        ["Gripper Acc (%)", f"{np.mean(gripper_accuracies)*100:.2f}", f"{np.std(gripper_accuracies)*100:.2f}",
         f"{np.min(gripper_accuracies)*100:.2f}", f"{np.max(gripper_accuracies)*100:.2f}"],
        ["Gripper MSE", f"{np.mean(gripper_mses):.6f}", f"{np.std(gripper_mses):.6f}",
         f"{np.min(gripper_mses):.6f}", f"{np.max(gripper_mses):.6f}"],
    ]

    for dim in range(num_dims):
        dim_mses = per_dim_mses_all[:, dim]
        table_data.append([
            f"Dim {dim} MSE", f"{np.mean(dim_mses):.6f}", f"{np.std(dim_mses):.6f}",
            f"{np.min(dim_mses):.6f}", f"{np.max(dim_mses):.6f}",
        ])

    table = ax.table(
        cellText=table_data, cellLoc="center", loc="center",
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(5):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title(
        f"Statistical Summary Across {len(all_episode_metrics)} Episodes",
        fontsize=14, fontweight="bold", pad=20,
    )
    plt.savefig(f"{save_path}/statistical_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved statistical_summary.png")

    # Plot 5: Episode-wise comparison heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(all_episode_metrics) * 0.4)))

    sns.heatmap(
        per_dim_mses_all, annot=False, cmap="YlOrRd", ax=axes[0],
        xticklabels=[f"Dim {i}" for i in range(num_dims)],
        yticklabels=[f"Ep {i}" for i in range(len(all_episode_metrics))],
        cbar_kws={"label": "MSE"},
    )
    axes[0].set_title("Per-Dimension MSE Heatmap (by Episode)")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Episode")

    combined_metrics = np.column_stack([overall_mses, gripper_accuracies, gripper_mses])
    combined_metrics_norm = (combined_metrics - combined_metrics.min(axis=0)) / (
        combined_metrics.max(axis=0) - combined_metrics.min(axis=0) + 1e-8
    )

    sns.heatmap(
        combined_metrics_norm, annot=False, cmap="coolwarm", ax=axes[1],
        xticklabels=["Overall MSE", "Gripper Acc", "Gripper MSE"],
        yticklabels=[f"Ep {i}" for i in range(len(all_episode_metrics))],
        cbar_kws={"label": "Normalized Value"},
    )
    axes[1].set_title("Overall Metrics Heatmap (Normalized, by Episode)")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Episode")

    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_heatmap.png", dpi=150)
    plt.close()
    log.info("Saved metrics_heatmap.png")

    log.info(f"\n{'='*80}")
    log.info("EVALUATION COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Processed {len(all_episode_metrics)} episodes")
    log.info(f"Average Overall MSE: {np.mean(overall_mses):.6f} +/- {np.std(overall_mses):.6f}")
    log.info(f"Average Gripper Accuracy: {np.mean(gripper_accuracies)*100:.2f}% +/- {np.std(gripper_accuracies)*100:.2f}%")
    log.info(f"All plots saved in {os.path.abspath('plots')}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
