import contextlib
import logging
import os
import random
import types
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES

OBS_TACTILE_IMAGES = "observation.tactile_images"
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

NUM_WORKERS = 4

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return default
    return [int(x.strip()) for x in value.split(",") if x.strip()]


CHECKPOINT_PATH = os.environ.get(
    "PLOT_CHECKPOINT_PATH",
    "/home/hk-project-p0024638/uhtfz/dev/outputs/train/2026-05-12/06-47-21_fm_transformer/checkpoints/024250/pretrained_model",
)
DATASET_PATH = "/hkfs/work/workspace/scratch/uhtfz-digit/data/test"
REPO_ID = "test"
VIDEO_BACKEND = os.environ.get("LEROBOT_VIDEO_BACKEND", "torchcodec")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

EPISODE_INDICES = _env_int_list("PLOT_EPISODE_INDICES", [56, 59, 62])
NUM_REPEATS = int(os.environ.get("PLOT_NUM_REPEATS", "2"))
SELECTED_TIMESTEPS = _env_int_list("PLOT_SELECTED_TIMESTEPS", [2, 40, 70])    # per episode: render full chunk + saliency at these timesteps
CHUNK_PLOT_DIMS = list(range(16, 23))   # which action dims to render in the per-timestep chunk plot
COMPUTE_SALIENCY = _env_bool("PLOT_COMPUTE_SALIENCY", True)
BATCH_SIZE = int(os.environ.get("PLOT_BATCH_SIZE", "64"))
SALIENCY_BATCH_SIZE = int(os.environ.get("PLOT_SALIENCY_BATCH_SIZE", "8"))
PREFETCH_FACTOR = 4
SALIENCY_HEATMAP_TIMESTEPS = int(os.environ.get("PLOT_SALIENCY_HEATMAP_TIMESTEPS", "8"))
SALIENCY_VIDEO = _env_bool("PLOT_SALIENCY_VIDEO", True)               # render per-frame saliency overlay as MP4 (run 0 only)
SALIENCY_VIDEO_FPS = int(os.environ.get("PLOT_SALIENCY_VIDEO_FPS", "15"))             # video framerate
SALIENCY_VIDEO_NORMALIZE = "per_frame"  # "per_frame" matches static heatmap; "global" preserves absolute magnitude

SALIENCY_MODALITIES = ("state", "rgb", "tactile")
PLOT_IMAGE_ATTENTION_SCORES = _env_bool("PLOT_IMAGE_ATTENTION_SCORES", True)      # plot rgb/tactile saliency scores over time as attention proxies


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_path_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return safe.strip("._") or "model"


def _model_name_from_checkpoint(checkpoint_path: str, policy_type: str) -> str:
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    model_name = os.path.basename(os.path.dirname(checkpoint_path))
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0:
            model_name = parts[idx - 1]
        if idx + 1 < len(parts):
            model_name = f"{model_name}__ckpt_{parts[idx + 1]}"
    return _sanitize_path_name(f"{policy_type}__{model_name}")


def _build_obs_dict(item):
    obs = {}
    for k in item:
        if k.startswith("observation.") or k == "task":
            obs[k] = item[k]
    return obs


def _prepare_structured_obs(agent, batch):
    if hasattr(agent, "_prepare_structured_observations"):
        return agent._prepare_structured_observations(batch)
    cfg = agent.config
    batch = dict(batch)
    if getattr(cfg, "rgb_image_features", None):
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in cfg.rgb_image_features], dim=-4)
    elif getattr(cfg, "image_features", None):
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in cfg.image_features], dim=-4)
    if getattr(cfg, "tactile_image_features", None):
        batch[OBS_TACTILE_IMAGES] = torch.stack([batch[k] for k in cfg.tactile_image_features], dim=-4)
    return batch


def get_image_keys_per_modality(config):
    out = {"rgb": [], "tactile": []}
    if getattr(config, "rgb_image_features", None):
        out["rgb"].extend(list(config.rgb_image_features))
    elif getattr(config, "image_features", None):
        out["rgb"].extend(list(config.image_features))
    if getattr(config, "tactile_image_features", None):
        out["tactile"].extend(list(config.tactile_image_features))
    return out


def resolve_saliency_keys(config, modalities):
    keys = {"state": [], "rgb": [], "tactile": []}
    if "state" in modalities and getattr(config, "robot_state_feature", None) and getattr(config, "use_robot_state", True):
        keys["state"].append("observation.state")
    has_rgb = bool(getattr(config, "rgb_image_features", None) or getattr(config, "image_features", None))
    if "rgb" in modalities and has_rgb:
        keys["rgb"].append(OBS_IMAGES)
    if "tactile" in modalities and getattr(config, "tactile_image_features", None):
        keys["tactile"].append(OBS_TACTILE_IMAGES)
    return keys


def _get_inference_model(agent):
    inference_model = getattr(agent, "_inference_model", None)
    if callable(inference_model):
        return inference_model()
    return agent


def _latest_obs_step(batch):
    out = dict(batch)
    for key in ("observation.state", "observation.environment_state"):
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim >= 3:
            out[key] = value[:, -1]
    return out


def _predict_action_chunk(agent, inner, batch):
    if hasattr(inner, "generate_actions"):
        return inner.generate_actions(batch)

    # ACTPolicy does not expose _inference_model/generate_actions. Its underlying
    # ACT module expects one current observation and image inputs split per camera.
    if hasattr(agent, "model") and agent.__class__.__name__ == "ACTPolicy":
        model_batch = _latest_obs_step(batch)
        if OBS_IMAGES in model_batch and torch.is_tensor(model_batch[OBS_IMAGES]):
            rgb = model_batch[OBS_IMAGES]
            if rgb.ndim == 6:
                rgb = rgb[:, -1]
            if rgb.ndim == 5:
                model_batch[OBS_IMAGES] = [rgb[:, i] for i in range(rgb.shape[1])]
        if OBS_TACTILE_IMAGES in model_batch and torch.is_tensor(model_batch[OBS_TACTILE_IMAGES]):
            tactile = model_batch[OBS_TACTILE_IMAGES]
            if tactile.ndim == 6:
                tactile = tactile[:, -1]
            if tactile.ndim == 5:
                model_batch[OBS_TACTILE_IMAGES] = [tactile[:, i] for i in range(tactile.shape[1])]
        return agent.model(model_batch)[0]

    if hasattr(agent, "predict_action_chunk"):
        return agent.predict_action_chunk(batch)

    raise AttributeError(
        f"Don't know how to predict action chunks for {agent.__class__.__name__} "
        f"(inner={inner.__class__.__name__})."
    )


# -----------------------------------------------------------------------------
# Single-run evaluation (chunk mode)
# -----------------------------------------------------------------------------
def _gather_episode_to_gpu(dataset, episode_idx, preprocessor, device,
                           heatmap_ts=None, image_keys=None):
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
    subset = Subset(dataset, list(range(from_idx, to_idx)))
    loader = DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"), persistent_workers=False,
        collate_fn=lambda b: b[0],
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    raw_images = None
    if heatmap_ts and image_keys:
        raw_images = {k: {} for k in image_keys}
    obs_list, gt_list, task_value = [], [], None
    for t, item in enumerate(loader):
        if raw_images is not None and t in heatmap_ts:
            for k in image_keys:
                if k in item and torch.is_tensor(item[k]):
                    raw_images[k][t] = item[k].detach().cpu().clone()
        obs = _build_obs_dict(item)
        obs = preprocessor(obs)
        if "task" in obs:
            task_value = obs.pop("task")
        obs = {
            k: (
                (v.squeeze(0) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 else v)
                .to(device, non_blocking=True)
                if torch.is_tensor(v) else v
            )
            for k, v in obs.items()
        }
        obs_list.append(obs)
        gt_list.append(item["action"])

    drop = {"observation.language.tokens", "observation.language.attention_mask",
            "observation.language.subtask.tokens", "observation.language.subtask.attention_mask"}
    keys = [k for k, v in obs_list[0].items() if torch.is_tensor(v) and k not in drop]
    stacked = {k: torch.stack([o[k] for o in obs_list], dim=0) for k in keys}
    gt_actions = torch.stack(gt_list, dim=0).cpu().numpy()
    return stacked, gt_actions, task_value, raw_images


def _build_window_indices(T, n_obs_steps, device):
    t = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(n_obs_steps, device=device).unsqueeze(0)
    return (t - (n_obs_steps - 1) + j).clamp_min_(0)


def evaluate_episode_chunk_mode(
    agent, dataset, episode_idx, preprocessor, postprocessor, saliency_keys, run_idx, n_action_steps,
):
    cfg = agent.config
    n_obs_steps = getattr(cfg, "n_obs_steps", 1)

    image_keys_per_modality = get_image_keys_per_modality(cfg)
    enable_grad_check = bool(saliency_keys) and any(saliency_keys[m] for m in saliency_keys)
    has_image_modality = bool(image_keys_per_modality["rgb"]) or bool(image_keys_per_modality["tactile"])
    capture_heatmaps = (run_idx == 0) and enable_grad_check and has_image_modality and SALIENCY_HEATMAP_TIMESTEPS > 0

    T_total = (
        dataset.meta.episodes[episode_idx]["dataset_to_index"]
        - dataset.meta.episodes[episode_idx]["dataset_from_index"]
    )
    if capture_heatmaps:
        # Video mode: capture every frame; static plot subsamples down to K later.
        if SALIENCY_VIDEO:
            heatmap_ts = set(range(T_total))
        else:
            heatmap_ts_arr = np.linspace(0, T_total - 1, SALIENCY_HEATMAP_TIMESTEPS, dtype=int)
            heatmap_ts = set(int(x) for x in heatmap_ts_arr.tolist())
        # Always include explicitly selected timesteps so we can render their saliency.
        heatmap_ts.update(int(t) for t in SELECTED_TIMESTEPS if 0 <= t < T_total)
        flat_image_keys = image_keys_per_modality["rgb"] + image_keys_per_modality["tactile"]
    else:
        heatmap_ts = set()
        flat_image_keys = []

    stacked, gt_actions, task_value, raw_images = _gather_episode_to_gpu(
        dataset, episode_idx, preprocessor, DEVICE,
        heatmap_ts=heatmap_ts, image_keys=flat_image_keys,
    )
    structured = _prepare_structured_obs(agent, stacked)

    queue_keys = {"observation.state", OBS_IMAGES, OBS_TACTILE_IMAGES, "observation.environment_state"}
    queue_keys = {k for k in queue_keys if k in structured}

    T = next(iter(structured.values())).shape[0]
    win = _build_window_indices(T, n_obs_steps, DEVICE)

    inner = _get_inference_model(agent)
    flow = getattr(inner, "flow", None)
    flow_sample = getattr(flow, "sample", None) if flow is not None else None
    unwrapped_fn = getattr(flow_sample, "__wrapped__", None)
    enable_grad = bool(saliency_keys) and any(saliency_keys[m] for m in saliency_keys)

    chunk_preds_chunks = []
    saliency_per_modality = {m: [] for m in saliency_keys}
    spatial_saliency = {"rgb": {}, "tactile": {}}
    modality_to_obs_key = {"rgb": OBS_IMAGES, "tactile": OBS_TACTILE_IMAGES}
    bs = SALIENCY_BATCH_SIZE if enable_grad else BATCH_SIZE
    warned_no_grad_chunk = False

    for start in tqdm(range(0, T, bs), desc=f"ep{episode_idx} run{run_idx}"):
        end = min(start + bs, T)
        B = end - start
        flat = win[start:end].reshape(-1)

        batch = {}
        for k, v in structured.items():
            if k in queue_keys:
                gathered = v.index_select(0, flat)
                batch[k] = gathered.reshape(B, n_obs_steps, *v.shape[1:])
            else:
                batch[k] = v[start:end]
        if task_value is not None:
            if isinstance(task_value, str):
                task_str = task_value
            elif isinstance(task_value, (list, tuple)) and len(task_value) > 0:
                task_str = task_value[0]
            else:
                task_str = str(task_value)
            batch["task"] = [task_str] * B

        tracked = {}
        if enable_grad:
            for modality, names in saliency_keys.items():
                for n in names:
                    if n in batch and torch.is_tensor(batch[n]) and batch[n].is_floating_point():
                        batch[n] = batch[n].detach().clone().requires_grad_(True)
                        tracked[(modality, n)] = batch[n]

        if enable_grad and tracked:
            patched = unwrapped_fn is not None
            if patched:
                flow.sample = types.MethodType(unwrapped_fn, flow)
            try:
                with torch.enable_grad():
                    chunk = _predict_action_chunk(agent, inner, batch)
            finally:
                if patched:
                    with contextlib.suppress(AttributeError):
                        del flow.sample

            loss = chunk.float().pow(2).sum()
            if loss.requires_grad:
                grads = torch.autograd.grad(loss, list(tracked.values()),
                                            retain_graph=False, allow_unused=True)
                modal_to_per_sample = {}
                for (modality, key), g in zip(tracked.keys(), grads):
                    if g is None:
                        continue
                    g_abs = g.detach().abs()
                    per_sample = g_abs.flatten(1).mean(dim=1).cpu().numpy()
                    modal_to_per_sample.setdefault(modality, []).append(per_sample)
                    if (capture_heatmaps and modality in ("rgb", "tactile")
                            and key == modality_to_obs_key.get(modality)
                            and g_abs.dim() == 6):
                        spatial = g_abs.mean(dim=(1, 3))
                        for t_global in heatmap_ts:
                            if start <= t_global < end:
                                spatial_saliency[modality][int(t_global)] = (
                                    spatial[t_global - start].cpu().numpy()
                                )
                for modality in saliency_per_modality:
                    if modality in modal_to_per_sample:
                        arr = np.mean(np.stack(modal_to_per_sample[modality], axis=0), axis=0)
                    else:
                        arr = np.full(B, np.nan, dtype=np.float32)
                    saliency_per_modality[modality].append(arr)
            else:
                if not warned_no_grad_chunk:
                    log.warning(
                        "%s action chunks do not require gradients; skipping saliency for this policy.",
                        agent.__class__.__name__,
                    )
                    warned_no_grad_chunk = True
                for modality in saliency_per_modality:
                    saliency_per_modality[modality].append(np.full(B, np.nan, dtype=np.float32))
        else:
            with torch.no_grad():
                chunk = _predict_action_chunk(agent, inner, batch)

        chunk_post = postprocessor(chunk.detach())
        chunk_preds_chunks.append(chunk_post.cpu().numpy())

    chunk_preds = np.concatenate(chunk_preds_chunks, axis=0)
    saliency_out = {
        m: (np.concatenate(arrs, axis=0) if arrs else np.array([]))
        for m, arrs in saliency_per_modality.items()
    }

    return {
        "episode_idx": episode_idx,
        "run_idx": run_idx,
        "num_samples": T,
        "n_action_steps": n_action_steps,
        "chunk_preds": chunk_preds,
        "chunk_starts": np.arange(T),
        "gt_actions": gt_actions,
        "saliency": saliency_out,
        "spatial_saliency": spatial_saliency if capture_heatmaps else None,
        "raw_images": raw_images,
        "image_keys_per_modality": image_keys_per_modality,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_chunks_overlaid(runs, episode_idx, save_dir):
    run0 = runs[0]
    chunks = run0["chunk_preds"]
    starts = run0["chunk_starts"]
    gt = run0["gt_actions"]
    T, n_act, A = chunks.shape

    fig, axes = plt.subplots(A, 1, figsize=(20, 3 * A), sharex=True)
    if A == 1:
        axes = [axes]

    max_chunks_drawn = 80
    stride = max(1, T // max_chunks_drawn)

    for d in range(A):
        ax = axes[d]
        ax.plot(np.arange(gt.shape[0]), gt[:, d], color="blue", linewidth=2, label="Ground Truth", zorder=3)
        for i in range(0, T, stride):
            s = starts[i]
            xs = np.arange(s, s + n_act)
            mask = xs < gt.shape[0]
            ax.plot(xs[mask], chunks[i, :mask.sum(), d], color="orange", alpha=0.25, linewidth=0.8)
        ax.plot([], [], color="orange", linewidth=2, label="Predicted chunks (overlay)")
        title = f"Ep {episode_idx} - {f'Dim {d}'} - chunks per obs (run 0)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time Step")

    plt.tight_layout()
    fp = os.path.join(save_dir, f"ep{episode_idx}_chunks_overlay.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    log.info(f"Saved {fp}")


def plot_variance_across_runs(runs, episode_idx, save_dir):
    first_actions = np.stack([r["chunk_preds"][:, 0, :] for r in runs], axis=0)
    gt = runs[0]["gt_actions"]
    R, T, A = first_actions.shape

    fig, axes = plt.subplots(A, 1, figsize=(15, 3 * A), sharex=True)
    if A == 1:
        axes = [axes]
    t = np.arange(T)
    for d in range(A):
        ax = axes[d]
        mean = first_actions[:, :, d].mean(axis=0)
        std = first_actions[:, :, d].std(axis=0)
        ax.plot(t, gt[:, d], color="blue", linewidth=2, label="Ground Truth")
        ax.plot(t, mean, color="orange", linewidth=2, linestyle="--", label=f"Pred mean (R={R})")
        ax.fill_between(t, mean - std, mean + std, color="orange", alpha=0.25, label="±1 std")
        title = f"Ep {episode_idx} - {f'Dim {d}'} - across {R} runs"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    fp = os.path.join(save_dir, f"ep{episode_idx}_variance_across_runs.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    log.info(f"Saved {fp}")


def plot_saliency(runs, episode_idx, save_dir):
    modalities = list(runs[0]["saliency"].keys())
    valid = [m for m in modalities if runs[0]["saliency"][m].size > 0]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(15, 5))
    for m in valid:
        stacked = np.stack([r["saliency"][m] for r in runs], axis=0)  # [R, T]
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        denom = mean.max() if mean.max() > 0 else 1.0
        mean_n = mean / denom
        std_n = std / denom
        t = np.arange(mean_n.shape[0])
        ax.plot(t, mean_n, label=f"{m} (max raw={denom:.2e})")
        ax.fill_between(t, mean_n - std_n, mean_n + std_n, alpha=0.2)
    ax.set_title(f"Ep {episode_idx} - Input saliency per modality (normalized, mean±std over runs)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("|grad of ||chunk||² w.r.t. input| (normalized)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fp = os.path.join(save_dir, f"ep{episode_idx}_saliency.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    log.info(f"Saved {fp}")


def plot_image_attention_scores_over_time(runs, episode_idx, save_dir):
    """Plot per-timestep image relevance scores for rgb/tactile.

    These are the same gradient-based saliency scores used for the plots, shown
    explicitly as over-time attention proxies.
    """
    saved = {}
    for modality in ("rgb", "tactile"):
        per_run = [
            r.get("saliency", {}).get(modality)
            for r in runs
            if r.get("saliency", {}).get(modality) is not None
            and r.get("saliency", {}).get(modality).size > 0
        ]
        if not per_run:
            continue

        stacked = np.stack(per_run, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        saved[modality] = {"per_run": stacked, "mean": mean, "std": std}

    if not saved:
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    for modality, values in saved.items():
        mean = values["mean"]
        std = values["std"]
        denom = mean.max() if mean.max() > 0 else 1.0
        t = np.arange(mean.shape[0])
        ax.plot(t, mean / denom, label=f"{modality} (max raw={denom:.2e})")
        ax.fill_between(t, (mean - std) / denom, (mean + std) / denom, alpha=0.2)
    ax.set_title(f"Ep {episode_idx} - Image attention scores over time (rgb/tactile)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized gradient-based attention proxy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"ep{episode_idx}_image_attention_scores_over_time.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Saved {plot_path}")


def plot_image_saliency_heatmaps(runs, episode_idx, save_dir):
    run0 = runs[0]
    spatial = run0.get("spatial_saliency")
    raw = run0.get("raw_images")
    cams_per_mod = run0.get("image_keys_per_modality") or {}
    if not spatial or not raw:
        return

    for modality, cam_keys in cams_per_mod.items():
        if not cam_keys or not spatial.get(modality):
            continue
        all_ts = sorted(spatial[modality].keys())
        # If video mode captured every frame, subsample down to K for the static plot.
        if len(all_ts) > SALIENCY_HEATMAP_TIMESTEPS:
            idx = np.linspace(0, len(all_ts) - 1, SALIENCY_HEATMAP_TIMESTEPS, dtype=int)
            timesteps = [all_ts[i] for i in idx]
        else:
            timesteps = all_ts
        n_cams = len(cam_keys)
        K = len(timesteps)
        fig, axes = plt.subplots(n_cams, K, figsize=(3 * K, 3 * n_cams), squeeze=False)
        for ci, cam_key in enumerate(cam_keys):
            for ki, t in enumerate(timesteps):
                ax = axes[ci, ki]
                ax.axis("off")
                heatmap_per_cam = spatial[modality].get(t)
                raw_t = raw.get(cam_key, {}).get(t)
                if heatmap_per_cam is None or raw_t is None or ci >= heatmap_per_cam.shape[0]:
                    continue

                img = raw_t.float()
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                img_np = np.clip(img.numpy(), 0.0, 1.0)
                if img_np.ndim == 3 and img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)

                hm = heatmap_per_cam[ci].astype(np.float32)
                if hm.max() > hm.min():
                    hm = (hm - hm.min()) / (hm.max() - hm.min())

                H_img, W_img = img_np.shape[:2]
                ax.imshow(img_np, extent=[0, W_img, H_img, 0])
                ax.imshow(hm, cmap="jet", alpha=0.5,
                          extent=[0, W_img, H_img, 0], aspect="auto")
                cam_label = cam_key.replace("observation.", "").replace("images.", "")
                ax.set_title(f"{cam_label}  t={t}", fontsize=8)
        plt.suptitle(
            f"Ep {episode_idx} — {modality} input saliency (|∂‖chunk‖² / ∂pixel|)",
            fontsize=12,
        )
        plt.tight_layout()
        fp = os.path.join(save_dir, f"ep{episode_idx}_{modality}_heatmaps.png")
        plt.savefig(fp, dpi=130)
        plt.close()
        log.info(f"Saved {fp}")


def make_saliency_video(runs, episode_idx, save_dir):
    """Per-frame heatmap-overlay video (run 0). Cameras are concatenated
       horizontally per frame. Heatmap colors are normalized globally over the
       whole episode so frames are visually comparable across time.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        log.warning("imageio not available — skipping saliency video")
        return

    run0 = runs[0]
    spatial = run0.get("spatial_saliency")
    raw = run0.get("raw_images")
    cams_per_mod = run0.get("image_keys_per_modality") or {}
    if not spatial or not raw:
        return

    cmap = plt.get_cmap("jet")

    for modality, cam_keys in cams_per_mod.items():
        if not cam_keys or not spatial.get(modality):
            continue
        timesteps = sorted(spatial[modality].keys())
        if len(timesteps) < 2:
            continue

        # Per-frame normalization (matches the static heatmap plot). For absolute
        # magnitude comparison across time use SALIENCY_VIDEO_NORMALIZE = "global".
        global_norm = SALIENCY_VIDEO_NORMALIZE == "global"
        if global_norm:
            all_vals = np.concatenate([spatial[modality][t].ravel() for t in timesteps])
            # Use 99th percentile as upper bound to avoid outlier-dominated scaling.
            vmin = float(all_vals.min())
            vmax = float(np.percentile(all_vals, 99))
            denom_g = max(vmax - vmin, 1e-12)

        frames = []
        for t in timesteps:
            heat = spatial[modality][t]  # [n_cams, H, W]
            row = []
            for ci, cam_key in enumerate(cam_keys):
                if ci >= heat.shape[0]:
                    continue
                raw_t = raw.get(cam_key, {}).get(t)
                if raw_t is None:
                    continue
                img = raw_t.float()
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                img_np = np.clip(img.numpy(), 0.0, 1.0)
                if img_np.ndim == 3 and img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)

                h = heat[ci].astype(np.float32)
                if global_norm:
                    hn = (h - vmin) / denom_g
                else:
                    h_min, h_max = h.min(), h.max()
                    hn = (h - h_min) / max(h_max - h_min, 1e-12)
                hn = np.clip(hn, 0.0, 1.0)
                heat_rgb = cmap(hn)[..., :3]  # drop alpha → [H, W, 3]

                # Resize heatmap to image grid via nearest-neighbor if needed.
                if heat_rgb.shape[:2] != img_np.shape[:2]:
                    H, W = img_np.shape[:2]
                    Hh, Wh = heat_rgb.shape[:2]
                    yy = (np.arange(H) * Hh / H).astype(int)
                    xx = (np.arange(W) * Wh / W).astype(int)
                    heat_rgb = heat_rgb[yy[:, None], xx]

                blended = 0.5 * img_np + 0.5 * heat_rgb
                blended = (blended * 255.0).clip(0, 255).astype(np.uint8)
                row.append(blended)

            if row:
                frame = np.concatenate(row, axis=1)
                # Pad to even dimensions (libx264 requires this).
                h, w = frame.shape[:2]
                if h % 2:
                    frame = np.concatenate([frame, frame[-1:]], axis=0)
                if w % 2:
                    frame = np.concatenate([frame, frame[:, -1:]], axis=1)
                frames.append(frame)

        if not frames:
            continue
        out_path = os.path.join(save_dir, f"ep{episode_idx}_{modality}_heatmap.mp4")
        try:
            iio.imwrite(out_path, np.stack(frames, axis=0),
                        fps=SALIENCY_VIDEO_FPS, codec="libx264")
            log.info(f"Saved {out_path}")
        except Exception as e:
            log.warning(f"libx264 failed ({e}); falling back to GIF")
            out_gif = os.path.splitext(out_path)[0] + ".gif"
            iio.imwrite(out_gif, np.stack(frames, axis=0),
                        duration=1.0 / SALIENCY_VIDEO_FPS, loop=0)
            log.info(f"Saved {out_gif}")


def plot_chunk_at_timestep(runs, episode_idx, t, save_dir):
    """One graph per (episode, t): for the action-chunk predicted at timestep t,
       overlay all runs alongside the GT actions in the [t, t+n_action_steps] window.
    """
    n_act = runs[0]["n_action_steps"]
    gt = runs[0]["gt_actions"]                        # [T_ep, A]
    T_ep, A = gt.shape
    if t < 0 or t >= T_ep:
        log.warning(f"Skipping ep{episode_idx} t={t}: out of range (T={T_ep})")
        return

    # Per-run chunk at this timestep: [n_act, A]
    chunks = [r["chunk_preds"][t] for r in runs]

    # GT window: starts at t, length n_act, clamped to episode end
    end = min(t + n_act, T_ep)
    gt_window = gt[t:end]                             # [n, A]
    n = end - t

    dims_to_plot = [d for d in CHUNK_PLOT_DIMS if 0 <= d < A]
    if not dims_to_plot:
        log.warning(f"Ep{episode_idx} t={t}: CHUNK_PLOT_DIMS yielded no valid dims for action_dim={A}")
        return
    n_rows = len(dims_to_plot)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    x_pred = np.arange(n_act)
    x_gt = np.arange(n)
    run_colors = ["orange", "red", "green", "purple", "brown"]
    for ax_idx, d in enumerate(dims_to_plot):
        ax = axes[ax_idx]
        ax.plot(x_gt, gt_window[:, d], color="blue", linewidth=2.5, label="GT")
        for ri, c in enumerate(chunks):
            color = run_colors[ri % len(run_colors)]
            ax.plot(x_pred, c[:, d], color=color, linewidth=1.0,
                    linestyle="--", label=f"Run {ri}")
        title = "Gripper" if d == A - 1 else f"Dim {d}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(f"Action step (chunk emitted at t={t})")
    plt.suptitle(f"Ep {episode_idx} — full action chunk at t={t} vs. GT", fontsize=12)
    plt.tight_layout()
    fp = os.path.join(save_dir, f"ep{episode_idx}_t{t}_chunk.png")
    plt.savefig(fp, dpi=130)
    plt.close()
    log.info(f"Saved {fp}")


def plot_saliency_at_timestep(runs, episode_idx, t, save_dir):
    """Spatial saliency heatmap at one specific timestep, per camera. Uses run 0."""
    run0 = runs[0]
    spatial = run0.get("spatial_saliency")
    raw = run0.get("raw_images")
    cams_per_mod = run0.get("image_keys_per_modality") or {}
    if not spatial or not raw:
        return

    for modality, cam_keys in cams_per_mod.items():
        if not cam_keys or not spatial.get(modality) or t not in spatial[modality]:
            continue
        heat = spatial[modality][t]                   # [n_cams, H, W]
        n_cams = len(cam_keys)
        fig, axes = plt.subplots(1, n_cams, figsize=(4 * n_cams, 4), squeeze=False)
        for ci, cam_key in enumerate(cam_keys):
            ax = axes[0, ci]
            ax.axis("off")
            raw_t = raw.get(cam_key, {}).get(t)
            if raw_t is None or ci >= heat.shape[0]:
                continue
            img = raw_t.float()
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img_np = np.clip(img.numpy(), 0.0, 1.0)
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            hm = heat[ci].astype(np.float32)
            if hm.max() > hm.min():
                hm = (hm - hm.min()) / (hm.max() - hm.min())
            H_img, W_img = img_np.shape[:2]
            ax.imshow(img_np, extent=[0, W_img, H_img, 0])
            ax.imshow(hm, cmap="jet", alpha=0.5,
                      extent=[0, W_img, H_img, 0], aspect="auto")
            cam_label = cam_key.replace("observation.", "").replace("images.", "")
            ax.set_title(cam_label, fontsize=10)
        plt.suptitle(f"Ep {episode_idx} — {modality} saliency at t={t}", fontsize=12)
        plt.tight_layout()
        fp = os.path.join(save_dir, f"ep{episode_idx}_t{t}_{modality}_saliency.png")
        plt.savefig(fp, dpi=130)
        plt.close()
        log.info(f"Saved {fp}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed_everywhere(SEED)

    log.info(f"Loading policy from {CHECKPOINT_PATH}")
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=CHECKPOINT_PATH)
    if hasattr(config, "compile_model"):
        config.compile_model = False
    policy_cls = get_policy_class(config.type)
    log.info(f"Detected policy type '{config.type}' -> {policy_cls.__name__}")

    plot_output_root = os.environ.get("PLOT_OUTPUT_ROOT", "plots")
    run_name = os.environ.get("PLOT_RUN_NAME")
    if run_name:
        save_dir = os.path.join(plot_output_root, _sanitize_path_name(run_name))
    else:
        model_name = _model_name_from_checkpoint(CHECKPOINT_PATH, config.type)
        save_dir = os.path.join(plot_output_root, f"{model_name}__{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}")
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Saving outputs in {os.path.abspath(save_dir)}")

    agent = policy_cls(config)

    state_dict = load_file(os.path.join(CHECKPOINT_PATH, "model.safetensors"), device=DEVICE)
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    agent = agent.to(DEVICE).eval()

    device_override = {"device_processor": {"device": DEVICE}}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=CHECKPOINT_PATH,
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )

    n_action_steps = getattr(config, "n_action_steps", 1)
    saliency_keys = resolve_saliency_keys(config, SALIENCY_MODALITIES) if COMPUTE_SALIENCY else {}
    log.info(f"n_action_steps={n_action_steps}, saliency_keys={saliency_keys}")

    log.info(f"Loading LeRobot dataset from {DATASET_PATH}")
    dataset = LeRobotDataset(root=DATASET_PATH, repo_id=REPO_ID, video_backend=VIDEO_BACKEND)
    log.info(f"Found {dataset.num_episodes} episodes")

    for ep_idx in EPISODE_INDICES:
        if ep_idx < 0 or ep_idx >= dataset.num_episodes:
            raise ValueError(f"EPISODE_INDEX={ep_idx} out of range (have {dataset.num_episodes})")

    for ep_idx in EPISODE_INDICES:
        log.info(f"\n{'='*80}\nEpisode {ep_idx}: running {NUM_REPEATS} repeats\n{'='*80}")
        runs = []
        for run_idx in range(NUM_REPEATS):
            torch.manual_seed(SEED + run_idx * 1000 + ep_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED + run_idx * 1000 + ep_idx)
                torch.cuda.empty_cache()
            res = evaluate_episode_chunk_mode(
                agent, dataset, ep_idx, preprocessor, postprocessor,
                saliency_keys, run_idx, n_action_steps,
            )
            runs.append(res)

        plot_chunks_overlaid(runs, ep_idx, save_dir)
        plot_variance_across_runs(runs, ep_idx, save_dir)
        if COMPUTE_SALIENCY:
            plot_saliency(runs, ep_idx, save_dir)
            if PLOT_IMAGE_ATTENTION_SCORES:
                plot_image_attention_scores_over_time(runs, ep_idx, save_dir)
            plot_image_saliency_heatmaps(runs, ep_idx, save_dir)
            if SALIENCY_VIDEO:
                make_saliency_video(runs, ep_idx, save_dir)

        # Per-(episode, timestep) graphs for the explicitly chosen timesteps.
        T_ep = runs[0]["gt_actions"].shape[0]
        for t in SELECTED_TIMESTEPS:
            if 0 <= t < T_ep:
                plot_chunk_at_timestep(runs, ep_idx, t, save_dir)
                if COMPUTE_SALIENCY:
                    plot_saliency_at_timestep(runs, ep_idx, t, save_dir)

        per_run_mse = []
        for r in runs:
            first = r["chunk_preds"][:, 0, :]
            mse = float(np.mean((first - r["gt_actions"]) ** 2))
            per_run_mse.append(mse)
        log.info(f"Ep {ep_idx} first-action MSE across runs: mean={np.mean(per_run_mse):.6f} std={np.std(per_run_mse):.6f}")

    log.info("DONE")


if __name__ == "__main__":
    main()
