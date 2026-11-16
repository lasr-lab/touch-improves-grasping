import torch

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
import numpy as np
import os
import time
from dataclasses import dataclass, field
from bag_recorder import SyncBagRecorder

from policy.beso.modeling_beso import BESOPolicy
from policy.beso_tactile.modeling_beso import BESOPolicy as BESOPolicyTactile
from policy.act_tactile.modeling_act import ACTPolicy
from policy.fm.modeling_fm import FMPolicy
from policy.fm_tactile.modeling_fm import FMPolicy as FMPolicyTactile
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.utils import build_inference_frame, make_robot_action
from robot_description.completeRobot import CompleteRobot
from robot_description.config_robot import CompleteRobotConfig
from safetensors.torch import load_file
from scripts.salience import set_seed_everywhere, resolve_saliency_keys, _prepare_structured_obs, _build_window_indices
from scripts.salience import plot_saliency, plot_image_saliency_heatmaps, make_saliency_video, plot_chunk_at_timestep, plot_saliency_at_timestep, get_image_keys_per_modality
import logging
from datetime import datetime
from lerobot.utils.constants import OBS_IMAGES
import types
import contextlib


log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Policy registry — select via DeployConfig.policy_type
# Variants: tactile policies use touch sensing, non-tactile use state only
# -----------------------------------------------------------------------------
POLICY_REGISTRY: dict[str, type] = {
    "beso": BESOPolicy,
    "beso_tactile": BESOPolicyTactile,
    "act_tactile": ACTPolicy,
    "fm": FMPolicy,
    "fm_tactile": FMPolicyTactile,
}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

SELECTED_TIMESTEPS = [2, 40, 70]    # per episode: render full chunk + saliency at these timesteps
COMPUTE_SALIENCY = True
BATCH_SIZE = 64
SALIENCY_BATCH_SIZE = 8
SALIENCY_VIDEO = True               # render per-frame saliency overlay as MP4 (run 0 only)

SALIENCY_MODALITIES = ("state", "rgb", "tactile")
OBS_TACTILE_IMAGES = "observation.tactile_images"


@dataclass
class DeployConfig:
    # Policy selection
    policy_type: str = "fm_tactile"
    model_id: str = "/path/to/model"
    dataset_id: str = "/path/to/dataset"

    # Episode control
    max_episodes: int = 1
    max_steps_per_episode: int = 25
    record: bool = False
    control_hz: float = 30.0
    skip_initial_actions: int = 20        # ignore first N steps due to warmup artifacts
    obs_log_every_n: int = 30             # log observation every N steps for saliency
    episode_reset_pause: float = 1.0
    episode_between_pause: float = 9.0

    # Bag recording
    bag_output_base: str = "/media/niklas/T9/inference"
    bag_topic_regex: str = "/lasr/.*"
    bag_expected_topics: list[str] = field(default_factory=lambda: [
        "/lasr/qpos", "/lasr/qvel", "/lasr/action",
        "/lasr/realsense/image", "/lasr/realsense/image_depth",
        "/lasr/digit360/image_0", "/lasr/digit360/image_1",
        "/lasr/digit360/image_2", "/lasr/digit360/image_3",
    ])
    bag_discovery_timeout: float = 10.0

    # Task
    task: str = "Pick up the ball on the left side"

    # Saliency plotting
    plot_name: str = "fm_trans_2_obs"


def load_policy(policy_type: str, model_id: str, device: str = DEVICE):
    """Instantiate a policy from the registry and load its weights."""
    if policy_type not in POLICY_REGISTRY:
        available = ", ".join(POLICY_REGISTRY.keys())
        raise ValueError(f"Unknown policy type '{policy_type}'. Available: {available}")

    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=model_id)
    model = POLICY_REGISTRY[policy_type](config)

    weights_path = os.path.join(model_id, "model.safetensors")
    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device), config


def fix_feature_spec(dataset_metadata):
    """Fix inconsistent feature spec where shape is (25,) but only 23 names listed."""
    for key in ["observation.state", "observation.qvel", "action"]:
        if key in dataset_metadata.features:
            ft = dataset_metadata.features[key]
            if ft["shape"] == (25,) and len(ft["names"]) == 23:
                ft["names"] = list(ft["names"]) + ["dummy_0", "dummy_1"]

def salience_main(deploy_cfg, config, agent, pre_post, ep_from_gpu):
    global end
    logging.basicConfig(level=logging.INFO)
    set_seed_everywhere(SEED)

    save_dir = f"plots/{deploy_cfg.plot_name}/{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving outputs in {os.path.abspath(save_dir)}")

    preprocessor, postprocessor = pre_post

    n_action_steps = getattr(config, "n_action_steps", 1)
    saliency_keys = resolve_saliency_keys(config, SALIENCY_MODALITIES) if COMPUTE_SALIENCY else {}
    print(f"n_action_steps={n_action_steps}, saliency_keys={saliency_keys}")

    all_results = {}

    runs = []
    stacked, gt_actions, task_value, raw_images = ep_from_gpu

    structured = _prepare_structured_obs(agent, stacked)

    queue_keys = {"observation.state", OBS_IMAGES, OBS_TACTILE_IMAGES, "observation.environment_state"}
    queue_keys = {k for k in queue_keys if k in structured}

    T = next(iter(structured.values())).shape[0]
    n_obs_steps = getattr(config, "n_obs_steps", 1)
    win = _build_window_indices(T, n_obs_steps, DEVICE)

    inner = agent._inference_model()
    flow = inner.flow
    unwrapped_fn = getattr(flow.sample, "__wrapped__", None)
    enable_grad = bool(saliency_keys) and any(saliency_keys[m] for m in saliency_keys)

    chunk_preds_chunks = []
    saliency_per_modality = {m: [] for m in saliency_keys}
    spatial_saliency = {"rgb": {}, "tactile": {}}
    modality_to_obs_key = {"rgb": OBS_IMAGES, "tactile": OBS_TACTILE_IMAGES}
    bs = SALIENCY_BATCH_SIZE if enable_grad else BATCH_SIZE
    capture_heatmaps = True

    start = 0

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
                chunk = inner.generate_actions(batch)
        finally:
            if patched:
                with contextlib.suppress(AttributeError):
                    del flow.sample

        loss = chunk.float().pow(2).sum()
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
                for t_global in range(deploy_cfg.max_steps_per_episode):
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
        with torch.no_grad():
            chunk = inner.generate_actions(batch)

    chunk_post = postprocessor(chunk.detach())
    chunk_preds_chunks.append(chunk_post.cpu().numpy())

    chunk_preds = np.concatenate(chunk_preds_chunks, axis=0)
    saliency_out = {
        m: (np.concatenate(arrs, axis=0) if arrs else np.array([]))
        for m, arrs in saliency_per_modality.items()
    }

    image_keys_per_modality = get_image_keys_per_modality(config)


    res = {
        "episode_idx": 0,
        "run_idx": 0,
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


    runs.append(res)
    all_results[0] = runs

    # plot_chunks_overlaid(runs, 0, save_dir)
    # plot_variance_across_runs(runs, 0, save_dir)
    if COMPUTE_SALIENCY:
        plot_saliency(runs, 0, save_dir)
        plot_image_saliency_heatmaps(runs, 0, save_dir)
        if SALIENCY_VIDEO:
            make_saliency_video(runs, 0, save_dir)
        return

    # Per-(episode, timestep) graphs for the explicitly chosen timesteps.
    T_ep = runs[0]["gt_actions"].shape[0]
    for t in SELECTED_TIMESTEPS:
        if 0 <= t < T_ep:
            plot_chunk_at_timestep(runs, 0, t, save_dir)
            if COMPUTE_SALIENCY:
                plot_saliency_at_timestep(runs, 0, t, save_dir)

    per_run_mse = []
    for r in runs:
        first = r["chunk_preds"][:, 0, :]
        mse = float(np.mean((first - r["gt_actions"]) ** 2))
        per_run_mse.append(mse)

    out = {
        ep: [
            {
                "run_idx": r["run_idx"],
                "chunk_preds": r["chunk_preds"],
                "chunk_starts": r["chunk_starts"],
                "gt_actions": r["gt_actions"],
                "saliency": r["saliency"],
            }
            for r in runs
        ]
        for ep, runs in all_results.items()
    }
    torch.save(out, os.path.join(save_dir, "all_results.pt"))
    print(f"Saved raw results to {os.path.join(save_dir, 'all_results.pt')}")
    print("DONE")


# Module-level counter used by salience_main (kept as global for compatibility)
end = 0


def setup_inference(cfg):
    """Initialize policy, dataset metadata, preprocessors, and robot."""
    model, config = load_policy(cfg.policy_type, cfg.model_id)
    dataset_metadata = LeRobotDatasetMetadata(cfg.dataset_id, root=cfg.dataset_id)
    fix_feature_spec(dataset_metadata)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=cfg.model_id)

    image_keys_per_modality = get_image_keys_per_modality(config)
    flat_image_keys = image_keys_per_modality["rgb"] + image_keys_per_modality["tactile"]
    raw_images = {k: {} for k in flat_image_keys}

    robot = CompleteRobot(CompleteRobotConfig())
    model.eval()

    return {
        "model": model,
        "config": config,
        "dataset_metadata": dataset_metadata,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "flat_image_keys": flat_image_keys,
        "raw_images": raw_images,
        "robot": robot,
    }


def log_observation(obs, action, flat_image_keys, obs_list, gt_list, raw_images):
    """Log observation data for saliency analysis."""
    global end
    obs = {
        k: (
            (v.squeeze(0) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 else v)
            .to(DEVICE, non_blocking=True)
            if torch.is_tensor(v) else v
        )
        for k, v in obs.items()
    }
    for k in flat_image_keys:
        if k in obs and torch.is_tensor(obs[k]):
            raw_images[k][end] = obs[k].detach().cpu().clone()
    obs_list.append(obs)
    gt_list.append(action)
    end += 1


def step_control_loop(cfg, i, model, robot, dataset_metadata, preprocessor, postprocessor,
                      flat_image_keys, obs_list, gt_list, raw_images, loop_period, start_time):
    """Execute one iteration of the control loop."""
    # Busy-wait to maintain target control rate
    while (time.time() - start_time) < loop_period:
        pass

    start_time = time.time()
    robot.spin()
    obs = robot.get_observation()

    obs_frame = build_inference_frame(
        observation=obs, ds_features=dataset_metadata.features, device=DEVICE, task=cfg.task
    )
    obs = preprocessor(obs_frame)

    action = model.select_action(obs)
    action = postprocessor(action)
    action = make_robot_action(action, dataset_metadata.features)

    if i > cfg.skip_initial_actions:
        robot.send_action(action)

    if i % cfg.obs_log_every_n == 0:
        log_observation(obs, action, flat_image_keys, obs_list, gt_list, raw_images)

    robot.spin()
    return start_time


def run_episode(cfg, episode_idx, model, robot, dataset_metadata, preprocessor, postprocessor,
                flat_image_keys, obs_list, gt_list, raw_images):
    """Run a single deployment episode with bag recording."""
    loop_period = 1.0 / cfg.control_hz

    robot.spin()
    while not robot.is_connected():
        robot.spin()
    print("Connected")
    model.reset()

    recorder = SyncBagRecorder(
        output_dir=f"{cfg.bag_output_base}/{cfg.policy_type}/episode_{episode_idx}",
        topic_regex=cfg.bag_topic_regex,
        expected_topics=cfg.bag_expected_topics,
        discovery_timeout=cfg.bag_discovery_timeout,
    )
    if cfg.record:
        if not recorder.start():
            return False
    print("Recording started — all topics discovered")

    start_time = time.time()
    for i in range(cfg.max_steps_per_episode):
        if i % 50 == 0:
            print(i)
        start_time = step_control_loop(
            cfg, i, model, robot, dataset_metadata, preprocessor, postprocessor,
            flat_image_keys, obs_list, gt_list, raw_images, loop_period, start_time
        )

    print(f"Episode {episode_idx} finished! Starting new episode...")
    recorder.stop()
    robot.reset()
    robot.spin()
    print(f"end: {end}")

    if episode_idx + 1 < cfg.max_episodes:
        time.sleep(cfg.episode_between_pause)
    time.sleep(cfg.episode_reset_pause)
    robot.reset_2()
    robot.spin()
    return True


def prepare_salience_data(obs_list, gt_list):
    """Prepare observation tensors for saliency analysis."""
    drop = {"observation.language.tokens", "observation.language.attention_mask",
            "observation.language.subtask.tokens", "observation.language.subtask.attention_mask"}
    keys = [k for k, v in obs_list[0].items() if torch.is_tensor(v) and k not in drop]
    stacked = {k: torch.stack([o[k] for o in obs_list], dim=0) for k in keys}
    gt_actions = torch.stack(gt_list, dim=0).cpu().numpy()
    gt_actions = None  # Intentionally discarded
    task_value = None
    return stacked, gt_actions, task_value


def main():
    global end

    cfg = DeployConfig()
    ctx = setup_inference(cfg)

    obs_list = []
    gt_list = []

    for j in range(cfg.max_episodes):
        success = run_episode(
            cfg, j,
            ctx["model"], ctx["robot"], ctx["dataset_metadata"],
            ctx["preprocessor"], ctx["postprocessor"],
            ctx["flat_image_keys"], obs_list, gt_list, ctx["raw_images"]
        )
        if not success:
            return

    stacked, gt_actions, task_value = prepare_salience_data(obs_list, gt_list)
    salience_main(cfg, ctx["config"], ctx["model"],
                  (ctx["preprocessor"], ctx["postprocessor"]),
                  (stacked, gt_actions, task_value, ctx["raw_images"]))

    print("end")


if __name__ == "__main__":
    main()
