from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("beso_tactile")
@dataclass
class BESOConfig(PreTrainedConfig):

    # Inputs / output structure
    n_obs_steps: int = 1
    horizon: int = 10
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 0

    # Vision backbone
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = False
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = True
    # "spatial_softmax": ResNet feature map → SpatialSoftmax → 2*num_kp features.
    # "avgpool":        ResNet → global avgpool → Linear(backbone_dim → vision_feature_dim).
    vision_pool_mode: str = "spatial_softmax"
    spatial_softmax_num_keypoints: int = 32
    vision_feature_dim: int = 512
    fixed_version: bool = True
    use_separate_rgb_encoder_per_camera: bool = True
    # tactile Encoder
    tactile_image_keys: list[str] = field(default_factory=list)
    tactile_key_substrings: tuple[str, ...] = ("digit",)
    tactile_encoder_type: str = "d360"  # "cnn" | "resnet18" | "resnet50" | "d360" | "t3" | "anytouch"
    tactile_channels: tuple[int, ...] = (32, 64, 128)
    tactile_backbone: str = "resnet18"
    tactile_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    tactile_feature_dim: int = 128
    tactile_d360_model_size: str = "base"
    tactile_d360_pretrained_model_id: str | None = "facebook/sparsh-x-img"
    tactile_d360_pretrained_filename: str = "d360_sparshx_img_base.pth"
    tactile_d360_freeze: bool = True
    tactile_d360_image_size: tuple[int, int] = (224, 224)
    tactile_d360_input_channels: int = 6
    tactile_d360_patch_size: int = 16
    tactile_d360_num_register_tokens: int = 1
    tactile_d360_pool: str = "register"
    tactile_d360_fusion_type: str = "vanilla"
    tactile_d360_fusion_layer: int = 8
    tactile_d360_num_bottlenecks: int = 4
    tactile_t3_repo_path: str | None = "/home/hk-project-p0024638/uhtfz/dev/t3"
    tactile_t3_checkpoint_dir: str | None = None
    tactile_t3_pretrained_model_id: str | None = "alanz-mit/FoundationTactile"
    tactile_t3_pretrained_repo_type: str = "dataset"
    tactile_t3_model_size: str = "t3_medium"
    tactile_t3_encoder_domain: str = "digit"
    tactile_t3_freeze: bool = True
    tactile_t3_image_size: tuple[int, int] = (224, 224)
    tactile_t3_input_channels: int = 3
    tactile_t3_patch_size: int = 16
    tactile_t3_mlp_ratio: float = 4.0
    tactile_t3_pool: str = "cls"
    tactile_anytouch_repo_path: str | None = "/home/hk-project-p0024638/uhtfz/dev/AnyTouch"
    tactile_anytouch_checkpoint_path: str | None = "/home/hk-project-p0024638/uhtfz/dev/model/checkpoint.pth"
    tactile_anytouch_clip_config_path: str | None = None
    tactile_anytouch_freeze: bool = True
    tactile_anytouch_image_size: tuple[int, int] = (224, 224)
    tactile_anytouch_input_channels: int = 3
    tactile_anytouch_sensor_type: int = 0
    tactile_anytouch_data_type: int = 0
    tactile_anytouch_use_sensor_token: bool = True
    tactile_anytouch_use_same_patchemb: bool = False
    tactile_anytouch_new_decoder_sensor_token: bool = False
    # If True, each camera (and each obs step) is fed to the transformer as its own token at
    # embed_dim. If False, camera features are concatenated along the feature axis and projected
    # into a single obs token per step.
    obs_tokens_per_camera: bool = True
    # If False, the robot_state feature is ignored even when present in the dataset — no state
    # tokens are added to the transformer input. Set to False to train image-only.
    use_robot_state: bool = False

    # Sequence backbone — "transformer" | "mamba" | "xlstm".
    backbone_type: str = "transformer"

    # MDT Transformer
    embed_dim: int = 512
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    n_heads: int = 8
    embed_pdrop: float = 0.0
    attn_pdrop: float = 0.3
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1
    goal_dim: int = 512
    goal_seq_len: int = 1
    goal_drop: float = 0.1
    use_ada_conditioning: bool = False
    use_noise_encoder: bool = False
    linear_output: bool = True

    # Mamba backbone (only used when backbone_type="mamba"). Requires `mamba-ssm`.
    mamba_layer: str = "Mamba2"  # "Mamba1" or "Mamba2"
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64

    # xLSTM backbone (only used when backbone_type="xlstm"). Requires `xlstm`.
    xlstm_num_heads: int = 8
    xlstm_chunkwise_kernel: str = "parallel--native_autograd"
    xlstm_context_length: int = 256

    # Language conditioning (CLIP)
    use_language_conditioning: bool = True
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_clip: bool = True

    # EDM Diffusion
    sampler_type: str = "ddim"
    num_sampling_steps: int = 4
    sigma_data: float = 0.5
    sigma_min: float = 0.001
    sigma_max: float = 80.0
    noise_scheduler: str = "exponential"
    sigma_sample_density_type: str = "loglogistic"

    # Loss
    do_mask_loss_for_padding: bool = False

    # EMA — exponential moving average of the full model used at inference time.
    use_ema: bool = True
    ema_decay: float = 0.995

    # Optimization
    compile_model: bool = False
    compile_mode: str = "max-autotune" #default max-autotune reduce-overhead

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.9)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.05
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        vision_backbone = self.vision_backbone.lower()
        if not (
            vision_backbone.startswith("resnet")
            or vision_backbone.startswith("vit")
        ):
            raise ValueError(
                f"`vision_backbone` must be a torchvision ResNet or ViT variant. Got {self.vision_backbone}."
            )

        if self.backbone_type not in ("transformer", "mamba", "xlstm"):
            raise ValueError(
                f"`backbone_type` must be one of 'transformer', 'mamba', 'xlstm'. Got {self.backbone_type!r}."
            )

        if self.vision_pool_mode not in ("spatial_softmax", "avgpool"):
            raise ValueError(
                f"`vision_pool_mode` must be 'spatial_softmax' or 'avgpool'. Got {self.vision_pool_mode}."
            )
        if self.tactile_encoder_type not in ("cnn", "resnet18", "resnet50", "d360", "t3", "anytouch"):
            raise ValueError(
                "`tactile_encoder_type` must be one of 'cnn', 'resnet18', 'resnet50', "
                "'d360', 't3' or 'anytouch'. "
                f"Got {self.tactile_encoder_type}."
            )
        if len(self.tactile_channels) == 0 or any(ch <= 0 for ch in self.tactile_channels):
            raise ValueError(f"`tactile_channels` must contain positive integers. Got {self.tactile_channels}.")
        if self.tactile_encoder_type in ("resnet18", "resnet50") and self.tactile_backbone != self.tactile_encoder_type:
            raise ValueError(
                "`tactile_backbone` must match `tactile_encoder_type`. "
                f"Got tactile_encoder_type={self.tactile_encoder_type!r} and tactile_backbone={self.tactile_backbone!r}."
            )
        if self.tactile_feature_dim <= 0:
            raise ValueError(f"`tactile_feature_dim` must be positive. Got {self.tactile_feature_dim}.")
        if self.tactile_encoder_type == "d360":
            if self.tactile_d360_model_size not in ("tinier", "tiny", "small", "base"):
                raise ValueError(
                    "`tactile_d360_model_size` must be one of 'tinier', 'tiny', 'small' or 'base'. "
                    f"Got {self.tactile_d360_model_size}."
                )
            if self.tactile_d360_pretrained_model_id is not None and self.tactile_d360_model_size != "base":
                raise ValueError(
                    "`tactile_d360_pretrained_model_id` currently points to base-sized Sparsh-X weights. "
                    "Set `tactile_d360_model_size='base'` or disable pretrained loading with "
                    "`tactile_d360_pretrained_model_id=None`."
                )
            if self.tactile_d360_pretrained_model_id is not None and self.tactile_d360_input_channels != 6:
                raise ValueError(
                    "`facebook/sparsh-x-img` expects 6 input channels. Keep "
                    "`tactile_d360_input_channels=6` or disable pretrained loading."
                )
            if (
                self.tactile_d360_pretrained_model_id is not None
                and tuple(self.tactile_d360_image_size) != (224, 224)
            ):
                raise ValueError(
                    "`facebook/sparsh-x-img` expects 224x224 inputs. Keep "
                    "`tactile_d360_image_size=(224, 224)` or disable pretrained loading."
                )
            if self.tactile_d360_patch_size <= 0:
                raise ValueError(f"`tactile_d360_patch_size` must be positive. Got {self.tactile_d360_patch_size}.")
            if self.tactile_d360_pretrained_model_id is not None and self.tactile_d360_patch_size != 16:
                raise ValueError(
                    "`facebook/sparsh-x-img` expects patch size 16. Keep `tactile_d360_patch_size=16` "
                    "or disable pretrained loading."
                )
            if self.tactile_d360_num_register_tokens < 0:
                raise ValueError(
                    "`tactile_d360_num_register_tokens` must be non-negative. "
                    f"Got {self.tactile_d360_num_register_tokens}."
                )
            if self.tactile_d360_pretrained_model_id is not None and self.tactile_d360_num_register_tokens != 1:
                raise ValueError(
                    "`facebook/sparsh-x-img` expects one register token. Keep "
                    "`tactile_d360_num_register_tokens=1` or disable pretrained loading."
                )
            if self.tactile_d360_pool not in ("register", "mean"):
                raise ValueError(
                    f"`tactile_d360_pool` must be 'register' or 'mean'. Got {self.tactile_d360_pool}."
                )
            if self.tactile_d360_fusion_type not in ("bottleneck", "vanilla"):
                raise ValueError(
                    "`tactile_d360_fusion_type` must be 'bottleneck' or 'vanilla'. "
                    f"Got {self.tactile_d360_fusion_type}."
                )
            if self.tactile_d360_pretrained_model_id is not None and self.tactile_d360_fusion_type != "vanilla":
                raise ValueError(
                    "`facebook/sparsh-x-img` was trained with vanilla fusion (no bottleneck tokens). "
                    "Keep `tactile_d360_fusion_type='vanilla'` or disable pretrained loading."
                )
            if self.tactile_d360_fusion_layer < 0:
                raise ValueError(
                    f"`tactile_d360_fusion_layer` must be non-negative. Got {self.tactile_d360_fusion_layer}."
                )
            if self.tactile_d360_num_bottlenecks < 0:
                raise ValueError(
                    "`tactile_d360_num_bottlenecks` must be non-negative. "
                    f"Got {self.tactile_d360_num_bottlenecks}."
                )
            if len(self.tactile_d360_image_size) != 2 or any(d <= 0 for d in self.tactile_d360_image_size):
                raise ValueError(
                    "`tactile_d360_image_size` must be a pair of positive integers. "
                    f"Got {self.tactile_d360_image_size}."
                )
            if self.tactile_d360_input_channels <= 0:
                raise ValueError(
                    f"`tactile_d360_input_channels` must be positive. Got {self.tactile_d360_input_channels}."
                )
        if self.tactile_encoder_type == "t3":
            if self.tactile_t3_model_size not in ("t3_medium", "t3_large"):
                raise ValueError(
                    f"`tactile_t3_model_size` must be 't3_medium' or 't3_large'. Got {self.tactile_t3_model_size}."
                )
            if len(self.tactile_t3_image_size) != 2 or any(d <= 0 for d in self.tactile_t3_image_size):
                raise ValueError("`tactile_t3_image_size` must be a pair of positive integers.")
            if self.tactile_t3_input_channels <= 0:
                raise ValueError("`tactile_t3_input_channels` must be positive.")
            if self.tactile_t3_patch_size <= 0:
                raise ValueError("`tactile_t3_patch_size` must be positive.")
            if any(d % self.tactile_t3_patch_size != 0 for d in self.tactile_t3_image_size):
                raise ValueError("`tactile_t3_patch_size` must divide both T3 image dimensions.")
            if self.tactile_t3_pool not in ("cls", "mean"):
                raise ValueError(f"`tactile_t3_pool` must be 'cls' or 'mean'. Got {self.tactile_t3_pool}.")
        if self.tactile_encoder_type == "anytouch":
            if len(self.tactile_anytouch_image_size) != 2 or any(d <= 0 for d in self.tactile_anytouch_image_size):
                raise ValueError("`tactile_anytouch_image_size` must be a pair of positive integers.")
            if self.tactile_anytouch_image_size[0] != self.tactile_anytouch_image_size[1]:
                raise ValueError("AnyTouch expects square tactile images.")
            if self.tactile_anytouch_input_channels <= 0:
                raise ValueError("`tactile_anytouch_input_channels` must be positive.")
            if self.tactile_anytouch_sensor_type < 0:
                raise ValueError("`tactile_anytouch_sensor_type` must be non-negative.")
            if self.tactile_anytouch_data_type not in (0,):
                raise ValueError("Only `tactile_anytouch_data_type=0` image inputs are supported.")

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if not self.tactile_image_keys and self.tactile_key_substrings:
            self.tactile_image_keys = [
                key
                for key in self.image_features
                if any(marker in key for marker in self.tactile_key_substrings)
            ]

        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        missing_tactile_keys = [key for key in self.tactile_image_keys if key not in self.image_features]
        if missing_tactile_keys:
            raise ValueError(
                "`tactile_image_keys` must refer to VISUAL input features. "
                f"Missing or non-visual keys: {missing_tactile_keys}."
            )

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.rgb_image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        if len(self.rgb_image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.rgb_image_features.items()))
            for key, image_ft in self.rgb_image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all RGB image shapes to match."
                    )
        if len(self.tactile_image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.tactile_image_features.items()))
            for key, image_ft in self.tactile_image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        "but we expect all tactile image shapes to match."
                    )

    @property
    def tactile_image_features(self) -> dict:
        tactile_keys = set(self.tactile_image_keys)
        return {key: ft for key, ft in self.image_features.items() if key in tactile_keys}

    @property
    def rgb_image_features(self) -> dict:
        tactile_keys = set(self.tactile_image_keys)
        return {key: ft for key, ft in self.image_features.items() if key not in tactile_keys}

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
