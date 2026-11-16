#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_tactile")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # tactile Encoder
    tactile_image_keys: list[str] = field(default_factory=list)
    tactile_key_substrings: tuple[str, ...] = ("digit",)
    tactile_encoder_type: str = "d360"  # "cnn" | "resnet18" | "resnet50" | "d360" | "t3" | "anytouch"
    tactile_channels: tuple[int, ...] = (32, 64, 128)
    tactile_backbone: str = "resnet18"
    tactile_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    tactile_feature_dim: int = 128
    use_group_norm: bool = False
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

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.tactile_encoder_type not in ("cnn", "resnet18", "resnet50", "d360", "t3", "anytouch"):
            raise ValueError(
                "`tactile_encoder_type` must be 'cnn', 'resnet18', 'resnet50', "
                "'d360', 't3' or 'anytouch'. "
                f"Got {self.tactile_encoder_type}."
            )
        if self.tactile_encoder_type in ("resnet18", "resnet50") and self.tactile_backbone != self.tactile_encoder_type:
            raise ValueError(
                "`tactile_backbone` must match `tactile_encoder_type`. "
                f"Got tactile_encoder_type={self.tactile_encoder_type!r} and tactile_backbone={self.tactile_backbone!r}."
            )
        if self.tactile_feature_dim <= 0:
            raise ValueError(f"`tactile_feature_dim` must be positive. Got {self.tactile_feature_dim}.")
        if self.tactile_encoder_type == "t3":
            if self.tactile_t3_model_size not in ("t3_medium", "t3_large"):
                raise ValueError(
                    f"`tactile_t3_model_size` must be 't3_medium' or 't3_large'. Got {self.tactile_t3_model_size}."
                )
            if len(self.tactile_t3_image_size) != 2 or any(d <= 0 for d in self.tactile_t3_image_size):
                raise ValueError(f"`tactile_t3_image_size` must be a pair of positive integers.")
            if self.tactile_t3_input_channels <= 0:
                raise ValueError(f"`tactile_t3_input_channels` must be positive.")
            if self.tactile_t3_patch_size <= 0:
                raise ValueError(f"`tactile_t3_patch_size` must be positive.")
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

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.tactile_image_keys and self.tactile_key_substrings:
            self.tactile_image_keys = [
                key
                for key in self.image_features
                if any(marker in key for marker in self.tactile_key_substrings)
            ]
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        missing_tactile_keys = [k for k in self.tactile_image_keys if k not in self.image_features]
        if missing_tactile_keys:
            raise ValueError(
                f"`tactile_image_keys` must reference visual input features. "
                f"Missing: {missing_tactile_keys}."
            )

    @property
    def tactile_image_features(self) -> dict:
        keys = set(self.tactile_image_keys)
        return {k: ft for k, ft in self.image_features.items() if k in keys}

    @property
    def rgb_image_features(self) -> dict:
        keys = set(self.tactile_image_keys)
        return {k: ft for k, ft in self.image_features.items() if k not in keys}

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
