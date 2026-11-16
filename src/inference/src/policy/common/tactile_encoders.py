from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision
from torch import Tensor


# ==================== helpers ====================


def _replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


# ==================== CNN encoder ====================


class TactileCnnEncoder(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self._printed_first_forward = False
        images_shape = next(iter(config.tactile_image_features.values())).shape
        in_channels = images_shape[0]
        layers: list[nn.Module] = []
        for out_channels in config.tactile_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=math.gcd(out_channels, 16), num_channels=out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = config.tactile_feature_dim
        self.out = nn.Linear(in_channels, self.feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using TactileCnnEncoder", flush=True)
            self._printed_first_forward = True
        x = self.cnn(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        return self.out(x)


# ==================== ResNet encoder ====================


class TactileResNetEncoder(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self._printed_first_forward = False
        self.backbone_name = config.tactile_backbone
        if config.tactile_backbone == "resnet18":
            backbone_model = torchvision.models.resnet18(weights=config.tactile_backbone_weights)
        elif config.tactile_backbone == "resnet50":
            backbone_model = torchvision.models.resnet50(weights=config.tactile_backbone_weights)
        else:
            raise ValueError(f"`tactile_backbone` must be 'resnet18' or 'resnet50'. Got {config.tactile_backbone}.")
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-1]))
        if getattr(config, "use_group_norm", False):
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.tactile_image_features.values())).shape
        in_channels = images_shape[0]
        if in_channels != 3:
            original_conv = self.backbone[0]
            self.backbone[0] = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )

        with torch.no_grad():
            backbone_out_dim = self.backbone(torch.zeros(1, in_channels, *images_shape[1:])).shape[1]
        self.feature_dim = config.tactile_feature_dim
        self.out = nn.Linear(backbone_out_dim, self.feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print(f"Using TactileResNetEncoder({self.backbone_name})", flush=True)
            self._printed_first_forward = True
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        return self.out(x)


class TactileResNet18Encoder(TactileResNetEncoder):
    pass


# ==================== D360 encoder (Sparsh-X) ====================


_D360_MODEL_SPECS = {
    "tinier": {"embed_dim": 96, "depth": 8, "num_heads": 3},
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}


def _import_d360_modules():
    try:
        from tactile_ssl.model.d360_transformer import D360Transformer
        from tactile_ssl.model.layers import Attention
        return D360Transformer, Attention
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Using tactile_encoder_type='d360' requires the sparsh tactile_ssl package. "
            "Install sparsh-multisensory-touch with `pip install -e .` from its repo checkout."
        ) from exc


def _resolve_d360_checkpoint(config: Any) -> str | None:
    if config.tactile_d360_pretrained_model_id is None:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Using pretrained D360 weights requires `huggingface_hub`. Install it or set "
            "`tactile_d360_pretrained_model_id=None` to train from scratch."
        ) from exc
    return hf_hub_download(
        repo_id=config.tactile_d360_pretrained_model_id,
        filename=config.tactile_d360_pretrained_filename,
    )


def _extract_d360_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("D360 checkpoint must contain a state dict.")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("D360 checkpoint `model` entry must contain a state dict.")
    prefixes = ("teacher_encoder.backbone.", "model_encoder.", "encoder.", "backbone.")
    for prefix in prefixes:
        extracted = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if extracted:
            return extracted
    return state_dict


class TactileD360Encoder(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self._printed_first_forward = False
        D360Transformer, Attention = _import_d360_modules()

        images_shape = next(iter(config.tactile_image_features.values())).shape
        in_channels = images_shape[0]
        encoder_channels = config.tactile_d360_input_channels
        height, width = config.tactile_d360_image_size
        patch_size = config.tactile_d360_patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "`tactile_d360_patch_size` must divide tactile image height and width. "
                f"Got image size {(height, width)} and patch size {patch_size}."
            )

        spec = _D360_MODEL_SPECS[config.tactile_d360_model_size]
        self.resize = torchvision.transforms.Resize((height, width))
        self.input_channel_adapter = (
            nn.Conv2d(in_channels, encoder_channels, kernel_size=1)
            if in_channels not in (1, 3, encoder_channels)
            else None
        )
        normalization = {
            "img": {
                "avg": [0.09562227, 0.26593734, 0.32464101],
                "std": [0.08179263, 0.17255498, 0.18509875],
                "div": 1,
            }
        }
        self.encoder = D360Transformer(
            use_img=True,
            use_mic=False,
            use_imu=False,
            use_pressure=False,
            sensor_sizes={"img": (height, width)},
            sensor_chans={"img": encoder_channels},
            patch_sizes={"img": patch_size},
            embed_dim=spec["embed_dim"],
            depth=spec["depth"],
            num_heads=spec["num_heads"],
            num_register_tokens=config.tactile_d360_num_register_tokens,
            fusion_type=config.tactile_d360_fusion_type,
            fusion_layer=config.tactile_d360_fusion_layer,
            num_bottlenecks=config.tactile_d360_num_bottlenecks,
            pos_embed_fn="sinusoidal",
            normalization=normalization,
            attn_class=Attention,
        )
        ckpt_path = _resolve_d360_checkpoint(config)
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(_extract_d360_state_dict(checkpoint), strict=True)
        if config.tactile_d360_freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.pool = config.tactile_d360_pool
        self.feature_dim = config.tactile_feature_dim
        self.out = nn.Linear(spec["embed_dim"], self.feature_dim)

    def _adapt_input(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != self.encoder.sensor_sizes["img"]:
            x = self.resize(x)
        encoder_channels = self.encoder.sensor_chans["img"]
        if x.shape[1] == encoder_channels:
            return x
        if x.shape[1] == 3 and encoder_channels == 6:
            return torch.cat([x, x], dim=1)
        if x.shape[1] == 1:
            return x.repeat(1, encoder_channels, 1, 1)
        if self.input_channel_adapter is not None:
            return self.input_channel_adapter(x)
        raise ValueError(
            f"Cannot adapt tactile image channels from {x.shape[1]} to {encoder_channels}."
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using TactileD360Encoder", flush=True)
            self._printed_first_forward = True
        x = self._adapt_input(x)
        outputs = self.encoder.forward_features({"img": x})
        reg_tokens = outputs["x_norm_regtokens"]["img"]
        patch_tokens = outputs["x_norm_patchtokens"]["img"]
        if self.pool == "register" and reg_tokens.shape[1] > 0:
            x = reg_tokens.mean(dim=1)
        else:
            x = patch_tokens.mean(dim=1)
        return self.out(x)


# ==================== T3 encoder ====================


_T3_MODEL_SPECS = {
    "t3_medium": {"embed_dim": 768, "encoder_depth": 3, "trunk_depth": 9, "num_heads": 12},
    "t3_large": {"embed_dim": 1024, "encoder_depth": 6, "trunk_depth": 9, "num_heads": 8},
}


def _import_t3_modules(repo_path: str | None):
    if repo_path:
        path = str(Path(repo_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from t3.models import TransformerTrunk, ViTEncoder
        return ViTEncoder, TransformerTrunk
    except ModuleNotFoundError as exc:
        if exc.name == "t3":
            raise ImportError(
                "Using tactile_encoder_type='t3' requires the T3 package. "
                "Install it with `pip install -e /home/hk-project-p0024638/uhtfz/dev/t3` "
                "or set `tactile_t3_repo_path` to a local T3 checkout."
            ) from exc
        raise ImportError(
            f"Using tactile_encoder_type='t3' requires the T3 dependency `{exc.name}`. "
            "Install the T3 project dependencies, for example with "
            "`pip install -e /home/hk-project-p0024638/uhtfz/dev/t3`."
        ) from exc


def _resolve_t3_component_checkpoint(config: Any, relative_path: str) -> str | None:
    if getattr(config, "tactile_t3_pretrained_model_id", None) is None:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Using pretrained T3 weights requires `huggingface_hub`. Install it or set "
            "`tactile_t3_pretrained_model_id=None` to train T3 from scratch."
        ) from exc
    return hf_hub_download(
        repo_id=config.tactile_t3_pretrained_model_id,
        repo_type=getattr(config, "tactile_t3_pretrained_repo_type", "dataset"),
        filename=relative_path,
    )


def _load_t3_component(module: nn.Module, path: Path, component_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"T3 {component_name} checkpoint not found: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise ValueError(f"T3 {component_name} checkpoint must contain a state dict: {path}")
    module.load_state_dict(state_dict, strict=True)


class TactileT3Encoder(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self._printed_first_forward = False
        ViTEncoder, TransformerTrunk = _import_t3_modules(getattr(config, "tactile_t3_repo_path", None))

        images_shape = next(iter(config.tactile_image_features.values())).shape
        in_channels = images_shape[0]
        height, width = getattr(config, "tactile_t3_image_size", (224, 224))
        encoder_channels = getattr(config, "tactile_t3_input_channels", 3)
        model_size = getattr(config, "tactile_t3_model_size", "t3_medium")
        spec = _T3_MODEL_SPECS.get(model_size)
        if spec is None:
            raise ValueError(
                f"`tactile_t3_model_size` must be one of {tuple(_T3_MODEL_SPECS)}. Got {model_size!r}."
            )
        embed_dim = spec["embed_dim"]
        patch_size = getattr(config, "tactile_t3_patch_size", 16)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "`tactile_t3_patch_size` must divide tactile image height and width. "
                f"Got image size {(height, width)} and patch size {patch_size}."
            )

        self.resize = torchvision.transforms.Resize((height, width))
        self.input_channel_adapter = (
            nn.Conv2d(in_channels, encoder_channels, kernel_size=1)
            if in_channels not in (1, 3, encoder_channels)
            else None
        )
        self.encoder = ViTEncoder(
            img_size=(height, width),
            patch_size=patch_size,
            in_chans=encoder_channels,
            embed_dim=embed_dim,
            depth=spec["encoder_depth"],
            num_heads=spec["num_heads"],
            mlp_ratio=getattr(config, "tactile_t3_mlp_ratio", 4.0),
        )
        self.trunk = TransformerTrunk(
            embed_dim=embed_dim,
            depth=spec["trunk_depth"],
            num_heads=spec["num_heads"],
            mlp_ratio=getattr(config, "tactile_t3_mlp_ratio", 4.0),
            pooling_type="none",
        )

        checkpoint_dir = getattr(config, "tactile_t3_checkpoint_dir", None)
        self.encoder_domain = getattr(config, "tactile_t3_encoder_domain", "digit")
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir).expanduser()
            _load_t3_component(self.encoder, checkpoint_dir / "encoders" / f"{self.encoder_domain}.pth", "encoder")
            _load_t3_component(self.trunk, checkpoint_dir / "trunk.pth", "trunk")
        else:
            checkpoint_root = f"models/{model_size}"
            encoder_path = _resolve_t3_component_checkpoint(
                config, f"{checkpoint_root}/encoders/{self.encoder_domain}.pth"
            )
            trunk_path = _resolve_t3_component_checkpoint(config, f"{checkpoint_root}/trunk.pth")
            if encoder_path is not None and trunk_path is not None:
                _load_t3_component(self.encoder, Path(encoder_path), "encoder")
                _load_t3_component(self.trunk, Path(trunk_path), "trunk")

        if getattr(config, "tactile_t3_freeze", True):
            self.encoder.requires_grad_(False)
            self.trunk.requires_grad_(False)
            self.encoder.eval()
            self.trunk.eval()

        self.pool = getattr(config, "tactile_t3_pool", "cls")
        self.feature_dim = config.tactile_feature_dim
        self.out = nn.Linear(embed_dim, self.feature_dim)

    def _adapt_input(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != self.encoder.patch_embed.img_size:
            x = self.resize(x)
        encoder_channels = self.encoder.patch_embed.proj.in_channels
        if x.shape[1] == encoder_channels:
            return x
        if x.shape[1] == 1:
            return x.repeat(1, encoder_channels, 1, 1)
        if x.shape[1] == 3 and encoder_channels == 6:
            return torch.cat([x, x], dim=1)
        if self.input_channel_adapter is not None:
            return self.input_channel_adapter(x)
        raise ValueError(f"Cannot adapt tactile image channels from {x.shape[1]} to {encoder_channels}.")

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print(f"Using TactileT3Encoder({self.encoder_domain})", flush=True)
            self._printed_first_forward = True
        x = self._adapt_input(x)
        tokens = self.trunk(self.encoder(x))
        if self.pool == "cls":
            x = tokens[:, 0]
        elif self.pool == "mean":
            x = tokens[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"`tactile_t3_pool` must be 'cls' or 'mean'. Got {self.pool}.")
        return self.out(x)


# ==================== AnyTouch encoder ====================


def _import_anytouch_modules(repo_path: str | None):
    if repo_path:
        path = str(Path(repo_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import transformers.models.clip.modeling_clip as clip_modeling

        if not hasattr(clip_modeling, "CLIP_VISION_INPUTS_DOCSTRING"):
            clip_modeling.CLIP_VISION_INPUTS_DOCSTRING = ""
        if not hasattr(clip_modeling, "CLIPVisionTransformer"):
            clip_modeling.CLIPVisionTransformer = clip_modeling.CLIPVisionModel

        from model.mae_model import TactileVideoMAE
        from transformers import AutoConfig
        from transformers.models.vit.configuration_vit import ViTConfig
        return TactileVideoMAE, AutoConfig, ViTConfig
    except ModuleNotFoundError as exc:
        if exc.name == "model":
            raise ImportError(
                "Using tactile_encoder_type='anytouch' requires the AnyTouch repository. "
                "Set `tactile_anytouch_repo_path` to its checkout."
            ) from exc
        raise ImportError(
            f"Using tactile_encoder_type='anytouch' requires the dependency `{exc.name}`. "
            "Install the AnyTouch requirements in the active environment."
        ) from exc


def _extract_anytouch_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("AnyTouch checkpoint must contain a state dict.")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise ValueError("AnyTouch checkpoint state must be a dict.")
    prefix = "touch_mae_model."
    extracted = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not extracted:
        raise ValueError("AnyTouch checkpoint does not contain `touch_mae_model.*` weights.")
    return extracted


class TactileAnyTouchEncoder(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self._printed_first_forward = False
        TactileVideoMAE, AutoConfig, ViTConfig = _import_anytouch_modules(
            getattr(config, "tactile_anytouch_repo_path", None)
        )

        images_shape = next(iter(config.tactile_image_features.values())).shape
        in_channels = images_shape[0]
        height, width = getattr(config, "tactile_anytouch_image_size", (224, 224))
        encoder_channels = getattr(config, "tactile_anytouch_input_channels", 3)
        if height != width:
            raise ValueError(f"AnyTouch expects square tactile images. Got {(height, width)}.")

        args = type("AnyTouchArgs", (), {
            "use_sensor_token": getattr(config, "tactile_anytouch_use_sensor_token", True),
            "use_same_patchemb": getattr(config, "tactile_anytouch_use_same_patchemb", False),
            "new_decoder_sensor_token": getattr(config, "tactile_anytouch_new_decoder_sensor_token", False),
            "norm_pix_loss": False,
            "mask_ratio": 0.75,
        })()
        anytouch_root = Path(getattr(config, "tactile_anytouch_repo_path", "/home/hk-project-p0024638/uhtfz/dev/AnyTouch"))
        clip_config_path = getattr(config, "tactile_anytouch_clip_config_path", None)
        if clip_config_path is None:
            clip_config_path = anytouch_root / "CLIP-ViT-L-14-DataComp.XL-s13B-b90K" / "config.json"
        clip_config_path = Path(clip_config_path).expanduser()
        clip_config = AutoConfig.from_pretrained(str(clip_config_path))
        decoder_config = ViTConfig()
        decoder_config.encoder_stride = 14
        decoder_config.hidden_size = 512
        decoder_config.intermediate_size = 2048
        decoder_config.num_attention_heads = 16
        decoder_config.num_hidden_layers = 8
        decoder_config.patch_size = 14

        self.resize = torchvision.transforms.Resize((height, width))
        self.input_channel_adapter = (
            nn.Conv2d(in_channels, encoder_channels, kernel_size=1)
            if in_channels not in (1, 3, encoder_channels)
            else None
        )
        self.encoder = TactileVideoMAE(
            args,
            clip_config,
            decoder_config,
            num_frames=1,
            add_time_attn=False,
            tube_size=1,
        )

        checkpoint_path = getattr(config, "tactile_anytouch_checkpoint_path", None)
        if checkpoint_path is not None:
            checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(_extract_anytouch_state_dict(checkpoint), strict=True)

        if getattr(config, "tactile_anytouch_freeze", True):
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        self.sensor_type = getattr(config, "tactile_anytouch_sensor_type", 0)
        self.data_type = getattr(config, "tactile_anytouch_data_type", 0)
        self.feature_dim = config.tactile_feature_dim
        self.out = nn.Linear(clip_config.projection_dim, self.feature_dim)

    def _adapt_input(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != self.resize.size:
            x = self.resize(x)
        encoder_channels = self.encoder.touch_model.embeddings.patch_embedding.in_channels
        if x.shape[1] == encoder_channels:
            return x
        if x.shape[1] == 1:
            return x.repeat(1, encoder_channels, 1, 1)
        if x.shape[1] == 6 and encoder_channels == 3:
            return x[:, :3]
        if self.input_channel_adapter is not None:
            return self.input_channel_adapter(x)
        raise ValueError(f"Cannot adapt tactile image channels from {x.shape[1]} to {encoder_channels}.")

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using TactileAnyTouchEncoder", flush=True)
            self._printed_first_forward = True
        x = self._adapt_input(x)
        sensor_type = torch.full((x.shape[0],), self.sensor_type, dtype=torch.long, device=x.device)
        features, _, _ = self.encoder.forward_encoder(
            x,
            sensor_type=sensor_type,
            data_type=self.data_type,
            use_mask=False,
        )
        return self.out(features)


# ==================== Factory ====================


def make_tactile_encoder(config: Any) -> nn.Module:
    t = config.tactile_encoder_type
    if t == "cnn":
        return TactileCnnEncoder(config)
    if t == "resnet18":
        return TactileResNetEncoder(config)
    if t == "resnet50":
        return TactileResNetEncoder(config)
    if t == "d360":
        return TactileD360Encoder(config)
    if t == "t3":
        return TactileT3Encoder(config)
    if t == "anytouch":
        return TactileAnyTouchEncoder(config)
    raise ValueError(f"Unknown tactile encoder type: {t!r}")
