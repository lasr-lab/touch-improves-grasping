import copy
import math
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor

from policy.fm_tactile.configuration_fm import FMConfig
from policy.common.backbones import build_sequence_encoder
from policy.common.tactile_encoders import make_tactile_encoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE,
)


OBS_TACTILE_IMAGES = "observation.tactile_images"


# ==================== Transformer Building Blocks ====================


class FMRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class FMSwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.project = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: Tensor) -> Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class FMAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float,
                 block_size: int = 100, causal: bool = False, bias: bool = False, qk_norm: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = FMRMSNorm(n_embd // self.n_head, eps=1e-6)
            self.k_norm = FMRMSNorm(n_embd // self.n_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        B, T, C = x.size()
        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FMMLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            FMSwishGLU(n_embd, 4 * n_embd),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class FMBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float,
                 mlp_pdrop: float, block_size: int = 100, causal: bool = True,
                 use_cross_attention: bool = False, bias: bool = False, qk_norm: bool = True):
        super().__init__()
        self.ln_1 = FMRMSNorm(n_embd, eps=1e-6)
        self.attn = FMAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = FMAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm)
            self.ln3 = FMRMSNorm(n_embd, eps=1e-6)
        self.ln_2 = FMRMSNorm(n_embd, eps=1e-6)
        self.mlp = FMMLP(n_embd, bias, mlp_pdrop)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context)
        x = x + self.mlp(self.ln_2(x))
        return x


class FMAdaLNZero(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, c: Tensor):
        return self.modulation(c).chunk(6, dim=-1)


class FMConditionedBlock(FMBlock):
    def __init__(self, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop,
                 block_size=100, causal=True, use_cross_attention=False, bias=False, qk_norm=True):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                         causal, use_cross_attention, bias, qk_norm)
        self.adaLN_zero = FMAdaLNZero(n_embd)

    def forward(self, x: Tensor, c: Tensor, context: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        x_attn = self.ln_1(x)
        x_attn = shift_msa + (x_attn * scale_msa)
        x = x + gate_msa * self.attn(x_attn)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context)
        x_mlp = self.ln_2(x)
        x_mlp = shift_mlp + (x_mlp * scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        return x


class FMTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float,
                 n_layers: int, block_size: int = 100, causal: bool = True, bias: bool = False,
                 mlp_pdrop: float = 0.0, qk_norm: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            FMBlock(embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                    causal=causal, bias=bias, qk_norm=qk_norm)
            for _ in range(n_layers)
        ])
        self.ln = FMRMSNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x)
        x = self.ln(x)
        return x


class FMTransformerFiLMEncoder(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, attn_pdrop: float, resid_pdrop: float,
                 n_layers: int, block_size: int = 100, causal: bool = True, bias: bool = False,
                 mlp_pdrop: float = 0.0, qk_norm: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            FMConditionedBlock(embed_dim, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size,
                               causal=causal, bias=bias, qk_norm=qk_norm)
            for _ in range(n_layers)
        ])
        self.ln = FMRMSNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        for layer in self.blocks:
            x = layer(x, c)
        x = self.ln(x)
        return x


class FMSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==================== FM Inner Model (vector field network v_θ) ====================


class FMInnerModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        embed_dim: int,
        embed_pdrop: float,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        goal_drop: float = 0.1,
        bias: bool = False,
        linear_output: bool = True,
        use_ada_conditioning: bool = False,
        use_pos_emb: bool = False,
        backbone_type: str = "transformer",
        backbone_kwargs: dict | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.use_ada_conditioning = use_ada_conditioning
        self.use_pos_emb = use_pos_emb

        seq_size = goal_seq_len + obs_seq_len + action_seq_len
        block_size = seq_size + 1  # +1 for time token

        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.drop = nn.Dropout(embed_pdrop)
        self.cond_mask_prob = goal_drop

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, goal_seq_len + action_seq_len, embed_dim))

        self.t_emb = nn.Sequential(
            FMSinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.backbone_type = backbone_type
        if backbone_type == "transformer":
            if use_ada_conditioning:
                self.encoder = FMTransformerFiLMEncoder(
                    embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop, n_layers=n_layers, block_size=block_size,
                    causal=True, bias=bias, mlp_pdrop=mlp_pdrop, qk_norm=True,
                )
            else:
                self.encoder = FMTransformerEncoder(
                    embed_dim=embed_dim, n_heads=n_heads, attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop, n_layers=n_layers, block_size=block_size,
                    causal=True, bias=bias, mlp_pdrop=mlp_pdrop, qk_norm=True,
                )
        else:
            self.encoder = build_sequence_encoder(
                backbone_type=backbone_type,
                embed_dim=embed_dim,
                n_layers=n_layers,
                use_ada_conditioning=use_ada_conditioning,
                **(backbone_kwargs or {}),
            )

        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), nn.GELU(), nn.Linear(100, action_dim)
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, FMInnerModel):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states: Tensor, actions: Tensor, goals: Tensor, t: Tensor) -> Tensor:
        b, n_obs, _ = states.size()
        _, t_a, _ = actions.size()

        goal_embed = self.goal_emb(goals)
        goal_embed = goal_embed + self.pos_emb[:, :self.goal_seq_len, :]
        goal_x = self.drop(goal_embed)

        action_embed = self.action_emb(actions)
        action_embed = action_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t_a), :]
        action_x = self.drop(action_embed)

        state_embed = self.tok_emb(states)
        if self.use_pos_emb:
            state_embed = state_embed + self.pos_emb[:, (self.goal_seq_len + t_a):(self.goal_seq_len + t_a + n_obs), :]
        state_x = self.drop(state_embed)

        t_in = einops.rearrange(t, "b -> b 1")
        emb_t = self.t_emb(t_in)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, "b d -> b 1 d")

        input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)

        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq, emb_t)
        else:
            encoder_output = self.encoder(input_seq)

        pred_v = self.action_pred(encoder_output[:, -self.action_seq_len:, :])
        return pred_v


# ==================== Rectified Flow wrapper ====================


class RectifiedFlow(nn.Module):
    def __init__(self, inner_model: FMInnerModel, logit_normal_t: bool = False):
        super().__init__()
        self.inner_model = inner_model
        self.logit_normal_t = logit_normal_t

    def loss(self, state: Tensor, action: Tensor, goal: Tensor) -> Tensor:
        b = action.size(0)
        if self.logit_normal_t:
            nt = torch.randn((b,), device=action.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,), device=action.device)
        texp = t.view([b, *([1] * len(action.shape[1:]))])
        z1 = torch.randn_like(action)
        zt = (1 - texp) * action + texp * z1

        v_pred = self.inner_model(state, zt, goal, t)
        target = z1 - action
        return (v_pred - target).pow(2).flatten(1).mean()

    @torch.no_grad()
    def sample(self, z: Tensor, state: Tensor, goal: Tensor, sample_steps: int = 50) -> Tensor:
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt_t = torch.tensor([dt] * b, device=z.device).view([b, *([1] * len(z.shape[1:]))])
        for i in range(sample_steps, 0, -1):
            t_val = i / sample_steps
            t = torch.tensor([t_val] * b, device=z.device)
            v = self.inner_model(state, z, goal, t)
            z = z - dt_t * v
        return z


# ==================== RGB Encoder ====================


class FMSpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class FMRgbEncoder(nn.Module):
    def __init__(self, config: FMConfig):
        super().__init__()
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None
        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        vision_backbone = config.vision_backbone.lower()
        backbone_model = getattr(torchvision.models, vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.is_vit = vision_backbone.startswith("vit")
        self.pool_mode = config.vision_pool_mode

        if self.is_vit:
            backbone_model.heads = nn.Identity()
            self.backbone = backbone_model
        elif self.pool_mode == "spatial_softmax":
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        else:
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-1]))

        if config.use_group_norm and not self.is_vit:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.rgb_image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)

        if self.is_vit:
            with torch.no_grad():
                backbone_out_dim = self.backbone(torch.zeros(dummy_shape)).shape[-1]
            self.pool = None
            self.feature_dim = config.vision_feature_dim
            self.out = nn.Linear(backbone_out_dim, self.feature_dim)
            self.relu = None
        elif self.pool_mode == "spatial_softmax":
            feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
            self.pool = FMSpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
            self.feature_dim = (
                config.vision_feature_dim if config.fixed_version else config.spatial_softmax_num_keypoints * 2
            )
            self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
            self.relu = nn.ReLU()
        else:
            with torch.no_grad():
                backbone_out_dim = self.backbone(torch.zeros(dummy_shape)).shape[1]
            self.pool = None
            self.feature_dim = config.vision_feature_dim
            self.out = nn.Linear(backbone_out_dim, self.feature_dim)
            self.relu = None

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = self.backbone(x)
        if self.is_vit:
            x = self.out(x)
        elif self.pool_mode == "spatial_softmax":
            x = torch.flatten(self.pool(x), start_dim=1)
            x = self.relu(self.out(x))
        else:
            x = torch.flatten(x, start_dim=1)
            x = self.out(x)
        return x


def _replace_submodules(root_module, predicate, func):
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
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


# ==================== Language Encoder ====================


class FMLanguageEncoder(nn.Module):
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", goal_dim: int = 512, freeze: bool = True):
        super().__init__()
        from transformers import CLIPTextModel, AutoTokenizer

        self.clip_text = CLIPTextModel.from_pretrained(clip_model_name)
        for param in self.clip_text.parameters():
            param.data = param.data.clone()
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
        clip_dim = self.clip_text.config.hidden_size

        if clip_dim != goal_dim:
            self.proj = nn.Linear(clip_dim, goal_dim)
        else:
            self.proj = None

        if freeze:
            for param in self.clip_text.parameters():
                param.requires_grad = False

    def tokenize(self, texts: list[str], device: torch.device) -> dict[str, Tensor]:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return {k: v.to(device) for k, v in encoded.items()}

    @torch.no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        outputs = self.clip_text(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.pooler_output.float()
        if self.proj is not None:
            emb = self.proj(emb)
        return emb.unsqueeze(1)


# ==================== FM Model ====================


def _collect_backbone_kwargs(config: FMConfig) -> dict:
    if config.backbone_type == "mamba":
        return dict(
            mamba_layer=config.mamba_layer,
            mamba_d_state=config.mamba_d_state,
            mamba_d_conv=config.mamba_d_conv,
            mamba_expand=config.mamba_expand,
            mamba_headdim=config.mamba_headdim,
        )
    if config.backbone_type == "xlstm":
        return dict(
            xlstm_num_heads=config.xlstm_num_heads,
            xlstm_chunkwise_kernel=config.xlstm_chunkwise_kernel,
            xlstm_context_length=config.xlstm_context_length,
        )
    return {}


class FMModel(nn.Module):
    def __init__(self, config: FMConfig):
        super().__init__()
        self.config = config

        self.num_cameras = len(config.rgb_image_features) if config.rgb_image_features else 0
        self.num_tactile_sensors = (
            len(config.tactile_image_features) if config.tactile_image_features else 0
        )
        self.num_visual_inputs = self.num_cameras + self.num_tactile_sensors
        self.per_cam_tokens = bool(config.obs_tokens_per_camera and self.num_visual_inputs > 0)
        obs_dim = 0
        per_cam_feature_dim = 0
        tactile_feature_dim = 0
        if config.rgb_image_features:
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [FMRgbEncoder(config) for _ in range(self.num_cameras)]
                self.rgb_encoder = nn.ModuleList(encoders)
                per_cam_feature_dim = encoders[0].feature_dim
            else:
                self.rgb_encoder = FMRgbEncoder(config)
                per_cam_feature_dim = self.rgb_encoder.feature_dim
            obs_dim += per_cam_feature_dim * self.num_cameras

        if config.tactile_image_features:
            tactile_encoders = [make_tactile_encoder(config) for _ in range(self.num_tactile_sensors)]
            self.tactile_encoder = nn.ModuleList(tactile_encoders)
            tactile_feature_dim = tactile_encoders[0].feature_dim
            obs_dim += tactile_feature_dim * self.num_tactile_sensors

        if config.robot_state_feature and config.use_robot_state:
            state_dim = config.robot_state_feature.shape[0]
            self.state_emb = nn.Linear(state_dim, config.embed_dim)
            self._has_state = True
        else:
            self._has_state = False

        if config.env_state_feature:
            obs_dim += config.env_state_feature.shape[0]

        if self.per_cam_tokens:
            if config.env_state_feature:
                raise NotImplementedError(
                    "env_state_feature is not supported with obs_tokens_per_camera=True. "
                    "Disable obs_tokens_per_camera or drop env_state_feature."
                )
            if self.num_cameras > 0 and per_cam_feature_dim != config.embed_dim:
                self.cam_proj = nn.Linear(per_cam_feature_dim, config.embed_dim)
            else:
                self.cam_proj = None
            if self.num_tactile_sensors > 0 and tactile_feature_dim != config.embed_dim:
                self.tactile_proj = nn.Linear(tactile_feature_dim, config.embed_dim)
            else:
                self.tactile_proj = None
            self.obs_proj = None
        else:
            if obs_dim != config.embed_dim and obs_dim > 0:
                self.obs_proj = nn.Linear(obs_dim, config.embed_dim)
            else:
                self.obs_proj = None
            self.cam_proj = None
            self.tactile_proj = None

        if config.use_language_conditioning:
            self.language_encoder = FMLanguageEncoder(
                clip_model_name=config.clip_model_name,
                goal_dim=config.goal_dim,
                freeze=config.freeze_clip,
            )
            self._has_language = True
        else:
            self._has_language = False

        action_dim = config.action_feature.shape[0]

        if self.per_cam_tokens:
            obs_seq_len = config.n_obs_steps * self.num_visual_inputs
        elif self.num_visual_inputs > 0 or config.env_state_feature:
            obs_seq_len = config.n_obs_steps
        else:
            obs_seq_len = 0
        if self._has_state:
            obs_seq_len += config.n_obs_steps

        inner_model = FMInnerModel(
            obs_dim=config.embed_dim,
            goal_dim=config.goal_dim,
            action_dim=action_dim,
            embed_dim=config.embed_dim,
            embed_pdrop=config.embed_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            mlp_pdrop=config.mlp_pdrop,
            n_layers=config.n_enc_layers,
            n_heads=config.n_heads,
            goal_seq_len=config.goal_seq_len,
            obs_seq_len=obs_seq_len,
            action_seq_len=config.horizon,
            goal_drop=config.goal_drop,
            bias=False,
            linear_output=config.linear_output,
            use_ada_conditioning=config.use_ada_conditioning,
            use_pos_emb=False,
            backbone_type=config.backbone_type,
            backbone_kwargs=_collect_backbone_kwargs(config),
        )

        self.flow = RectifiedFlow(inner_model, logit_normal_t=config.logit_normal_t)

        self.action_dim = action_dim
        self.horizon = config.horizon
        self.sampling_steps = config.sampling_steps

    def _encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        if OBS_IMAGES in batch:
            batch_size, n_obs_steps = batch[OBS_IMAGES].shape[:2]
        elif OBS_TACTILE_IMAGES in batch:
            batch_size, n_obs_steps = batch[OBS_TACTILE_IMAGES].shape[:2]
        elif OBS_STATE in batch:
            batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        elif OBS_ENV_STATE in batch:
            batch_size, n_obs_steps = batch[OBS_ENV_STATE].shape[:2]
        else:
            raise ValueError("No observation key found in batch.")

        if self.per_cam_tokens:
            token_groups = []
            if self.config.rgb_image_features and self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                per_cam = [
                    enc(imgs) for enc, imgs in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                per_cam = [
                    einops.rearrange(e, "(b s) f -> b s f", b=batch_size, s=n_obs_steps)
                    for e in per_cam
                ]
                cam_feats = torch.stack(per_cam, dim=2)
                if self.cam_proj is not None:
                    cam_feats = self.cam_proj(cam_feats)
                token_groups.append(cam_feats)
            elif self.config.rgb_image_features:
                flat = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                cam_feats = einops.rearrange(
                    flat, "(b s n) f -> b s n f", b=batch_size, s=n_obs_steps, n=self.num_cameras
                )
                if self.cam_proj is not None:
                    cam_feats = self.cam_proj(cam_feats)
                token_groups.append(cam_feats)
            if self.config.tactile_image_features:
                images_per_sensor = einops.rearrange(batch[OBS_TACTILE_IMAGES], "b s n ... -> n (b s) ...")
                tactile = [
                    enc(imgs) for enc, imgs in zip(self.tactile_encoder, images_per_sensor, strict=True)
                ]
                tactile = [
                    einops.rearrange(e, "(b s) f -> b s f", b=batch_size, s=n_obs_steps)
                    for e in tactile
                ]
                tactile_feats = torch.stack(tactile, dim=2)
                if self.tactile_proj is not None:
                    tactile_feats = self.tactile_proj(tactile_feats)
                token_groups.append(tactile_feats)
            obs_features = einops.rearrange(torch.cat(token_groups, dim=2), "b s n f -> b (s n) f")
        else:
            features_list = []
            if self.config.rgb_image_features:
                if self.config.use_separate_rgb_encoder_per_camera:
                    images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                    img_features = torch.cat([
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ])
                    img_features = einops.rearrange(
                        img_features, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                    )
                else:
                    img_features = self.rgb_encoder(
                        einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                    )
                    img_features = einops.rearrange(
                        img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                    )
                features_list.append(img_features)

            if self.config.tactile_image_features:
                images_per_sensor = einops.rearrange(batch[OBS_TACTILE_IMAGES], "b s n ... -> n (b s) ...")
                tactile_features = torch.cat([
                    encoder(images)
                    for encoder, images in zip(self.tactile_encoder, images_per_sensor, strict=True)
                ])
                tactile_features = einops.rearrange(
                    tactile_features, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
                features_list.append(tactile_features)

            if self.config.env_state_feature:
                features_list.append(batch[OBS_ENV_STATE])

            if features_list:
                obs_features = torch.cat(features_list, dim=-1)
                if self.obs_proj is not None:
                    obs_features = self.obs_proj(obs_features)
            else:
                any_obs = next(iter(batch.values()))
                obs_features = torch.zeros(
                    batch_size, n_obs_steps, self.config.embed_dim, device=any_obs.device
                )

        if self._has_state:
            state_tokens = self.state_emb(batch[OBS_STATE])
            obs_features = torch.cat([obs_features, state_tokens], dim=1)

        return obs_features

    def _encode_goal(self, batch: dict[str, Tensor], batch_size: int, device, dtype) -> Tensor:
        if self._has_language and OBS_LANGUAGE_TOKENS in batch:
            goal = self.language_encoder(
                input_ids=batch[OBS_LANGUAGE_TOKENS],
                attention_mask=batch.get(OBS_LANGUAGE_ATTENTION_MASK),
            ).to(dtype)
        elif self._has_language and "task" in batch:
            task_strings = batch["task"]
            if isinstance(task_strings, str):
                task_strings = [task_strings]
            tokenized = self.language_encoder.tokenize(task_strings, device)
            goal = self.language_encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask"),
            ).to(dtype)
        else:
            goal = torch.zeros(batch_size, self.config.goal_seq_len, self.config.goal_dim,
                               device=device, dtype=dtype)
        return goal

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        perceptual_emb = self._encode_observations(batch)
        batch_size = perceptual_emb.shape[0]
        goal = self._encode_goal(batch, batch_size, device, dtype)
        z = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=dtype)
        actions = self.flow.sample(z, perceptual_emb, goal, sample_steps=self.sampling_steps)
        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        device = get_device_from_parameters(self)
        perceptual_emb = self._encode_observations(batch)
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        goal = self._encode_goal(batch, batch_size, device, actions.dtype)
        loss = self.flow.loss(perceptual_emb, actions, goal)
        return loss


# ==================== FM Policy ====================


class FMPolicy(PreTrainedPolicy):

    config_class = FMConfig
    name = "fm_tactile"

    def __init__(self, config: FMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.fm = FMModel(config)
        if getattr(config, "use_ema", False):
            self.fm_ema = copy.deepcopy(self.fm)
            for p in self.fm_ema.parameters():
                p.requires_grad_(False)
            self.fm_ema.eval()
        else:
            self.fm_ema = None
        # if config.compile_model:
        if False:
            torch.set_float32_matmul_precision("high")
            self.fm.compute_loss = torch.compile(self.fm.compute_loss, mode=config.compile_mode)
            inference_model = self.fm_ema if self.fm_ema is not None else self.fm
            inference_model.generate_actions = torch.compile(
                inference_model.generate_actions,
                mode=config.compile_mode,
            )
        self.reset()

    def get_optim_params(self) -> dict:
        return self.fm.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fm_ema is not None:
            self.fm_ema.eval()
        return self

    @torch.no_grad()
    def update(self):
        if self.fm_ema is None:
            return
        decay = self.config.ema_decay
        ema_params = dict(self.fm_ema.named_parameters())
        for name, p in self.fm.named_parameters():
            ema_p = ema_params.get(name)
            if ema_p is None:
                continue
            ema_p.mul_(decay).add_(p.detach().to(ema_p.dtype), alpha=1.0 - decay)
        ema_buffers = dict(self.fm_ema.named_buffers())
        for name, b in self.fm.named_buffers():
            ema_b = ema_buffers.get(name)
            src = b.detach().to(device=b.device)
            if ema_b is None or ema_b.shape != src.shape:
                module = self.fm_ema
                parts = name.split(".")
                for part in parts[:-1]:
                    module = getattr(module, part)
                module.register_buffer(parts[-1], src.clone())
            else:
                ema_b.copy_(src.to(ema_b.dtype))

    def _inference_model(self) -> "FMModel":
        return self.fm_ema if self.fm_ema is not None else self.fm

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.robot_state_feature and self.config.use_robot_state:
            self._queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.rgb_image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.tactile_image_features:
            self._queues[OBS_TACTILE_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        language_keys = [k for k in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, "task") if k in batch]
        queued = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        for k in language_keys:
            queued[k] = batch[k]
        actions = self._inference_model().generate_actions(queued)
        return actions

    def _prepare_structured_observations(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.rgb_image_features or self.config.tactile_image_features:
            batch = dict(batch)
            if self.config.rgb_image_features:
                batch[OBS_IMAGES] = torch.stack(
                    [batch[key] for key in self.config.rgb_image_features], dim=-4
                )
            if self.config.tactile_image_features:
                batch[OBS_TACTILE_IMAGES] = torch.stack(
                    [batch[key] for key in self.config.tactile_image_features], dim=-4
                )
        return batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)
        batch = self._prepare_structured_observations(batch)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self._prepare_structured_observations(batch)
        if self.config.rgb_image_features or self.config.tactile_image_features:
            for key in self.config.rgb_image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            for key in self.config.tactile_image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            if self.config.rgb_image_features:
                batch[OBS_IMAGES] = torch.stack(
                    [batch[key] for key in self.config.rgb_image_features], dim=-4
                )
            if self.config.tactile_image_features:
                batch[OBS_TACTILE_IMAGES] = torch.stack(
                    [batch[key] for key in self.config.tactile_image_features], dim=-4
                )
        loss = self.fm.compute_loss(batch)
        return loss, None
