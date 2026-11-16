from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ==================== AdaLN-Zero-style FiLM (γ, β, gate) ====================


class _BlockFiLM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, c: Tensor):
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return shift, scale, gate


# ==================== Mamba ====================


def _make_mamba_block(
    embed_dim: int,
    layer_idx: int,
    layer_type: str,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
) -> nn.Module:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.mamba2 import Mamba2
    except ImportError as e:
        raise ImportError(
            "Mamba backbone requires the `mamba-ssm` package. "
            "Install with: pip install mamba-ssm"
        ) from e

    if layer_type.lower() == "mamba2":
        return Mamba2(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            layer_idx=layer_idx,
        )
    elif layer_type.lower() == "mamba1":
        return Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
    else:
        raise ValueError(f"Unknown mamba_layer: {layer_type!r}. Use 'Mamba1' or 'Mamba2'.")


class MambaEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        layer_type: str = "Mamba2",
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self._printed_first_forward = False
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([
            _make_mamba_block(embed_dim, i, layer_type, d_state, d_conv, expand, headdim)
            for i in range(n_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using MambaEncoder", flush=True)
            self._printed_first_forward = True
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.ln(x)


class MambaFiLMEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        layer_type: str = "Mamba2",
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self._printed_first_forward = False
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim, elementwise_affine=False)
                                    for _ in range(n_layers)])
        self.blocks = nn.ModuleList([
            _make_mamba_block(embed_dim, i, layer_type, d_state, d_conv, expand, headdim)
            for i in range(n_layers)
        ])
        self.films = nn.ModuleList([_BlockFiLM(embed_dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using MambaFiLMEncoder", flush=True)
            self._printed_first_forward = True
        c_vec = c.squeeze(1) if c.ndim == 3 else c
        for norm, block, film in zip(self.norms, self.blocks, self.films):
            shift, scale, gate = film(c_vec)
            shift, scale, gate = shift[:, None], scale[:, None], gate[:, None]
            h = norm(x) * (1 + scale) + shift
            x = x + gate * block(h)
        return self.ln(x)


# ==================== xLSTM ====================


def _make_xlstm_stack(
    embed_dim: int,
    n_layers: int,
    num_heads: int,
    chunkwise_kernel: str,
    context_length: int,
) -> nn.Module:
    try:
        from xlstm import (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
        )
    except ImportError as e:
        raise ImportError(
            "xLSTM backbone requires the `xlstm` package. "
            "Install with: pip install xlstm"
        ) from e

    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=num_heads,
            )
        ),
        context_length=context_length,
        num_blocks=n_layers,
        embedding_dim=embed_dim,
    )
    return xLSTMBlockStack(cfg)


class XLSTMEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        num_heads: int = 8,
        chunkwise_kernel: str = "parallel--native_autograd",
        context_length: int = 256,
    ):
        super().__init__()
        self._printed_first_forward = False
        self.stack = _make_xlstm_stack(embed_dim, n_layers, num_heads, chunkwise_kernel, context_length)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using XLSTMEncoder", flush=True)
            self._printed_first_forward = True
        return self.ln(self.stack(x))


class XLSTMFiLMEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        num_heads: int = 8,
        chunkwise_kernel: str = "parallel--native_autograd",
        context_length: int = 256,
    ):
        super().__init__()
        self._printed_first_forward = False
        self.stack = _make_xlstm_stack(embed_dim, n_layers, num_heads, chunkwise_kernel, context_length)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.film = _BlockFiLM(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        if not self._printed_first_forward:
            print("Using XLSTMFiLMEncoder", flush=True)
            self._printed_first_forward = True
        h = self.stack(x)
        c_vec = c.squeeze(1) if c.ndim == 3 else c
        shift, scale, gate = self.film(c_vec)
        shift, scale, gate = shift[:, None], scale[:, None], gate[:, None]
        h = self.norm(h) * (1 + scale) + shift
        return self.ln(x + gate * h)


# ==================== Factory ====================


def build_sequence_encoder(
    backbone_type: str,
    embed_dim: int,
    n_layers: int,
    use_ada_conditioning: bool,
    *,
    # Mamba kwargs
    mamba_layer: str = "Mamba2",
    mamba_d_state: int = 128,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mamba_headdim: int = 64,
    # xLSTM kwargs
    xlstm_num_heads: int = 8,
    xlstm_chunkwise_kernel: str = "parallel--native_autograd",
    xlstm_context_length: int = 256,
) -> nn.Module:
    if backbone_type == "mamba":
        cls = MambaFiLMEncoder if use_ada_conditioning else MambaEncoder
        return cls(
            embed_dim=embed_dim,
            n_layers=n_layers,
            layer_type=mamba_layer,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
        )
    elif backbone_type == "xlstm":
        cls = XLSTMFiLMEncoder if use_ada_conditioning else XLSTMEncoder
        return cls(
            embed_dim=embed_dim,
            n_layers=n_layers,
            num_heads=xlstm_num_heads,
            chunkwise_kernel=xlstm_chunkwise_kernel,
            context_length=xlstm_context_length,
        )
    else:
        raise ValueError(
            f"backbone_type {backbone_type!r} is not handled by build_sequence_encoder. "
            "Transformer should be built inline by the caller."
        )
