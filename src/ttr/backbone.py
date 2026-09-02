"""Backbone: a thin wrapper over timm VisionTransformer with an explicit token layout.

Layout of the token axis, always: [cls] [trained registers] [test-time registers] [patches].
"""

from __future__ import annotations

from collections.abc import Callable

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

from ttr.config import BackboneCfg


class CaptureHandle:
    def __init__(self) -> None:
        self.data: dict[int, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.data.clear()


class Backbone(nn.Module):
    def __init__(self, model: VisionTransformer) -> None:
        super().__init__()
        if not isinstance(model, VisionTransformer):
            raise TypeError("Backbone only wraps timm VisionTransformer")
        self.model = model
        self.embed_dim = model.embed_dim
        self.depth = len(model.blocks)
        ps = model.patch_embed.patch_size
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.num_cls = 1 if model.cls_token is not None else 0
        self.num_trained_reg = int(getattr(model, "num_reg_tokens", 0) or 0)
        self.num_tt_reg = 0
        self.tt_reg_init = "zeros"

    # ---- layout -----------------------------------------------------------------
    def grid(self, img_hw: tuple[int, int]) -> tuple[int, int]:
        return img_hw[0] // self.patch_size, img_hw[1] // self.patch_size

    def num_patches(self, img_hw: tuple[int, int]) -> int:
        h, w = self.grid(img_hw)
        return h * w

    def prefix_len(self) -> int:
        return self.num_cls + self.num_trained_reg + self.num_tt_reg

    def patch_slice(self) -> slice:
        return slice(self.prefix_len(), None)

    def tt_reg_slice(self) -> slice:
        start = self.num_cls + self.num_trained_reg
        return slice(start, start + self.num_tt_reg)

    def set_tt_registers(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        self.num_tt_reg = n

    # ---- forward ----------------------------------------------------------------
    def forward_tokens(self, x: Tensor) -> Tensor:
        m = self.model
        x = m.patch_embed(x)
        x = m._pos_embed(x)  # prepends cls + trained registers, adds pos embed
        if self.num_tt_reg > 0:
            b, _, c = x.shape
            reg = x.new_zeros(b, self.num_tt_reg, c)
            p = self.num_cls + self.num_trained_reg
            x = torch.cat([x[:, :p], reg, x[:, p:]], dim=1)
        x = m.patch_drop(x)
        x = m.norm_pre(x)
        for blk in m.blocks:
            x = blk(x)
        return m.norm(x)

    def forward_features(self, x: Tensor) -> Tensor:
        h, w = self.grid(tuple(x.shape[-2:]))
        t = self.forward_tokens(x)[:, self.patch_slice()]
        return t.transpose(1, 2).reshape(t.shape[0], self.embed_dim, h, w)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)

    # ---- introspection ------------------------------------------------------------
    def set_fused_attention(self, fused: bool) -> None:
        for blk in self.model.blocks:
            blk.attn.fused_attn = fused

    def _layers(self, layers: list[int] | None) -> list[int]:
        return list(range(self.depth)) if layers is None else list(layers)

    def capture(self, what: str, layers: list[int] | None = None) -> CaptureHandle:
        """Capture per-layer tensors from the next forward passes.

        what="resid": block output (B, T, C).
        what="mlp_act": MLP hidden activation after the nonlinearity (B, T, hidden).
        what="attn": attention probabilities (B, heads, T, T); needs fused_attn=False.
        """
        handle = CaptureHandle()
        for i in self._layers(layers):
            blk = self.model.blocks[i]
            if what == "resid":
                mod, pick = blk, lambda m, inp, out: out
            elif what == "mlp_act":
                mod, pick = blk.mlp.act, lambda m, inp, out: out
            elif what == "attn":
                if blk.attn.fused_attn:
                    raise RuntimeError("call set_fused_attention(False) before capturing attention")
                mod, pick = blk.attn.attn_drop, lambda m, inp, out: inp[0]
            else:
                raise ValueError(f"unknown capture target {what!r}")

            def _hook(m, inp, out, i=i, pick=pick):
                handle.data[i] = pick(m, inp, out).detach()

            handle._handles.append(mod.register_forward_hook(_hook))
        return handle

    def add_mlp_hook(
        self, layer: int, fn: Callable[[nn.Module, tuple, Tensor], Tensor | None]
    ) -> torch.utils.hooks.RemovableHandle:
        """Forward hook on blocks[layer].mlp.act; fn may return a replacement tensor."""
        return self.model.blocks[layer].mlp.act.register_forward_hook(fn)


def tiny_backbone(
    depth: int = 2,
    embed_dim: int = 32,
    heads: int = 2,
    img: int = 56,
    patch: int = 14,
    reg_tokens: int = 0,
    mlp_ratio: float = 2.0,
) -> Backbone:
    """Random tiny ViT for CPU tests. Same code path as the real models."""
    m = VisionTransformer(
        img_size=img,
        patch_size=patch,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=heads,
        num_classes=0,
        reg_tokens=reg_tokens,
        mlp_ratio=mlp_ratio,
        dynamic_img_size=True,
    )
    m.eval()
    return Backbone(m)


def build_backbone(cfg: BackboneCfg) -> Backbone:
    if cfg.name == "tiny":
        bb = tiny_backbone(reg_tokens=0)
    else:
        m = timm.create_model(
            cfg.name,
            pretrained=cfg.pretrained,
            num_classes=0,
            img_size=cfg.img_size,
            dynamic_img_size=True,
        )
        m.eval()
        bb = Backbone(m)

    if cfg.registers == "trained":
        if bb.num_trained_reg == 0:
            raise ValueError(
                f"{cfg.name} has no trained registers; use a *_reg4_* "
                "checkpoint for registers=trained"
            )
    elif cfg.registers == "test_time":
        bb.set_tt_registers(cfg.num_test_time_registers)
    elif cfg.registers != "none":
        raise ValueError(f"unknown registers mode {cfg.registers!r}")

    if cfg.capture_attention:
        bb.set_fused_attention(False)
    return bb


def normalization_for(bb: Backbone) -> tuple[list[float], list[float]]:
    """Mean/std the checkpoint was trained with; ImageNet defaults for random models."""
    pc = getattr(bb.model, "pretrained_cfg", None) or {}
    mean = list(pc.get("mean", (0.485, 0.456, 0.406)))
    std = list(pc.get("std", (0.229, 0.224, 0.225)))
    return mean, std
