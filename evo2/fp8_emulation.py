"""
FP8 (e4m3) emulation for the StripedHyena input projections on Apple Silicon.

Why this exists
---------------
The 1B/20B/40B Evo 2 checkpoints set ``use_fp8_input_projections: True``: the qkv
input projection (vortex ``TELinear``) was *trained* with NVIDIA Transformer
Engine (TE) using per-tensor **delayed scaling** in e4m3. TE is CUDA/Hopper only,
so on Mac vortex falls back to a plain bf16 ``F.linear`` — which is numerically a
different operation than what the weights were trained for, and the model's
next-token accuracy collapses (the "FP8-degraded" warning).

This module reproduces TE's forward GEMM numerically, in plain PyTorch, so it
runs on MPS (and CPU). It is **bit-exact emulation**, not hardware FP8: on
M1–M4 there is no speed benefit (the point is *accuracy*, making the FP8
checkpoints usable). On an M5 — whose GPU Neural Accelerators support FP8
natively — the ``_quantize_e4m3`` body is the natural seam to swap for a real
FP8 matmul (``torch._scaled_mm`` once MPS exposes it, or an MLX kernel); the
per-tensor scales recovered here are exactly what such a path needs.

What TE actually does (verified against the evo2_1b_base checkpoint)
-------------------------------------------------------------------
Each ``*.projections._extra_state`` blob stores ``scale_fwd`` with three slots:
  slot 0 = activation (input) scale, slot 1 = weight scale, slot 2 = unused.
TE's forward for a single GEMM is:

    x_q = round_e4m3(x * act_scale)            # saturating, RNE
    w_q = round_e4m3(W * weight_scale)
    y   = (x_q @ w_q.T) / (act_scale * weight_scale) + bias

with ``scale = fp8_max / amax`` and ``fp8_max = 448`` for e4m3. The weight scale
recovered from the checkpoint matches ``448 / W.abs().max()`` to the digit, and
the activation scale is the delayed-scaling value from calibration (it cannot be
recomputed at inference time — it must come from the checkpoint).
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import torch
import torch.nn as nn

# e4m3fn: 4 exponent bits, 3 mantissa bits, bias 7, max normal 448, no inf.
FP8_E4M3_MAX = 448.0
_E4M3_MIN_EXP = -6  # smallest normal binade; subnormals share this exponent
_E4M3_MANTISSA_BITS = 3


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round a real-valued tensor to the e4m3fn grid, returned in the input dtype.

    Pure-tensor (MPS-safe) emulation. Bit-exact vs ``x.to(torch.float8_e4m3fn)``
    for all in-range values; values above 448 saturate to 448 (TE clamps before
    casting, so saturation — not NaN — is the behaviour we want here).
    """
    orig_dtype = x.dtype
    xf = x.float()
    sign = torch.sign(xf)
    ax = xf.abs()

    # Per-element binade exponent, floored into the representable range.
    e = torch.floor(torch.log2(ax.clamp_min(1e-30)))
    e = torch.clamp(e, min=_E4M3_MIN_EXP)
    # Mantissa step within the binade for 3 mantissa bits.
    step = torch.exp2(e - _E4M3_MANTISSA_BITS)
    q = torch.round(ax / step) * step
    q = torch.clamp(q, max=FP8_E4M3_MAX)

    # Flush values below half the smallest subnormal to zero.
    smallest_subnormal = 2.0 ** (_E4M3_MIN_EXP - _E4M3_MANTISSA_BITS)
    q = torch.where(ax < smallest_subnormal / 2, torch.zeros_like(q), q)

    return (sign * q).to(orig_dtype)


class Fp8EmulatedLinear(nn.Module):
    """Drop-in replacement for vortex's fallback ``TELinear`` that reproduces
    Transformer Engine's per-tensor e4m3 forward GEMM.

    Matches the fallback's ``(output, bias_or_None)`` return convention and its
    ``weight``/``bias`` parameter naming so the checkpoint loads unchanged.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        act_scale: float,
        weight_scale: float,
        skip_bias_add: bool = False,
    ):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.te_return_bias = skip_bias_add and (bias is not None)

        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

        # Per-tensor scales from the checkpoint's TE extra_state (slot 0/1 of
        # scale_fwd). Stored as buffers so they ride device moves with the module.
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))
        self.register_buffer("weight_scale", torch.tensor(float(weight_scale)))

    def forward(self, x):
        w = self.weight
        # Quantize activations and weights to e4m3 with their per-tensor scales,
        # matmul, then undo the scaling (TE's dequant).
        x_q = quantize_e4m3(x.to(w.dtype) * self.act_scale)
        w_q = quantize_e4m3(w * self.weight_scale)
        inv = 1.0 / (self.act_scale * self.weight_scale)
        out = torch.nn.functional.linear(x_q, w_q) * inv
        if self.bias is not None:
            out = out + self.bias
        out = out.to(w.dtype)
        if self.te_return_bias:
            return out, self.bias
        return out, None


def extract_projection_scales(checkpoint_path: str) -> Dict[str, Dict[str, float]]:
    """Read per-projection forward scales from a raw checkpoint's TE extra_state.

    Must be called on the on-disk checkpoint: vortex strips ``._extra_state``
    keys when Transformer Engine is absent, so the scales are gone by the time
    the model is built. Returns ``{module_path: {"act": float, "weight": float}}``
    keyed by the module path (e.g. ``"blocks.0.projections"``).
    """
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "module" in sd:
        sd = sd["module"]

    scales: Dict[str, Dict[str, float]] = {}
    for key, value in sd.items():
        if not key.endswith(".projections._extra_state"):
            continue
        module_path = key[: -len("._extra_state")]
        try:
            if hasattr(value, "read"):
                value.seek(0)
                meta = torch.load(value, map_location="cpu", weights_only=False)
            elif isinstance(value, (bytes, bytearray)):
                meta = torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
            else:
                continue
            scale_fwd = meta["scale_fwd"]  # (3,): [act, weight, unused]
            scales[module_path] = {
                "act": float(scale_fwd[0]),
                "weight": float(scale_fwd[1]),
            }
        except Exception:
            # A projection we can't decode is skipped; the caller leaves that
            # layer in its bf16 fallback rather than guessing a scale.
            continue
    return scales


def apply_fp8_emulation(model: nn.Module, checkpoint_path: str) -> int:
    """Swap every fallback ``TELinear`` projection in ``model`` for an
    ``Fp8EmulatedLinear`` carrying that layer's checkpoint scales.

    Returns the number of projections replaced. Layers whose scales could not be
    recovered are left untouched (bf16 fallback).
    """
    scales = extract_projection_scales(checkpoint_path)
    if not scales:
        return 0

    replaced = 0
    modules = dict(model.named_modules())
    for module_path, sc in scales.items():
        parent_path, _, attr = module_path.rpartition(".")
        parent = modules.get(parent_path)
        if parent is None:
            continue
        old = getattr(parent, attr, None)
        if old is None or not hasattr(old, "weight"):
            continue
        new = Fp8EmulatedLinear(
            weight=old.weight.data,
            bias=old.bias.data if getattr(old, "bias", None) is not None else None,
            act_scale=sc["act"],
            weight_scale=sc["weight"],
            skip_bias_add=getattr(old, "te_return_bias", False),
        ).to(old.weight.device)
        setattr(parent, attr, new)
        replaced += 1
    return replaced
