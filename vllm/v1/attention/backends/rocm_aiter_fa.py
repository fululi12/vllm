# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with AiterFlashAttention."""

import math
import os
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import ClassVar

import torch

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import get_cu_count
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_prefills_and_extends,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024
USING_SHUFFLE_LAYOUT = False #True

# When enabled, the Walsh-Hadamard Transform is applied to K/V before FP4
# quantization during cache writes, spreading outlier energy uniformly across
# all head dimensions and significantly improving FP4 quantization accuracy.
# The transform is self-inverse, so Q and the attention output are rotated/
# inverse-rotated at decode and extend time with no change to the final result.
# Set VLLM_FP4_HADAMARD=1 to enable.  Only applies when kv_cache_dtype=fp4
# with per-token quantization.
_FP4_HADAMARD = os.environ.get("VLLM_FP4_HADAMARD", "0") == "1"

# When the C++ cache-write kernel is compiled with FP4_USE_IN_KERNEL_WHT=1,
# set this to "1" so Python skips the cache-write-side WHT (the kernel does
# sign-flipping + WHT in float32 internally).  Python-side WHT is still used
# for Q rotation at decode time and K/V inverse-rotation in the extend path.
_FP4_IN_KERNEL_WHT = os.environ.get("VLLM_FP4_IN_KERNEL_WHT", "0") == "1"

# Per-channel K + per-token V quantization mode for FP4 KV cache.
# When enabled, K uses per-channel scales (one scale per head-dim element,
# shared across all tokens, computed from the first prefill batch) and V
# uses per-token scales (one scale per token, shared across head-dim).
# During decode, K channel scales are absorbed into Q before QK^T.
# Set VLLM_FP4_PER_CHANNEL_K=1 to enable.
_FP4_PER_CHANNEL_K = os.environ.get("VLLM_FP4_PER_CHANNEL_K", "0") == "1"

# MXFP4 (OCP MX) block-scale mode for FP4 KV cache.
# Uses E8M0 (power-of-2) scales stored as uint8 instead of FP32.
# Both K and V use per-block-32 quantization with E8M0 scales.
# Scale storage is 4× smaller than FP32 per-block scales.
# Set VLLM_FP4_MXFP4=1 to enable.
_FP4_MXFP4 = os.environ.get("VLLM_FP4_MXFP4", "0") == "1"

# NVFP4-style block-scale mode for FP4 KV cache (inspired by NVIDIA NVFP4).
# Uses FP8 E4M3 FNUZ scaling factors (1 byte each, same storage as E8M0)
# but with 3 mantissa bits of precision per scale, reducing quantization
# error by ~5% compared to MXFP4 E8M0 on typical LLM benchmarks.
# Both K and V use per-block-32 quantization with FP8 E4M3 scales.
# Set VLLM_FP4_NVFP4=1 to enable.
_FP4_NVFP4 = os.environ.get("VLLM_FP4_NVFP4", "0") == "1"
_NVFP4_OP_CHECKED = False

# Signal offset for PA kernel to distinguish NVFP4 E4M3 from MXFP4 E8M0.
# fp4_num_k/v_blocks <= -_NVFP4_SIGNAL_OFFSET → NVFP4 E4M3 mode
_NVFP4_SIGNAL_OFFSET = 1000

# AMXFP4 (Asymmetric Microscaling FP4) block-scale mode for FP4 KV cache.
# Based on arXiv:2411.09909, the Block Maximum (BM) element uses E0M3 encoding
# (3 mantissa bits) for higher outlier precision, while all other elements use
# standard E2M1.  A 1-byte BM index per block adds only 0.25 bits/element
# overhead, recovering ~90% of the MXFP4→BF16 accuracy gap.
# Uses E8M0 shared scales (same as MXFP4) plus BM index metadata.
# Set VLLM_FP4_AMXFP4=1 to enable.
_FP4_AMXFP4 = os.environ.get("VLLM_FP4_AMXFP4", "0") == "1"
_AMXFP4_OP_CHECKED = False

# Signal offset for PA kernel to distinguish AMXFP4 from NVFP4/MXFP4.
# fp4_num_k/v_blocks <= -_AMXFP4_SIGNAL_OFFSET → AMXFP4 mode
_AMXFP4_SIGNAL_OFFSET = 2000


def _ensure_amxfp4_op_available():
    """Lazy check: disable AMXFP4 mode if the C++ op wasn't compiled."""
    global _FP4_AMXFP4, _AMXFP4_OP_CHECKED
    if _AMXFP4_OP_CHECKED:
        return
    _AMXFP4_OP_CHECKED = True
    if not _FP4_AMXFP4:
        return
    if not hasattr(torch.ops, "_C_cache_ops") or not hasattr(
        torch.ops._C_cache_ops, "reshape_and_cache_flash_fp4_amxfp4"
    ):
        logger.warning(
            "VLLM_FP4_AMXFP4=1 is set but reshape_and_cache_flash_fp4_amxfp4 "
            "is not available in the compiled C++ extension. "
            "Falling back to NVFP4/MXFP4/per-token mode. Please rebuild vLLM "
            "with the updated C++ files "
            "(cache.h, cache_kernels.cu, torch_bindings.cpp)."
        )
        _FP4_AMXFP4 = False


def _ensure_nvfp4_op_available():
    """Lazy check: disable NVFP4 mode if the C++ op wasn't compiled."""
    global _FP4_NVFP4, _NVFP4_OP_CHECKED
    if _NVFP4_OP_CHECKED:
        return
    _NVFP4_OP_CHECKED = True
    if not _FP4_NVFP4:
        return
    if not hasattr(torch.ops, "_C_cache_ops") or not hasattr(
        torch.ops._C_cache_ops, "reshape_and_cache_flash_fp4_nvfp4"
    ):
        logger.warning(
            "VLLM_FP4_NVFP4=1 is set but reshape_and_cache_flash_fp4_nvfp4 "
            "is not available in the compiled C++ extension. "
            "Falling back to MXFP4 mode if VLLM_FP4_MXFP4=1, or default "
            "per-token mode. Please rebuild vLLM with the updated C++ files "
            "(cache.h, cache_kernels.cu, torch_bindings.cpp)."
        )
        _FP4_NVFP4 = False

# Per-channel K scale accuracy tuning.
# Clip sigma: use min(absmax, rms * sigma) per channel to reduce outlier
# sensitivity.  0 = disabled (raw absmax).  Typical range: 4–6.
_FP4_PCK_CLIP_SIGMA = float(
    os.environ.get("VLLM_FP4_PCK_CLIP_SIGMA", "0")
)
# MSE refinement iterations for K channel scales.  Default: 2.
_FP4_PCK_MSE_ITERS = int(
    os.environ.get("VLLM_FP4_PCK_MSE_ITERS", "2")
)
# Safety margin multiplied into K channel scales after computation.
# Prevents future decode tokens from being clipped.  Default: 1.0.
_FP4_PCK_SAFETY_MARGIN = float(
    os.environ.get("VLLM_FP4_PCK_SAFETY_MARGIN", "1.0")
)


def _fp4_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Simulate FP4 E2M1 quantize-then-dequant in float32.

    FP4 E2M1 representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    Each value is mapped to the nearest representable magnitude,
    preserving sign.
    """
    sign = x.sign()
    ax = x.abs()
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device, dtype=torch.float32,
    )
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device, dtype=torch.float32,
    )
    idx = torch.bucketize(ax, boundaries)
    return sign * values[idx]


def _compute_k_channel_scales(
    k_data: torch.Tensor,
    clip_sigma: float = 0.0,
    mse_iters: int = 2,
    safety_margin: float = 1.0,
) -> torch.Tensor:
    """Compute per-channel K scales from prefill data with optimizations.

    Args:
        k_data: float32 tensor [num_tokens, num_kv_heads, head_size]
        clip_sigma: outlier clipping sigma (0 = disabled)
        mse_iters: MSE-optimal refinement iterations
        safety_margin: multiplicative safety margin (>= 1.0)

    Returns:
        k_channel_scales: [num_kv_heads, head_size] float32
    """
    FP4_MAX = 6.0

    k_ch_max = k_data.abs().amax(dim=0)
    effective_max = k_ch_max

    if clip_sigma > 0:
        k_ch_rms = (k_data ** 2).mean(dim=0).sqrt()
        clip_val = k_ch_rms * clip_sigma
        effective_max = torch.minimum(k_ch_max, clip_val)

    k_ch_scales = (effective_max / FP4_MAX).clamp(min=1e-12)

    if mse_iters > 0:
        for _ in range(mse_iters):
            scaled = k_data / k_ch_scales.unsqueeze(0)
            q_vals = _fp4_round_trip(scaled)
            xq = (k_data * q_vals).sum(dim=0)
            qq = (q_vals * q_vals).sum(dim=0)
            k_ch_scales = torch.where(
                qq > 0,
                (xq / qq).clamp(min=1e-12),
                k_ch_scales,
            )

    if safety_margin > 1.0:
        k_ch_scales = k_ch_scales * safety_margin

    return k_ch_scales


def _fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension.

    Computed in float32 for numerical stability.  The normalized WHT is
    its own inverse (involutory): applying it twice returns the original.
    The last dimension must be a power of 2.
    """
    shape = x.shape
    n = shape[-1]
    x = x.contiguous().reshape(-1, n).float()
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :] + x[:, :, 1, :]
        b = x[:, :, 0, :] - x[:, :, 1, :]
        x = torch.stack([a, b], dim=2).reshape(-1, n)
        h *= 2
    return (x * (1.0 / math.sqrt(n))).reshape(shape)


@lru_cache(maxsize=8)
def _get_fp4_hadamard_signs(n: int, device: torch.device) -> torch.Tensor:
    """Generate deterministic random ±1 sign vector for QuIP#-style
    randomized Hadamard rotation.

    Uses Knuth's multiplicative hash so that the same signs are
    reproducible in the C++ in-kernel WHT path.
    """
    indices = torch.arange(n, dtype=torch.int64)
    hashes = (indices * 2654435761) & 0xFFFFFFFF
    signs = torch.where(
        (hashes & 0x80000000) != 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    return signs.to(device)


def _hadamard_rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Forward rotation: T(x) = WHT(D · x)."""
    return _fast_walsh_hadamard_transform(x * signs)


def _hadamard_inv_rotate(
    x: torch.Tensor, signs: torch.Tensor,
) -> torch.Tensor:
    """Inverse rotation: T⁻¹(y) = D · WHT(y).

    Since both D (diag ±1) and normalized WHT are self-inverse,
    T⁻¹ = D⁻¹ ∘ WHT⁻¹ = D ∘ WHT.
    """
    return _fast_walsh_hadamard_transform(x) * signs


@lru_cache(maxsize=1)
def get_static_kvscale(
        k_scale_float: float,
        v_scale_float: float,
        num_kv_heads: int,
        num_blocks: int,
        block_size: int,
        device,
    ):
    k_scale = torch.empty((num_kv_heads, num_blocks * block_size),
                        dtype=torch.float32,
                        device=device)
    v_scale = torch.empty((num_kv_heads, num_blocks * block_size),
                        dtype=torch.float32,
                        device=device)
    k_scale.fill_(k_scale_float)
    v_scale.fill_(v_scale_float)
    return k_scale, v_scale


_FP4_MAX_TOKENS_PER_HEAD_LEGACY = 32 * 1024 * 1024  # kept for reference


@lru_cache(maxsize=1)
def get_fp4_per_token_kscale(
        k_scale_float: float,
        num_kv_heads: int,
        total_tokens: int,
        device,
    ):
    """Allocate per-token K scale buffer matching aiter FP4 paged attention
    kernel layout.  The kernel indexes k_scale as:
      k_scale_ptr[kv_head_idx * total_tokens + physical_token_idx]
    where ``total_tokens`` is the per-head stride passed to the kernel.
    """
    needed = num_kv_heads * total_tokens
    return torch.full(
        (needed,), k_scale_float, dtype=torch.float32, device=device
    )


FP4_QUANT_BLOCK_SIZE = 32


def _get_or_create_fp4_pertoken_scales(
    layer: torch.nn.Module,
    num_kv_heads: int,
    total_tokens: int,
    device,
    head_size: int = 128,
):
    """Allocate (once) per-block K and V dequant-scale buffers.

    Both K and V use per-block-32 quantization scales.  The layout is::

        [num_kv_heads * num_blocks, total_tokens]

    where ``num_blocks = head_size // FP4_QUANT_BLOCK_SIZE``.  The PA
    kernel indexes as::

        scale_ptr[kv_head_idx * (num_blocks * total_tokens)
                  + block_idx * total_tokens + physical_token_idx]

    Returns (k_scales_flat, v_scales_flat, k_scales_2d, v_scales_2d).
    """
    if hasattr(layer, "_fp4_k_dequant_scales_flat"):
        return (
            layer._fp4_k_dequant_scales_flat,
            layer._fp4_v_dequant_scales_flat,
            layer._fp4_k_dequant_scales_2d,
            layer._fp4_v_dequant_scales_2d,
        )

    num_k_blocks = max(1, head_size // FP4_QUANT_BLOCK_SIZE)
    num_v_blocks = max(1, head_size // FP4_QUANT_BLOCK_SIZE)

    k_flat_size = num_kv_heads * num_k_blocks * total_tokens
    k_flat = torch.zeros(k_flat_size, dtype=torch.float32, device=device)
    k_2d = k_flat.view(num_kv_heads * num_k_blocks, total_tokens)

    v_flat_size = num_kv_heads * num_v_blocks * total_tokens
    v_flat = torch.zeros(v_flat_size, dtype=torch.float32, device=device)
    v_2d = v_flat.view(num_kv_heads * num_v_blocks, total_tokens)

    layer._fp4_k_dequant_scales_flat = k_flat
    layer._fp4_v_dequant_scales_flat = v_flat
    layer._fp4_k_dequant_scales_2d = k_2d
    layer._fp4_v_dequant_scales_2d = v_2d
    layer._fp4_num_k_blocks = num_k_blocks
    layer._fp4_num_v_blocks = num_v_blocks
    layer._fp4_k_scale_stride_h = num_k_blocks * total_tokens
    layer._fp4_v_scale_stride_h = num_v_blocks * total_tokens

    return k_flat, v_flat, k_2d, v_2d


def _get_or_create_fp4_per_channel_k_scales(
    layer: torch.nn.Module,
    num_kv_heads: int,
    total_tokens: int,
    device,
    head_size: int = 128,
):
    """Allocate (once) per-channel K and per-token V scale buffers.

    K channel scales: ``[num_kv_heads, head_size]`` — static, computed from
    the first prefill batch and frozen.

    V per-token scales: ``[num_kv_heads, total_tokens]`` — dynamic, written
    at cache-write time.

    Returns (k_channel_scales, v_scales_flat, v_scales_2d, k_scales_ready).
    ``k_scales_ready`` is False when K channel scales have not yet been
    computed (first call) and True after initialization.
    """
    if hasattr(layer, "_fp4_k_channel_scales"):
        return (
            layer._fp4_k_channel_scales,
            layer._fp4_v_pertoken_scales_flat,
            layer._fp4_v_pertoken_scales_2d,
            True,
        )

    k_ch = torch.ones(
        num_kv_heads, head_size, dtype=torch.float32, device=device
    )

    v_flat_size = num_kv_heads * total_tokens
    v_flat = torch.zeros(v_flat_size, dtype=torch.float32, device=device)
    v_2d = v_flat.view(num_kv_heads, total_tokens)

    layer._fp4_k_channel_scales = k_ch
    layer._fp4_v_pertoken_scales_flat = v_flat
    layer._fp4_v_pertoken_scales_2d = v_2d
    layer._fp4_v_scale_stride_h = total_tokens
    layer._fp4_num_v_blocks = 1
    layer._fp4_num_k_blocks = 0
    layer._fp4_k_channel_scales_ready = False

    return k_ch, v_flat, v_2d, False


def _get_or_create_fp4_mxfp4_scales(
    layer: torch.nn.Module,
    num_kv_heads: int,
    total_tokens: int,
    device,
    head_size: int = 128,
):
    """Allocate (once) MXFP4 E8M0 per-block-32 scale buffers as uint8.

    Both K and V use per-block-32 quantization with E8M0 scales.
    Layout: ``[num_kv_heads * num_blocks, total_tokens]`` as uint8.

    Returns (k_e8m0_flat, v_e8m0_flat, k_e8m0_2d, v_e8m0_2d).
    """
    if hasattr(layer, "_fp4_mxfp4_k_scales_flat"):
        return (
            layer._fp4_mxfp4_k_scales_flat,
            layer._fp4_mxfp4_v_scales_flat,
            layer._fp4_mxfp4_k_scales_2d,
            layer._fp4_mxfp4_v_scales_2d,
        )

    num_blocks = max(1, head_size // FP4_QUANT_BLOCK_SIZE)

    k_flat_size = num_kv_heads * num_blocks * total_tokens
    k_flat = torch.zeros(k_flat_size, dtype=torch.uint8, device=device)
    k_2d = k_flat.view(num_kv_heads * num_blocks, total_tokens)

    v_flat_size = num_kv_heads * num_blocks * total_tokens
    v_flat = torch.zeros(v_flat_size, dtype=torch.uint8, device=device)
    v_2d = v_flat.view(num_kv_heads * num_blocks, total_tokens)

    layer._fp4_mxfp4_k_scales_flat = k_flat
    layer._fp4_mxfp4_v_scales_flat = v_flat
    layer._fp4_mxfp4_k_scales_2d = k_2d
    layer._fp4_mxfp4_v_scales_2d = v_2d
    layer._fp4_num_k_blocks = num_blocks
    layer._fp4_num_v_blocks = num_blocks
    layer._fp4_k_scale_stride_h = num_blocks * total_tokens
    layer._fp4_v_scale_stride_h = num_blocks * total_tokens
    layer._fp4_mxfp4_active = True

    return k_flat, v_flat, k_2d, v_2d


def _fp8_e4m3_to_float(raw: torch.Tensor) -> torch.Tensor:
    """Convert uint8 FP8 E4M3 FNUZ bytes to float32 scale values.

    E4M3 FNUZ: bias=8, normal = 2^(E-8) * (1 + M/8),
    subnormal = 2^(-7) * (M/8).  Only positive values for scales.
    """
    exp_bits = ((raw.long() >> 3) & 0xF)
    mantissa = (raw.long() & 0x7)
    is_subnormal = (exp_bits == 0)
    normal = torch.pow(
        2.0, (exp_bits - 8).float()
    ) * (1.0 + mantissa.float() / 8.0)
    subnormal = (2.0 ** -7.0) * (mantissa.float() / 8.0)
    result = torch.where(is_subnormal, subnormal, normal)
    result = result * (raw != 0).float()
    return result


def _get_or_create_fp4_nvfp4_scales(
    layer: torch.nn.Module,
    num_kv_heads: int,
    total_tokens: int,
    device,
    head_size: int = 128,
):
    """Allocate (once) NVFP4 FP8 E4M3 per-block-32 scale buffers as uint8.

    Identical layout to MXFP4 but scales are encoded as FP8 E4M3 FNUZ
    instead of E8M0, providing 3 mantissa bits of precision per scale.

    Returns (k_fp8_flat, v_fp8_flat, k_fp8_2d, v_fp8_2d).
    """
    if hasattr(layer, "_fp4_nvfp4_k_scales_flat"):
        return (
            layer._fp4_nvfp4_k_scales_flat,
            layer._fp4_nvfp4_v_scales_flat,
            layer._fp4_nvfp4_k_scales_2d,
            layer._fp4_nvfp4_v_scales_2d,
        )

    num_blocks = max(1, head_size // FP4_QUANT_BLOCK_SIZE)

    k_flat_size = num_kv_heads * num_blocks * total_tokens
    k_flat = torch.zeros(k_flat_size, dtype=torch.uint8, device=device)
    k_2d = k_flat.view(num_kv_heads * num_blocks, total_tokens)

    v_flat_size = num_kv_heads * num_blocks * total_tokens
    v_flat = torch.zeros(v_flat_size, dtype=torch.uint8, device=device)
    v_2d = v_flat.view(num_kv_heads * num_blocks, total_tokens)

    layer._fp4_nvfp4_k_scales_flat = k_flat
    layer._fp4_nvfp4_v_scales_flat = v_flat
    layer._fp4_nvfp4_k_scales_2d = k_2d
    layer._fp4_nvfp4_v_scales_2d = v_2d
    layer._fp4_num_k_blocks = num_blocks
    layer._fp4_num_v_blocks = num_blocks
    layer._fp4_k_scale_stride_h = num_blocks * total_tokens
    layer._fp4_v_scale_stride_h = num_blocks * total_tokens
    layer._fp4_nvfp4_active = True

    return k_flat, v_flat, k_2d, v_2d


def _get_or_create_fp4_amxfp4_scales(
    layer: torch.nn.Module,
    num_kv_heads: int,
    total_tokens: int,
    device,
    head_size: int = 128,
):
    """Allocate (once) AMXFP4 E8M0 scale + BM index buffers as uint8.

    For the PA kernel, scales and BM indices are packed into a single buffer
    per K/V with doubled row count:
      - Rows [0, num_blocks) : E8M0 shared scales
      - Rows [num_blocks, 2*num_blocks) : BM indices (0-31)

    Returns (k_scales_2d, v_scales_2d, k_bm_2d, v_bm_2d, k_packed_2d, v_packed_2d).
    The *_packed_2d tensors are for the PA kernel (concatenated scales + BM idx).
    """
    if hasattr(layer, "_fp4_amxfp4_k_scales_2d"):
        return (
            layer._fp4_amxfp4_k_scales_2d,
            layer._fp4_amxfp4_v_scales_2d,
            layer._fp4_amxfp4_k_bm_2d,
            layer._fp4_amxfp4_v_bm_2d,
            layer._fp4_amxfp4_k_packed_2d,
            layer._fp4_amxfp4_v_packed_2d,
        )

    num_blocks = max(1, head_size // FP4_QUANT_BLOCK_SIZE)

    k_scale_flat = torch.zeros(
        num_kv_heads * num_blocks * total_tokens,
        dtype=torch.uint8, device=device)
    k_scales_2d = k_scale_flat.view(num_kv_heads * num_blocks, total_tokens)

    k_bm_flat = torch.zeros(
        num_kv_heads * num_blocks * total_tokens,
        dtype=torch.uint8, device=device)
    k_bm_2d = k_bm_flat.view(num_kv_heads * num_blocks, total_tokens)

    v_scale_flat = torch.zeros(
        num_kv_heads * num_blocks * total_tokens,
        dtype=torch.uint8, device=device)
    v_scales_2d = v_scale_flat.view(num_kv_heads * num_blocks, total_tokens)

    v_bm_flat = torch.zeros(
        num_kv_heads * num_blocks * total_tokens,
        dtype=torch.uint8, device=device)
    v_bm_2d = v_bm_flat.view(num_kv_heads * num_blocks, total_tokens)

    k_packed = torch.zeros(
        num_kv_heads * num_blocks * 2, total_tokens,
        dtype=torch.uint8, device=device)
    v_packed = torch.zeros(
        num_kv_heads * num_blocks * 2, total_tokens,
        dtype=torch.uint8, device=device)

    layer._fp4_amxfp4_k_scales_2d = k_scales_2d
    layer._fp4_amxfp4_v_scales_2d = v_scales_2d
    layer._fp4_amxfp4_k_bm_2d = k_bm_2d
    layer._fp4_amxfp4_v_bm_2d = v_bm_2d
    layer._fp4_amxfp4_k_packed_2d = k_packed
    layer._fp4_amxfp4_v_packed_2d = v_packed
    layer._fp4_num_k_blocks = num_blocks
    layer._fp4_num_v_blocks = num_blocks
    layer._fp4_k_scale_stride_h = num_blocks * 2 * total_tokens
    layer._fp4_v_scale_stride_h = num_blocks * 2 * total_tokens
    layer._fp4_amxfp4_active = True

    return k_scales_2d, v_scales_2d, k_bm_2d, v_bm_2d, k_packed, v_packed


if current_platform.is_rocm():
    import aiter

    from vllm.triton_utils import tl, triton

    def block_size(x, head_dim):
        return min(65536 // x.element_size(), triton.next_power_of_2(head_dim))

    def num_programs(total_tokens):
        return min(total_tokens, get_cu_count())

    @triton.jit
    def cp_mha_gather_cache_kernel(
        key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
        value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
        key_ptr,  # [num_tokens, num_heads, head_size]
        value_ptr,  # [num_tokens, num_heads, head_size]
        block_table_ptr,  # [num_batches, max_block_num]
        cu_seqlens_kv_ptr,  # [num_batches + 1]
        token_to_batch_ptr,  # [max_cum_tokens]
        seq_start_ptr,  # [num_batches]
        k_scale_ptr,
        v_scale_ptr,
        num_heads,
        head_size,
        x,
        max_block_num,
        DEQUANT: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        CACHE_FORMAT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        if DEQUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)

        key_ptr_offset = key_ptr + token_id * head_size * num_heads
        value_ptr_offset = value_ptr + token_id * head_size * num_heads
        batch_idx = tl.load(token_to_batch_ptr + token_id)
        batch_start = tl.load(seq_start_ptr + batch_idx)
        token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
        batch_offset = token_id - token_start + batch_start
        block_offset = batch_offset // PAGE_SIZE
        block_id = tl.load(
            block_table_ptr + max_block_num * batch_idx + block_offset
        ).to(tl.int64)
        slot_id = batch_offset % PAGE_SIZE

        if CACHE_FORMAT == "NHD":
            # for kv cache layout as
            # K: [num_blocks, page_size, num_head, head_dim]
            # V: [num_blocks, page_size, num_head, head_dim]
            key_cache_ptr_offset = (
                key_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + slot_id * num_heads * head_size
            )
            value_cache_ptr_offset = (
                value_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + slot_id * num_heads * head_size
            )

            for i in tl.range(0, head_size * num_heads, BLOCK_SIZE):
                mask = (col_offsets + i) < head_size * num_heads
                k_reg = tl.load(key_cache_ptr_offset + col_offsets + i, mask=mask)
                v_reg = tl.load(value_cache_ptr_offset + col_offsets + i, mask=mask)
                if DEQUANT:
                    k_dtype = k_reg.dtype
                    v_dtype = v_reg.dtype
                    k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
                    v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
                tl.store(key_ptr_offset + col_offsets + i, k_reg, mask=mask)
                tl.store(value_ptr_offset + col_offsets + i, v_reg, mask=mask)
        elif CACHE_FORMAT == "SHUFFLE":
            # for kv cache layout as
            # K: [num_blocks, num_head, head_dim // x, page_size, x]
            # V: [num_blocks, num_head, page_size // x, head_dim, x]
            key_cache_ptr_offset = (
                key_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + slot_id * x
            )
            value_cache_ptr_offset = (
                value_cache_ptr
                + block_id * num_heads * head_size * PAGE_SIZE
                + (slot_id // x) * head_size * x
                + slot_id % x
            )

            for i in tl.range(0, head_size * num_heads, BLOCK_SIZE):
                offset = col_offsets + i
                mask = offset < head_size * num_heads
                k_reg_offset = (
                    (offset // head_size) * head_size * PAGE_SIZE
                    + (offset % head_size) // x * PAGE_SIZE * x
                    + (offset % head_size) % x
                )
                v_reg_offset = (offset // head_size) * head_size * PAGE_SIZE + (
                    offset % head_size
                ) * x
                k_reg = tl.load(key_cache_ptr_offset + k_reg_offset)
                v_reg = tl.load(value_cache_ptr_offset + v_reg_offset)
                if DEQUANT:
                    k_dtype = k_reg.dtype
                    v_dtype = v_reg.dtype
                    k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
                    v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
                tl.store(key_ptr_offset + col_offsets + i, k_reg, mask=mask)
                tl.store(value_ptr_offset + col_offsets + i, v_reg, mask=mask)

    def _fp4_gather_and_dequant_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        token_to_batch: torch.Tensor,
        seq_starts: torch.Tensor,
        total_tokens: int,
        k_dequant_scales_2d: torch.Tensor | None = None,
        v_dequant_scales_2d: torch.Tensor | None = None,
    ):
        """Gather FP4-packed KV entries from paged cache and dequantize
        to bf16/fp16.

        Cache layout: ``[num_blocks, page_size, num_heads, head_dim // 2]``
        (each ``uint8`` byte packs two FP4 E2M1 values).
        Output layout: ``[total_tokens, num_heads, head_dim]`` in model dtype.
        """
        page_size = key_cache.shape[1]
        num_heads = key_cache.shape[2]
        packed_dim = key_cache.shape[3]
        head_dim = packed_dim * 2
        max_block_num = block_tables.size(1)

        kc_u8 = key_cache.view(torch.uint8)
        vc_u8 = value_cache.view(torch.uint8)

        token_ids = torch.arange(
            total_tokens, device=key_cache.device, dtype=torch.int64
        )
        batch_idx = token_to_batch[:total_tokens].long()
        batch_start = seq_starts[batch_idx]
        token_start = cu_seqlens_kv[batch_idx]
        batch_offset = (token_ids - token_start + batch_start).long()
        block_offset = batch_offset // page_size
        slot_id = batch_offset % page_size
        block_offset = block_offset.clamp(0, max_block_num - 1)
        block_id = block_tables[batch_idx, block_offset].long()

        packed_k = kc_u8[block_id, slot_id]
        packed_v = vc_u8[block_id, slot_id]

        # FP4 E2M1 dequantization: nibble → float via lookup table.
        # Use float32 for all intermediate math to avoid bf16/fp16
        # rounding during scale multiplication.
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=key.device,
        )

        k_lo = (packed_k & 0x0F).long()
        k_hi = ((packed_k >> 4) & 0x0F).long()
        k_vals = torch.stack([lut[k_lo], lut[k_hi]], dim=-1)
        k_out = k_vals.reshape(total_tokens, num_heads, head_dim)

        v_lo = (packed_v & 0x0F).long()
        v_hi = ((packed_v >> 4) & 0x0F).long()
        v_vals = torch.stack([lut[v_lo], lut[v_hi]], dim=-1)
        v_out = v_vals.reshape(total_tokens, num_heads, head_dim)

        if k_dequant_scales_2d is not None and v_dequant_scales_2d is not None:
            physical_slot_idx = block_id * page_size + slot_id

            is_per_channel_k = (
                k_dequant_scales_2d.shape[0] == num_heads
                and k_dequant_scales_2d.shape[1] == head_dim
            )
            is_uint8_k = k_dequant_scales_2d.dtype == torch.uint8
            is_uint8_v = v_dequant_scales_2d.dtype == torch.uint8
            is_nvfp4_mode = _FP4_NVFP4
            is_amxfp4_mode = _FP4_AMXFP4

            if is_per_channel_k:
                k_out = k_out * k_dequant_scales_2d.unsqueeze(0)
            elif is_uint8_k:
                if is_amxfp4_mode:
                    num_k_blocks = k_dequant_scales_2d.shape[0] // \
                        (num_heads * 2)
                    total_krows = num_heads * num_k_blocks
                    k_e8m0_raw = k_dequant_scales_2d[
                        :total_krows, physical_slot_idx]
                    k_bm_raw = k_dequant_scales_2d[
                        total_krows:, physical_slot_idx]
                    k_sc_float = torch.pow(
                        2.0, k_e8m0_raw.float() - 127.0)
                    k_sc_float = k_sc_float.reshape(
                        num_heads, num_k_blocks, total_tokens
                    ).permute(2, 0, 1)
                    k_scales = k_sc_float.repeat_interleave(
                        FP4_QUANT_BLOCK_SIZE, dim=-1)
                    k_out = k_out * k_scales

                    bm_lut = torch.tensor(
                        [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                         -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0,
                         -7.5],
                        dtype=torch.float32, device=key.device)
                    k_bm_idx = k_bm_raw.reshape(
                        num_heads, num_k_blocks, total_tokens
                    ).permute(2, 0, 1).long()
                    for blk in range(num_k_blocks):
                        blk_start = blk * FP4_QUANT_BLOCK_SIZE
                        bm_pos = k_bm_idx[:, :, blk]
                        global_pos = blk_start + bm_pos
                        packed_byte_idx = global_pos // 2
                        nib_in_byte = global_pos % 2
                        raw_byte = packed_k[
                            torch.arange(total_tokens,
                                         device=key.device
                                         ).unsqueeze(1),
                            torch.arange(num_heads,
                                         device=key.device
                                         ).unsqueeze(0),
                            packed_byte_idx]
                        nibble = torch.where(
                            nib_in_byte == 0,
                            raw_byte & 0x0F,
                            (raw_byte >> 4) & 0x0F).long()
                        bm_correct = bm_lut[nibble]
                        scale_at_bm = k_sc_float[:, :, blk]
                        corrected = bm_correct * scale_at_bm
                        k_out[
                            torch.arange(total_tokens,
                                         device=key.device
                                         ).unsqueeze(1),
                            torch.arange(num_heads,
                                         device=key.device
                                         ).unsqueeze(0),
                            global_pos] = corrected
                else:
                    num_k_blocks = k_dequant_scales_2d.shape[0] // \
                        num_heads
                    k_sc_raw = k_dequant_scales_2d[
                        :, physical_slot_idx]
                    if is_nvfp4_mode:
                        k_sc_float = _fp8_e4m3_to_float(k_sc_raw)
                    else:
                        k_sc_float = torch.pow(
                            2.0, k_sc_raw.float() - 127.0)
                    k_sc_float = k_sc_float.reshape(
                        num_heads, num_k_blocks, total_tokens
                    ).permute(2, 0, 1)
                    k_scales = k_sc_float.repeat_interleave(
                        FP4_QUANT_BLOCK_SIZE, dim=-1)
                    k_out = k_out * k_scales
            else:
                num_k_blocks = k_dequant_scales_2d.shape[0] // num_heads
                if num_k_blocks > 1:
                    k_sc = k_dequant_scales_2d[:, physical_slot_idx]
                    k_sc = k_sc.reshape(
                        num_heads, num_k_blocks, total_tokens
                    ).permute(2, 0, 1)
                    k_scales = k_sc.repeat_interleave(
                        FP4_QUANT_BLOCK_SIZE, dim=-1
                    )
                    k_out = k_out * k_scales
                else:
                    k_scales = k_dequant_scales_2d[:, physical_slot_idx]
                    k_scales = k_scales.T.unsqueeze(-1)
                    k_out = k_out * k_scales

            if is_uint8_v:
                if is_amxfp4_mode:
                    num_v_blocks = v_dequant_scales_2d.shape[0] // \
                        (num_heads * 2)
                    total_vrows = num_heads * num_v_blocks
                    v_e8m0_raw = v_dequant_scales_2d[
                        :total_vrows, physical_slot_idx]
                    v_bm_raw = v_dequant_scales_2d[
                        total_vrows:, physical_slot_idx]
                    v_sc_float = torch.pow(
                        2.0, v_e8m0_raw.float() - 127.0)
                    v_sc_float = v_sc_float.reshape(
                        num_heads, num_v_blocks, total_tokens
                    ).permute(2, 0, 1)
                    v_scales = v_sc_float.repeat_interleave(
                        FP4_QUANT_BLOCK_SIZE, dim=-1)
                    v_out = v_out * v_scales

                    bm_lut_v = torch.tensor(
                        [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                         -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0,
                         -7.5],
                        dtype=torch.float32, device=key.device)
                    v_bm_idx = v_bm_raw.reshape(
                        num_heads, num_v_blocks, total_tokens
                    ).permute(2, 0, 1).long()
                    for blk in range(num_v_blocks):
                        blk_start = blk * FP4_QUANT_BLOCK_SIZE
                        bm_pos = v_bm_idx[:, :, blk]
                        global_pos = blk_start + bm_pos
                        packed_byte_idx = global_pos // 2
                        nib_in_byte = global_pos % 2
                        raw_byte = packed_v[
                            torch.arange(total_tokens,
                                         device=key.device
                                         ).unsqueeze(1),
                            torch.arange(num_heads,
                                         device=key.device
                                         ).unsqueeze(0),
                            packed_byte_idx]
                        nibble = torch.where(
                            nib_in_byte == 0,
                            raw_byte & 0x0F,
                            (raw_byte >> 4) & 0x0F).long()
                        bm_correct = bm_lut_v[nibble]
                        scale_at_bm = v_sc_float[:, :, blk]
                        corrected = bm_correct * scale_at_bm
                        v_out[
                            torch.arange(total_tokens,
                                         device=key.device
                                         ).unsqueeze(1),
                            torch.arange(num_heads,
                                         device=key.device
                                         ).unsqueeze(0),
                            global_pos] = corrected
                else:
                    num_v_blocks = v_dequant_scales_2d.shape[0] // \
                        num_heads
                    v_sc_raw = v_dequant_scales_2d[
                        :, physical_slot_idx]
                    if is_nvfp4_mode:
                        v_sc_float = _fp8_e4m3_to_float(v_sc_raw)
                    else:
                        v_sc_float = torch.pow(
                            2.0, v_sc_raw.float() - 127.0)
                    v_sc_float = v_sc_float.reshape(
                        num_heads, num_v_blocks, total_tokens
                    ).permute(2, 0, 1)
                    v_scales = v_sc_float.repeat_interleave(
                        FP4_QUANT_BLOCK_SIZE, dim=-1)
                    v_out = v_out * v_scales
            else:
                num_v_blocks = v_dequant_scales_2d.shape[0] // num_heads
                if num_v_blocks > 1:
                    v_sc = v_dequant_scales_2d[:, physical_slot_idx]
                    v_sc = v_sc.reshape(
                        num_heads, num_v_blocks, total_tokens
                    ).permute(2, 0, 1)
                    v_scales = v_sc.repeat_interleave(
                        FP4_QUANT_BLOCK_SIZE, dim=-1
                    )
                    v_out = v_out * v_scales
                else:
                    v_scales = v_dequant_scales_2d[:, physical_slot_idx]
                    v_scales = v_scales.T.unsqueeze(-1)
                    v_out = v_out * v_scales

        if _FP4_HADAMARD and k_dequant_scales_2d is not None:
            signs = _get_fp4_hadamard_signs(
                k_out.shape[-1], k_out.device)
            k_out = _hadamard_inv_rotate(k_out, signs)
            v_out = _hadamard_inv_rotate(v_out, signs)

        key[:total_tokens] = k_out.to(key.dtype)
        value[:total_tokens] = v_out.to(value.dtype)

    def cp_mha_gather_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        token_to_batch: torch.Tensor,
        seq_starts: torch.Tensor,
        dequant: bool,
        kv_cache_layout: str,
        total_tokens: int,
        k_dequant_scales_2d: torch.Tensor | None = None,
        v_dequant_scales_2d: torch.Tensor | None = None,
    ):
        head_dim = key.shape[2]

        if key_cache.shape[3] != head_dim:
            _fp4_gather_and_dequant_cache(
                key_cache, value_cache, key, value,
                block_tables, cu_seqlens_kv, token_to_batch, seq_starts,
                total_tokens,
                k_dequant_scales_2d, v_dequant_scales_2d,
            )
            return

        assert kv_cache_layout in ["NHD", "SHUFFLE"], (
            "kv_cache_layout only support v0, NHD, HND, SHUFFLE"
        )
        x = 16 // key_cache.element_size()
        page_size = key_cache.shape[1]
        num_heads = key_cache.shape[2]

        grid = lambda meta: (total_tokens,)
        cp_mha_gather_cache_kernel[grid](
            key_cache,
            value_cache,
            key,
            value,
            block_tables,
            cu_seqlens_kv,
            token_to_batch,
            seq_starts,
            k_scales,
            v_scales,
            num_heads,
            head_dim,
            x,
            block_tables.size(1),
            DEQUANT=dequant,
            PAGE_SIZE=page_size,
            CACHE_FORMAT=kv_cache_layout,
            BLOCK_SIZE=head_dim,
        )

    @triton.jit
    def reshape_and_cache_shuffle_kernel(
        key_ptr,  # [num_tokens, num_kv_heads, head_size]
        value_ptr,  # [num_tokens, num_kv_heads, head_size]
        key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
        slot_mapping_ptr,  # [num_tokens]
        k_scale_ptr,
        v_scale_ptr,
        x,
        k_stride0,
        v_stride0,
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        QUANT: tl.constexpr,
    ):
        tid = tl.program_id(0)
        head_id = tl.program_id(1)
        offset = tl.arange(0, BLOCK_SIZE)
        src_offset_k = tid * k_stride0 + head_id * head_size
        src_offset_v = tid * v_stride0 + head_id * head_size
        slot_id = tl.load(slot_mapping_ptr + tid)
        if slot_id < 0:
            return
        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        dst_offset = (
            block_id * num_kv_heads * head_size * block_size
            + head_id * head_size * block_size
        )
        dst_k_shuffle_offset = (
            dst_offset + offset // x * block_size * x + block_offset * x + offset % x
        )
        dst_v_shuffle_offset = (
            dst_offset
            + block_offset // x * head_size * x
            + offset * x
            + block_offset % x
        )
        k_val = tl.load(key_ptr + src_offset_k + offset)
        v_val = tl.load(value_ptr + src_offset_v + offset)
        if QUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)
        tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
        tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)

    def reshape_and_cache_shuffle_triton(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
    ):
        num_tokens = slot_mapping.shape[0]
        _, num_kv_heads, head_size = key.shape
        num_blocks, block_size, _, _ = key_cache.shape
        x = 16 // key_cache.element_size()
        k_cache_template = torch.empty(
            [num_blocks, num_kv_heads, head_size // x, block_size, x],
            dtype=key_cache.dtype,
            device="meta",
        )
        v_cache_template = torch.empty(
            [num_blocks, num_kv_heads, block_size // x, head_size, x],
            dtype=value_cache.dtype,
            device="meta",
        )
        new_key_cache = key_cache.view_as(k_cache_template)
        new_value_cache = value_cache.view_as(v_cache_template)
        QUANT = False
        if kv_cache_dtype.startswith("fp8"):
            QUANT = True
            
        kv_cache_torch_dtype = (current_platform.fp8_dtype() if "fp8" in kv_cache_dtype else torch.int8)
        new_key_cache = new_key_cache.view(kv_cache_torch_dtype)
        new_value_cache = new_value_cache.view(kv_cache_torch_dtype)

        # aiter.reshape_and_cache_with_pertoken_quant(
        #     key, value, new_key_cache, new_value_cache, k_scales, v_scales, slot_mapping.flatten(), True)
        
            
        grid = (
            num_tokens,
            num_kv_heads,
        )
        reshape_and_cache_shuffle_kernel[grid](
            key,
            value,
            new_key_cache,
            new_value_cache,
            slot_mapping,
            k_scales,
            v_scales,
            x,
            key.stride(0),
            value.stride(0),
            block_size,
            head_size,
            num_kv_heads,
            BLOCK_SIZE=head_size,
            QUANT=QUANT,
        )


logger = init_logger(__name__)


@dataclass
class AiterFlashAttentionDecodeMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


@dataclass
class AiterFlashAttentionPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


@dataclass
class AiterChunkSlidingWindowMetadata:
    swa_seqlens: torch.Tensor
    swa_cu_seqlens: torch.Tensor
    swa_seq_starts: torch.Tensor
    swa_token_to_batch: torch.Tensor
    swa_max_seqlens: int
    swa_total_tokens: int
    swa_workspace: torch.Tensor


@dataclass
class AiterChunkContextMetadata:
    workspace: torch.Tensor
    cu_seq_lens_chunk: torch.Tensor
    chunk_starts: torch.Tensor
    token_to_batch: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    num_chunks: int
    total_token_per_batch: list[int]
    swa_metadata: AiterChunkSlidingWindowMetadata | None


@dataclass
class AiterFlashAttentionChunkPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    chunk_context_metadata: AiterChunkContextMetadata


@dataclass
class AiterFlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor

    # prefill and deocde split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_extends: int
    num_extend_tokens: int

    decode_metadata: AiterFlashAttentionDecodeMetadata | None
    prefill_metadata: AiterFlashAttentionPrefillMetadata | None
    extend_metadata: AiterFlashAttentionChunkPrefillMetadata | None

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    total_tokens: int


class AiterFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[AiterFlashAttentionMetadata]
):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None
        self.total_tokens: int = 0

        sliding_window_configs: set[tuple[int, int] | None] = set()
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer in layers.values():
            assert isinstance(layer.impl, AiterFlashAttentionImpl)
            sliding_window_configs.add(layer.impl.sliding_window)

        while len(sliding_window_configs) > 0:
            sliding_window_config = sliding_window_configs.pop()
            if sliding_window_config is not None and sliding_window_config[0] != -1:
                assert self.aot_sliding_window is None, (
                    "Aiter Flash ATTENTION can only support one valid sliding window!"
                )
                self.aot_sliding_window = sliding_window_config

        self.extend_workspace = torch.empty(
            [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.headdim],
            dtype=self.model_config.dtype,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        self.total_tokens = (
            self.model_config.max_model_len
            * self.vllm_config.scheduler_config.max_num_partial_prefills
        )
        res = self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.total_tokens = 0
        return res

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "AiterFlashAttentionMetadata":
        split_ret = split_decodes_prefills_and_extends(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )

        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            num_extend_tokens,
            num_prefill_tokens,
        ) = split_ret

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        seq_lens = common_attn_metadata.seq_lens.cpu()

        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AiterFlashAttentionDecodeMetadata(
                max_query_len=query_lens_cpu[:num_decodes].max().item(),
                min_query_len=query_lens_cpu[:num_decodes].min().item(),
                max_seq_len=seq_lens[:num_decodes].max().item(),
                query_start_loc=common_attn_metadata.query_start_loc[: num_decodes + 1],
            )

        prefill_metadata = None
        if num_prefills > 0:
            query_lens_for_prefill = query_lens_cpu[num_decodes + num_extends :]
            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes + num_extends :
            ]
            prefill_metadata = AiterFlashAttentionPrefillMetadata(
                max_query_len=query_lens_for_prefill.max().item(),
                min_query_len=query_lens_for_prefill.min().item(),
                max_seq_len=seq_lens[num_decodes + num_extends :].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
            )

        extend_metadata = None
        if num_extends > 0:
            num_extends_slice = slice(num_decodes, num_decodes + num_extends)
            query_lens_for_extend = query_lens_cpu[num_extends_slice]
            seq_lens_for_extend = seq_lens[num_extends_slice]
            computed_kv_lens = seq_lens_for_extend - query_lens_for_extend
            swa_metadata = None
            if self.aot_sliding_window is not None:
                swa_seqlen_for_extend = torch.minimum(
                    seq_lens_for_extend,
                    query_lens_for_extend + self.aot_sliding_window[0] + 1,
                )
                cu_seq_lens = torch.zeros(
                    num_extends + 1,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                torch.cumsum(
                    swa_seqlen_for_extend,
                    dim=0,
                    dtype=cu_seq_lens.dtype,
                    out=cu_seq_lens[1:],
                )
                token_to_seq = torch.arange(
                    0,
                    num_extends,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                token_to_seq = torch.repeat_interleave(
                    token_to_seq, swa_seqlen_for_extend
                )
                fetched_shape = cu_seq_lens[-1].item()
                # TODO(ganyi): Maybe reuse these 2 buffer from extend_workspace
                swa_workspace = torch.empty(
                    (2, fetched_shape, self.num_heads_kv, self.headdim),
                    dtype=self.vllm_config.model_config.dtype,
                    device=self.device,
                )

                seq_starts = seq_lens_for_extend - swa_seqlen_for_extend
                max_seqlen_k = swa_seqlen_for_extend.max().item()
                total_tokens = cu_seq_lens[-1].item()

                swa_metadata = AiterChunkSlidingWindowMetadata(
                    swa_seqlens=swa_seqlen_for_extend.to(
                        self.device, non_blocking=True
                    ),
                    swa_cu_seqlens=cu_seq_lens.to(self.device, non_blocking=True),
                    swa_seq_starts=seq_starts.to(self.device, non_blocking=True),
                    swa_token_to_batch=token_to_seq.to(self.device, non_blocking=True),
                    swa_max_seqlens=max_seqlen_k,
                    swa_total_tokens=total_tokens,
                    swa_workspace=swa_workspace,
                )

            # allocate the equal amount of workspace for
            # each chunk prefill request
            max_context_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends
            num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk)

            chunk_starts = (
                torch.arange(num_chunks, dtype=torch.int32)
                .unsqueeze(1)
                .expand(-1, num_extends)
                * max_context_chunk
            )
            chunk_ends = torch.min(
                computed_kv_lens.unsqueeze(0), chunk_starts + max_context_chunk
            )
            chunk_seq_lens = (chunk_ends - chunk_starts).clamp(
                min=0
            )  # [num_chunks, num_extends]
            cu_seq_lens_cpu = torch.zeros(
                [num_chunks, num_extends + 1], dtype=torch.int32, pin_memory=True
            )
            torch.cumsum(
                chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
            )
            max_cum_tokens = cu_seq_lens_cpu[:, -1].max().item()

            range_idx = torch.arange(max_cum_tokens, dtype=torch.int32)[None, None, :]
            idx_to_batch_tensor = range_idx == cu_seq_lens_cpu[:, 1:][:, :, None]
            idx_to_batch_tensor = idx_to_batch_tensor.sum(
                dim=1
            )  # [num_chunks, max_cum_tokens]
            token_to_batch_tensor = torch.cumsum(idx_to_batch_tensor, dim=1)

            chunk_context_metadata = AiterChunkContextMetadata(
                workspace=self.extend_workspace,
                cu_seq_lens_chunk=cu_seq_lens_cpu.to(self.device, non_blocking=True),
                chunk_starts=chunk_starts.to(self.device, non_blocking=True),
                seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                seq_lens=chunk_seq_lens,
                token_to_batch=token_to_batch_tensor.to(self.device, non_blocking=True),
                num_chunks=num_chunks,
                total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(),
                swa_metadata=swa_metadata,
            )

            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes : num_decodes + num_extends + 1
            ]
            seq_lens_device = common_attn_metadata.seq_lens[num_extends_slice]
            cu_seq_lens = torch.zeros(
                num_extends + 1, dtype=torch.int32, device=seq_lens_device.device
            )
            torch.cumsum(
                seq_lens_device, dim=0, dtype=cu_seq_lens.dtype, out=cu_seq_lens[1:]
            )
            extend_metadata = AiterFlashAttentionChunkPrefillMetadata(
                max_query_len=query_lens_for_extend.max().item(),
                min_query_len=query_lens_for_extend.min().item(),
                max_seq_len=seq_lens[num_extends_slice].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
                chunk_context_metadata=chunk_context_metadata,
            )

        num_actual_kv_tokens = torch.sum(seq_lens).item()

        use_cascade = common_prefix_len > 0

        attn_metadata = AiterFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_extends=num_extends,
            num_extend_tokens=num_extend_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
            extend_metadata=extend_metadata,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            total_tokens=self.total_tokens,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class AiterFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AiterFlashAttentionImpl"]:
        return AiterFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AiterFlashAttentionMetadataBuilder"]:
        return AiterFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # FP4 uses 2 values per byte → half the elements per slot.
        if (cache_dtype_str or "").startswith("fp4"):
            return (2, num_blocks, block_size, num_kv_heads, head_size // 2)
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class AiterFlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0.0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type not in [AttentionType.DECODER, AttentionType.ENCODER_DECODER]:
            raise NotImplementedError(
                "Encoder self-attention is not implemented for FlashAttentionImpl"
            )

    def extend_for_sliding_window(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key_cache,
        value_cache,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        k_scale: float,
        v_scale: float,
        fp4_k_scales_2d: torch.Tensor | None = None,
        fp4_v_scales_2d: torch.Tensor | None = None,
    ):
        assert attn_metadata.extend_metadata is not None
        assert attn_metadata.extend_metadata.chunk_context_metadata is not None
        chunked_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        swa_metadata = chunked_metadata.swa_metadata
        assert swa_metadata is not None
        swa_cu_seqlens = swa_metadata.swa_cu_seqlens
        swa_seq_starts = swa_metadata.swa_seq_starts
        swa_token_to_batch = swa_metadata.swa_token_to_batch
        swa_max_seqlens = swa_metadata.swa_max_seqlens
        swa_total_tokens = swa_metadata.swa_total_tokens
        key_fetched, value_fetched = (
            swa_metadata.swa_workspace[0],
            swa_metadata.swa_workspace[1],
        )
        cp_mha_gather_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key_fetched,
            value=value_fetched,
            block_tables=block_table,
            k_scales=k_scale,
            v_scales=v_scale,
            cu_seqlens_kv=swa_cu_seqlens,
            token_to_batch=swa_token_to_batch,
            seq_starts=swa_seq_starts,
            dequant=False,
            kv_cache_layout="NHD",
            total_tokens=swa_total_tokens,
            k_dequant_scales_2d=fp4_k_scales_2d,
            v_dequant_scales_2d=fp4_v_scales_2d,
        )

        aiter.flash_attn_varlen_func(
            q=query,
            k=key_fetched,
            v=value_fetched,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=swa_cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=swa_max_seqlens,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=False,
            out=output,
        )

    def extend_forward(
        self,
        attn_metadata: AiterFlashAttentionMetadata,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: float,
        v_scale: float,
        fp4_k_scales_2d: torch.Tensor | None = None,
        fp4_v_scales_2d: torch.Tensor | None = None,
    ):
        if self.sliding_window[0] != -1:
            self.extend_for_sliding_window(
                attn_metadata,
                query,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q,
                max_seqlen_q,
                block_table,
                k_scale,
                v_scale,
                fp4_k_scales_2d,
                fp4_v_scales_2d,
            )
            return
        out, lse = aiter.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            min_seqlen_q=min_seqlen_q,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=True,
        )
        assert attn_metadata.extend_metadata is not None
        chunk_context_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        num_chunks = chunk_context_metadata.num_chunks
        workspace = chunk_context_metadata.workspace
        cu_seqlens_kv = chunk_context_metadata.cu_seq_lens_chunk
        max_seqlens = chunk_context_metadata.max_seq_lens
        chunk_starts = chunk_context_metadata.chunk_starts
        token_to_batch = chunk_context_metadata.token_to_batch
        total_token_per_batch = chunk_context_metadata.total_token_per_batch
        key_fetched, value_fetched = workspace[0], workspace[1]
        chunked_output = None
        chunked_lse = None
        for chunk_idx in range(num_chunks):
            cp_mha_gather_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                key=key_fetched,
                value=value_fetched,
                block_tables=block_table,
                k_scales=k_scale,
                v_scales=v_scale,
                cu_seqlens_kv=cu_seqlens_kv[chunk_idx],
                token_to_batch=token_to_batch[chunk_idx],
                seq_starts=chunk_starts[chunk_idx],
                dequant=False,
                kv_cache_layout="SHUFFLE" if USING_SHUFFLE_LAYOUT else "NHD",
                total_tokens=total_token_per_batch[chunk_idx],
                k_dequant_scales_2d=fp4_k_scales_2d,
                v_dequant_scales_2d=fp4_v_scales_2d,
            )

            suf_out, suf_lse = aiter.flash_attn_varlen_func(
                q=query,
                k=key_fetched,
                v=value_fetched,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlens[chunk_idx],
                min_seqlen_q=min_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                return_lse=True,
            )
            if chunked_output is None:
                chunked_output = suf_out
                chunked_lse = suf_lse
            else:
                tmp_output = torch.empty_like(out)
                tmp_lse = torch.empty_like(lse)
                merge_attn_states(
                    output=tmp_output,
                    output_lse=tmp_lse,
                    prefix_output=chunked_output,
                    prefix_lse=chunked_lse,
                    suffix_output=suf_out,
                    suffix_lse=suf_lse,
                )
                chunked_output = tmp_output
                chunked_lse = tmp_lse

        merge_attn_states(
            output=output,
            prefix_output=chunked_output,
            prefix_lse=chunked_lse,
            suffix_output=out,
            suffix_lse=lse,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AiterFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with AiterFlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is
        # executed in eager-mode PyTorch. Thus, we need to be careful
        # about any CPU overhead in this method. For example, `view`
        # and `slice` (or `[:n]`) operations are surprisingly slow even
        # in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping
            # is not padded. However, we don't need to do
            # key[:num_actual_tokens] and value[:num_actual_tokens] because
            # the reshape_and_cache_flash op uses the slot_mapping's shape
            # to determine the number of actual tokens.

            if self.kv_cache_dtype.startswith("fp4") and getattr(
                layer, "fp4_per_token_quant", False
            ):
                _ensure_amxfp4_op_available()
                _ensure_nvfp4_op_available()
                num_kv_heads = key_cache.size(2)
                total_tokens = key_cache.size(0) * key_cache.size(1)
                head_size = key.shape[-1]

                cache_key = key
                cache_value = value
                if _FP4_HADAMARD and not _FP4_IN_KERNEL_WHT:
                    signs = _get_fp4_hadamard_signs(
                        key.shape[-1], key.device)
                    cache_key = _hadamard_rotate(
                        key, signs).to(key.dtype)
                    cache_value = _hadamard_rotate(
                        value, signs).to(value.dtype)

                if _FP4_AMXFP4:
                    (
                        k_scales_2d, v_scales_2d,
                        k_bm_2d, v_bm_2d,
                        k_packed_2d, v_packed_2d,
                    ) = _get_or_create_fp4_amxfp4_scales(
                        layer, num_kv_heads, total_tokens,
                        key_cache.device, head_size,
                    )
                    torch.ops._C_cache_ops \
                        .reshape_and_cache_flash_fp4_amxfp4(
                            cache_key,
                            cache_value,
                            key_cache,
                            value_cache,
                            k_scales_2d,
                            v_scales_2d,
                            k_bm_2d,
                            v_bm_2d,
                            attn_metadata.slot_mapping,
                            self.kv_cache_dtype,
                        )
                    num_blk = layer._fp4_num_k_blocks
                    num_h = num_kv_heads
                    k_packed_2d[:num_h * num_blk, :].copy_(k_scales_2d)
                    k_packed_2d[num_h * num_blk:, :].copy_(k_bm_2d)
                    v_packed_2d[:num_h * num_blk, :].copy_(v_scales_2d)
                    v_packed_2d[num_h * num_blk:, :].copy_(v_bm_2d)
                elif _FP4_NVFP4:
                    (
                        k_fp8_flat, v_fp8_flat,
                        k_fp8_2d, v_fp8_2d,
                    ) = _get_or_create_fp4_nvfp4_scales(
                        layer, num_kv_heads, total_tokens,
                        key_cache.device, head_size,
                    )
                    torch.ops._C_cache_ops \
                        .reshape_and_cache_flash_fp4_nvfp4(
                            cache_key,
                            cache_value,
                            key_cache,
                            value_cache,
                            k_fp8_2d,
                            v_fp8_2d,
                            attn_metadata.slot_mapping,
                            self.kv_cache_dtype,
                        )
                elif _FP4_MXFP4:
                    (
                        k_e8m0_flat, v_e8m0_flat, k_e8m0_2d, v_e8m0_2d,
                    ) = _get_or_create_fp4_mxfp4_scales(
                        layer, num_kv_heads, total_tokens,
                        key_cache.device, head_size,
                    )
                    torch.ops._C_cache_ops \
                        .reshape_and_cache_flash_fp4_mxfp4(
                            cache_key,
                            cache_value,
                            key_cache,
                            value_cache,
                            k_e8m0_2d,
                            v_e8m0_2d,
                            attn_metadata.slot_mapping,
                            self.kv_cache_dtype,
                        )
                elif _FP4_PER_CHANNEL_K:
                    (
                        k_ch_scales, v_flat, v_2d, k_ready,
                    ) = _get_or_create_fp4_per_channel_k_scales(
                        layer, num_kv_heads, total_tokens,
                        key_cache.device, head_size,
                    )
                    if not k_ready:
                        k_data = cache_key.float()
                        optimized_scales = _compute_k_channel_scales(
                            k_data,
                            clip_sigma=_FP4_PCK_CLIP_SIGMA,
                            mse_iters=_FP4_PCK_MSE_ITERS,
                            safety_margin=_FP4_PCK_SAFETY_MARGIN,
                        )
                        k_ch_scales.copy_(optimized_scales)
                        layer._fp4_k_channel_scales_ready = True
                    torch.ops._C_cache_ops \
                        .reshape_and_cache_flash_fp4_per_channel_k_per_token_v(
                            cache_key,
                            cache_value,
                            key_cache,
                            value_cache,
                            k_ch_scales,
                            v_2d,
                            attn_metadata.slot_mapping,
                            self.kv_cache_dtype,
                        )
                else:
                    (
                        k_flat, v_flat, k_2d, v_2d,
                    ) = _get_or_create_fp4_pertoken_scales(
                        layer, num_kv_heads, total_tokens,
                        key_cache.device, head_size,
                    )
                    torch.ops._C_cache_ops \
                        .reshape_and_cache_flash_with_pertoken_quant(
                            cache_key,
                            cache_value,
                            key_cache,
                            value_cache,
                            k_2d,
                            v_2d,
                            attn_metadata.slot_mapping,
                            self.kv_cache_dtype,
                        )
            elif USING_SHUFFLE_LAYOUT:
                num_blocks, block_size, num_kv_heads, head_size = \
                    key_cache.shape

                k_scale, v_scale = get_static_kvscale(
                    1.0, 1.0, num_kv_heads,
                    num_blocks, block_size, kv_cache.device,
                )

                layer._k_scale = k_scale
                layer._v_scale = v_scale

                reshape_and_cache_shuffle_triton(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )
            else:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.kv_cache_dtype.startswith("fp8") or self.kv_cache_dtype.startswith("fp4"):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # decode:extend:prefill
        query = query[:num_actual_tokens]
        if key is not None:
            key = key[:num_actual_tokens]
        if value is not None:
            value = value[:num_actual_tokens]

        output_actual_tokens = output[:num_actual_tokens]

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_extends = attn_metadata.num_extends

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.num_extend_tokens
        if not attn_metadata.use_cascade:
            # calculate for pure prefills
            if num_prefills > 0:
                assert attn_metadata.prefill_metadata is not None

                prefill_query = query[num_decode_tokens + num_extend_tokens :]
                prefill_key = key[num_decode_tokens + num_extend_tokens :]
                prefill_value = value[num_decode_tokens + num_extend_tokens :]

                aiter.flash_attn_varlen_func(
                    q=prefill_query,
                    k=prefill_key,
                    v=prefill_value,
                    cu_seqlens_q=attn_metadata.prefill_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.prefill_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.prefill_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.prefill_metadata.max_seq_len,
                    min_seqlen_q=1,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    out=output_actual_tokens[num_decode_tokens + num_extend_tokens :],
                )

            # calculate for extends
            if num_extends > 0:
                assert attn_metadata.extend_metadata is not None
                extend_tokens_slice = slice(
                    num_decode_tokens, num_decode_tokens + num_extend_tokens
                )
                extend_querys = query[extend_tokens_slice]
                extend_keys = key[extend_tokens_slice]
                extend_values = value[extend_tokens_slice]
                extend_outputs = output[extend_tokens_slice]
                fp4_k_scales_2d = None
                fp4_v_scales_2d = None
                if self.kv_cache_dtype.startswith("fp4") and getattr(
                    layer, "fp4_per_token_quant", False
                ):
                    if _FP4_AMXFP4 and hasattr(
                        layer, "_fp4_amxfp4_k_packed_2d"
                    ):
                        fp4_k_scales_2d = \
                            layer._fp4_amxfp4_k_packed_2d
                        fp4_v_scales_2d = \
                            layer._fp4_amxfp4_v_packed_2d
                    elif _FP4_NVFP4 and hasattr(
                        layer, "_fp4_nvfp4_k_scales_2d"
                    ):
                        fp4_k_scales_2d = \
                            layer._fp4_nvfp4_k_scales_2d
                        fp4_v_scales_2d = \
                            layer._fp4_nvfp4_v_scales_2d
                    elif _FP4_MXFP4 and hasattr(
                        layer, "_fp4_mxfp4_k_scales_2d"
                    ):
                        fp4_k_scales_2d = \
                            layer._fp4_mxfp4_k_scales_2d
                        fp4_v_scales_2d = \
                            layer._fp4_mxfp4_v_scales_2d
                    elif _FP4_PER_CHANNEL_K and hasattr(
                        layer, "_fp4_k_channel_scales"
                    ):
                        fp4_k_scales_2d = \
                            layer._fp4_k_channel_scales
                        fp4_v_scales_2d = \
                            layer._fp4_v_pertoken_scales_2d
                    else:
                        fp4_k_scales_2d = getattr(
                            layer, "_fp4_k_dequant_scales_2d", None
                        )
                        fp4_v_scales_2d = getattr(
                            layer, "_fp4_v_dequant_scales_2d", None
                        )
                self.extend_forward(
                    attn_metadata=attn_metadata,
                    query=extend_querys,
                    key=extend_keys,
                    value=extend_values,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=extend_outputs,
                    cu_seqlens_q=attn_metadata.extend_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.extend_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.extend_metadata.max_seq_len,
                    min_seqlen_q=1,
                    block_table=attn_metadata.block_table[
                        num_decodes : num_decodes + num_extends
                    ],
                    slot_mapping=attn_metadata.slot_mapping[
                        num_decodes : num_decodes + num_extends
                    ],
                    k_scale=layer._k_scale,
                    v_scale=layer._v_scale,
                    fp4_k_scales_2d=fp4_k_scales_2d,
                    fp4_v_scales_2d=fp4_v_scales_2d,
                )

            # calculate for decodes
            if num_decodes > 0:
                assert attn_metadata.decode_metadata is not None
                if (self.sliding_window[0] != -1
                        and not self.kv_cache_dtype.startswith("fp4")):
                    assert not USING_SHUFFLE_LAYOUT, (
                        "Sliding window with shuffle layout is not supported yet."
                    )
                    from aiter.ops.triton.unified_attention import (
                        unified_attention,
                    )

                    descale_shape = (
                        attn_metadata.query_start_loc[:num_decodes].shape[0] - 1,
                        key_cache.shape[2],
                    )
                    unified_attention(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=attn_metadata.query_start_loc[:num_decodes],
                        max_seqlen_q=1,  # optimize this
                        seqused_k=attn_metadata.seq_lens[:num_decodes],
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=attn_metadata.block_table[:num_decodes],
                        softcap=self.logits_soft_cap,
                        q_descale=None,
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                    )
                    return
                assert attn_metadata.decode_metadata is not None
                if USING_SHUFFLE_LAYOUT:
                    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
                    x = 16 // key_cache.element_size()
                    k_cache_template = torch.empty(
                        [num_blocks, num_kv_heads, head_size // x, block_size, x],
                        dtype=key_cache.dtype,
                        device="meta",
                    )
                    v_cache_template = torch.empty(
                        [num_blocks, num_kv_heads, block_size // x, head_size, x],
                        dtype=value_cache.dtype,
                        device="meta",
                    )
                    new_key_cache = key_cache.view_as(k_cache_template)
                    new_value_cache = value_cache.view_as(v_cache_template)
                    aiter.pa_fwd_asm(
                        Q=query[:num_decode_tokens],
                        K=new_key_cache,
                        V=new_value_cache,
                        block_tables=attn_metadata.block_table[:num_decodes],
                        context_lens=attn_metadata.seq_lens[:num_decodes],
                        block_tables_stride0=attn_metadata.block_table[
                            :num_decodes
                        ].stride(0),
                        K_QScale=layer._k_scale,
                        V_QScale=layer._v_scale,
                        out_=output[:num_decode_tokens],
                    )
                else:
                    _, num_heads, head_size = query.shape
                    nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
                    num_seqs = attn_metadata.seq_lens.shape[0]
                    max_num_partitions = (
                        attn_metadata.max_seq_len + _PARTITION_SIZE_ROCM - 1
                    ) // _PARTITION_SIZE_ROCM

                    workspace_buffer = torch.empty(
                        (num_seqs * num_heads * max_num_partitions * head_size)
                        * nbytes_per_qo_elem
                        + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
                        dtype=torch.uint8,
                        device=output.device,
                    )

                    k_scale_stride_h = 0
                    v_scale_stride_h = 0
                    fp4_num_k_blocks = 1
                    fp4_num_v_blocks = 1
                    fp4_per_channel_k_active = False
                    fp4_mxfp4_active = False
                    fp4_nvfp4_active = False
                    fp4_amxfp4_active = False
                    if self.kv_cache_dtype.startswith("fp4") and getattr(
                        layer, "fp4_per_token_quant", False
                    ):
                        if _FP4_AMXFP4 and hasattr(
                            layer, "_fp4_amxfp4_k_packed_2d"
                        ):
                            fp4_amxfp4_active = True
                            k_scale_for_pa = \
                                layer._fp4_amxfp4_k_packed_2d \
                                    .contiguous().view(-1)
                            v_scale_for_pa = \
                                layer._fp4_amxfp4_v_packed_2d \
                                    .contiguous().view(-1)
                            k_scale_stride_h = \
                                layer._fp4_k_scale_stride_h
                            v_scale_stride_h = \
                                layer._fp4_v_scale_stride_h
                            num_k_blks = layer._fp4_num_k_blocks
                            num_v_blks = layer._fp4_num_v_blocks
                            fp4_num_k_blocks = \
                                -(num_k_blks + _AMXFP4_SIGNAL_OFFSET)
                            fp4_num_v_blocks = \
                                -(num_v_blks + _AMXFP4_SIGNAL_OFFSET)
                        elif _FP4_NVFP4 and hasattr(
                            layer, "_fp4_nvfp4_k_scales_flat"
                        ):
                            fp4_nvfp4_active = True
                            k_scale_for_pa = \
                                layer._fp4_nvfp4_k_scales_flat
                            v_scale_for_pa = \
                                layer._fp4_nvfp4_v_scales_flat
                            k_scale_stride_h = \
                                layer._fp4_k_scale_stride_h
                            v_scale_stride_h = \
                                layer._fp4_v_scale_stride_h
                            num_k_blks = layer._fp4_num_k_blocks
                            num_v_blks = layer._fp4_num_v_blocks
                            fp4_num_k_blocks = \
                                -(num_k_blks + _NVFP4_SIGNAL_OFFSET)
                            fp4_num_v_blocks = \
                                -(num_v_blks + _NVFP4_SIGNAL_OFFSET)
                        elif _FP4_MXFP4 and hasattr(
                            layer, "_fp4_mxfp4_k_scales_flat"
                        ):
                            fp4_mxfp4_active = True
                            k_scale_for_pa = \
                                layer._fp4_mxfp4_k_scales_flat
                            v_scale_for_pa = \
                                layer._fp4_mxfp4_v_scales_flat
                            k_scale_stride_h = \
                                layer._fp4_k_scale_stride_h
                            v_scale_stride_h = \
                                layer._fp4_v_scale_stride_h
                            num_k_blks = layer._fp4_num_k_blocks
                            num_v_blks = layer._fp4_num_v_blocks
                            fp4_num_k_blocks = -num_k_blks
                            fp4_num_v_blocks = -num_v_blks
                        elif _FP4_PER_CHANNEL_K and hasattr(
                            layer, "_fp4_k_channel_scales"
                        ):
                            fp4_per_channel_k_active = True
                            k_scale_for_pa = torch.ones(
                                1, dtype=torch.float32,
                                device=output.device,
                            )
                            v_scale_for_pa = \
                                layer._fp4_v_pertoken_scales_flat
                            v_scale_stride_h = \
                                layer._fp4_v_scale_stride_h
                            fp4_num_k_blocks = 0
                            fp4_num_v_blocks = 1
                        else:
                            k_scale_for_pa = \
                                layer._fp4_k_dequant_scales_flat
                            v_scale_for_pa = \
                                layer._fp4_v_dequant_scales_flat
                            k_scale_stride_h = \
                                layer._fp4_k_scale_stride_h
                            v_scale_stride_h = \
                                layer._fp4_v_scale_stride_h
                            fp4_num_k_blocks = getattr(
                                layer, "_fp4_num_k_blocks", 1)
                            fp4_num_v_blocks = getattr(
                                layer, "_fp4_num_v_blocks", 1)
                    elif self.kv_cache_dtype.startswith("fp4"):
                        total_tokens = \
                            key_cache.size(0) * key_cache.size(1)
                        k_scale_for_pa = get_fp4_per_token_kscale(
                            layer._k_scale.flatten()[0].item(),
                            key_cache.size(2),
                            total_tokens,
                            key_cache.device,
                        )
                        v_scale_for_pa = layer._v_scale
                        k_scale_stride_h = total_tokens
                    else:
                        k_scale_for_pa = layer._k_scale
                        v_scale_for_pa = layer._v_scale

                    use_hadamard = (
                        _FP4_HADAMARD
                        and self.kv_cache_dtype.startswith("fp4")
                        and getattr(layer, "fp4_per_token_quant", False)
                    )
                    decode_q = query[:num_decode_tokens]
                    if use_hadamard:
                        signs = _get_fp4_hadamard_signs(
                            decode_q.shape[-1], decode_q.device)
                        decode_q = _hadamard_rotate(
                            decode_q, signs).to(query.dtype)

                    if fp4_per_channel_k_active:
                        k_ch = layer._fp4_k_channel_scales
                        gqa_ratio = self.num_heads // self.num_kv_heads
                        k_ch_q = k_ch.repeat_interleave(
                            gqa_ratio, dim=0
                        )
                        decode_q = (
                            decode_q.float() * k_ch_q.unsqueeze(0)
                        ).to(decode_q.dtype)

                    torch.ops.aiter.paged_attention_v1(
                        output[:num_decode_tokens],
                        workspace_buffer,
                        decode_q,
                        key_cache,
                        value_cache,
                        self.scale,
                        attn_metadata.block_table[:num_decodes],
                        attn_metadata.query_start_loc[:num_decodes],
                        attn_metadata.seq_lens[:num_decodes],
                        attn_metadata.max_seq_len,
                        self.alibi_slopes,
                        self.kv_cache_dtype,
                        "NHD",
                        self.logits_soft_cap,
                        k_scale_for_pa,
                        v_scale_for_pa,
                        None,
                        _PARTITION_SIZE_ROCM,
                        k_scale_stride_h=k_scale_stride_h,
                        v_scale_stride_h=v_scale_stride_h,
                        fp4_num_k_blocks=fp4_num_k_blocks,
                        fp4_num_v_blocks=fp4_num_v_blocks,
                    )

                    if use_hadamard:
                        output[:num_decode_tokens] = \
                            _hadamard_inv_rotate(
                                output[:num_decode_tokens], signs
                            ).to(output.dtype)
        else:
            raise NotImplementedError(
                "Cascade attention is not implemented for ROCM AITER"
            )

        return output

