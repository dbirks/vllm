# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 quantization of KV rows for the Inkling2 SM12x KV cache.

Device-side helpers used by the attention prologue writer
(`ops/sconv.py::_kv_conv_tile`) to store one (token, kv_head) K or V row of
`D` values as `D // 2 + D // 16` bytes:

    bytes [0, D//2)          packed E2M1 (element 2j low nibble, 2j+1 high nibble)
    bytes [D//2, D//2+D//16) one FP8 E4M3 scale per 16-element group

The bit-exact reference is `ops/nvfp4_kv_ref.py`; only portable `tl` ops are
used (no inline PTX) so the path also runs under `TRITON_INTERPRET=1`. The E4M3
scale is rounded with integer ops instead of `x.to(tl.float8e4nv)` because the
Triton CPU interpreter flushes E4M3 subnormals and saturates overflow to 256
instead of 448, which would make the interpreted result differ from both the
reference and the compiled CUDA path.
"""

from vllm.triton_utils import tl, triton

# Elements per NVFP4 scale group, and the value a group scale divides by.
NVFP4_GROUP = 16
_NVFP4_AMAX_DIV = 6.0
# Smallest E4M3 normal (2^-6), largest E4M3 finite value, E4M3 subnormal step.
_E4M3_MIN_NORMAL = 0.015625
_E4M3_MAX = 448.0
_E4M3_SUBNORMAL_SCALE = 512.0  # 2^9 = 1 / subnormal step
# fp32 magic constant for ties-to-even rounding to integers (double rounding).
_ROUND_MAGIC = 8388608.0  # 2^23


@triton.jit
def _e4m3_bits(x):
    """Byte of round-to-nearest-even E4M3FN(`x`) for `x >= 0` (NaN -> 0x7F).

    Matches `x.to(torch.float8_e4m3fn)` bit for bit, including the subnormal
    multiples of 2^-9 and the saturation of everything above 448 to 448.
    """
    b = x.to(tl.int32, bitcast=True)
    # Normal range: drop the fp32 mantissa to 3 bits, ties to even.
    low = b & 0x000FFFFF
    keep = b - low
    carry = (low > 0x00080000) | ((low == 0x00080000) & ((keep >> 20) & 1) != 0)
    exp_mant = ((keep + tl.where(carry, 0x00100000, 0)) >> 20) & 0x0FFF
    normal = (((exp_mant >> 3) - 120) << 3) | (exp_mant & 0x007)
    # Subnormal range: E4M3 steps are multiples of 2^-9; scaling by 2^9 is
    # exact, and adding/subtracting 2^23 rounds to an integer ties-to-even.
    sub = ((x * _E4M3_SUBNORMAL_SCALE + _ROUND_MAGIC) - _ROUND_MAGIC).to(tl.int32)
    sub = tl.minimum(sub, 8)
    bits = tl.where(x >= _E4M3_MAX, 126, tl.where(x < _E4M3_MIN_NORMAL, sub, normal))
    return tl.where(x != x, 127, bits).to(tl.uint8)


@triton.jit
def _e2m1_nibbles(y, scale):
    """E2M1 nibble of each `y` given its group `scale` (already an E4M3 float).

    `y` is the raw value; the code is `y / scale` rounded to the nearest E2M1
    magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} with ties taken by the lower code
    (what `argmin` of the distances does in the reference), sign bit 3, and 0
    for a zero scale.
    """
    a = tl.minimum(tl.abs(y), 6.0)
    code = (
        (a > 0.25).to(tl.int32)
        + (a > 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a > 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a > 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    code = tl.where(scale == 0, 0, code)
    return code + tl.where((y < 0) & (code > 0), 8, 0)


@triton.jit
def nvfp4_quantize_rows(y, D: tl.constexpr, GROUP: tl.constexpr):
    """Quantize `y` [R, D] float32 to NVFP4.

    Returns (packed, scales): `packed` is uint8 [R, D//2] of two E2M1 nibbles
    per byte (element 2j low, 2j+1 high), `scales` is uint8 [R, D//GROUP] of
    E4M3 scale bytes. Matches `nvfp4_kv_ref.quantize_rows`.
    """
    R: tl.constexpr = y.shape[0]
    G: tl.constexpr = D // GROUP
    tl.static_assert(D % GROUP == 0)
    groups = tl.reshape(y, (R, G, GROUP))
    amax = tl.max(tl.abs(groups), axis=2)
    scales = _e4m3_bits(tl.div_rn(amax, _NVFP4_AMAX_DIV))
    scale = scales.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # A zero scale (all-zero group, or one too small for E4M3) keeps nibbles 0.
    safe = tl.where(scale == 0, 1.0, scale)
    safe_full = tl.reshape(tl.broadcast_to(safe[:, :, None], (R, G, GROUP)), (R, D))
    q = tl.reshape(tl.div_rn(groups, safe[:, :, None]), (R, D // 2, 2))
    sf = tl.reshape(safe_full, (R, D // 2, 2))
    y_even, y_odd = tl.split(q)
    sf_even, sf_odd = tl.split(sf)
    nib_even = _e2m1_nibbles(y_even, sf_even)
    nib_odd = _e2m1_nibbles(y_odd, sf_odd)
    packed = (nib_even | (nib_odd << 4)).to(tl.uint8)
    return packed, scales


@triton.jit
def nvfp4_store_rows(
    cache_ptr, row_bytes, y, mask, D: tl.constexpr, GROUP: tl.constexpr
):
    """Write each row of `y` [R, D] float32 as one NVFP4 KV row.

    `cache_ptr` is the uint8 KV cache base and `row_bytes` the [R] byte offset
    of each row's first byte (the cache's last dim is in bytes, so the caller's
    element strides are byte strides); `mask` is the [R] rows to write.
    """
    packed, scales = nvfp4_quantize_rows(y, D, GROUP)
    NB: tl.constexpr = D // 2
    G: tl.constexpr = D // GROUP
    col = tl.arange(0, NB)
    grp = tl.arange(0, G)
    row_mask = mask[:, None]
    tl.store(cache_ptr + row_bytes[:, None] + col[None, :], packed, mask=row_mask)
    tl.store(cache_ptr + row_bytes[:, None] + NB + grp[None, :], scales, mask=row_mask)


@triton.jit
def nvfp4_store_kv_rows(
    cache_ptr,
    kv_slot,
    y,
    kv_block_size,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_head,
    head,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Store the `y` [R, D] rows of one KV head into the paged NVFP4 cache.

    `kv_slot` is the [R] destination slot of each token (< 0 skips the row) and
    the strides are `key_cache.stride()[:3]` of the uint8
    `[num_blocks, block_size, num_kv_heads, D//2 + D//16]` cache, i.e. exactly
    the byte offsets the bf16/fp8 path addresses in elements.
    """
    safe = tl.maximum(kv_slot, 0)
    row_bytes = (
        (safe // kv_block_size) * stride_kv_blk
        + (safe % kv_block_size) * stride_kv_slot
        + head * stride_kv_head
    )
    nvfp4_store_rows(cache_ptr, row_bytes, y, kv_slot >= 0, D, GROUP)
