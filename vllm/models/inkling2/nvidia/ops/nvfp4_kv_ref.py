"""Reference (pure torch) NVFP4 KV-cache row format for Inkling2 on SM12x.

THE CONTRACT. Every kernel that writes or reads an Inkling2 NVFP4 KV cache must match this file bit for bit.

One cache "row" = one (block, kv_head, slot) for K or for V, head_dim D (128 for Inkling 2 Mini):
    bytes [0, D//2)          packed E2M1 values; byte j holds element 2j in the LOW nibble, 2j+1 in the HIGH nibble
    bytes [D//2, D//2+D//16) one FP8 E4M3 scale per 16-element group g (elements 16g .. 16g+15)
Row size = D//2 + D//16 = 72 bytes for D=128 (FP8 KV used 128, BF16 256).

E2M1 nibble: bit 3 = sign, bits 0-2 = magnitude code -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
Scale: s_g = e4m3(amax_g / 6) (round to nearest e4m3); an all-zero group stores scale 0 and nibbles 0.
Quantize: q = round_to_nearest_e2m1(x / float(s_g)) (ties to the even code), clamp |q| <= 6.
Dequantize: x' = e2m1_value(nibble) * float(s_g).

The KV cache tensor stays uint8 with the same [num_blocks, block_size, num_kv_heads, 2*ROW] page shape the
fp8 path views (K row then V row), so _split_kv_cache can still split the last dim in half and kernels keep their
block/slot/head strides; they detect NVFP4 from the row width (D//2 + D//16) or a constexpr flag.
"""

import torch

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
GROUP = 16


def nvfp4_row_bytes(head_dim: int) -> int:
    assert head_dim % GROUP == 0
    return head_dim // 2 + head_dim // GROUP


def quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """x: [..., D] float -> uint8 [..., D//2 + D//16] in the contract layout."""
    D = x.shape[-1]
    xf = x.float().reshape(-1, D // GROUP, GROUP)
    amax = xf.abs().amax(-1, keepdim=True)
    scale_e4m3 = (amax / 6.0).to(torch.float8_e4m3fn)
    scale = scale_e4m3.float()
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = (xf / safe).clamp(-6.0, 6.0)
    vals = E2M1_VALUES.to(x.device)
    # nearest magnitude code; ties go to the even code (lower index of an equidistant pair is even here)
    dist = (y.abs().unsqueeze(-1) - vals).abs()
    code = dist.argmin(-1).to(torch.uint8)
    code = torch.where(scale == 0, torch.zeros_like(code), code)
    sign = ((y < 0) & (code > 0)).to(torch.uint8) << 3
    nib = (code | sign).reshape(-1, D)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    scales = scale_e4m3.reshape(-1, D // GROUP).view(torch.uint8)
    out = torch.cat([packed, scales], dim=-1)
    return out.reshape(*x.shape[:-1], nvfp4_row_bytes(D))


def dequantize_rows(rows: torch.Tensor, head_dim: int, dtype=torch.float32) -> torch.Tensor:
    """uint8 [..., D//2 + D//16] -> [..., D] dtype."""
    D = head_dim
    r = rows.reshape(-1, nvfp4_row_bytes(D))
    packed, scales = r[:, : D // 2], r[:, D // 2 :]
    lo, hi = packed & 0xF, packed >> 4
    nib = torch.stack([lo, hi], dim=-1).reshape(-1, D).long()
    mag = E2M1_VALUES.to(rows.device)[nib & 7]
    val = torch.where((nib & 8) != 0, -mag, mag)
    s = scales.contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(GROUP, dim=-1)
    return (val * s).to(dtype).reshape(*rows.shape[:-1], D)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1000, 128) * 3
    x[5] = 0
    rt = dequantize_rows(quantize_rows(x), 128)
    rel = (rt - x).norm() / x.norm()
    assert quantize_rows(x).shape == (1000, 72)
    assert torch.equal(quantize_rows(rt), quantize_rows(x)), "requantizing a dequantized row must be idempotent"
    assert rt[5].abs().sum() == 0
    print(f"ok: row bytes 72, relative error {rel:.4f}")
