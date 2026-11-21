# NVFP4 KV: CPU test of the NVFP4 KV-cache writer (task B).
#
#   uv run --no-project --with torch --with triton --with numpy \
#       --index-url https://download.pytorch.org/whl/cpu \
#       --extra-index-url https://pypi.org/simple python tests/test_nvfp4_writer.py
#
# Drives `ops/nvfp4_kv_write.nvfp4_store_kv_rows`, the device function the
# attention-prologue writer kernel (`ops/sconv.py::_kv_conv_tile`) calls for its
# NVFP4 store, and asserts the stored bytes equal `nvfp4_kv_ref.quantize_rows`
# byte for byte. The two op modules are loaded by file path with
# `vllm.triton_utils` stubbed, so no vLLM internals are needed.
"""Exercise the NVFP4 KV write path against the reference contract."""

import importlib.util
import os
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

OPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "vllm",
    "models",
    "inkling2",
    "nvidia",
    "ops",
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(OPS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Stub vllm.triton_utils (the real module re-exports exactly these two).
_pkg = types.ModuleType("vllm")
_pkg.__path__ = []  # type: ignore[attr-defined]
_utils = types.ModuleType("vllm.triton_utils")
_utils.tl = tl  # type: ignore[attr-defined]
_utils.triton = triton  # type: ignore[attr-defined]
sys.modules.setdefault("vllm", _pkg)
sys.modules["vllm.triton_utils"] = _utils

wr = _load("nvfp4_kv_write", "nvfp4_kv_write.py")
ref = _load("nvfp4_kv_ref", "nvfp4_kv_ref.py")
GROUP = 16


@triton.jit
def _write_rows(
    x_ptr,  # [R, D] float32 rows to store
    cache_ptr,  # uint8 KV cache base
    slot_ptr,  # [R] int32 KV slot (< 0: skip the write)
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_head,
    kv_block_size,
    head,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    R: tl.constexpr,
):
    """`_kv_conv_tile`'s NVFP4 store: the same helper with the same arguments."""
    rows = tl.arange(0, R)
    x = tl.load(x_ptr + rows[:, None] * D + tl.arange(0, D)[None, :])
    slot = tl.load(slot_ptr + rows)
    wr.nvfp4_store_kv_rows(
        cache_ptr,
        slot,
        x,
        kv_block_size,
        stride_kv_blk,
        stride_kv_slot,
        stride_kv_head,
        head,
        D,
        GROUP,
    )


@triton.jit
def _e4m3_bits_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """`_e4m3_bits` alone, for a dense sweep of the scale rounding."""
    for off in range(0, n, BLOCK):
        i = off + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out_ptr + i, wr._e4m3_bits(tl.load(x_ptr + i, mask=m)), mask=m)


def write_rows(x: torch.Tensor, block_size: int, num_heads: int, head: int, slots):
    """Store `x` [R, D] into a paged uint8 cache; return the cache and rows."""
    R, D = x.shape
    row = D // 2 + D // GROUP
    num_blocks = (
        int(max(s // block_size for s in (s if s >= 0 else 0 for s in slots))) + 1
    )
    cache = torch.full(
        (num_blocks, block_size, num_heads, row), 0xA5, dtype=torch.uint8
    )
    slot_t = torch.tensor(slots, dtype=torch.int32)
    _write_rows[(1,)](
        x,
        cache,
        slot_t,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        block_size,
        head,
        D=D,
        GROUP=GROUP,
        R=R,
    )
    return cache, row


def check_rows(name, x, slots=None, block_size=16, num_heads=2, head=1):
    """Assert the writer's bytes equal the contract's for each row of `x`."""
    R0, D = x.shape
    # The prologue kernel always processes a power-of-2 token tile (BT = 4|8);
    # pad with skipped rows (negative slot) to match.
    R = max(1, triton.next_power_of_2(R0))
    if R > R0:
        x = torch.cat([x, torch.zeros(R - R0, D, dtype=x.dtype)])
    row = D // 2 + D // GROUP
    slots = list(range(R0)) if slots is None else list(slots)
    assert len(slots) == R0
    slots += [-1] * (R - R0)
    cache, row = write_rows(x, block_size, num_heads, head, slots)
    want = ref.quantize_rows(x)
    assert want.shape == (R, row), (want.shape, R, row)
    for i, slot in enumerate(slots):
        if slot < 0:
            continue
        stored = cache[slot // block_size, slot % block_size, head]
        assert torch.equal(stored, want[i]), (
            f"{name}: row {i} (slot {slot}) differs\n"
            f" got {stored.tolist()}\nwant {want[i].tolist()}"
        )
    if any(s < 0 for s in slots):
        # Skipped rows must be untouched: no byte of any written row is 0xA5 in
        # the data area of a real row unless it was written; check by count.
        untouched = (cache != 0xA5).sum().item()
        written = sum(1 for s in slots if s >= 0) * row
        assert untouched <= written, (untouched, written)
    print(f"ok: {name}: {R0} rows (tile {R}) x D={D} match quantize_rows")


def make_rows(D: int, gen: torch.Generator, n: int, scale) -> torch.Tensor:
    return torch.rand(n, D, generator=gen) * 2 * scale - scale


def check_plumbing():
    """`_kv_conv_tile` itself cannot run here: ops/norm.rsqrt is inline PTX, which
    Triton rejects under TRITON_INTERPRET. Check the wiring that calls the store
    helper above, since the quantized bytes are only right if it is intact."""
    src = open(os.path.join(OPS_DIR, "sconv.py")).read()
    need = [
        # the constexpr reaches the writer kernel and selects the NVFP4 store
        "KV_NVFP4: tl.constexpr",
        "nvfp4_store_kv_rows(",
        # the launcher derives it from the packed row width, head_dim from q
        "kv_nvfp4 = kv_row_bytes == nvfp4_kv_cache_full_dim(head_dim)",
        "KV_NVFP4=kv_nvfp4",
    ]
    for needle in need:
        assert needle in src, f"sconv.py lost its NVFP4 writer plumbing: {needle!r}"
    assert src.count("KV_NVFP4: tl.constexpr") == 2, "prologue + writer constexpr"
    print(f"ok: sconv.py wires KV_NVFP4 to nvfp4_store_kv_rows ({len(need)} checks)")


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    ties_mid = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.0, 0.5, 6.0]

    for D in (128, 64):
        # Random rows at several magnitudes, positive and negative.
        for n in (4, 8):
            for mag in (0.5, 3.0, 200.0):
                check_rows(
                    f"rand n={n} D={D} mag={mag}",
                    make_rows(D, gen, n, mag),
                    slots=list(range(n)),
                )
        x = make_rows(D, gen, 8, 5.0)
        # One all-zero group, one all-zero row, one row with a single element.
        x[0, :GROUP] = 0.0
        x[1] = 0.0
        x[2] = 0.0
        x[2, 3] = -1.25
        x[3, :] = 0.0
        x[4, 0] = 0.0
        x[4, 1] = -0.0
        check_rows(f"zeros D={D}", x)
        # Groups too small for an E4M3 scale (scale rounds to 0, nibbles 0).
        x = make_rows(D, gen, 4, 1e-4)
        check_rows(f"subnormal-scale D={D}", x)
        # Ties against the E2M1 codes: build groups whose E4M3 scale is exactly
        # 2**k (amax = 3 * 2**k -> amax / 6 = 2**k), then place elements on the
        # midpoints between adjacent codes, so y = x / scale is a midpoint.
        rows = []
        for k in (-3, 0, 4):
            s = 2.0**k
            row = torch.tensor([m * 3.0 * s for m in ties_mid], dtype=torch.float32)
            row = row.repeat(D // row.numel() + 1)[:D].clone()
            # Every group must actually see 3*s (= 6*s amax) as its max.
            row = row.reshape(D // GROUP, GROUP)
            row[:, 0] = 3.0 * s
            row = row.reshape(D) * torch.where(torch.arange(D) % 2 == 0, 1.0, -1.0)
            rows.append(row)
        check_rows(f"e2m1 ties D={D}", torch.stack(rows))
        # Negative ties, same construction with the opposite signs.
        check_rows(f"e2m1 ties neg D={D}", -torch.stack(rows))
        # Values beyond the E2M1 range clamp to code 6 (nibble 7).
        x = make_rows(D, gen, 4, 40.0)
        x[:, 0] = 1e30
        x[:, 1] = -1e30
        check_rows(f"clamp D={D}", x)
        # Paged placement: multiple blocks/heads, and negative slots skip.
        check_rows(
            f"paged D={D}",
            make_rows(D, gen, 8, 2.0),
            slots=[0, 15, 16, 47, -1, 3, -1, 32],
            block_size=16,
            num_heads=2,
            head=0,
        )
        check_rows(
            f"paged head1 D={D}",
            make_rows(D, gen, 8, 2.0),
            slots=[0, 15, 16, 47, -1, 3, -1, 32],
            block_size=16,
            num_heads=2,
            head=1,
        )

    # The E4M3 scale rounding alone, over a dense sweep: subnormals, exact
    # midpoints between adjacent codes, the 448 saturation and NaN.
    codes = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    codes = codes[codes.isfinite()]
    mids = (codes[:-1] + codes[1:]) / 2
    xs = torch.cat(
        [
            torch.rand(400000, generator=gen) * 448.0,
            torch.rand(100000, generator=gen) * 0.02,
            torch.rand(50000, generator=gen) * 5000.0,
            mids,
            codes,
            codes * (1 + 1e-7),
            codes * (1 - 1e-7),
            torch.tensor([0.0, 448.0, 449.0, 464.0, 480.0, 1e30, float("nan")]),
        ]
    ).float()
    out = torch.zeros(xs.numel(), dtype=torch.uint8)
    _e4m3_bits_kernel[(1,)](xs, out, xs.numel(), BLOCK=1024)
    want = xs.to(torch.float8_e4m3fn).view(torch.uint8)
    bad = (out != want).nonzero().flatten()
    assert bad.numel() == 0, (
        f"_e4m3_bits: {bad.numel()} of {xs.numel()} differ, e.g. "
        + ", ".join(
            f"{xs[i].item()!r}: got {out[i].item()} want {want[i].item()}"
            for i in bad[:10]
        )
    )
    print(f"ok: _e4m3_bits matches torch on all {xs.numel()} swept values")

    # Round-trip sanity: the contract's own dequantization of what we stored.
    x = make_rows(128, gen, 32, 3.0)
    cache, row = write_rows(x, 16, 2, 1, list(range(32)))
    stored = cache[:, :, 1].reshape(-1, row)[: x.shape[0]]
    deq = ref.dequantize_rows(stored, 128)
    rel = (deq - x).norm() / x.norm()
    assert 0.0 < rel < 0.15, rel
    assert torch.equal(ref.quantize_rows(deq), ref.quantize_rows(x))
    print(f"ok: dequantized rows re-quantize identically (relative error {rel:.4f})")
    check_plumbing()
    print("PASS")


if __name__ == "__main__":
    main()
