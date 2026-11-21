# SPDX-License-Identifier: Apache-2.0
"""NVFP4 KV reader tests for the SM12x Triton attention kernel.

(a) `_load_nvfp4_kv` must reproduce `nvfp4_kv_ref.dequantize_rows` exactly.
(b) the real `_rel_attention_kernel` of
    `vllm/models/inkling2/nvidia/ops/attention_sm90.py`, launched through
    `rel_attention()` over a paged uint8 NVFP4 cache, must match a torch
    attention reference computed on the dequantized K/V.

Everything runs under the Triton CPU interpreter (TRITON_INTERPRET=1), so no
GPU is needed:

    uv run --no-project --with torch --with triton --with numpy \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple python tests/test_nvfp4_reader.py

The query dtype is float16 instead of production bfloat16 because Triton
3.9.0's CPU interpreter `tl.dot` returns garbage for bf16 operands (fp16 and
fp32 are fine); fp16 is exact for E2M1 x E4M3 values anyway.
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import importlib.util
import sys
import types

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = os.path.normpath(
    os.path.join(HERE, "..", "vllm", "models", "inkling2", "nvidia", "ops")
)
D = 128
ROW = D // 2 + D // 16  # 72 bytes: 64 packed E2M1 + 8 E4M3 scales
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# attention_sm90.py only pulls `tl, triton` out of vllm.triton_utils; stub that
# module so the file loads outside a full vLLM install (no GPU needed).
_vllm = types.ModuleType("vllm")
_triton_utils = types.ModuleType("vllm.triton_utils")
_triton_utils.tl = tl
_triton_utils.triton = triton
_vllm.triton_utils = _triton_utils
sys.modules.setdefault("vllm", _vllm)
sys.modules["vllm.triton_utils"] = _triton_utils

ref = _load_by_path("nvfp4_kv_ref", os.path.join(OPS, "nvfp4_kv_ref.py"))
attn = _load_by_path("attention_sm90", os.path.join(OPS, "attention_sm90.py"))
_load_nvfp4_kv = attn._load_nvfp4_kv


@triton.jit
def _probe_loader(
    KV, OFF, KEEP, OUT, DType: tl.constexpr, DIM: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Call `_load_nvfp4_kv` on gathered byte offsets, masking out KEEP == 0."""
    idx = tl.arange(0, BLOCK_N)
    offs = tl.load(OFF + idx).to(tl.int64)
    mask = tl.load(KEEP + idx) != 0
    tile = _load_nvfp4_kv(KV, offs, mask, DType, DIM)
    tl.store(OUT + idx[:, None] * DIM + tl.arange(0, DIM)[None, :], tile)


def _read_rows(rows, offsets, keep):
    """rows: flat uint8 cache; offsets: byte offsets of the rows to read."""
    n = len(offsets)
    block_n = max(16, triton.next_power_of_2(n))  # tl.arange wants a power of 2
    pad = [0] * (block_n - n)
    out = torch.zeros(block_n, D, dtype=torch.float32)
    _probe_loader[(1,)](
        rows,
        torch.tensor(list(offsets) + pad, dtype=torch.int32),
        torch.tensor(list(keep) + pad, dtype=torch.int32),
        out,
        DType=tl.float32,
        DIM=D,
        BLOCK_N=block_n,
    )
    return out[:n]


def _special_rows():
    """Rows covering every nibble code and the e4m3 scale extremes."""
    codes = torch.arange(16, dtype=torch.uint8).repeat(8)  # 128 nibbles = 64 B
    all_nibbles = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)

    def scaled(scale_byte, nibbles=None):
        scales = torch.full((D // 16,), scale_byte, dtype=torch.uint8)
        return torch.cat([all_nibbles if nibbles is None else nibbles, scales])

    return torch.stack(
        [
            scaled(0x38),  # scale 1.0
            scaled(0x00),  # zero scales
            scaled(0x01),  # smallest e4m3 subnormal
            scaled(0x7E),  # largest finite e4m3 (448)
            scaled(0x7E, torch.zeros(D // 2, dtype=torch.uint8)),  # zero nibbles
        ]
    )


def test_loader_matches_reference():
    """(a) _load_nvfp4_kv == nvfp4_kv_ref.dequantize_rows, bit exactly."""
    torch.manual_seed(0)
    cache = torch.cat(
        [ref.quantize_rows(torch.randn(24, D) * 4.0), _special_rows()], dim=0
    ).contiguous()
    assert cache.shape == (24 + 5, ROW)

    order = list(range(0, cache.shape[0], 2))[::-1]  # scattered, descending
    keep = [1] * len(order)
    keep[-1] = 0  # masked row must read as zeros
    got = _read_rows(cache, [i * ROW for i in order], keep)
    want = ref.dequantize_rows(cache[order[:-1]], D)
    assert torch.equal(got[:-1], want), "dequantized rows differ from the contract"
    assert torch.equal(got[-1], torch.zeros(D)), "masked row must load as zeros"
    print(
        f"ok loader: {len(order)} rows read ({len(order) - 1} compared, 1 masked), "
        "bit-exact vs dequantize_rows"
    )


def _deq(rows):
    """Whole paged cache -> float32 [num_blocks, page, kv_heads, D]."""
    nb, page, kvh, _ = rows.shape
    return ref.dequantize_rows(rows, D).reshape(nb, page, kvh, D).float()


def _make_cache(num_blocks, page, kv_heads, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(num_blocks, page, kv_heads, D, generator=g) * 2.0
    v = torch.randn(num_blocks, page, kv_heads, D, generator=g) * 2.0
    k[0, 0] = 0  # an all-zero row exercises the scale == 0 encoding
    return ref.quantize_rows(k), ref.quantize_rows(v), k, v


def _attention_reference(q, keys, values, softmax_scale):
    """Draft-mode (no rel bias, no window) attention for one request.

    `keys`/`values` are [n_keys, kv_heads, D] in logical key order; with DRAFT
    set the kernel puts every query at position seqlen_k - 1, so every query
    sees every key and the softmax is a plain row-wise softmax.
    """
    _, kvh, _ = keys.shape
    group = q.shape[1] // kvh
    kf = keys.repeat_interleave(group, dim=1).float()  # head = kv_head*G + g
    vf = values.repeat_interleave(group, dim=1).float()
    s = torch.einsum("thd,khd->thk", q.float(), kf) * softmax_scale
    m = s.amax(-1, keepdim=True)
    p = torch.exp2((s - m) * LOG2E)  # the kernel's log2 domain
    denom = p.sum(-1)
    # p is rounded to the query dtype before the PV dot, but summed in fp32.
    o = torch.einsum("thk,khd->thd", p.to(q.dtype).float(), vf) / denom[:, :, None]
    # (H, T) like the kernel's lse; m is natural here, the kernel's is log2.
    lse = (m.squeeze(-1) + torch.log2(denom) * LN2).transpose(0, 1)
    return o, lse


def _run_case(name, num_blocks, page, kv_heads, group, q_lens, bt_perm, seed):
    """One launch per request, checked against the torch reference twice over."""
    num_reqs = len(q_lens)
    per_req = num_blocks // num_reqs
    assert all(len(p) == per_req for p in bt_perm), "one page list per request"
    assert len({b for p in bt_perm for b in p}) == per_req * num_reqs, "pages overlap"
    kq, vq, k_raw, v_raw = _make_cache(num_blocks, page, kv_heads, seed)
    # `bt_perm` holds absolute, disjoint block ids, so pages are shuffled.
    block_table = torch.tensor(bt_perm, dtype=torch.int32)
    seqlens = [per_req * page] * num_reqs

    def logical(cache):  # paged -> [total_keys, kv_heads, D] in key order
        pairs = zip(block_table, seqlens, strict=True)
        pages = [cache[bt[: n // page]] for bt, n in pairs]
        return torch.cat(pages, dim=0).reshape(-1, kv_heads, D)

    keys_raw, vals_raw = logical(k_raw), logical(v_raw)
    keys_deq, vals_deq = logical(_deq(kq)), logical(_deq(vq))
    heads, scale = kv_heads * group, 1.0 / D**0.5

    starts = [0]
    for n in seqlens:
        starts.append(starts[-1] + n)
    for req in range(num_reqs):
        n_k, n_q = seqlens[req], q_lens[req]
        k0 = starts[req]
        gen = torch.Generator().manual_seed(seed + req)
        # fp16, not production bf16: the interpreter's tl.dot breaks on bf16.
        q = (torch.randn(n_q, heads, D, generator=gen) * 0.5).to(torch.float16)
        out = torch.empty(n_q, heads, D, dtype=torch.float32)
        lse = torch.empty(heads, n_q, dtype=torch.float32)
        attn.rel_attention(
            q,
            kq,
            vq,
            out,
            lse,
            None,  # rel_logits=None -> DRAFT mode, no relative bias
            block_table=block_table[req : req + 1],
            cache_seqlens=torch.tensor([n_k], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, n_q], dtype=torch.int32),
            max_query_len=n_q,
            softmax_scale=scale,
            window=0,
            num_splits=1,
            mode=attn.MODE_OUT_LSE,
        )
        want, want_lse = _attention_reference(
            q, keys_deq[k0 : k0 + n_k], vals_deq[k0 : k0 + n_k], scale
        )
        err = (out - want).abs().max().item()
        rel = ((out - want).norm() / want.norm()).item()
        lse_err = (lse - want_lse).abs().max().item()
        # Tight: what is left is fp16 rounding of the probabilities and the
        # order of the online-softmax rescaling.
        assert rel < 1e-3, f"{name} req{req}: {rel:.2e} vs dequantized reference"
        assert lse_err < 1e-4, f"{name} req{req}: lse {lse_err:.2e}"
        # Same reference on the *pre-quantization* K/V: reports the NVFP4 error.
        orig, _ = _attention_reference(
            q, keys_raw[k0 : k0 + n_k], vals_raw[k0 : k0 + n_k], scale
        )
        q_err = ((out - orig).norm() / orig.norm()).item()
        assert rel < 1e-2 * q_err, (
            f"{name} req{req}: reader {rel:.0e} vs quant {q_err:.0e}"
        )
        block_m = attn._block_m(n_q, group)
        print(
            f"ok {name} req{req}: q({n_q},{heads},{D}) keys={n_k} page={page} "
            f"BLOCK_M={block_m} | vs dequant max|d|={err:.2e} rel={rel:.2e} "
            f"lse|d|={lse_err:.2e} | vs pre-quant rel={q_err:.4f}"
        )


def test_end_to_end_paged_cache():
    """(b) kernel over a paged NVFP4 cache == torch attention on the same rows."""
    # 2 blocks x 16 slots, 1 page per request, 2 kv heads x group 2, 1 token.
    _run_case(
        "small", num_blocks=2, page=16, kv_heads=2, group=2,
        q_lens=[1, 1], bt_perm=[[0], [1]], seed=11,
    )
    # 6 blocks x 32 slots (3 pages = 96 keys per request), 1 kv head x group 4,
    # 32 query tokens -> BLOCK_M=128, BLOCK_N=64: two key iterations, each one
    # spanning two cache pages of the block table.
    _run_case(
        "multi", num_blocks=6, page=32, kv_heads=1, group=4,
        q_lens=[32, 32], bt_perm=[[1, 3, 5], [2, 4, 0]], seed=23,
    )
    print("ok end-to-end")


def test_kv_nvfp4_flag():
    """The launcher picks NVFP4 from the 72-byte row, never from a D-wide one."""
    assert ROW == 72 == D // 2 + D // 16
    torch.manual_seed(3)
    rows = ref.quantize_rows(torch.randn(2, 16, 1, D) * 2)
    assert rows.shape == (2, 16, 1, ROW) and rows.dtype is torch.uint8
    assert not (D == D // 2 + D // 16), "an FP8/BF16 row must not look like NVFP4"
    print("ok flag: shape[-1] == D//2 + D//16 detects NVFP4, D does not")


if __name__ == "__main__":
    test_loader_matches_reference()
    test_kv_nvfp4_flag()
    test_end_to_end_paged_cache()
    print("ALL TESTS PASSED")
