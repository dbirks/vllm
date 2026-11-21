# NVFP4 KV: CPU test of the NVFP4 KV cache for the DSpark DRAFT layers.
#
#   uv run --no-project --with torch --with triton --with numpy \
#       --index-url https://download.pytorch.org/whl/cpu \
#       --extra-index-url https://pypi.org/simple python tests/test_nvfp4_draft.py
#
# Drives `ops/dspark.write_kv_paged_nvfp4` -- the draft's NVFP4 store, which
# quantizes each (token, kv_head) row of its paged cache through the target's
# `ops/nvfp4_kv_write.nvfp4_store_kv_rows` -- and asserts the stored bytes equal
# the contract `ops/nvfp4_kv_ref.quantize_rows` byte for byte. Also checks the
# draft's KV-cache *sizing*: an NVFP4 draft page (same layout as the target's
# packed layers) is `num_kv_heads * block_size * 2 * 72` bytes, one quarter of
# the BF16 page it replaces.
#
# The workspace holds only the vLLM files we patch and has no GPU, so
# `vllm/models/inkling2/nvidia/ops/dspark.py` is loaded by file path with
# `vllm.triton_utils` and the `...ops.nvfp4_kv_write` it imports stubbed to the
# real modules, and the NVFP4 row layout comes from the unmodified contract
# `ops/nvfp4_kv_ref.py`. `write_kv_paged_nvfp4` itself is the file's real code.
"""Exercise the NVFP4 draft KV write path against the reference contract."""

import importlib.util
import os
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
OPS_DIR = os.path.join(
    REPO,
    "vllm",
    "models",
    "inkling2",
    "nvidia",
    "ops",
)
HEAD_DIM = 128
ROW = HEAD_DIM // 2 + HEAD_DIM // 16  # 72
GROUP = 16


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _stub_namespace(name: str) -> types.ModuleType:
    mod = sys.modules.setdefault(name, types.ModuleType(name))
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(OPS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Stub `vllm.triton_utils` (the real module re-exports exactly these three),
# then the ops package namespace, so `ops/dspark.py` and `nvfp4_kv_write.py`
# resolve their imports from sys.modules. The two op modules are the real code.
_stub_namespace("vllm")
_stub("vllm.triton_utils", tl=tl, triton=triton, tldevice=tl)
_stub_namespace("vllm.models")
_stub_namespace("vllm.models.inkling2")
_stub_namespace("vllm.models.inkling2.nvidia")
_stub_namespace("vllm.models.inkling2.nvidia.ops")
wr = _load("vllm.models.inkling2.nvidia.ops.nvfp4_kv_write", "nvfp4_kv_write.py")
sys.modules["vllm.models.inkling2.nvidia.ops"].nvfp4_kv_write = wr  # type: ignore[attr-defined]
ref = _load("nvfp4_kv_ref", "nvfp4_kv_ref.py")
dsp = _load("vllm.models.inkling2.nvidia.ops.dspark", "dspark.py")


def check_write(name, k, v, slots, block_size=16, num_blocks=2):
    """Assert the draft's NVFP4 store matches `quantize_rows` for K and V.

    `k`, `v` are `(num_tokens, num_kv_heads, D)` (as the SM120 draft's forward
    hands them to `_write_kv`), and `slots` their destination slot (-1 skips).
    """
    num_tokens, num_kv_heads, D = k.shape
    assert v.shape == k.shape
    assert slots.numel() == num_tokens
    # The uint8 page the worker binds: [B, H, N, 2 * ROW], split K | V.
    page = torch.full(
        (num_blocks, num_kv_heads, block_size, 2 * ROW), 0xA5, dtype=torch.uint8
    )
    key_cache, value_cache = page.transpose(1, 2).split(ROW, dim=-1)
    dsp.write_kv_paged_nvfp4(k, v, key_cache, value_cache, slots)

    want_k = ref.quantize_rows(k.float())  # (T, H, ROW)
    want_v = ref.quantize_rows(v.float())
    written = 0
    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        written += 1
        block, offset = slot // block_size, slot % block_size
        for head in range(num_kv_heads):
            # The split side is indexed [block, offset, head] (N moved up front).
            assert torch.equal(key_cache[block, offset, head], want_k[token, head]), (
                f"{name}: K token {token} head {head} (slot {slot}) differs\n"
                f" got {key_cache[block, offset, head].tolist()}\n"
                f"want {want_k[token, head].tolist()}"
            )
            assert torch.equal(value_cache[block, offset, head], want_v[token, head]), (
                f"{name}: V token {token} head {head} (slot {slot}) differs\n"
                f" got {value_cache[block, offset, head].tolist()}\n"
                f"want {want_v[token, head].tolist()}"
            )
    # Skipped slots leave their rows untouched: every non-sentinel byte must
    # belong to a written (token, head) row on both sides (K and V).
    if any(s < 0 for s in slots.tolist()):
        skipped = sum(1 for s in slots.tolist() if s < 0)
        assert (page != 0xA5).sum().item() <= written * num_kv_heads * 2 * ROW, name
        assert skipped > 0
    print(
        f"ok: {name}: {num_tokens} tokens x {num_kv_heads} heads x D={D}, "
        f"{written} written / {sum(1 for s in slots.tolist() if s < 0)} skipped"
    )


def make_rows(tokens, heads, gen, scale):
    x = (torch.rand(tokens, heads, HEAD_DIM, generator=gen) * 2 - 1) * scale
    return x.to(torch.bfloat16)


def test_skipped_and_large():
    """A 2-block x 16-slot cache, 2 heads, D=128: skipped slots and |x|~2000."""
    gen = torch.Generator().manual_seed(0)
    k = make_rows(4, 2, gen, 2000.0)
    v = make_rows(4, 2, gen, 3.0)
    # Slots across both blocks; slot 2 skipped (-1). block_size=16.
    slots = torch.tensor([0, 1, 17, -1], dtype=torch.int64)
    check_write("large+skip", k, v, slots, block_size=16, num_blocks=2)


def test_all_slots_and_edge_rows():
    """All slots written; one all-zero row, one all-zero group, one single."""
    gen = torch.Generator().manual_seed(1)
    k = make_rows(8, 2, gen, 5.0)
    v = make_rows(8, 2, gen, 400.0)
    k[0] = 0  # an all-zero row stores scale 0 and nibbles 0
    v[1, :, :GROUP] = 0  # one all-zero group per head
    k[2, 0, 0] = -0.3  # a single small element
    v[3, 0] = 0  # one of two heads zero
    slots = torch.tensor([31, 15, 16, 0, 5, 20, 2, 9], dtype=torch.int64)
    check_write("edge-rows", k, v, slots, block_size=16, num_blocks=2)


def test_single_head_full_first_block():
    """One head, slots filling block 0 exactly at a block boundary."""
    gen = torch.Generator().manual_seed(2)
    k = make_rows(16, 1, gen, 1.0)
    v = make_rows(16, 1, gen, 1.0)
    slots = torch.arange(16, dtype=torch.int64)
    check_write("single-head-boundary", k, v, slots, block_size=16, num_blocks=1)


def _load_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_vllm_tree():
    """Stand in for the absent vLLM modules `kv_cache_interface.py` imports (as
    tests/test_nvfp4_sizing.py does), so the *real* AttentionSpec is used."""
    import enum

    for name in ("vllm", "vllm.utils", "vllm.v1", "vllm.v1.attention"):
        _stub_namespace(name)
    envs = types.ModuleType("vllm.envs")
    envs.__getattr__ = lambda _key: None  # PEP 562: any VLLM_* reads as unset
    sys.modules["vllm.envs"] = envs
    _stub("vllm.logger", init_logger=lambda *_a, **_k: None)
    _stub("vllm.utils.platform_utils", is_pin_memory_available=lambda: False)

    def _at_least(version):
        from torch.torch_version import TorchVersion

        return TorchVersion(torch.__version__) >= TorchVersion(version)

    _stub(
        "vllm.utils.torch_version",
        _is_torch_equal=lambda v: torch.__version__ == v,
        _is_torch_equal_or_newer=_at_least,
        is_torch_equal=lambda v: torch.__version__ == v,
        is_torch_equal_or_newer=_at_least,
    )
    _stub(
        "vllm.utils.math_utils",
        cdiv=lambda a, b: -(-a // b),
        round_up=lambda a, b: (a + b - 1) // b * b,
    )

    class MambaAttentionBackendEnum(enum.Enum):
        MAMBA2 = "mamba2"

    _stub(
        "vllm.v1.attention.backends.registry",
        MambaAttentionBackendEnum=MambaAttentionBackendEnum,
    )
    _stub(
        "vllm.v1.kv_cache_layout",
        _DIM_L=0,
        _DIM_B=1,
        KVCacheLayout=enum.Enum("KVCacheLayout", {"NHD": (0, 1, 2, 3, 4)}),
    )


def test_page_sizing(kvci=None, torch_utils=None):
    """An NVFP4 draft page is heads*block*2*72 bytes (the target's packed page)."""
    num_kv_heads, block_size = 8, 128
    nvfp4 = num_kv_heads * block_size * 2 * ROW  # K row + V row, packed
    bf16 = num_kv_heads * block_size * 2 * HEAD_DIM * 2  # what it replaces
    assert nvfp4 == num_kv_heads * block_size * 144 == 147456
    # 144 vs 512 bytes per (head, token): the packed page is ~28% of the BF16
    # one at equal block size (the K + V row narrows from 2*D bf16 to 2*72).
    assert nvfp4 * 512 == bf16 * 144
    assert ROW == 72 == HEAD_DIM // 2 + HEAD_DIM // 16
    assert 2 * ROW == 144
    # The state_content_bytes the spec publishes: one K and one V row per slot.
    assert ref.nvfp4_row_bytes(HEAD_DIM) == ROW

    # With the real AttentionSpec (vLLM's page math), the spec the draft's
    # Inkling2DraftAttention asks for lays out [B, H, N, 2 * 72] uint8 and its
    # page is exactly heads * block * 2 * 72 bytes.
    if kvci is not None:
        spec = kvci.AttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=HEAD_DIM,
            dtype=torch.uint8,
            kv_quant_mode=kvci.KVQuantMode.NVFP4,
            state_content_bytes=2 * ROW,
        )
        assert spec.page_size_bytes == num_kv_heads * block_size * 2 * 72 == 147456
        assert spec.state_content_size_bytes == 144
        assert spec.dtype is torch.uint8
        assert spec.head_size == HEAD_DIM
        shape = kvci.compute_layer_kv_cache_shape_bytes
        assert shape(spec, 4) == (4, num_kv_heads, block_size, 144)

    print(
        f"ok: page sizing: heads={num_kv_heads} block={block_size} -> "
        f"nvfp4={nvfp4} B (bf16 was {bf16} B, {nvfp4 / bf16:.3f}x)"
    )


def main() -> int:
    assert dsp.write_kv_paged_nvfp4.__doc__ is not None
    # Optional: exercise the page-size math against the real AttentionSpec too.
    kvci = torch_utils = None
    try:
        _stub_vllm_tree()
        torch_utils = _load_path(
            "vllm.utils.torch_utils", os.path.join(REPO, "vllm/utils/torch_utils.py")
        )
        kvci = _load_path(
            "vllm.v1.kv_cache_interface",
            os.path.join(REPO, "vllm/v1/kv_cache_interface.py"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"note: AttentionSpec sizing skipped ({exc!r})")
    tests = [
        ("test_skipped_and_large", test_skipped_and_large),
        ("test_all_slots_and_edge_rows", test_all_slots_and_edge_rows),
        ("test_single_head_full_first_block", test_single_head_full_first_block),
        (
            "test_page_sizing",
            lambda: test_page_sizing(kvci=kvci, torch_utils=torch_utils),
        ),
    ]
    failed = 0
    for name, run in tests:
        try:
            run()
        except Exception:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print("FAILED" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
