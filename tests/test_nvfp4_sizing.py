# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 KV: CPU sizing + split test for Inkling2 on SM12x.

Runs on CPU and without an installed vLLM. This workspace holds only the vLLM
files we patch, so `vllm/models/inkling2/nvidia/attention.py` cannot be
imported as a module (it pulls in `vllm.compilation`, `vllm.distributed`,
tilelang, ...). The two functions it owns for this task are therefore lifted
out of the file with `ast` and executed against the *real* `AttentionSpec`
(`vllm/v1/kv_cache_interface.py`) and the *real* `vllm/utils/torch_utils.py`
helpers, both loaded from their paths with the handful of absent vLLM modules
stubbed. Nothing here re-implements the code under test: the byte-layout
reference `ops/nvfp4_kv_ref.py` is the contract, loaded unmodified.

Run:
  uv run --no-project --with torch --with triton --with numpy \
     --index-url https://download.pytorch.org/whl/cpu \
     --extra-index-url https://pypi.org/simple python tests/test_nvfp4_sizing.py
"""

import ast
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"  # project rule (this test needs no triton)

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# loading the real code without an installed vLLM
# ---------------------------------------------------------------------------


def _stub(name: str, **attrs) -> types.ModuleType:
    """Register a stand-in module (for vLLM files absent from this repo)."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules.setdefault(parent, types.ModuleType(parent)), child, mod)
    return mod


def _load(name: str, path: Path) -> types.ModuleType:
    """Execute `path` as module `name`; its imports resolve via sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules.setdefault(parent, types.ModuleType(parent)), child, mod)
    spec.loader.exec_module(mod)
    return mod


def _lift(path: Path, func: str, namespace: dict):
    """Compile one function/method out of `path` into `namespace` and return it.

    The method keeps its `self` parameter; callers pass a stand-in `self`.
    """
    tree = ast.parse(path.read_text())
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func
    ]
    assert len(found) == 1, f"{func}: {len(found)} definitions in {path}"
    module = ast.Module(body=[found[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[func]


def _envs_stub(name: str) -> types.ModuleType:
    """`vllm.envs`: any VLLM_* reads as unset (None) in this workspace."""
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda _key: None  # PEP 562
    sys.modules[name] = mod
    return mod


def _stub_vllm_tree() -> None:
    """The vLLM modules that `torch_utils.py` / `kv_cache_interface.py` import
    but that this scratch workspace does not contain."""
    for name in ("vllm", "vllm.utils", "vllm.v1", "vllm.v1.attention"):
        sys.modules.setdefault(name, types.ModuleType(name))
    _envs_stub("vllm.envs")
    _stub("vllm.logger", init_logger=lambda *_a, **_k: None)
    _stub(
        "vllm.utils.platform_utils",
        is_pin_memory_available=lambda: False,
    )

    def _torch_version_at_least(version: str) -> bool:
        from torch.torch_version import TorchVersion

        return TorchVersion(torch.__version__) >= TorchVersion(version)

    _stub(
        "vllm.utils.torch_version",
        _is_torch_equal=lambda v: torch.__version__ == v,
        _is_torch_equal_or_newer=_torch_version_at_least,
        is_torch_equal=lambda v: torch.__version__ == v,
        is_torch_equal_or_newer=_torch_version_at_least,
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


def _make_managers():
    """Stand-ins for the two manager classes (their module is not in this
    workspace). They record the spec they were handed and return a config with
    `.spec`, like `KVCacheLayerConfig`, so the test also checks that the layer
    still registers with the manager it did before."""
    calls: list[str] = []

    class _Cfg:
        def __init__(self, kind, spec):
            self.kind, self.spec = kind, spec

    class FullAttentionManager:
        @staticmethod
        def layer_config(spec, **_kw):
            calls.append("full")
            return _Cfg("full", spec)

    class SlidingWindowManager:
        @staticmethod
        def layer_config(spec, **_kw):
            calls.append("swa")
            return _Cfg("swa", spec)

    return FullAttentionManager, SlidingWindowManager, calls


# ---------------------------------------------------------------------------
# tests (helpers are injected by main(); the module has no vllm import)
# ---------------------------------------------------------------------------


def test_row_contract(ref):
    """One NVFP4 row is 64 packed fp4 bytes + 8 fp8 scales at D=128."""
    assert ref.nvfp4_row_bytes(HEAD_DIM) == 72 == HEAD_DIM // 2 + HEAD_DIM // 16
    assert 2 * ref.nvfp4_row_bytes(HEAD_DIM) == 144  # K row + V row per slot


def test_page_sizing(kv_layer_config, kvci, calls):
    """nvfp4 pages hold 72 bytes per K/V row; the fp8/bf16 page is unchanged."""
    block_size, num_kv_heads = 16, 8

    def spec_for(dtype_str, blocks=block_size):
        calls.clear()
        cfg = kv_layer_config(
            SimpleNamespace(
                cache_config=SimpleNamespace(
                    block_size=blocks, cache_dtype=dtype_str
                ),
                model_config=None,
            ),
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            sliding_window=None,
        )
        assert calls == ["full"], calls  # still a FullAttentionManager layer
        return cfg.spec

    nvfp4, fp8, bf16 = (spec_for(d) for d in ("nvfp4", "fp8", "bfloat16"))

    assert nvfp4.page_size_bytes == num_kv_heads * block_size * 2 * 72 == 18432
    assert fp8.page_size_bytes == num_kv_heads * block_size * 2 * 128 == 32768
    assert bf16.page_size_bytes == num_kv_heads * block_size * 2 * 128 * 2 == 65536
    # 72/128 of the fp8 page, at every block size.
    assert spec_for("nvfp4", 8).page_size_bytes == 9216
    assert spec_for("fp8", 8).page_size_bytes == 16384
    # uint8 storage, logical head_size unchanged, packed state width published.
    assert nvfp4.dtype is torch.uint8 and nvfp4.head_size == HEAD_DIM
    assert nvfp4.kv_quant_mode is kvci.KVQuantMode.NVFP4
    assert nvfp4.state_content_size_bytes == 2 * 72
    assert fp8.state_content_size_bytes == 2 * 128
    # The fp8 spec is untouched: it keeps the default quant mode that the FP8
    # kernels dispatch on today, and derives its width from head_size.
    assert fp8.kv_quant_mode is kvci.KVQuantMode.NONE
    assert fp8.state_content_bytes is None
    assert bf16.kv_quant_mode is kvci.KVQuantMode.NONE
    assert bf16.dtype is torch.bfloat16
    # The per-layer allocation the spec asks the worker for: bytes of
    # [blocks, kv heads, tokens per block, K row + V row].
    shape = kvci.compute_layer_kv_cache_shape_bytes
    assert shape(nvfp4, 4) == (4, num_kv_heads, block_size, 144)
    assert shape(fp8, 4) == (4, num_kv_heads, block_size, 256)

    # A sliding-window layer with the same cache still registers as such.
    calls.clear()
    cfg = kv_layer_config(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=block_size, cache_dtype="nvfp4"),
            model_config=None,
        ),
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        sliding_window=2048,
    )
    assert calls == ["swa"], calls
    assert cfg.spec.page_size_bytes == 18432


def test_split_nvfp4(split, ref):
    """The uint8 page splits at 72 bytes and round-trips through the contract."""
    blocks, heads, page = 3, 8, 16
    torch.manual_seed(0)
    k_src = torch.randn(blocks, heads, page, HEAD_DIM) * 3.0
    v_src = torch.randn(blocks, heads, page, HEAD_DIM) * 3.0
    k_src[0, 0, 0] = 0  # an all-zero group stores scale 0 and nibbles 0
    v_src[1, 2, 3] = 0

    k_rows = ref.quantize_rows(k_src)  # (B, H, N, 72) uint8, contract layout
    v_rows = ref.quantize_rows(v_src)
    assert k_rows.shape == (blocks, heads, page, 72)
    # What the worker hands the layer as `kv_cache`: [B, H, N, 2 * 72] uint8.
    kv_cache = torch.cat([k_rows, v_rows], dim=-1).contiguous()

    self = SimpleNamespace(kv_mode="nvfp4", head_dim=HEAD_DIM)
    key_cache, value_cache = split(self, kv_cache)
    assert key_cache.dtype is torch.uint8 and value_cache.dtype is torch.uint8
    assert key_cache.shape == value_cache.shape == (blocks, page, heads, 72)
    # Bytes survive the split exactly (K first, V second, no copy).
    assert torch.equal(key_cache.transpose(1, 2), k_rows)
    assert torch.equal(value_cache.transpose(1, 2), v_rows)
    # What the launchers assert about a KV side: equal strides, unit last one.
    assert key_cache.stride() == value_cache.stride()
    assert key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    assert key_cache.storage_offset() == 0 and value_cache.storage_offset() == 72
    assert key_cache.data_ptr() + 72 == value_cache.data_ptr()

    # Dequantizing the split cache matches dequantizing the source rows and
    # requantizing is idempotent, i.e. the cache really is the contract format.
    for cache, rows in ((key_cache, k_rows), (value_cache, v_rows)):
        # `rows` are [B, H, N, *]; the split sides are [B, N, H, *].
        cache = cache.transpose(1, 2)
        deq = ref.dequantize_rows(cache, HEAD_DIM)
        assert deq.shape == ref.dequantize_rows(rows, HEAD_DIM).shape
        assert torch.equal(deq, ref.dequantize_rows(rows, HEAD_DIM))
        assert torch.equal(ref.quantize_rows(deq), rows)
    assert ref.dequantize_rows(key_cache, HEAD_DIM)[0, 0, 0].abs().sum() == 0


def test_split_dense_paths(split):
    """bf16 and fp8 still split at head_dim with their original dtypes."""
    blocks, heads, page = 2, 8, 16
    torch.manual_seed(1)

    bf16 = torch.randn(blocks, heads, page, 2 * HEAD_DIM, dtype=torch.bfloat16)
    key_cache, value_cache = split(
        SimpleNamespace(kv_mode="bf16", head_dim=HEAD_DIM), bf16
    )
    assert key_cache.dtype is torch.bfloat16
    assert key_cache.shape == (blocks, page, heads, HEAD_DIM)
    assert torch.equal(key_cache.transpose(1, 2), bf16[..., :HEAD_DIM])
    assert torch.equal(value_cache.transpose(1, 2), bf16[..., HEAD_DIM:])

    fp8_bytes = torch.randint(
        0, 256, (blocks, heads, page, 2 * HEAD_DIM), dtype=torch.uint8
    )
    key_cache, value_cache = split(
        SimpleNamespace(kv_mode="fp8", head_dim=HEAD_DIM), fp8_bytes
    )
    assert key_cache.dtype is torch.float8_e4m3fn
    assert key_cache.shape == (blocks, page, heads, HEAD_DIM)
    assert torch.equal(
        key_cache.view(torch.uint8).transpose(1, 2), fp8_bytes[..., :HEAD_DIM]
    )
    assert torch.equal(
        value_cache.view(torch.uint8).transpose(1, 2), fp8_bytes[..., HEAD_DIM:]
    )


def main() -> int:
    _stub_vllm_tree()
    ref = _load(
        "nvfp4_kv_ref", REPO / "vllm/models/inkling2/nvidia/ops/nvfp4_kv_ref.py"
    )
    torch_utils = _load("vllm.utils.torch_utils", REPO / "vllm/utils/torch_utils.py")
    kvci = _load("vllm.v1.kv_cache_interface", REPO / "vllm/v1/kv_cache_interface.py")
    assert kvci.KVQuantMode.NVFP4 == 5
    assert torch_utils.nvfp4_kv_cache_full_dim(HEAD_DIM) == 72
    assert torch_utils.kv_cache_dtype_str_to_dtype("nvfp4", None) is torch.uint8

    attention_py = REPO / "vllm/models/inkling2/nvidia/attention.py"
    full, swa, calls = _make_managers()
    kv_layer_config = _lift(
        attention_py,
        "_kv_cache_layer_config",
        {
            "AttentionSpec": kvci.AttentionSpec,
            "KVCacheLayerConfig": kvci.KVCacheLayerConfig,
            "KVQuantMode": kvci.KVQuantMode,
            "FullAttentionManager": full,
            "SlidingWindowManager": swa,
            "VllmConfig": object,
            "kv_cache_dtype_str_to_dtype": torch_utils.kv_cache_dtype_str_to_dtype,
            "get_dtype_size": torch_utils.get_dtype_size,
            "nvfp4_kv_cache_full_dim": torch_utils.nvfp4_kv_cache_full_dim,
        },
    )
    split = _lift(
        attention_py,
        "_split_kv_cache",
        {
            "torch": torch,
            "nvfp4_kv_cache_full_dim": torch_utils.nvfp4_kv_cache_full_dim,
            "canonicalize_singleton_dim_strides": (
                torch_utils.canonicalize_singleton_dim_strides
            ),
        },
    )

    tests = [
        ("test_row_contract", lambda: test_row_contract(ref)),
        ("test_page_sizing", lambda: test_page_sizing(kv_layer_config, kvci, calls)),
        ("test_split_nvfp4", lambda: test_split_nvfp4(split, ref)),
        ("test_split_dense_paths", lambda: test_split_dense_paths(split)),
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
