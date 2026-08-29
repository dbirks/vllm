# vllm-flashnext:sm120

Prebuilt image for serving **Qwen3.8-Flash-Next** (180B/6B MoE + 51B N-gram PLE) on a
single **RTX PRO 6000 (SM120 / Blackwell)**.

## What this image is

A **thin overlay**, not a vLLM source build. It starts from the prebuilt base image
`vllm/vllm-openai:qwen38-flash-next` and copies **two Python files** into vLLM's
installed `site-packages`:

| Overlay file (in this dir) | Destination in the image |
|---|---|
| `worker_image_quant.py` | `vllm/v1/ple_offload/worker.py` |
| `ple_layer_quant.py`    | `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` |

These two files are **primitive-ai's INT4 PLE-quant overlay**, vendored verbatim from
<https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant> (`worker_image_quant.py`,
`ple_layer_quant.py`). They let the ~51B N-gram PLE be served as an INT4 overlay
mmap'd into host RAM (RSS ~33GB, reclaimable) and CPU-offloaded, instead of the
95GB BF16 PLE — which is what makes the model fit alongside the mixed-NVFP4/FP8
MoE weights on the 46GB (no-swap) node. The files are **not modified**: our runtime
fixes (model-index ple-bf16 stripping, `CUDA_DISABLE_CONTROL`) are applied on the
K8s-manifest / model-data side, not baked into the image.

Base image is multi-arch; we build/push **linux/amd64** only.

Published as: `ghcr.io/dbirks/vllm-flashnext:sm120`

## Building

Built on GitHub-hosted runners via `.github/workflows/flashnext-sm120.yml`
(manual `workflow_dispatch`, input `tag`, default `sm120`). Do **not** build locally —
the base image is ~15GB.

```
gh workflow run flashnext-sm120.yml --repo dbirks/vllm --ref flashnext-sm120 -f tag=sm120
```

## Runtime env vars required at deploy

The image alone is not enough; the serving Deployment must set:

| Env var | Value | Why |
|---|---|---|
| `VLLM_PLE_QUANT_DIR` | path to the INT4 PLE-quant overlay dir | tells the overlay where the mmap'd INT4 PLE shards live |
| `VLLM_PLE_CPU_OFFLOAD` | `1` | keep the PLE on CPU, off the GPU |
| `VLLM_GDN_DECODE_KERNEL` | `triton` | GatedDeltaNet decode kernel path that works on SM120 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | PLE offload spawns a child process |
| `MAX_JOBS` | `1` | limit build/compile parallelism (host RAM ceiling) |
| `CUDA_DISABLE_CONTROL` | `true` | CUDA IPC deadlocks under HAMi vGPU without this |

**Model index caveat:** if the `ple-bf16` shards are **not** present in the model
directory, the model index (`*.safetensors.index.json` / weight map) must have its
`ple-bf16` references removed, or loading fails looking for missing shards. This is a
data-side fix, done on the mounted model, not in the image.

## Provenance / license

The two overlay files are authored by primitive-ai (Apache-2.0 headers retained) and
vendored here unmodified solely to bake them into our own serving image. Upstream:
<https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant>.
