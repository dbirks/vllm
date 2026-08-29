# syntax=docker/dockerfile:1
#
# vllm-flashnext:sm120 — thin overlay for serving Qwen3.8-Flash-Next on RTX PRO 6000 (SM120).
#
# This is NOT a vLLM source build. It layers primitive-ai's INT4 PLE-quant overlay
# on top of the prebuilt base image by copying two Python files into site-packages.
# See README.md for the full story and the runtime env vars required at deploy time.

FROM vllm/vllm-openai:qwen38-flash-next

# INT4 PLE CPU-offload worker (mmap'd quant overlay) and the quantized PLE layer.
# Source: https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant
COPY worker_image_quant.py /usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py
COPY ple_layer_quant.py    /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
