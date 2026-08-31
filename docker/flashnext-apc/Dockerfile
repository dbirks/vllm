# syntax=docker/dockerfile:1
#
# vllm-flashnext:apc-a — hybrid-cache (APC) experiment overlay for Qwen3.8-Flash-Next on SM120.
#
# Candidate A for home-k8s issue #115 (prefix_cache_hits_total = 0 on qwen4_exp).
# This is the PRODUCTION sm120 overlay (Flash-Next base runtime + INT4 PLE offload)
# with two reviewed-but-unmerged upstream vLLM hybrid-cache correctness patches applied
# to the installed vLLM in site-packages at build time:
#
#   #53798  vllm/v1/worker/gpu/model_states/mamba_hybrid.py
#           seed align-mode mamba state_idx in mamba block units (not cache_config.block_size)
#   #54076  vllm/v1/core/sched/scheduler.py
#           align-mode chunk split stops on the Mamba group's own block size (MambaSpec.block_size)
#
# Built FROM the pinned production overlay DIGEST so this image differs from prod by ONLY
# these two patches -> a clean A/B for the off-prod qualification in apps/flashnext-apc-test.yaml.hold.
# NOT a vLLM source build; NOT copied from the fork tree. apply-patches.sh fails the build hard
# (strict, --fuzz=0) if the base has drifted, dumping the base context to re-anchor the patch.
#
# See PATCHES.md for provenance.

FROM ghcr.io/dbirks/vllm-flashnext:sm120@sha256:f7186f70baf43061c042dfc8e9b296b75476f287b7c8d8394a7d2e287ed61e82

USER root

COPY patches/ /tmp/patches/
COPY apply-patches.sh /tmp/apply-patches.sh

RUN bash /tmp/apply-patches.sh && rm -rf /tmp/patches /tmp/apply-patches.sh

LABEL org.opencontainers.image.description="Qwen3.8-Flash-Next SM120 overlay + APC candidate A (vLLM #53798 + #54076)"
