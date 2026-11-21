# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2.speculator.Inkling2DSparkSpeculator"

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        super().adapt_draft_model_config(spec)
        if current_platform.is_device_capability_family(120):
            # NVFP4 KV: the draft's cache dtype is read only from this draft's
            # own `SpeculativeConfig.kv_cache_dtype` (`--speculative-config
            # '{..., "kv_cache_dtype": "nvfp4"}'`), never from
            # `vllm_config.cache_config.cache_dtype`, so a target running with
            # `--kv-cache-dtype nvfp4` leaves the draft at auto/bfloat16 and
            # passes here unchanged. The draft now has NVFP4 kernels (see
            # nvidia/dspark.py + ops/dspark.py): its cache uses the same packed
            # fp4 + per-16-element E4M3-scale row format as the target's NVFP4
            # layers, whose scales (unlike the target's unit-scale FP8) can
            # describe the draft's unnormalized values -- max |x| = 448 * 6 =
            # 2688. Accept `nvfp4` alongside auto/bfloat16; reject the rest.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "auto"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16", "nvfp4"):
                raise ValueError(
                    "Inkling2 DSpark on SM12 requires BF16 or NVFP4 draft KV "
                    "(the target's --kv-cache-dtype does not apply to the draft)"
                )
