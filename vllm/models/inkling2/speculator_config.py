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
            # The target's clipped, unit-scale FP8 cache does not describe the
            # draft's unnormalized values. Keep the draft cache in BF16.
            #
            # NVFP4 KV: this reads only the *draft's* own dtype
            # (`SpeculativeConfig.kv_cache_dtype`, set by the user's
            # `--speculative-kv-cache-dtype`), never
            # `vllm_config.cache_config.cache_dtype`, so a target running with
            # `--kv-cache-dtype nvfp4` leaves the draft at auto/bfloat16 and
            # passes here unchanged. A draft that explicitly asks for nvfp4 is
            # still rejected: the draft has no NVFP4 kernels.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "auto"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):
                raise ValueError(
                    "Inkling2 DSpark on SM12 requires BF16 draft KV (the "
                    "target's --kv-cache-dtype does not apply to the draft)"
                )
