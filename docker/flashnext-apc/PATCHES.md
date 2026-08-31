# APC candidate A — patch provenance

Image: `ghcr.io/dbirks/vllm-flashnext:apc-a` (immutable: `:apc-a-<git-sha>`)
Base:  `ghcr.io/dbirks/vllm-flashnext:sm120@sha256:f7186f70baf43061c042dfc8e9b296b75476f287b7c8d8394a7d2e287ed61e82`
       (= Flash-Next base runtime `vLLM 0.1.dev20073+g8e685d198` + INT4 PLE overlay)

Goal: get non-zero `prefix_cache_hits_total` on qwen4_exp (Qwen3.8-Flash-Next) by
fixing two known hybrid-`align`-mode cache invariants.

IMPORTANT — these patches are RE-ANCHORED against the base image's ACTUAL code, not
taken verbatim from the PRs. The base runtime (`vLLM 0.1.dev20073+g8e685d198`, commit
`8e685d198` which is NOT on upstream main) ships a customized Flash-Next
`mamba_hybrid.py` / `scheduler.py`, so the upstream PR diffs do not apply as-is. Each
patch re-expresses the PR's semantic fix against the base's real lines; applied
strictly with `--fuzz=0` at build time. To regenerate, re-run the base-file dump
workflow and re-diff (see below).

| Patch file | Upstream PR | Ported | Target module | What it fixes |
|---|---|---|---|---|
| `patches/53798-mamba-align-seed.patch` | vllm-project/vllm#53798 (OPEN) | full | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | Worker seeds the resumed align-mode recurrent-state block using the Mamba group's block size (`_mamba_block_size`, captured in `_get_mamba_group_info`), not the generic `cache_config.block_size`; a finer divisor points the precopy source outside the request's block-table row. Adds a fail-fast assert if a request is admitted with computed tokens before the mamba block size resolves. |
| `patches/54076-mamba-align-chunk-split.patch` | vllm-project/vllm#54076 (OPEN) | hunks 1+2 only | `vllm/v1/core/sched/scheduler.py` | Scheduler `_mamba_block_aligned_split` clips chunks on `MambaSpec.block_size` (the grid the worker actually checkpoints), not the group-minimum `cache_config.block_size`. `__init__` computes `self.mamba_state_block_size` from the single Mamba group; the split uses it with a `cache_config.block_size` fallback. |

**#54076 hunk 3 is deliberately OMITTED.** The PR's third hunk rewrites the `stops`
tuple in `_mamba_block_aligned_split` around a `use_internal_checkpoint` flag. The base
image's variant of this method has no internal-checkpoint path, so that hunk is not
portable and not applicable here. Only the headline block-size-source fix (hunks 1+2,
the piece that maps to home-k8s #115's 0-hit symptom) is ported. Revisit hunk 3 only if
Candidate A shows reuse but leaves interior state slots null on multi-block chunks.

Both PRs target the same hybrid-`align` prefix-cache path flagged by upstream #45238
(align veto -> 0% hits on multi-turn), the exact symptom in home-k8s #115.

## Deliberately excluded
- MTP/EAGLE align fixes (#48375, #50897): the Flash-Next Deployment does not pass
  `--speculative-config`, so MTP is inactive. Excluded to keep attribution clean.
- Candidate B (#53479, direct state materialization for diverged/growing prefixes):
  only built if A is stable but reuse stays weak. #53479 and #54076 both touch
  `_mamba_block_aligned_split`, so B must be a reviewed local merge, not blind stacking.

## Re-vendoring (when the base image digest changes)
Because these are re-anchored (not verbatim PR diffs), regenerate them against the new
base's real files rather than re-running `gh pr diff`:
1. Run the `flashnext-base-dump` workflow (pins the base digest, uploads `mamba_hybrid.py`
   + `scheduler.py` as the `base-files` artifact); `gh run download ... -n base-files`.
2. Apply the same semantic edits (see the table above and #53798 / #54076 for intent) to
   copies of those files, `python3 -m py_compile` them, and
   `diff -u --label a/<path> --label b/<path> orig edit > patches/<name>.patch`.
3. Dry-run strictly against the base files (`patch -p1 --fuzz=0 --dry-run`) before building.

The build itself fails hard (`--fuzz=0`, all-or-nothing) and dumps the base region if a
patch does not apply, so a drifted base can never yield a partially patched image.
