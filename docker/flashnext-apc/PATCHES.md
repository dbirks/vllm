# APC candidate A — patch provenance

Image: `ghcr.io/dbirks/vllm-flashnext:apc-a` (immutable: `:apc-a-<git-sha>`)
Base:  `ghcr.io/dbirks/vllm-flashnext:sm120@sha256:f7186f70baf43061c042dfc8e9b296b75476f287b7c8d8394a7d2e287ed61e82`
       (= Flash-Next base runtime `vLLM 0.1.dev20073+g8e685d198` + INT4 PLE overlay)

Goal: get non-zero `prefix_cache_hits_total` on qwen4_exp (Qwen3.8-Flash-Next) by
fixing two known hybrid-`align`-mode cache invariants. Both patches are the SOURCE
hunks only (test files dropped); applied strictly with `--fuzz=0` at build time.

| Patch file | Upstream PR | State when vendored | Target module | What it fixes |
|---|---|---|---|---|
| `patches/53798-mamba-align-seed.patch` | vllm-project/vllm#53798 | OPEN (2026-08-31) | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | Worker seeds the resumed align-mode recurrent-state block using the Mamba group's block size, not the generic `cache_config.block_size`; a wrong divisor points the precopy source outside the request's block-table row. |
| `patches/54076-mamba-align-chunk-split.patch` | vllm-project/vllm#54076 | OPEN (2026-08-31) | `vllm/v1/core/sched/scheduler.py` | Scheduler `_mamba_block_aligned_split` stops chunks on `MambaSpec.block_size` (the state grid the worker actually checkpoints), not the group-minimum `cache_config.block_size`; otherwise no reusable state materializes and the Mamba group publishes no / a misaligned prefix hash. |

Both PRs target the same hybrid-`align` prefix-cache path flagged by upstream #45238
(align veto -> 0% hits on multi-turn), the exact symptom in home-k8s #115.

## Deliberately excluded
- MTP/EAGLE align fixes (#48375, #50897): the Flash-Next Deployment does not pass
  `--speculative-config`, so MTP is inactive. Excluded to keep attribution clean.
- Candidate B (#53479, direct state materialization for diverged/growing prefixes):
  only built if A is stable but reuse stays weak. #53479 and #54076 both touch
  `_mamba_block_aligned_split`, so B must be a reviewed local merge, not blind stacking.

## Re-vendoring
```
gh pr diff 53798 --repo vllm-project/vllm | awk '/^diff --git.*mamba_hybrid.py/{p=1} /^diff --git/{if($0!~/mamba_hybrid.py/)p=0} p' > patches/53798-mamba-align-seed.patch
gh pr diff 54076 --repo vllm-project/vllm | awk '/^diff --git.*sched\/scheduler.py/{p=1} /^diff --git/{if($0!~/sched\/scheduler.py/)p=0} p' > patches/54076-mamba-align-chunk-split.patch
```
If either PR is force-pushed and the base drifts, the build fails hard and dumps the
base region; re-anchor the hunk and rebuild. Never publish a partially patched image.
