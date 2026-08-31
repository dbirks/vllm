#!/usr/bin/env bash
# Apply the reviewed APC (automatic prefix caching) patches to the installed vLLM
# in the base image, strictly. Fail HARD on any drift so a mispatched image can
# never be published. See PATCHES.md for provenance.
#
# Invariant: both files must patch fully or not at all. #54076 hunk 1 DEFINES
# self.mamba_state_block_size, which hunks 2-3 (and correctness) depend on; a
# partial apply would be a runtime AttributeError, so every hunk is required.
set -euo pipefail

SITE=/usr/local/lib/python3.12/dist-packages
cd "$SITE"

command -v patch >/dev/null 2>&1 || {
  echo "=== installing patch ==="
  apt-get update && apt-get install -y --no-install-recommends patch
  rm -rf /var/lib/apt/lists/*
}

python3 -c "import vllm; print('base vLLM:', vllm.__version__)"

targets="vllm/v1/worker/gpu/model_states/mamba_hybrid.py vllm/v1/core/sched/scheduler.py"
for f in $targets; do
  [ -f "$f" ] || { echo "FATAL: patch target missing in base image: $f"; exit 3; }
done

# 1) Strict dry-run gate for every patch (no fuzz). On failure, dump the reject
#    plus the real base regions so the patch can be re-anchored in one iteration.
for p in /tmp/patches/*.patch; do
  echo "=== dry-run (strict, --fuzz=0): $p ==="
  if ! patch -p1 --fuzz=0 --dry-run < "$p"; then
    echo "########## FATAL: patch does not apply cleanly to base (drift): $p ##########"
    patch -p1 --fuzz=0 < "$p" || true
    echo "----- rejects -----"
    find . -name '*.rej' -print -exec cat {} \;
    echo "----- base scheduler.py __init__ region (lines 320-440) -----"
    sed -n '320,440p' vllm/v1/core/sched/scheduler.py || true
    echo "----- base mamba_hybrid.py region (lines 85-170) -----"
    sed -n '85,170p' vllm/v1/worker/gpu/model_states/mamba_hybrid.py || true
    exit 1
  fi
done

# 2) All patches verified appliable: apply for real, strictly.
for p in /tmp/patches/*.patch; do
  echo "=== apply: $p ==="
  patch -p1 --fuzz=0 < "$p"
done

find . -name '*.orig' -delete

# 3) Byte-compile the touched modules so any syntax error fails the build here,
#    not at engine start on the node.
python3 -m py_compile $targets

echo "=== APC candidate A patched OK (#53798 + #54076) ==="
