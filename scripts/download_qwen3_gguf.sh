#!/usr/bin/env bash
#
# Download Qwen3-0.6B as both F16 and Q8_0 GGUF — the cheapest text-only Qwen3
# checkpoints — into brolm's gitignored weights/ directory.
#
# Why two quants:
#   - BF16 (~1.2 GB) loads as plain dense weights and exercises brolm's gguf
#     loader on every backend (CPU + CUDA + Metal). brolm widens BF16 to FP32
#     on CPU and narrows to FP16 on GPU. This is the variant the gated
#     end-to-end test (tests/test_qwen_gguf.cpp) targets.
#   - Q8_0 (~640 MB) exercises the quant-carrier path: brotensor keeps the
#     on-disk dtype and dispatches through linear_forward_batched_q8_0_fp16.
#     CUDA-only at runtime (no CPU/Metal quant kernels yet).
#
# The default is unsloth/Qwen3-0.6B-GGUF rather than Qwen's official Qwen3
# GGUF repo because Qwen only publishes Q8_0; unsloth offers BF16 + every
# common quant.
#
# Qwen3-0.6B was chosen as the smallest text-only Qwen3 — Qwen3.5 (already in
# scripts/download_qwen35.sh) is vision-language only. The two stacks share
# attention, RoPE, GQA, and the SwiGLU MLP, so verifying the gguf path on
# Qwen3 transfers to the Qwen3.5 family once it gets its own gguf loaders.
#
# Requires `hf` on PATH and an authenticated session.
#
# Usage:
#   scripts/download_qwen3_gguf.sh [DEST_DIR]
#
# Environment overrides:
#   REPO=Qwen/Qwen3-1.7B-GGUF  scripts/download_qwen3_gguf.sh   # different size
#   QUANTS="Q4_K_M f16"        scripts/download_qwen3_gguf.sh   # different mix

set -euo pipefail

REPO="${REPO:-unsloth/Qwen3-0.6B-GGUF}"
QUANTS="${QUANTS:-BF16 Q8_0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST="${1:-${REPO_ROOT}/weights/$(basename "${REPO}")}"

if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' (Hugging Face CLI) not found on PATH." >&2
    echo "       Install with: pip install -U huggingface_hub" >&2
    exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
    echo "error: not authenticated. Run 'hf auth login' first." >&2
    exit 1
fi

echo "Repo:        ${REPO}"
echo "Destination: ${DEST}"
echo "Quants:      ${QUANTS}"
echo

mkdir -p "${DEST}"

# Qwen's GGUF repos store one .gguf per quant at the repo root. The filename
# convention is "<model>-<Quant>.gguf" (e.g. Qwen3-0.6B-Q8_0.gguf,
# Qwen3-0.6B-f16.gguf). hf download with --include grabs only the files we ask
# for, so we don't pull every quant in the repo.
model_base="$(basename "${REPO}" -GGUF)"
include_args=()
for q in ${QUANTS}; do
    include_args+=(--include "${model_base}-${q}.gguf")
done

hf download "${REPO}" --local-dir "${DEST}" "${include_args[@]}"

echo
echo "Done. Files in ${DEST}:"
ls -lh "${DEST}"/*.gguf
