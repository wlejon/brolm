#!/usr/bin/env bash
#
# Download Qwen3-1.7B as a Q8_0 GGUF — the context-aware translation model for
# listen-lab's "correctness" tier — into brolm's gitignored weights/ directory.
#
# Why this model:
#   The listen-lab translator runs two tiers. NLLB-200 (encoder-decoder MT) is
#   the fast, per-sentence tier that keeps up with the live transcript. But NLLB
#   is sentence-level and pro-drop languages (Japanese) lose meaning when each
#   line is translated in isolation. The slow "correctness" tier instead feeds an
#   instruction LLM the running, speaker-tagged dialogue as CONTEXT and re-asks
#   for the latest line(s), recovering dropped pronouns / referents / who-speaks.
#
#   Qwen3-8B does this well but is ~9 GB — disproportionate next to a pipeline
#   that is otherwise a couple GB. Qwen3-1.7B (~1.8 GB at Q8_0) is the smallest
#   text-only Qwen3 above 0.6B and is the size/quality sweet spot for this job.
#   (Qwen3 publishes 0.6B / 1.7B / 4B / 8B / 14B…; there is no 1.5B — 1.7B is the
#   closest. bro.lm.loadQwen loads Qwen3 GGUF + ChatML, so we stay in-family.)
#
#   Q8_0 exercises brotensor's quant-carrier path (CUDA linear_forward_*_q8_0),
#   matching how the other Qwen3 GGUFs run; it is the quant Qwen publishes.
#
#   DO NOT quant small models hard. Sub-~4B models degrade sharply below ~Q6 and
#   start to malfunction (broken instruction-following, garbled output). Keep
#   these at Q8_0 (or Q6_K at the lowest); reserve Q4_K_M for the big models.
#
# Requires `hf` on PATH and an authenticated session.
#
# Usage:
#   scripts/download_qwen3_translate.sh [DEST_DIR]
#
# Environment overrides:
#   REPO=Qwen/Qwen3-4B-GGUF  scripts/download_qwen3_translate.sh   # step up to 4B
#   QUANTS="Q6_K"            scripts/download_qwen3_translate.sh   # Q6 floor for small models

set -euo pipefail

REPO="${REPO:-Qwen/Qwen3-1.7B-GGUF}"
QUANTS="${QUANTS:-Q8_0}"

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

# Qwen's GGUF repos store one .gguf per quant at the repo root, named
# "<model>-<Quant>.gguf" (e.g. Qwen3-1.7B-Q8_0.gguf). --include grabs only the
# quant(s) we ask for instead of the whole repo.
model_base="$(basename "${REPO}" -GGUF)"
include_args=()
for q in ${QUANTS}; do
    include_args+=(--include "${model_base}-${q}.gguf")
done

hf download "${REPO}" --local-dir "${DEST}" "${include_args[@]}"

echo
echo "Done. Files in ${DEST}:"
ls -lh "${DEST}"/*.gguf
