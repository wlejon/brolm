#!/usr/bin/env bash
#
# Download google/gemma-2-2b — the smallest Gemma-2 open-weight model — into
# brolm's gitignored weights/ directory.
#
# Why this model: Gemma-2 is a decoder-only LLM with a SentencePiece-BPE
# tokenizer (~256k vocab, byte-fallback, no prefix space). The 2B checkpoint is
# the cheapest Gemma-2 to develop and validate brolm's gemma_tokenizer (and
# future Gemma-2 decoder support) against — ~5 GB of BF16 safetensors plus the
# tokenizer/config files. Gemma-2 also ships at 9B / 27B; set the REPO override
# to grab a larger size.
#
# google/gemma-2-2b is a GATED repo: accept the license on its model page and
# authenticate (`hf auth login`) before this will download. Downloads resume
# automatically if rerun.
#
# Requires the Hugging Face CLI (`hf`) on PATH and an authenticated session
# (`hf auth whoami` should succeed).
#
# Usage:
#   scripts/download_gemma2_2b.sh [DEST_DIR]
#
#   DEST_DIR  Optional. Where to place the weights. Defaults to
#             <repo>/weights/<model> (weights/ is gitignored).
#
# Environment overrides:
#   REPO=google/gemma-2-9b   scripts/download_gemma2_2b.sh   # grab a bigger size
#   REPO=google/gemma-2-2b-it scripts/download_gemma2_2b.sh  # instruction-tuned

set -euo pipefail

REPO="${REPO:-google/gemma-2-2b}"

# Resolve the repo root from this script's location, independent of CWD.
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
    echo "       google/gemma-2-2b is gated — accept its license on the" >&2
    echo "       model page first: https://huggingface.co/${REPO}" >&2
    exit 1
fi

echo "Repo:        ${REPO}"
echo "Destination: ${DEST}"
echo "Size:        gemma-2-2b is ~5 GB (safetensors shards + tokenizer/config)"
echo

mkdir -p "${DEST}"

# Pull the model + tokenizer/config: the safetensors shards plus their index,
# config.json, and both tokenizer files. brolm's gemma_tokenizer reads
# tokenizer.json; tokenizer.model is the SentencePiece source for reference.
# --include keeps it to what the loaders consume (skips e.g. the GGUF mirror,
# flax/pytorch .bin duplicates) while still grabbing every safetensors shard.
hf download "${REPO}" --local-dir "${DEST}" \
    --include "config.json" \
    --include "tokenizer.json" \
    --include "tokenizer.model" \
    --include "tokenizer_config.json" \
    --include "special_tokens_map.json" \
    --include "generation_config.json" \
    --include "*.safetensors" \
    --include "*.safetensors.index.json"

# The "*.safetensors" glob is unreliable across hf-CLI versions — some skip the
# shards silently while still exiting 0, leaving a weightless directory. Read the
# shard names straight out of the index and explicitly fetch any that are missing
# (or came down truncated), so a successful exit always means a complete model.
INDEX="${DEST}/model.safetensors.index.json"
if [[ -f "${INDEX}" ]]; then
    shards="$(grep -oE 'model-[0-9]+-of-[0-9]+\.safetensors' "${INDEX}" | sort -u)"
    missing=()
    for shard in ${shards}; do
        # Re-fetch unless the file exists and is plausibly a real shard (>1 MiB);
        # hf download is incremental, so present files are a no-op.
        if [[ ! -f "${DEST}/${shard}" ]] || \
           [[ "$(stat -c %s "${DEST}/${shard}" 2>/dev/null || echo 0)" -lt 1048576 ]]; then
            missing+=("--include" "${shard}")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Fetching ${missing[*]} shard(s) the glob skipped..."
        hf download "${REPO}" --local-dir "${DEST}" "${missing[@]}"
    fi
    # Hard-fail if any indexed shard is still absent.
    for shard in ${shards}; do
        if [[ ! -f "${DEST}/${shard}" ]]; then
            echo "error: shard ${shard} missing after download — model is incomplete." >&2
            exit 1
        fi
    done
fi

echo
echo "Done. ${REPO} is in: ${DEST}"
ls -lh "${DEST}"
