#!/usr/bin/env bash
#
# Download Qwen/Qwen3-VL-4B-Instruct into brolm's gitignored weights/
# directory.
#
# Why this model: it's the checkpoint brolm's qwen3vl_{config,tokenizer,
# preprocessor,vision,text,vl}.h support was built and validated against — a
# dense (non-hybrid) 36-layer text backbone with full-rotary M-RoPE plus a
# 24-block vision tower with DeepStack intermediate-feature injection. ~8 GB
# of bf16 safetensors. Qwen3-VL also ships at 2B / 8B / 30B-A3B / 32B /
# 235B-A22B; set the REPO override below to grab a different size (larger
# MoE sizes are not yet supported by brolm's dense-only text backbone).
#
# Requires the Hugging Face CLI (`hf`) on PATH and an authenticated session
# (`hf auth whoami` should succeed). Downloads resume automatically if rerun.
#
# Usage:
#   scripts/download_qwen3vl.sh [DEST_DIR]
#
#   DEST_DIR  Optional. Where to place the weights. Defaults to
#             <repo>/weights/<model> (weights/ is gitignored).
#
# Environment overrides:
#   REPO=Qwen/Qwen3-VL-2B-Instruct  scripts/download_qwen3vl.sh   # grab a smaller size

set -euo pipefail

REPO="${REPO:-Qwen/Qwen3-VL-4B-Instruct}"

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
    exit 1
fi

echo "Repo:        ${REPO}"
echo "Destination: ${DEST}"
echo "Size:        Qwen3-VL-4B-Instruct is ~8 GB (bf16 safetensors shards + tokenizer/config)"
echo

mkdir -p "${DEST}"

# Pull the model + tokenizer/config: the safetensors shards plus their index,
# config.json, and the tokenizer files. brolm's qwen3vl_tokenizer reads
# vocab.json/merges.txt directly; tokenizer.json/tokenizer_config.json are
# grabbed for reference. --include skips GGUF/pytorch-.bin duplicates while
# still catching every safetensors shard.
hf download "${REPO}" --local-dir "${DEST}" \
    --include "config.json" \
    --include "generation_config.json" \
    --include "preprocessor_config.json" \
    --include "chat_template.jinja" \
    --include "vocab.json" \
    --include "merges.txt" \
    --include "tokenizer.json" \
    --include "tokenizer_config.json" \
    --include "special_tokens_map.json" \
    --include "*.safetensors" \
    --include "*.safetensors.index.json"

# The "*.safetensors" glob is unreliable across hf-CLI versions — some skip
# shards silently while still exiting 0, leaving a weightless directory. Read
# the shard names straight out of the index and explicitly fetch any that are
# missing (or came down truncated), so a successful exit always means a
# complete model.
INDEX="${DEST}/model.safetensors.index.json"
if [[ -f "${INDEX}" ]]; then
    shards="$(grep -oE '[A-Za-z0-9_.-]*-[0-9]+-of-[0-9]+\.safetensors' "${INDEX}" | sort -u)"
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
