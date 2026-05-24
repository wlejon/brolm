#!/usr/bin/env bash
#
# Download Qwen3.5-0.8B — the smallest Qwen3.5 open-weight model — into brolm's
# gitignored weights/ directory.
#
# Note: every Qwen3.5 release is a vision-language model. There is no text-only
# Qwen3.5. The preprocessor / vision tower / hybrid text backbone / top-level
# driver live in brolm/qwen35_{preprocessor,vision,text,vl}.h.
#
# Why this model: the Qwen3.6 open-weight checkpoints (27B dense, 35B-A3B MoE)
# are both 55+ GB. Qwen3.5 and Qwen3.6 share the same hybrid architecture
# (Gated DeltaNet linear-attention layers + periodic Gated Attention), so the
# 0.8B Qwen3.5 model — ~1.8 GB — is the cheapest checkpoint to develop and
# validate that architecture support against. The Qwen3.5 Small series also
# offers 2B / 4B / 9B; set the REPO override below to grab a larger one.
#
# Requires the Hugging Face CLI (`hf`) on PATH and an authenticated session
# (`hf auth whoami` should succeed). Downloads resume automatically if rerun.
#
# Usage:
#   scripts/download_qwen35.sh [DEST_DIR]
#
#   DEST_DIR  Optional. Where to place the weights. Defaults to
#             <repo>/weights/<model> (weights/ is gitignored).
#
# Environment overrides:
#   REPO=Qwen/Qwen3.5-2B  scripts/download_qwen35.sh   # grab a bigger size

set -euo pipefail

REPO="${REPO:-Qwen/Qwen3.5-0.8B}"

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
echo "Size:        Qwen3.5-0.8B is ~1.8 GB (1 safetensors shard + tokenizer/config)"
echo

mkdir -p "${DEST}"

# Pull the whole repo: the safetensors weights, model.safetensors.index.json,
# config.json, and the tokenizer files (tokenizer.json, tokenizer_config.json,
# vocab.json, merges.txt). brolm's loaders consume the safetensors shards plus
# vocab.json/merges.txt directly.
hf download "${REPO}" --local-dir "${DEST}"

echo
echo "Done. ${REPO} is in: ${DEST}"
