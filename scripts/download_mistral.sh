#!/usr/bin/env bash
#
# Download Mistral-Small-3.1-24B-Instruct-2503 — the Mistral 3.1 vision-language
# checkpoint — into brolm's gitignored weights/ directory.
#
# Mistral 3.1 is a VLM = a 24B Mistral text decoder + a Pixtral vision encoder.
# brolm support is being built up incrementally (see the mistral3_* headers as
# they land); the tokenizer (brolm::mistral::Tokenizer, fed by tekken.json) is
# in first.
#
# Size warning: the full bf16 checkpoint is ~48 GB (there is no smaller 3.1
# release). Until the decoder/vision loaders land, the only thing brolm consumes
# is the tokenizer — so by default this script fetches ONLY the tokenizer +
# config files (~15 MB), which is all the gated tokenizer test needs. Set
# FULL=1 to pull the entire repo (weights included).
#
# This is a gated model: accept the terms on its Hugging Face page while signed
# in before downloading, or `hf download` will 403.
#
# Requires the Hugging Face CLI (`hf`) on PATH and an authenticated session
# (`hf auth whoami` should succeed). Downloads resume automatically if rerun.
#
# Usage:
#   scripts/download_mistral.sh [DEST_DIR]
#
#   DEST_DIR  Optional. Where to place the weights. Defaults to
#             <repo>/weights/<model> (weights/ is gitignored).
#
# Environment overrides:
#   FULL=1   scripts/download_mistral.sh        # pull the full ~48 GB checkpoint
#   REPO=mistralai/Mistral-Small-3.1-24B-Base-2503 scripts/download_mistral.sh

set -euo pipefail

REPO="${REPO:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
FULL="${FULL:-0}"

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
if [ "${FULL}" = "1" ]; then
    echo "Mode:        FULL (~48 GB: safetensors shards + tokenizer + config)"
else
    echo "Mode:        tokenizer-only (~15 MB; set FULL=1 for the weights)"
fi
echo

mkdir -p "${DEST}"

if [ "${FULL}" = "1" ]; then
    # Whole repo: safetensors shards, model.safetensors.index.json, config.json,
    # and the tokenizer (tekken.json + the json configs). brolm's loaders consume
    # the safetensors shards plus tekken.json directly — Mistral needs no conversion.
    hf download "${REPO}" --local-dir "${DEST}"
else
    # Tokenizer milestone only needs tekken.json; the rest are small and handy
    # for reference / the upcoming config parser. The gated tokenizer test reads
    # tekken.json from $BROLM_MISTRAL_DIR (= this DEST).
    hf download "${REPO}" --local-dir "${DEST}" \
        --include tekken.json \
        --include tokenizer_config.json \
        --include special_tokens_map.json \
        --include config.json
fi

echo
echo "Done. ${REPO} is in: ${DEST}"
echo
echo "Run the gated tokenizer test against it with:"
echo "  BROLM_MISTRAL_DIR='${DEST}' ctest --test-dir build -C Release -R mistral_tokenizer"
