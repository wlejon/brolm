#!/usr/bin/env bash
#
# Download a quantized GGUF of the Mistral-Small-3.1-24B text decoder into
# brolm's gitignored weights/ directory, for the GGUF load path
# (brolm::mistral3::TextModel::load_weights(gguf::File)).
#
# Mistral 3.1 is a VLM, but llama.cpp splits the conversion into a text-model
# GGUF (the decoder brolm loads) plus a separate `mmproj-*` GGUF for the Pixtral
# vision tower. This script fetches the TEXT model by default; pass MMPROJ=1 to
# also grab the vision projector for later multimodal work.
#
# Default quant is Q4_K_M (~14 GB) — it fits a single 24 GB GPU with room for
# the KV cache, and uses only Q4_K / Q6_K blocks, which are exactly the quant
# dtypes brolm dispatches through brotensor's CUDA kernels (alongside Q8_0).
# AVOID IQ*, Q2_K, Q3_K, Q5_K, Q4_0/Q4_1 quants: brotensor has no kernels for
# them and the load/forward will fail.
#
# The quant path is GPU-only (brotensor's dequant + quant-matmul kernels are
# CUDA-only), so the downloaded file is loadable only by a brolm built with
# -DBROTENSOR_WITH_CUDA=ON.
#
# Requires the Hugging Face CLI (`hf`) on PATH. The bartowski GGUF repo is
# public (ungated), so no login is needed. Downloads resume if rerun.
#
# Usage:
#   scripts/download_mistral_gguf.sh [DEST_DIR]
#
# Environment overrides:
#   QUANT=Q6_K  scripts/download_mistral_gguf.sh   # pick a different quant
#   MMPROJ=1    scripts/download_mistral_gguf.sh   # also fetch the vision mmproj
#   REPO=...    scripts/download_mistral_gguf.sh   # a different GGUF repo

set -euo pipefail

REPO="${REPO:-bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF}"
QUANT="${QUANT:-Q4_K_M}"
MMPROJ="${MMPROJ:-0}"

# bartowski names files "<model>-<QUANT>.gguf" with the model basename below.
MODEL_BASE="mistralai_Mistral-Small-3.1-24B-Instruct-2503"
GGUF_FILE="${MODEL_BASE}-${QUANT}.gguf"
MMPROJ_FILE="mmproj-${MODEL_BASE}-f16.gguf"

# Resolve the repo root from this script's location, independent of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST="${1:-${REPO_ROOT}/weights/Mistral-Small-3.1-24B-Instruct-2503-GGUF}"

if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' (Hugging Face CLI) not found on PATH." >&2
    echo "       Install with: pip install -U huggingface_hub" >&2
    exit 1
fi

echo "Repo:        ${REPO}"
echo "Quant:       ${QUANT}  (${GGUF_FILE})"
echo "Destination: ${DEST}"
[ "${MMPROJ}" = "1" ] && echo "Also:        ${MMPROJ_FILE} (vision mmproj)"
echo

mkdir -p "${DEST}"

hf download "${REPO}" "${GGUF_FILE}" --local-dir "${DEST}"

if [ "${MMPROJ}" = "1" ]; then
    hf download "${REPO}" "${MMPROJ_FILE}" --local-dir "${DEST}"
fi

echo
echo "Done. ${GGUF_FILE} is in: ${DEST}"
echo
echo "Load it with a CUDA-enabled build and the gated test:"
echo "  cmake -B build-cuda -DBROTENSOR_WITH_CUDA=ON"
echo "  cmake --build build-cuda --config Release --target brolm_test_mistral3_gguf"
echo "  BROLM_MISTRAL_GGUF='${DEST}/${GGUF_FILE}' \\"
echo "    ./build-cuda/tests/Release/brolm_test_mistral3_gguf"
