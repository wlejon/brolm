#!/usr/bin/env bash
# Parity gate for the LLM2Vec bidirectional LLaMA text encoder. Default mode
# `synth` builds a small random-weight LLaMA and compares FP32-vs-FP32 (exact
# architecture check, forced onto the CPU backend so brolm runs FP32). Mode
# `real` runs a real merged checkpoint (weights/llm2vec-llama3-8b) on CPU float32
# both sides (slow — 8B on CPU).
#
# The Python reference (scripts/llm2vec_ref.py) is an independent HF-convention
# LLaMA forward with the causal mask dropped; in synth mode it self-checks its
# CAUSAL output against transformers' own LlamaModel before dumping, so the
# bidirectional reference is HF-faithful by construction.
#
# Usage: scripts/llm2vec_parity.sh [synth|real] [--weights DIR] [--len L] [--bin PATH]
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="synth"
WEIGHTS="weights/llm2vec-llama3-8b"
LEN=6
BIN="build/tools/Release/brolm_llm2vec_encode.exe"

while [ $# -gt 0 ]; do
  case "$1" in
    synth|real) MODE="$1"; shift ;;
    --weights)  WEIGHTS="${2:?--weights needs a value}"; shift 2 ;;
    --len)      LEN="${2:?--len needs a value}"; shift 2 ;;
    --bin)      BIN="${2:?--bin needs a value}"; shift 2 ;;
    -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
done

if [ ! -f "$BIN" ]; then
  echo "error: brolm_llm2vec_encode not found at $BIN" >&2
  echo "       build it: cmake --build build --config Release --target brolm_llm2vec_encode" >&2
  exit 1
fi

python scripts/llm2vec_ref.py "$MODE" .parity --weights "$WEIGHTS" --len "$LEN"
read -r L HID < .parity/llm2vec_dims.txt

if [ "$MODE" = "synth" ]; then
  WDIR=.parity/llm2vec_synth
else
  WDIR="$WEIGHTS"
fi

# brolm runs FP32 only on the CPU backend; force it so the compute dtype matches
# the FP32 reference exactly.
BROTENSOR_DEFAULT_DEVICE=CPU "$BIN" \
  --weights-dir "$WDIR" \
  --ids .parity/llm2vec_ids.i32 \
  --out .parity/llm2vec_hidden_mine.f32

python - "$L" "$HID" <<'PY'
import sys
import numpy as np
L, H = int(sys.argv[1]), int(sys.argv[2])
ref  = np.fromfile('.parity/llm2vec_hidden_ref.f32',  dtype='<f4')
mine = np.fromfile('.parity/llm2vec_hidden_mine.f32', dtype='<f4')
assert ref.size == L * H, (ref.size, L * H)
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
print('shape        (%d, %d)' % (L, H))
print('cosine       %.8f' % cos)
print('rel L2 err   %.8f' % rel)
print('ref  std %.5f  mine std %.5f  maxabsdiff %.6f'
      % (ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))
# Gate: FP32-vs-FP32 should be tight.
ok = (cos > 0.9999) and (rel < 1e-3)
print('PARITY %s' % ('OK' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PY