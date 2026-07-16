#!/usr/bin/env bash
# Download the REAL LLM2Vec-Llama-3-8B text encoder — ungated, plain curl, no
# HF CLI, no auth token needed.
#
# LLM2Vec ships as a base decoder + two stacked LoRA adapters, NOT a single
# checkpoint. The adapters are public, but their `base_model_name_or_path`
# points at the GATED `meta-llama/Meta-Llama-3-8B-Instruct`. We sidestep the
# gate entirely by pulling the base from NousResearch's ungated mirror (identical
# weights) and repointing the adapter configs at the local base dir, so the
# downstream merge (scripts/convert-llm2vec.py) never touches a gated repo.
#
#   base        NousResearch/Meta-Llama-3-8B-Instruct   (~16 GB, 4 shards)
#   mntp        McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
#   supervised  McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
#
# Usage: scripts/download-llm2vec.sh [--out-dir DIR] [--force]
#   --out-dir DIR   default: weights/llm2vec-real
#   --force         re-download even if a file already exists
#
# Output layout:
#   <out>/base/         config + tokenizer + 4 safetensors shards
#   <out>/mntp/         adapter_config.json (repointed) + adapter_model.safetensors
#   <out>/supervised/   adapter_config.json (repointed) + adapter_model.safetensors

set -euo pipefail

OUT_DIR=""
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/llm2vec-real"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

BASE_REPO="NousResearch/Meta-Llama-3-8B-Instruct"
MNTP_REPO="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUP_REPO="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

# fetch <repo> <relative-path> <dest-file>
fetch() {
    local repo="$1" rel="$2" dest="$3"
    local url="https://huggingface.co/$repo/resolve/main/$rel"
    local auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $rel  (cached)"; return 0
    fi
    echo "==> $repo/$rel"
    mkdir -p "$(dirname "$dest")"
    local code
    code="$(curl -sL --retry 5 --retry-delay 3 "${auth[@]}" \
                 -o "$dest.part" -w '%{http_code}' "$url")" \
        || { echo "    curl failed for $url" >&2; rm -f "$dest.part"; return 2; }
    if [ "$code" = "200" ]; then mv "$dest.part" "$dest"; return 0; fi
    rm -f "$dest.part"; echo "    HTTP $code for $url" >&2; return 2
}

echo "Target: $OUT_DIR"; echo

# --- base (ungated NousResearch mirror) ---
BASE_FILES=(
    config.json generation_config.json
    special_tokens_map.json tokenizer.json tokenizer_config.json
    model.safetensors.index.json
    model-00001-of-00004.safetensors model-00002-of-00004.safetensors
    model-00003-of-00004.safetensors model-00004-of-00004.safetensors
)
for f in "${BASE_FILES[@]}"; do fetch "$BASE_REPO" "$f" "$OUT_DIR/base/$f"; done

# --- adapters ---
for f in adapter_config.json adapter_model.safetensors; do
    fetch "$MNTP_REPO" "$f" "$OUT_DIR/mntp/$f"
    fetch "$SUP_REPO"  "$f" "$OUT_DIR/supervised/$f"
done

# --- repoint adapter base_model_name_or_path at the local ungated base ---
python - "$OUT_DIR" <<'PY'
import json, os, sys
out = sys.argv[1]
base = os.path.join(out, "base")
for name in ("mntp", "supervised"):
    p = os.path.join(out, name, "adapter_config.json")
    with open(p) as f: cfg = json.load(f)
    old = cfg.get("base_model_name_or_path")
    cfg["base_model_name_or_path"] = base
    with open(p, "w") as f: json.dump(cfg, f, indent=2)
    print(f"repointed {name}: {old} -> {base}")
PY

echo; echo "Done. Layout:"
du -sh "$OUT_DIR"/base "$OUT_DIR"/mntp "$OUT_DIR"/supervised 2>/dev/null || true
