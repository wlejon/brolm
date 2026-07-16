#!/usr/bin/env python3
"""Merge the LLM2Vec LoRA adapters into Llama-3 and write a brolm-loadable dir.

`brolm::llm2vec::Encoder` loads a plain HF LLaMA safetensors checkpoint (the
`model.embed_tokens.weight` / `model.layers.N.*` / `model.norm.weight` name
scheme). The LLM2Vec release does NOT ship as such a checkpoint — it is the base
`meta-llama/Meta-Llama-3-8B-Instruct` decoder plus two stacked LoRA adapters:

  1. McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp             (bidirectional
                                                                   MNTP fine-tune)
  2. McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised  (contrastive)

The LLM2Vec attention change (bidirectional / causal-mask-dropped) is applied at
INFERENCE by the C++ encoder — it is not a weight change — so once the two LoRA
deltas are folded into the base projection weights the result is an ordinary
LLaMA checkpoint the C++ loader reads directly. This script does that fold:

  1. Assembles base + mntp + supervised via the `llm2vec` package (its
     from_pretrained loads the base, merges MNTP, then applies the supervised
     adapter — exactly the stack the model card documents).
  2. Merges the remaining (supervised) adapter with merge_and_unload().
  3. Remaps the merged LlamaModel state-dict onto brolm's `model.`-prefixed HF
     names, drops non-persistent buffers (rotary inv_freq), casts to bf16.
  4. Writes model.safetensors + config.json + the tokenizer into --out.

The encoder never runs an lm_head, so none is written (the C++ loader ties the
unused head to the embedding matrix).

Requires: torch, transformers, peft, safetensors, and the `llm2vec` package
(`pip install llm2vec`), plus HF access to the gated
`meta-llama/Meta-Llama-3-8B-Instruct` (`huggingface-cli login`).

Usage:
  scripts/convert-llm2vec.py [--out DIR] [--mntp REPO] [--supervised REPO]
                             [--dtype bf16|fp16|fp32] [--dump-keys]

  --out DIR         output directory (default: weights/llm2vec-llama3-8b)
  --mntp REPO       base+MNTP repo   (default: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp)
  --supervised REPO supervised LoRA  (default: <mntp>-supervised)
  --dtype           saved weight dtype (default: bf16)
  --dump-keys       print the merged (name, shape, dtype) list and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_DEF_MNTP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
_DEF_SUP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def remap_key(k: str):
    """Map a merged LlamaModel state-dict key onto brolm's HF name, or None to
    drop it. LlamaModel keys have no `model.` prefix and no lm_head; brolm reads
    `model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`."""
    # Non-persistent RoPE buffers — brolm recomputes RoPE closed-form.
    if "rotary_emb.inv_freq" in k or k.endswith(".attention_scaling"):
        return None
    # A PeftModel that was not fully unwrapped can leave a base_model.model.
    # prefix; strip it defensively.
    for pref in ("base_model.model.", "base_model."):
        if k.startswith(pref):
            k = k[len(pref):]
    # LlamaForCausalLM already carries the `model.` prefix + lm_head; LlamaModel
    # (what LLM2Vec loads) does not. Normalise to the `model.`-prefixed scheme.
    if k.startswith("model."):
        return k
    if k == "lm_head.weight":
        return k  # harmless if present; the encoder ignores it
    return "model." + k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="weights/llm2vec-llama3-8b")
    ap.add_argument("--mntp", default=_DEF_MNTP)
    ap.add_argument("--supervised", default=None)
    ap.add_argument("--local-dir", default=None,
                    help="offline merge: a dir holding base/ mntp/ supervised/ "
                         "(from scripts/download-llm2vec.sh). Uses peft directly, "
                         "no llm2vec package and no gated repo.")
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    ap.add_argument("--dump-keys", action="store_true")
    args = ap.parse_args()

    dtype = _DTYPES[args.dtype]
    tok_src = args.mntp  # where the tokenizer is loaded from at the end

    if args.local_dir:
        # ── Offline peft-only stack: base + MNTP + supervised, all local ──
        # This is the ungated path. The llm2vec package pins an old transformers
        # and monkeypatches attention; we don't need it — merging LoRA deltas is
        # pure peft (W += B@A), and the bidirectional-attention change is applied
        # at inference by the C++ encoder, not baked into the weights.
        from pathlib import Path as _P
        root = _P(args.local_dir)
        base_dir, mntp_dir, sup_dir = root / "base", root / "mntp", root / "supervised"
        for d in (base_dir, mntp_dir, sup_dir):
            if not d.is_dir():
                ap.error(f"--local-dir missing subdir: {d}")
        tok_src = str(base_dir)

        from transformers import AutoModel
        from peft import PeftModel

        print(f"loading base {base_dir}", flush=True)
        model = AutoModel.from_pretrained(str(base_dir), torch_dtype=dtype)
        print(f"applying + merging MNTP adapter {mntp_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(mntp_dir)).merge_and_unload()
        print(f"applying + merging supervised adapter {sup_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(sup_dir)).merge_and_unload()
    else:
        supervised = args.supervised or (args.mntp + "-supervised"
                                         if args.mntp == _DEF_MNTP else None)
        if supervised is None:
            ap.error("--supervised is required when --mntp is not the default")

        try:
            from llm2vec import LLM2Vec
        except ImportError:
            print("error: the 'llm2vec' package is required (pip install "
                  "llm2vec), or use --local-dir for the offline peft path",
                  file=sys.stderr)
            return 1

        # Assemble base + MNTP (merged inside from_pretrained) + supervised.
        print(f"loading {args.mntp}\n     + {supervised}", flush=True)
        l2v = LLM2Vec.from_pretrained(
            args.mntp,
            peft_model_name_or_path=supervised,
            device_map="cpu",
            torch_dtype=dtype,
        )
        model = l2v.model
        # Fold the still-attached supervised LoRA into the base weights.
        if hasattr(model, "merge_and_unload"):
            print("merging supervised adapter...", flush=True)
            model = model.merge_and_unload()

    sd = model.state_dict()

    remapped = {}
    for k, v in sd.items():
        nk = remap_key(k)
        if nk is None:
            continue
        remapped[nk] = v.to(dtype).contiguous()

    if args.dump_keys:
        for k in sorted(remapped):
            t = remapped[k]
            print(f"{k:60s} {tuple(t.shape)} {t.dtype}")
        print(f"\n{len(remapped)} tensors")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    st_path = out / "model.safetensors"
    print(f"writing {st_path} ({len(remapped)} tensors, {args.dtype})", flush=True)
    save_file(remapped, str(st_path), metadata={"format": "pt"})

    # config.json — the flat LLaMA keys brolm::llm2vec::Config reads.
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    config = {
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads",
                                       cfg.num_attention_heads),
        "head_dim": head_dim,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": getattr(cfg, "rope_theta", 500000.0),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", 8192),
    }
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {out / 'config.json'}", flush=True)

    # Tokenizer — the base Llama-3 BPE (tokenizer.json + specials).
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_src)
        tok.save_pretrained(out)
        print(f"wrote tokenizer to {out}", flush=True)
    except Exception as e:  # noqa: BLE001 - tokenizer is convenience, not fatal
        print(f"warning: could not save tokenizer ({e}); copy tokenizer.json "
              f"from {args.mntp} manually", file=sys.stderr)

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
