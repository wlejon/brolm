#!/usr/bin/env python
# Reference for the LLM2Vec bidirectional LLaMA text encoder, to diff against
# brolm::llm2vec::Encoder (tools/llm2vec_encode, CLI backend of
# scripts/llm2vec_parity.sh).
#
# LLM2Vec = a decoder-only LLaMA with the causal mask DROPPED (bidirectional
# attention); everything else — RoPE, GQA, RMSNorm, SwiGLU — is stock LLaMA.
# So the reference here is an independent torch implementation of the LLaMA
# forward in the exact Hugging Face convention (split-half rotate_half RoPE on
# HF-layout weights, the layout brolm loads and permutes internally), run with
# NO causal mask.
#
# To prove that independent forward really is the HF convention (and not just
# "some" LLaMA), synth mode SELF-CHECKS it against transformers' own LlamaModel
# in CAUSAL mode over the same weights before dumping — if our causal output
# matches transformers, our bidirectional output is HF-faithful by construction.
# The one deviation LLM2Vec makes (dropping the mask) is the only difference
# between the self-check and the dumped reference.
#
# Two modes:
#   synth  (default) — build a SMALL random-weight LLaMA in float32, save it
#       (config.json + model.safetensors, HF `model.`-prefixed names) to a synth
#       dir, self-check causal vs transformers, then dump the BIDIRECTIONAL
#       per-token hidden states + the token ids. Both sides FP32 => exact
#       architecture parity, always fits in memory.
#   real   — load a real merged checkpoint dir (config.json + model.safetensors,
#       e.g. weights/llm2vec-llama3-8b from scripts/convert-llm2vec.py), run the
#       bidirectional forward on CPU float32, dump. Slow (8B on CPU).
#
# Dumps (raw little-endian) into <outdir>:
#   llm2vec_ids.i32          int32 token ids                 (the C++ reads these)
#   llm2vec_hidden_ref.f32   float32 (L, hidden) reference    (the shell diffs this)
#   llm2vec_dims.txt         "L hidden"
#   llm2vec_synth/           synth weights dir (synth mode only)
#
# Usage: python scripts/llm2vec_ref.py [synth|real] [outdir] [--weights DIR]
#        [--len L] [--no-check]
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch


# ── HF-convention LLaMA forward (independent of transformers) ────────────────

def rms_norm(x, w, eps):
    # x: (L, H) float32. HF computes the norm in float32.
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * w


def rotate_half(x):
    # HF split-half convention: (x1, x2) -> (-x2, x1).
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(L, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(L).float()
    freqs = torch.outer(pos, inv_freq)          # (L, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (L, head_dim)
    return emb.cos(), emb.sin()                 # (L, head_dim) each


def apply_rope(x, cos, sin):
    # x: (L, n_heads, head_dim); cos/sin: (L, head_dim).
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + rotate_half(x) * sin


def llama_forward(w, ids, cfg, causal):
    """w: dict of HF-layout tensors (NO `model.` prefix). ids: (L,) int64.
    Returns (L, hidden) float32 hidden states after the final norm."""
    H   = cfg["hidden_size"]
    nL  = cfg["num_hidden_layers"]
    n_q = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd  = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    L = ids.shape[0]
    rep = n_q // nkv

    h = w["embed_tokens.weight"][ids]           # (L, H)
    cos, sin = rope_cos_sin(L, hd, theta)

    mask = None
    if causal:
        mask = torch.full((L, L), float("-inf")).triu(1)   # (L,L), 0 on/below diag

    for i in range(nL):
        p = f"layers.{i}."
        # ── attention ──
        residual = h
        x = rms_norm(h, w[p + "input_layernorm.weight"], eps)
        q = (x @ w[p + "self_attn.q_proj.weight"].T).view(L, n_q, hd)
        k = (x @ w[p + "self_attn.k_proj.weight"].T).view(L, nkv, hd)
        v = (x @ w[p + "self_attn.v_proj.weight"].T).view(L, nkv, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA: repeat kv heads to match q heads.
        k = k.repeat_interleave(rep, dim=1)     # (L, n_q, hd)
        v = v.repeat_interleave(rep, dim=1)
        # (n_q, L, L) scores
        qh = q.permute(1, 0, 2)                 # (n_q, L, hd)
        kh = k.permute(1, 0, 2)
        vh = v.permute(1, 0, 2)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / (hd ** 0.5)
        if mask is not None:
            scores = scores + mask[None, :, :]
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, vh)            # (n_q, L, hd)
        ctx = ctx.permute(1, 0, 2).reshape(L, n_q * hd)
        h = residual + ctx @ w[p + "self_attn.o_proj.weight"].T
        # ── MLP (SwiGLU) ──
        residual = h
        x = rms_norm(h, w[p + "post_attention_layernorm.weight"], eps)
        gate = x @ w[p + "mlp.gate_proj.weight"].T
        up   = x @ w[p + "mlp.up_proj.weight"].T
        act  = torch.nn.functional.silu(gate) * up
        h = residual + act @ w[p + "mlp.down_proj.weight"].T

    return rms_norm(h, w["norm.weight"], eps)


# ── transformers cross-check (synth only) ────────────────────────────────────

def transformers_causal(w, ids, cfg):
    """Load the same weights into transformers' LlamaModel and run its stock
    CAUSAL forward — the authority our llama_forward(causal=True) must match."""
    from transformers import LlamaConfig, LlamaModel
    lc = LlamaConfig(
        vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        attn_implementation="eager",
    )
    m = LlamaModel(lc).eval().float()
    m.load_state_dict(w, strict=False)
    with torch.no_grad():
        out = m(input_ids=ids[None]).last_hidden_state[0]
    return out


# ── weight construction / IO ─────────────────────────────────────────────────

def synth_config():
    H, n_q, hd = 32, 4, 8
    return dict(
        vocab_size=48, hidden_size=H, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=n_q,
        num_key_value_heads=2, head_dim=hd, rms_norm_eps=1e-5,
        rope_theta=500000.0, max_position_embeddings=128,
    )


def synth_weights(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, F = cfg["hidden_size"], cfg["intermediate_size"]
    n_q, nkv, hd = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                    cfg["head_dim"])
    V = cfg["vocab_size"]
    qd, kvd = n_q * hd, nkv * hd

    def r(*shape):
        return torch.randn(*shape, generator=g).float() * 0.1

    w = {"embed_tokens.weight": r(V, H), "norm.weight": r(H).abs() + 0.5}
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = r(H).abs() + 0.5
        w[p + "self_attn.q_proj.weight"] = r(qd, H)
        w[p + "self_attn.k_proj.weight"] = r(kvd, H)
        w[p + "self_attn.v_proj.weight"] = r(kvd, H)
        w[p + "self_attn.o_proj.weight"] = r(H, qd)
        w[p + "post_attention_layernorm.weight"] = r(H).abs() + 0.5
        w[p + "mlp.gate_proj.weight"] = r(F, H)
        w[p + "mlp.up_proj.weight"] = r(F, H)
        w[p + "mlp.down_proj.weight"] = r(H, F)
    return w


def load_real(weights_dir):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(weights_dir, "config.json")))
    cfg.setdefault("head_dim",
                   cfg["hidden_size"] // cfg["num_attention_heads"])
    sd = load_file(os.path.join(weights_dir, "model.safetensors"))
    # Strip the `model.` prefix brolm/HF ForCausalLM carry; llama_forward wants
    # the bare LlamaModel names.
    w = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        if nk == "lm_head.weight":
            continue
        w[nk] = v.float()
    return cfg, w


def save_synth(out, cfg, w):
    from safetensors.torch import save_file
    sdir = os.path.join(out, "llm2vec_synth")
    os.makedirs(sdir, exist_ok=True)
    # brolm reads `model.`-prefixed HF names.
    prefixed = {("model." + k): v.contiguous() for k, v in w.items()}
    save_file(prefixed, os.path.join(sdir, "model.safetensors"))
    json.dump(cfg, open(os.path.join(sdir, "config.json"), "w"), indent=2)
    return sdir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="synth", choices=["synth", "real"])
    ap.add_argument("outdir", nargs="?", default=".parity")
    ap.add_argument("--weights", default="weights/llm2vec-llama3-8b",
                    help="real-mode checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--len", type=int, default=6, help="token-sequence length")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the transformers causal cross-check (synth)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "synth":
        cfg = synth_config()
        w = synth_weights(cfg)
        sdir = save_synth(args.outdir, cfg, w)
        print(f"synth weights -> {sdir}")
    else:
        cfg, w = load_real(args.weights)
        print(f"real weights  <- {args.weights}  (hidden={cfg['hidden_size']} "
              f"layers={cfg['num_hidden_layers']})")

    L = args.len
    rng = np.random.default_rng(1234)
    ids_np = rng.integers(0, cfg["vocab_size"], size=L).astype("<i4")
    ids = torch.from_numpy(ids_np.astype("int64"))

    with torch.no_grad():
        # Self-check: our causal forward must match transformers' LlamaModel.
        if args.mode == "synth" and not args.no_check:
            ours_c = llama_forward(w, ids, cfg, causal=True)
            ref_c = transformers_causal(w, ids, cfg)
            d = float((ours_c - ref_c).abs().max())
            rel = float((ours_c - ref_c).norm() / ref_c.norm())
            print(f"[self-check] causal vs transformers: maxabs {d:.3e}  relL2 {rel:.3e}")
            if d > 1e-4:
                print("ERROR: causal forward does NOT match transformers "
                      "(HF-convention bug in the reference)", file=sys.stderr)
                return 1
            print("[self-check] OK — reference is HF-faithful")

        hidden = llama_forward(w, ids, cfg, causal=False)   # BIDIRECTIONAL

    hidden_np = hidden.float().cpu().numpy().astype("<f4")
    ids_np.tofile(os.path.join(args.outdir, "llm2vec_ids.i32"))
    hidden_np.tofile(os.path.join(args.outdir, "llm2vec_hidden_ref.f32"))
    with open(os.path.join(args.outdir, "llm2vec_dims.txt"), "w") as f:
        f.write(f"{L} {cfg['hidden_size']}\n")
    print(f"dumped ids ({L}) + bidirectional hidden ({L}x{cfg['hidden_size']}) "
          f"to {args.outdir}")
    print(f"  ref hidden: mean {hidden_np.mean():.5f}  std {hidden_np.std():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
