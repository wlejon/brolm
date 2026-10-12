#!/usr/bin/env python3
"""Convert an open_clip ViT checkpoint into the layout bro.lm.loadClip reads.

brolm's CLIP (src/clip.cpp, src/clip_image.cpp, src/clip_score.cpp) loads
weights under Hugging Face `transformers` CLIPModel names, and the JS binding
(src/api/native_lm_clip.cpp) reads the architecture from a config.json beside
them (text_config / vision_config / projection_dim). open_clip checkpoints such
as imageomics/bioclip (ViT-B/16), imageomics/bioclip-2 (ViT-L/14) and
imageomics/bioclip-2.5-vith14 (ViT-H/14: 32 x 1280 vision with 80-wide heads,
read from open_clip_config.json's head_width; 24 x 1024 text) ship
neither: they carry open_clip's own state dict plus an open_clip_config.json.
This script rewrites one into the other; the tensors themselves are copied
bit for bit except where the two layouts differ:

  * attention: open_clip packs q/k/v into one `attn.in_proj_weight` (3D, D) and
    `in_proj_bias` (3D,), in that order; HF splits them into q_proj / k_proj /
    v_proj.
  * projections: open_clip applies `x @ proj` with `visual.proj` (Dv, E) and
    `text_projection` (Dt, E); HF stores nn.Linear weights (E, D), so both are
    transposed.
  * the vision tower's `ln_pre` / `ln_post` become HF's `pre_layrnorm` (HF's
    spelling) / `post_layernorm`, and the text tower's `ln_final` becomes
    `final_layer_norm`. open_clip pools exactly as HF does (ln_post on the CLS
    token, ln_final then the argmax-id (EOT) token), so no other change is
    needed.
  * the activation: open_clip builds nn.GELU (exact, erf) unless the model
    config says `quick_gelu: true`; OpenAI's original CLIP and HF's
    openai/clip-* use QuickGELU. The choice is written as `hidden_act`
    ("gelu" or "quick_gelu") in both sub-configs, which brolm honours.
    No BioCLIP config (1, 2 or 2.5) sets quick_gelu, so open_clip (and therefore every
    published BioCLIP result computed through it) runs them with exact GELU;
    `--act` overrides that for experiments.
  * logit_scale is kept as the raw (log-space) parameter, as in HF.

Text: open_clip pads token ids with 0 after EOT where HF / brolm's tokenizer
pad with EOT itself. The text tower is causal and pools at the first EOT, so
the padding never reaches the pooled feature; nothing to convert.

Output directory: model.safetensors (F32 by default), config.json, vocab.json,
merges.txt (copied from the source dir; open_clip's BPE is the OpenAI one) and
the preprocessing stats in config.json's `preprocess` block for callers that
want them.

--verify loads the result into transformers' CLIPModel and the source into
open_clip, runs a seeded random image and three prompts through both, and
prints the largest difference of the normalised embeddings (expect ~1e-6).

Requires: torch, safetensors, numpy; open_clip_torch + transformers for
--verify. CPU only.

Usage:
  scripts/convert-openclip.py SRC_DIR OUT_DIR [--act gelu|quick_gelu]
                              [--dtype fp32|fp16] [--verify]

  SRC_DIR  an open_clip HF repo snapshot (open_clip_config.json plus
           open_clip_model.safetensors or open_clip_pytorch_model.bin,
           vocab.json, merges.txt), e.g. `hf download imageomics/bioclip-2`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def load_state_dict(src: Path) -> dict:
    st = src / "open_clip_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    bin_path = src / "open_clip_pytorch_model.bin"
    if not bin_path.exists():
        sys.exit(f"no open_clip_model.safetensors or open_clip_pytorch_model.bin in {src}")
    sd = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    # Training checkpoints wrap the weights and prefix them with the DDP name.
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return {k.removeprefix("module."): v for k, v in sd.items()}


def count_layers(sd: dict, prefix: str) -> int:
    n = 0
    while f"{prefix}{n}.ln_1.weight" in sd:
        n += 1
    return n


def split_attention(sd: dict, out: dict, src: str, dst: str) -> None:
    w = sd[src + "attn.in_proj_weight"]
    b = sd[src + "attn.in_proj_bias"]
    d = w.shape[1]
    for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
        out[f"{dst}self_attn.{name}.weight"] = w[i * d:(i + 1) * d]
        out[f"{dst}self_attn.{name}.bias"] = b[i * d:(i + 1) * d]
    out[dst + "self_attn.out_proj.weight"] = sd[src + "attn.out_proj.weight"]
    out[dst + "self_attn.out_proj.bias"] = sd[src + "attn.out_proj.bias"]


def convert_block(sd: dict, out: dict, src: str, dst: str) -> None:
    split_attention(sd, out, src, dst)
    for a, b in (("ln_1", "layer_norm1"), ("ln_2", "layer_norm2"),
                 ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2")):
        out[f"{dst}{b}.weight"] = sd[f"{src}{a}.weight"]
        out[f"{dst}{b}.bias"] = sd[f"{src}{a}.bias"]


def convert(sd: dict) -> tuple[dict, dict]:
    """Return (HF-named tensors, architecture facts read off the shapes)."""
    out: dict = {}
    if "visual.conv1.weight" not in sd or "visual.proj" not in sd:
        sys.exit("not an open_clip ViT checkpoint (no visual.conv1 / visual.proj); "
                 "ModifiedResNet and timm towers are not supported")

    # Vision tower.
    v = "vision_model."
    out[v + "embeddings.patch_embedding.weight"] = sd["visual.conv1.weight"]
    out[v + "embeddings.class_embedding"] = sd["visual.class_embedding"]
    out[v + "embeddings.position_embedding.weight"] = sd["visual.positional_embedding"]
    out[v + "pre_layrnorm.weight"] = sd["visual.ln_pre.weight"]
    out[v + "pre_layrnorm.bias"] = sd["visual.ln_pre.bias"]
    nv = count_layers(sd, "visual.transformer.resblocks.")
    for i in range(nv):
        convert_block(sd, out, f"visual.transformer.resblocks.{i}.", f"{v}encoder.layers.{i}.")
    out[v + "post_layernorm.weight"] = sd["visual.ln_post.weight"]
    out[v + "post_layernorm.bias"] = sd["visual.ln_post.bias"]
    out["visual_projection.weight"] = sd["visual.proj"].t()

    # Text tower.
    t = "text_model."
    out[t + "embeddings.token_embedding.weight"] = sd["token_embedding.weight"]
    out[t + "embeddings.position_embedding.weight"] = sd["positional_embedding"]
    nt = count_layers(sd, "transformer.resblocks.")
    for i in range(nt):
        convert_block(sd, out, f"transformer.resblocks.{i}.", f"{t}encoder.layers.{i}.")
    out[t + "final_layer_norm.weight"] = sd["ln_final.weight"]
    out[t + "final_layer_norm.bias"] = sd["ln_final.bias"]
    out["text_projection.weight"] = sd["text_projection"].t()
    out["logit_scale"] = sd["logit_scale"]

    conv = sd["visual.conv1.weight"]
    facts = {
        "vision_layers": nv,
        "vision_width": conv.shape[0],
        "patch_size": conv.shape[2],
        "vision_tokens": sd["visual.positional_embedding"].shape[0],
        "vision_mlp": sd["visual.transformer.resblocks.0.mlp.c_fc.weight"].shape[0],
        "text_layers": nt,
        "text_width": sd["token_embedding.weight"].shape[1],
        "vocab_size": sd["token_embedding.weight"].shape[0],
        "context_length": sd["positional_embedding"].shape[0],
        "text_mlp": sd["transformer.resblocks.0.mlp.c_fc.weight"].shape[0],
        "embed_dim": sd["visual.proj"].shape[1],
    }
    return out, facts


def build_config(oc: dict, facts: dict, act: str) -> dict:
    mc = oc["model_cfg"]
    vc, tc = mc["vision_cfg"], mc["text_cfg"]
    grid = vc["image_size"] // vc["patch_size"]
    if 1 + grid * grid != facts["vision_tokens"] or vc["patch_size"] != facts["patch_size"]:
        sys.exit("open_clip_config.json disagrees with the checkpoint's vision shapes")
    if tc["width"] != facts["text_width"] or tc["layers"] != facts["text_layers"]:
        sys.exit("open_clip_config.json disagrees with the checkpoint's text shapes")
    # open_clip's defaults: 64-wide heads in the vision tower (heads = width /
    # head_width), the text config's own `heads`.
    v_heads = facts["vision_width"] // vc.get("head_width", 64)
    pre = oc.get("preprocess_cfg", {})
    return {
        "architectures": ["CLIPModel"],
        "model_type": "clip",
        "projection_dim": facts["embed_dim"],
        "logit_scale_init_value": 2.6592,
        "converted_from": "open_clip",
        "text_config": {
            "model_type": "clip_text_model",
            "vocab_size": facts["vocab_size"],
            "max_position_embeddings": facts["context_length"],
            "hidden_size": facts["text_width"],
            "num_attention_heads": tc["heads"],
            "num_hidden_layers": facts["text_layers"],
            "intermediate_size": facts["text_mlp"],
            "layer_norm_eps": 1e-5,
            "hidden_act": act,
            "projection_dim": facts["embed_dim"],
            "bos_token_id": 49406,
            "eos_token_id": 49407,
            "pad_token_id": 1,
        },
        "vision_config": {
            "model_type": "clip_vision_model",
            "image_size": vc["image_size"],
            "patch_size": facts["patch_size"],
            "num_channels": 3,
            "hidden_size": facts["vision_width"],
            "num_attention_heads": v_heads,
            "num_hidden_layers": facts["vision_layers"],
            "intermediate_size": facts["vision_mlp"],
            "layer_norm_eps": 1e-5,
            "hidden_act": act,
            "projection_dim": facts["embed_dim"],
        },
        "preprocess": {
            "image_mean": pre.get("mean", [0.48145466, 0.4578275, 0.40821073]),
            "image_std": pre.get("std", [0.26862954, 0.26130258, 0.27577711]),
            "interpolation": pre.get("interpolation", "bicubic"),
            "resize_mode": pre.get("resize_mode", "shortest"),
        },
    }


def _features(x):
    # transformers 5 returns a ModelOutput whose pooler_output is the
    # projected feature; 4.x returned the tensor itself.
    return x if torch.is_tensor(x) else x.pooler_output


def verify(src: Path, out_dir: Path, act: str) -> None:
    import open_clip
    from transformers import CLIPConfig, CLIPModel

    torch.manual_seed(0)
    ref, _, _ = open_clip.create_model_and_transforms(f"local-dir:{src}")
    ref.eval()
    tok = open_clip.get_tokenizer(f"local-dir:{src}")
    cfg = CLIPConfig.from_pretrained(str(out_dir))
    hf = CLIPModel.from_pretrained(str(out_dir), config=cfg, dtype=torch.float32).eval()

    size = cfg.vision_config.image_size
    img = torch.randn(1, 3, size, size)
    prompts = ["a photo of a bird.", "a photo of a mammal.", "Zenaida macroura"]
    ids = tok(prompts)
    with torch.no_grad():
        ri = torch.nn.functional.normalize(ref.encode_image(img), dim=-1)
        rt = torch.nn.functional.normalize(ref.encode_text(ids), dim=-1)
        # HF wants EOT padding and an attention mask is unnecessary (causal,
        # pooled at the first EOT); swap open_clip's 0 padding for EOT.
        hf_ids = ids.clone()
        hf_ids[hf_ids == 0] = 49407
        hi = torch.nn.functional.normalize(_features(hf.get_image_features(pixel_values=img)), dim=-1)
        ht = torch.nn.functional.normalize(_features(hf.get_text_features(input_ids=hf_ids)), dim=-1)
    di = (ri - hi).abs().max().item()
    dt = (rt - ht).abs().max().item()
    print(f"verify ({act}): image max|diff| {di:.2e}, text max|diff| {dt:.2e}")
    if max(di, dt) > 1e-4:
        sys.exit("verify FAILED: converted weights disagree with open_clip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--act", choices=("gelu", "quick_gelu"), default=None,
                    help="activation to record (default: from open_clip_config.json)")
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    from safetensors.torch import save_file

    oc = json.loads((args.src / "open_clip_config.json").read_text())
    act = args.act or ("quick_gelu" if oc["model_cfg"].get("quick_gelu") else "gelu")
    sd = load_state_dict(args.src)
    tensors, facts = convert(sd)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tensors = {k: v.to(dtype).contiguous() for k, v in tensors.items()}
    cfg = build_config(oc, facts, act)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "model.safetensors"),
              metadata={"format": "pt", "converted_from": "open_clip",
                        "converter": "brolm/scripts/convert-openclip.py"})
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in ("vocab.json", "merges.txt"):
        if (args.src / name).exists():
            shutil.copyfile(args.src / name, args.out / name)
    print(f"wrote {args.out}: {len(tensors)} tensors, vision {facts['vision_layers']}x"
          f"{facts['vision_width']} /{facts['patch_size']}, text {facts['text_layers']}x"
          f"{facts['text_width']}, embed {facts['embed_dim']}, act {act}")
    if args.verify:
        verify(args.src, args.out, act)


if __name__ == "__main__":
    main()
