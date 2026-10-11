#!/usr/bin/env python3
"""Regenerate the BioCLIP / BioCLIP 2 parity fixtures.

Validates brolm's CLIP towers (src/clip.cpp, src/clip_image.cpp,
src/clip_score.cpp) on open_clip checkpoints converted by
scripts/convert-openclip.py, against the genuine upstream model: open_clip's
own CLIP built from the imageomics release (open_clip_config.json + its
weights), FP32 on the CPU. For each model it dumps:

    * the architecture (so the C++ test needs nothing but the converted
      model.safetensors) and the activation open_clip built,
    * a deterministic synthetic 224 x 224 RGB image (gradients, rings and a
      dark ellipse; no photograph) and the normalised pixel tensor open_clip's
      ToTensor + Normalize makes of it. The image is already 224 square, so
      resize and centre crop are identities and only the normalisation runs:
      the comparison is of the model, not of two resamplers,
    * vision stages: the post-LN CLS feature after the first block alone (a
      1-layer truncation: ln_pre, block 0, ln_post), after all blocks, and the
      projected, L2-normalised image embedding,
    * per prompt: the open_clip token ids up to and including EOT, the
      ln_final hidden states over those positions, the 1-layer truncated
      pooled feature, and the projected, normalised text embedding,
    * the cosine similarities, exp(logit_scale) and the softmax over prompts.

Prompts follow the BioCLIP recipe (taxonomic name + common name) plus a plain
one and a mixed-case one with punctuation for the tokenizer.

The fixtures are gitignored (*.bin): regenerate locally to enable the numeric
checks in tests/test_bioclip_parity.cpp (it skips when they are absent).

Requirements: torch, open_clip_torch, numpy (CPU). The open_clip repos:
    hf download imageomics/bioclip   --local-dir ~/models/bioclip
    hf download imageomics/bioclip-2 --local-dir ~/models/bioclip-2

Usage: tests/ref/gen_bioclip_fixture.py [--bioclip DIR] [--bioclip2 DIR]

Fixture layout (little-endian, the CLAP fixtures' format): the magic
"BIOCFIX1", int32 record count, then per record: int32 name length, the name
bytes, int32 dtype (0 f32, 1 i32, 2 utf-8 bytes), int32 ndim, int32 dims[ndim],
the data.
"""
import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "tests", "fixtures")

PROMPTS = [
    "a photo of Animalia Chordata Aves Columbiformes Columbidae Zenaida macroura "
    "with common name Mourning Dove.",
    "a photo of Animalia Chordata Aves Pelecaniformes Ardeidae Ardea herodias "
    "with common name Great Blue Heron.",
    "a photo of Animalia Chordata Mammalia Rodentia Sciuridae Tamias striatus "
    "with common name Eastern Chipmunk.",
    "a photo of a bird.",
    "Black-capped Chickadee, Poecile atricapillus!",
]


def synthetic_image(size):
    """RGB uint8 (size, size, 3): smooth gradients, concentric rings and a dark
    ellipse, enough structure that every patch differs."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    r = 0.15 + 0.7 * x
    g = 0.2 + 0.6 * y * (1.0 - 0.3 * x)
    d = np.sqrt((x - 0.62) ** 2 + (y - 0.4) ** 2)
    b = 0.5 + 0.4 * np.sin(d * 40.0)
    ell = ((x - 0.4) / 0.22) ** 2 + ((y - 0.62) / 0.12) ** 2 < 1.0
    rgb = np.stack([r, g, b], axis=-1)
    rgb[ell] *= 0.25
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


class Writer:
    def __init__(self):
        self.recs = []

    def add(self, name, arr, dtype=None):
        if isinstance(arr, str):
            data = arr.encode("utf-8")
            self.recs.append((name, 2, [len(data)], data))
            return
        a = np.asarray(arr)
        if dtype == "i32" or a.dtype.kind in "iub":
            a = a.astype("<i4")
            code = 1
        else:
            a = a.astype("<f4")
            code = 0
        self.recs.append((name, code, list(a.shape) or [1], a.tobytes()))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"BIOCFIX1")
            f.write(struct.pack("<i", len(self.recs)))
            for name, code, dims, data in self.recs:
                nb = name.encode("utf-8")
                f.write(struct.pack("<i", len(nb)))
                f.write(nb)
                f.write(struct.pack("<ii", code, len(dims)))
                f.write(struct.pack("<%di" % len(dims), *dims))
                f.write(data)


def vision_stages(model, pixels):
    """(1-layer pooled, full pooled, embedding) for one (1, 3, S, S) input,
    walking open_clip's VisionTransformer by hand so the truncation is the
    same computation as the C++ num_layers=1 encoder."""
    v = model.visual
    x = v.conv1(pixels)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x)
    blocks = v.transformer.resblocks
    one = v.ln_post(blocks[0](x)[:, 0])
    for blk in blocks:
        x = blk(x)
    pooled = v.ln_post(x[:, 0])
    emb = F.normalize(pooled @ v.proj, dim=-1)
    ref = F.normalize(model.encode_image(pixels), dim=-1)
    assert torch.allclose(emb, ref, atol=1e-6), "manual vision walk != encode_image"
    return one[0], pooled[0], emb[0]


def text_stages(model, ids):
    """Per prompt: (ln_final hidden (n, D) up to EOT, 1-layer pooled, embedding)."""
    eot = ids.argmax(dim=-1)
    x = model.token_embedding(ids) + model.positional_embedding
    blocks = model.transformer.resblocks
    one = model.ln_final(blocks[0](x, attn_mask=model.attn_mask))
    for blk in blocks:
        x = blk(x, attn_mask=model.attn_mask)
    hidden = model.ln_final(x)
    rows = torch.arange(ids.shape[0])
    emb = F.normalize(hidden[rows, eot] @ model.text_projection, dim=-1)
    ref = F.normalize(model.encode_text(ids), dim=-1)
    assert torch.allclose(emb, ref, atol=1e-6), "manual text walk != encode_text"
    out = []
    for i in range(ids.shape[0]):
        n = int(eot[i]) + 1
        out.append((hidden[i, :n], one[i, eot[i]], emb[i]))
    return out


def generate(src, name):
    import json
    import open_clip

    torch.set_grad_enabled(False)
    model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{src}")
    model.eval()
    tok = open_clip.get_tokenizer(f"local-dir:{src}")
    oc = json.load(open(os.path.join(src, "open_clip_config.json")))["model_cfg"]
    quick = bool(oc.get("quick_gelu", False))
    act = model.visual.transformer.resblocks[0].mlp.gelu
    assert isinstance(act, open_clip.transformer.QuickGELU) == quick, "activation not as configured"

    v = model.visual
    size = v.image_size[0] if isinstance(v.image_size, (tuple, list)) else v.image_size
    patch = v.conv1.kernel_size[0]
    vw = v.conv1.out_channels
    tw = model.token_embedding.weight.shape[1]
    w = Writer()
    w.add("arch/vision", [size, patch, vw, v.transformer.resblocks[0].attn.num_heads,
                          len(v.transformer.resblocks),
                          v.transformer.resblocks[0].mlp.c_fc.out_features], "i32")
    w.add("arch/text", [model.token_embedding.weight.shape[0], model.positional_embedding.shape[0],
                        tw, model.transformer.resblocks[0].attn.num_heads,
                        len(model.transformer.resblocks),
                        model.transformer.resblocks[0].mlp.c_fc.out_features], "i32")
    w.add("arch/embed_dim", [v.proj.shape[1]], "i32")
    w.add("arch/hidden_act", "quick_gelu" if quick else "gelu")

    rgb = synthetic_image(size)
    # What open_clip's eval transform does to an RGB image that is already
    # size x size: ToTensor (/255) then Normalize; check against the real one.
    from PIL import Image
    pre = json.load(open(os.path.join(src, "open_clip_config.json"))).get("preprocess_cfg", {})
    mean = torch.tensor(pre.get("mean", [0.48145466, 0.4578275, 0.40821073])).view(3, 1, 1)
    std = torch.tensor(pre.get("std", [0.26862954, 0.26130258, 0.27577711])).view(3, 1, 1)
    pixels = ((torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0) - mean) / std
    via = preprocess(Image.fromarray(rgb))
    assert torch.allclose(pixels, via, atol=1e-5), "hand normalisation != open_clip preprocess"
    pixels = pixels.unsqueeze(0)

    w.add("image/rgb", rgb, "i32")
    w.add("image/pixels", pixels[0].numpy())
    one, pooled, img_emb = vision_stages(model, pixels)
    w.add("image/pooled_l1", one.numpy())
    w.add("image/pooled", pooled.numpy())
    w.add("image/embed", img_emb.numpy())

    w.add("prompt_texts", "\n".join(PROMPTS))
    ids = tok(PROMPTS)
    stages = text_stages(model, ids)
    text_embs = []
    for i, (hidden, one_t, emb) in enumerate(stages):
        n = hidden.shape[0]
        w.add(f"prompt/{i}/ids", ids[i, :n].numpy(), "i32")
        w.add(f"prompt/{i}/hidden", hidden.numpy())
        w.add(f"prompt/{i}/pooled_l1", one_t.numpy())
        w.add(f"prompt/{i}/embed", emb.numpy())
        text_embs.append(emb)

    T = torch.stack(text_embs)
    sim = img_emb @ T.T
    scale = model.logit_scale.exp().item()
    probs = (sim * scale).softmax(dim=-1)
    w.add("sim", sim.numpy())
    w.add("logit_scale", [scale])
    w.add("probs", probs.numpy())
    out = os.path.join(OUT_DIR, f"{name}.bin")
    w.save(out)
    np.set_printoptions(precision=4, suppress=True)
    print(f"{name}: act {'quick_gelu' if quick else 'gelu'}, scale {scale:.2f}, sim {sim.numpy()}, "
          f"probs {probs.numpy()}")
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bioclip", default="~/models/bioclip")
    ap.add_argument("--bioclip2", default="~/models/bioclip-2")
    args = ap.parse_args()
    for src, name in ((args.bioclip, "bioclip"), (args.bioclip2, "bioclip-2")):
        src = os.path.expanduser(src)
        if os.path.exists(os.path.join(src, "open_clip_config.json")):
            generate(src, name)
        else:
            print(f"skip {name}: no open_clip_config.json in {src}")


if __name__ == "__main__":
    main()
