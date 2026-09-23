"""Reference fixture for bro.lm.loadModernBert: tests/ref/modernbert_encode.json.

Runs Hugging Face's ModernBertModel over the encoder of a Laya checkpoint
(encoder/config.json + the "encoder." tensors of model.safetensors, the
tokenizer under tokenizer/) and records, per text:

  - the input ids as the HF tokenizer produces them with
    add_special_tokens=True ([CLS] ... [SEP]),
  - last_hidden_state rows 0 (the [CLS] row) and L-1, and the mean over all
    rows, in fp32 (no autocast: the numeric truth).

One text is longer than the 128-token local-attention window, so the
sliding layers are exercised.

Usage (needs torch + transformers + safetensors; CUDA optional):
    USE_TF=0 python tests/ref/gen_modernbert_encode.py [LAYA_DIR]
LAYA_DIR defaults to ../laya relative to the brolm checkout.
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
LAYA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(HERE, "..", "..", "..", "laya"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel  # noqa: E402

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    " ".join("Sentence number %d talks about invoices, refunds and a duplicate charge." % i
             for i in range(20)),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModernBertConfig.from_json_file(os.path.join(LAYA_DIR, "encoder", "config.json"))
    cfg._attn_implementation = "sdpa"
    model = ModernBertModel(cfg)
    state = {}
    with safe_open(os.path.join(LAYA_DIR, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("encoder."):
                state[k[len("encoder."):]] = f.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    if missing or unexpected:
        sys.exit("state dict mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    model = model.to(dev).float().eval()
    tok = AutoTokenizer.from_pretrained(os.path.join(LAYA_DIR, "tokenizer"))

    cases = []
    for text in TEXTS:
        ids = tok(text, add_special_tokens=True)["input_ids"]
        with torch.no_grad():
            h = model(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0].float().cpu()
        cases.append({
            "text": text,
            "ids": ids,
            "first": [round(float(x), 6) for x in h[0]],
            "last": [round(float(x), 6) for x in h[-1]],
            "mean": [round(float(x), 6) for x in h.mean(0)],
        })
        print("%3d tokens  |h0|=%.4f" % (len(ids), float(h[0].norm())))

    out = os.path.join(HERE, "modernbert_encode.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": "laya (english) encoder", "hidden_size": cfg.hidden_size, "cases": cases},
                  f, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
