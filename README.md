# brolm

Language- and text-model inference for the bro stack — the text counterpart to
[brodiffusion](https://github.com/wlejon/brodiffusion). Pure C++20, built on
[brotensor](https://github.com/wlejon/brotensor) (tensor + compute kernels) and
[bromath](https://github.com/wlejon/bromath) (scalar / RNG helpers). Runs
**CPU-by-default and on a GPU when one is available** — FP32 on the CPU backend,
FP16 on a GPU — with the device chosen at runtime.

brolm turns token sequences into embeddings (encoder models) and generates them
(decoder models). It owns the tokenizers, the transformer building blocks, and
the safetensors / Hugging Face weight loaders. brodiffusion depends on brolm for
its default text encoders.

## Scope

- **Tokenizers** — BPE (CLIP), Unigram/SentencePiece (T5), behind one interface.
- **Transformer LM core** — encoder-style (CLIP text, T5) and decoder-style
  (LLMs), composed from brotensor ops.
- **Alignment adapters** — trainable projection/pooling that retargets an
  encoder or LLM's hidden states into a diffusion denoiser's conditioning.
- **Generation** — sampling and KV-cache for decoder LLMs.

## Components

| Header | Purpose |
|---|---|
| `brolm/tokenizer.h` | CLIP BPE tokenizer (`vocab.json` + `merges.txt`) |
| `brolm/tokenizer_t5.h` | T5 SentencePiece Unigram tokenizer (`tokenizer.json`) |
| `brolm/clip.h` | CLIP ViT-L/14 text encoder |
| `brolm/clip_image.h` | CLIP ViT-L/14 vision encoder |
| `brolm/clip_score.h` | CLIP image/text similarity scoring |
| `brolm/t5.h` | T5-XXL encoder (encoder-only) |

## Build

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

bromath and brotensor are resolved as standalone sibling repos at `../bromath`
and `../brotensor`, with a `third_party/` submodule fallback — the pattern
documented in `bro/docs/multi-repo-workflow.md`.
