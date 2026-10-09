# brolm

[![CI](https://github.com/wlejon/brolm/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brolm/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/brolm/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brolm/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Language- and text-model inference for the bro stack — the text counterpart to
[brodiffusion](https://github.com/wlejon/brodiffusion). Pure C++20, built on
[brotensor](https://github.com/wlejon/brotensor) (tensor + compute kernels) and
[bromath](https://github.com/wlejon/bromath) (scalar / RNG helpers). The device
is picked at runtime — a GPU when one is registered, CPU otherwise — and the
compute dtype follows it: FP16 on a GPU, FP32 on CPU.

brolm turns token sequences into embeddings (encoder models) and generates them
(decoder models). It owns the tokenizers, the transformer building blocks, and
the safetensors / Hugging Face weight loaders. brodiffusion depends on brolm for
its default text encoders. Host-side image decoding and resampling for
multimodal models go through [broimage](https://github.com/wlejon/broimage).

brolm is one of the engine libraries of the
[bro ecosystem](https://github.com/wlejon/bro/blob/main/docs/ecosystem.md):
[bro](https://github.com/wlejon/bro) links it under `BRO_WITH_LM` and exposes it
to apps as `bro.lm` through the JavaScript binding in `src/api/` (`brolm_api`),
which needs [bronze](https://github.com/wlejon/bronze) and
[brass](https://github.com/wlejon/brass); brodiffusion and brosoundml build on it.

**Platforms and backends.** The library builds and its self-contained tests pass
on Windows (MSVC), Linux (GCC and Clang) and macOS (arm64). Models run on any
brotensor backend: CPU (FP32), CUDA, Metal or Vulkan (FP16/BF16). Quantised GGUF
weights need a GPU backend (see GGUF below). CI covers the CPU build only; GPU
inference is checked on hardware that has the weights.

## Scope

- **Tokenizers** — BPE (CLIP), Unigram/SentencePiece (T5), behind one interface.
- **Transformer LM core** — encoder-style (CLIP text, T5) and decoder-style
  (LLMs), composed from brotensor ops.
- **Alignment adapters** — trainable projection/pooling that retargets an
  encoder or LLM's hidden states into a diffusion denoiser's conditioning.
- **Generation** — sampling (greedy / temperature / top-k / top-p), KV-cache,
  and the autoregressive generate loop for decoder LLMs, plus chat-template
  rendering (`<|im_start|>role…<|im_end|>`) on the Qwen3 tokenizer.

## Components

| Header | Purpose |
|---|---|
| `brolm/tokenizer.h` | CLIP BPE tokenizer (`vocab.json` + `merges.txt`) |
| `brolm/tokenizer_t5.h` | T5 SentencePiece Unigram tokenizer (`tokenizer.json`) |
| `brolm/clip.h` | CLIP ViT-L/14 text encoder |
| `brolm/clip_image.h` | CLIP ViT-L/14 vision encoder |
| `brolm/clip_score.h` | CLIP image/text similarity scoring |
| `brolm/t5.h` | T5-XXL encoder (encoder-only) |
| `brolm/qwen_tokenizer.h` | Qwen3 byte-level BPE tokenizer + `apply_chat_template` |
| `brolm/whisper_tokenizer.h` | Whisper byte-level BPE tokenizer (GPT-2 family) + language/task/timestamp specials |
| `brolm/qwen.h` | Qwen3 decoder LLM — GQA, QK-norm, RoPE, SwiGLU, KV-cache |
| `brolm/qwen_generate.h` | Sampling (greedy / temperature / top-k / top-p) + autoregressive generation |
| `brolm/qwen35_config.h` | Qwen3.5-VL typed config (text + vision + multimodal token IDs) |
| `brolm/qwen35_tokenizer.h` | Qwen3.5-VL tokenizer — Qwen3 BPE + the 33 vision/video/think/tool specials |
| `brolm/qwen35_preprocessor.h` | Image smart-resize → patch tensor + grid_thw + M-RoPE position IDs |
| `brolm/qwen35_vision.h` | Qwen3.5-VL ViT vision tower (12 blocks + patch merger) |
| `brolm/qwen35_text.h` | Qwen3.5-VL hybrid text backbone — full-attention layers with attn-output-gate + M-RoPE interleaved with Gated DeltaNet linear-attention layers |
| `brolm/qwen35_vl.h` | Top-level VLM driver: tokenize → vision tower → embed splice → text prefill → sample |
| `brolm/qwen3vl_*.h` | Qwen3-VL (2B–32B dense): config, tokenizer, image preprocessor, vision tower, text decoder, prompt assembly and the top-level `qwen3vl_vl.h` driver |
| `brolm/mistral3_*.h`, `brolm/mistral_tokenizer.h` | Mistral 3.1 (Small 3.1, Pixtral): Tekken tokenizer, Pixtral preprocessor and vision tower, multimodal projector, text decoder, generation, and the `mistral3_vl.h` driver |
| `brolm/gemma2.h`, `gemma2_config.h`, `gemma_tokenizer.h` | Gemma-2 decoder LLM and its SentencePiece-BPE tokenizer |
| `brolm/llama3_tokenizer.h`, `brolm/llm2vec.h` | Llama-3 tokenizer; LLM2Vec, a LLaMA-family decoder used as a bidirectional text encoder |
| `brolm/nllb.h`, `nllb_config.h`, `tokenizer_nllb.h` | NLLB-200 encoder-decoder translation |
| `brolm/modernbert.h`, `modernbert_config.h` | ModernBERT encoder |
| `brolm/laya*.h` | Laya decision model on ModernBERT: choice / score / yes-no questions, a request scheduler packing concurrent callers into one forward per device, and soft-token gradients |
| `brolm/sampler.h`, `brolm/grammar.h`, `brolm/grammar_jit.h` | Samplers and penalties (min-p, DRY, repetition); regex/grammar-constrained decoding with DFA logit masks, JIT-compiled through brass |
| `brolm/alignment_adapter.h` | Trainable adapter: LLM hidden states → diffusion-conditioning tensors |
| `brolm/api.h` | The bronze JavaScript binding (`bro.lm`) |

The CLIP, Qwen3, Qwen3.5-VL, and Whisper tokenizers share a single byte-level
BPE core in `brolm::detail::bpe` (`include/brolm/detail/byte_level_bpe.h`) —
each family-specific tokenizer adds only its own pre-tokenization regex and
special-token table on top.

## Build

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

A plain clone is all it takes. Every dependency — bromath, brotensor, broimage,
and bronze with brass — is pinned to a commit in `CMakeLists.txt`
(`bro_dependency()`, `cmake/bro_deps.cmake`) and resolves the way every repo in
the ecosystem resolves one: an existing target wins (bro adds them first), then
a working tree beside this one (`../bromath`, `../brotensor`, `../broimage`,
`../bronze`, ...), then the pinned commit, fetched at configure. See bro's
[multi-repo workflow](https://github.com/wlejon/bro/blob/main/docs/multi-repo-workflow.md)
for the layout. `-DFETCHCONTENT_SOURCE_DIR_<NAME>=<path>` points one dependency
anywhere else. bronze and brass compile inside this build tree, because the
binding has to be compiled against the same bronze as the program that loads
it. Pass `-DBROTENSOR_WITH_CUDA=ON`,
`-DBROTENSOR_WITH_METAL=ON` or `-DBROTENSOR_WITH_VULKAN=ON` (the AMD path) to
forward the GPU backend selection to brotensor.

CMake options:

- `BROLM_TESTS` (default `ON` when built standalone) — build the unit/integration suite under `tests/`.
- `BROLM_TOOLS` (default `ON` when built standalone) — build the ad-hoc tool drivers under `tools/` (not run by ctest).
- `BROLM_INSTALL` (default `OFF`) — install the static library and public headers. brolm is meant to be consumed via `add_subdirectory`, not `find_package`; no CMake package config is generated.

## GGUF

In addition to Hugging Face safetensors, brolm loads `.gguf` checkpoints
(llama.cpp's format) directly. Tokenizer vocab, merges, and special-token IDs
are read from the file's metadata; model weights are read by their ggml tensor
names. The same model class is used either way — only the loader entry point
differs:

```cpp
brotensor::gguf::File f("weights/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf");

auto tok = brolm::qwen::Tokenizer::from_gguf(f);
auto cfg = brolm::qwen::Qwen3Config::from_gguf(f);
brolm::qwen::Qwen3Model model(cfg);
model.load_weights(f);
```

GGUF covers Qwen3 (model + tokenizer + config), T5 (model + tokenizer + config)
and Whisper (tokenizer). BF16 weights load on every backend. On-disk quants
(Q4_K / Q6_K / Q8_0) are kept in their original dtype and dispatched through
brotensor's quant-carrier kernels, which exist on the GPU backends (CUDA, Metal
and Vulkan) — the CPU backend is FP32-only and cannot run a quantised GGUF. Dense tensors whose downstream op is
dense-only (embedding lookup, RMSNorm gamma) are dequantised to the compute dtype
on load.

A helper script pulls the smallest text-only Qwen3 in both BF16 and Q8_0:

```bash
scripts/download_qwen3_gguf.sh                              # unsloth/Qwen3-0.6B-GGUF
REPO=Qwen/Qwen3-1.7B-GGUF scripts/download_qwen3_gguf.sh    # different size
```

## Multimodal

**[Qwen3.5-VL](docs/qwen35-vl.md)** — image + text in, generated text out. Loads
the official `Qwen/Qwen3.5-*` Hugging Face safetensors directly, with a 12-block
ViT vision tower and a hybrid text backbone that interleaves full-attention layers
with Gated DeltaNet linear-attention layers. `docs/qwen35-vl.md` covers the
pipeline, the architecture, the weight download, and the CLI driver.

```cpp
#include "brolm/qwen35_vl.h"

brolm::qwen35::VLM vlm(cfg);
vlm.load_from_directory("weights/Qwen3.5-0.8B");
std::string out = vlm.generate(prompt, { img });
```

## CI

Builds and tests a plain clone on Linux (GCC + Clang), Windows (MSVC) and
macOS/arm64, building the whole stack from source at the pinned dependency
commits.

What a green run does and does not mean: `weights/` is 86 GB and gitignored, so a
runner never has it. The model tests gate on the checkpoint being present and skip
without it; the rest build their fixtures synthetically and run for real. So green
means "brolm compiles everywhere and its self-contained tests pass" — not that the
models produce correct output. That check needs the weights and stays on hardware
that has them.

Coverage of `src/` + `include/brolm/` lands in each run's job summary
(`-DBROLM_COVERAGE=ON` locally; GCC/Clang only), and understates the loading and
generation paths for the same reason. [CodeQL](.github/workflows/codeql.yml) runs
weekly and on every push: brolm parses safetensors and GGUF checkpoints, tokenizer
vocab/merges, and model config JSON, and indexes buffers using shapes and offsets
those files declare — all of it attacker-shaped input if a user loads a model
someone else built. The dependencies are built ahead of the traced build so their
findings stay in their own repos.

## Versioning

Pre-1.0. Consumers build this repo from source (a working tree beside them or
a pinned commit), so a tag is a pin point rather than a compatibility promise.

## License

[MIT](LICENSE)