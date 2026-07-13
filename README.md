# brolm

[![CI](https://github.com/wlejon/brolm/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brolm/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/brolm/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brolm/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Language- and text-model inference for the bro stack — the text counterpart to
[brodiffusion](https://github.com/wlejon/brodiffusion). Pure C++20, built on
[brotensor](https://github.com/wlejon/brotensor) (tensor + compute kernels) and
[bromath](https://github.com/wlejon/bromath) (scalar / RNG helpers). Runs
**CPU-by-default and on a GPU when one is available** — FP32 on the CPU backend,
FP16 on a GPU — with the device chosen at runtime.

brolm turns token sequences into embeddings (encoder models) and generates them
(decoder models). It owns the tokenizers, the transformer building blocks, and
the safetensors / Hugging Face weight loaders. brodiffusion depends on brolm for
its default text encoders. Host-side image decoding and resampling for
multimodal models go through [broimage](https://github.com/wlejon/broimage).

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
| `brolm/alignment_adapter.h` | Trainable adapter: LLM hidden states → diffusion-conditioning tensors |

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

bromath, brotensor, and broimage are resolved as standalone sibling repos at
`../bromath`, `../brotensor`, and `../broimage`, with a `third_party/` submodule
fallback — the pattern documented in `bro/docs/multi-repo-workflow.md`. Override
any of them with `-DBROMATH_DIR=...`, `-DBROTENSOR_DIR=...`,
`-DBROIMAGE_DIR=...`. Pass `-DBROTENSOR_WITH_CUDA=ON` or
`-DBROTENSOR_WITH_METAL=ON` to forward the GPU backend selection to brotensor.

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

Supported today: Qwen3 (model + tokenizer + config), T5 (model + tokenizer +
config), Whisper (tokenizer). BF16 weights load on every backend; on-disk
quants (Q4_K / Q6_K / Q8_0) are kept in their original dtype and dispatched
through brotensor's quant-carrier kernels (CUDA-only at the moment). Dense
tensors whose downstream op is dense-only (embedding lookup, RMSNorm gamma) are
dequantised to the compute dtype on load.

A helper script pulls the smallest text-only Qwen3 in both BF16 and Q8_0:

```bash
scripts/download_qwen3_gguf.sh                              # unsloth/Qwen3-0.6B-GGUF
REPO=Qwen/Qwen3-1.7B-GGUF scripts/download_qwen3_gguf.sh    # different size
```

## Qwen3.5-VL

Qwen3.5-VL is the first multimodal model fully supported in brolm. The pipeline
loads the official `Qwen/Qwen3.5-*` Hugging Face safetensors directly — no
conversion step.

```cpp
#include "brolm/qwen35_vl.h"

brolm::qwen35::VLMConfig cfg;
cfg.max_new_tokens = 64;
brolm::qwen35::VLM vlm(cfg);
vlm.load_from_directory("weights/Qwen3.5-0.8B");

// (3, H, W) float pixels in [0, 1] — caller owns image decoding.
brolm::qwen35::ImageInput img{ pixels.data(), H, W };

const std::string prompt =
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|>\n"
    "<|im_start|>assistant\n";

std::string out = vlm.generate(prompt, { img });
```

Architecture notes:
- **Vision tower** (`qwen35_vision.h`): 12-block ViT, learned 48×48 position
  table bilinearly interpolated to the runtime patch grid, qkv + 2D rotary
  attention, GELU-tanh MLP, patch merger that 2×2-shuffles to text hidden size.
- **Hybrid text backbone** (`qwen35_text.h`): 24 layers in a `[L,L,L,F]×6`
  schedule (read from `config.json`'s `layer_types`). Full-attention layers use
  GQA (8 q-heads, 2 kv-heads), per-head q/k RMSNorm, partial M-RoPE (rotates
  64/256 dims with `mrope_section=[11,11,10]` over (t,h,w)), and a sigmoid
  attention-output gate. Linear-attention layers run the Gated DeltaNet
  recurrence with a depthwise causal conv1d front-end. KV-cache for full layers;
  recurrent + conv-shift state for linear layers.
- **Preprocessor** (`qwen35_preprocessor.h`): mirrors HF
  `Qwen2VLImageProcessorFast` — smart-resize to a multiple of `patch_size *
  merge_size`, clamp to the min/max pixel budget, normalize with mean/std=0.5,
  patchify in HF's exact merger-block-major order with the temporal axis
  duplicated for static images, and emit per-token M-RoPE `(t, h, w)` streams
  for the full image+text sequence.

Weight download (requires `hf` CLI authenticated):

```bash
scripts/download_qwen35.sh           # Qwen/Qwen3.5-0.8B by default
REPO=Qwen/Qwen3.5-2B scripts/download_qwen35.sh
```

An ad-hoc CLI driver under `tools/` runs the full pipeline against a real image
file:

```bash
build/tools/Release/brolm_run_qwen35_image \
    weights/Qwen3.5-0.8B path/to/image.png "Describe the image."
```

It decodes the image via broimage, area-resamples it down to a per-tool pixel
cap (~512×512 by default — vision token count is linear in pixels and ViT
attention cost quadratic in tokens), and prints the model's reply.

Status: the unconditional CPU test suite passes, and the gated real-checkpoint
suite loads the official 0.8B safetensors and runs every stage (config /
tokenizer / preprocessor / vision tower / hybrid text prefill+decode parity /
full VLM end-to-end) without NaN. Bit-exact numerical parity against HF `transformers`
is not yet asserted — that's the remaining validation step before this is
production-ready. The MTP head present in the checkpoint is loaded by the text
model but not yet wired into a speculative-decoding pass.

## CI

Builds and tests on Linux (GCC + Clang), Windows (MSVC) and macOS/arm64. Each job
checks out bromath, brotensor and broimage alongside this repo and builds the whole
stack from source, so a breaking change in a sibling fails here rather than in
whoever next builds brolm by hand.

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
someone else built. The siblings are built ahead of the traced build so their
findings stay in their own repos.

## Versioning

Pre-1.0. Siblings vendor this repo via `add_subdirectory` and build from source,
so a tag is a pin point rather than a compatibility promise.

## License

[MIT](LICENSE)