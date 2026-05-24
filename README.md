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
| `brolm/qwen_tokenizer.h` | Qwen3 byte-level BPE tokenizer |
| `brolm/qwen.h` | Qwen3 decoder LLM — GQA, QK-norm, RoPE, SwiGLU, KV-cache |
| `brolm/qwen_generate.h` | Sampling (greedy / temperature / top-k / top-p) + autoregressive generation |
| `brolm/qwen35_config.h` | Qwen3.5-VL typed config (text + vision + multimodal token IDs) |
| `brolm/qwen35_tokenizer.h` | Qwen3.5-VL tokenizer — Qwen3 BPE + the 33 vision/video/think/tool specials |
| `brolm/qwen35_preprocessor.h` | Image smart-resize → patch tensor + grid_thw + M-RoPE position IDs |
| `brolm/qwen35_vision.h` | Qwen3.5-VL ViT vision tower (12 blocks + patch merger) |
| `brolm/qwen35_text.h` | Qwen3.5-VL hybrid text backbone — full-attention layers with attn-output-gate + M-RoPE interleaved with Gated DeltaNet linear-attention layers |
| `brolm/qwen35_vl.h` | Top-level VLM driver: tokenize → vision tower → embed splice → text prefill → sample |
| `brolm/alignment_adapter.h` | Trainable adapter: LLM hidden states → diffusion-conditioning tensors |

## Build

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

bromath and brotensor are resolved as standalone sibling repos at `../bromath`
and `../brotensor`, with a `third_party/` submodule fallback — the pattern
documented in `bro/docs/multi-repo-workflow.md`.

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

Status: all 18 unit + integration tests pass; the gated real-checkpoint suite
loads the official 0.8B safetensors and runs every stage (config / tokenizer /
preprocessor / vision tower / hybrid text prefill+decode parity / full VLM
end-to-end) without NaN. Bit-exact numerical parity against HF `transformers`
is not yet asserted — that's the remaining validation step before this is
production-ready. The MTP head present in the checkpoint is loaded by the text
model but not yet wired into a speculative-decoding pass.

## License

[MIT](LICENSE)