# Qwen3.5-VL

A vision-language model: image plus text in, generated text out. brolm loads the
official `Qwen/Qwen3.5-*` Hugging Face safetensors directly — there is no
conversion step.

```cpp
#include "brolm/qwen35_vl.h"

brolm::qwen35::VLMConfig cfg;
cfg.max_new_tokens = 64;
brolm::qwen35::VLM vlm(cfg);
vlm.load_from_directory("weights/Qwen3.5-0.8B");

// (3, H, W) float pixels in [0, 1] — the caller owns image decoding.
brolm::qwen35::ImageInput img{ pixels.data(), H, W };

const std::string prompt =
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|>\n"
    "<|im_start|>assistant\n";

std::string out = vlm.generate(prompt, { img });
```

`qwen35_vl.h` is the top-level driver — tokenize, run the vision tower, splice the
image embeddings into the text stream, prefill, sample.

## Architecture

**Vision tower** (`qwen35_vision.h`) — a 12-block ViT. The learned 48×48 position
table is bilinearly interpolated to whatever patch grid the runtime image produces.
Attention is qkv with 2D rotary embeddings; the MLP is GELU-tanh. A patch merger
2×2-shuffles the output to the text hidden size.

**Hybrid text backbone** (`qwen35_text.h`) — 24 layers on a `[L,L,L,F]×6` schedule,
read from `config.json`'s `layer_types`. The two layer kinds differ:

- *Full-attention* layers use GQA (8 query heads, 2 KV heads), per-head q/k RMSNorm,
  partial M-RoPE (rotating 64 of 256 dims, `mrope_section=[11,11,10]` over `(t,h,w)`),
  and a sigmoid attention-output gate. They carry a KV-cache.
- *Linear-attention* layers run the Gated DeltaNet recurrence behind a depthwise
  causal conv1d front-end. They carry recurrent state plus a conv shift buffer
  instead of a KV-cache.

**Preprocessor** (`qwen35_preprocessor.h`) — mirrors HF's `Qwen2VLImageProcessorFast`.
It smart-resizes to a multiple of `patch_size * merge_size`, clamps to the min/max
pixel budget, normalizes with mean/std = 0.5, patchifies in HF's exact
merger-block-major order (with the temporal axis duplicated for a static image), and
emits the per-token M-RoPE `(t, h, w)` streams for the whole image+text sequence.

## Weights

Requires an authenticated `hf` CLI.

```bash
scripts/download_qwen35.sh           # Qwen/Qwen3.5-0.8B by default
REPO=Qwen/Qwen3.5-2B scripts/download_qwen35.sh
```

## CLI driver

`tools/` carries an ad-hoc driver that runs the pipeline against a real image file:

```bash
build/tools/Release/brolm_run_qwen35_image \
    weights/Qwen3.5-0.8B path/to/image.png "Describe the image."
```

It decodes the image through broimage and area-resamples it down to a pixel cap
(~512×512 by default) before handing it to the model: vision token count grows
linearly with pixels, and ViT attention cost grows quadratically with tokens, so an
uncapped image gets expensive fast.

## Validation

The checkpoint-gated test suite loads the official 0.8B safetensors and exercises
every stage — config, tokenizer, preprocessor, vision tower, hybrid text
prefill/decode parity, and the full VLM end to end — asserting finite outputs
throughout. It does not assert bit-exact numerical parity against HF
`transformers`; the tests check structure and finiteness, not agreement with the
reference implementation to the last bit.

The MTP head in the checkpoint is loaded by the text model but is not wired into a
speculative-decoding pass, so it does no work at inference time.
