# CLAUDE.md — brolm

Project-specific guidance for Claude Code. Read this before editing.

## What this repo is

`brolm` is the language- and text-model inference library for the **bro** stack —
the text counterpart to `brodiffusion`. Pure C++20. Owns tokenizers, transformer
building blocks (encoder + decoder), safetensors / HF weight loaders, sampling
and KV-cache, and the trainable alignment adapter that retargets LLM hidden
states into diffusion conditioning.

CPU-by-default (FP32 scalar backend); a GPU backend (FP16) is enabled by
forwarding `BROTENSOR_WITH_CUDA=ON` or `BROTENSOR_WITH_METAL=ON` to brotensor.
brolm itself ships **no GPU kernels** — it composes brotensor ops.

## Sibling dependencies

Three sibling repos, resolved at `../<name>` with a `third_party/<name>`
fallback (the `bro/docs/multi-repo-workflow.md` pattern):

- **bromath** — header-only scalar / RNG helpers.
- **brotensor** — tensors + compute kernels (CPU + optional GPU backend).
- **broimage** — host-side image decode and geometric resampling. Used by the
  Qwen3.5-VL CLI driver and any future multimodal preprocessing that needs to
  touch raw image files. Pixel-tensor preprocessing inside the model
  (smart-resize → patches → M-RoPE) lives in `qwen35_preprocessor.h` and is
  brolm's, not broimage's.

Override paths with `-DBROMATH_DIR=...`, `-DBROTENSOR_DIR=...`,
`-DBROIMAGE_DIR=...`.

## Build & test

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

CMake options: `BROLM_TESTS` (default ON standalone), `BROLM_TOOLS` (default ON
standalone), `BROLM_INSTALL` (default OFF — brolm is consumed via
`add_subdirectory`, not `find_package`; no package config is generated, by
design — see the comment in `CMakeLists.txt`).

Tests live in `tests/`, one file per public header. They are plain
`int main()` executables (manual asserts + nonzero exit on failure — no gtest)
registered with `add_test`, so `ctest` drives them. CPU-only tests are
unconditional; tests gated on real HF / GGUF checkpoints look for files under
`weights/` (or an env-var path override) and skip cleanly when absent.

The ad-hoc CLI driver `tools/run_qwen35_image.cpp` (binary:
`brolm_run_qwen35_image`) is **not** run by ctest — it's for eyeballing the
full image → reply pipeline.

## Source layout

```
include/brolm/
  version.h                — library version constants + version_string()
  detail/                  — internal helpers (not part of the public surface)
    byte_level_bpe.h       — shared GPT-2-family BPE core (CLIP, Qwen, Whisper)
    weights.h              — uniform Source over safetensors + gguf (incl. quant
                             dequant); model loaders stay container-agnostic
    compute.h, device.h    — backend dispatch glue
    json.h                 — minimal JSON reader for HF config files
  tokenizer.h              — CLIP BPE
  tokenizer_t5.h           — T5 SentencePiece Unigram
  tokenizer_nllb.h         — NLLB-200 SPM Unigram + FLORES-200 language codes
  detail/spm_unigram.h     — shared SentencePiece Unigram core (T5, NLLB)
  qwen_tokenizer.h         — Qwen3 byte-level BPE
  qwen35_tokenizer.h       — Qwen3.5-VL (Qwen3 BPE + 33 vision/think/tool specials)
  whisper_tokenizer.h      — Whisper byte-level BPE + lang/task/timestamp specials
  clip.h, clip_image.h, clip_score.h
  t5.h
  nllb_config.h, nllb.h     — NLLB-200 (M2M-100) encoder-decoder translation
                              (encoder + decoder w/ cross-attn + beam search +
                              Translator); sinusoidal positions, LayerNorm, ReLU
  qwen.h, qwen_generate.h
  qwen35_config.h, qwen35_preprocessor.h, qwen35_vision.h,
  qwen35_text.h, qwen35_vl.h
  alignment_adapter.h
src/                       — one .cpp per public header (CLIP tokenizer is
                             `tokenizer_clip.cpp` because `tokenizer.h` is the
                             generic name); plus `version.cpp` and
                             `detail/byte_level_bpe.cpp`
tests/                     — one `int main()` test per header, run via ctest
tools/                     — ad-hoc driver binaries
scripts/                   — shell-only helpers (e.g. weight download)
weights/                   — real HF checkpoints used by gated tests (gitignored)
```

When adding a new model family: header in `include/brolm/`, impl in `src/`, test
in `tests/`, wire all three into `CMakeLists.txt` (top-level `target_sources`
and `tests/CMakeLists.txt`).

## Conventions

- **C++20, pure C++.** No Python anywhere in this stack — not for scripts, not
  for validation, not for weight conversion. Shell or C++ only.
- **No CUDA/Metal in this repo.** GPU paths must go through brotensor ops; if a
  kernel is missing, add it to brotensor.
- **Tokenizers share `brolm::detail::bpe`.** New byte-level-BPE families should
  reuse it and add only their pre-tokenization regex and special-token table.
- **Load HF safetensors directly** — no offline conversion step. Tokenizers
  read `vocab.json` + `merges.txt` (or `tokenizer.json`); models read the
  sharded safetensors + `config.json` from a directory. `.gguf` checkpoints
  (llama.cpp format) load through the same model classes via `from_gguf` /
  `load_weights(gguf::File)`; both containers go through the
  `brolm::detail::weights::Source` adapter so loaders stay format-agnostic.
  Supported via GGUF today: Qwen3 (model + tokenizer + config), T5 (model +
  tokenizer + config), Whisper (tokenizer). On-disk quants (Q4_K / Q6_K / Q8_0)
  keep their dtype and dispatch through brotensor's CUDA quant kernels.
- **Numerical parity work in progress.** Qwen3.5-VL passes structural and
  no-NaN checks against the real 0.8B checkpoint; bit-exact parity vs HF
  `transformers` is not yet asserted end-to-end. When touching attention,
  RoPE, RMSNorm, or quantization paths, prefer adding a parity test over
  hand-checking a single trace.

## Git / workflow

- Commit directly to `main` unless explicitly asked to branch.
- One capability per commit. Describe commits by what they add or fix — never
  use "phase 1/2/N" language in commit messages, comments, or PR descriptions.
- For large mechanical changes (e.g. sweeping a new tokenizer through every
  call site), chunk into ~2k-LOC pieces and commit each chunk.
