---
name: Bug report
about: A model loads wrongly or produces output that differs from the reference, a tokenizer mismatch, a crash, or a test fails
labels: bug
---

**The model:** the checkpoint (Hugging Face repo id, or the GGUF file and its
quant type) and the call — prompt, sampling settings, seed.

**What the reference produces** (Hugging Face transformers or llama.cpp with
the same weights and settings; token ids are the most useful for tokenizer
bugs):

**What brolm produced instead** (the tokens or text, the error, a crash, or
the failing `ctest --output-on-failure` output — paste it):

```
```

**Environment:**
- OS:
- brotensor backend (CPU / CUDA / Metal / Vulkan), GPU and driver version:
- Compiler / toolchain (MSVC / GCC / Clang):
- Called from C++ or from JavaScript (`bro.lm`):
- brolm commit, and brotensor commit if built from siblings:
