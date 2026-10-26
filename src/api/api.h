#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace brolm::api {

// ---- Access ------------------------------------------------------------------
//
// Every loader asks the host's access gate (embed/embed_access.h,
// bronze::embed::checkAccess) about each path the app named, after the path
// resolver and before anything is opened, stat'ed or queued, on the calling JS
// thread. The gate is handed the path absolute and lexically normal, in UTF-8,
// with AccessKind::Read:
//   loadQwen / loadModel (the GGUF), loadMistral (the GGUF and tokenizerPath),
//   loadGemma2, loadQwen35, loadQwen3VL, loadNllb, loadLaya, loadLayaAsync and
//   new LayaModel (the checkpoint directory: a directory covers what is in it),
//   loadModernBert (the directory and each of configPath / tokenizerPath /
//   weightsPath given), loadTokenizer (the tokenizer.json, or vocabPath and
//   mergesPath), loadLlama3Tokenizer (the file), loadClip / loadClipModel
//   (vocabPath, mergesPath, weightsPath, textPath, imagePath, projectionPath,
//   configPath), loadT5 (tokenizerPath, ggufPath, weightsPath, each of shards).
// A file the loader looks for on its own beside the weights (loadMistral's
// tekken.json, loadClip's config.json) is asked about too and skipped when
// refused. Nothing in bro.lm writes a file.
//
// A refusal is an Error carrying the gate's sentence as its message, code
// "EACCES" and `path` (the path as the gate was handed it), named
// "PermissionRevoked" for a grant taken back while the app ran. It is thrown
// from the call, including a call given opts.onReady (nothing is queued);
// loadLayaAsync's promise is rejected with it instead. Nothing is read.

/// Sets the asset path resolver callback.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

/// Mounts `bro.lm` into the current Bronze realm.
void installLM();

/// Drains completed background LM jobs and fires their JS callbacks (onToken, onDone, onError),
/// and settles this thread's LayaModel promises (loadLayaAsync / predictAsync). Promise reactions
/// are queued as microtasks; a host that wants them to run this frame drains microtasks after.
void tickLMAsync();

/// Stops every LayaModel's device threads and frees their replicas, and drops this thread's
/// unsettled Laya promises. Call at engine shutdown, before the runtime and brotensor go away.
void shutdownLM();

} // namespace brolm::api

using brolm::api::installLM;
