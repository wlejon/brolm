#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace brolm::api {

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
