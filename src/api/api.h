#pragma once

#include "embed/embed.h"

namespace brolm::api {

/// Mounts `bro.lm` into the current Bronze realm.
void installLM();

/// Drains completed background LM jobs and fires their JS callbacks (onToken, onDone, onError).
void tickLMAsync();

} // namespace brolm::api

using brolm::api::installLM;
