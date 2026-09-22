#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace brolm::api {

/// Sets the asset path resolver callback.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

/// Mounts `bro.lm` into the current Bronze realm.
void installLM();

/// Drains completed background LM jobs and fires their JS callbacks (onToken, onDone, onError).
void tickLMAsync();

} // namespace brolm::api

using brolm::api::installLM;
