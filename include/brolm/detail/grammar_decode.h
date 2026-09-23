#pragma once

// One grammar-constrained decode: a working copy of the grammar plus the
// per-id token-text table it masks against. The decode loops that do not go
// through detail::generate (the VLM drivers) mask each logits row with
// mask() before sampling and advance with accept() after.

#include "brolm/grammar.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::detail {

class GrammarDecode {
public:
    // `grammar` null = unconstrained (every call below is a no-op). The
    // table must cover the ids the model can emit; an id past it, and any id
    // whose text is empty (a control token), is never allowed except as
    // `eos_id` once the grammar accepts. Both must outlive this object.
    GrammarDecode(const Grammar* grammar, const std::vector<std::string>* token_text) {
        if (!grammar) return;
        if (!token_text || token_text->empty())
            throw std::runtime_error("grammar-constrained decode needs the model's token text");
        state_.emplace(grammar->clone());
        text_ = token_text;
    }

    bool active() const { return state_.has_value(); }

    // Mask every token the grammar cannot take next. Returns false when no
    // token is left — the grammar is complete with no stop token allowed, or
    // the vocabulary cannot continue it — and the decode must end here.
    bool mask(float* row, int vocab, int eos_id) const {
        if (!state_) return true;
        state_->mask_logits(row, vocab, *text_, eos_id);
        for (int i = 0; i < vocab; ++i)
            if (!(std::isinf(row[i]) && row[i] < 0)) return true;
        return false;
    }

    void accept(int id) {
        if (!state_) return;
        if (id >= 0 && static_cast<size_t>(id) < text_->size()) state_->accept((*text_)[static_cast<size_t>(id)]);
    }

    bool accepted() const { return !state_ || state_->is_accepted(); }

private:
    std::optional<Grammar> state_;
    const std::vector<std::string>* text_ = nullptr;
};

}  // namespace brolm::detail
