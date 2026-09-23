#include "brolm/detail/hf_bpe.h"

#include <cstdio>
#include <queue>
#include <stdexcept>

namespace brolm::detail::hfbpe {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("hfbpe::Model: " + msg); }

std::size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // stray continuation / invalid lead: one byte, as a char of its own
}

struct Symbol {
    int32_t c;
    int32_t prev;
    int32_t next;
    int32_t len;  // bytes
};

struct QueuedMerge {
    int32_t pos;
    int32_t rank;
    int32_t new_id;
    // std::priority_queue pops the "largest": make that the lowest rank, then
    // the lowest position (tokenizers' Merge ordering).
    bool operator<(const QueuedMerge& o) const {
        if (rank != o.rank) return rank > o.rank;
        return pos > o.pos;
    }
};

}  // namespace

void Model::load(const j::Value& model) {
    if (!model.is_object()) fail("'model' is not an object");
    const std::string type = model.get_string("type", "BPE");
    if (type != "BPE") fail("model type '" + type + "' is not BPE");
    if (const j::Value* v = model.find("continuing_subword_prefix"); v && !v->is_null() && !v->as_string().empty())
        fail("continuing_subword_prefix is not supported");
    if (const j::Value* v = model.find("end_of_word_suffix"); v && !v->is_null() && !v->as_string().empty())
        fail("end_of_word_suffix is not supported");
    byte_fallback_ = model.get_bool("byte_fallback", false);
    fuse_unk_ = model.get_bool("fuse_unk", false);
    ignore_merges_ = model.get_bool("ignore_merges", false);

    const j::Value* vocab = model.find("vocab");
    if (!vocab || !vocab->is_object()) fail("missing 'model.vocab' object");
    vocab_.clear();
    vocab_.reserve(vocab->as_object().size());
    for (const auto& [tok, idv] : vocab->as_object()) {
        if (idv.is_number()) vocab_.emplace(tok, static_cast<int32_t>(idv.as_number()));
    }

    unk_id_ = -1;
    if (const j::Value* u = model.find("unk_token"); u && u->is_string()) unk_id_ = id(u->as_string());
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        byte_ids_[b] = id(buf);
    }

    merges_.clear();
    const j::Value* merges = model.find("merges");
    if (!merges || !merges->is_array()) return;
    const auto& arr = merges->as_array();
    merges_.reserve(arr.size());
    std::string a, b;
    for (std::size_t i = 0; i < arr.size(); ++i) {
        if (arr[i].is_array() && arr[i].as_array().size() == 2) {
            a = arr[i].as_array()[0].as_string();
            b = arr[i].as_array()[1].as_string();
        } else if (arr[i].is_string()) {  // legacy "a b" form
            const std::string& line = arr[i].as_string();
            const auto sp = line.find(' ');
            if (sp == std::string::npos) fail("merge " + std::to_string(i) + " is not 'a b'");
            a = line.substr(0, sp);
            b = line.substr(sp + 1);
        } else {
            fail("merge " + std::to_string(i) + " is malformed");
        }
        const int32_t ia = id(a), ib = id(b), inew = id(a + b);
        if (ia < 0 || ib < 0 || inew < 0) fail("merge " + std::to_string(i) + " references a token outside the vocab");
        merges_[pair_key(ia, ib)] = Merge{static_cast<int32_t>(i), inew};  // a later duplicate wins, as a HashMap collect
    }
}

int32_t Model::id(std::string_view token) const {
    const auto it = vocab_.find(token);
    return it == vocab_.end() ? -1 : it->second;
}

void Model::encode_word(std::string_view w, std::vector<int32_t>& out) const {
    if (w.empty()) return;
    if (ignore_merges_) {
        if (const int32_t whole = id(w); whole >= 0) {
            out.push_back(whole);
            return;
        }
    }

    // Initial symbols (tokenizers BPE::merge_word).
    std::vector<Symbol> sym;
    sym.reserve(w.size());
    auto add = [&](int32_t c, int32_t len) {
        const int32_t n = static_cast<int32_t>(sym.size());
        sym.push_back(Symbol{c, n - 1, -1, len});
        if (n > 0) sym[static_cast<std::size_t>(n - 1)].next = n;
    };
    int32_t unk_len = 0;  // a pending (possibly fused) unk run
    bool unk_pending = false;
    for (std::size_t i = 0; i < w.size();) {
        const std::size_t n = std::min(utf8_len(static_cast<unsigned char>(w[i])), w.size() - i);
        const std::string_view ch = w.substr(i, n);
        i += n;
        if (const int32_t cid = id(ch); cid >= 0) {
            if (unk_pending) {
                add(unk_id_, unk_len);
                unk_pending = false;
            }
            add(cid, static_cast<int32_t>(n));
            continue;
        }
        if (byte_fallback_) {
            bool all = true;
            for (char c : ch) all = all && byte_ids_[static_cast<unsigned char>(c)] >= 0;
            if (all) {
                // tokenizers adds the byte tokens without flushing a pending unk.
                for (char c : ch) add(byte_ids_[static_cast<unsigned char>(c)], 1);
                continue;
            }
        }
        if (unk_id_ < 0) continue;  // no unk token: the character is dropped
        if (unk_pending && fuse_unk_) {
            unk_len += static_cast<int32_t>(n);
        } else {
            if (unk_pending) add(unk_id_, unk_len);
            unk_pending = true;
            unk_len = static_cast<int32_t>(n);
        }
    }
    if (unk_pending) add(unk_id_, unk_len);
    if (sym.empty()) return;

    // Word::merge_all.
    std::priority_queue<QueuedMerge> queue;
    auto lookup = [&](int32_t a, int32_t b) -> const Merge* {
        const auto it = merges_.find(pair_key(a, b));
        return it == merges_.end() ? nullptr : &it->second;
    };
    for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
        if (const Merge* m = lookup(sym[i].c, sym[i + 1].c)) {
            queue.push(QueuedMerge{static_cast<int32_t>(i), m->rank, m->new_id});
        }
    }
    while (!queue.empty()) {
        const QueuedMerge top = queue.top();
        queue.pop();
        Symbol& cur = sym[static_cast<std::size_t>(top.pos)];
        if (cur.len == 0 || cur.next < 0) continue;
        const int32_t next_pos = cur.next;
        const Symbol right = sym[static_cast<std::size_t>(next_pos)];
        const Merge* m = lookup(cur.c, right.c);
        if (!m || m->new_id != top.new_id) continue;  // expired entry

        cur.c = top.new_id;
        cur.len += right.len;
        cur.next = right.next;
        sym[static_cast<std::size_t>(next_pos)].len = 0;
        if (right.next >= 0 && static_cast<std::size_t>(right.next) < sym.size()) {
            sym[static_cast<std::size_t>(right.next)].prev = top.pos;
        }
        if (cur.prev >= 0) {
            if (const Merge* pm = lookup(sym[static_cast<std::size_t>(cur.prev)].c, cur.c)) {
                queue.push(QueuedMerge{cur.prev, pm->rank, pm->new_id});
            }
        }
        if (cur.next >= 0 && static_cast<std::size_t>(cur.next) < sym.size()) {
            if (const Merge* nm = lookup(cur.c, sym[static_cast<std::size_t>(cur.next)].c)) {
                queue.push(QueuedMerge{top.pos, nm->rank, nm->new_id});
            }
        }
    }
    for (const Symbol& s : sym) {
        if (s.len != 0) out.push_back(s.c);
    }
}

}  // namespace brolm::detail::hfbpe
