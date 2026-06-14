#include "brolm/detail/spm_normalizer.h"

#include <cstdint>
#include <cstring>

namespace brolm::detail::spm {

namespace {

// Darts-clone double-array unit accessors (SentencePiece vendors this layout).
inline bool     dc_has_leaf(std::uint32_t u) { return ((u >> 8) & 1) == 1; }
inline std::uint32_t dc_value(std::uint32_t u) { return u & ((1u << 31) - 1); }
inline std::uint32_t dc_label(std::uint32_t u) { return u & ((1u << 31) | 0xffu); }
inline std::uint32_t dc_offset(std::uint32_t u) {
    return (u >> 10) << ((u & (1u << 9)) >> 6);
}

inline std::uint32_t u32le(std::string_view b, std::size_t off) {
    return static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[off])) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[off + 1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[off + 2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[off + 3])) << 24);
}

// Byte length of the UTF-8 character starting at lead byte `b` (1 on a
// malformed lead, so the normalize loop always advances).
inline std::size_t utf8_len(std::uint8_t b) {
    if (b < 0x80) return 1;
    if ((b >> 5) == 0x6) return 2;
    if ((b >> 4) == 0xe) return 3;
    if ((b >> 3) == 0x1e) return 4;
    return 1;
}

}  // namespace

PrecompiledNormalizer::PrecompiledNormalizer(std::string_view charsmap) {
    // Layout: uint32 trie_blob_size (LE) | trie units | normalized-string blob.
    if (charsmap.size() < 4) return;
    const std::uint32_t trie_size = u32le(charsmap, 0);
    if (static_cast<std::size_t>(4) + trie_size > charsmap.size()) return;
    if (trie_size % 4 != 0) return;

    const std::size_t n_units = trie_size / 4;
    units_.resize(n_units);
    for (std::size_t i = 0; i < n_units; ++i)
        units_[i] = u32le(charsmap, 4 + i * 4);

    const std::size_t norm_off = static_cast<std::size_t>(4) + trie_size;
    normalized_.assign(charsmap.data() + norm_off, charsmap.size() - norm_off);
}

bool PrecompiledNormalizer::longest_match(const std::uint8_t* key,
                                          std::size_t key_len,
                                          std::uint32_t& out_len,
                                          std::uint32_t& out_val) const {
    std::size_t id = 0;
    std::uint32_t unit = units_[0];
    id ^= dc_offset(unit);
    bool found = false;
    for (std::size_t i = 0; i < key_len; ++i) {
        const std::uint8_t k = key[i];
        id ^= k;
        if (id >= units_.size()) break;
        unit = units_[id];
        if (dc_label(unit) != k) break;
        id ^= dc_offset(unit);
        if (id >= units_.size()) break;
        if (dc_has_leaf(unit)) {
            // A leaf hangs off the current node; its value lives in units_[id].
            out_val = dc_value(units_[id]);
            out_len = static_cast<std::uint32_t>(i + 1);
            found = true;   // later (longer) leaves overwrite → keeps the longest
        }
    }
    return found;
}

std::string PrecompiledNormalizer::normalize(std::string_view input) const {
    if (units_.empty()) return std::string(input);

    std::string out;
    out.reserve(input.size());
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(input.data());
    std::size_t i = 0;
    while (i < input.size()) {
        std::uint32_t mlen = 0, mval = 0;
        if (longest_match(p + i, input.size() - i, mlen, mval) &&
            mval < normalized_.size()) {
            // Replacement is the null-terminated string at normalized_[mval]
            // (may be empty → a deletion).
            out.append(normalized_.c_str() + mval);
            i += mlen;
        } else {
            std::size_t clen = utf8_len(p[i]);
            if (i + clen > input.size()) clen = 1;
            out.append(reinterpret_cast<const char*>(p + i), clen);
            i += clen;
        }
    }
    return out;
}

}  // namespace brolm::detail::spm
