#include "brolm/detail/unicode.h"

#include "unicode_tables.h"

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace brolm::detail::unicode {

namespace {

using tables::Range;

template <std::size_t N>
bool in_ranges(const Range (&r)[N], uint32_t cp) {
    // Ranges are sorted and disjoint: the first one whose hi reaches cp is
    // the only candidate.
    auto it = std::lower_bound(
        std::begin(r), std::end(r), cp,
        [](const Range& a, uint32_t v) { return a.hi < v; });
    return it != std::end(r) && it->lo <= cp;
}

// Hangul syllable arithmetic (UAX #15 §3.12).
constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161,
                   kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount;
constexpr uint32_t kSCount = kLCount * kNCount;

bool is_hangul_syllable(uint32_t cp) { return cp >= kSBase && cp < kSBase + kSCount; }

const tables::Decomp* find_decomp(uint32_t cp) {
    auto it = std::lower_bound(
        std::begin(tables::kDecomp), std::end(tables::kDecomp), cp,
        [](const tables::Decomp& d, uint32_t v) { return d.cp < v; });
    return (it != std::end(tables::kDecomp) && it->cp == cp) ? &*it : nullptr;
}

// Primary composite of (a, b), or 0 when the pair does not compose.
uint32_t compose_pair(uint32_t a, uint32_t b) {
    // Hangul L + V -> LV, LV + T -> LVT.
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    }
    if (is_hangul_syllable(a) && (a - kSBase) % kTCount == 0 &&
        b > kTBase && b < kTBase + kTCount) {
        return a + (b - kTBase);
    }
    auto it = std::lower_bound(
        std::begin(tables::kCompose), std::end(tables::kCompose),
        std::pair<uint32_t, uint32_t>{a, b},
        [](const tables::Compose& c, const std::pair<uint32_t, uint32_t>& v) {
            return c.first != v.first ? c.first < v.first : c.second < v.second;
        });
    if (it != std::end(tables::kCompose) && it->first == a && it->second == b) {
        return it->composite;
    }
    return 0;
}

constexpr uint32_t kEscapeBase = 0xDC00;

}  // namespace

const char* version() { return tables::kUnicodeVersion; }

// ─── UTF-8 ─────────────────────────────────────────────────────────────────

uint32_t decode_utf8(std::string_view s, std::size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { ++i; return c; }
    std::size_t len;
    uint32_t cp;
    if ((c & 0xE0) == 0xC0)      { len = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
    else                         { ++i; return kEscapeBase + c; }
    if (i + len > s.size())      { ++i; return kEscapeBase + c; }
    for (std::size_t k = 1; k < len; ++k) {
        const unsigned char d = static_cast<unsigned char>(s[i + k]);
        if ((d & 0xC0) != 0x80)  { ++i; return kEscapeBase + c; }
        cp = (cp << 6) | (d & 0x3F);
    }
    static constexpr uint32_t kMinForLen[] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < kMinForLen[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        ++i;
        return kEscapeBase + c;  // overlong, surrogate, or out of range
    }
    i += len;
    return cp;
}

void encode_utf8(uint32_t cp, std::string& out) {
    if (cp >= kEscapeBase && cp <= kEscapeBase + 0xFF) {
        out += static_cast<char>(cp - kEscapeBase);
        return;
    }
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ─── Properties ────────────────────────────────────────────────────────────

bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return in_ranges(tables::kLetter, cp);
}

bool is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(tables::kNumber, cp);
}

bool is_white_space(uint32_t cp) {
    if (cp < 0x80) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D);
    return in_ranges(tables::kWhiteSpace, cp);
}

uint8_t combining_class(uint32_t cp) {
    if (cp < 0x300) return 0;
    auto it = std::lower_bound(
        std::begin(tables::kCcc), std::end(tables::kCcc), cp,
        [](const tables::CccRange& a, uint32_t v) { return a.hi < v; });
    return (it != std::end(tables::kCcc) && it->lo <= cp) ? it->ccc : 0;
}

// ─── NFC ───────────────────────────────────────────────────────────────────

bool is_nfc(std::string_view s) {
    std::size_t i = 0;
    uint8_t last_ccc = 0;
    while (i < s.size()) {
        if (static_cast<unsigned char>(s[i]) < 0x80) {
            ++i;
            last_ccc = 0;
            continue;
        }
        const uint32_t cp = decode_utf8(s, i);
        const uint8_t ccc = combining_class(cp);
        if (ccc != 0 && last_ccc > ccc) return false;   // out of canonical order
        if (in_ranges(tables::kNfcQcNo, cp)) return false;
        if (in_ranges(tables::kNfcQcMaybe, cp)) return false;  // "Maybe": go the long way
        last_ccc = ccc;
    }
    return true;
}

void nfc(std::vector<uint32_t>& cps) {
    // 1. Canonical decomposition (the table holds full NFD sequences).
    std::vector<uint32_t> d;
    d.reserve(cps.size() + 8);
    for (const uint32_t cp : cps) {
        if (is_hangul_syllable(cp)) {
            const uint32_t s = cp - kSBase;
            d.push_back(kLBase + s / kNCount);
            d.push_back(kVBase + (s % kNCount) / kTCount);
            if (const uint32_t t = s % kTCount) d.push_back(kTBase + t);
        } else if (const tables::Decomp* e = find_decomp(cp)) {
            for (std::size_t k = 0; k < e->len; ++k) {
                d.push_back(tables::kDecompPool[e->offset + k]);
            }
        } else {
            d.push_back(cp);
        }
    }

    // 2. Canonical ordering: each run of non-starters sorts by ccc, stably.
    for (std::size_t i = 0; i < d.size();) {
        if (combining_class(d[i]) == 0) { ++i; continue; }
        std::size_t j = i + 1;
        while (j < d.size() && combining_class(d[j]) != 0) ++j;
        if (j - i > 1) {
            std::stable_sort(d.begin() + static_cast<std::ptrdiff_t>(i),
                             d.begin() + static_cast<std::ptrdiff_t>(j),
                             [](uint32_t a, uint32_t b) {
                                 return combining_class(a) < combining_class(b);
                             });
        }
        i = j;
    }

    // 3. Canonical composition. A character composes with the last starter
    // unless an intervening kept character blocks it (ccc >= its own, or a
    // starter — which would have become the last starter itself).
    std::vector<uint32_t> out;
    out.reserve(d.size());
    std::ptrdiff_t starter = -1;
    uint8_t last_ccc = 0;
    for (const uint32_t cp : d) {
        const uint8_t ccc = combining_class(cp);
        if (starter >= 0) {
            const bool intervening =
                out.size() - 1 != static_cast<std::size_t>(starter);
            const bool blocked = intervening && last_ccc >= ccc;
            if (!blocked) {
                if (const uint32_t comp = compose_pair(
                        out[static_cast<std::size_t>(starter)], cp)) {
                    out[static_cast<std::size_t>(starter)] = comp;
                    continue;
                }
            }
        }
        if (ccc == 0) starter = static_cast<std::ptrdiff_t>(out.size());
        out.push_back(cp);
        last_ccc = ccc;
    }
    cps.swap(out);
}

std::string nfc(std::string_view s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) cps.push_back(decode_utf8(s, i));
    nfc(cps);
    std::string out;
    out.reserve(s.size());
    for (const uint32_t cp : cps) encode_utf8(cp, out);
    return out;
}

}  // namespace brolm::detail::unicode
