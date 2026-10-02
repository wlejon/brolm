#pragma once

#include <cstddef>
#include <vector>

namespace brolm::qwen35::internal {

struct MRopePairing {
    std::vector<int> t_pairs, h_pairs, w_pairs;
};

inline MRopePairing mrope_pairing(int d_t, int d_h, int d_w) {
    const int half = d_t + d_h + d_w;
    std::vector<char> owner(static_cast<std::size_t>(half), 'T');
    for (int k = 0; k < d_h; ++k) owner[static_cast<std::size_t>(1 + 3 * k)] = 'H';
    for (int k = 0; k < d_w; ++k) owner[static_cast<std::size_t>(2 + 3 * k)] = 'W';
    MRopePairing p;
    for (int j = 0; j < half; ++j) {
        switch (owner[static_cast<std::size_t>(j)]) {
            case 'T': p.t_pairs.push_back(j); break;
            case 'H': p.h_pairs.push_back(j); break;
            case 'W': p.w_pairs.push_back(j); break;
        }
    }
    return p;
}

}  // namespace brolm::qwen35::internal
