#pragma once

// Private state of brolm::laya::LayaGrad, shared by laya_grad.cpp (packing,
// head, scorer, soft-row gradients) and laya_grad_encoder.cpp (the ModernBERT
// forward with per-layer checkpoints and its recompute backward).

#include "brolm/laya_grad.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <vector>

namespace brolm::laya {

namespace bt = ::brotensor;

struct LayaGrad::State {
    int T = 0, N = 0, K = 0, S = 0;  // rows, items, markers, soft rows (sequence order)
    int soft_table_rows = 0;
    std::vector<int> markers_per_item;

    // Device INT32 index block: ids | pos | bounds | type | markers | soft_seq | soft_tab.
    bt::Tensor idx;
    bt::Tensor v_ids, v_pos, v_bounds, v_type, v_markers, v_soft_seq, v_soft_tab;
    bt::Tensor soft_seq_rows;  // (S, D) soft table rows gathered into sequence order

    bt::Tensor ws;  // FP32 split-K workspace for every forward linear

    // Encoder: embeds (pre embedding-norm), h_in[i] = input of layer i
    // (h_in[0] after the embedding norm), h_in[L] = pre final-norm, enc_out.
    bt::Tensor embeds;
    std::vector<bt::Tensor> h_in;
    bt::Tensor enc_out;

    // Encoder layer scratch (forward and recompute).
    bt::Tensor h, a, qkv, attn, m, pre, geglu;
    bt::Tensor xhat1, mean1, rstd1, xhat2, mean2, rstd2;
    bt::Tensor zero_beta, dgamma, dbeta;  // bias-free LayerNorm plumbing
    bt::Tensor neg_sin_full, neg_sin_slid;  // RoPE inverse: rotate by -theta
    const void* neg_sin_src = nullptr;      // the encoder table they were built from

    // Head layers (pre-LN nn.TransformerEncoderLayer), per layer.
    struct HeadSave {
        bt::Tensor x_in, ln1_y, ln1_xhat, ln1_mean, ln1_rstd, qkv, attn, x_mid;
        bt::Tensor ln2_y, ln2_xhat, ln2_mean, ln2_rstd, act;
    };
    std::vector<HeadSave> head;
    bt::Tensor type_rows, x_final;

    // Scorer.
    bt::Tensor m_rows, s_y, s_xhat, s_mean, s_rstd, pre1, g1, wide;

    // Backward scratch.
    bt::Tensor g_a, g_b, g_c, g_qkv, g_wide, g_rows, g_rows32;
};

}  // namespace brolm::laya
