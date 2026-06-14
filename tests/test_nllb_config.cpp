// NLLB-200 (M2M-100) config parse test. Verifies the typed config mirrors a
// real nllb-200-distilled-600M config.json and that validate() rejects
// inconsistent dimensions.

#include "brolm/nllb_config.h"

#include <cstdio>
#include <string>

namespace nllb = brolm::nllb;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Trimmed copy of facebook/nllb-200-distilled-600M config.json.
static const char* kConfig = R"json({
  "architectures": ["M2M100ForConditionalGeneration"],
  "model_type": "m2m_100",
  "activation_function": "relu",
  "d_model": 1024,
  "encoder_layers": 12,
  "decoder_layers": 12,
  "encoder_attention_heads": 16,
  "decoder_attention_heads": 16,
  "encoder_ffn_dim": 4096,
  "decoder_ffn_dim": 4096,
  "vocab_size": 256206,
  "max_position_embeddings": 1024,
  "scale_embedding": true,
  "bos_token_id": 0,
  "pad_token_id": 1,
  "eos_token_id": 2,
  "decoder_start_token_id": 2
})json";

int main() {
    auto cfg = nllb::NllbConfig::from_json_text(kConfig);

    CHECK(cfg.d_model == 1024);
    CHECK(cfg.encoder_layers == 12);
    CHECK(cfg.decoder_layers == 12);
    CHECK(cfg.encoder_attention_heads == 16);
    CHECK(cfg.decoder_attention_heads == 16);
    CHECK(cfg.encoder_ffn_dim == 4096);
    CHECK(cfg.decoder_ffn_dim == 4096);
    CHECK(cfg.vocab_size == 256206);
    CHECK(cfg.max_position_embeddings == 1024);
    CHECK(cfg.scale_embedding == true);
    CHECK(cfg.head_dim() == 64);
    CHECK(cfg.pad_token_id == 1);
    CHECK(cfg.bos_token_id == 0);
    CHECK(cfg.eos_token_id == 2);
    CHECK(cfg.decoder_start_token_id == 2);

    // Defaults apply when keys are absent.
    {
        auto def = nllb::NllbConfig::from_json_text("{\"model_type\":\"m2m_100\"}");
        CHECK(def.d_model == 1024);
        CHECK(def.vocab_size == 256206);
    }

    // Wrong model_type must throw.
    {
        bool threw = false;
        try { (void)nllb::NllbConfig::from_json_text(
                  "{\"model_type\":\"t5\"}"); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    // Inconsistent dims must throw in validate().
    {
        bool threw = false;
        try { (void)nllb::NllbConfig::from_json_text(
                  "{\"model_type\":\"m2m_100\",\"d_model\":1000,"
                  "\"encoder_attention_heads\":16}"); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    if (g_failures == 0) std::printf("nllb_config: OK\n");
    else std::fprintf(stderr, "nllb_config: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
