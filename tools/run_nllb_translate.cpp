// Ad-hoc driver: load a converted NLLB-200 checkpoint and translate text.
//
// usage: run_nllb_translate <model_dir> <src_lang> <tgt_lang> <text...>
//   model_dir  directory with config.json + tokenizer.json + model.safetensors
//              (the output of scripts/convert-nllb.py)
//   src_lang   FLORES-200 source code, e.g. eng_Latn
//   tgt_lang   FLORES-200 target code, e.g. fra_Latn
//   text...    the sentence to translate (remaining args are joined by spaces)
//
// Not run by ctest — for eyeballing the full translate pipeline against a real
// checkpoint. Reads BROLM_NLLB_BEAMS for the beam width (default 5).

#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the optional beam override

#include "brolm/nllb.h"

#include "brotensor/runtime.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <model_dir> <src_lang> <tgt_lang> <text...>\n", argv[0]);
        return 1;
    }
    const std::string model_dir = argv[1];
    const std::string src_lang  = argv[2];
    const std::string tgt_lang  = argv[3];
    std::string text = argv[4];
    for (int i = 5; i < argc; ++i) { text += ' '; text += argv[i]; }

    try {
        brotensor::init();

        brolm::nllb::Translator tr = brolm::nllb::Translator::load(model_dir);

        brolm::nllb::BeamOptions opts;
        if (const char* nb = std::getenv("BROLM_NLLB_BEAMS")) {
            if (nb[0]) opts.num_beams = std::atoi(nb);
        }

        const std::string out = tr.translate(text, src_lang, tgt_lang, opts);
        std::printf("%s\n", out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
