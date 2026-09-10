#!/usr/bin/env python3
# Generates tests/ref/qwen_tokenizer_unicode.json — the Hugging Face oracle
# for brolm::qwen::Tokenizer's Unicode pre-tokenization + NFC normalization,
# consumed by tests/test_qwen_tokenizer_unicode.cpp.
#
# Two reference tokenizers:
#   * OmniVoice's tokenizer.json (Qwen2/Qwen3 byte-level BPE + the seven
#     OmniVoice specials), loaded with `tokenizers` — the "ids" of every case;
#   * Qwen3-TTS's vocab.json + merges.txt pair, loaded through transformers'
#     Qwen2TokenizerFast conversion — the "ids_tts" of every case whose text
#     carries none of that tokenizer's added tokens (same vocab, different
#     merges, so the ids differ and both loaders get checked).
# Plus "nfc" pairs from the tokenizer.json's own NFC normalizer, cross-checked
# against unicodedata, for the normalizer on its own.
#
# Offline, one-off: never run at build or run time. The output is checked in;
# regenerate when the corpus or a reference tokenizer changes:
#   python scripts/gen_qwen_tokenizer_fixture.py [--omnivoice DIR] [--tts DIR]
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import unicodedata

from tokenizers import Tokenizer
import tokenizers

OMNIVOICE_SPECIALS = [
    "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
    "<|instruct_end|>", "<|text_start|>", "<|text_end|>",
]

# ── corpus ──────────────────────────────────────────────────────────────────

ENGLISH = [
    "Hello, world!",
    "It's a beautiful day, isn't it? We'll see... they've gone; I'd say we're done.",
    "IT'S LOUD. HE'LL SHOUT. THEY'RE HERE. WE'VE WON. I'M IN. SHE'D KNOW.",
    "it'S mixed'Case 'Re 'vE 'LL 'D 'M 'T 's",
    "don't can't won't shouldn't I'm you're they'd we'll it's",
    "'twas the night — 'tis true, o'clock, rock'n'roll, y'all, ma'am",
    "Dr. Smith's e-mail: john.smith@example.com (re: \"Q3 results\")!",
    "The quick brown fox jumps over the lazy dog.",
    "Wait... what?! No way -- really? Yes; absolutely: 100% sure.",
    "  leading and trailing spaces  ",
    "Tabs\tbetween\twords\tand\ta\ttrailing\ttab\t",
    "Line one\nLine two\r\nLine three\rLine four\n\n",
    "A sentence ending in a newline.\n",
    "Multiple    spaces   between    words.",
    "hyphen-ated words, under_scores, CamelCase, snake_case_var, kebab-case-name",
    "Quotes 'single' and \"double\" and `backticks`.",
    "[laughter] That was funny [sigh] but tiring [cough] excuse me [breath]",
    "[laughter][sigh][cough] [laugh] [giggle] [gasp]",
    "Wow [laughter], really? [sigh]... okay [cough].",
    "Price: $1,234.56 (approx. €1.100,00 or £999) at 12:30pm on 2024-01-15.",
    "Mr. O'Neil's 3rd-place finish (2nd overall) wasn't bad!",
    "hello",
    " hello",
    "hello ",
    "hello world",
    "HELLO WORLD",
    "a",
    " ",
    "\n",
    "",
]

CODE = [
    "def foo(x: int) -> int:\n    return x * 2\n",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    "SELECT id, name FROM users WHERE age >= 18 AND status = 'active';",
    "const f = async (a, b) => { await fetch(`/api/${a}?b=${b}`); };",
    "#include <vector>\nstd::vector<int> v{1, 2, 3};\n",
    "if x != y and not z:\n\tprint(\"nope\")\n",
    "let re = /^[a-z]+\\d{2,}$/gi;",
    "x = 0x1F + 0b1010 - 0o17 + 1e-9 + 3.14f",
    "git commit -m \"fix: handle NUL\\x00 bytes\" && git push --force-with-lease",
    "{\"key\": [1, 2, {\"nested\": null}], \"ok\": true}",
    "<div class=\"a b\" data-x='1'>&amp;&lt;&gt;</div>",
    "printf(\"%s: %d\\n\", name, count);",
    "a=b;c=d;e=f;",
    "    indented\n        more indented\n    back\n",
    "path/to/file.txt C:\\Users\\name\\Documents ~/.config",
    "arr[0][1] = obj->field.sub::value; // comment /* block */",
]

NUMBERS = [
    "123", "1234567890", " 123", "123 ", "a1b2c3", "1a2b3c", "3.14159", "1,000,000",
    "2024-01-15T12:34:56Z", "+1 (555) 010-9999", "0", "007", "1/2 3/4",
    "²³¹ ½ ¼ ¾ Ⅻ Ⅳ ⅷ ①②③ ⑽ ㊷", "١٢٣٤٥ ٠", "०१२३४५६७८९", "๐๑๒๓๔๕", "𝟘𝟙𝟚 𝟎𝟏𝟐",
    "v2.0.1-beta.3+build.42", "No. 5", "5th 21st 100th",
]

WHITESPACE = [
    "a b", "a  b", "a   b", "a    b", "a\tb", "a\t\tb", "a \t b", "a\t b", "a \tb",
    "a\nb", "a\n\nb", "a \nb", "a\n b", "a \n b", "a\n\n\nb", "a\r\nb", "a\r\n\r\nb",
    "a \r\n b", "a\rb", "a\n \n b", "\n\n a", "  \n", " \n ", "\n ", " \n", "\t\n\t",
    "a\v\fb", "a\u000b\u000cb", "x\u0085y", "a\u0085\u0085b", "x\u00a0y", "a\u00a0\u00a0b",
    "a\u2000b", "a\u2003\u2003b", "a\u3000b", "\u3000\u3000中", "a\u1680\u1680b",
    "a\u2028b", "a\u2029\u2029b", "a\u202fb", "a\u205f\u205fb", "a\u200bb", "a\u200b\u200bb",
    "a\ufeffb", "\ufeffBOM at start", "a\u180e\u180eb", "trailing   ", "   leading",
    "  ", "   ", "    ", "\t", "\t\t", "\n\n\n", " \t\n \t\n", "end.\n\n\n",
    "word \n\n", "word\n\n ", "word\n\n  next",
]

CHINESE = [
    "你好，世界！",
    "今天天气很好，我们去公园散步吧。",
    "中文分词测试：标点、符号「引号」《书名》（括号）【方括号】……——",
    "简体：汉字 软件 国家 图书馆",
    "繁體：漢字 軟體 國家 圖書館",
    "這是一個繁體中文的句子，包含標點符號。",
    "混合 mixed 中英文 English 文本 text 123 数字。",
    "北京市海淀区中关村大街1号",
    "价格：¥1,234.50，折扣20%！",
    "他说：“你好吗？”她回答：“很好，谢谢。”",
    "一二三四五六七八九十百千万亿零",
    "𠀀𠀁𠀂 扩展B区 𪚥 𫝀",
    "日本語と中国語：東京 北京 上海 大阪",
    "中华人民共和国",
    "中華民國",
]

JAPANESE = [
    "こんにちは、世界！",
    "東京タワーは高さ333メートルです。",
    "漢字とひらがなとカタカナとABCと123の混在テキスト。",
    "私はAppleのiPhoneを使っています。",
    "ｶﾀｶﾅ半角 ＡＢＣ全角 １２３全角数字",
    "「引用」『二重引用』（括弧）【墨付き括弧】〜波ダッシュ〜・中黒…",
    "すもももももももものうち",
    "ラーメン、うどん、そば、寿司、天ぷら",
    "お疲れ様でした。また明日！",
    "ゔゕゖ ヷヸヹヺ ㋐㋑ ㍻㍼",
]

KOREAN = [
    "안녕하세요, 세계!",
    "한국어 형태소 분석 테스트입니다.",
    "서울특별시 강남구 테헤란로 123",
    "ㄱㄴㄷㄹ ㅏㅑㅓㅕ 자모 단독",
    "각갂갃 힣 가나다라마바사",
    "한글과 English와 日本語와 123",
    "\u1100\u1161\u11a8 \u1112\u1175\u11c2 (jamo sequences)",
    "된장찌개 김치찌개 부대찌개 순두부찌개",
]

CYRILLIC_GREEK = [
    "Привет, мир!",
    "Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Москва — столица России. Київ — столиця України. Мінск — сталіца Беларусі.",
    "Ё ё Й й Ъ ъ Ь ь Ѐ ѐ Ѓ ѓ Є є Ѕ ѕ І і Ї ї Ј ј Љ љ Њ њ Ћ ћ Ќ ќ Ў ў Џ џ",
    "български сръпски македонски казакша қазақша монгол хэл",
    "Ӏ ӏ Ә ә Ғ ғ Қ қ Ң ң Ө ө Ұ ұ Ү ү Һ һ",
    "Γειά σου Κόσμε!",
    "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν.",
    "αβγδεζηθικλμνξοπρστυφχψω ΑΒΓΔ ς σ",
    "Ωμέγα Ω ω Ώ ώ ϊ ϋ ΐ ΰ",
]

ARABIC_HEBREW = [
    "مرحبا بالعالم!",
    "اللغة العربية جميلة، وهي لغة القرآن الكريم.",
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "أرقام: ١٢٣ و 123 و ۱۲۳ فارسی اردو پښتو",
    "سلام دنیا! این یک متن فارسی است.",
    "يَا أَيُّهَا النَّاسُ",
    "ﷲ ﷺ ﻻ ﻷ ﻵ لا لأ",
    "שלום עולם!",
    "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ",
    "עברית עם ניקוד: שָׁלוֹם",
    "ייִדיש: אַ גוטן טאָג",
    "Mixed RTL/LTR: hello שלום مرحبا world",
]

INDIC = [
    "नमस्ते दुनिया!",
    "हिन्दी में एक वाक्य: कि की कु कू कृ के कै को कौ कं कः क़ ख़ ग़ ज़ ड़ ढ़ फ़",
    "क्ष त्र ज्ञ श्र द्ध द्व ट्ट",
    "संस्कृतम् ऋषिः ॐ",
    "মহাত্মা গান্ধী বাংলাদেশ কলকাতা",
    "আমি বাংলায় গান গাই",
    "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਪੰਜਾਬੀ",
    "ગુજરાતી ભાષા અમદાવાદ",
    "ଓଡ଼ିଆ ଭାଷା ଭୁବନେଶ୍ୱର",
    "தமிழ் மொழி சென்னை வணக்கம்",
    "తెలుగు భాష హైదరాబాద్ నమస్కారం",
    "ಕನ್ನಡ ಭಾಷೆ ಬೆಂಗಳೂರು ನಮಸ್ಕಾರ",
    "മലയാളം ഭാഷ തിരുവനന്തപുരം നമസ്കാരം",
    "සිංහල භාෂාව කොළඹ ආයුබෝවන්",
    "मराठी नेपाली भोजपुरी मैथिली",
    "কি কী কু কূ কে কৈ কো কৌ ক্ষ জ্ঞ",
    "நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "అది ఏమిటి? ఇది ఏమిటి?",
]

SEASIA = [
    "สวัสดีชาวโลก",
    "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ แต่มีการเว้นวรรคระหว่างประโยค",
    "กรุงเทพมหานคร อมรรัตนโกสินทร์ มหินทรายุธยา",
    "ก็ ก่ ก้ ก๊ ก๋ กำ กิ กี กึ กื กุ กู เก แก โก ใก ไก",
    "ສະບາຍດີ ພາສາລາວ ວຽງຈັນ",
    "ຂ້ອຍຮັກເຈົ້າ",
    "ជំរាបសួរ ភាសាខ្មែរ ភ្នំពេញ",
    "မင်္ဂလာပါ မြန်မာဘာသာ ရန်ကုန်",
    "ជា​ភាសា​ខ្មែរ (with ZWSP)",
    "Bahasa Indonesia dan Bahasa Melayu; Tagalog: Kumusta ka?",
]

EUROPEAN = [
    "Xin chào thế giới!",
    "Tiếng Việt có dấu: ă â đ ê ô ơ ư à ả ã á ạ ầ ẩ ẫ ấ ậ ằ ẳ ẵ ắ ặ",
    "Tôi yêu Việt Nam. Hà Nội, Thành phố Hồ Chí Minh, Đà Nẵng.",
    "Nguyễn Văn Trỗi đường Lê Lợi quận 1",
    "İstanbul'da güneşli bir gün. Iğdır ığdır ıi İı",
    "TÜRKÇE: çğıöşü ÇĞİÖŞÜ",
    "Ilık ılık, ışık ışık, İzmir izmir",
    "Straße Fußball Grüße größer weiß ß ẞ",
    "Die Bäume, Äpfel, Öl, Übung — ä ö ü Ä Ö Ü",
    "Château, garçon, naïve, cœur, æquo, Œuvre, façade",
    "¿Qué tal? ¡Hola! Español: ñ á é í ó ú ü",
    "Zażółć gęślą jaźń — polski",
    "Příliš žluťoučký kůň úpěl ďábelské ódy — čeština",
    "Árvíztűrő tükörfúrógép — magyar",
    "Suomi: Hyvää päivää! ääkköset",
    "Íslenska: Þórður Ægir Ýr Öxi ð þ æ ö",
    "Português: ação coração não são pão",
    "Română: ă â î ș ț Ă Â Î Ș Ț",
    "Latviešu: Ā ā Č č Ē ē Ģ ģ Ī ī Ķ ķ Ļ ļ Ņ ņ Š š Ū ū Ž ž",
    "Lietuvių: ą č ę ė į š ų ū ž",
    "Nederlands: ij IJ ĳ Ĳ",
    "Esperanto: ĉ ĝ ĥ ĵ ŝ ŭ",
]

EMOJI = [
    "Hello 👋 world 🌍!",
    "👨‍👩‍👧‍👦 family, 👩‍💻 coder, 🏳️‍🌈 flag, 🏴‍☠️ pirate",
    "👍🏻👍🏼👍🏽👍🏾👍🏿 skin tones",
    "🇺🇸🇯🇵🇩🇪🇫🇷 flags 🇬🇧",
    "😀😃😄😁😆😅😂🤣",
    "I ❤️ NY ❤ ♥ ♡",
    "☀️☁️☂️ ☀ ☁ ☂ ✈️ ✈",
    "text👍text 👍 text",
    "emoji😀attached 😀 and 😀😀😀 runs",
    "🧑🏽‍🚀 astronaut 🧙🏻‍♀️ witch 🧜🏿‍♂️ merman",
    "1️⃣2️⃣3️⃣ #️⃣ *️⃣",
    "™ © ® ° ± × ÷ µ ¶ § ¤ ¢ £ ¥ €",
    "→ ← ↑ ↓ ⇒ ⇔ ∀ ∃ ∈ ∉ ∑ ∏ √ ∞ ≈ ≠ ≤ ≥",
    "♠♣♥♦ ♔♕♖ ♩♪♫♬ ☺☻ ☭ ☯ ☮",
]

PUNCT = [
    "「こんにちは」と言った。",
    "全角記号：！？。、，；：（）［］｛｝＜＞＝＋－＊／＼＃＄％＆＠",
    "Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ　ａｂｃ　１２３",
    "“Smart quotes” and ‘single smart quotes’ and „German quotes“ and «guillemets»",
    "Dashes: hyphen-minus - en dash – em dash — figure dash ‒ horizontal bar ―",
    "Ellipsis… and three dots... and ⋯ and ‥",
    "Bullet • list · middle dot ‧ hyphenation point ・ katakana middle dot",
    "Math: a² + b² = c²; ∫₀¹ x dx = ½; ∂f/∂x; ℝ ℕ ℤ ℚ ℂ",
    "Currency: ₹100 ₽200 ₩300 ₪400 ₫500 ₴600 ₦700 ₱800 ฿900 ₿1",
    "Box: ┌─┐│└┘ ╔═╗║╚╝ ░▒▓█ ▲▼◀▶ ■□●○",
    "Arrows‐and‑hyphens‒and–dashes—and―bars",
    "Special: \u00ad soft hyphen, \u200c ZWNJ, \u200d ZWJ, \u2060 WJ, \u2061 FA",
    "Combining: a\u0301 e\u0300 o\u0302 u\u0308 n\u0303 c\u0327 (decomposed)",
    "Precomposed: á è ô ü ñ ç",
]

SPECIALS = [
    "<|denoise|>",
    "<|lang_start|>en<|lang_end|>",
    "<|lang_start|>zh<|lang_end|><|text_start|>你好<|text_end|>",
    "<|instruct_start|>Speak slowly and calmly.<|instruct_end|><|text_start|>Hello there.<|text_end|>",
    "<|denoise|><|lang_start|>ja<|lang_end|><|instruct_start|>元気に<|instruct_end|><|text_start|>こんにちは<|text_end|>",
    "text before <|denoise|> text after",
    "<|text_start|> leading space<|text_end|>",
    "<|text_start|>trailing space <|text_end|>",
    "<|text_start|>\n\nnewlines\n\n<|text_end|>",
    "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|><|denoise|><|endoftext|>",
    "<|lang_start|>ar<|lang_end|><|text_start|>مرحبا بالعالم<|text_end|>",
    "<|lang_start|>hi<|lang_end|><|text_start|>नमस्ते दुनिया<|text_end|>",
    "<|lang_start|>th<|lang_end|><|text_start|>สวัสดีชาวโลก<|text_end|>",
    "<|lang_start|>ko<|lang_end|><|text_start|>안녕하세요<|text_end|>",
    "<|lang_start|>ru<|lang_end|><|text_start|>Привет, мир!<|text_end|>",
    "<|lang_star|>not a special<|lang_end|",
    "<|denoise|><|denoise|>",
    "a<|denoise|>b<|lang_start|>c<|lang_end|>d",
    "<|text_start|>[laughter] Ha ha [sigh]<|text_end|>",
    "<|vision_start|><|image_pad|><|vision_end|> other Qwen specials",
    "<think>not special but added</think> <tool_call>x</tool_call>",
]

OTHER_SCRIPTS = [
    "ქართული ენა თბილისი",
    "Հայերեն լեզու Երևան",
    "አማርኛ ቋንቋ አዲስ አበባ",
    "བོད་སྐད་ ལྷ་ས་",
    "ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ",
    "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ",
    "ꆈꌠꁱꂷ",
    "ⵜⴰⵎⴰⵣⵉⵖⵜ",
    "ߒߞߏ",
    "ܠܫܢܐ ܣܘܪܝܝܐ",
    "Ⲙⲉⲧⲣⲉⲙⲛ̀ⲭⲏⲙⲓ",
    "𐌲𐌿𐍄𐌹𐍃𐌺𐌰",
    "𐎠𐎼𐎹",
    "ᚠᚢᚦᚨᚱᚲ",
    "ʻŌlelo Hawaiʻi",
    "Ɛ̀ʋɛ̀gbè Yorùbá Igbo Hausa",
    "𝐁𝐨𝐥𝐝 𝘪𝘵𝘢𝘭𝘪𝘤 𝓈𝒸𝓇𝒾𝓅𝓉 𝔣𝔯𝔞𝔨𝔱𝔲𝔯",
    "ⓐⓑⓒ ⒜⒝⒞ ㍿ ㈱",
]

EDGE = [
    "it'ſ", "'ſ", "'S", "'T", "'M", "'D", "'RE", "'VE", "'LL", "'Re", "'vE", "'lL",
    "'ﬅ", "'ß", "'ẞ", "'K", "'x", "'", "''", "'''", "''s", "' s", "'s's's",
    "a'sb", "a'S", "'d'd", "'dog", "'twas", "we'ren't",
    "x\u0301", "\u0301x", "\u0301", "\u0301\u0301",
    "\u00e9", "e\u0301", "\u1e0b\u0323", "d\u0323\u0307", "\u1e9b\u0323",
    "\u212b", "\u2126", "\u0958", "\u0f73", "\uf900", "\u2000",
    "a\u0301\u0327", "a\u0327\u0301", "\u1100\u1161\u11a8", "\uac01",
    "\u1100\u1161", "\u1100\u1161\u11a8\u11a8", "\u1161\u11a8",
    "\ud7a3\ud7a4", "\u0344", "\u0f81", "\u0cc0",
    "\U0001f600", "\U00010d50\U00010d50", "a\U00010d40\U00010d40b",
    "\u0e01\u0e31\u0e19", "\u0e31", "ก\u0e48\u0e49",
    "\U0001F1E6\U0001F1E7", "\U0001F1E6 \U0001F1E7",
    "\u00b2", "2\u00b2", "\u00b2\u00b2\u00b2\u00b2", "1\u00b22\u00b3",
    "\u00e9\u00e9\u00e9 caf\u00e9 CAF\u00c9", "na\u00efve",
    "\u0130", "\u0131", "I\u0307", "i\u0307",
    "\u1e9e\u00df", "\u00df\u00df",
    "ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ ﬆ",
    "℃ ℉ № ℡ ™ ℠",
    "Ⅰ Ⅱ Ⅲ ⅰ ⅱ ⅲ", "ⅠⅡⅢ",
    "٪ ٫ ٬ ۔", "،", "؟",
    "\u0600\u0601 \u06dd",
    "\u0000", "a\u0000b", "\u0001\u0002\u0003", "\u007f", "a\u007fb",
    "\u00ff\u00fe", "\ufffd", "\ufffe", "\uffff", "\U0010ffff",
    "\ue000\ue001 private use",
    "\U000e0001\U000e007f tags",
    "\u1160\u1160", "\u115f\u1160",
]

LONG = [
    "The Unicode Standard is a character coding system designed to support the "
    "worldwide interchange, processing, and display of the written texts of the "
    "diverse languages and technical disciplines of the modern world. 統一碼是一個"
    "字符編碼系統。ユニコードは文字コードの業界標準です。유니코드는 전 세계의 모든 "
    "문자를 컴퓨터에서 일관되게 표현하고 다룰 수 있도록 설계된 산업 표준이다。"
    "Юникод — стандарт кодирования символов. يونيكود هو معيار لترميز الأحرف. "
    "यूनिकोड एक कैरेक्टर एन्कोडिंग मानक है। ยูนิโคดเป็นมาตรฐานการเข้ารหัสอักขระ",
    "Mixed 123 数字 and ٤٥٦ and ४५६ digits with spaces   and\ttabs\nand newlines\r\n"
    "and “quotes” and 「brackets」 and — dashes — and … ellipses … done.",
    "<|lang_start|>en<|lang_end|><|instruct_start|>Read this like a news anchor, "
    "steady and clear.<|instruct_end|><|text_start|>Good evening. Tonight's top "
    "story: scientists have confirmed that 3.14 is, in fact, delicious. [laughter] "
    "In other news, the weather will be partly cloudy with a 40% chance of "
    "rain.<|text_end|>",
    "<|lang_start|>zh<|lang_end|><|text_start|>晚上好。今晚的头条新闻：科学家证实，"
    "3.14确实很好吃。[laughter]其他新闻，天气将多云，有40%的降雨概率。<|text_end|>",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat.",
]

SCRIPT_SAMPLES = [
    "你好世界", "こんにちは", "안녕하세요", "Привет", "Γειά", "مرحبا", "שלום",
    "नमस्ते", "বাংলা", "தமிழ்", "สวัสดี", "ສະບາຍດີ", "ខ្មែរ", "မြန်မာ", "Việt",
    "İstanbul", "Straße", "😀", "👨‍👩‍👧", "ქართული", "Հայերեն", "አማርኛ", "𐌲𐌿𐍄",
    "café", "naïve", "Ω", "²", "½",
]


def variants(sample: str) -> list[str]:
    return [
        sample,
        " " + sample,
        sample + " ",
        sample + "\n",
        "x " + sample + " y",
        sample + "123",
        "123" + sample,
        sample + "!",
        "(" + sample + ")",
        sample + "  \n  " + sample,
    ]


def nfc_stress(rng: random.Random) -> list[str]:
    """Deterministic strings that exercise decomposition, reordering and
    composition: precomposed and decomposed Latin/Greek/Vietnamese, Hangul
    jamo, out-of-order combining marks, singletons, exclusions."""
    bases = "aeiouyAEIOUcnsCNSgGkKzZ" + "αεηιουωΑΕΗΙΟΥΩ" + "กขค" + "\u0915\u0917\u0921"
    marks = ["\u0300", "\u0301", "\u0302", "\u0303", "\u0308", "\u0327", "\u0323",
             "\u0307", "\u030a", "\u0306", "\u0304", "\u030c", "\u0328", "\u0331",
             "\u0345", "\u0314", "\u0342", "\u0e48", "\u0e49", "\u093c", "\u0940"]
    precomposed = list("áàâãäåāăąçćčďđéèêëēėęěğģíìîïīįıķĺļľłńņňñóòôõöøōőŕŗřśşšţťūůűųźżž"
                       "ÁÀÂÃÄÅĀĂĄÇĆČĎĐÉÈÊËĒĖĘĚĞĢÍÌÎÏĪĮİĶĹĻĽŁŃŅŇÑÓÒÔÕÖØŌŐŔŖŘŚŞŠŢŤŪŮŰŲŹŻŽ"
                       "ầẩẫấậằẳẵắặềểễếệồổỗốộờởỡớợừửữứựỳỷỹýỵ"
                       "ᾳᾴᾶᾷᾀᾁᾂῂῃῄῆῇὰάἀἁἂἃἄἅἆἇ")
    singletons = ["\u212b", "\u2126", "\u2000", "\u2001", "\u0340", "\u0341", "\u0343",
                  "\u0374", "\u037e", "\u0387", "\u1fef", "\u1ffd", "\uf900", "\uf901",
                  "\u2f800", "\u1e9b", "\u0f73", "\u0f75", "\u0f81", "\u0958", "\u0959"]
    jamo_l = [chr(c) for c in range(0x1100, 0x1113)]
    jamo_v = [chr(c) for c in range(0x1161, 0x1176)]
    jamo_t = [chr(c) for c in range(0x11a8, 0x11c3)]
    out = []
    for _ in range(40):
        parts = []
        for _ in range(rng.randint(2, 8)):
            kind = rng.random()
            if kind < 0.3:
                parts.append(rng.choice(bases) + "".join(
                    rng.choice(marks) for _ in range(rng.randint(1, 4))))
            elif kind < 0.5:
                parts.append(rng.choice(precomposed) + "".join(
                    rng.choice(marks) for _ in range(rng.randint(0, 2))))
            elif kind < 0.65:
                parts.append(rng.choice(jamo_l) + rng.choice(jamo_v) +
                             (rng.choice(jamo_t) if rng.random() < 0.6 else ""))
            elif kind < 0.75:
                parts.append(chr(rng.randint(0xac00, 0xd7a3)) +
                             (rng.choice(jamo_t) if rng.random() < 0.3 else ""))
            elif kind < 0.85:
                parts.append(rng.choice(singletons))
            else:
                parts.append(rng.choice(["", " ", "x", "1", "。", "\n"]))
        out.append("".join(parts))
    return out


def build_corpus() -> list[str]:
    rng = random.Random(20240915)
    corpus: list[str] = []
    for group in (ENGLISH, CODE, NUMBERS, WHITESPACE, CHINESE, JAPANESE, KOREAN,
                  CYRILLIC_GREEK, ARABIC_HEBREW, INDIC, SEASIA, EUROPEAN, EMOJI,
                  PUNCT, SPECIALS, OTHER_SCRIPTS, EDGE, LONG):
        corpus.extend(group)
    for s in SCRIPT_SAMPLES:
        corpus.extend(variants(s))
    corpus.extend(nfc_stress(rng))
    # Dedupe, keep order.
    seen = set()
    out = []
    for s in corpus:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnivoice", default=os.path.join(here, "..", "..", "OmniVoice"))
    ap.add_argument("--tts", default=os.path.join(
        here, "..", "..", "brosoundml", "weights", "qwen-tts", "0.6B-Base"))
    ap.add_argument("--out", default=os.path.join(
        here, "..", "tests", "ref", "qwen_tokenizer_unicode.json"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.omnivoice, "tokenizer.json"))
    added = set(t.content for t in tok.get_added_tokens_decoder().values())
    assert all(s in added for s in OMNIVOICE_SPECIALS), "OmniVoice specials missing"

    tts = None
    tts_added: set[str] = set()
    if os.path.isfile(os.path.join(args.tts, "vocab.json")):
        from transformers import AutoTokenizer
        tts = AutoTokenizer.from_pretrained(args.tts)
        tts_added = set(tts.get_added_vocab().keys())

    corpus = build_corpus()
    cases = []
    for s in corpus:
        case = {"text": s, "ids": tok.encode(s).ids}
        if tts is not None and not any(a in s for a in tts_added | added):
            case["ids_tts"] = tts(s, add_special_tokens=False)["input_ids"]
        cases.append(case)

    nfc_pairs = []
    for s in corpus:
        norm = tok.normalizer.normalize_str(s)
        assert norm == unicodedata.normalize("NFC", s), repr(s)
        if norm != s:
            nfc_pairs.append({"in": s, "out": norm})

    fixture = {
        "generator": "scripts/gen_qwen_tokenizer_fixture.py",
        "tokenizers_version": tokenizers.__version__,
        "unicode_version": unicodedata.unidata_version,
        "python_version": sys.version.split()[0],
        "extra_special_tokens": OMNIVOICE_SPECIALS,
        "case_count": len(cases),
        "tts_case_count": sum(1 for c in cases if "ids_tts" in c),
        "cases": cases,
        "nfc": nfc_pairs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fixture, f, ensure_ascii=True, indent=None, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {args.out}: {len(cases)} cases ({fixture['tts_case_count']} with "
          f"ids_tts), {len(nfc_pairs)} nfc pairs, tokenizers {tokenizers.__version__}, "
          f"Unicode {unicodedata.unidata_version}")


if __name__ == "__main__":
    main()
