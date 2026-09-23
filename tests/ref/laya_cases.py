"""Extra fixture calls for the non-English Laya checkpoints (see gen_laya_parity.py).

multilingual_calls():   states and questions in several languages and scripts.
typed_decisions_calls(): the four typed-decisions workflows (invoice processing,
                         security incidents, customer service, agent traces).
HARD_TOKENIZER_STRINGS: inputs that stress a Metaspace/ByteFallback BPE —
                         scripts without spaces, combining marks, emoji
                         sequences, characters outside the vocabulary (byte
                         fallback), whitespace runs, literal metaspaces and the
                         tokenizer's added tokens.
"""

DEPT_ES = {"type": "choice", "instructions": "¿Qué departamento debe atender esta solicitud?",
           "criteria": {"facturación": "facturas, pagos, reembolsos", "técnico": "errores, caídas del sistema",
                        "ventas": "precios, contratos nuevos", "otro": "todo lo demás"}}
URGENCY_HI = {"type": "score", "instructions": "यह अनुरोध कितना ज़रूरी है?",
              "criteria": ["ज़रूरी नहीं", "जल्द", "बहुत ज़रूरी, काम रुका हुआ है"]}
REFUND_ZH = {"type": "noul", "instructions": "用户是否明确要求退款？"}
CHURN_AR = {"type": "noul", "instructions": "هل يهدد المستخدم بإلغاء الاشتراك أو المغادرة؟",
            "criteria": {"false": "لا يوجد تهديد", "true": "يهدد بالإلغاء"}}
SENTIMENT_JA = {"type": "choice", "instructions": "この文章の感情は？", "criteria": ["ポジティブ", "ネガティブ", "中立"]}
TOPIC_RU = {"type": "choice", "instructions": "Какая тема сообщения?",
            "criteria": {"оплата": "счета и платежи", "доставка": "сроки и адреса", "техника": "ошибки приложения"}}
LANG_Q = {"type": "choice", "instructions": "Which language is the state written in?",
          "criteria": {"english": "", "hindi": "", "chinese": "", "arabic": "", "spanish": "", "japanese": "",
                       "russian": "", "korean": "", "thai": "", "swahili": ""}}

ZH_PARA = ("我们三月份被重复扣款了两次，发票号是4411。请今天把重复的款项退回，否则我们将取消订阅。"
           "客服电话一直占线，邮件也没有人回复，这已经是本月第三次出现类似的问题了。")


def multilingual_calls():
    out = []
    out.append(("hi_email", {"from": "user@acme.in", "subject": "चालान #4411 पर दोहरा शुल्क",
                             "body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। वरना मैं सदस्यता रद्द कर दूँगा।"},
                {"department": DEPT_ES, "urgency": URGENCY_HI, "refund": REFUND_ZH, "churn": CHURN_AR}, {}))
    out.append(("zh_text", ZH_PARA, {"refund": REFUND_ZH, "lang": LANG_Q, "sentiment": SENTIMENT_JA}, {}))
    out.append(("ar_text", "مرحبا، تم خصم المبلغ مرتين من بطاقتي. أريد استرداد المبلغ فوراً وإلا سألغي حسابي.",
                {"churn": CHURN_AR, "lang": LANG_Q, "topic": TOPIC_RU}, {}))
    out.append(("es_ticket", {"canal": "correo", "mensaje": "La aplicación se cierra al abrir el carrito desde la "
                                                            "actualización 3.2. ¡Ya van tres veces este mes!"},
                {"department": DEPT_ES, "lang": LANG_Q}, {}))
    out.append(("ja_review", "この製品は本当に素晴らしいです！配送も早くて、梱包も丁寧でした。星五つ⭐⭐⭐⭐⭐",
                {"sentiment": SENTIMENT_JA, "lang": LANG_Q}, {}))
    out.append(("ru_ko_th_sw", "Здравствуйте! 배송이 아직 안 왔어요. การจัดส่งล่าช้ามาก Habari, mzigo wangu haujafika.",
                {"lang": LANG_Q, "topic": TOPIC_RU}, {}))
    out.append(("mixed_scripts_emoji", "Order #99 🚀 — naïve café, 東京, Москва, मुंबई, القاهرة, 서울 👨‍👩‍👧 🇮🇳 1️⃣ "
                                       "é (decomposed)  nbsp 　ideographic\tTAB  two  spaces",
                {"lang": LANG_Q, "emoji": {"type": "noul", "instructions": "¿Contiene emojis? 😀"}}, {}))
    out.append(("added_tokens_state", "<start_of_turn>user\nPlease <b>refund</b> me.<end_of_turn>\n\n\n<mask> "
                                      "<table><tr><td>4411</td></tr></table>\t\tdone ▁▁ <unused5> [@BOS@] <2mass>",
                {"refund": REFUND_ZH, "fmt": {"type": "choice", "instructions": "Formato del <mask> texto?",
                                              "criteria": {"html": "marcado <b>html</b>", "chat": "un chat",
                                                           "plain": "texto plano"}}}, {}))
    long_zh = ZH_PARA * 30
    out.append(("zh_long_right", long_zh, {"refund": REFUND_ZH, "churn": CHURN_AR}, {}))
    out.append(("zh_long_left", long_zh, {"refund": REFUND_ZH}, {"truncate_left": True}))
    conv_hi = {"channel": "chat", "conversation": ["ग्राहक: मेरा डैशबोर्ड खाली है।", "एजेंट: कृपया स्क्रीनशॉट भेजें।",
                                                   "ग्राहक: यह इस महीने तीसरी बार है, मैं कंपनी बदलने की सोच रहा हूँ।"] * 12}
    out.append(("hi_conversation_left", conv_hi, {"churn": CHURN_AR, "urgency": URGENCY_HI}, {"truncate_left": True}))
    return out


INVOICE = {"vendor": "Nordwind Logistics GmbH", "invoice_no": "NW-2024-0931", "po_number": "PO-7781",
           "amount": 18450.00, "currency": "EUR", "due": "2024-10-15",
           "lines": [{"desc": "Freight Hamburg-Rotterdam", "qty": 3, "unit": 4800.0},
                     {"desc": "Fuel surcharge", "qty": 1, "unit": 4050.0}],
           "notes": "Bank details changed: please pay to the new IBAN DE89 3704 0044 0532 0130 00 from now on."}
INCIDENT = {"source": "EDR", "host": "fin-ws-042", "user": "j.doe",
            "events": ["powershell.exe -enc JABjAGwAaQBlAG4AdAA...", "outbound 443 to 185.220.101.4 (tor exit)",
                       "lsass.exe handle opened by rundll32.exe", "new local admin account 'svc_backup2'"],
            "first_seen": "02:14 UTC", "business_hours": False}
TRACE = {"task": "Book the cheapest refundable flight LHR->JFK on 12 Oct",
         "steps": [{"tool": "search_flights", "result": "14 options"},
                   {"tool": "filter", "args": {"refundable": True}, "result": "3 options"},
                   {"tool": "book", "args": {"flight": "VS3", "fare": "Economy Light"},
                    "result": "booked, non-refundable"}],
         "final_answer": "Booked the cheapest refundable flight VS3."}


def typed_decisions_calls():
    out = []
    out.append(("td_invoice", INVOICE, {
        "approve": {"type": "choice", "instructions": "What should accounts payable do with this invoice?",
                    "criteria": {"approve": "matches PO, pay on schedule", "hold": "needs verification first",
                                 "reject": "invalid or fraudulent"}},
        "bank_change": {"type": "noul", "instructions": "Does the invoice request a change of payment details?"},
        "risk": {"type": "score", "instructions": "Payment-fraud risk",
                 "criteria": ["none", "low", "moderate", "high", "critical"]}}, {}))
    out.append(("td_security", INCIDENT, {
        "severity": {"type": "score", "instructions": "Incident severity",
                     "criteria": ["informational", "low", "medium", "high", "critical"]},
        "category": {"type": "choice", "instructions": "Most likely incident category?",
                     "criteria": {"credential_theft": "", "ransomware": "", "insider": "", "false_positive": ""}},
        "isolate": {"type": "noul", "instructions": "Should the host be isolated immediately?"}}, {}))
    out.append(("td_customer", {"channel": "email", "tier": "enterprise",
                                "body": "Third outage this month. Our SLA says 99.9%. We expect service credits and "
                                        "a call with your VP of Engineering, or legal will get involved."},
                {"route": {"type": "choice", "instructions": "Route the ticket",
                           "criteria": {"support_l1": "", "support_l2": "", "account_manager": "", "legal": ""}},
                 "credit": {"type": "noul", "instructions": "Is the customer asking for service credits?"},
                 "tone": {"type": "score", "instructions": "Escalation level of the tone",
                          "criteria": ["calm", "concerned", "angry", "threatening"]}}, {}))
    out.append(("td_trace", TRACE, {
        "success": {"type": "noul", "instructions": "Did the agent complete the task correctly?"},
        "failure": {"type": "choice", "instructions": "Primary failure mode, if any",
                    "criteria": {"none": "", "wrong_tool_args": "", "hallucinated_result": "",
                                 "ignored_constraint": "", "incomplete": ""}},
        "honesty": {"type": "score", "instructions": "How faithful is the final answer to the trace?",
                    "criteria": ["contradicts it", "partly faithful", "faithful"]}}, {}))
    return out


HARD_TOKENIZER_STRINGS = [
    # scripts without spaces, conjuncts, RTL
    "东京是日本的首都，也是世界上最大的城市之一。", "我们三月份被重复扣款了两次", "東京で会議があります。",
    "カタカナとひらがなと漢字", "한국어 텍스트를 토큰화합니다", "नमस्ते दुनिया। क्षत्रिय द्वार श्री",
    "मुझसे दो बार शुल्क लिया गया", "مرحبا بالعالم", "العربية: ١٢٣٤٥ ٦٧٨", "עִבְרִית עם ניקוד",
    "ภาษาไทยไม่มีช่องว่างระหว่างคำ", "ភាសាខ្មែរ", "ქართული ენა", "தமிழ் மொழி", "አማርኛ", "Ελληνικά ΚΕΦΑΛΑΙΑ",
    "Привет, мир! ЁЖИК ёжик", "Tiếng Việt có dấu: Hà Nội", "Türkçe İı Ğğ Şş",
    # combining marks / normalisation forms (this tokenizer does not normalise)
    "é vs é", "Café", "ñö", "Z͑͠a͛l͆g̔o̒",
    "ﬁ ligature ß ẞ", "Ω Ω Å Å",
    # emoji sequences
    "🚀🔥👍🏽", "👨‍👩‍👧‍👦", "🇮🇳🇯🇵🇺🇸", "1️⃣ #️⃣", "🏳️‍🌈", "🫨🫎🪿", "☺︎ ☺️",
    # characters outside the vocabulary -> byte fallback (and control bytes)
    "\x00\x01\x02", "\x7f\x1b[0m", "�", "\U00020000\U0002a6d6", "", "\U000e0041\U000e007f",
    "\U00013000 hieroglyph", "ꙮ", "\U0001f9ff\U0001fae8", "mixed\x00null", "​‌‍⁠",
    # whitespace runs
    " ", "  ", "     ", "a b", "a  b", "a   b", " a", "  a", "a ", "a  ", "\t", "\t\t", "\t\t\t", "a\tb", "a\t\tb",
    "\n", "\n\n", "\n\n\n", "a\nb", "a\n\nb", "a \n b", "a\n  b", "\r\n", "a\r\nb", "\n \n", " \n", "\n ",
    " ", "a b", "　", "a　b", " em en", "x" + " " * 40 + "y", "\n" * 35,
    "\t" * 35, " \t \t ", "trailing\n\n", "   leading",
    # literal metaspaces and byte-token spellings
    "▁", "▁▁", "x▁▁y", "▁▁▁▁ four", "▁a ▁b", "<0x41>", "<0xE4><0xB8>",
    # added / special tokens appearing in text
    "<bos>", "<eos>x<pad>", "<mask>", "a <mask> b", "<unk>", "<start_of_turn>user\nhi<end_of_turn>",
    "<b>bold</b> <i>it</i>", "<table><tr><td>1</td></tr></table>", "<unused0><unused99>", "[@BOS@]", "<2mass>",
    "[toxicity=0]", "<h1>Title</h1>\n\n\n<code>x</code>", "<s>no</s>", "[MASK] [CLS] [SEP]", "<|endoftext|>",
    # mixed
    "Order #4411 — 重复扣款 — दोहरा शुल्क — خصم مرتين 🚀", "JSON: {\"k\": \"值\", \"n\": 1.5e-3}",
    "https://例子.测试/路径?q=值&x=1", "user@例え.jp", "C++ / C# -> ok", "12,345.67 ₹ € ¥ £", "a" * 300,
]
