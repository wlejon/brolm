"""Reference fixture for brolm's Laya port: tests/ref/laya_parity.json.

Runs the upstream reference (rl_common.py / rl_agent_api.py shipped beside the
weights) over a varied set of calls and records, per question:

  - input ids and [MASK] marker positions (build_sequence, exact),
  - raw scorer logits and raw act logits, in fp32 (no autocast: the numeric
    truth) and under the checkpoint's own bf16 autocast (what upstream ships;
    the fp32-vs-bf16 gap is the noise floor a half-precision port is judged
    against),
  - the act probability and the temperature-calibrated probabilities,
    computed exactly as RLAgent.system_one does,
  - or the ValueError text when the options do not fit.

JSON states are stored both as the reference serialisation (json.dumps,
ensure_ascii=False) and as the compact form JSON.stringify produces, so the
C++ side can check its Python-style re-spacer.

One fixture per checkpoint of the Laya family (the repo root is English, the
other two are subfolders):

    english          -> laya_parity.json
    multilingual     -> laya_parity_multilingual.json
    typed-decisions  -> laya_parity_typed_decisions.json

The non-English fixtures add their own calls (laya_cases.py): states and
questions in several languages and scripts for multilingual, the four
typed-decisions workflows for typed-decisions, and a harder tokenizer battery
(CJK, Devanagari, Arabic, emoji, whitespace runs, byte-fallback characters,
added tokens) for the Metaspace/ByteFallback BPE.

Usage (needs torch + transformers + safetensors; CUDA optional):
    USE_TF=0 python tests/ref/gen_laya_parity.py [CHECKPOINT] [LAYA_DIR]
CHECKPOINT defaults to english; LAYA_DIR to ../laya relative to the brolm
checkout.
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = {"english": ("", "laya_parity.json"),
               "multilingual": ("multilingual", "laya_parity_multilingual.json"),
               "typed-decisions": ("typed-decisions", "laya_parity_typed_decisions.json")}
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "english"
if CHECKPOINT not in CHECKPOINTS:
    sys.exit("unknown checkpoint %r (one of %s)" % (CHECKPOINT, ", ".join(CHECKPOINTS)))
LAYA_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(os.path.join(HERE, "..", "..", "..", "laya"))
LAYA_DIR = os.path.join(LAYA_ROOT, CHECKPOINTS[CHECKPOINT][0]) if CHECKPOINTS[CHECKPOINT][0] else LAYA_ROOT
sys.path.insert(0, LAYA_ROOT)  # rl_common.py / rl_agent_api.py live at the repo root
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_agent_api import RLAgent  # noqa: E402
from rl_common import QTYPES, build_sequence, collate_items, render_options, serialize_state, temp_bucket  # noqa: E402
import laya_cases  # noqa: E402

OUT = os.path.join(HERE, CHECKPOINTS[CHECKPOINT][1])

EMAIL = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}

DEPT = {"type": "choice", "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts", "other": "everything else"}}
URGENCY = {"type": "score", "instructions": "How urgent is this request?",
           "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]}
CHURN = {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
REFUND = {"type": "noul", "instructions": "Does the user explicitly request a refund?"}

LOREM = ("The quarterly report shows revenue grew 12% while support tickets fell. Customers in the EU asked about "
         "GDPR exports, and two enterprise accounts reported intermittent 502 errors from the billing API. ")
TICKET_LINES = ["Customer: my dashboard has been blank since the 3.2 upgrade.",
                "Agent: could you share a screenshot and your browser version?",
                "Customer: Chrome 128, screenshot attached. It is still blank after clearing the cache.",
                "Agent: thanks, escalating to engineering.",
                "Customer: this is the third time this month, I am considering switching providers."]

INTENTS = ["activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
           "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
           "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
           "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
           "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
           "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working",
           "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
           "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
           "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
           "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
           "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
           "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
           "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
           "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
           "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
           "top_up_limits", "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
           "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
           "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
           "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
           "wrong_exchange_rate_for_cash_withdrawal"]


def calls(tok, cfg):
    """(name, state, questions, overrides). overrides: max_len / head_max_len / truncate_left."""
    max_len, head_max_len = cfg["max_len"], cfg["head_max_len"]
    out = []
    out.append(("readme_email", EMAIL, {"department": DEPT, "urgency": URGENCY, "churn_risk": CHURN,
                                        "refund_requested": REFUND}, {}))
    out.append(("short_text", "I love this product, five stars!", {
        "sentiment": {"type": "choice", "instructions": "What is the sentiment?",
                      "criteria": ["positive", "negative", "neutral"]},
        "spam": {"type": "noul", "instructions": "Is this spam?",
                 "criteria": {"false": "a genuine customer message", "true": "unsolicited advertising"}},
        "binary_choice": {"type": "choice", "instructions": "Pick one", "criteria": {"yes": "", "no": ""}},
        "score2": {"type": "score", "instructions": "Is it praise?", "criteria": ["no", "yes"]},
    }, {}))
    out.append(("empty_state", "", {"q": {"type": "noul", "instructions": "Is there any content?"},
                                     "s": {"type": "score", "instructions": "Rate completeness.",
                                           "criteria": ["none", "some", "a lot", "everything", "more"]}}, {}))
    seven = {"c%d" % i: "category number %d of seven" % i for i in range(7)}
    twelve = {w: "" for w in ["world", "sports", "business", "science", "health", "politics", "travel", "food",
                              "music", "film", "fashion", "education"]}
    thirty = {"label_%02d" % i: "a longer description for label %d that talks about several things at once" % i
              for i in range(30)}
    out.append(("many_options", LOREM, {"seven": {"type": "choice", "instructions": "Which category?", "criteria": seven},
                                        "twelve": {"type": "choice", "instructions": "Topic?", "criteria": twelve},
                                        "thirty": {"type": "choice", "instructions": "Which label fits best?",
                                                   "criteria": thirty}}, {}))
    out.append(("banking77_default", "I was charged twice for the same coffee purchase yesterday.",
                {"intent": {"type": "choice", "instructions": "Which banking intent is this?", "criteria": INTENTS}}, {}))
    out.append(("banking77_raised", "I was charged twice for the same coffee purchase yesterday.",
                {"intent": {"type": "choice", "instructions": "Which banking intent is this?", "criteria": INTENTS}},
                {"max_len": 1024, "head_max_len": 512}))
    out.append(("too_many_options", "x", {"huge": {"type": "choice", "instructions": "pick",
                                                   "criteria": ["option %d" % i for i in range(200)]}}, {}))
    long_text = (LOREM * 40).strip()
    lq = {"urgent": {"type": "noul", "instructions": "Is anything urgent?"},
          "topic": {"type": "choice", "instructions": "Main topic?", "criteria": {"finance": "", "support": "",
                                                                                   "legal": ""}}}
    out.append(("long_right", long_text, lq, {}))
    out.append(("long_left", long_text, lq, {"truncate_left": True}))
    out.append(("long_1024", long_text, lq, {"max_len": 1024}))
    out.append(("long_2048_left", long_text + " " + long_text, lq, {"max_len": 2048, "truncate_left": True}))
    # near max_len: the longest word prefix whose state still fits untruncated
    # for the first question (exactly max_len tokens, or one short).
    words = long_text.split(" ")
    q0 = to_internal(lq["urgent"])
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        st = " ".join(words[:mid])
        n_state = len(tok(st, add_special_tokens=False)["input_ids"])
        n_head = len(build_sequence(tok, "", q0, max_len, head_max_len)[0]) - 1
        if n_head + n_state + 1 <= max_len:
            lo = mid
        else:
            hi = mid - 1
    out.append(("near_max", " ".join(words[:lo]), lq, {}))
    out.append(("near_max_plus1", " ".join(words[:lo + 1]), lq, {}))
    conv = {"channel": "chat", "customer_tier": "gold", "conversation": TICKET_LINES * 8}
    cq = {"churn": {"type": "noul", "instructions": "Will the customer churn?"},
          "sentiment": {"type": "score", "instructions": "Customer frustration level",
                        "criteria": ["calm", "mildly annoyed", "frustrated", "furious"]}}
    out.append(("conversation_left", conv, cq, {"truncate_left": True}))
    out.append(("conversation_right", conv, cq, {}))
    out.append(("list_state", TICKET_LINES, cq, {}))
    out.append(("unicode", "Café crème brûlée — naïve résumé. 東京で会議があります。 Ünïcödé 🚀🔥 emoji. "
                           "مرحبا بالعالم. नमस्ते दुनिया। Café (combining). Ωmega ß ẞ ﬁ ligature.",
                {"lang": {"type": "choice", "instructions": "Dominant language?",
                          "criteria": {"english": "", "japanese": "", "arabic": "", "hindi": "", "mixed": ""}},
                 "emoji": {"type": "noul", "instructions": "Does it contain emoji? 😀"}}, {}))
    out.append(("specials_and_spaces",
                "Contact me at |||EMAIL_ADDRESS||| or |||PHONE_NUMBER|||.  Two  spaces,   three,\t\ttabs\n\nnewlines\r\n"
                "[MASK] should vanish; [CLS] and [SEP] literal. Numbers 1234567890 and 3.14159, it's they're we'll "
                "I'M DON'T. <|endoftext|> end.      ",
                {"odd": {"type": "noul", "instructions": "Is the [MASK] text odd?"},
                 "fmt": {"type": "choice", "instructions": "Format?",
                         "criteria": {"email": "an [MASK] email", "chat": "chat message", "log": "machine log"}}},
                {}))
    nested = {"order": {"id": 99812, "items": [{"sku": "A-1", "qty": 2, "price": 19.99},
                                               {"sku": "B-7", "qty": 1, "price": 5.5}],
                        "notes": "Deliver after 5pm \"please\"\nback door", "gift": True, "coupon": None},
              "customer": {"name": "Zoë Ångström", "vip": False}}
    out.append(("json_nested", nested, {"fraud": {"type": "noul", "instructions": "Is this order likely fraud?"},
                                        "value": {"type": "score", "instructions": "Order value",
                                                  "criteria": ["low", "medium", "high"]},
                                        "dept": DEPT}, {}))
    out.append(("long_instructions", EMAIL, {
        "long": {"type": "choice",
                 "instructions": "Considering the full context of the message, the history of the account, the "
                                 "tone used, any explicit or implicit threats, the monetary amounts mentioned, and "
                                 "the company's written policy on refunds and escalations, which team should own "
                                 "the follow-up and why would they be the right owner for this particular case? " * 3,
                 "criteria": DEPT["criteria"]}}, {}))
    out.append(("small_head_budget", EMAIL, {"department": DEPT, "urgency": URGENCY}, {"head_max_len": 48}))
    return out


TOKENIZER_STRINGS = [
    "", " ", "  ", "   ", "a", " a", "a ", "a  b", "a   b", "a\tb", "a\t\tb", "a\nb", "a\n\nb", "a \n b", "a\n  b",
    "end.\n", "end.\n\nNext", "x!\n\n", "hi!!!  ", "  leading", "trailing  ", "\n", "\r\n", "\r\n\r\n",
    "it's", "IT'S", "we're they've I'm you'll he'd", "WE'RE THEY'VE", "don't won't can't", "'s 's 'S",
    "1234567890", "3.14159", "12,345.67", "v1.2.3-rc4", "x1y2z3", " 42", "٣٤٥ ٦", "²³ ½",
    "hello, world!", "(parens) [brackets] {braces}", "email@example.com", "https://example.com/a?b=c&d=e",
    "#hashtag @mention $100 50% a&b", "C++ / C# -> ok", "\"quoted\" 'single'", "...", "---", "—–-",
    "Café crème brûlée", "Café", "naïve résumé", "東京で会議があります。", "한국어 텍스트", "Привет, мир!",
    "مرحبا بالعالم", "नमस्ते दुनिया।", "🚀🔥 emoji 👍🏽", "👨‍👩‍👧", "ﬁ ligature ß ẞ", "Ωmega µ",
    " nbsp ", "zero​width", "　ideographic space", "tab\tand nbsp",
    "|||EMAIL_ADDRESS|||", "a|||PHONE_NUMBER|||b", "[CLS] [SEP] [PAD] [UNK]", "<|endoftext|>", "[unused3]x",
    "       seven spaces", "x" + " " * 30 + "y", "\t\t\t", "a" * 200, "ab" * 50,
]


def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"])
    return {"t": t, "ins": ins, "crit": crit}


def run_model(agent, items, dtype):
    b = collate_items([items], agent.tok.pad_token_id)
    dev = agent.device
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dtype or torch.bfloat16,
                                         enabled=dtype is not None and dev.type == "cuda"):
        logits, act = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                  b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits.float().cpu().numpy(), act.float().cpu().numpy()


def calibrate(agent, qt, logits):
    k = len(logits)
    T = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    z = np.asarray(logits, dtype=np.float64) / T
    p = np.exp(z - z.max())
    return T, (p / p.sum()).tolist()


def main():
    agent = RLAgent(LAYA_DIR)
    fixtures = []
    all_calls = calls(agent.tok, agent.cfg)
    strings = list(TOKENIZER_STRINGS)
    if CHECKPOINT == "multilingual":
        all_calls += laya_cases.multilingual_calls()
        strings += laya_cases.HARD_TOKENIZER_STRINGS
    elif CHECKPOINT == "typed-decisions":
        all_calls += laya_cases.typed_decisions_calls()
        all_calls += laya_cases.multilingual_calls()[:2]
    for name, state, questions, ov in all_calls:
        max_len = ov.get("max_len", agent.cfg["max_len"])
        head_max_len = ov.get("head_max_len", agent.cfg["head_max_len"])
        tl = bool(ov.get("truncate_left", False))
        entry = {"name": name, "state": serialize_state(state), "max_len": max_len, "head_max_len": head_max_len,
                 "truncate_left": tl, "questions_json": json.dumps(questions, ensure_ascii=False),
                 "questions": []}
        if not isinstance(state, str):
            entry["state_compact"] = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        items, error = [], None
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len, truncate_left=tl)
            k = len(render_options(q))
            entry["questions"].append({"id": qid, "def": qdef, "ids": ids, "markers": markers, "n_options": k})
            if len(markers) != k and error is None:
                error = "question %r: options do not fit in head_max_len=%d tokens" % (qid, head_max_len)
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers),
                          "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
        if error:
            entry["error"] = error
            fixtures.append(entry)
            print("%-20s error: %s" % (name, error))
            continue
        lg32, act32 = run_model(agent, items, None)
        lg16, act16 = run_model(agent, items, torch.bfloat16)
        for r, qe in enumerate(entry["questions"]):
            k = len(qe["markers"])
            qt = QTYPES[qe["def"]["type"]]
            a32 = torch.softmax(torch.tensor(act32[r], dtype=torch.float64), -1).tolist()
            a16 = torch.softmax(torch.tensor(act16[r], dtype=torch.float64), -1).tolist()
            T, probs = calibrate(agent, qt, lg32[r, :k])
            _, probs16 = calibrate(agent, qt, lg16[r, :k])
            qe.update({"qtype": qt, "temperature": T,
                       "logits_fp32": lg32[r, :k].tolist(), "logits_bf16": lg16[r, :k].tolist(),
                       "act_logits_fp32": act32[r].tolist(), "act_logits_bf16": act16[r].tolist(),
                       "act_probability_fp32": a32[0], "act_probability_bf16": a16[0],
                       "probs_fp32": probs, "probs_bf16": probs16})
        fixtures.append(entry)
        print("%-20s %d questions, lens %s" % (name, len(items), [len(it["ids"]) for it in items]))
    tok_cases = [{"text": s, "ids": agent.tok(s, add_special_tokens=False)["input_ids"]} for s in strings]
    t = agent.tok
    special = {"cls": t.cls_token_id, "sep": t.sep_token_id, "pad": t.pad_token_id, "mask": t.mask_token_id,
               "mask_token": t.mask_token}
    doc = {"generator": "tests/ref/gen_laya_parity.py", "torch": torch.__version__, "tokenizer": tok_cases,
           "calls": fixtures}
    if CHECKPOINT != "english":  # the English fixture predates these keys; keep it byte-stable
        doc.update({"checkpoint": CHECKPOINT, "subdir": CHECKPOINTS[CHECKPOINT][0], "special": special,
                    "config": {k: agent.cfg[k] for k in ("encoder", "max_len", "head_max_len", "max_prefixes")}})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
