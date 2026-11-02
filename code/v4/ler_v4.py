"""Fourth study (PREREGISTRATION_V4.md, release prereg-v4): what the answer-polarity part W follows (True/False
answers, the negated question), whether post-training or the chat template produces it, the order of the solved
examples in both framings, a paraphrased prompt, and direction words against numbers at the level of the combined
effect.

Everything that the earlier studies registered (data, splits, XGBoost, cards, solved examples, the registered prompts,
the reading of the answer-word log-probabilities, the statistics a and a', the cue card and the vectorised estimators)
comes from ler_main.py (main-2.0) and ler_v3.py (v3-1.1), which are not changed (their SHA-256 are checked below).
This module adds only what the fourth study needs:

  1. the new prompt formats (True/False statements, the negated question, the paraphrased prompt, the net-number
     card), the prompt builder and the checks that each new prompt differs from the registered one only where the
     protocol says;
  2. the prompt grids, the models, the decisions of the engineering check and the check against the protocol text;
  3. the reading of a framing with its own answer words, the plain and chat formats, generated answers and their
     parsing;
  4. the smoke test and the engineering check;
  5. a fake model for the CPU tests;
  6. the registered analysis: validity, O and W of every pair, the changes against the registered pair, the 2 x 2
     comparison of R10, the movement comparison of R13, the decision rules of R8 to R13 and the descriptive Q14.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

import ler_main as L
import ler_v3 as V

V4_VERSION = "v4-1.2"
MAIN_SHA256 = "0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e"     # ler_main.py, main-2.0
V3_SHA256 = "89d4c0c0ed995e82aa472eeac3f22ea0728583c7c8c84a01c87befe18f5e92ec"       # ler_v3.py, v3-1.1
PROTOCOL_V4_TAG = "prereg-v4"
PROTOCOL_V4_URL = f"{L._RAW}/{PROTOCOL_V4_TAG}/PREREGISTRATION_V4.md"
PROTOCOL_AT_V4_URLS = {name: f"{L._RAW}/{PROTOCOL_V4_TAG}/{name}"
                       for name in ("PREREGISTRATION.md", "PREREGISTRATION_V2.md", "PREREGISTRATION_V3.md")}
V4_MARKERS = ["## 4. Models", "## 5. Prompt formats", "## 6. Prompts", "### R8.", "### R9.", "### Joint reading",
              "### R10.", "### R11.", "### R12.", "### R13.", "### Q14.", "## 8. Analysis", "## 10."]
N_BOOT = 10_000
N_BOOT_BOUND = 2_000
SPLIT_CONDS = ("C0", "C2", "C4")

# =============================================================================================================
# 1. Prompt formats (PREREGISTRATION_V4.md, Section 5)
# =============================================================================================================
TF_WORDS = ("True", "False")
YN_WORDS = ("Yes", "No")
TF_INSTRUCTION = "Is this statement true or false? Answer with True or False."
STATEMENT = "Statement: "
QUESTION = "Question: "
# 5.1: the statements are the answer-option sentences of the second study (ler_main.CHOICE_OPTIONS), word for word
NEG_Q = {"physionet2012": "Will this patient not die during the current hospital stay?",
         "physionet2019": "Will this patient not develop sepsis later in this ICU stay?",
         "diabetes130": "Will this patient not be readmitted within 30 days of discharge?"}
# 5.4: the paraphrased prompt
PARA_EXAMPLES_HEADER = "Four earlier cases and what actually happened to them:"
PARA_LABEL = "Case"
PARA_PATIENT_HEADER = "The case to assess:"
PARA_INSTRUCTION = "Reply with Yes or No."
PARA_TASK = {
    "physionet2012": "estimate whether a patient in the intensive care unit (ICU) will die before leaving the hospital, "
                     "using the measurements recorded in the first 48 hours of the ICU stay.",
    "physionet2019": "estimate whether a patient in the intensive care unit (ICU) will develop sepsis after the first "
                     "24 hours of the ICU stay, using the measurements recorded in those 24 hours.",
    "diabetes130": "estimate whether a patient with diabetes who was treated in hospital will be admitted to hospital "
                   "again within 30 days of going home, using the information recorded during the stay."}
PARA_Q = {
    "physionet2012": {"alarm": "Is this patient going to die before leaving the hospital?",
                      "reversed": "Is this patient going to leave the hospital alive?"},
    "physionet2019": {"alarm": "Is this patient going to get sepsis later in this ICU stay?",
                      "reversed": "Is this patient going to remain sepsis-free for the rest of this ICU stay?"},
    "diabetes130": {"alarm": "Is this patient going to be admitted again within 30 days of going home?",
                    "reversed": "Is this patient going to stay out of hospital for the 30 days after going home?"}}
# 5.5: the net-number card; the cue is ler_v3.CUE_TEXT
NET_TEXT = ("Evidence from a machine-learning model (XGBoost) trained on other patients: taken together, the {n} "
            "factors that most influenced its risk estimate for this patient have a combined impact of {impact} on the "
            "risk of {outcome}.")

# framing -> how the question is asked, the two answer words, and whether the first word stands for the alarming
# outcome; 'named_alarm' = the question or statement names the alarming outcome
FRAMINGS_V4 = {
    "alarm": dict(form="registered", words=YN_WORDS, pos_alarm=True, named_alarm=True),
    "reversed": dict(form="registered", words=YN_WORDS, pos_alarm=False, named_alarm=False),
    "tf_alarm": dict(form="statement", words=TF_WORDS, pos_alarm=True, named_alarm=True),
    "tf_reversed": dict(form="statement", words=TF_WORDS, pos_alarm=False, named_alarm=False),
    "negated": dict(form="negated", words=YN_WORDS, pos_alarm=False, named_alarm=True),
    "para_alarm": dict(form="paraphrase", words=YN_WORDS, pos_alarm=True, named_alarm=True),
    "para_reversed": dict(form="paraphrase", words=YN_WORDS, pos_alarm=False, named_alarm=False),
}
# example-set / format labels (the record column shot_set):
#   A     = registered solved examples and card;   Asf  = set A in the order safe, alarming, safe, alarming
#   Acue  = the cue of the third study;            Anet = the net-number card
#   Abase = the registered prompt rendered as plain text (an instruction-tuned model; the second study's label)
#   Achat = the registered prompt rendered through the instruction-tuned model's chat template (a base model)
SETS_V4 = ("A", "Asf", "Acue", "Anet", "Abase", "Achat")


def answer_words(framing: str) -> tuple:
    return FRAMINGS_V4[framing]["words"]


def pos_alarm(framing: str) -> bool:
    return FRAMINGS_V4[framing]["pos_alarm"]


def _impact(x: float) -> str:
    return ("+" if x > 0 else "-") + f"{abs(x):.2f}"


def net_text(b, card) -> str:
    s = float(sum(it["shap"] for it in card))
    return NET_TEXT.format(n=len(card), impact=_impact(s), outcome=b.spec["outcome"])


def build_prompt(b, rid, cond, shots="A", qline=None, words=YN_WORDS, positive_alarm=True, final=L.ANSWER_INSTRUCTION,
                 task=None, ex_header=L.EXAMPLES_HEADER, label="Example", pt_header=L.PATIENT_HEADER):
    """The prompt structure of ler_main.Bundle.prompt with its fixed strings as parameters. With the registered strings
    it returns exactly ler_main's prompt (the tests check this). -> (text, card_dir)."""
    if cond not in SPLIT_CONDS:
        raise ValueError(cond)
    pos_word, neg_word = words
    ex = []
    for k, (vals, yy) in enumerate(b.shots[shots], 1):
        pos = (yy == 1) if positive_alarm else (yy == 0)
        ex.append(f"{label} {k}\n{b._block(vals)}\n{qline}\nAnswer: {pos_word if pos else neg_word}")
    parts = [f"Task: {b.spec['task'] if task is None else task}", ex_header, *ex,
             pt_header + "\n" + b._block(b.values[rid])]
    card = b.shown_card(rid, cond)
    if card is not None:
        parts.append(b.card_text(card))
    parts.append(f"{qline}\n{final}")
    cdir = None if card is None else int(sum(it["shap"] for it in card) > 0)
    return "\n\n".join(parts), cdir


def prompt_v4(b, rid, cond: str, framing: str, shot: str = "A"):
    """-> (text, card_dir) of one prompt of the fourth study. The registered framings with sets A, Asf, Abase and Achat
    are ler_main's registered prompts (Abase and Achat differ only in how they are rendered)."""
    if cond not in SPLIT_CONDS:
        raise ValueError(f"condition {cond} is not used in the fourth study")
    if shot not in SETS_V4:
        raise ValueError(shot)
    form = FRAMINGS_V4[framing]["form"]
    if shot in ("Acue", "Anet"):
        if framing != "alarm" or cond == "C0":
            raise ValueError("the cue and the net-number card are used with C2 and C4 in the alarm framing only")
        if shot == "Acue":
            return V.prompt_v3(b, rid, cond, "alarm", "cue")
        text, cdir = b.prompt(rid, cond, "alarm", "A")
        card = b.shown_card(rid, cond)
        old = b.card_text(card)
        if text.count(old) != 1:
            raise RuntimeError("the registered evidence block is not found exactly once")
        return text.replace(old, net_text(b, card)), cdir
    if form == "registered":
        return b.prompt(rid, cond, framing, "Asf" if shot == "Asf" else "A")
    if shot != "A":
        raise ValueError(f"framing {framing} is used with set A only")
    ds = b.name
    if form == "statement":
        st = L.CHOICE_OPTIONS[ds]["alarm" if pos_alarm(framing) else "safe"]
        return build_prompt(b, rid, cond, "A", STATEMENT + st, TF_WORDS, pos_alarm(framing), TF_INSTRUCTION)
    if form == "negated":
        return build_prompt(b, rid, cond, "A", QUESTION + NEG_Q[ds], YN_WORDS, False, L.ANSWER_INSTRUCTION)
    if form == "paraphrase":
        q = PARA_Q[ds]["alarm" if pos_alarm(framing) else "reversed"]
        return build_prompt(b, rid, cond, "A", QUESTION + q, YN_WORDS, pos_alarm(framing), PARA_INSTRUCTION,
                            task=PARA_TASK[ds], ex_header=PARA_EXAMPLES_HEADER, label=PARA_LABEL,
                            pt_header=PARA_PATIENT_HEADER)
    raise ValueError(framing)


def _swap_answers(text: str, mapping: dict) -> str:
    return re.sub(r"(?m)^Answer: (\w+)$", lambda m: "Answer: " + mapping.get(m.group(1), m.group(1)), text)


def check_prompt_v4(b, rid, cond, framing, shot, text, cdir):
    """Assertions for one prompt of the fourth study (run notebook and tests): the prompt is the registered prompt with
    exactly the change that Section 5 of the protocol describes, and its card direction is the registered one."""
    ds = b.name
    form = FRAMINGS_V4[framing]["form"]
    ref_fr = framing if form == "registered" else ("alarm" if pos_alarm(framing) or form == "negated" else "reversed")
    ref_set = "Asf" if shot == "Asf" else "A"
    ref, rdir = b.prompt(rid, cond, ref_fr, ref_set)
    assert cdir == rdir, "card direction differs from the registered prompt"
    ev = "Evidence from a machine-learning model (XGBoost) trained on other patients"
    assert text.count(ev) == (0 if cond == "C0" else 1), "evidence block missing or repeated"
    if shot in ("A", "Asf", "Abase", "Achat") and form == "registered":
        assert text == ref
    elif shot == "Acue":
        V.check_format_prompt(b, rid, cond, "alarm", "cue", text, cdir)
    elif shot == "Anet":
        card = b.shown_card(rid, cond)
        new = net_text(b, card)
        assert text == ref.replace(b.card_text(card), new) and text.count(new) == 1
        assert ("combined impact of +" in new) == (cdir == 1) and "increase" not in new and "decrease" not in new
    elif form == "statement":
        st = L.CHOICE_OPTIONS[ds]["alarm" if pos_alarm(framing) else "safe"]
        q = b.question(ref_fr)
        assert text.count(STATEMENT + st + "\n") == 5 and QUESTION not in text
        back = text.replace(STATEMENT + st + "\n" + TF_INSTRUCTION, QUESTION + q + "\n" + L.ANSWER_INSTRUCTION)
        back = _swap_answers(back.replace(STATEMENT + st + "\n", QUESTION + q + "\n"), {"True": "Yes", "False": "No"})
        assert back == ref, "True/False prompt differs from the registered prompt outside the statement lines"
    elif form == "negated":
        assert text.count(QUESTION + NEG_Q[ds] + "\n") == 5
        back = _swap_answers(text.replace(QUESTION + NEG_Q[ds] + "\n", QUESTION + b.question("alarm") + "\n"),
                             {"Yes": "No", "No": "Yes"})
        assert back == ref, "negated prompt differs from the registered alarm prompt outside the question lines"
    elif form == "paraphrase":
        q = PARA_Q[ds]["alarm" if pos_alarm(framing) else "reversed"]
        assert text.startswith(f"Task: {PARA_TASK[ds]}\n\n{PARA_EXAMPLES_HEADER}\n\n")
        assert text.count(PARA_PATIENT_HEADER + "\n") == 1 and L.PATIENT_HEADER not in text
        assert L.EXAMPLES_HEADER not in text and L.ANSWER_INSTRUCTION not in text
        assert re.findall(rf"(?m)^{PARA_LABEL} (\d)$", text) == ["1", "2", "3", "4"]
        assert not re.search(r"(?m)^Example \d$", text)
        assert text.count(QUESTION + q + "\n") == 5 and text.endswith(QUESTION + q + "\n" + PARA_INSTRUCTION)
        back = text.replace(f"Task: {PARA_TASK[ds]}", f"Task: {b.spec['task']}", 1)
        back = back.replace(PARA_EXAMPLES_HEADER, L.EXAMPLES_HEADER, 1).replace(PARA_PATIENT_HEADER, L.PATIENT_HEADER, 1)
        back = re.sub(rf"(?m)^{PARA_LABEL} (\d)$", r"Example \1", back)
        back = back.replace(QUESTION + q + "\n" + PARA_INSTRUCTION, QUESTION + b.question(ref_fr) + "\n"
                            + L.ANSWER_INSTRUCTION)
        back = back.replace(QUESTION + q + "\n", QUESTION + b.question(ref_fr) + "\n")
        assert back == ref, "paraphrased prompt differs from the registered prompt outside the paraphrased lines"
    else:
        raise AssertionError(f"no check for {framing} / {shot}")
    return True


# =============================================================================================================
# 2. Grids, models, decisions of the engineering check, protocol check
# =============================================================================================================
GRID_DEF = {
    "split": [(c, f, "A") for f in L.FRAMINGS for c in SPLIT_CONDS],
    "tf": [(c, f, "A") for f in ("tf_alarm", "tf_reversed") for c in SPLIT_CONDS],
    "neg": [(c, "negated", "A") for c in SPLIT_CONDS],
    "order": [(c, f, "Asf") for f in L.FRAMINGS for c in SPLIT_CONDS],
    "para": [(c, f, "A") for f in ("para_alarm", "para_reversed") for c in SPLIT_CONDS],
    "aggr": [(c, "alarm", s) for s in ("Acue", "Anet") for c in ("C2", "C4")],
    "plain": [(c, f, "Abase") for f in L.FRAMINGS for c in SPLIT_CONDS],
    "chat": [(c, f, "Achat") for f in L.FRAMINGS for c in SPLIT_CONDS],
}
INSTRUCT_GRIDS = ("split", "tf", "neg", "order", "para", "aggr")      # 31 prompts per patient
BASE_GRIDS = ("split", "chat")                                          # 12 prompts per patient
OPTIONAL_GRIDS = ("tf", "neg", "order", "para", "aggr")                 # may be dropped by rule (i) of Section 10


def grid_v4(grids) -> list:
    """(condition, framing, set) of a run, in a fixed order."""
    out = []
    for g in grids:
        out += GRID_DEF[g]
    if len(set(out)) != len(out):
        raise ValueError("duplicate grid entries")
    return out


# The models (Section 4). 'chat_model' = the instruction-tuned model whose chat template a base model is rendered with.
MODELS_V4 = {
    "meta-llama/Llama-3.1-8B-Instruct": dict(kind="instruct", family="llama", tag="llama-instruct",
                                             revision="0e9e39f249a16976918f6564b8830bc894c89659", developer="Meta"),
    "Qwen/Qwen2.5-7B-Instruct": dict(kind="instruct", family="qwen", tag="qwen-instruct",
                                     revision="a09a35458c702b33eeacc393d103063234e8bc28", developer="Alibaba (Qwen team)"),
    "allenai/Olmo-3-7B-Instruct": dict(kind="instruct", family="olmo3", tag="olmo3-instruct",
                                       revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
                                       developer="Allen Institute for AI"),
    "tiiuae/Falcon3-7B-Instruct": dict(kind="instruct", family="falcon3", tag="falcon3-7b",
                                       revision="1e57a0ecd176c7c139f289c60a74e57f887c3dfb",
                                       developer="Technology Innovation Institute"),
    "01-ai/Yi-1.5-9B-Chat-16K": dict(kind="instruct", family="yi", tag="yi15-9b",
                                     revision="2f8753bed95bd4fede3fd64f8305428620b4b212", developer="01.AI"),
    "meta-llama/Llama-3.1-8B": dict(kind="base", family="llama", tag="llama-base",
                                    revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", developer="Meta",
                                    chat_model="meta-llama/Llama-3.1-8B-Instruct"),
    "Qwen/Qwen2.5-7B": dict(kind="base", family="qwen", tag="qwen-base",
                            revision="d149729398750b98c0af14eb82c78cfe92750796", developer="Alibaba (Qwen team)",
                            chat_model="Qwen/Qwen2.5-7B-Instruct"),
    "allenai/Olmo-3-1025-7B": dict(kind="base", family="olmo3", tag="olmo3-base",
                                   revision="a81bae42db3975be1671e27b9c9a56da1a9f980f",
                                   developer="Allen Institute for AI", chat_model="allenai/Olmo-3-7B-Instruct"),
}
R10_FAMILY_ORDER = ("llama", "qwen", "olmo3")
INSTRUCT_TAGS = [m["tag"] for m in MODELS_V4.values() if m["kind"] == "instruct"]
TAG2ID = {m["tag"]: mid for mid, m in MODELS_V4.items()}


def family_tags(family: str) -> tuple:
    """(base tag, instruct tag) of an R10 family."""
    base = next(m["tag"] for m in MODELS_V4.values() if m["family"] == family and m["kind"] == "base")
    inst = next(m["tag"] for m in MODELS_V4.values() if m["family"] == family and m["kind"] == "instruct")
    return base, inst


# Decisions taken from the engineering check by the rules of Section 10, set before the tag. None = the check has not
# been recorded yet; a real run refuses to start then.
#   formats_not_run: tag -> grids of OPTIONAL_GRIDS that rule (i) removes
#   plain_run:       instruction-tuned tags whose I.plain cell is run (rule ii)
#   r10_families:    the R10 families used (rule iii), in the registered order
# Both checks (2026-11-02): results/v4/engcheck_v4.zip (first check, ler_v4.py v4-1.0) and results/v4/engcheck_v4_run2.zip
# (second check of Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and Llama-3.1-8B, ler_v4.py v4-1.1); a model's result is
# taken from the second check where it was checked again. decisions_from_check() of the merged results gives exactly
# this (the tests check it).
ENGCHECK_DECISIONS = {"formats_not_run": {}, "plain_run": ["llama-instruct", "qwen-instruct"],
                      "r10_families": ["llama", "qwen"]}
ENGCHECK_ZIP_SHA256 = ("805ebbac4f896f50f2e00ddccf4c14585373da7d63e3dcd3cb7a04150f610e7f",    # engcheck_v4.zip
                       "c1ed543c5c39ca5b4cb1e5d3e61014c21a108f39392883864a6d680c099efbc1")    # engcheck_v4_run2.zip
ENGCHECK_CODE_SHA256 = ("5ef560a98d3045139bf6d4d3d98ac47615d784b5ff4e9ec7341367890015f41c",   # ler_v4.py v4-1.0
                        "c2ed9d95bde5906fc5e0fc3bd94e205dbb9014581394c3b2020694a665268cdd")   # ler_v4.py v4-1.1


def model_grids(model_id: str, decisions=None) -> tuple:
    """The grids one model is run with, after the decisions of the engineering check."""
    d = ENGCHECK_DECISIONS if decisions is None else decisions
    m = MODELS_V4[model_id]
    if d is None:
        raise RuntimeError("the decisions of the engineering check are not recorded in ler_v4.py (ENGCHECK_DECISIONS)")
    if m["kind"] == "base":
        if m["family"] not in d["r10_families"]:
            raise RuntimeError(f"{model_id}: its family is not used for R10 (engineering check)")
        return BASE_GRIDS
    drop = set(d.get("formats_not_run", {}).get(m["tag"], ()))
    grids = tuple(g for g in INSTRUCT_GRIDS if g not in drop)
    if m["tag"] in d.get("plain_run", ()) and m["family"] in d["r10_families"]:
        grids += ("plain",)
    return grids


def protocol_strings_v4() -> list:
    """Every fixed string the fourth study adds, the two code hashes, and every model id and revision, for the check
    against PREREGISTRATION_V4.md."""
    out = [TF_INSTRUCTION, PARA_EXAMPLES_HEADER, PARA_PATIENT_HEADER, PARA_INSTRUCTION, NET_TEXT, V.CUE_TEXT,
           MAIN_SHA256, V3_SHA256, V4_VERSION]
    out += list(NEG_Q.values()) + list(PARA_TASK.values())
    out += [q for d in PARA_Q.values() for q in d.values()]
    out += [s for d in L.CHOICE_OPTIONS.values() for s in d.values()]
    for mid, m in MODELS_V4.items():
        out += [mid, m["revision"]]
    out += list(ENGCHECK_ZIP_SHA256 or ()) + list(ENGCHECK_CODE_SHA256 or ())
    return out


def check_against_protocol_v4(text: str) -> list:
    """Strings of the fourth study missing from PREREGISTRATION_V4.md, and missing section markers (should be empty)."""
    flat = re.sub(r"\s+", " ", text.replace("`", ""))
    miss = [s for s in protocol_strings_v4() if re.sub(r"\s+", " ", s) not in flat]
    return miss + [m for m in V4_MARKERS if m not in text]


# =============================================================================================================
# 3. Reading, formats, generated answers
# =============================================================================================================
def render_v4(text: str, model_kind: str, shot: str, tok, chat_tok=None) -> str:
    """plain (Abase, and every prompt of a base model except Achat): text + newline + 'Answer:' after the BOS token;
    chat (Achat, and every prompt of an instruction-tuned model except Abase): the chat template of the
    instruction-tuned model + 'Answer:'. For a base model in the chat format, chat_tok is the instruction-tuned
    model's tokenizer."""
    if shot == "Abase":
        return L.render(text, "base", tok)
    if shot == "Achat":
        if chat_tok is None:
            raise ValueError("the chat format of a base model needs the instruction-tuned model's tokenizer")
        return L.render(text, "instruct", chat_tok)
    return L.render(text, model_kind, tok)


def read_v4(be, rendered, framing):
    """Readings of prompts of one framing (or of framings with the same answer words) -> dicts {lp_yes, lp_no,
    n_imputed}; lp_yes / lp_no hold the first and the second answer word ('Yes'/'No' or 'True'/'False')."""
    return [{"lp_yes": r["lp_pos"], "lp_no": r["lp_neg"], "n_imputed": r["n_imputed"]}
            for r in be.answer_logprobs(rendered, answer_words(framing))]


def finish_record_v4(rec: dict, r: dict, framing: str) -> dict:
    """ler_main.finish_record for any framing of the fourth study: the first answer word stands for the alarming
    outcome when pos_alarm(framing), exactly as 'Yes' does in the registered alarm framing."""
    rec = L.finish_record(rec, r, "alarm" if pos_alarm(framing) else "reversed")
    rec["answer_words"] = "/".join(answer_words(framing))
    return rec


GREEDY_SEED = 2031
GREEDY_PER_CLASS = 50
GREEDY_MAX_TOKENS = {"prefill": 8, "free": 512}     # free: 64 in the first and 256 in the second check (Section 10)
GREEDY_GRID = GRID_DEF["split"]
_STRIP = re.compile(r"[*_`#>]")


def greedy_patients(b) -> list:
    """50 test patients per class, drawn with seed 2031 from the sorted patient ids of the bundle's current sample."""
    ids = sorted(b.ids)
    rs = np.random.RandomState(GREEDY_SEED)
    out = []
    for cls in (1, 0):
        pool = [i for i in ids if b.y[i] == cls]
        k = min(GREEDY_PER_CLASS, len(pool))
        out += [pool[j] for j in sorted(rs.choice(len(pool), k, replace=False))]
    return sorted(out)


def render_free(text: str, tok) -> str:
    """The prompt through the chat template without the 'Answer:' prefill (free mode)."""
    r = L.render(text, "instruct", tok)
    if not r.endswith(L.PREFILL):
        raise RuntimeError("rendered prompt does not end with the prefill")
    return r[:-len(L.PREFILL)]


def parse_generated(text, words=YN_WORDS):
    """Section 5.7: after removing * _ ` # >, the first word if it is one of the answer words (any case); otherwise the
    first answer word after 'answer' (with 'is', ':' or both in between, or nothing); otherwise None (unparsed)."""
    t = _STRIP.sub(" ", text or "")
    alt = "|".join(re.escape(w) for w in words)
    canon = {w.lower(): w for w in words}
    m = re.match(rf"\W*({alt})\b", t, re.I)
    if m:
        return canon[m.group(1).lower()]
    m = re.search(rf"\banswer(?:\s+is)?\s*:?\s*({alt})\b", t, re.I)
    if m:
        return canon[m.group(1).lower()]
    return None


def generate(be, prompts, max_tokens):
    """Greedy continuations of rendered prompts."""
    if hasattr(be, "generate_from_text"):
        return be.generate_from_text(prompts, max_tokens)
    ids = [be.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    return be.generate_text(ids, max_tokens)


def download_tokenizer(model_id, cache_dir, revision=None, token=None):
    """The tokenizer files (and the small JSON files) of a model, without its weights."""
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi(token=token).model_info(model_id, revision=revision).sha
    path = snapshot_download(model_id, revision=sha, cache_dir=cache_dir, token=token,
                             allow_patterns=["*.json", "*.model", "tokenizer*", "*.txt", "*.jinja", "*.tiktoken"],
                             ignore_patterns=["*.safetensors", "*.bin", "*.pt"])
    return path, sha


def same_token_ids(rendered, tok_a, tok_b) -> dict:
    """Whether two tokenizers give exactly the same token ids for every rendered prompt (base model in chat format)."""
    bad = 0
    for r in rendered:
        if tok_a.encode(r, add_special_tokens=False) != tok_b.encode(r, add_special_tokens=False):
            bad += 1
    return {"n": len(rendered), "n_differ": bad, "identical": bad == 0}


TF_SANITY = {
    "sky_true": (STATEMENT + "The sky is usually blue on a clear day.\n" + TF_INSTRUCTION, TF_WORDS, True),
    "fire_false": (STATEMENT + "Fire is usually cold to the touch.\n" + TF_INSTRUCTION, TF_WORDS, False),
}
SANITY_V4 = {k: v for k, v in L.SANITY_PROMPTS.items() if v[1] == YN_WORDS} | TF_SANITY


# =============================================================================================================
# 4. Smoke test and engineering check
# =============================================================================================================
def smoke_v4(model_path, kind="instruct", prompts_path=None):
    """Run in a subprocess before vLLM is loaded in the notebook kernel: loads the model with its window, runs the
    exact-reading check of Deviation D2 for Yes/No and for True/False prompts (recorded; the runs read the top 1,000,
    Deviation D5) and the sky prompt."""
    import torch
    tp = 2 if torch.cuda.device_count() >= 2 else 1
    t0 = time.time()
    win = V.model_window(model_path)
    be = L.VLLMBackend(model_path, tensor_parallel_size=tp, max_model_len=win, read="token_ids")
    prompts = [L.render("Is the sky usually blue on a clear day?\n" + L.ANSWER_INSTRUCTION, kind, be.tokenizer)]
    tf = []
    if prompts_path:
        data = json.load(open(prompts_path))
        prompts += data.get("yesno", [])
        tf = data.get("tf", [])
    check = be.check_exact(prompts, YN_WORDS)
    check_tf = be.check_exact(tf, TF_WORDS) if tf else None
    read = "token_ids" if check.get("ok") and (check_tf is None or check_tf.get("ok")) else "topk"
    be.read = read
    r = be.answer_logprobs(prompts[:1], YN_WORDS)
    res = {"ok": True, "load_seconds": time.time() - t0, "tensor_parallel_size": tp, "max_model_len": win, "read": read,
           "exact_check": check, "exact_check_tf": check_tf, "sky_prompt": r[0], "info": be.info()}
    print("SMOKE_V4_RESULT " + json.dumps(res, default=str))
    return res


def engineering_check_v4(model_id, kind, prompts_path, cache_dir, revision=None, chat_model=None, chat_revision=None):
    """BEFORE registration (PREREGISTRATION_V4.md, Section 10): can this model be run as registered, and in which
    formats? Engineering only: download, the vLLM float16 load with the model window, the chat template and its
    default system text, single-token answer words, the exact-reading check (D2) for Yes/No and True/False, finiteness,
    the median answer-word mass per framing and format, the imputed share under the top-1,000 reading (D5), prompt
    lengths, the time per prompt, the obvious questions, the generated answers' time and parse share (instruction-tuned
    models) and, for a base model, the token identity of the chat format. No accuracy, no answer direction, no
    agreement and no asymmetry is computed; the prompts are dev-sample prompts. Prints one ENGCHECK_V4_RESULT line."""
    res = {"model_id": model_id, "kind": kind, "ok": False, "problems": [],
           "code": {"ler_v4": V4_VERSION, "ler_v3": V.V3_VERSION, "ler_main": L.CODE_VERSION}}
    t0 = time.time()
    try:
        path, sha = L.download_model(model_id, cache_dir, revision, os.environ.get("HF_TOKEN"))
        res.update(revision=sha, download_seconds=round(time.time() - t0, 1))
        cfg = json.load(open(os.path.join(path, "config.json")))
        tcfg = cfg.get("text_config") or {}
        res["config"] = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                         "dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tcfg.get("torch_dtype"),
                         "quantization": (cfg.get("quantization_config") or {}).get("quant_method")}
        win = V.model_window(path)
        res["max_model_len"] = win
        chat_tok = None
        if kind == "base" and chat_model:
            from transformers import AutoTokenizer
            cpath, csha = download_tokenizer(chat_model, cache_dir, chat_revision, os.environ.get("HF_TOKEN"))
            chat_tok = AutoTokenizer.from_pretrained(cpath)
            res["chat_model"], res["chat_revision"] = chat_model, csha
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        t1 = time.time()
        be = L.VLLMBackend(path, tensor_parallel_size=tp, max_model_len=win, read="token_ids")
        res.update(load_seconds=round(time.time() - t1, 1), tensor_parallel_size=tp)
        tok = be.tokenizer
        res["template_system_text"] = L.template_system_text(tok) if kind == "instruct" else (
            L.template_system_text(chat_tok) if chat_tok is not None else None)
        res["single_token_words"] = {}
        for pair in (YN_WORDS, TF_WORDS):
            try:
                be.ensure_words(pair)
                res["single_token_words"]["/".join(pair)] = True
            except Exception as e:
                res["single_token_words"]["/".join(pair)] = False
                res["problems"].append(f"no single-token variant for {pair}: {e!r}"[:300])
        res["variants"] = {w: [[i, be.strings[i]] for i in ids] for w, ids in be.variants.items()}
        items = json.load(open(prompts_path))
        groups = {}
        for it in items:
            if it["grid"] not in (BASE_GRIDS if kind == "base" else INSTRUCT_GRIDS + ("plain",)):
                continue
            groups.setdefault((it["set"], it["grid"], it["framing"], it["shot"]), []).append(it)
        rendered = {k: [render_v4(it["text"], kind, k[3], tok, chat_tok) for it in v] for k, v in groups.items()}
        ntok = [len(tok.encode(r, add_special_tokens=False)) for v in rendered.values() for r in v]
        res["prompt_tokens"] = {"median": int(np.median(ntok)), "max": int(max(ntok))}
        if max(ntok) >= win - 1:
            res["problems"].append(f"a prompt is longer than the model window ({max(ntok)} >= {win - 1})")
        if chat_tok is not None:
            res["chat_token_identity"] = same_token_ids([r for k, v in rendered.items() if k[3] == "Achat" for r in v],
                                                        chat_tok, tok)
        for k, rs in rendered.items():
            if k[0] == "check" and k[2] in ("alarm", "tf_alarm") and k[3] in ("A", "Achat"):
                res[f"rendered_tail_{k[1]}_{k[2]}"] = rs[-1][-600:]
        ex = {}
        for k, rs in rendered.items():
            if k[0] == "check" and k[3] in ("A", "Achat") and k[2] in ("alarm", "tf_alarm"):
                ex.setdefault(answer_words(k[2]), []).extend(rs[:12])
        for pair, rs in ex.items():
            if res["single_token_words"].get("/".join(pair)):
                res["exact_check_" + "/".join(pair)] = be.check_exact(rs, pair)
        be.read = "topk"
        res["readings"] = {}
        for (sset, grid, fr, shot), rs in sorted(rendered.items()):
            pair = answer_words(fr)
            if not res["single_token_words"].get("/".join(pair)) or sset == "greedy":
                continue
            t1 = time.time()
            rows = be.answer_logprobs(rs, pair)
            el = time.time() - t1
            res["readings"][f"{sset}|{grid}|{fr}|{shot}"] = dict(L._mass(rows), seconds=round(el, 1),
                                                               seconds_per_prompt=round(el / max(1, len(rs)), 4))
        san = {}
        for name, (text, pair, first) in SANITY_V4.items():
            if not res["single_token_words"].get("/".join(pair)):
                continue
            r = be.answer_logprobs([L.render(text, kind, tok)], pair)[0]
            san[name] = bool((r["lp_pos"] > r["lp_neg"]) == first)
        res["sanity_right_word_more_likely"] = san
        if kind == "instruct":
            gi = [it for it in items if it["set"] == "greedy"]
            for mode in ("prefill", "free"):
                ps = [L.render(it["text"], "instruct", tok) if mode == "prefill" else render_free(it["text"], tok)
                      for it in gi]
                t1 = time.time()
                outs = generate(be, ps, GREEDY_MAX_TOKENS[mode])
                el = time.time() - t1
                parsed = [parse_generated(o) for o in outs]
                res[f"generated_{mode}"] = {"n": len(outs), "parsed_share": float(np.mean([p is not None for p in parsed]))
                                            if outs else float("nan"), "seconds_per_prompt": round(el / max(1, len(outs)), 4),
                                            "examples": [o[:160] for o in outs[:3]]}
        # decisions by the rules of Section 10 (technical quantities only)
        fmt_ok = {}
        for g in (BASE_GRIDS if kind == "base" else INSTRUCT_GRIDS + ("plain",)):
            ks = [k for k in res["readings"] if k.startswith(f"check|{g}|")]
            fin = all(res["readings"][k]["finite_share"] == 1 for k in ks)
            mass = all(res["readings"][k]["mass_median"] >= L.VALIDITY_MIN_MASS for k in ks)
            fmt_ok[g] = bool(ks) and fin and mass
        res["formats_ok"] = fmt_ok
        if kind == "base":
            res["r10_eligible"] = bool(fmt_ok.get("split") and fmt_ok.get("chat")
                                       and (res.get("chat_token_identity") or {}).get("identical"))
        if not fmt_ok.get("split"):
            res["problems"].append("the registered prompts do not reach a median mass of 0.5 with finite readings")
        if res["config"].get("quantization"):
            res["problems"].append(f"quantized weights ({res['config']['quantization']}); the study runs 16-bit weights")
        res["ok"] = not res["problems"]
    except Exception as e:
        import traceback
        res["problems"].append(f"error: {e!r}"[:500])
        res["traceback"] = traceback.format_exc()[-3000:]
    res["total_seconds"] = round(time.time() - t0, 1)
    print("ENGCHECK_V4_RESULT " + json.dumps(res, default=str))
    return res


def decisions_from_check(results: dict) -> dict:
    """The rules of Section 10 applied to the engineering-check results (model id -> result) -> ENGCHECK_DECISIONS."""
    out = {"formats_not_run": {}, "plain_run": [], "r10_families": []}
    for mid, m in MODELS_V4.items():
        r = results.get(mid) or {}
        ok = r.get("formats_ok") or {}
        if m["kind"] == "instruct":
            drop = [g for g in OPTIONAL_GRIDS if not ok.get(g)]
            if drop:
                out["formats_not_run"][m["tag"]] = drop
    for fam in R10_FAMILY_ORDER:
        base, inst = family_tags(fam)
        rb = results.get(TAG2ID[base]) or {}
        ri = results.get(TAG2ID[inst]) or {}
        if rb.get("r10_eligible") and (ri.get("formats_ok") or {}).get("split") and len(out["r10_families"]) < 2:
            out["r10_families"].append(fam)
    for fam in out["r10_families"]:          # rule (ii): the I.plain cell of an R10 family
        inst = family_tags(fam)[1]
        if ((results.get(TAG2ID[inst]) or {}).get("formats_ok") or {}).get("plain"):
            out["plain_run"].append(inst)
    return out


def access_problems(model_ids, token=None, cache_dir=None) -> dict:
    """Gated models whose pinned revision cannot be downloaded with this token (model id -> the error, first line).
    Run at the start of a notebook so that a refused download stops it before GPU time is spent."""
    from huggingface_hub import hf_hub_download
    out = {}
    for mid in model_ids:
        try:
            hf_hub_download(mid, "config.json", revision=MODELS_V4[mid]["revision"], token=token, cache_dir=cache_dir)
        except Exception as e:
            out[mid] = f"{type(e).__name__}: " + (str(e).strip().splitlines() or [""])[0][:300]
    return out


ACCESS_HELP = ("Check three things: (1) on huggingface.co, logged in to the account whose token you use, the page of each "
               "model says that you have been granted access; (2) in Kaggle, Add-ons -> Secrets, the secret HF_TOKEN is "
               "attached to this notebook and holds that account's token; (3) a fine-grained token has the permission "
               "'Read access to contents of all public gated repos you can access' (a 'read' token always has it).")


def merge_checks(*results) -> dict:
    """Engineering-check results of several checks (model id -> result), later checks replacing earlier ones."""
    out = {}
    for r in results:
        out.update(r)
    return out


def parse_result(output: str, key: str):
    return V.parse_result(output, key)


# =============================================================================================================
# 5. Fake model for the CPU tests (never used on Kaggle)
# =============================================================================================================
class _MockTokV4(L._MockTok):
    def __init__(self):
        super().__init__()
        self.vocab.update({"True": 9, "ĠTrue": 10, "False": 11, "ĠFalse": 12})


class MockV4(L.MockBackend):
    """Deterministic fake model that reads every prompt of the fourth study. The evidence (registered card, cue,
    net-number card) moves the log-odds of the alarming outcome with `alarm_gain` / `safe_gain`; whenever a card is
    shown, the answer is pulled by `pull[words]` toward the second answer word ('No' or 'False'), which gives
    W = -2 * pull. `mode` says what the pull follows:
      'polarity': the second answer word in every format (Yes/No and True/False, with their own sizes);
      'named':    denying the outcome the question names (so the negated question pulls toward 'Yes').
    `render_factor` scales the pull per format ('chat' or 'plain'); `order_pull` adds a shift toward the outcome of the
    last solved example whenever a card is shown (an outcome part that follows the example order); `para_factor`
    scales the pull in the paraphrased prompt; `cue_impact` and `net_gain` set how much the cue and the net-number card
    move the model; `plain_mass` sets the answer-word mass of plain prompts."""
    name = "mock"
    _CUE = V.MockV3._CUE
    _NET = re.compile(r"have a combined impact of ([+-]\d+\.\d+) on the risk of")
    _FULL = re.compile(r"\(impact ([+-]\d+\.\d+)\)")

    def __init__(self, pull=None, mode="polarity", render_factor=None, order_pull=0.0, para_factor=1.0,
                 cue_impact=1.0, net_gain=0.1, plain_mass=None, gen_flip=0.0, **kw):
        super().__init__([s["q_reversed"] for s in L.SPEC.values()], **kw)
        self.tokenizer = _MockTokV4()
        self.variants, self.strings = L.find_word_variants(self.tokenizer)
        self.pull = {"Yes/No": 0.0, "True/False": 0.0}
        self.pull.update(pull or {})
        self.mode, self.order_pull, self.para_factor = mode, float(order_pull), float(para_factor)
        self.render_factor = {"chat": 1.0, "plain": 1.0}
        self.render_factor.update(render_factor or {})
        self.cue_impact, self.net_gain, self.plain_mass, self.gen_flip = float(cue_impact), float(net_gain), plain_mass, gen_flip
        self.qinfo = {}
        for ds, s in L.SPEC.items():
            self.qinfo[s["q_alarm"]] = ("registered", True, True)
            self.qinfo[s["q_reversed"]] = ("registered", False, False)
            self.qinfo[NEG_Q[ds]] = ("negated", False, True)
            self.qinfo[PARA_Q[ds]["alarm"]] = ("paraphrase", True, True)
            self.qinfo[PARA_Q[ds]["reversed"]] = ("paraphrase", False, False)
        self.statements = {}
        for o in L.CHOICE_OPTIONS.values():
            self.statements[o["alarm"]] = True
            self.statements[o["safe"]] = False

    def _evidence(self, part):
        m = self._CUE.search(part)
        if m:
            return (1.0 if m.group(1) == "increase" else -1.0) * self.cue_impact
        m = self._NET.search(part)
        if m:
            return float(m.group(1)) * self.net_gain
        x = self._FULL.findall(part)
        return sum(float(v) for v in x) if x else None

    def logodds(self, p):
        """-> (log-odds of the first answer word, answer words, mass) for one rendered prompt."""
        for h in (L.PATIENT_HEADER, PARA_PATIENT_HEADER):
            if h in p:
                head, part = p.rsplit(h, 1)
                break
        key = part.split("\n\n")[0]
        base = 3.0 * (L._u(key, self.seed) - 0.5) + self.lean
        if key in self.truth:
            base += self.signal * (2 * self.truth[key] - 1)
        s = self._evidence(part)
        shown = s is not None
        s = s or 0.0
        z = self.scale * (base + (self.alarm_gain * s if s > 0 else self.safe_gain * s))   # log-odds of the alarm outcome
        if TF_INSTRUCTION in p:
            st = re.findall(re.escape(STATEMENT) + r"(.*)\n" + re.escape(TF_INSTRUCTION), p)[-1]
            form, pa, named = "statement", self.statements[st], self.statements[st]
            words = TF_WORDS
        else:
            q = re.findall(r"Question: (.*)\n(?:" + re.escape(L.ANSWER_INSTRUCTION) + "|" + re.escape(PARA_INSTRUCTION) + ")",
                           p)[-1]
            form, pa, named = self.qinfo[q]
            words = YN_WORDS
        render = "chat" if "<|im_start|>" in p else "plain"
        if shown and self.order_pull:
            ans = re.findall(r"\nAnswer: (\w+)\n", head + "\n")
            last_first_word = ans[-1] == words[0]
            last_alarm = last_first_word == pa
            z += self.order_pull * (1 if last_alarm else -1)
        zp = z if pa else -z                                        # log-odds of the first answer word
        if shown:
            pull = self.pull["/".join(words)] * self.render_factor[render] * (self.para_factor if form == "paraphrase" else 1)
            if self.mode == "polarity":
                zp -= pull
            elif self.mode == "named":            # toward denying the named outcome
                affirms = pa == named                 # the first word affirms the named outcome
                zp += -pull if affirms else pull
            else:
                raise ValueError(self.mode)
        mass = 0.80 + 0.19 * L._u(key, "m")
        if render == "plain" and self.plain_mass is not None:
            mass = float(self.plain_mass)
        return zp, words, mass

    def word_logprobs(self, prompts, words=YN_WORDS):
        wanted = self.wanted_ids(words)
        ids_of = {YN_WORDS: (2, 1, 4, 3), TF_WORDS: (10, 9, 12, 11)}
        res = []
        for p in prompts:
            zp, w, mass = self.logodds(p)
            if tuple(words) != w:
                raise RuntimeError(f"mock: prompt asks for {w}, reading asks for {words}")
            ids = ids_of[w]
            lmy = math.log(mass) - math.log1p(math.exp(-zp)) if zp > -30 else math.log(mass) + zp
            lmn = math.log(mass) - math.log1p(math.exp(zp)) if zp < 30 else math.log(mass) - zp
            full = {ids[0]: lmy + math.log(0.9), ids[1]: lmy + math.log(0.1), ids[2]: lmn + math.log(0.9),
                    ids[3]: lmn + math.log(0.1)}
            other = math.log(1 - mass) - math.log(8) if mass < 1 else -50.0
            res.append(({t: full.get(t, other) for t in wanted}, 0))
        return res

    def generate_from_text(self, prompts, max_tokens):
        out = []
        for p in prompts:
            zp, w, _ = self.logodds(p)
            first = zp > 0
            if abs(zp) < self.gen_flip:
                first = not first
            word = w[0] if first else w[1]
            if p.endswith(L.PREFILL):
                out.append(" " + word)
            else:
                h = L._u(p[-200:], "g")
                out.append(f"**{word}**" if h < 0.4 else (f"Based on the data, the answer is {word}." if h < 0.8
                                                          else "I cannot tell from this information."))
        return out


# =============================================================================================================
# 6. Analysis (PREREGISTRATION_V4.md, Sections 7 and 8)
# =============================================================================================================
PAIRS = {"registered": (("alarm", "A"), ("reversed", "A")),
         "tf": (("tf_alarm", "A"), ("tf_reversed", "A")),
         "neg": (("alarm", "A"), ("negated", "A")),
         "order": (("alarm", "Asf"), ("reversed", "Asf")),
         "para": (("para_alarm", "A"), ("para_reversed", "A")),
         "plain": (("alarm", "Abase"), ("reversed", "Abase")),
         "chat": (("alarm", "Achat"), ("reversed", "Achat"))}
TESTED_PAIRS = ("tf", "neg", "order", "para")
# eligible cells (Section 7): R6 / R7 eligible cells of the third study without Phi-4-mini, with their signs
ELIGIBLE = {
    "W": {"qwen-instruct|physionet2012": 1, "qwen-instruct|physionet2019": -1,
          "llama-instruct|physionet2012": -1, "llama-instruct|diabetes130": -1,
          "olmo3-instruct|physionet2012": -1, "olmo3-instruct|physionet2019": -1,
          "falcon3-7b|physionet2012": 1, "falcon3-7b|physionet2019": 1, "falcon3-7b|diabetes130": 1,
          "yi15-9b|physionet2012": 1, "yi15-9b|physionet2019": 1},
    "O": {"qwen-instruct|physionet2012": -1, "llama-instruct|physionet2012": -1, "llama-instruct|physionet2019": -1,
          "olmo3-instruct|physionet2012": 1, "falcon3-7b|physionet2019": 1, "falcon3-7b|diabetes130": 1,
          "yi15-9b|physionet2019": -1},
}
R10_ELIGIBLE = {"llama": {"physionet2012": -1, "diabetes130": -1},
                "qwen": {"physionet2012": 1, "physionet2019": -1},
                "olmo3": {"physionet2012": -1, "physionet2019": -1}}


def slice_v4(df: pd.DataFrame, framing: str, shot: str = "A") -> pd.DataFrame:
    """C0, C2 and C4 records of one framing and set. The cue and the net-number card (C2, C4) use the C0 records of
    the registered alarm framing."""
    if shot in ("Acue", "Anet"):
        a = df[(df["framing"] == "alarm") & (df["shot_set"] == "A") & (df["cond"] == "C0")]
        v = df[(df["framing"] == "alarm") & (df["shot_set"] == shot) & df["cond"].isin(("C2", "C4"))]
        return pd.concat([a, v], ignore_index=True)
    return df[(df["framing"] == framing) & (df["shot_set"] == shot) & df["cond"].isin(SPLIT_CONDS)]


def has_slice(df, framing, shot) -> bool:
    return bool(((df["framing"] == framing) & (df["shot_set"] == shot)).any())


def _mass_median(r):
    return float(r["yn_mass"].astype(float).fillna(0.0).median()) if len(r) else float("nan")


def _la_ls(sl: pd.DataFrame) -> pd.DataFrame:
    """Per patient with valid C2 and C4 readings: la, ls = log-odds of the alarming outcome with the alarm-pointing and
    with the safe-pointing card."""
    d = sl[(sl["valid"] == 1) & sl["cond"].isin(("C2", "C4"))]
    if not len(d):
        return pd.DataFrame(columns=["la", "ls"])
    w = d.pivot_table(index="id", columns="cond", values=["lo_alarm", "card_dir"], aggfunc="first")
    need = [("lo_alarm", "C2"), ("lo_alarm", "C4"), ("card_dir", "C2")]
    if not all(k in w.columns for k in need):
        return pd.DataFrame(columns=["la", "ls"])
    w = w.dropna(subset=need)
    cd = w[("card_dir", "C2")].astype(int).to_numpy()
    l2, l4 = w[("lo_alarm", "C2")].to_numpy(float), w[("lo_alarm", "C4")].to_numpy(float)
    return pd.DataFrame({"la": np.where(cd == 1, l2, l4), "ls": np.where(cd == 1, l4, l2)}, index=w.index)


def _c0(sl):
    d = sl[(sl["valid"] == 1) & (sl["cond"] == "C0")]
    return d.set_index("id")["lo_alarm"].astype(float)


def split_validity(df: pd.DataFrame, shot: str = "A") -> dict:
    """The registered cell rule for one format: median mass >= 0.5 in the alarm and in the reversed framing (C0, C2,
    C4), and C0 accuracy >= 0.55 in the alarm framing."""
    al, rv = slice_v4(df, "alarm", shot), slice_v4(df, "reversed", shot)
    c0 = al[(al["cond"] == "C0") & (al["valid"] == 1)]
    acc = float(c0["correct"].mean()) if len(c0) else float("nan")
    ma, mr = _mass_median(al), _mass_median(rv)
    return {"alarm_median_mass": ma, "reversed_median_mass": mr, "c0_accuracy": acc, "n_c0_valid": int(len(c0)),
            "passes": bool(ma >= L.VALIDITY_MIN_MASS and mr >= L.VALIDITY_MIN_MASS and acc >= L.VALIDITY_MIN_C0_ACC)}


def framing_validity(df: pd.DataFrame, framing: str, shot: str) -> dict:
    """Section 8, a new framing in a cell: (i) median mass >= 0.5 over its records; (ii) mean movement > 0; (iii)
    positive Spearman correlation of its C0 log-odds of the alarming outcome with those of the registered alarm
    question. For the cue and the net-number card only (i), over their C2 and C4 records."""
    from scipy.stats import spearmanr
    own = df[(df["framing"] == framing) & (df["shot_set"] == shot)]
    m = _mass_median(own)
    out = {"median_mass": m, "n_records": int(len(own))}
    if shot in ("Acue", "Anet"):
        out["passes"] = bool(len(own) and m >= L.VALIDITY_MIN_MASS)
        return out
    sl = slice_v4(df, framing, shot)
    ll = _la_ls(sl)
    mv = float((ll["la"] - ll["ls"]).mean()) if len(ll) else float("nan")
    a, b = _c0(sl), _c0(slice_v4(df, "alarm", "A"))
    ids = a.index.intersection(b.index)
    rho = float(spearmanr(a.loc[ids], b.loc[ids]).correlation) if len(ids) >= 10 else float("nan")
    out.update(movement=mv, c0_spearman_with_alarm=rho,
               passes=bool(len(own) and m >= L.VALIDITY_MIN_MASS and mv > 0 and rho > 0))
    return out


def validity_v4(df: pd.DataFrame) -> dict:
    split = split_validity(df, "A")
    fr = {}
    for framing in FRAMINGS_V4:
        for shot in SETS_V4:
            if (framing, shot) in (("alarm", "A"), ("reversed", "A")) or not has_slice(df, framing, shot):
                continue
            v = framing_validity(df, framing, shot)
            v["passes"] = bool(v["passes"] and split["passes"])
            fr[f"{framing}/{shot}"] = v
    ok = {"alarm/A": split["passes"], "reversed/A": split["passes"]}
    ok.update({k: v["passes"] for k, v in fr.items()})
    pairs = {}
    for p, (s1, s2) in PAIRS.items():
        k1, k2 = f"{s1[0]}/{s1[1]}", f"{s2[0]}/{s2[1]}"
        if k1 in ok and k2 in ok:
            pairs[p] = {"passes": bool(ok[k1] and ok[k2])}
    if "alarm/Acue" in fr and "alarm/Anet" in fr:
        pairs["aggregate"] = {"passes": bool(fr["alarm/Acue"]["passes"] and fr["alarm/Anet"]["passes"])}
    for shot in ("Abase", "Achat"):
        if has_slice(df, "alarm", shot):
            fr[f"split/{shot}"] = split_validity(df, shot)
    return {"split": split, "framings": fr, "pairs": pairs}


def pair_comparison(sources: dict, pairs: dict, contrasts: dict = None, n_boot: int = N_BOOT) -> dict | None:
    """sources: slice name -> records of one framing and format (C0, C2, C4); pairs: name -> (first slice, second
    slice); contrasts: name -> (pair x, pair y, part) for part(x) - part(y). On the patients with valid C0, C2 and C4
    readings in every source (records of two runs pair by patient id), one paired bootstrap (seed 2026): a, a' and the
    movement of every slice, O = (s1 + s2) / 2 and W = (s1 - s2) / 2 of every pair, and the contrasts."""
    wides = {}
    for k, v in sources.items():
        w = L._wide(v)
        if w is None:
            return None
        wides[k] = w
    ids = None
    for w in wides.values():
        ids = w.index if ids is None else ids.intersection(w.index)
    ids = sorted(ids)
    if len(ids) < 10:
        return None
    xs = {k: L.asym_arrays(w.loc[ids]) for k, w in wides.items()}
    fns = {}
    for k, x in xs.items():
        fns[f"a|{k}"] = lambda W, x=x: V.stat_a(x, W)
        fns[f"ap|{k}"] = lambda W, x=x: V.stat_ap(x, W)
        fns[f"m|{k}"] = lambda W, x=x: V.stat_move(x, W)
    est, dr = V.bootstrap(len(ids), fns, n_boot)
    out = {"n": len(ids), "slices": {}, "pairs": {}, "contrasts": {}}
    for k in xs:
        out["slices"][k] = {"s": V._two(est, dr, f"a|{k}", f"ap|{k}"), "movement": V._summ(est[f"m|{k}"], dr[f"m|{k}"])}
    for p, (s1, s2) in pairs.items():
        for st in ("a", "ap"):
            for part, sg in (("O", 1), ("W", -1)):
                est[f"{part}_{st}|{p}"] = (est[f"{st}|{s1}"] + sg * est[f"{st}|{s2}"]) / 2
                dr[f"{part}_{st}|{p}"] = (dr[f"{st}|{s1}"] + sg * dr[f"{st}|{s2}"]) / 2
        est[f"m|{p}"] = (est[f"m|{s1}"] + est[f"m|{s2}"]) / 2
        dr[f"m|{p}"] = (dr[f"m|{s1}"] + dr[f"m|{s2}"]) / 2
        out["pairs"][p] = {"O": V._two(est, dr, f"O_a|{p}", f"O_ap|{p}"), "W": V._two(est, dr, f"W_a|{p}", f"W_ap|{p}"),
                           "movement": V._summ(est[f"m|{p}"], dr[f"m|{p}"])}
    for c, (px, py, part) in (contrasts or {}).items():
        for st in ("a", "ap"):
            est[f"d{st}|{c}"] = est[f"{part}_{st}|{px}"] - est[f"{part}_{st}|{py}"]
            dr[f"d{st}|{c}"] = dr[f"{part}_{st}|{px}"] - dr[f"{part}_{st}|{py}"]
        out["contrasts"][c] = V._two(est, dr, f"da|{c}", f"dap|{c}")
    return out


def _src(df, framing, shot, prefix=""):
    return {f"{prefix}{framing}/{shot}": slice_v4(df, framing, shot)}


def compare_pair(df: pd.DataFrame, name: str, n_boot: int = N_BOOT) -> dict | None:
    """The registered pair and pair `name` of one cell, on the patients valid in all their framings: O, W of both, and
    the changes dO, dW = part(name) - part(registered). For the negated pair also the wording contrast D (the W of
    the pair reversed / negated)."""
    reg, new = PAIRS["registered"], PAIRS[name]
    srcs, pairs = {}, {"registered": tuple(f"{f}/{s}" for f, s in reg), name: tuple(f"{f}/{s}" for f, s in new)}
    for f, s in reg + new:
        srcs.update(_src(df, f, s))
    if name == "neg":
        pairs["wording"] = ("reversed/A", "negated/A")
    return pair_comparison(srcs, pairs, {"dW": (name, "registered", "W"), "dO": (name, "registered", "O")}, n_boot)


def aggregate_comparison(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict | None:
    """R13: in the alarm framing, the movement mean(la - ls) with the registered card, the cue and the net-number card,
    on the patients with valid C2 and C4 readings with all three; dm = m(cue) - m(net-number)."""
    ll = {s: _la_ls(slice_v4(df, "alarm", s)) for s in ("A", "Acue", "Anet")}
    ids = None
    for v in ll.values():
        ids = v.index if ids is None else ids.intersection(v.index)
    ids = sorted(ids) if ids is not None else []
    if len(ids) < 10:
        return None
    dif = {s: (v.loc[ids, "la"] - v.loc[ids, "ls"]).to_numpy(float) for s, v in ll.items()}
    fns = {f"m|{s}": (lambda W, x=x: (W @ x) / W.sum(1)) for s, x in dif.items()}
    fns["dm"] = lambda W: (W @ (dif["Acue"] - dif["Anet"])) / W.sum(1)
    est, dr = V.bootstrap(len(ids), fns, n_boot)
    return {"n": len(ids), "movement": {s: V._summ(est[f"m|{s}"], dr[f"m|{s}"]) for s in dif},
            "dm": V._summ(est["dm"], dr["dm"])}


def static_lean(df: pd.DataFrame, pair: str) -> float:
    """Q14c: median over patients of the C0 log-odds of the first answer word, averaged over the two framings."""
    (f1, s1), (f2, s2) = PAIRS[pair]
    vals = []
    for f, s in ((f1, s1), (f2, s2)):
        d = df[(df["framing"] == f) & (df["shot_set"] == s) & (df["cond"] == "C0") & (df["valid"] == 1)]
        vals.append((d["lp_yes"].astype(float) - d["lp_no"].astype(float)).groupby(d["id"]).first())
    if not all(len(v) for v in vals):
        return float("nan")
    ids = vals[0].index.intersection(vals[1].index)
    return float(((vals[0].loc[ids] + vals[1].loc[ids]) / 2).median()) if len(ids) else float("nan")


def descriptives_v4(df: pd.DataFrame) -> dict:
    """Q14e: per framing and set, accuracy by condition, the share of C2 / C4 answers that follow the card, the
    movement, the median mass and the imputed share."""
    out = {}
    for (f, s), g in df.groupby(["framing", "shot_set"]):
        v = g[g["valid"] == 1]
        ev = v[v["cond"].isin(("C2", "C4"))]
        ll = _la_ls(g)
        out[f"{f}/{s}"] = {"by_condition": L._accuracy_table(v, [c for c in SPLIT_CONDS if (v["cond"] == c).any()]),
                           "follow_rate": float((ev["pred_alarm"] == ev["card_dir"]).mean()) if len(ev) else float("nan"),
                           "movement": float((ll["la"] - ll["ls"]).mean()) if len(ll) else float("nan"),
                           "median_mass": _mass_median(g), "imputed_share": float((g["n_imputed"] > 0).mean()),
                           "n_records": int(len(g))}
    return out


def analyse_cell_v4(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Everything for one model x dataset of an instruction-tuned model: validity, the registered pair, each new pair
    against the registered pair, the aggregate cards, the static leans and descriptives."""
    cell = {"n_records": int(len(df)), "n_invalid": int((df["valid"] == 0).sum()),
            "n_records_imputed": int((df["n_imputed"] > 0).sum()), "validity": validity_v4(df)}
    cell["registered"] = pair_comparison({**_src(df, "alarm", "A"), **_src(df, "reversed", "A")},
                                         {"registered": ("alarm/A", "reversed/A")}, n_boot=n_boot)
    cell["pairs"] = {}
    for name in ("tf", "neg", "order", "para", "plain"):
        (f1, s1), (f2, s2) = PAIRS[name]
        if has_slice(df, f1, s1) and has_slice(df, f2, s2):
            cell["pairs"][name] = compare_pair(df, name, n_boot)
    cell["aggregate"] = aggregate_comparison(df, n_boot) if has_slice(df, "alarm", "Acue") else None
    cell["lean"] = {p: static_lean(df, p) for p in PAIRS if has_slice(df, *PAIRS[p][0]) and has_slice(df, *PAIRS[p][1])}
    cell["descriptives"] = descriptives_v4(df)
    return cell


def analyse_2x2(df_base: pd.DataFrame, df_inst: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """R10 for one family and dataset. Primary: B.plain, B.chat and I.chat on the patients valid in all three, with
    P = W(I.chat) - W(B.chat) and F = W(B.chat) - W(B.plain). Secondary (if I.plain was run): the four cells, with
    P' = W(I.plain) - W(B.plain) and F' = W(I.chat) - W(I.plain)."""
    out = {"validity": {"B_plain": split_validity(df_base, "A"), "B_chat": split_validity(df_base, "Achat"),
                        "I_chat": split_validity(df_inst, "A")}}
    src = {**_src(df_base, "alarm", "A", "B:"), **_src(df_base, "reversed", "A", "B:"),
           **_src(df_base, "alarm", "Achat", "B:"), **_src(df_base, "reversed", "Achat", "B:"),
           **_src(df_inst, "alarm", "A", "I:"), **_src(df_inst, "reversed", "A", "I:")}
    pairs = {"B_plain": ("B:alarm/A", "B:reversed/A"), "B_chat": ("B:alarm/Achat", "B:reversed/Achat"),
             "I_chat": ("I:alarm/A", "I:reversed/A")}
    out["primary"] = pair_comparison(src, pairs, {"P": ("I_chat", "B_chat", "W"), "F": ("B_chat", "B_plain", "W"),
                                                  "PO": ("I_chat", "B_chat", "O"), "FO": ("B_chat", "B_plain", "O")},
                                     n_boot)
    if has_slice(df_inst, "alarm", "Abase"):
        out["validity"]["I_plain"] = split_validity(df_inst, "Abase")
        src2 = dict(src, **_src(df_inst, "alarm", "Abase", "I:"), **_src(df_inst, "reversed", "Abase", "I:"))
        pairs2 = dict(pairs, I_plain=("I:alarm/Abase", "I:reversed/Abase"))
        out["secondary"] = pair_comparison(src2, pairs2, {"P2": ("I_plain", "B_plain", "W"),
                                                          "F2": ("I_chat", "I_plain", "W")}, n_boot)
    else:
        out["secondary"] = None
    return out


# ---------------------------------------------------------------- decision rules
def _pv(r):
    p = L._get(r, "p")
    return p if L._finite(p) else None


def kept_test(cells: dict, part: str, pair: str) -> dict:
    """Section 7, 'Kept and removed', for one part (W or O) and one pair over its eligible cells."""
    ent_s, ent_r = {}, {}
    for k, sign in ELIGIBLE[part].items():
        c = cells.get(k)
        comp = (c.get("pairs") or {}).get(pair) if c else None
        r = L._get(comp, "pairs", pair, part)
        ch = L._get(comp, "contrasts", f"d{part}")
        valid = bool(c and c["validity"]["split"]["passes"] and L._get(c, "validity", "pairs", pair, "passes"))
        ent_s[k] = {"p": _pv(r), "direction_ok": V._both(r, sign), "included": valid and _pv(r) is not None,
                    "sign_registered": sign, "a": L._get(r, "a", "est"), "a_prime": L._get(r, "a_prime", "est"),
                    "opposite": bool(valid and V._both(r, -sign) and V._below(_pv(r)))}
        ent_r[k] = {"p": _pv(ch), "direction_ok": V._both(ch, -sign), "included": valid and _pv(ch) is not None,
                    "change_a": L._get(ch, "a", "est"), "change_a_prime": L._get(ch, "a_prime", "est")}
    fs, fr = L.holm_family(ent_s), L.holm_family(ent_r)
    for k in fs:
        fs[k]["reduced"] = fr[k]["supported"]
        fs[k].update(change_a=fr[k]["change_a"], change_a_prime=fr[k]["change_a_prime"], change_p_holm=fr[k]["p_holm"])
    n = len(fs)
    ns = sum(v["supported"] for v in fs.values())
    nr = sum(v["reduced"] for v in fs.values())
    kept = ns > n / 2
    removed = (not kept) and nr > n / 2
    return {"cells": fs, "n_eligible": n, "n_supported": int(ns), "n_reduced": int(nr),
            "n_valid": int(sum(v["included"] for v in fs.values())), "n_opposite": int(sum(v["opposite"] for v in fs.values())),
            "result": "kept" if kept else ("removed" if removed else "not settled")}


JOINT_READING = {("kept", "kept"): "answer polarity", ("removed", "kept"): "the answer words",
                 ("kept", "removed"): "the named outcome", ("removed", "removed"): "the Yes/No question as worded"}


def joint_reading(r8: str, r9: str) -> str:
    return JOINT_READING.get((r8, r9), "not settled by this study")


def r10(fam: dict, families) -> dict:
    """R10 over the eligible cells of the families used. P: predicted to have the eligible sign (Holm); F: two-sided
    (Holm), 'in the direction of W' when both versions have the eligible sign and the Holm p < 0.05."""
    ent_p, ent_f = {}, {}
    for f in families:
        for ds, sign in R10_ELIGIBLE[f].items():
            k = f"{f}|{ds}"
            c = fam.get(k)
            P = L._get(c, "primary", "contrasts", "P")
            F = L._get(c, "primary", "contrasts", "F")
            vb, vc, vi = (bool(L._get(c, "validity", x, "passes")) for x in ("B_plain", "B_chat", "I_chat"))
            ent_p[k] = {"p": _pv(P), "direction_ok": V._both(P, sign), "included": vc and vi and _pv(P) is not None,
                        "sign": sign, "a": L._get(P, "a", "est"), "a_prime": L._get(P, "a_prime", "est")}
            ent_f[k] = {"p": _pv(F), "direction_ok": V._both(F, sign), "included": vb and vc and _pv(F) is not None,
                        "sign": sign, "a": L._get(F, "a", "est"), "a_prime": L._get(F, "a_prime", "est"),
                        "opposite_direction": V._both(F, -sign)}
    fp, ff = L.holm_family(ent_p), L.holm_family(ent_f)
    for k in ff:
        ff[k]["opposite"] = bool(ff[k]["included"] and ff[k]["opposite_direction"] and ff[k]["p_holm"] < 0.05)
    n = len(fp)
    n_p = sum(v["supported"] for v in fp.values())
    n_f = sum(v["supported"] for v in ff.values())
    sup = n >= 1 and n_p > n / 2
    if not n:
        interp = "not testable"
    elif sup and n_f == 0:
        interp = "post-training produces the part, not the chat template"
    elif not sup and n_f > n / 2:
        interp = "the chat template produces the part, not post-training"
    elif sup and n_f > n / 2:
        interp = "both contribute"
    else:
        interp = "not separated by this study"
    return {"families": list(families), "P": fp, "F": ff, "n_eligible": n, "n_P_supported": int(n_p),
            "n_F_in_direction": int(n_f), "supported": bool(sup), "interpretation": interp}


def r13(cells: dict, tags) -> dict:
    ent = {}
    for t in tags:
        for ds in L.DATASETS:
            k = f"{t}|{ds}"
            c = cells.get(k)
            a = c.get("aggregate") if c else None
            valid = bool(c and c["validity"]["split"]["passes"] and L._get(c, "validity", "pairs", "aggregate", "passes"))
            dm = L._get(a, "dm")
            ent[k] = {"p": _pv(dm), "direction_ok": bool(L._finite(L._get(dm, "est")) and L._get(dm, "est") > 0),
                      "included": valid and _pv(dm) is not None, "dm": L._get(dm, "est"),
                      "m_cue": L._get(a, "movement", "Acue", "est"), "m_net": L._get(a, "movement", "Anet", "est"),
                      "m_full": L._get(a, "movement", "A", "est")}
    fam = L.holm_family(ent)
    nv = sum(v["included"] for v in fam.values())
    ns = sum(v["supported"] for v in fam.values())
    return {"cells": fam, "n_valid": int(nv), "n_supported": int(ns), "supported": bool(nv >= 1 and ns > nv / 2)}


def confirmatory_v4(cells: dict, fam: dict, families, tags) -> dict:
    out = {}
    out["R8"] = {"W": kept_test(cells, "W", "tf"), "O": kept_test(cells, "O", "tf")}
    out["R9"] = {"W": kept_test(cells, "W", "neg"), "O": kept_test(cells, "O", "neg")}
    out["joint_reading"] = joint_reading(out["R8"]["W"]["result"], out["R9"]["W"]["result"])
    out["R10"] = r10(fam, families)
    out["R11"] = {"W": kept_test(cells, "W", "order"), "O": kept_test(cells, "O", "order")}
    out["R12"] = {"W": kept_test(cells, "W", "para"), "O": kept_test(cells, "O", "para")}
    out["R13"] = r13(cells, tags)
    for r in ("R8", "R9", "R11", "R12"):
        out[r]["prediction_holds"] = {p: out[r][p]["result"] == "kept" for p in ("W", "O")}
    if out["R11"]["O"]["result"] == "removed":
        out["R11"]["O_reading"] = "the outcome part depends on the order of the solved examples"
    elif out["R11"]["O"]["result"] == "kept":
        out["R11"]["O_reading"] = "the outcome part is not produced by the outcome of the last solved example"
    else:
        out["R11"]["O_reading"] = "not settled"
    return out


def lean_vs_w(cells: dict) -> dict:
    """Q14c: per pair, sign agreement of W and the static lean in the valid cells where W is conclusive, and the
    Spearman correlation over the valid cells."""
    from scipy.stats import spearmanr
    out = {}
    for p in ("registered",) + TESTED_PAIRS:
        xs, ys, agree, n = [], [], 0, 0
        for k, c in cells.items():
            comp = c["registered"] if p == "registered" else (c.get("pairs") or {}).get(p)
            w = L._get(comp, "pairs", p, "W")
            lean = (c.get("lean") or {}).get(p)
            valid = c["validity"]["split"]["passes"] and (p == "registered" or L._get(c, "validity", "pairs", p, "passes"))
            if not (valid and w and L._finite(lean)):
                continue
            xs.append(lean)
            ys.append(L._get(w, "a_prime", "est"))
            if w["conclusive"]:
                n += 1
                agree += int(np.sign(lean) == w["sign"])
        rho = float(spearmanr(xs, ys).correlation) if len(xs) >= 4 else float("nan")
        out[p] = {"n_valid": len(xs), "n_conclusive": n, "n_sign_agrees": agree, "spearman_lean_W_a_prime": rho}
    return out


def replay_v4(new: pd.DataFrame, old: pd.DataFrame, shots=("A",), framings=L.FRAMINGS) -> dict:
    """Q14a: shared prompts scored again (same key and text) against an earlier run."""
    keep = lambda d: d[d["shot_set"].isin(shots) & d["framing"].isin(framings) & d["cond"].isin(SPLIT_CONDS)]
    m = keep(new).merge(keep(old), on="key", suffixes=("", "_old"))
    out = {"n_shared_prompts": int(len(m)), "n_text_differs": int((m["text_sha"] != m["text_sha_old"]).sum())}
    v = m[(m["valid"] == 1) & (m["valid_old"] == 1) & (m["text_sha"] == m["text_sha_old"])]
    d = (v["lo_alarm"].astype(float) - v["lo_alarm_old"].astype(float)).abs()
    out.update(n_compared=int(len(v)), max_abs_diff=float(d.max()) if len(d) else float("nan"),
               median_abs_diff=float(d.median()) if len(d) else float("nan"),
               share_same_answer=float((v["pred_alarm"] == v["pred_alarm_old"]).mean()) if len(v) else float("nan"))
    return out


def decision_E(d: pd.DataFrame) -> float:
    """Added false alarms minus added missed alarms in one framing, from decisions pred_alarm (0/1): added false
    alarms = FPR with the alarm-pointing card - FPR at C0 (y = 0); added missed alarms = FNR with the safe-pointing
    card - FNR at C0 (y = 1). Uses patients with all three decisions."""
    w = d.pivot_table(index="id", columns="cond", values=["pred_alarm", "card_dir", "y"], aggfunc="first")
    need = [("pred_alarm", c) for c in SPLIT_CONDS] + [("card_dir", "C2")]
    if not all(k in w.columns for k in need):
        return float("nan")
    w = w.dropna(subset=need)
    if not len(w):
        return float("nan")
    cd = w[("card_dir", "C2")].astype(int).to_numpy()
    y = w[("y", "C0")].astype(int).to_numpy()
    c0 = w[("pred_alarm", "C0")].astype(int).to_numpy()
    pa = np.where(cd == 1, w[("pred_alarm", "C2")], w[("pred_alarm", "C4")]).astype(int)
    ps = np.where(cd == 1, w[("pred_alarm", "C4")], w[("pred_alarm", "C2")]).astype(int)
    if not (y == 0).any() or not (y == 1).any():
        return float("nan")
    add_fa = pa[y == 0].mean() - c0[y == 0].mean()
    add_miss = (1 - ps[y == 1]).mean() - (1 - c0[y == 1]).mean()
    return float(add_fa - add_miss)


def greedy_summary(gen: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """Q14d for one model: per mode, the parsed share, the agreement with the first-token reading, and E_W from the
    generated answers and from the first-token reading on the same prompts (own threshold)."""
    out = {}
    base = rec[["key", "dataset", "id", "cond", "framing", "y", "card_dir", "valid", "lp_yes", "lp_no"]]
    for mode, g in gen.groupby("mode"):
        m = g.merge(base, on="key", how="left", suffixes=("", "_r"))
        parsed = m["parsed"].notna()
        ok = parsed & (m["valid"] == 1)
        first = np.where(m["lp_yes"].astype(float) > m["lp_no"].astype(float), "Yes", "No")
        agree = float((m.loc[ok, "parsed"] == first[ok.to_numpy()]).mean()) if ok.any() else float("nan")
        res = {"n": int(len(m)), "parsed_share": float(parsed.mean()), "agreement": agree, "by_dataset": {}}
        for ds in L.DATASETS:
            Es = {}
            for src in ("generated", "first_token"):
                Ef = {}
                for fr in L.FRAMINGS:
                    d = m[(m["dataset"] == ds) & (m["framing"] == fr) & ok].copy()
                    yes = (d["parsed"] == "Yes") if src == "generated" else (d["lp_yes"].astype(float) > d["lp_no"].astype(float))
                    d["pred_alarm"] = (yes if fr == "alarm" else ~yes).astype(int)
                    Ef[fr] = decision_E(d)
                Es[src] = (Ef["alarm"] - Ef["reversed"]) / 2
            res["by_dataset"][ds] = {"E_W_generated": Es["generated"], "E_W_first_token": Es["first_token"]}
        out[mode] = res
    return out


def halve_less_likely_v4(df: pd.DataFrame) -> pd.DataFrame:
    """The imputation bound of Deviation D5 (the second study's method): in every record with an imputed variant, the
    probability of the less likely answer word is halved, and the record is scored again."""
    df = df.copy()
    m = df["n_imputed"] > 0
    lower = df["lp_yes"] < df["lp_no"]
    df.loc[m & lower, "lp_yes"] -= np.log(2)
    df.loc[m & ~lower, "lp_no"] -= np.log(2)
    keep = ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir", "xgb_p")
    rows = [finish_record_v4({k: r[k] for k in keep if k in r} | {"n_imputed": r["n_imputed"]},
                             {"lp_yes": r["lp_yes"], "lp_no": r["lp_no"], "n_imputed": r["n_imputed"]}, r["framing"])
            for r in df[m].to_dict("records")]
    out = pd.concat([df[~m], pd.DataFrame(rows)], ignore_index=True)
    out["card_dir"] = out["card_dir"].astype(float)
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "smoke":
        smoke_v4(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "instruct", sys.argv[4] if len(sys.argv) > 4 else None)
    elif len(sys.argv) >= 6 and sys.argv[1] == "engcheck":
        a = sys.argv + ["None"] * 4
        nz = lambda x: None if x == "None" else x
        engineering_check_v4(a[2], a[3], a[4], a[5], nz(a[6]), nz(a[7]), nz(a[8]))
