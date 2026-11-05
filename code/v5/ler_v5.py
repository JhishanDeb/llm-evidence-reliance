"""Fifth study (PREREGISTRATION_V5.md, registered on OSF and released as prereg-v5): does the presence of a card
change a model's answer prior, and what remains of the polarity part W once it is measured against a card without
content, on patients that no earlier study used?

Everything the earlier studies registered (data, splits, XGBoost, cards, solved examples, the registered prompts, the
True/False and answer-option formats, the numbers card, the reading of the answer-word log-probabilities) comes from
ler_main.py (main-2.0), ler_v3.py (v3-1.1) and ler_v4.py (v4-1.2), which are not changed (their SHA-256 are checked
below). This module adds only what the fifth study needs:

  1. the placebo card (the patient's own five factors, each with "does not change the risk ... (impact 0.00)"), the
     numbers card at three scales, the risk-probability evidence, and the checks that each new prompt differs from a
     registered prompt only in its evidence block;
  2. the fresh patients: PhysioNet 2012 set C (never used before) and the PhysioNet 2019 patients of the test and dev
     pools that no earlier study used;
  3. the prompt grids, the models, the decisions of the engineering check and the check against the protocol text;
  4. reading every framing with its own answer words, and free generation;
  5. the engineering check;
  6. a fake model for the CPU tests;
  7. the registered analysis: validity, the static lean with and without the placebo card (R15), W against the
     placebo card at its neutral point (R16), the decision-level polarity effect of generated answers (R17), and the
     registered questions Q18 to Q22;
  8. the diagnostics of Q21 and Q22 and the imputation bound.

v5-0.1 ran both engineering checks; v5-1.0 (registered) adds the run and analysis functions (sections 3 and 8), the
merge of the two checks, and the recorded decisions, fingerprints and hashes. No prompt, reading or sample changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ler_main as L
import ler_v3 as V
import ler_v4 as W4

V5_VERSION = "v5-1.0"                  # registered (prereg-v5); v5-0.1 ran the two engineering checks
MAIN_SHA256 = "0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e"     # ler_main.py, main-2.0
V3_SHA256 = "89d4c0c0ed995e82aa472eeac3f22ea0728583c7c8c84a01c87befe18f5e92ec"       # ler_v3.py, v3-1.1
V4_SHA256 = "8a23f2ece854317185a720606d1bf3b427ea8e5589bf9ec9577f9ea4941af0a7"       # ler_v4.py, v4-1.2
PROTOCOL_V5_TAG = "prereg-v5"
PROTOCOL_V5_URL = f"{L._RAW}/{PROTOCOL_V5_TAG}/PREREGISTRATION_V5.md"
PROTOCOL_AT_V5_URLS = {name: f"{L._RAW}/{PROTOCOL_V5_TAG}/{name}" for name in
                       ("PREREGISTRATION.md", "PREREGISTRATION_V2.md", "PREREGISTRATION_V3.md", "PREREGISTRATION_V4.md")}
V5_MARKERS = ["## 3. Patients", "## 4. Models", "## 5. Prompts", "### R15.", "### R16.", "### R17.", "### Q18.",
              "### Q19.", "### Q20.", "### Q21.", "### Q22.", "## 8. Analysis", "## 10."]
DATASETS_V5 = ("physionet2012", "physionet2019")
N_BOOT = 10_000
N_BOOT_BOUND = 2_000                  # the imputation bound (Q21f), as in the fourth study
YN, TF, AB = W4.YN_WORDS, W4.TF_WORDS, L.ANSWER_WORDS["choice"]

# =============================================================================================================
# 1. Prompts (PREREGISTRATION_V5.md, Section 5)
# =============================================================================================================
PLACEBO_LINE = "{k}. {feature} = {value}: does not change the risk of {outcome} (impact 0.00)"
PROB_TEXT = ("Evidence from a machine-learning model (XGBoost) trained on other patients: its estimated risk of "
             "{outcome} for this patient is {p}%.")
DOSE_SETS = {"Anum05": 0.5, "Anum1": 1.0, "Anum2": 2.0}          # the numbers card of the third study, impacts scaled
CONDS = ("C0", "CP", "C2", "C4", "CQ2", "CQ4")
# C0 no evidence; CP the placebo card; C2 the true card; C4 the flipped card; CQ2 the risk model's probability;
# CQ4 one minus that probability
FRAMINGS_V5 = {
    "alarm": dict(words=YN, pos_alarm=True), "reversed": dict(words=YN, pos_alarm=False),
    "tf_alarm": dict(words=TF, pos_alarm=True), "tf_reversed": dict(words=TF, pos_alarm=False),
    "choice_alarm_A": dict(words=AB, pos_alarm=True), "choice_safe_A": dict(words=AB, pos_alarm=False),
    "para_alarm": dict(words=YN, pos_alarm=True), "para_reversed": dict(words=YN, pos_alarm=False),
}
PAIRS_V5 = {"yn": ("alarm", "reversed"), "pa": ("para_alarm", "para_reversed"), "tf": ("tf_alarm", "tf_reversed"),
            "ab": ("choice_alarm_A", "choice_safe_A")}


def answer_words(framing: str) -> tuple:
    return FRAMINGS_V5[framing]["words"]


def placebo_text(b, card) -> str:
    """The registered evidence block with the same header, factors and values, without any direction or size."""
    lines = [PLACEBO_LINE.format(k=k, feature=L.display_name(b.name, it["col"]),
                                 value=L.display_value(b.name, it["col"], it["value"], b.id_maps), outcome=b.spec["outcome"])
             for k, it in enumerate(card, 1)]
    return L.EVIDENCE_HEADER + "\n" + "\n".join(lines)


def prob_percent(b, rid) -> int:
    """The risk model's probability for the patient, in whole percent (the value shown in CQ2)."""
    return int(round(100 * float(b.xgb_p[rid])))


def prob_text(b, rid, cond) -> str:
    p = prob_percent(b, rid)
    return PROB_TEXT.format(outcome=b.spec["outcome"], p=p if cond == "CQ2" else 100 - p)


def _base(b, rid, cond, framing):
    """The registered prompt of a framing (ler_main for Yes/No and answer options, ler_v4 for True/False and the
    paraphrased Yes/No prompt)."""
    if framing in ("tf_alarm", "tf_reversed", "para_alarm", "para_reversed"):
        return W4.prompt_v4(b, rid, cond, framing, "A")
    return b.prompt(rid, cond, framing, "A")


def _swap_block(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("the registered evidence block is not found exactly once")
    return text.replace(old, new)


def prompt_v5(b, rid, cond: str, framing: str, shot: str = "A"):
    """-> (text, card_dir) of one prompt of the fifth study. card_dir = 1 if the evidence points to the alarming
    outcome, 0 if it points to the safe outcome, None without direction (C0, CP, and CQ at exactly 50%)."""
    if framing not in FRAMINGS_V5 or cond not in CONDS:
        raise ValueError((cond, framing))
    if shot in DOSE_SETS:
        if framing != "alarm" or cond not in ("C2", "C4"):
            raise ValueError("the scaled numbers cards are used with C2 and C4 in the alarm framing only")
        text, cdir = b.prompt(rid, cond, "alarm", "A")
        card = b.shown_card(rid, cond)
        scaled = [dict(it, shap=float(it["shap"]) * DOSE_SETS[shot]) for it in card]
        return _swap_block(text, b.card_text(card), V.card_text_v3(b, scaled, "numbers")), cdir
    if shot != "A":
        raise ValueError(shot)
    if cond in ("C0", "C2", "C4"):
        return _base(b, rid, cond, framing)
    text, _ = _base(b, rid, "C2", framing)
    true_card = b.shown_card(rid, "C2")
    if cond == "CP":
        return _swap_block(text, b.card_text(true_card), placebo_text(b, true_card)), None
    if framing not in ("alarm", "reversed"):
        raise ValueError("the probability evidence is used with the Yes/No question only")
    p = prob_percent(b, rid) if cond == "CQ2" else 100 - prob_percent(b, rid)
    cdir = None if p == 50 else int(p > 50)
    return _swap_block(text, b.card_text(true_card), prob_text(b, rid, cond)), cdir


def check_prompt_v5(b, rid, cond, framing, shot, text, cdir):
    """Assertions for one prompt (run notebook and tests): the prompt is a registered prompt with exactly the change
    that Section 5 of the protocol describes."""
    ev = "Evidence from a machine-learning model (XGBoost) trained on other patients"
    assert text.count(ev) == (0 if cond == "C0" else 1), "evidence block missing or repeated"
    if shot in DOSE_SETS:
        ref, rdir = b.prompt(rid, cond, "alarm", "A")
        assert cdir == rdir
        if shot == "Anum1":
            assert text == V.prompt_v3(b, rid, cond, "alarm", "numbers")[0], "the x1 card is not the numbers card"
        block = text.split(L.EVIDENCE_HEADER, 1)[1].split("\n\n")[0]
        assert "increases" not in block and "decreases" not in block and block.count("(impact ") == len(b.cards[rid])
        card = b.shown_card(rid, cond)
        assert text == ref.replace(b.card_text(card), V.card_text_v3(
            b, [dict(it, shap=float(it["shap"]) * DOSE_SETS[shot]) for it in card], "numbers"))
        return True
    ref, rdir = _base(b, rid, cond if cond in ("C0", "C2", "C4") else "C2", framing)
    if cond in ("C0", "C2", "C4"):
        assert text == ref and cdir == rdir
        return True
    card = b.shown_card(rid, "C2")
    if cond == "CP":
        new = placebo_text(b, card)
        assert cdir is None and text == ref.replace(b.card_text(card), new)
        assert new.count("(impact 0.00)") == len(card) and "increases" not in new and "decreases" not in new
        return True
    p = prob_percent(b, rid) if cond == "CQ2" else 100 - prob_percent(b, rid)
    assert text == ref.replace(b.card_text(card), prob_text(b, rid, cond)) and f"is {p}%." in text
    assert cdir == (None if p == 50 else int(p > 50))
    return True


# =============================================================================================================
# 2. Fresh patients (Section 3)
# =============================================================================================================
FRESH_SEED = 2032
FRESH_PER_CLASS = 300
GEN_SEED = 2033
GEN_PER_CLASS = 150            # the generation subsample; 100 if rule (iii) of Section 10 applies (it does: see
                               # ENGCHECK_DECISIONS["gen_per_class"], which the runs use)
# the registered test samples of Studies 1 to 4 (run logs), rebuilt and checked before they are excluded
REGISTERED_TEST_FP = {"physionet2012": {"test_ids": "eb7ca19be05d8b46", "test_cards": "e9abe0533092f138"},
                      "physionet2019": {"test_ids": "cda9d34771cde56d", "test_cards": "b0334b1289dee4da"}}
# recorded by the engineering check (both checks built the same samples); every run stops if its samples differ
FRESH_FP = {"physionet2012": {"fresh_ids": "c9555e028ee16b9e", "fresh_cards": "fa60197aec5ea6fa"},
            "physionet2019": {"fresh_ids": "9940fba7c625991c", "fresh_cards": "dd0a04f4080e72a3"}}


def load_p12_set_c(data_dir, extra_roots=(), log=print) -> pd.DataFrame:
    """PhysioNet 2012 set C (4,000 ICU stays, never used in the earlier studies) with its outcomes, in the format of
    ler_main.load_physionet2012."""
    d = Path(data_dir) / "physionet2012"
    d.mkdir(parents=True, exist_ok=True)
    rec_dir, outc = L._p12_get_set("c", d, [d, *extra_roots], log)
    recs = L._p12_records(rec_dir)
    if len(recs) < L.P12_MIN_PER_SET:
        raise RuntimeError(f"PhysioNet 2012 set-c: {len(recs)} records found, expected 4,000")
    feats = pd.DataFrame([L.p12_read_record(fp) for fp in recs])
    o = pd.read_csv(outc)
    o.columns = [c.strip() for c in o.columns]
    o = o[["RecordID", "In-hospital_death"]].rename(columns={"RecordID": "id", "In-hospital_death": "y"})
    o["id"] = o["id"].astype(int).astype(str)
    df = feats.merge(o, on="id", how="inner")
    df = df[df["y"].isin([0, 1])].reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    return df


def _pick(ids_pos, ids_neg, per_class, seed):
    k = min(per_class, len(ids_pos), len(ids_neg))
    rs = np.random.RandomState(seed)
    pick = list(rs.choice(np.array(sorted(ids_pos), dtype=object), k, replace=False)) + \
        list(rs.choice(np.array(sorted(ids_neg), dtype=object), k, replace=False))
    return [pick[i] for i in rs.permutation(len(pick))], k


def build_fresh(b, set_c: pd.DataFrame | None = None, pilot_ids=(), log=print) -> dict:
    """Builds the fresh sample of a bundle that holds the registered test sample (sample='test'), stores it as the
    bundle's sample 'fresh' and switches to it. PhysioNet 2012: eligible set C stays (at least 10 of the 16 features);
    PhysioNet 2019: eligible patients of the test and dev pools outside the registered test and dev samples. Up to 300
    per class (all eligible positives and as many negatives if fewer), seed 2032. -> info with fingerprints."""
    if "test" not in b._samples:
        raise RuntimeError("build the bundle with sample='test' first")
    test_ids, dev_ids = set(b._samples["test"]["ids"]), set(b._samples["dev"]["ids"])
    if b.name == "physionet2012":
        if set_c is None:
            raise ValueError("PhysioNet 2012 needs set C")
        df = set_c.copy()
        df = df[~df["id"].isin(set(map(str, pilot_ids)))]
        overlap = set(df["id"]) & set(b.df["id"])
        if overlap:
            raise RuntimeError(f"{len(overlap)} set C ids also occur in sets A and B")
        X = df.reindex(columns=b.features).astype(float)
        source = "set C"
    elif b.name == "physionet2019":
        rows = np.concatenate([b.test_pool, b.dev_pool])
        df = b.df.iloc[rows]
        df = df[~df["id"].isin(test_ids | dev_ids)]
        X = df[b.features]
        source = "test and dev pools minus the registered test and dev samples"
    else:
        raise ValueError(b.name)
    el = L.n_measured(X) >= L.MIN_MEASURED
    df, X = df[el], X[el]
    y = df["y"].to_numpy().astype(int)
    ids = df["id"].astype(str).to_numpy()
    pick, k = _pick(ids[y == 1], ids[y == 0], FRESH_PER_CLASS, FRESH_SEED)
    if k < 10:
        raise RuntimeError(f"{b.name}: only {k} fresh eligible patients per class")
    pos = {r: i for i, r in enumerate(ids)}
    rows = [pos[r] for r in pick]
    Xp = X.iloc[rows]
    sv, _ = L.shap_values(b.model, Xp, b.categorical)
    b.xgb_p = dict(zip(pick, b.model.predict_proba(Xp)[:, 1]))
    b.cards, b.card_dir, b.values = {}, {}, {}
    for i, rid in enumerate(pick):
        order = np.argsort(-np.abs(sv[i]), kind="mergesort")[:L.TOP_K]
        b.cards[rid] = [{"col": b.features[j], "value": Xp.iloc[i, j], "shap": float(sv[i, j])} for j in order]
        b.card_dir[rid] = int(sum(c["shap"] for c in b.cards[rid]) > 0)
        b.values[rid] = {c: Xp.iloc[i][c] for c in b.features}
    b.ids, b.patients = list(pick), np.array(rows)
    b.y = {rid: int(y[pos[rid]]) for rid in pick}
    b.partner = {rid: None for rid in pick}
    b._samples["fresh"] = b._state()
    b.sample = "fresh"
    fp = {"fresh_ids": L._fp(sorted(b.ids)),
          "fresh_cards": L._fp({r: [(c["col"], round(c["shap"], 6)) for c in b.cards[r]] for r in sorted(b.ids)})}
    info = {"source": source, "eligible": int(len(df)), "eligible_pos": int((y == 1).sum()),
            "eligible_neg": int((y == 0).sum()), "per_class": int(k), "n": len(pick),
            "fallback_rule_applied": bool(k < FRESH_PER_CLASS),
            "card_points_alarm_share": float(np.mean(list(b.card_dir.values()))),
            "prob_exactly_50": int(sum(prob_percent(b, r) == 50 for r in pick)),
            "overlap_with_earlier_samples": len(set(pick) & (test_ids | dev_ids)), "fingerprint": fp}
    b.info["fresh_sample"] = info
    log(f"  {b.name}: fresh sample from {source}: {len(pick)} patients ({k} per class; eligible positives "
        f"{info['eligible_pos']}){' -> FALLBACK RULE' if k < FRESH_PER_CLASS else ''}")
    return info


def gen_patients(b, per_class=None) -> list:
    """The generation subsample: per class, `per_class` patients of the fresh sample, seed 2033, sorted ids."""
    per = GEN_PER_CLASS if per_class is None else per_class
    ids = sorted(b.ids)
    rs = np.random.RandomState(GEN_SEED)
    out = []
    for cls in (1, 0):
        pool = [i for i in ids if b.y[i] == cls]
        k = min(per, len(pool))
        out += [pool[j] for j in sorted(rs.choice(len(pool), k, replace=False))]
    return sorted(out)


# =============================================================================================================
# 3. Grids, models, decisions of the engineering check, protocol check
# =============================================================================================================
COND4 = ("C0", "CP", "C2", "C4")
GRID_DEF = {
    "yn": [(c, f, "A") for f in PAIRS_V5["yn"] for c in COND4],
    "pa": [(c, f, "A") for f in PAIRS_V5["pa"] for c in COND4],
    "tf": [(c, f, "A") for f in PAIRS_V5["tf"] for c in COND4],
    "ab": [(c, f, "A") for f in PAIRS_V5["ab"] for c in COND4],
    "dose": [(c, "alarm", s) for s in DOSE_SETS for c in ("C2", "C4")],
    "prob": [(c, f, "A") for f in PAIRS_V5["yn"] for c in ("CQ2", "CQ4")],
}
GRIDS = ("yn", "pa", "tf", "ab", "dose", "prob")                 # 42 scored prompts per patient
OPTIONAL_GRIDS = ("pa", "tf", "ab", "dose", "prob")              # may be dropped by rule (ii) of Section 10
GEN_GRID = [(c, f, "A") for f in PAIRS_V5["yn"] for c in ("C0", "C2", "C4")]   # free generation
GEN_MAX_TOKENS = 512

MODELS_V5 = {
    "meta-llama/Llama-3.1-8B-Instruct": dict(tag="llama-instruct", family="Llama 3.1",
                                             revision="0e9e39f249a16976918f6564b8830bc894c89659"),
    "Qwen/Qwen2.5-7B-Instruct": dict(tag="qwen-instruct", family="Qwen2.5", revision="a09a35458c702b33eeacc393d103063234e8bc28"),
    "Qwen/Qwen3-8B": dict(tag="qwen3-instruct", family="Qwen3", revision="b968826d9c46dd6066d109eabc6255188de91218"),
    "allenai/Olmo-3-7B-Instruct": dict(tag="olmo3-instruct", family="Olmo 3", revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
    "ibm-granite/granite-3.3-8b-instruct": dict(tag="granite3-instruct", family="Granite 3.3",
                                                revision="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"),
    "tiiuae/Falcon3-7B-Instruct": dict(tag="falcon3-7b", family="Falcon3", revision="1e57a0ecd176c7c139f289c60a74e57f887c3dfb"),
    "01-ai/Yi-1.5-9B-Chat-16K": dict(tag="yi15-9b", family="Yi-1.5", revision="2f8753bed95bd4fede3fd64f8305428620b4b212"),
}
TAG2ID = {m["tag"]: mid for mid, m in MODELS_V5.items()}
# Decisions taken from the engineering check by the rules of Section 10, set before registration. None = the check has
# not been recorded yet; a real run refuses to start then.
#   not_run:       model ids that are not run (rule i)
#   grids_not_run: tag -> grids of OPTIONAL_GRIDS that are not run (rule ii)
#   gen_per_class: patients per class in the generation subsample (rule iii)
# Rules (i) to (iii) on both checks (the second check, of Yi-1.5 only, replaces the first for that model):
ENGCHECK_DECISIONS = {"not_run": [], "grids_not_run": {"qwen3-instruct": ["tf"]}, "gen_per_class": 100,
                      "projected_generation_h_at_150": 9.14}
ENGCHECK_ZIP_SHA256 = ("a2adc60251ffac8d078c5e7fe6d148a54f913a004bfa6e67835cc56bae91b0b7",    # engcheck_v5.zip
                       "6f44b4bd695bfc763490041323fdc9f49489d631d0ea126cfc59bb0da2fc4b01")    # engcheck_v5_run2.zip
ENGCHECK_CODE_SHA256 = ("b9aeb9b2dca6cdddff343ac35b6647969920e6aa68cb746388d0b4e03d2a2e64",)   # ler_v5.py v5-0.1, both checks
SCORED_PER_PATIENT = sum(len(GRID_DEF[g]) for g in GRIDS)
GEN_HOURS_LIMIT = 8.0


def grid_v5(grids) -> list:
    out = []
    for g in grids:
        out += GRID_DEF[g]
    if len(set(out)) != len(out):
        raise ValueError("duplicate grid entries")
    return out


def model_grids(model_id: str, decisions=None) -> tuple:
    d = ENGCHECK_DECISIONS if decisions is None else decisions
    if d is None:
        raise RuntimeError("the decisions of the engineering check are not recorded in ler_v5.py (ENGCHECK_DECISIONS)")
    if model_id in d.get("not_run", ()):
        raise RuntimeError(f"{model_id} is not run (engineering check, rule i)")
    drop = set(d.get("grids_not_run", {}).get(MODELS_V5[model_id]["tag"], ()))
    return tuple(g for g in GRIDS if g not in drop)


def protocol_strings_v5() -> list:
    """Every fixed string the fifth study adds, the code hashes, the seeds, and every model id and revision, for the
    check against PREREGISTRATION_V5.md."""
    out = [PLACEBO_LINE, PROB_TEXT, MAIN_SHA256, V3_SHA256, V4_SHA256, V.NUMBERS_LINE, L.EVIDENCE_HEADER,
           W4.TF_INSTRUCTION, L.CHOICE_QUESTION, L.CHOICE_INSTRUCTION, W4.PARA_INSTRUCTION]
    for mid, m in MODELS_V5.items():
        out += [mid, m["revision"]]
    for ds, fp in REGISTERED_TEST_FP.items():
        out += list(fp.values())
    for ds, fp in (FRESH_FP or {}).items():
        out += list(fp.values())
    out += list(ENGCHECK_ZIP_SHA256 or ()) + list(ENGCHECK_CODE_SHA256 or ())
    return out


def check_against_protocol_v5(text: str) -> list:
    flat = re.sub(r"\s+", " ", text.replace("`", ""))
    miss = [s for s in protocol_strings_v5() if re.sub(r"\s+", " ", s) not in flat]
    return miss + [m for m in V5_MARKERS if m not in text]


# =============================================================================================================
# 4. Reading and generation
# =============================================================================================================
def render_v5(text: str, tok) -> str:
    return L.render(text, "instruct", tok)


def read_v5(be, rendered, framing):
    return [{"lp_yes": r["lp_pos"], "lp_no": r["lp_neg"], "n_imputed": r["n_imputed"]}
            for r in be.answer_logprobs(rendered, answer_words(framing))]


def finish_record_v5(rec: dict, r: dict, framing: str) -> dict:
    """lo_alarm = log-odds of the alarming outcome in every framing (the first answer word stands for it when
    pos_alarm)."""
    if framing in L.CHOICE_FRAMINGS:
        return L.finish_record(rec, r, framing)
    return W4.finish_record_v4(rec, r, framing)


render_free = W4.render_free
parse_generated = W4.parse_generated
generate = W4.generate
SANITY_V5 = dict(W4.SANITY_V4) | {k: v for k, v in L.SANITY_PROMPTS.items() if v[1] == AB}


# =============================================================================================================
# 5. Engineering check (before registration; dev-sample prompts only)
# =============================================================================================================
def engineering_check_v5(model_id, prompts_path, cache_dir, revision=None):
    """Can this model be run as registered, and with which grids? Engineering only: download, the vLLM float16 load
    with the model window, the chat template and its default system text, single-token answer words (Yes/No,
    True/False, A/B), the exact-reading check (Deviation D2), finite readings, the median answer-word mass per framing
    and grid, the imputed share under the top-1,000 reading (Deviation D5), prompt lengths, the time per scored prompt,
    the obvious questions, and the time and parse share of freely generated answers. No accuracy, no answer direction,
    no agreement and no asymmetry is computed. Prints one ENGCHECK_V5_RESULT line."""
    res = {"model_id": model_id, "ok": False, "problems": [],
           "code": {"ler_v5": V5_VERSION, "ler_v4": W4.V4_VERSION, "ler_v3": V.V3_VERSION, "ler_main": L.CODE_VERSION}}
    t0 = time.time()
    try:
        path, sha = L.download_model(model_id, cache_dir, revision, os.environ.get("HF_TOKEN"))
        res.update(revision=sha, download_seconds=round(time.time() - t0, 1))
        cfg = json.load(open(os.path.join(path, "config.json")))
        res["config"] = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                         "dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
                         "quantization": (cfg.get("quantization_config") or {}).get("quant_method")}
        win = V.model_window(path)
        res["max_model_len"] = win
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        t1 = time.time()
        be = L.VLLMBackend(path, tensor_parallel_size=tp, max_model_len=win, read="token_ids")
        res.update(load_seconds=round(time.time() - t1, 1), tensor_parallel_size=tp)
        tok = be.tokenizer
        res["template_system_text"] = L.template_system_text(tok)
        res["single_token_words"] = {}
        for pair in (YN, TF, AB):
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
            if it["set"] in ("check", "timing"):
                groups.setdefault((it["set"], it["grid"], it["framing"], it["shot"], it["cond"]), []).append(it)
        rendered = {k: [render_v5(it["text"], tok) for it in v] for k, v in groups.items()}
        ntok = [len(tok.encode(r, add_special_tokens=False)) for v in rendered.values() for r in v]
        res["prompt_tokens"] = {"median": int(np.median(ntok)), "max": int(max(ntok))}
        if max(ntok) >= win - 1:
            res["problems"].append(f"a prompt is longer than the model window ({max(ntok)} >= {win - 1})")
        for k, rs in rendered.items():
            if k[0] == "check" and k[2] in ("alarm",) and k[4] in ("CP", "CQ2") and k[3] == "A":
                res[f"rendered_tail_{k[4]}"] = rs[-1][-700:]
        ex = {}
        for k, rs in rendered.items():
            if k[0] == "check" and k[3] == "A":
                ex.setdefault(answer_words(k[2]), []).extend(rs[:6])
        for pair, rs in ex.items():
            if res["single_token_words"].get("/".join(pair)):
                res["exact_check_" + "/".join(pair)] = be.check_exact(rs[:12], pair)
        be.read = "topk"
        res["readings"] = {}
        by = {}
        for (sset, grid, fr, shot, cond), rs in rendered.items():
            by.setdefault((sset, grid, fr, shot), []).extend(rs)
        for (sset, grid, fr, shot), rs in sorted(by.items()):
            pair = answer_words(fr)
            if not res["single_token_words"].get("/".join(pair)):
                continue
            t1 = time.time()
            rows = be.answer_logprobs(rs, pair)
            el = time.time() - t1
            res["readings"][f"{sset}|{grid}|{fr}|{shot}"] = dict(L._mass(rows), seconds=round(el, 1),
                                                               seconds_per_prompt=round(el / max(1, len(rs)), 4))
        san = {}
        for name, (text, pair, first) in SANITY_V5.items():
            if not res["single_token_words"].get("/".join(pair)):
                continue
            r = be.answer_logprobs([L.render(text, "instruct", tok)], pair)[0]
            san[name] = bool((r["lp_pos"] > r["lp_neg"]) == first)
        res["sanity_right_word_more_likely"] = san
        gi = [it for it in items if it["set"] == "gen"]
        ps = [render_free(it["text"], tok) for it in gi]
        t1 = time.time()
        outs = generate(be, ps, GEN_MAX_TOKENS)
        el = time.time() - t1
        parsed = [parse_generated(o) for o in outs]
        res["generated_free"] = {"n": len(outs), "parsed_share": float(np.mean([p is not None for p in parsed]))
                                 if outs else float("nan"), "seconds_per_prompt": round(el / max(1, len(outs)), 4),
                                 "examples": [o[:160] for o in outs[:3]]}
        ok = {}
        for g in GRIDS:
            ks = [k for k in res["readings"] if k.startswith(f"check|{g}|")]
            ok[g] = bool(ks) and all(res["readings"][k]["finite_share"] == 1 for k in ks) and \
                all(res["readings"][k]["mass_median"] >= L.VALIDITY_MIN_MASS for k in ks)
        res["grids_ok"] = ok
        if not ok.get("yn"):
            res["problems"].append("the Yes/No prompts do not reach a median mass of 0.5 with finite readings")
        if res["config"].get("quantization"):
            res["problems"].append(f"quantized weights ({res['config']['quantization']}); the study runs 16-bit weights")
        res["ok"] = not res["problems"]
    except Exception as e:
        import traceback
        res["problems"].append(f"error: {e!r}"[:500])
        res["traceback"] = traceback.format_exc()[-3000:]
    res["total_seconds"] = round(time.time() - t0, 1)
    print("ENGCHECK_V5_RESULT " + json.dumps(res, default=str))
    return res


def projected_hours(r: dict, n_patients: int, n_gen_patients: int, grids=GRIDS) -> dict:
    """Projected run time of one model from its check: scored prompts at the timing speed, generated answers at the
    measured speed."""
    spp = ((r.get("readings") or {}).get("timing|yn|alarm|A") or {}).get("seconds_per_prompt") or 0.0
    n_scored = n_patients * sum(len(GRID_DEF[g]) for g in grids)
    gpp = (r.get("generated_free") or {}).get("seconds_per_prompt") or 0.0
    return {"scoring_h": spp * n_scored / 3600, "generation_h": gpp * n_gen_patients * len(GEN_GRID) / 3600}


def merge_checks(*checks) -> dict:
    """The results of several engineering checks (each: model id -> result), in the order they were run; a later check
    replaces an earlier one for every model it ran (Section 10)."""
    out = {}
    for c in checks:
        out.update(c)
    return out


def decisions_from_check(results: dict, n_patients: int, n_gen_patients: int) -> dict:
    """Rules (i) to (iii) of Section 10 applied to the engineering-check results (model id -> result).
    n_patients: fresh patients over both datasets; n_gen_patients: generation subsample at 150 per class, both datasets."""
    out = {"not_run": [], "grids_not_run": {}, "gen_per_class": GEN_PER_CLASS}
    for mid in MODELS_V5:
        r = results.get(mid) or {}
        ok = r.get("grids_ok") or {}
        if not r or not ok.get("yn") or not (r.get("single_token_words") or {}).get("Yes/No"):
            out["not_run"].append(mid)
            continue
        drop = [g for g in OPTIONAL_GRIDS if not ok.get(g)]
        if drop:
            out["grids_not_run"][MODELS_V5[mid]["tag"]] = drop
    run = [m for m in MODELS_V5 if m not in out["not_run"]]
    gen_h = sum(projected_hours(results[m], n_patients, n_gen_patients)["generation_h"] for m in run)
    out["projected_generation_h_at_150"] = round(gen_h, 2)
    if gen_h > GEN_HOURS_LIMIT:
        out["gen_per_class"] = 100
    return out


def access_problems(model_ids, token=None, cache_dir=None) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for mid in model_ids:
        try:
            hf_hub_download(mid, "config.json", revision=MODELS_V5[mid]["revision"], token=token, cache_dir=cache_dir)
        except Exception as e:
            out[mid] = f"{type(e).__name__}: " + (str(e).strip().splitlines() or [""])[0][:300]
    return out


ACCESS_HELP = W4.ACCESS_HELP
parse_result = W4.parse_result


# =============================================================================================================
# 6. Fake model for the CPU tests (never used on Kaggle)
# =============================================================================================================
class _MockTokV5(L._MockTok):
    def __init__(self):
        super().__init__()
        self.vocab.update({"True": 9, "ĠTrue": 10, "False": 11, "ĠFalse": 12})


class MockV5(L.MockBackend):
    """Deterministic fake model that reads every prompt of the fifth study. Its belief about the alarming outcome is
    moved by the evidence with `alarm_gain` / `safe_gain` (an outcome asymmetry if they differ). On the scale of the
    answer words it adds a static preference `prior` for the second word ('No', 'False'; for letters `letter_prior` for
    'B'); whenever any evidence block is shown, that preference is multiplied by `card_prior_factor` (a card that
    changes the prior). A real card or probability (not the placebo) also pulls the answer by `pull` toward the second
    word (an asymmetry in updating, W = -2 pull). `prob_gain` sets how much the probability moves the belief, and
    `dose_power` how the numbers card's effect grows with its impacts (0: the size of the impacts is ignored)."""
    name = "mock"
    _IMP = re.compile(r"\(impact ([+-]?\d+\.\d+)\)")
    _PROB = re.compile(r"its estimated risk of [^\n]* for this patient is (\d+)%\.")

    def __init__(self, prior=0.0, letter_prior=0.0, card_prior_factor=1.0, pull=0.0, prob_gain=1.0, dose_power=1.0,
                 gen_flip=0.0, **kw):
        super().__init__([s["q_reversed"] for s in L.SPEC.values()] + [q["reversed"] for q in W4.PARA_Q.values()], **kw)
        self.tokenizer = _MockTokV5()
        self.variants, self.strings = L.find_word_variants(self.tokenizer)
        self.prior, self.letter_prior, self.cpf, self.pull = float(prior), float(letter_prior), float(card_prior_factor), float(pull)
        self.prob_gain, self.dose_power, self.gen_flip = float(prob_gain), float(dose_power), float(gen_flip)
        self.statements = {}
        for o in L.CHOICE_OPTIONS.values():
            self.statements[o["alarm"]] = True
            self.statements[o["safe"]] = False

    def logodds(self, p):
        hdr = L.PATIENT_HEADER if L.PATIENT_HEADER in p else W4.PARA_PATIENT_HEADER
        head, part = p.rsplit(hdr, 1)
        key = part.split("\n\n")[0]
        base = 3.0 * (L._u(key, self.seed) - 0.5) + self.lean
        if key in self.truth:
            base += self.signal * (2 * self.truth[key] - 1)
        shown = "Evidence from a machine-learning model (XGBoost)" in part
        real = False
        s = 0.0
        m = self._PROB.search(part)
        if m:
            q = min(max(int(m.group(1)), 1), 99) / 100
            s, real = self.prob_gain * math.log(q / (1 - q)), True
        else:
            imps = [float(x) for x in self._IMP.findall(part)]
            if imps and any(abs(x) > 0 for x in imps):
                real = True
                tot = sum(imps)
                if ": effect on the risk of" in part and self.dose_power != 1.0:
                    tot = math.copysign(abs(tot) ** self.dose_power, tot)
                s = tot
        z = self.scale * (base + (self.alarm_gain * s if s > 0 else self.safe_gain * s))
        if W4.TF_INSTRUCTION in p:
            st = re.findall(re.escape(W4.STATEMENT) + r"(.*)\n" + re.escape(W4.TF_INSTRUCTION), p)[-1]
            pa, words, prior = self.statements[st], TF, self.prior
        elif L.CHOICE_INSTRUCTION in p:
            opt_a = re.findall(r"\nA: (.*)\nB: ", p)[-1]
            pa, words, prior = self.statements[opt_a], AB, self.letter_prior
        else:
            q = re.findall(r"Question: (.*)\n(?:" + re.escape(L.ANSWER_INSTRUCTION) + "|" + re.escape(W4.PARA_INSTRUCTION) + ")",
                           p)[-1]
            pa, words, prior = q not in self.reversed_q, YN, self.prior
        zp = z if pa else -z
        zp -= prior * (self.cpf if shown else 1.0)
        if real and words != AB:
            zp -= self.pull
        mass = 0.80 + 0.19 * L._u(key, "m")
        return zp, words, mass

    def word_logprobs(self, prompts, words=YN):
        wanted = self.wanted_ids(words)
        ids_of = {YN: (2, 1, 4, 3), TF: (10, 9, 12, 11), AB: (6, 5, 8, 7)}
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
            h = L._u(p[-200:], "g")
            out.append(f"**{word}**" if h < 0.5 else (f"Based on the data, the answer is {word}." if h < 0.9
                                                      else "I cannot tell from this information."))
        return out


# =============================================================================================================
# 7. Analysis (PREREGISTRATION_V5.md, Sections 7 and 8)
# =============================================================================================================
TEMPLATE_SD = {"threshold": {"a": 0.597, "ap": 0.362}, "neutral": {"a": 0.453, "ap": 0.382}}   # Study 4, Section 8
SUPPORT_WINDOW, SUPPORT_MIN, DSTAR_MIN = 1.0, 30, 0.5
R15_CELLS = ("qwen3-instruct|physionet2012", "qwen3-instruct|physionet2019")
R16_CELLS = {"falcon3-7b|physionet2012": 1, "falcon3-7b|physionet2019": 1, "yi15-9b|physionet2019": 1,
             "granite3-instruct|physionet2012": -1}


def slice_v5(df, framing, shot="A", conds=COND4):
    return df[(df["framing"] == framing) & (df["shot_set"] == shot) & df["cond"].isin(conds)]


def _mass_median(r):
    return float(r["yn_mass"].astype(float).fillna(0.0).median()) if len(r) else float("nan")


def wide_pair(df, pair, base="C0", ev=("C2", "C4")):
    """Per patient with valid readings of `base` and both evidence conditions in both framings of `pair`: the arrays
    of each framing (l0 = the baseline reading, la / ls = readings with the alarm- / safe-pointing evidence, t = +1 if
    the true evidence points to the alarming outcome), with z and y. -> (ids, x1, x2) or None."""
    out = []
    for fr in PAIRS_V5[pair]:
        d = df[(df["framing"] == fr) & (df["shot_set"] == "A") & (df["valid"] == 1) & df["cond"].isin((base,) + ev)]
        w = d.pivot_table(index="id", columns="cond", values=["lo_alarm", "card_dir", "y", "xgb_p"], aggfunc="first")
        need = [("lo_alarm", c) for c in (base,) + ev] + [("card_dir", ev[0])]
        if not all(k in w.columns for k in need):
            return None
        out.append(w.dropna(subset=need))
    ids = sorted(out[0].index.intersection(out[1].index))
    if len(ids) < 10:
        return None
    xs = []
    for w in out:
        w = w.loc[ids]
        cd = w[("card_dir", ev[0])].astype(int).to_numpy()
        e0, e1 = (w[("lo_alarm", c)].to_numpy(float) for c in ev)
        p = np.clip(w[("xgb_p", base)].to_numpy(float), 1e-7, 1 - 1e-7)
        xs.append({"l0": w[("lo_alarm", base)].to_numpy(float), "la": np.where(cd == 1, e0, e1),
                   "ls": np.where(cd == 1, e1, e0), "t": np.where(cd == 1, 1.0, -1.0), "z": np.log(p / (1 - p)),
                   "y": w[("y", base)].to_numpy(float)})
    return ids, xs[0], xs[1]


def _solve(X, W, d):
    if X.ndim == 2:
        A = np.einsum("bn,nk,nl->bkl", W, X, X); r = np.einsum("bn,nk,n->bk", W, X, d)
    else:
        A = np.einsum("bn,bnk,bnl->bkl", W, X, X); r = np.einsum("bn,bnk,n->bk", W, X, d)
    if not np.all(np.isfinite(A)):
        return np.full(r.shape, np.nan)
    try:
        return np.linalg.solve(A, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(A[i], r[i], rcond=None)[0] for i in range(len(A))])


def _wstd(X, W):
    n = W.sum(1, keepdims=True)
    m = (W * X).sum(1, keepdims=True) / n
    return np.sqrt((W * (X - m) ** 2).sum(1, keepdims=True) / n)


def stat_a(x, W, shift=None):
    """a (Deviation D1) with the baseline shifted by `shift` (one value per resample) before the odd polynomial."""
    l0, la, ls, t, z, y = (x[k] for k in ("l0", "la", "ls", "t", "z", "y"))
    n = W.sum(1, keepdims=True)
    zs = (z[None, :] - (W * z).sum(1, keepdims=True) / n) / _wstd(np.broadcast_to(z, W.shape), W)
    Z = np.stack([np.ones_like(zs), zs, zs ** 2, zs ** 3, np.broadcast_to(y, zs.shape)], -1)
    e = np.einsum("bnk,bk->bn", Z, _solve(Z, W, l0))
    s = _wstd(e, W)
    if shift is not None:
        e = e - shift[:, None]
    return _solve(np.stack([np.ones_like(e), e, (e / s) ** 3, np.broadcast_to(t, e.shape)], -1), W,
                  (la - l0) + (ls - l0))[:, 0]


def stat_ap(x, W, shift=None):
    l0, la, ls, t = (x[k] for k in ("l0", "la", "ls", "t"))
    L0 = np.broadcast_to(l0, W.shape)
    s = _wstd(L0, W)
    if shift is not None:
        L0 = L0 - shift[:, None]
    return _solve(np.stack([np.ones_like(L0), L0, (L0 / s) ** 3, np.broadcast_to(t, L0.shape)], -1), W,
                  (la - l0) + (ls - l0))[:, 0]


def wmedian(v, W):
    o = np.argsort(v, kind="stable")
    cum = np.cumsum(W[:, o], axis=1)
    return v[o][np.argmax(cum >= W.sum(1, keepdims=True) / 2, axis=1)]


def _boot(n, fns, n_boot=N_BOOT, seed=L.BOOT_SEED, chunk=500):
    rng = np.random.default_rng(seed)
    one = np.ones((1, n))
    est = {k: float(f(one)[0]) for k, f in fns.items()}
    draws = {k: [] for k in fns}
    for start in range(0, n_boot, chunk):
        Wt = rng.multinomial(n, np.full(n, 1.0 / n), size=min(chunk, n_boot - start)).astype(float)
        for k, f in fns.items():
            draws[k].append(f(Wt))
    return est, {k: np.concatenate(v) for k, v in draws.items()}


def _p_boot(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))) if len(b) else float("nan")


def _p_norm(est, se, sd):
    z = est / math.sqrt(se * se + sd * sd)
    return float(math.erfc(abs(z) / math.sqrt(2)))


def _part(est, dr, k_a, k_ap, reading):
    """One part (O or W) in both versions: estimates, bootstrap intervals, the joint test over patients (the
    registered test, Section 8), and the same test with the Study 4 template variance added (registered secondary)."""
    ea, eap = est[k_a], est[k_ap]
    ba, bap = dr[k_a], dr[k_ap]
    sa, sap = float(np.nanstd(ba)), float(np.nanstd(bap))
    same = bool(np.isfinite(ea) and np.isfinite(eap) and np.sign(ea) == np.sign(eap))
    p_pat = max(_p_boot(ba), _p_boot(bap))
    sd = TEMPLATE_SD[reading]
    p_tpl = max(_p_norm(ea, sa, sd["a"]), _p_norm(eap, sap, sd["ap"])) if same else 1.0
    ci = lambda b: [float(v) for v in np.nanpercentile(b, [2.5, 97.5])]
    return {"a": ea, "ap": eap, "a_ci": ci(ba), "ap_ci": ci(bap), "se_a": sa, "se_ap": sap,
            "sign": int(np.sign(ea)) if same else 0, "p": p_pat if same else 1.0, "p_template": p_tpl,
            "conclusive": bool(same and p_pat < 0.05), "conclusive_template": bool(same and p_tpl < 0.05)}


def analyse_pair(df, pair, base="C0", ev=("C2", "C4"), n_boot=N_BOOT):
    """O and W of one pair at the threshold and at the neutral point of the baseline `base` (C0 or CP), the static
    lean, the patients near both reading points, and the change a card would have to make to the prior (delta*)."""
    got = wide_pair(df, pair, base, ev)
    if got is None:
        return None
    ids, x1, x2 = got
    n = len(ids)
    lean_i = (x1["l0"] - x2["l0"]) / 2

    def lam(Wt):
        return wmedian(lean_i, Wt)

    fns = {"lam": lam,
           "a1": lambda Wt: stat_a(x1, Wt), "a2": lambda Wt: stat_a(x2, Wt),
           "ap1": lambda Wt: stat_ap(x1, Wt), "ap2": lambda Wt: stat_ap(x2, Wt),
           "na1": lambda Wt: stat_a(x1, Wt, lam(Wt)), "na2": lambda Wt: stat_a(x2, Wt, -lam(Wt)),
           "nap1": lambda Wt: stat_ap(x1, Wt, lam(Wt)), "nap2": lambda Wt: stat_ap(x2, Wt, -lam(Wt))}
    est, dr = _boot(n, fns, n_boot)
    out = {"n": n, "lean": est["lam"], "lean_ci": [float(v) for v in np.percentile(dr["lam"], [2.5, 97.5])]}
    for reading, pre in (("threshold", ""), ("neutral", "n")):
        res = {}
        for part, sg in (("O", 1), ("W", -1)):
            for v in ("a", "ap"):
                est[f"{part}{v}"] = (est[f"{pre}{v}1"] + sg * est[f"{pre}{v}2"]) / 2
                dr[f"{part}{v}"] = (dr[f"{pre}{v}1"] + sg * dr[f"{pre}{v}2"]) / 2
            res[part] = _part(est, dr, f"{part}a", f"{part}ap", reading)
        w = res["W"]
        res["delta_star"] = (abs(min(w["a"], w["ap"], key=abs)) / (2 * abs(est["lam"]))
                             if w["sign"] != 0 and est["lam"] != 0 else None)
        out[reading] = res
    lam0 = est["lam"]
    out["support"] = {"threshold": int(min((abs(x1["l0"]) <= SUPPORT_WINDOW).sum(), (abs(x2["l0"]) <= SUPPORT_WINDOW).sum())),
                      "neutral": int(min((abs(x1["l0"] - lam0) <= SUPPORT_WINDOW).sum(),
                                         (abs(x2["l0"] + lam0) <= SUPPORT_WINDOW).sum()))}
    out["readable"] = {r: bool(out[r]["W"]["conclusive"] and out["support"][r] >= SUPPORT_MIN and
                               (out[r]["delta_star"] or 0) >= DSTAR_MIN) for r in ("threshold", "neutral")}
    return out


def lean_change(df, pair="yn", n_boot=N_BOOT):
    """R15: the static lean with the placebo card minus the lean without a card, on the patients with valid C0 and CP
    readings in both framings, with a paired bootstrap; and the placebo's own shift of the alarm log-odds averaged
    over the framings (an outcome effect of the placebo)."""
    w = []
    for fr in PAIRS_V5[pair]:
        d = df[(df["framing"] == fr) & (df["shot_set"] == "A") & (df["valid"] == 1) & df["cond"].isin(("C0", "CP"))]
        p = d.pivot_table(index="id", columns="cond", values="lo_alarm", aggfunc="first")
        if not all(c in p.columns for c in ("C0", "CP")):
            return None
        w.append(p.dropna(subset=["C0", "CP"]))
    ids = sorted(w[0].index.intersection(w[1].index))
    if len(ids) < 10:
        return None
    l0 = (w[0].loc[ids, "C0"].to_numpy(float) - w[1].loc[ids, "C0"].to_numpy(float)) / 2
    lp = (w[0].loc[ids, "CP"].to_numpy(float) - w[1].loc[ids, "CP"].to_numpy(float)) / 2
    sh = ((w[0].loc[ids, "CP"] - w[0].loc[ids, "C0"]).to_numpy(float) + (w[1].loc[ids, "CP"] - w[1].loc[ids, "C0"]).to_numpy(float)) / 2
    fns = {"l0": lambda Wt: wmedian(l0, Wt), "lp": lambda Wt: wmedian(lp, Wt),
           "d": lambda Wt: wmedian(lp, Wt) - wmedian(l0, Wt), "shift": lambda Wt: (Wt @ sh) / Wt.sum(1)}
    est, dr = _boot(len(ids), fns, n_boot)
    ci = lambda b: [float(v) for v in np.percentile(b, [2.5, 97.5])]
    return {"n": len(ids), "lean_c0": est["l0"], "lean_cp": est["lp"], "delta": est["d"], "delta_ci": ci(dr["d"]),
            "p": _p_boot(dr["d"]), "placebo_outcome_shift": est["shift"], "placebo_outcome_shift_ci": ci(dr["shift"])}


def validity_v5(df) -> dict:
    """Section 8: a cell is valid if the Yes/No pair has a median answer-word mass of at least 0.5 in both framings
    (C0, CP, C2, C4) and the no-card accuracy in the alarm framing is at least 0.55 (the rule of every earlier study).
    Another pair is used in a valid cell if each of its framings has a median mass of at least 0.5, a positive mean
    movement between the alarm- and the safe-pointing card, and a positive Spearman correlation of its C0 readings with
    those of the alarm question. The scaled numbers cards and the probability evidence are used if their median mass is
    at least 0.5."""
    from scipy.stats import spearmanr
    al, rv = slice_v5(df, "alarm"), slice_v5(df, "reversed")
    c0 = al[(al["cond"] == "C0") & (al["valid"] == 1)]
    acc = float(c0["correct"].mean()) if len(c0) else float("nan")
    out = {"yn": {"alarm_median_mass": _mass_median(al), "reversed_median_mass": _mass_median(rv), "c0_accuracy": acc}}
    out["yn"]["passes"] = bool(out["yn"]["alarm_median_mass"] >= L.VALIDITY_MIN_MASS and
                               out["yn"]["reversed_median_mass"] >= L.VALIDITY_MIN_MASS and acc >= L.VALIDITY_MIN_C0_ACC)
    ref = c0.set_index("id")["lo_alarm"].astype(float)
    for pair in ("pa", "tf", "ab"):
        res, ok = {}, out["yn"]["passes"]
        for fr in PAIRS_V5[pair]:
            own = slice_v5(df, fr)
            if not len(own):
                ok = False
                continue
            v = own[own["valid"] == 1]
            pv = v.pivot_table(index="id", columns="cond", values=["lo_alarm", "card_dir"], aggfunc="first")
            mv = float("nan")
            if ("lo_alarm", "C2") in pv.columns and ("lo_alarm", "C4") in pv.columns:
                pv = pv.dropna(subset=[("lo_alarm", "C2"), ("lo_alarm", "C4"), ("card_dir", "C2")])
                cd = pv[("card_dir", "C2")].astype(int).to_numpy()
                la = np.where(cd == 1, pv[("lo_alarm", "C2")], pv[("lo_alarm", "C4")])
                ls = np.where(cd == 1, pv[("lo_alarm", "C4")], pv[("lo_alarm", "C2")])
                mv = float(np.mean(la - ls)) if len(la) else float("nan")
            c = v[v["cond"] == "C0"].set_index("id")["lo_alarm"].astype(float)
            ii = c.index.intersection(ref.index)
            rho = float(spearmanr(c.loc[ii], ref.loc[ii]).correlation) if len(ii) >= 10 else float("nan")
            res[fr] = {"median_mass": _mass_median(own), "movement": mv, "c0_spearman_with_alarm": rho}
            ok = ok and res[fr]["median_mass"] >= L.VALIDITY_MIN_MASS and mv > 0 and rho > 0
        res["passes"] = bool(ok)
        out[pair] = res
    for name, (fr, shots, conds) in {"dose": ("alarm", tuple(DOSE_SETS), ("C2", "C4")),
                                     "prob": (None, ("A",), ("CQ2", "CQ4"))}.items():
        d = df[df["shot_set"].isin(shots) & df["cond"].isin(conds)]
        if fr:
            d = d[d["framing"] == fr]
        out[name] = {"median_mass": _mass_median(d), "passes": bool(len(d) and out["yn"]["passes"] and
                                                                    _mass_median(d) >= L.VALIDITY_MIN_MASS)}
    return out


def dose_movement(df, n_boot=N_BOOT):
    """Q19: in the alarm framing, the mean movement m = mean(la - ls) with the registered card and the numbers card at
    each scale, on the patients with valid C2 and C4 readings with all of them; dm = m(x2) - m(x0.5)."""
    mv = {}
    for s in ("A",) + tuple(DOSE_SETS):
        d = df[(df["framing"] == "alarm") & (df["shot_set"] == s) & (df["valid"] == 1) & df["cond"].isin(("C2", "C4"))]
        p = d.pivot_table(index="id", columns="cond", values=["lo_alarm", "card_dir"], aggfunc="first")
        if not all(k in p.columns for k in (("lo_alarm", "C2"), ("lo_alarm", "C4"), ("card_dir", "C2"))):
            return None
        p = p.dropna(subset=[("lo_alarm", "C2"), ("lo_alarm", "C4"), ("card_dir", "C2")])
        cd = p[("card_dir", "C2")].astype(int).to_numpy()
        mv[s] = pd.Series(np.where(cd == 1, p[("lo_alarm", "C2")] - p[("lo_alarm", "C4")],
                                   p[("lo_alarm", "C4")] - p[("lo_alarm", "C2")]), index=p.index)
    ids = None
    for v in mv.values():
        ids = v.index if ids is None else ids.intersection(v.index)
    ids = sorted(ids)
    if len(ids) < 10:
        return None
    arr = {s: v.loc[ids].to_numpy(float) for s, v in mv.items()}
    fns = {s: (lambda Wt, x=x: (Wt @ x) / Wt.sum(1)) for s, x in arr.items()}
    fns["dm"] = lambda Wt: (Wt @ (arr["Anum2"] - arr["Anum05"])) / Wt.sum(1)
    est, dr = _boot(len(ids), fns, n_boot)
    ci = lambda b: [float(v) for v in np.percentile(b, [2.5, 97.5])]
    return {"n": len(ids), "movement": {s: {"est": est[s], "ci": ci(dr[s])} for s in arr},
            "dm": {"est": est["dm"], "ci": ci(dr["dm"]), "p": _p_boot(dr["dm"])}}


def decision_EW(gen: pd.DataFrame, n_boot=N_BOOT):
    """R17: the decision-level polarity effect from freely generated answers, E_W = (E_alarm - E_reversed) / 2 with
    E = added false alarms - added missed alarms (ler_v4.decision_E), on the patients whose six answers were all
    parsed, with a paired bootstrap over patients."""
    g = gen[gen["parsed"].notna()].copy()
    yes = g["parsed"] == "Yes"
    g["pred_alarm"] = np.where(g["framing"] == "alarm", yes, ~yes).astype(int)
    full = g.groupby("id")["key"].count()
    ids = sorted(full[full == len(GEN_GRID)].index)
    if len(ids) < 10:
        return {"n": len(ids), "E_W": float("nan"), "ci": [float("nan")] * 2, "sign": 0}
    g = g[g["id"].isin(ids)]
    per = {}
    for fr in PAIRS_V5["yn"]:
        w = g[g["framing"] == fr].pivot_table(index="id", columns="cond", values=["pred_alarm", "card_dir", "y"],
                                              aggfunc="first").loc[ids]
        cd = w[("card_dir", "C2")].astype(int).to_numpy()
        per[fr] = {"y": w[("y", "C0")].astype(int).to_numpy(), "c0": w[("pred_alarm", "C0")].astype(int).to_numpy(),
                   "pa": np.where(cd == 1, w[("pred_alarm", "C2")], w[("pred_alarm", "C4")]).astype(int),
                   "ps": np.where(cd == 1, w[("pred_alarm", "C4")], w[("pred_alarm", "C2")]).astype(int)}

    def E(Wt, x):
        n0, n1 = Wt @ (x["y"] == 0), Wt @ (x["y"] == 1)
        fa = (Wt @ ((x["y"] == 0) * x["pa"]) - Wt @ ((x["y"] == 0) * x["c0"])) / n0
        miss = (Wt @ ((x["y"] == 1) * (1 - x["ps"])) - Wt @ ((x["y"] == 1) * (1 - x["c0"]))) / n1
        return fa - miss

    est, dr = _boot(len(ids), {"EW": lambda Wt: (E(Wt, per["alarm"]) - E(Wt, per["reversed"])) / 2}, n_boot)
    ci = [float(v) for v in np.nanpercentile(dr["EW"], [2.5, 97.5])]
    return {"n": len(ids), "E_W": est["EW"], "ci": ci, "sign": int(np.sign(est["EW"])) if (ci[0] > 0 or ci[1] < 0) else 0}


def analyse_cell_v5(df, gen=None, n_boot=N_BOOT) -> dict:
    """Everything of one model x dataset cell."""
    val = validity_v5(df)
    out = {"validity": val, "lean_change": lean_change(df, "yn", n_boot), "pairs": {}}
    for pair in ("yn", "pa", "tf", "ab"):
        if pair != "yn" and not val[pair]["passes"]:
            continue
        out["pairs"][pair] = {"C0": analyse_pair(df, pair, "C0", n_boot=n_boot), "CP": analyse_pair(df, pair, "CP", n_boot=n_boot)}
    if val["dose"]["passes"]:
        out["dose"] = dose_movement(df, n_boot)
    if val["prob"]["passes"]:
        out["prob"] = {"C0": analyse_pair(df, "yn", "C0", ("CQ2", "CQ4"), n_boot), "CP": analyse_pair(df, "yn", "CP", ("CQ2", "CQ4"), n_boot)}
    if gen is not None and len(gen):
        out["generated"] = decision_EW(gen, n_boot)
        out["generated"]["parsed_share"] = float(gen["parsed"].notna().mean())
    return out


def _holm(ps):
    ps = list(ps)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [1.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def confirmatory_v5(cells: dict) -> dict:
    """The decision rules of Section 7. cells: 'tag|dataset' -> analyse_cell_v5 output."""
    out = {}
    # R15: in Qwen3-8B, the placebo card weakens the preference for 'No' (delta > 0) in both datasets
    r15 = {}
    for k in R15_CELLS:
        c = cells.get(k)
        ok = c is not None and c["validity"]["yn"]["passes"] and c["lean_change"] is not None
        r15[k] = {"testable": ok, **({"delta": c["lean_change"]["delta"], "p": c["lean_change"]["p"]} if ok else {})}
    adj = _holm([r15[k]["p"] if r15[k]["testable"] else 1.0 for k in R15_CELLS])
    for k, a in zip(R15_CELLS, adj):
        r15[k]["p_holm"] = a
        r15[k]["supported"] = bool(r15[k]["testable"] and r15[k]["delta"] > 0 and a < 0.05)
    out["R15"] = {"cells": r15, "supported": all(v["supported"] for v in r15.values())}
    # R16: W against the placebo card at its neutral point (registered prompt, joint test over patients, Holm over the
    # four cells) has the predicted sign
    r16 = {}
    for k, s in R16_CELLS.items():
        c = cells.get(k)
        w = (((c or {}).get("pairs") or {}).get("yn") or {}).get("CP")
        ok = bool(c is not None and c["validity"]["yn"]["passes"] and w is not None)
        r16[k] = {"predicted": s, "testable": ok}
        if ok:
            wn = w["neutral"]["W"]
            r16[k].update(sign=wn["sign"], p=wn["p"], a=wn["a"], ap=wn["ap"])
    adj = _holm([r16[k]["p"] if r16[k]["testable"] else 1.0 for k in R16_CELLS])
    for k, a in zip(R16_CELLS, adj):
        v = r16[k]
        v["p_holm"] = a
        v["supported"] = bool(v["testable"] and v["sign"] == v["predicted"] and a < 0.05)
        v["opposite"] = bool(v["testable"] and v["sign"] == -v["predicted"] and a < 0.05)
    n_sup = sum(v["supported"] for v in r16.values())
    out["R16"] = {"cells": r16, "n_supported": n_sup,
                  "supported": bool(n_sup >= 3 and not any(v["opposite"] for v in r16.values()))}
    # R17: the decision-level polarity effect of generated answers has the predicted sign (interval excludes 0)
    r17 = {}
    for k, s in R16_CELLS.items():
        g = (cells.get(k) or {}).get("generated")
        ok = bool(g is not None and g["n"] >= 100 and cells[k]["validity"]["yn"]["passes"])
        r17[k] = {"predicted": s, "testable": ok, **({"E_W": g["E_W"], "ci": g["ci"], "n": g["n"]} if g else {})}
        r17[k]["supported"] = bool(ok and g["sign"] == s)
    n17 = sum(v["supported"] for v in r17.values())
    out["R17"] = {"cells": r17, "n_supported": n17, "supported": bool(n17 >= 3)}
    return out



# =============================================================================================================
# 8. Registered questions Q21 and Q22, and the imputation bound (PREREGISTRATION_V5.md, Section 7)
# =============================================================================================================
# Q21b: the sign of the conclusive O and W of the registered Yes/No pair against C0 on the earlier patients (Studies 1
# to 3; 0 = not conclusive; None = the cell failed the validity rule), from the paper's analysis files.
EARLIER_SIGNS = {
    "llama-instruct|physionet2012": {"threshold": {"W": -1, "O": -1}, "neutral": {"W": 0, "O": 1}},
    "llama-instruct|physionet2019": {"threshold": {"W": 0, "O": -1}, "neutral": {"W": 0, "O": -1}},
    "qwen-instruct|physionet2012": {"threshold": {"W": 1, "O": -1}, "neutral": {"W": 1, "O": -1}},
    "qwen-instruct|physionet2019": {"threshold": {"W": -1, "O": 0}, "neutral": {"W": 0, "O": 0}},
    "qwen3-instruct|physionet2012": {"threshold": {"W": -1, "O": -1}, "neutral": {"W": 1, "O": -1}},
    "qwen3-instruct|physionet2019": {"threshold": {"W": 0, "O": -1}, "neutral": {"W": 1, "O": 0}},
    "olmo3-instruct|physionet2012": {"threshold": {"W": -1, "O": 1}, "neutral": {"W": 0, "O": 1}},
    "olmo3-instruct|physionet2019": {"threshold": {"W": -1, "O": 0}, "neutral": {"W": -1, "O": 1}},
    "granite3-instruct|physionet2012": {"threshold": {"W": -1, "O": 1}, "neutral": {"W": -1, "O": 1}},
    "granite3-instruct|physionet2019": None,
    "falcon3-7b|physionet2012": {"threshold": {"W": 1, "O": 1}, "neutral": {"W": 1, "O": 0}},
    "falcon3-7b|physionet2019": {"threshold": {"W": 1, "O": 1}, "neutral": {"W": 1, "O": 1}},
    "yi15-9b|physionet2012": {"threshold": {"W": 1, "O": -1}, "neutral": {"W": 1, "O": -1}},
    "yi15-9b|physionet2019": {"threshold": {"W": 1, "O": -1}, "neutral": {"W": 1, "O": -1}},
}


def _csign(r):
    return int(r["sign"]) if r and r.get("conclusive") else 0


def earlier_agreement(cells: dict) -> dict:
    """Q21b: O and W of the Yes/No pair against C0 on the fresh patients, against the same cell on the earlier patients."""
    out, tally = {}, {}
    for k, old in EARLIER_SIGNS.items():
        c = cells.get(k)
        yn = (((c or {}).get("pairs") or {}).get("yn") or {}).get("C0")
        if c is None or old is None or yn is None or not c["validity"]["yn"]["passes"]:
            out[k] = {"compared": False, "earlier_valid": old is not None, "fresh_valid": bool(c and c["validity"]["yn"]["passes"])}
            continue
        row = {"compared": True}
        for rd in ("threshold", "neutral"):
            for part in ("W", "O"):
                a, b = old[rd][part], _csign(yn[rd][part])
                row[f"{rd}|{part}"] = {"earlier": a, "fresh": b}
                if a != 0:
                    t = tally.setdefault(f"{rd}|{part}", {"earlier_conclusive": 0, "same": 0, "opposite": 0, "not_conclusive": 0})
                    t["earlier_conclusive"] += 1
                    t["same" if b == a else "opposite" if b == -a else "not_conclusive"] += 1
        out[k] = row
    return {"cells": out, "tally": tally}


def descriptives_v5(df: pd.DataFrame) -> dict:
    """Q21e for one cell: per framing and set, the invalid records, the median answer-word mass, the share of imputed
    readings, the accuracy by condition and the share of answers that follow the evidence (C2, C4, CQ2, CQ4)."""
    out = {}
    for (fr, shot), g in df.groupby(["framing", "shot_set"]):
        gv = g[g["valid"] == 1]
        ev = gv[gv["card_dir"].notna()]
        out[f"{fr}|{shot}"] = {
            "n": int(len(g)), "invalid": int((g["valid"] == 0).sum()), "median_mass": _mass_median(g),
            "imputed_share": float((g["n_imputed"] > 0).mean()),
            "accuracy": {c: float(x["correct"].mean()) for c, x in gv.groupby("cond")},
            "follow": {c: float((x["pred_alarm"].astype(int) == x["card_dir"].astype(int)).mean()) for c, x in ev.groupby("cond")}}
    return out


def _full_ids(gen: pd.DataFrame) -> list:
    g = gen[gen["parsed"].notna()]
    n = g.groupby("id")["key"].count()
    return sorted(n[n == len(GEN_GRID)].index)


def generated_summary_v5(gen: pd.DataFrame, rec: pd.DataFrame, n_boot=N_BOOT) -> dict:
    """Q21d for one cell: the parsed share, the agreement of the parsed answers with the first-token reading of the
    same prompts ('Yes' when its log-odds against 'No' is above 0), and E_W from the generated answers (R17) and from
    the first-token reading of the same patients."""
    base = rec[rec["shot_set"] == "A"][["key", "valid", "lp_yes", "lp_no"]]
    m = gen.merge(base, on="key", how="left")
    parsed = m["parsed"].notna()
    ok = parsed & (m["valid"] == 1)
    first = np.where(m["lp_yes"].astype(float) > m["lp_no"].astype(float), "Yes", "No")
    agree = float((m.loc[ok, "parsed"] == first[ok.to_numpy()]).mean()) if ok.any() else float("nan")
    ids = _full_ids(gen)
    ft = m[m["id"].isin(ids)].copy()
    ft["parsed"] = np.where(ft["valid"] == 1, np.where(ft["lp_yes"].astype(float) > ft["lp_no"].astype(float), "Yes", "No"), None)
    ft.loc[ft["valid"] != 1, "parsed"] = None
    return {"n": int(len(gen)), "parsed_share": float(parsed.mean()), "agreement": agree, "n_full_patients": len(ids),
            "E_W_generated": decision_EW(gen, n_boot), "E_W_first_token": decision_EW(ft, n_boot)}


def r16_template(cells: dict) -> dict:
    """Q22: R16 recomputed with the test that adds the template variance of the fourth study (no decision rule)."""
    r = {}
    for k, s in R16_CELLS.items():
        c = cells.get(k)
        w = (((c or {}).get("pairs") or {}).get("yn") or {}).get("CP")
        ok = bool(c is not None and c["validity"]["yn"]["passes"] and w is not None)
        r[k] = {"predicted": s, "testable": ok}
        if ok:
            r[k].update(sign=w["neutral"]["W"]["sign"], p=w["neutral"]["W"]["p_template"])
    adj = _holm([r[k]["p"] if r[k]["testable"] else 1.0 for k in R16_CELLS])
    for k, a in zip(R16_CELLS, adj):
        v = r[k]
        v["p_holm"] = a
        v["supported"] = bool(v["testable"] and v["sign"] == v["predicted"] and a < 0.05)
        v["opposite"] = bool(v["testable"] and v["sign"] == -v["predicted"] and a < 0.05)
    n = sum(v["supported"] for v in r.values())
    return {"cells": r, "n_supported": n, "would_be_supported": bool(n >= 3 and not any(v["opposite"] for v in r.values()))}


def template_counts(cells: dict) -> dict:
    """Q22: for every pair, baseline and reading point, the conclusive readings of W and O and how many of them stay
    conclusive with the template variance added."""
    out = {}
    for c in cells.values():
        for pair, by in (c.get("pairs") or {}).items():
            for base, res in by.items():
                if not res:
                    continue
                for rd in ("threshold", "neutral"):
                    for part in ("W", "O"):
                        x = res[rd][part]
                        t = out.setdefault(f"{pair}|{base}|{rd}|{part}", {"conclusive": 0, "stay_conclusive": 0})
                        t["conclusive"] += int(x["conclusive"])
                        t["stay_conclusive"] += int(x["conclusive"] and x["conclusive_template"])
    return out


def halve_less_likely_v5(df: pd.DataFrame) -> pd.DataFrame:
    """The imputation bound of Deviation D5 (Q21f): in every record with an imputed variant, the probability of the
    less likely answer word is halved, and the record is scored again."""
    df = df.copy()
    m = df["n_imputed"] > 0
    lower = df["lp_yes"] < df["lp_no"]
    df.loc[m & lower, "lp_yes"] -= np.log(2)
    df.loc[m & ~lower, "lp_no"] -= np.log(2)
    keep = ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir", "xgb_p", "n_tokens", "text_sha")
    rows = [finish_record_v5({k: r[k] for k in keep if k in r},
                             {"lp_yes": r["lp_yes"], "lp_no": r["lp_no"], "n_imputed": r["n_imputed"]}, r["framing"])
            for r in df[m].to_dict("records")]
    out = pd.concat([df[~m], pd.DataFrame(rows)], ignore_index=True)
    out["card_dir"] = out["card_dir"].astype(float)
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "engcheck":
        a = sys.argv + ["None"]
        engineering_check_v5(a[2], a[3], a[4], None if a[5] == "None" else a[5])
