"""Offline tests of ler_v4.py (fourth study). CPU only.

usage: python unit_tests_v4.py CODE_DIR FAKE_DATA_DIR RUNS3_DIR PROTOCOL_V4
  CODE_DIR       code/v4 (ler_main.py, ler_v3.py and ler_v4.py)
  FAKE_DATA_DIR  fake data of the three datasets
  RUNS3_DIR      unpacked real run outputs of the third study (v3_qwen-instruct/, v3_olmo3-instruct/, ...)
  PROTOCOL_V4    PREREGISTRATION_V4.md

1. The registered code is unchanged; the protocol strings are in PREREGISTRATION_V4.md; grid sizes.
2. Prompts: the builder reproduces the registered prompts; every new prompt is the registered prompt with exactly the
   registered change; altered prompts are caught.
3. Readings: answer words, the mapping of the first answer word, the fake model reads every format the right way.
4. Estimators: the registered pair equals the third study's split on real records; pairing across runs.
5. Decision rules of R8 to R13 and the joint reading on hand-made cells; validity rules.
6. Generated answers, the imputation bound, the decision-level difference, the decisions of the engineering check.
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

CODE, DATA, RUNS3, PROT = sys.argv[1:5]
sys.path.insert(0, CODE)
os.environ.setdefault("LER_P12_MIN_PER_SET", "1000")
os.environ.setdefault("LER_P19_MIN_PER_SET", "500")
import ler_main as L  # noqa: E402
import ler_v3 as V  # noqa: E402
import ler_v4 as W  # noqa: E402

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------------------------ 1. code, protocol
check("ler_main.py is main-2.0 and ler_v3.py is v3-1.1",
      L.file_sha256(os.path.join(CODE, "ler_main.py")) == W.MAIN_SHA256 == V.MAIN_SHA256
      and L.file_sha256(os.path.join(CODE, "ler_v3.py")) == W.V3_SHA256)
check("protocol strings, models, revisions and sections in PREREGISTRATION_V4.md",
      W.check_against_protocol_v4(open(PROT).read()) == [])
check("a missing string is reported", W.check_against_protocol_v4(open(PROT).read().replace(W.NEG_Q["physionet2019"], "x"))
      == [W.NEG_Q["physionet2019"]])
check("grid sizes: 31 per instruction-tuned model (37 with plain), 12 per base model, no duplicates",
      len(W.grid_v4(W.INSTRUCT_GRIDS)) == 31 and len(W.grid_v4(W.INSTRUCT_GRIDS + ("plain",))) == 37
      and len(W.grid_v4(W.BASE_GRIDS)) == 12 and len(set(W.grid_v4(W.INSTRUCT_GRIDS + ("plain", "chat")))) == 43)
check("models: five instruction-tuned and three base models with 40-character revisions; families pair up",
      W.INSTRUCT_TAGS == ["llama-instruct", "qwen-instruct", "olmo3-instruct", "falcon3-7b", "yi15-9b"]
      and all(len(m["revision"]) == 40 for m in W.MODELS_V4.values())
      and [W.family_tags(f) for f in W.R10_FAMILY_ORDER] == [("llama-base", "llama-instruct"),
                                                             ("qwen-base", "qwen-instruct"), ("olmo3-base", "olmo3-instruct")])
check("revisions of the instruction-tuned models are those of the third study",
      all(W.MODELS_V4[m]["revision"] == r["revision"] for m, r in V.REGISTERED_MODELS_V3.items() if m in W.MODELS_V4))
check("framings: the first answer word stands for the alarming outcome exactly in alarm, tf_alarm, para_alarm",
      [f for f in W.FRAMINGS_V4 if W.pos_alarm(f)] == ["alarm", "tf_alarm", "para_alarm"]
      and W.answer_words("tf_reversed") == ("True", "False") and W.answer_words("negated") == ("Yes", "No"))

# ------------------------------------------------------------------------------------------------ 2. prompts
B = {}
for ds in L.DATASETS:
    if ds == "physionet2019":
        df, f, c, i = L.load_physionet2019(DATA, (), log=lambda *a: None, workers=4)
    else:
        df, f, c, i = L.LOADERS[ds](DATA, (), log=lambda *a: None)
    B[ds] = L.Bundle(ds, df, f, c, i, pilot_ids=[str(x) for x in L.PILOT_IDS_FALLBACK] if ds == "physionet2012" else (),
                     sample="test", log=lambda *a: None)
same_reg, same_build, n_checked, bad = True, True, 0, []
for ds, b in B.items():
    for rid in b.ids[:40]:
        for cond in W.SPLIT_CONDS:
            for fr in L.FRAMINGS:
                for s in ("A", "Asf"):
                    ref = b.prompt(rid, cond, fr, s)
                    same_build &= W.build_prompt(b, rid, cond, s, "Question: " + b.question(fr), W.YN_WORDS,
                                                 fr == "alarm") == ref
                for s in ("A", "Asf", "Abase", "Achat"):
                    same_reg &= W.prompt_v4(b, rid, cond, fr, s) == b.prompt(rid, cond, fr, "Asf" if s == "Asf" else "A")
        for cond, fr, shot in W.grid_v4(W.INSTRUCT_GRIDS + ("plain", "chat")):
            t, cd = W.prompt_v4(b, rid, cond, fr, shot)
            try:
                W.check_prompt_v4(b, rid, cond, fr, shot, t, cd)
                n_checked += 1
            except AssertionError as e:
                bad.append((ds, rid, cond, fr, shot, str(e)))
check("build_prompt with the registered strings = ler_main's registered prompt (sets A and Asf)", same_build)
check("registered framings with sets A, Asf, Abase, Achat are ler_main's prompts", same_reg)
check(f"every new prompt has exactly the registered change ({n_checked} prompts)", not bad and n_checked > 3000)
b = B["physionet2012"]
rid = b.ids[0]
pick = lambda cond, fr, s="A": W.prompt_v4(b, rid, cond, fr, s)
t_tf, _ = pick("C2", "tf_alarm")
t_tr, _ = pick("C2", "tf_reversed")
t_ng, _ = pick("C2", "negated")
t_pa, _ = pick("C0", "para_alarm")
t_net2, d2 = pick("C2", "alarm", "Anet")
t_net4, d4 = pick("C4", "alarm", "Anet")
check("True/False prompt ends with the statement and the True/False instruction",
      t_tf.endswith("Statement: The patient dies during the current hospital stay.\n" + W.TF_INSTRUCTION)
      and t_tr.endswith("Statement: The patient survives the current hospital stay.\n" + W.TF_INSTRUCTION)
      and "Question:" not in t_tf and "Answer: Yes" not in t_tf)
exa = [yy for _, yy in b.shots["A"]]
ans_ng = [ln.split(": ")[1] for ln in t_ng.splitlines() if ln.startswith("Answer: ")]
ans_tf = [ln.split(": ")[1] for ln in t_tf.splitlines() if ln.startswith("Answer: ")]
check("negated question: 'No' for alarming examples, 'Yes' for safe ones; True/False: 'True' for alarming ones",
      ans_ng == ["No" if y else "Yes" for y in exa] and ans_tf == ["True" if y else "False" for y in exa]
      and t_ng.endswith(W.NEG_Q["physionet2012"] + "\n" + L.ANSWER_INSTRUCTION))
check("paraphrased prompt: task, headers, labels, question and instruction replaced",
      t_pa.startswith("Task: " + W.PARA_TASK["physionet2012"]) and W.PARA_EXAMPLES_HEADER in t_pa
      and "\nCase 1\n" in t_pa and "Example 1" not in t_pa and W.PARA_PATIENT_HEADER in t_pa
      and t_pa.endswith(W.PARA_Q["physionet2012"]["alarm"] + "\n" + W.PARA_INSTRUCTION) and L.EXAMPLES_HEADER not in t_pa)
s2 = sum(it["shap"] for it in b.shown_card(rid, "C2"))
check("net-number card: one sentence with the signed sum, flipped in C4, no direction word",
      f"combined impact of {W._impact(s2)} on the risk of death." in t_net2
      and f"combined impact of {W._impact(-s2)} on the risk of death." in t_net4 and d2 == 1 - d4
      and "increases the risk" not in t_net2 and "(impact" not in t_net2)
check("the cue is the third study's cue prompt", pick("C2", "alarm", "Acue") == V.prompt_v3(b, rid, "C2", "alarm", "cue"))
check("refused: C6, the cue in the reversed framing or with C0, True/False with Asf",
      raises(lambda: pick("C6", "alarm")) and raises(lambda: pick("C2", "reversed", "Acue"))
      and raises(lambda: pick("C0", "alarm", "Anet")) and raises(lambda: pick("C2", "tf_alarm", "Asf")))
cd2 = pick("C2", "tf_alarm")[1]
tamper = {"True/False with an extra space": ("tf_alarm", "A", t_tf.replace("Statement: The", "Statement:  The", 1)),
          "negated with one answer not swapped": ("negated", "A", t_ng.replace("Answer: No", "Answer: Yes", 1)),
          "paraphrase with the registered patient header": ("para_alarm", "A",
                                                             pick("C2", "para_alarm")[0].replace(W.PARA_PATIENT_HEADER, L.PATIENT_HEADER)),
          "net card with a flipped sign": ("alarm", "Anet", t_net2.replace(W._impact(s2), W._impact(-s2))),
          "True/False with a changed patient value": ("tf_alarm", "A", t_tf.replace(": ", ":  ", 3))}
caught = {k: raises(lambda f=f, s=s, t=t: W.check_prompt_v4(b, rid, "C2", f, s, t, cd2 if f != "alarm" else d2),
                    AssertionError) for k, (f, s, t) in tamper.items()}
check(f"altered prompts are caught: {caught}", all(caught.values()))

# ------------------------------------------------------------------------------------------------ 3. readings
rec = W.finish_record_v4({"y": 1}, {"lp_yes": np.log(0.7), "lp_no": np.log(0.2), "n_imputed": 0}, "tf_alarm")
rec2 = W.finish_record_v4({"y": 1}, {"lp_yes": np.log(0.7), "lp_no": np.log(0.2), "n_imputed": 0}, "negated")
check("finish_record_v4: 'True' = alarm in tf_alarm, 'Yes' = safe in the negated question",
      rec["pred_alarm"] == 1 and rec["lo_alarm"] > 0 and rec["answer_words"] == "True/False" and rec["correct"] == 1
      and rec2["pred_alarm"] == 0 and rec2["lo_alarm"] < 0 and rec2["answer_words"] == "Yes/No")
mk = W.MockV4(truth={}, pull={"Yes/No": 0.0, "True/False": 0.0})
tok = mk.tokenizer
good = True
for fr, shot in [("alarm", "A"), ("reversed", "A"), ("tf_alarm", "A"), ("tf_reversed", "A"), ("negated", "A"),
                 ("para_alarm", "A"), ("para_reversed", "A"), ("alarm", "Asf"), ("alarm", "Acue"), ("alarm", "Anet")]:
    t2, c2 = pick("C2", fr, shot)
    t4, c4 = pick("C4", fr, shot)
    z2, w2, _ = mk.logodds(L.render(t2, "instruct", tok))
    z4, w4, _ = mk.logodds(L.render(t4, "instruct", tok))
    sg = 1 if W.pos_alarm(fr) else -1           # log-odds of the first word = sg * log-odds of the alarming outcome
    good &= w2 == W.answer_words(fr) and np.sign(sg * (z2 - z4)) == (1 if c2 == 1 else -1)
check("the fake model reads every format, its answer words and the direction of the card", good)
r_tf = mk.word_logprobs([L.render(t_tf, "instruct", tok)], ("True", "False"))
check("the fake model refuses a reading with the wrong answer words, reads True/False tokens",
      raises(lambda: mk.word_logprobs([L.render(t_tf, "instruct", tok)], ("Yes", "No"))) and len(r_tf[0][0]) == 4)
mt = L._MockTok()
check("render_v4: Abase is plain text, Achat uses the given chat tokenizer, the rest the model's own format",
      W.render_v4("X", "instruct", "Abase", mt) == "X\nAnswer:" and W.render_v4("X", "base", "A", mt) == "X\nAnswer:"
      and W.render_v4("X", "base", "Achat", mt, mt).startswith("<|im_start|>user\nX")
      and W.render_v4("X", "instruct", "A", mt).endswith("assistant\nAnswer:") and raises(lambda: W.render_v4("X", "base", "Achat", mt)))
check("render_free removes exactly the prefill", W.render_free("X", mt) == W.render_v4("X", "instruct", "A", mt)[:-len("Answer:")])

# ------------------------------------------------------------------------------------------------ 4. estimators
def rec_(path):
    return pd.read_csv(path, low_memory=False, dtype={"id": str})


Q = rec_(os.path.join(RUNS3, "v3_qwen-instruct", "records_qwen-instruct.csv.gz"))
O3 = rec_(os.path.join(RUNS3, "v3_olmo3-instruct", "records_olmo3-instruct.csv.gz"))
worst, ci_same = 0.0, True
for df_ in (Q, O3):
    for ds in L.DATASETS:
        d = df_[df_["dataset"] == ds]
        ref = V.compare_formats(d, ("full",), n_boot=300)["formats"]["full"]
        new = W.pair_comparison({"alarm/A": W.slice_v4(d, "alarm"), "reversed/A": W.slice_v4(d, "reversed")},
                                {"registered": ("alarm/A", "reversed/A")}, n_boot=300)["pairs"]["registered"]
        for part in ("O", "W"):
            for v in ("a", "a_prime"):
                worst = max(worst, abs(ref[part][v]["est"] - new[part][v]["est"]))
                ci_same &= np.allclose(ref[part][v]["ci"], new[part][v]["ci"])
            ci_same &= ref[part]["p"] == new[part]["p"]
check(f"registered pair = the third study's split on real records (max diff {worst:.1e}, same intervals and p)",
      worst < 1e-10 and ci_same)
res3 = json.load(open(os.path.join(RUNS3, "analysis_work", "analysis_v3_results", "results_v3.json")))
d = Q[Q["dataset"] == "physionet2012"]
new = W.analyse_cell_v4(d, n_boot=200)
old = res3["cells"]["qwen-instruct|physionet2012"]["split"]["formats"]["full"]
check("analyse_cell_v4 on a third-study cell reproduces its registered W and O estimates",
      all(abs(new["registered"]["pairs"]["registered"][p][v]["est"] - old[p][v]["est"]) < 1e-10
          for p in ("O", "W") for v in ("a", "a_prime")))
d12 = Q[Q["dataset"] == "physionet2012"]
f = W.analyse_2x2(d12.assign(shot_set=d12["shot_set"].replace({"A": "A"})).pipe(
    lambda x: pd.concat([x, x[x["shot_set"] == "A"].assign(shot_set="Achat")])), d12, n_boot=200)
pp = f["primary"]
check("2 x 2: a base run identical to the instruction-tuned run gives P = F = 0 exactly (pairing by patient)",
      all(abs(pp["contrasts"][c][v]["est"]) < 1e-12 for c in ("P", "F") for v in ("a", "a_prime"))
      and f["secondary"] is None and pp["n"] > 500)
# validity of a framing: a model that ignores the negation fails (ii) and (iii)
fake = d12[(d12["shot_set"] == "A") & (d12["framing"] == "alarm")].copy()
ign = fake.assign(framing="negated", lo_alarm=-fake["lo_alarm"], key=fake["key"].str.replace("|alarm|", "|negated|"))
rd = W.framing_validity(pd.concat([d12, ign]), "negated", "A")
good_ = W.framing_validity(pd.concat([d12, fake.assign(framing="negated")]), "negated", "A")
check("framing validity: a reading that ignores 'not' fails (movement < 0, C0 correlation < 0); a correct one passes",
      not rd["passes"] and rd["movement"] < 0 and rd["c0_spearman_with_alarm"] < 0 and good_["passes"])


# ------------------------------------------------------------------------------------------------ 5. decision rules
def part(est_a, est_ap, p):
    sign = int(np.sign(est_a)) if np.sign(est_a) == np.sign(est_ap) else 0
    return {"a": {"est": est_a}, "a_prime": {"est": est_ap}, "p": p, "sign": sign, "conclusive": sign != 0 and p < 0.05}


def cell(sign_w, sign_o, pairs, valid=True, pvalid=None):
    """pairs: name -> (W value as a multiple of the cell's sign, its p, change p) ; O follows the same rule."""
    c = {"validity": {"split": {"passes": valid}, "pairs": {}}, "pairs": {}}
    for name, (fw, pw, pcw, fo, po, pco) in pairs.items():
        c["validity"]["pairs"][name] = {"passes": valid and (pvalid or {}).get(name, True)}
        w, o = sign_w * fw, sign_o * fo
        c["pairs"][name] = {"pairs": {name: {"W": part(w, w, pw), "O": part(o, o, po)}},
                            "contrasts": {"dW": part(w - sign_w, w - sign_w, pcw), "dO": part(o - sign_o, o - sign_o, pco)}}
    return c


def build(spec):
    """spec: pair -> function(cell key) -> (fw, pw, pcw, fo, po, pco)."""
    out = {}
    keys = set(W.ELIGIBLE["W"]) | set(W.ELIGIBLE["O"])
    for k in keys:
        out[k] = cell(W.ELIGIBLE["W"].get(k, -1), W.ELIGIBLE["O"].get(k, 1), {p: fn(k) for p, fn in spec.items()})
    return out


KEEP = lambda k: (1.0, 1e-5, 0.9, 1.0, 1e-5, 0.9)           # part unchanged, conclusive
GONE = lambda k: (0.02, 0.8, 1e-5, 0.02, 0.8, 1e-5)          # part near 0, change conclusive toward 0
WEAK = lambda k: (0.5, 0.2, 0.3, 0.5, 0.2, 0.3)              # neither
cells = build({"tf": KEEP, "neg": KEEP, "order": KEEP, "para": KEEP})
kt = W.kept_test(cells, "W", "tf")
check("kept: W supported in all 11 eligible cells", kt["result"] == "kept" and kt["n_supported"] == 11 and kt["n_reduced"] == 0)
cells = build({"tf": GONE, "neg": KEEP})
check("removed with True/False, kept with the negated question -> 'the answer words'",
      W.kept_test(cells, "W", "tf")["result"] == "removed" and W.kept_test(cells, "W", "neg")["result"] == "kept"
      and W.joint_reading(W.kept_test(cells, "W", "tf")["result"], W.kept_test(cells, "W", "neg")["result"]) == "the answer words")
cells = build({"tf": KEEP, "neg": GONE})
check("joint reading: kept / removed -> 'the named outcome'", W.joint_reading("kept", "removed") == "the named outcome"
      and W.kept_test(cells, "W", "neg")["result"] == "removed")
check("joint reading: the other cells of the table", W.joint_reading("kept", "kept") == "answer polarity"
      and W.joint_reading("removed", "removed") == "the Yes/No question as worded"
      and W.joint_reading("not settled", "kept") == "not settled by this study"
      and W.joint_reading("kept", "not settled") == "not settled by this study")
cells = build({"tf": WEAK})
check("neither supported nor reduced -> 'not settled'", W.kept_test(cells, "W", "tf")["result"] == "not settled")
n6 = list(W.ELIGIBLE["W"])[:6]
cells = build({"tf": lambda k: KEEP(k) if k in n6 else GONE(k)})
r6_ = W.kept_test(cells, "W", "tf")
n5 = list(W.ELIGIBLE["W"])[:5]
cells = build({"tf": lambda k: KEEP(k) if k in n5 else GONE(k)})
r5_ = W.kept_test(cells, "W", "tf")
check("kept needs more than half: 6 of 11 kept; 5 of 11 with 6 reduced -> removed",
      r6_["result"] == "kept" and r5_["result"] == "removed" and r5_["n_reduced"] == 6)
cells = build({"tf": KEEP})
for k in list(W.ELIGIBLE["W"])[:6]:
    cells[k]["validity"]["pairs"]["tf"]["passes"] = False
r_ = W.kept_test(cells, "W", "tf")
check("an invalid format enters with p = 1: 6 of 11 invalid -> not kept, not removed", r_["result"] == "not settled"
      and r_["n_valid"] == 5 and r_["cells"][list(W.ELIGIBLE["W"])[0]]["p_used"] == 1.0)
cells = build({"tf": lambda k: (-1.0, 1e-5, 1e-5, 1.0, 1e-5, 0.9)})
r_ = W.kept_test(cells, "W", "tf")
check("a part that flips sign is 'opposite' and reduced -> removed", r_["n_opposite"] == 11 and r_["result"] == "removed")
cells = build({"order": KEEP})
check("O eligible: 7 cells; kept with the order pair", W.kept_test(cells, "O", "order")["n_eligible"] == 7
      and W.kept_test(cells, "O", "order")["result"] == "kept")
cells = build({"tf": KEEP, "neg": KEEP, "order": lambda k: (1.0, 1e-5, 0.9, 0.02, 0.8, 1e-5), "para": KEEP})
cf = W.confirmatory_v4(cells, {}, [], [])
check("confirmatory_v4: R11 O removed -> 'depends on the order of the solved examples'; R10 not testable without families",
      cf["R11"]["O"]["result"] == "removed" and "depends on the order" in cf["R11"]["O_reading"]
      and cf["R10"]["interpretation"] == "not testable" and cf["joint_reading"] == "answer polarity"
      and cf["R8"]["prediction_holds"] == {"W": True, "O": True})


def fam_cell(P, F, valid=(True, True, True)):
    return {"validity": {"B_plain": {"passes": valid[0]}, "B_chat": {"passes": valid[1]}, "I_chat": {"passes": valid[2]}},
            "primary": {"contrasts": {"P": P, "F": F}}}


def fams(fp, ff):
    out = {}
    for f in ("llama", "qwen"):
        for ds, s in W.R10_ELIGIBLE[f].items():
            out[f"{f}|{ds}"] = fam_cell(part(s * fp[0], s * fp[0], fp[1]), part(s * ff[0], s * ff[0], ff[1]))
    return out


check("R10: P supported, F not -> 'post-training produces the part, not the chat template'",
      W.r10(fams((1.0, 1e-5), (0.0, 0.9)), ["llama", "qwen"])["interpretation"] == "post-training produces the part, not the chat template")
check("R10: F in the direction of W, P not -> 'the chat template produces the part, not post-training'",
      W.r10(fams((0.0, 0.9), (1.0, 1e-5)), ["llama", "qwen"])["interpretation"] == "the chat template produces the part, not post-training")
check("R10: both -> 'both contribute'; neither -> 'not separated by this study'",
      W.r10(fams((1.0, 1e-5), (1.0, 1e-5)), ["llama", "qwen"])["interpretation"] == "both contribute"
      and W.r10(fams((0.0, 0.9), (0.0, 0.9)), ["llama", "qwen"])["interpretation"] == "not separated by this study")
fx = fams((1.0, 1e-5), (0.0, 0.9))
fx["llama|physionet2012"]["validity"]["B_chat"]["passes"] = False
fx["llama|diabetes130"]["validity"]["B_chat"]["passes"] = False
r_ = W.r10(fx, ["llama", "qwen"])
check("R10: B.chat invalid in two of four cells -> P supported in 2, not more than half -> not supported",
      r_["n_P_supported"] == 2 and not r_["supported"] and not r_["P"]["llama|physionet2012"]["included"])
fx = fams((-1.0, 1e-5), (0.0, 0.9))
check("R10: P opposite to W is not support", W.r10(fx, ["llama", "qwen"])["n_P_supported"] == 0)
check("R10 eligible cells of the families", W.r10(fams((1, 1e-5), (0, 0.9)), ["llama", "olmo3"])["n_eligible"] == 4
      and set(W.r10(fams((1, 1e-5), (0, 0.9)), ["llama"])["P"]) == {"llama|physionet2012", "llama|diabetes130"})
agg = {f"{t}|{ds}": {"validity": {"split": {"passes": True}, "pairs": {"aggregate": {"passes": True}}},
                     "aggregate": {"dm": {"est": 0.5 if ds != "diabetes130" else -0.1, "p": 1e-6 if ds != "diabetes130" else 0.5},
                                   "movement": {"A": {"est": 2.0}, "Acue": {"est": 1.5}, "Anet": {"est": 1.0}}}}
       for t in W.INSTRUCT_TAGS for ds in L.DATASETS}
r_ = W.r13(agg, W.INSTRUCT_TAGS)
check("R13: cue > net in 10 of 15 valid cells -> supported", r_["supported"] and r_["n_supported"] == 10 and r_["n_valid"] == 15)
for k in list(agg)[:8]:
    agg[k]["validity"]["pairs"]["aggregate"]["passes"] = False
r_ = W.r13(agg, W.INSTRUCT_TAGS)
check("R13: 'more than half of the valid cells' counts valid cells only", r_["n_valid"] == 7 and r_["supported"] == (r_["n_supported"] > 3.5))

# ------------------------------------------------------------------------------------------------ 6. other parts
cases = {"**No**, because": "No", "Based on the data, the answer is Yes.": "Yes", "The answer: no.": "No",
         "I cannot tell.": None, " Yes": "Yes", "Answer is: YES": "Yes", "There is no clear sign": None, '"No"': "No",
         "Nope": None, "> Yes.": "Yes", "Answer:No": "No", "True": None}
check("parse_generated follows the registered rule", all(W.parse_generated(k) == v for k, v in cases.items())
      and W.parse_generated("False.", W.TF_WORDS) == "False")
gp = W.greedy_patients(b)
check("greedy subsample: 50 per class, sorted, deterministic, from the test sample",
      len(gp) == 100 and sorted(gp) == gp and gp == W.greedy_patients(b) and sum(b.y[i] for i in gp) == 50
      and set(gp) <= set(b.ids))
dd = pd.DataFrame({"key": ["k1", "k2"], "dataset": "x", "id": ["1", "2"], "cond": "C0", "framing": ["tf_alarm", "negated"],
                   "shot_set": "A", "y": 1, "card_dir": np.nan, "xgb_p": 0.5, "lp_yes": [np.log(0.6), np.log(0.1)],
                   "lp_no": [np.log(0.3), np.log(0.8)], "n_imputed": [1, 0]})
dd = pd.DataFrame([W.finish_record_v4(dict(r), r, r["framing"]) for r in dd.to_dict("records")])
hb = W.halve_less_likely_v4(dd).set_index("key")
check("imputation bound: only imputed records change, the less likely word is halved",
      abs(hb.at["k1", "lp_no"] - (np.log(0.3) - np.log(2))) < 1e-12 and abs(hb.at["k1", "lp_yes"] - np.log(0.6)) < 1e-12
      and abs(hb.at["k2", "lp_yes"] - np.log(0.1)) < 1e-12 and hb.at["k1", "answer_words"] == "True/False")
rows = []
for i in range(8):
    y = int(i < 4)
    for cond, pa in (("C0", 0), ("C2", 1 if i % 2 == 0 else 0), ("C4", 1)):
        rows.append({"id": str(i), "cond": cond, "pred_alarm": pa, "y": y, "card_dir": 1 if cond == "C2" else 0})
# y=0 (i 4..7): alarm-pointing card = C2: decisions 1,0,1,0 -> FPR 0.5 vs 0 at C0 -> +0.5
# y=1 (i 0..3): safe-pointing card = C4: decisions 1 -> FNR 0 vs 1 at C0 -> -1 ; E = 0.5 - (-1) = 1.5
check("decision-level difference E on a hand-made case", abs(W.decision_E(pd.DataFrame(rows)) - 1.5) < 1e-12)
res_ = {}
for mid, m in W.MODELS_V4.items():
    if m["kind"] == "instruct":
        fo = {g: True for g in W.INSTRUCT_GRIDS + ("plain",)}
        if m["tag"] == "olmo3-instruct":
            fo["neg"] = False
        if m["tag"] == "qwen-instruct":
            fo["plain"] = False
        res_[mid] = {"formats_ok": fo}
    else:
        res_[mid] = {"r10_eligible": m["family"] != "llama"}
dec = W.decisions_from_check(res_)
check("engineering-check rules: a failed format is dropped; plain run only in an R10 family where it passes; first two eligible families",
      dec == {"formats_not_run": {"olmo3-instruct": ["neg"]}, "plain_run": ["olmo3-instruct"], "r10_families": ["qwen", "olmo3"]})
check("model grids from the decisions: plain only in an R10 family; a base model outside R10 is refused",
      W.model_grids("allenai/Olmo-3-7B-Instruct", dec) == ("split", "tf", "order", "para", "aggr", "plain")
      and W.model_grids("meta-llama/Llama-3.1-8B-Instruct", dec) == W.INSTRUCT_GRIDS
      and W.model_grids("Qwen/Qwen2.5-7B", dec) == W.BASE_GRIDS
      and raises(lambda: W.model_grids("meta-llama/Llama-3.1-8B", dec)) and raises(lambda: W.model_grids("allenai/Olmo-3-1025-7B")))
RES4 = os.path.join(os.path.dirname(os.path.abspath(PROT)), "results", "v4")
import zipfile  # noqa: E402
checks = []
for name, h in zip(("engcheck_v4.zip", "engcheck_v4_run2.zip"), W.ENGCHECK_ZIP_SHA256):
    zp = os.path.join(RES4, name)
    with zipfile.ZipFile(zp) as z:
        checks.append((L.file_sha256(zp) == h, json.loads(z.read("engcheck_v4.json")),
                       hashlib.sha256(z.read("ler_v4.py")).hexdigest()))
check("the two committed check outputs have the registered hashes and contain the registered earlier code versions",
      all(c[0] for c in checks) and tuple(c[2] for c in checks) == W.ENGCHECK_CODE_SHA256)
check("ENGCHECK_DECISIONS = the rules of Section 10 applied to both checks (second replacing first)",
      W.decisions_from_check(W.merge_checks(checks[0][1], checks[1][1])) == W.ENGCHECK_DECISIONS
      == {"formats_not_run": {}, "plain_run": ["llama-instruct", "qwen-instruct"], "r10_families": ["llama", "qwen"]})
check("registered grids: 37 prompts per patient for Llama and Qwen instruct, 31 for the others, 12 per base model",
      [len(W.grid_v4(W.model_grids(m))) for m in W.MODELS_V4 if m != "allenai/Olmo-3-1025-7B"] == [37, 37, 31, 31, 31, 12, 12])
check("same_token_ids", W.same_token_ids(["a b c", "d"], mt, mt) == {"n": 2, "n_differ": 0, "identical": True})
check("TF sanity prompts use the registered instruction", all(t.endswith(W.TF_INSTRUCTION) for t, _, _ in W.TF_SANITY.values()))
r_ = W.replay_v4(O3, O3)
check("replay: identical runs agree", r_["n_compared"] > 10000 and r_["max_abs_diff"] == 0 and r_["share_same_answer"] == 1)

n_fail = sum(not x for _, x in ok)
print(f"\n{len(ok) - n_fail} of {len(ok)} checks passed")
sys.exit(1 if n_fail else 0)
