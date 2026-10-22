"""Unit tests of the second-study analysis (main-2.0) on simulated records with a known truth.
usage: python unit_tests_v2.py V2_DIR"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import ler_main as L  # noqa: E402

L.BOOT_RESAMPLES = 400
rng = np.random.default_rng(7)


def simulate(n=400, ga=1.0, gs=2.0, cga=None, csa=None, letter=0.0, lean=-0.3, invalid_share=0.0, mass=0.9):
    """Records of one model x dataset: alarm framing (C0, C2, C4, set A) and both option orders. ga / gs = shift of
    the alarm log-odds under an alarm- / safe-pointing card (Yes/No); cga / csa = the same in the answer-option framing.
    The asymmetry d = ga - gs, so a ~ ga - gs."""
    cga = ga if cga is None else cga
    csa = gs if csa is None else csa
    y = (np.arange(n) % 2).astype(int)
    z = 1.2 * (2 * y - 1) + rng.normal(0, 1.3, n)
    l0 = lean + 2.0 * z + rng.normal(0, 1.5, n)
    t01 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * (z - 0.5)))).astype(int)        # true card points to alarm
    xgb_p = 1 / (1 + np.exp(-z))
    rows = []
    for i in range(n):
        for cond in ("C0", "C2", "C4"):
            cdir = None if cond == "C0" else (t01[i] if cond == "C2" else 1 - t01[i])
            for fr in ("alarm",) + L.CHOICE_FRAMINGS:
                choice = L.is_choice(fr)
                g_al, g_sf = (cga, csa) if choice else (ga, gs)
                lo = l0[i] + (0 if cdir is None else (g_al if cdir == 1 else -g_sf)) + rng.normal(0, 0.05)
                pos_alarm = L.positive_means_alarm(fr)
                lo_pos = (lo if pos_alarm else -lo) + (letter if choice else 0.0)          # log-odds of Yes / A
                lpy = math.log(mass) - math.log1p(math.exp(-lo_pos))
                lpn = math.log(mass) - math.log1p(math.exp(lo_pos))
                if rng.random() < invalid_share:
                    lpy = float("nan")
                rec = {"dataset": "physionet2012", "id": str(i), "cond": cond, "framing": fr, "shot_set": "A",
                       "y": int(y[i]), "card_dir": cdir, "xgb_p": float(xgb_p[i]), "key": f"{i}|{cond}|{fr}"}
                rows.append(L.finish_record(rec, {"lp_yes": lpy, "lp_no": lpn, "n_imputed": 0}, fr))
    df = pd.DataFrame(rows)
    df["card_dir"] = df["card_dir"].astype(float)
    return df


# 1. pooling the two option orders removes the letter bias and reproduces the alarm-framing log-odds
df = simulate(letter=0.9, cga=1.0, csa=2.0)
pooled = L.choice_pooled(df)
alarm = df[df["framing"] == "alarm"].set_index(["id", "cond"])["lo_alarm"]
p = pooled.set_index(["id", "cond"])["lo_alarm"]
assert len(p) == len(alarm) and float((p - alarm.loc[p.index]).abs().max()) < 0.2
raw = L.choice_records(df)
assert abs(float((raw["lp_yes"] - raw["lp_no"]).mean()) - 0.9) < 0.05          # the letter bias is in the raw data
print("choice_pooled: the mean of the two orders removes a letter bias of 0.9")

# 2. an invalid reading in either order drops that patient and condition
df_inv = simulate(n=200, invalid_share=0.05)
pooled = L.choice_pooled(df_inv)
v = L.choice_records(df_inv)
bad = set(map(tuple, v[v["valid"] == 0][["id", "cond"]].to_numpy()))
assert bad and not (set(map(tuple, pooled[["id", "cond"]].to_numpy())) & bad)
print(f"choice_pooled: {len(bad)} patient-conditions with an invalid order left out")

# 3. validity of an answer-option cell
vc = L.validity_choice(df)
assert vc["passes"] and abs(vc["median_ab_mass"] - 0.9) < 1e-9
low = simulate(n=200, mass=0.3)
assert not L.validity_choice(low)["passes"]
print("validity_choice: mass rule works")

# 4. outcome-driven asymmetry: present in both framings, framing difference ~ 0
c = L.analyse_choice_cell(simulate(ga=1.0, gs=2.0, letter=0.5))
a_ab, a0_ab = c["R2"]["adjusted"]["a"], c["R2"]["reading_adjusted"]["a"]
assert -1.4 < a_ab < -0.6 and -1.4 < a0_ab < -0.6, (a_ab, a0_ab)
assert c["R2"]["h3_test"]["p"] < 0.05 and L._both_negative(c["R2"])
fd = c["framing_difference"]
assert abs(fd["diff"]) < 0.3 and abs(fd["reading_adjusted_diff"]["diff"]) < 0.3, fd["diff"]
print(f"outcome-driven: a(A/B) {a_ab:+.2f}, a'(A/B) {a0_ab:+.2f}, framing difference {fd['diff']:+.2f}")

# 5. word-driven asymmetry: present with Yes/No, absent with A/B -> framing difference ~ +1
c = L.analyse_choice_cell(simulate(ga=1.0, gs=2.0, cga=1.5, csa=1.5, letter=-0.4))
fd = c["framing_difference"]
assert abs(c["R2"]["adjusted"]["a"]) < 0.3 and 0.6 < fd["diff"] < 1.4 and fd["test"]["both_positive"]
assert fd["test"]["p"] < 0.05
print(f"word-driven: a(A/B) {c['R2']['adjusted']['a']:+.2f}, framing difference {fd['diff']:+.2f} (p {fd['test']['p']:.3f})")

# 6. stage difference: a(later) - a(earlier)
early, late = simulate(ga=1.5, gs=1.5), simulate(ga=1.0, gs=2.5)
sd = L.stage_difference(late, early)
assert -1.9 < sd["diff"] < -1.1 and sd["test"]["both_negative"], sd["diff"]
print(f"stage_difference: {sd['diff']:+.2f} (truth -1.5)")

# 7. decision rules of confirmatory_v2 (synthetic cell results)
def cell(a, a0, p, valid=True, auroc=0.7):
    return {"validity": {"passes": valid, "median_yn_mass": 0.9 if valid else 0.3},
            "by_condition": {"C0": {"auroc": auroc}},
            "H3": {"adjusted": {"a": a}, "reading_adjusted": {"a": a0}, "h3_test": {"p": p}, "delta": {"delta": -0.1}}}


def ccell(a, a0, p, valid=True, fd=None, valid_yn=True):
    return {"validity": {"passes": valid}, "validity_yesno": {"passes": valid_yn},
            "R2": {"adjusted": {"a": a}, "reading_adjusted": {"a": a0}, "h3_test": {"p": p}},
            "framing_difference": fd}


new = L.models_v2("new")
five = L.models_v2(("new", "v1"), "instruct")
assert len(new) == 3 and len(five) == 5
cells = {}
for i, tag in enumerate(new):
    for j, ds in enumerate(L.DATASETS):
        ok = (i < 2 and j < 2)                      # two models significant in two datasets each
        cells[f"{tag}|{ds}"] = cell(-1.0, -1.1, 1e-4 if ok else 0.5)
conf = L.confirmatory_v2(cells, {}, {})
assert conf["R1"]["overall"]["supported_overall"] and conf["R1"]["overall"]["models_ok"] == 2
cells[f"{new[1]}|{L.DATASETS[1]}"] = cell(-1.0, 0.2, 1e-4)    # a' in the wrong direction -> not supported
conf = L.confirmatory_v2(cells, {}, {})
assert not conf["R1"]["overall"]["supported_overall"]
cells[f"{new[1]}|{L.DATASETS[1]}"] = cell(-1.0, -1.0, 1e-4, valid=False)    # excluded cell -> p = 1
conf = L.confirmatory_v2(cells, {}, {})
assert not conf["R1"]["overall"]["supported_overall"] and conf["R1"]["cells"][f"{new[1]}|{L.DATASETS[1]}"]["p_used"] == 1
print("R1 rule: 2 of 3 models with 2 of 3 datasets; wrong-direction and excluded cells do not count")

word_fd = {"test": {"p": 0.001, "both_positive": True}, "a_y": -1.0, "reading_adjusted_diff": {"a_y": -1.2}}
v1 = L.models_v2("v1", "instruct")
assert v1 == ["qwen-instruct", "llama-instruct"]
# R1 not met by any new model -> R2 eligible models = the two first-study models; majority of 2 = 2
r1_none = {f"{t}|{ds}": cell(-1.0, -1.0, 0.5) for t in new for ds in L.DATASETS}
cc = {f"{t}|{ds}": ccell(-1.0, -1.0, 1e-4) for t in v1 for ds in L.DATASETS[:2]}
r2 = L.confirmatory_v2(r1_none, cc, {})["R2"]
assert r2["overall"]["eligible_models"] == v1 and r2["overall"]["models_needed"] == 2
assert r2["overall"]["supported_overall"] and r2["interpretation"] == "follows the outcome"
assert len(r2["cells"]) == 6                                   # Holm over the eligible models' cells only
# all three new models meet R1 -> 5 eligible, majority 3
r1_all = {f"{t}|{ds}": cell(-1.0, -1.0, 1e-4) for t in new for ds in L.DATASETS}
cc5 = {f"{t}|{ds}": ccell(-1.0, -1.0, 1e-4) for t in v1 + new[:1] for ds in L.DATASETS[:2]}
r2 = L.confirmatory_v2(r1_all, cc5, {})["R2"]
assert len(r2["overall"]["eligible_models"]) == 5 and r2["overall"]["models_needed"] == 3
assert r2["overall"]["supported_overall"] and len(r2["cells"]) == 15
cc5b = {f"{t}|{ds}": ccell(-1.0, -1.0, 1e-4) for t in v1 for ds in L.DATASETS[:2]}
assert not L.confirmatory_v2(r1_all, cc5b, {})["R2"]["overall"]["supported_overall"]     # 2 of 5 is not a majority
# supported and weaker with answer options in a majority -> outcome, larger with Yes/No
cc = {f"{t}|{ds}": ccell(-1.0, -1.0, 1e-4, fd=word_fd) for t in v1 for ds in L.DATASETS[:2]}
assert L.confirmatory_v2(r1_none, cc, {})["R2"]["interpretation"] == "follows the outcome, and is larger with the Yes/No question"
# not supported, weaker with answer options in a majority -> depends on the Yes/No question format
cc = {f"{t}|{ds}": ccell(-0.05, 0.02, 0.6, fd=word_fd) for t in v1 for ds in L.DATASETS}
assert L.confirmatory_v2(r1_none, cc, {})["R2"]["interpretation"] == "depends on the Yes/No question format"
# a cell counts only if a and a' are below 0 with Yes/No (a positive Yes/No asymmetry growing with A/B is not 'weaker')
pos_fd = dict(word_fd, a_y=0.3, reading_adjusted_diff={"a_y": 0.2})
cc = {f"{t}|{ds}": ccell(-0.05, 0.02, 0.6, fd=pos_fd) for t in v1 for ds in L.DATASETS}
r2 = L.confirmatory_v2(r1_none, cc, {})["R2"]
assert r2["interpretation"] == "unresolved" and all(v["cells_counted"] == 0 for v in r2["framing_comparison"].values())
# the comparison needs both framings valid
cc = {f"{t}|{ds}": ccell(-0.05, 0.02, 0.6, fd=word_fd, valid_yn=False) for t in v1 for ds in L.DATASETS}
assert L.confirmatory_v2(r1_none, cc, {})["R2"]["interpretation"] == "unresolved"
print("R2: eligible models, majority rule, Holm over eligible cells, four interpretation labels")

r3 = {ds: {"diff": -0.8, "reading_adjusted_diff": {"diff": -0.7}, "h4_test": {"p": 1e-4}} for ds in L.DATASETS}
cells3 = {f"{t}|{ds}": cell(-1, -1, 0.01) for t in ("olmo3-instruct", "olmo3-base") for ds in L.DATASETS}
assert L.confirmatory_v2(cells3, {}, r3)["R3"]["overall"]["supported_overall"]
for ds in L.DATASETS[:2]:
    cells3[f"olmo3-base|{ds}"] = cell(-1, -1, 0.01, auroc=0.52)        # fails the AUROC rule
# a base model with a strong lean: accuracy rule fails, AUROC rule passes -> still tested
lean = {f"{t}|{ds}": cell(-1, -1, 0.01) for t in ("olmo3-instruct", "olmo3-base") for ds in L.DATASETS}
lean[f"olmo3-base|{L.DATASETS[0]}"]["validity"]["passes"] = False      # e.g. C0 accuracy 0.50 from answering 'alarm'
assert L.confirmatory_v2(lean, {}, r3)["R3"]["cells"][f"olmo3|{L.DATASETS[0]}"]["included"]
out = L.confirmatory_v2(cells3, {}, r3)["R3"]
assert not out["overall"]["supported_overall"] and not out["cells"][f"olmo3|{L.DATASETS[0]}"]["both_valid"]
r3b = {ds: {"diff": 0.8, "reading_adjusted_diff": {"diff": 0.7}, "h4_test": {"p": 1e-4}} for ds in L.DATASETS}
cells3 = {f"{t}|{ds}": cell(-1, -1, 0.01) for t in ("olmo3-instruct", "olmo3-base") for ds in L.DATASETS}
assert not L.confirmatory_v2(cells3, {}, r3b)["R3"]["overall"]["supported_overall"]
print("R3 rule: direction, validity of both models, 2 of 3 datasets")
print("ALL V2 UNIT TESTS PASSED")
