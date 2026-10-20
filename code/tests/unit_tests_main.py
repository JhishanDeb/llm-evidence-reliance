import sys, math, time
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
import ler_main as L  # noqa: E402
L.BOOT_RESAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

# ---- finish_record edge cases
r = L.finish_record({"y": 1}, {"lp_yes": -0.1, "lp_no": -2.4}, "alarm")
assert r["valid"] == 1 and r["pred_alarm"] == 1 and abs(r["lo_alarm"] - 2.3) < 1e-12 and r["correct"] == 1
r = L.finish_record({"y": 1}, {"lp_yes": -0.1, "lp_no": -2.4}, "reversed")
assert r["pred_alarm"] == 0 and abs(r["lo_alarm"] + 2.3) < 1e-12 and abs(r["p_alarm"] - (1 - r["p_yes"])) < 1e-15
r = L.finish_record({"y": 0}, {"lp_yes": -900.0, "lp_no": -0.001}, "alarm")      # p_yes underflows, log-odds exact
assert r["valid"] == 1 and r["p_yes_raw"] == 0.0 and abs(r["lo_alarm"] + 899.999) < 1e-9 and r["pred_alarm"] == 0
for bad in ({"lp_yes": -math.inf, "lp_no": -math.inf}, {"lp_yes": float("nan"), "lp_no": -1.0},
            {"lp_yes": -1.0, "lp_no": -math.inf}):
    r = L.finish_record({"y": 0}, bad, "alarm")
    assert r["valid"] == 0 and r["pred_alarm"] == -1, bad
assert abs(L.logsumexp([math.log(0.2), math.log(0.3)]) - math.log(0.5)) < 1e-12
assert L.logsumexp([-math.inf, -math.inf]) == -math.inf and math.isnan(L.logsumexp([float("nan"), 0.0]))
print("finish_record / logsumexp ok")

# ---- Q3 four-way table
q = L.q3_interpretation
assert q([0.1, 0.5], [0.2, 0.9]).startswith("follows the outcome, toward alarm")
assert q([-0.5, -0.1], [-0.9, -0.2]).startswith("follows the outcome, toward safe")
assert q([0.1, 0.5], [-0.9, -0.2]) == 'follows the answer word "Yes"'
assert q([-0.5, -0.1], [0.2, 0.9]) == 'follows the answer word "No"'
assert q([-0.1, 0.5], [0.2, 0.9]) == "inconclusive" and q(None, [0.2, 0.9]) == "inconclusive"
print("Q3 table ok")

# ---- synthetic study: records from a logit model with chosen asymmetry, lean and card-truth effect
def synth(n=600, alarm_gain=1.0, safe_gain=1.0, lean=0.0, truth=0.0, adopt=False, seed=0, mass=0.95, eta=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = i % 2
        z = 1.2 * (2 * y - 1) + rng.normal(0, 1.3)                    # XGBoost log-odds
        l0 = lean + 3 * z + rng.normal(0, 4)
        cd = int(rng.random() < 1 / (1 + np.exp(-1.5 * (z - 1))))      # true card direction, follows XGBoost
        xp = 1 / (1 + np.exp(-z))
        eta_i = rng.normal(0, eta) if eta else 0.0
        for cond in L.CONDITIONS:
            card = None if cond == "C0" else (cd if cond == "C2" else 1 - cd)
            if card is None:
                lo = l0 + eta_i
            else:
                sgn = 1 if card == 1 else -1
                k = (alarm_gain if card == 1 else safe_gain) * (1 + truth * (cond == "C2"))
                lo = (1 - 0.3 * k) * l0 + 0.3 * k * sgn * 8 if adopt else l0 + sgn * 3 * k
            lo += rng.normal(0, 1)
            for fr in L.FRAMINGS:
                lo_yes = lo if fr == "alarm" else -lo
                rec = {"dataset": "x", "id": f"p{i:04d}", "cond": cond, "framing": fr, "shot_set": "A", "y": y,
                       "card_dir": card, "xgb_p": xp}
                lpy = math.log(mass) - math.log1p(math.exp(-lo_yes)) if lo_yes > -30 else math.log(mass) + lo_yes
                lpn = math.log(mass) - math.log1p(math.exp(lo_yes)) if lo_yes < 30 else math.log(mass) - lo_yes
                rows.append(L.finish_record(rec, {"lp_yes": lpy, "lp_no": lpn}, fr))
            if cond in ("C0", "C2", "C4"):
                for s in ("B", "C"):
                    rec = {"dataset": "x", "id": f"p{i:04d}", "cond": cond, "framing": "alarm", "shot_set": s, "y": y,
                           "card_dir": card, "xgb_p": xp}
                    rows.append(L.finish_record(rec, {"lp_yes": -math.log1p(math.exp(-lo)) if lo > -30 else lo,
                                                      "lp_no": -math.log1p(math.exp(lo)) if lo < 30 else -lo}, "alarm"))
    return pd.DataFrame(rows)

t0 = time.time()
cases = [("symmetric add, lean +", dict(lean=3.0), "zero"), ("symmetric add, lean -", dict(lean=-3.0), "zero"),
         ("symmetric adopt, lean +, truth", dict(lean=3.0, adopt=True, truth=0.5), "zero"),
         ("symmetric add, lean +, noisy C0", dict(lean=3.0, eta=2.0), "zero"),
         ("alarm asymmetry, lean +", dict(lean=3.0, alarm_gain=1.5), "pos"),
         ("alarm asymmetry adopt, lean +", dict(lean=3.0, alarm_gain=1.5, adopt=True), "pos"),
         ("safe asymmetry", dict(safe_gain=1.5), "neg")]
for name, kw, expect in cases:
    a = L.asymmetry(L.slice_records(synth(**kw)))["adjusted"]
    lo, hi = a["ci"]
    ok = {"zero": lo < 0 < hi or abs(a["a"]) < 0.25, "pos": lo > 0, "neg": hi < 0}[expect]
    print(f"  {name:32s} a = {a['a']:+.3f} [{lo:+.3f}, {hi:+.3f}] p = {a['p_boot']:.3g} -> expected {expect}: {ok}")
    assert ok, name
print(f"adjusted asymmetry behaves as designed ({time.time() - t0:.0f}s)")

# ---- full cell + families, with one invalid cell and the H4 difference
df_asym = synth(alarm_gain=1.6, lean=2.0, seed=1)
df_sym = synth(seed=2)
cell = L.analyse_cell(df_asym)
assert cell["validity"]["passes"], cell["validity"]
assert cell["H3"]["adjusted"]["a"] > 0 and cell["Q3_interpretation"].startswith("follows the outcome, toward alarm")
assert set(cell["exploratory"]["robustness_example_sets"]) == {"B", "C"}
low_mass = synth(seed=3, mass=0.3)
assert not L.validity(low_mass)["passes"]
cells = {}
for tag in L.MODEL_TAGS:
    for ds in L.DATASETS:
        cells[f"{tag}|{ds}"] = cell if "instruct" in tag else L.analyse_cell(df_sym, exploratory=False)
cells["llama-instruct|diabetes130"] = L.analyse_cell(low_mass, exploratory=False)
h4 = {f"{f}|{ds}": L.adjusted_difference(df_asym, df_sym) for f in ("qwen", "llama") for ds in L.DATASETS}
h4["llama|diabetes130"] = None
conf = L.confirmatory(cells, h4)
fam = conf["primary"]["H3"]["cells"]
assert fam["llama-instruct|diabetes130"]["p_used"] == 1.0 and not fam["llama-instruct|diabetes130"]["supported"]
assert fam["qwen-instruct|physionet2012"]["supported"]
assert conf["primary"]["H3"]["overall"]["supported_overall"]          # 3/3 qwen, 2/3 llama
assert conf["secondary"]["H4"]["cells"]["llama|diabetes130"]["p_used"] == 1.0
assert conf["secondary"]["H4"]["cells"]["qwen|physionet2012"]["supported"]
print("H3 overall:", conf["primary"]["H3"]["overall"])
print("H4 qwen|physionet2012:", {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in h4["qwen|physionet2012"].items() if k != "secondary_delta_diff"})
# Holm with m = 6 and excluded cells at p = 1
f = L.holm_family({"a": {"p": 0.01, "direction_ok": True, "included": True},
                   "b": {"p": 0.001, "direction_ok": True, "included": False},
                   "c": {"p": 0.02, "direction_ok": False, "included": True}})
assert abs(f["a"]["p_holm"] - 0.03) < 1e-12 and f["a"]["supported"] and not f["b"]["supported"] and not f["c"]["supported"]
print("all main analysis tests passed")
