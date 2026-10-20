import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import ler_dev as L  # noqa: E402

# synthetic records with a built-in asymmetry: alarm-pointing cards move the model more than safe-pointing ones
rng = np.random.default_rng(0)
rows = []
for i in range(300):
    y = int(i % 2)
    cd = int(rng.random() < 0.4)
    p0 = 0.35 + 0.3 * y
    for cond in ("C0", "C2", "C4", "C6"):
        card = None if cond == "C0" else (cd if cond == "C2" else 1 - cd)
        shift = 0 if card is None else (0.45 if card == 1 else -0.15)
        pa = float(np.clip(p0 + shift + rng.normal(0, 0.1), 0.01, 0.99))
        for fr in ("alarm", "reversed"):
            py = pa if fr == "alarm" else 1 - pa
            rec = {"dataset": "x", "id": f"p{i:04d}", "cond": cond, "framing": fr, "shot_set": "A", "y": y,
                   "card_dir": card}
            rows.append(L.finish_record(rec, {"p_yes_raw": 0.9 * py, "p_no_raw": 0.9 * (1 - py)}, fr))
rows.append(L.finish_record({"dataset": "x", "id": "zz", "cond": "C0", "framing": "alarm", "shot_set": "A", "y": 1,
                             "card_dir": None}, {"p_yes_raw": 0.0, "p_no_raw": 0.0}, "alarm"))
df = pd.DataFrame(rows)
z = df[df.id == "zz"].iloc[0]
assert z["valid"] == 0 and z["pred_alarm"] == -1, "zero mass must be invalid, not 'No'"

res = L.analyse_cell(df)
r3 = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
print("by_condition:", {c: round(v["accuracy"], 3) for c, v in res["by_condition"].items()}, "invalid", res["n_invalid"])
print("H1", r3(res["H1"]))
print("H2", r3(res["H2"]))
print("H3", r3(res["H3"]))
print("Q3", round(res["Q3_reversed"]["delta"], 3), res["Q3_interpretation"])
print("H5", r3(res["H5"]))
print("H6", r3(res["H6"]))
assert res["H3"]["delta"] > 0 and res["H3"]["delta_ci"][0] > 0
assert res["Q3_interpretation"].startswith("follows the outcome")
assert res["H2"]["diff"] < 0

d4 = L.delta_difference(df, df)
print("H4 on identical data:", r3(d4))
assert abs(d4["diff"]) < 1e-12

cells = L.holm_across({"a": res, "b": L.analyse_cell(df)})
print("Holm H3:", cells["a"]["H3"]["p_holm"], cells["a"]["H3"]["supported"])

print("wilson 50/100:", [round(x, 4) for x in L.wilson(50, 100)])        # ~ [0.4038, 0.5962]
h = L.holm([0.01, 0.04, 0.03, float("nan")])
print("holm:", h)
assert np.allclose(h[:3], [0.03, 0.06, 0.06])

from sklearn.metrics import roc_auc_score  # noqa: E402
yy = rng.integers(0, 2, 200)
s1 = yy + rng.normal(0, 1, 200)
s2 = rng.normal(0, 1, 200)
r = L.delong_paired(yy, s1, s2)
assert abs(r["auc1"] - roc_auc_score(yy, s1)) < 1e-9 and abs(r["auc2"] - roc_auc_score(yy, s2)) < 1e-9
print("delong ok, p =", round(r["p"], 6))

from sklearn.linear_model import LogisticRegression  # noqa: E402
X = rng.normal(size=(300, 2))
t = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, 300) > 0).astype(int)
w = L.fit_logistic(X, t)
sk = LogisticRegression(C=1e9, max_iter=2000).fit(X, t)
print("logistic weights", np.round(w, 4), "| sklearn", np.round(np.r_[sk.coef_[0], sk.intercept_], 4))
assert np.allclose(w, np.r_[sk.coef_[0], sk.intercept_], atol=1e-3)

mc = L.mcnemar_exact([1, 1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 0])   # 4 vs 0 discordant -> p = 0.125
print("mcnemar:", mc)
assert abs(mc["p"] - 0.125) < 1e-12
print("all analysis tests passed")
