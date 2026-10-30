"""Supplement to the registered Study 3 analysis: the imputation bound of Deviation D5 (first study).

PREREGISTRATION_V3.md, Section 10, states that the imputation bound of Deviation D5 is reported for every cell with
imputed readings, but the registered analysis notebook does not compute it. This script computes it with the method
of the second study's analysis notebook (code/v2/src/analysis_v2_src.py, halve_less_likely): in every record with an
imputed variant, the probability of the less likely answer word is halved, which bounds the effect of the imputation,
and the statistics are recomputed (2,000 resamples, as in the second study). It uses the registered code at the tag
(ler_main.py main-2.0, ler_v3.py v3-1.1) and changes no decision.
usage: python supplement_d5.py RUNS_DIR(unpacked v3_*) RESULTS_JSON CODE_DIR OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS, RES, CODE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, CODE)
import ler_main as L  # noqa: E402
import ler_v3 as V  # noqa: E402

res = json.load(open(RES))


def halve_less_likely(df):          # copied from the second study's registered analysis notebook
    df = df.copy()
    m = df["n_imputed"] > 0
    lower = df["lp_yes"] < df["lp_no"]
    df.loc[m & lower, "lp_yes"] -= np.log(2)
    df.loc[m & ~lower, "lp_no"] -= np.log(2)
    rows = [L.finish_record({k: r[k] for k in ("dataset", "id", "cond", "framing", "shot_set", "y", "card_dir", "xgb_p")}
                            | {"n_imputed": r["n_imputed"]}, {"lp_yes": r["lp_yes"], "lp_no": r["lp_no"],
                                                               "n_imputed": r["n_imputed"]}, r["framing"])
            for r in df[m].to_dict("records")]
    out = pd.concat([df[~m], pd.DataFrame(rows)], ignore_index=True)
    out["card_dir"] = out["card_dir"].astype(float)
    return out


out = {}
for d in sorted(RUNS.glob("v3_*")):
    tag = d.name[3:]
    df = pd.read_csv(d / f"records_{tag}.csv.gz", low_memory=False, dtype={"id": str})
    if int((df["n_imputed"] > 0).sum()) == 0:
        continue
    adj = halve_less_likely(df)
    for ds in L.DATASETS:
        sub, raw = adj[adj["dataset"] == ds], df[df["dataset"] == ds]
        key = f"{tag}|{ds}"
        cell = res["cells"][key]
        ent = {"share_imputed": float((raw["n_imputed"] > 0).mean()), "formats": {}}
        for f in ("full",) + V.CHANGED:
            share = float((raw[raw["shot_set"].isin(["A", V.FMT_SET[f]]) & raw["framing"].isin(L.FRAMINGS)]["n_imputed"] > 0).mean())
            comp = V.compare_formats(sub, ("full",) if f == "full" else ("full", f), n_boot=2000)
            r = comp["formats"][f]
            reg = (cell["split"] if f == "full" else cell["formats"][f])["formats"][f]
            e = {"share_imputed": share}
            for part in ("O", "W"):
                e[part] = {"bounded_a": r[part]["a"]["est"], "bounded_a_prime": r[part]["a_prime"]["est"],
                           "bounded_p": r[part]["p"], "bounded_sign": r[part]["sign"],
                           "registered_a": reg[part]["a"]["est"], "registered_a_prime": reg[part]["a_prime"]["est"],
                           "registered_sign": reg[part]["sign"], "registered_p": reg[part]["p"]}
            ent["formats"][f] = e
        if cell.get("choice"):
            pooled = L.choice_pooled(sub)
            _b = L.BOOT_RESAMPLES
            L.BOOT_RESAMPLES = 2000
            try:
                a = L.asymmetry(pooled) if pooled is not None else None
            finally:
                L.BOOT_RESAMPLES = _b
            ent["answer_options"] = {"bounded_a": L._get(a, "adjusted", "a"), "bounded_a_prime": L._get(a, "reading_adjusted", "a"),
                                     "registered_a": cell["choice"]["asymmetry"]["adjusted"]["a"],
                                     "registered_a_prime": cell["choice"]["asymmetry"]["reading_adjusted"]["a"]}
        out[key] = ent
    print("done", tag, flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
# summary: largest change and any change of the joint reading's sign or conclusiveness
rows = []
for k, e in out.items():
    for f, x in e["formats"].items():
        for part in ("O", "W"):
            y = x[part]
            rows.append(dict(cell=k, card=f, part=part, share=x["share_imputed"],
                             d_a=y["bounded_a"] - y["registered_a"], d_ap=y["bounded_a_prime"] - y["registered_a_prime"],
                             reg_sign=y["registered_sign"], bnd_sign=y["bounded_sign"],
                             reg_concl=y["registered_sign"] != 0 and y["registered_p"] < 0.05,
                             bnd_concl=y["bounded_sign"] != 0 and y["bounded_p"] < 0.05))
t = pd.DataFrame(rows)
t["max_abs_change"] = t[["d_a", "d_ap"]].abs().max(axis=1)
print(t.groupby("cell")["max_abs_change"].max().round(3).to_string())
ch = t[(t.reg_sign != t.bnd_sign) | (t.reg_concl != t.bnd_concl)]
print("cells/cards where the joint reading changes:\n", ch[["cell", "card", "part", "share", "d_a", "d_ap", "reg_sign",
                                                            "bnd_sign", "reg_concl", "bnd_concl"]].round(3).to_string(index=False))
