"""Checks that the registered analysis recovers the effects built into the fake models of run_mock_v5.sh.
usage: python check_mock_v5.py RESULTS_JSON"""
import json
import sys

R = json.load(open(sys.argv[1]))
C, K = R["cells"], R["confirmatory"]
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def W(cell, pair="yn", base="CP", reading="neutral", part="W"):
    return C[cell]["pairs"][pair][base][reading][part]


pos = lambda r: r["conclusive"] and r["sign"] == 1
neg = lambda r: r["conclusive"] and r["sign"] == -1
small = lambda r, lim=0.35: abs(r["a"]) < lim and abs(r["ap"]) < lim
DS = ("physionet2012", "physionet2019")
check("every run analysed in both datasets (7 models x 2)", len(C) == 14)
check("R15 supported: the placebo weakens Qwen3's 'No' preference in both datasets",
      K["R15"]["supported"] and all(K["R15"]["cells"][f"qwen3-instruct|{d}"]["delta"] > 0.5 for d in DS))
check("Qwen3: W against the placebo near 0 at the neutral point (no asymmetry in updating)",
      all(small(W(f"qwen3-instruct|{d}")) for d in DS))
check("Qwen3: W against no card is not near 0 (the card's change of the prior enters it)",
      all(not small(W(f"qwen3-instruct|{d}", base="C0"), 0.3) for d in DS))
check("Qwen3: True/False not run (engineering-check decision) and absent from the analysis",
      all("tf" not in C[f"qwen3-instruct|{d}"]["pairs"] for d in DS) and not C["qwen3-instruct|physionet2012"]["validity"]["tf"]["passes"])
check("R16 supported: 4 of 4 cells with the predicted sign", K["R16"]["supported"] and K["R16"]["n_supported"] == 4)
check("Falcon3 and Yi-1.5: W > 0 against the placebo; Granite: W < 0",
      pos(W("falcon3-7b|physionet2012")) and pos(W("falcon3-7b|physionet2019")) and pos(W("yi15-9b|physionet2019"))
      and neg(W("granite3-instruct|physionet2012")))
check("Falcon3: W > 0 also with True/False and the paraphrase; letters near 0 (the fake pull skips letters)",
      pos(W("falcon3-7b|physionet2012", "tf")) and pos(W("falcon3-7b|physionet2012", "pa"))
      and small(W("falcon3-7b|physionet2012", "ab")))
check("Granite ignores the size of the impacts (dm near 0); Falcon3 does not (dm > 0)",
      abs(C["granite3-instruct|physionet2012"]["dose"]["dm"]["est"]) < 0.1 and C["falcon3-7b|physionet2012"]["dose"]["dm"]["est"] > 0.5)
check("probability evidence: Falcon3 W > 0 against both baselines",
      pos(C["falcon3-7b|physionet2012"]["prob"]["CP"]["neutral"]["W"]) and pos(C["falcon3-7b|physionet2012"]["prob"]["C0"]["neutral"]["W"]))
r17 = K["R17"]["cells"]
check("R17: PhysioNet 2012 cells testable and supported (Falcon3 +, Granite -); the small fake PhysioNet 2019 sample "
      "is not testable (fewer than 100 patients), so R17 is not supported",
      r17["falcon3-7b|physionet2012"]["supported"] and r17["granite3-instruct|physionet2012"]["supported"]
      and not r17["falcon3-7b|physionet2019"]["testable"] and not K["R17"]["supported"])
g = R["q21"]["generated"]
check("generated answers: about 90% parsed; agreement below 1 only for Llama (built-in flips)",
      all(0.8 < x["parsed_share"] < 0.97 for x in g.values()) and g["llama-instruct|physionet2012"]["agreement"] < 0.99
      and g["falcon3-7b|physionet2012"]["agreement"] > 0.99)
check("generated and first-token E_W have the same sign in the Falcon3 and Granite PhysioNet 2012 cells",
      all(g[k]["E_W_generated"]["E_W"] * g[k]["E_W_first_token"]["E_W"] > 0
          for k in ("falcon3-7b|physionet2012", "granite3-instruct|physionet2012")))
e = R["q21"]["earlier"]
check("Q21b compares 13 cells (Granite on PhysioNet 2019 was invalid earlier)",
      sum(v["compared"] for v in e["cells"].values()) == 13 and not e["cells"]["granite3-instruct|physionet2019"]["compared"])
t = R["q22"]
check("Q22: R16 recomputed with the template variance; counts of readings that stay conclusive",
      t["R16_with_template_variance"]["n_supported"] <= K["R16"]["n_supported"] and
      all(v["stay_conclusive"] <= v["conclusive"] for v in t["counts"].values()))
check("Q21a: the placebo's own outcome shift is computed in every valid cell",
      all(c["lean_change"] is not None for c in C.values()))
check("imputation bound skipped (no imputed readings in the fake runs)", "R15" not in R["bound"])
check("descriptives for every framing and set of every cell",
      all(len(d) >= 9 for d in R["q21"]["descriptives"].values()))
print(f"\n{sum(ok)} of {len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
