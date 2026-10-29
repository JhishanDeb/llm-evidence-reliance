"""Offline tests of ler_v3.py (third study). CPU only.

usage: python unit_tests_v3.py CODE_DIR FAKE_DATA_DIR RUNS_DIR SPLIT_JSON PROTOCOL_V3
  CODE_DIR       code/v3 (ler_main.py and ler_v3.py)
  FAKE_DATA_DIR  fake data of the three datasets (make_fake_p12.py, make_fake_p19.py, make_fake_d130.py)
  RUNS_DIR       unpacked real run outputs of the earlier studies (main_qwen-instruct/, v2_olmo3-instruct/, ...)
  SPLIT_JSON     the exploratory split of the paper (review_analyses.json), for the estimator check
  PROTOCOL_V3    PREREGISTRATION_V3.md

1. ler_main.py is main-2.0; the registered prompts are unchanged; every changed-card prompt is the registered prompt
   with exactly the registered change; the protocol strings are in PREREGISTRATION_V3.md.
2. The vectorised a and a' equal ler_main's estimators on real records, and a weighted fit with resample counts
   equals the ordinary fit on the resampled rows.
3. The split of compare_formats() reproduces the paper's exploratory split on the real records (estimates).
4. The decision rules of R4 to R7 on hand-made cells.
5. The model window rule.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

CODE, DATA, RUNS, SPLIT, PROT = sys.argv[1:6]
sys.path.insert(0, CODE)
os.environ.setdefault("LER_P12_MIN_PER_SET", "1000")
os.environ.setdefault("LER_P19_MIN_PER_SET", "500")
import ler_main as L  # noqa: E402
import ler_v3 as V  # noqa: E402

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


# ------------------------------------------------------------------------------------------------ 1. prompts
check("ler_main.py is main-2.0", L.file_sha256(os.path.join(CODE, "ler_main.py")) == V.MAIN_SHA256)
check("protocol strings and sections in PREREGISTRATION_V3.md", V.check_against_protocol_v3(open(PROT).read()) == [])
check("FULL_LINE template reproduces the registered card line format",
      V.FULL_LINE.format(k=1, feature="F", value="7", word="increases", outcome="death", impact="+0.42")
      == "1. F = 7: increases the risk of death (impact +0.42)")
check("grid sizes 20 and 26", len(V.grid_v3(V.EARLIER_GRIDS)) == 20 and len(V.grid_v3(V.NEW_GRIDS)) == 26)
check("no duplicate grid entries", len(set(V.grid_v3(V.NEW_GRIDS))) == 26)

B = {}
for ds in L.DATASETS:
    if ds == "physionet2019":
        df, f, c, i = L.load_physionet2019(DATA, (), log=lambda *a: None, workers=4)
    else:
        df, f, c, i = L.LOADERS[ds](DATA, (), log=lambda *a: None)
    B[ds] = L.Bundle(ds, df, f, c, i, pilot_ids=[str(x) for x in L.PILOT_IDS_FALLBACK] if ds == "physionet2012" else (),
                     sample="test", log=lambda *a: None)
n_checked, same_reg, bad = 0, True, []
for ds, b in B.items():
    for rid in b.ids[:60]:
        for cond, fr, shot in V.grid_v3(V.NEW_GRIDS):
            fmt = V.SET_FMT[shot]
            t, cd = V.prompt_v3(b, rid, cond, fr, fmt)
            if fmt == "full" or cond == "C0":
                same_reg &= (t, cd) == b.prompt(rid, cond, fr, "A")
            else:
                try:
                    V.check_format_prompt(b, rid, cond, fr, fmt, t, cd)
                    n_checked += 1
                except AssertionError:
                    bad.append((ds, rid, cond, fr, fmt))
check("registered-card and C0 prompts are the registered prompts", same_reg)
check(f"every changed-card prompt has exactly the registered change ({n_checked} prompts)", not bad and n_checked > 1000)
b = B["physionet2012"]
rid = b.ids[0]
card = b.shown_card(rid, "C2")
w = V.card_text_v3(b, card, "words")
n = V.card_text_v3(b, card, "numbers")
m = V.card_text_v3(b, card, "matched")
cue2 = V.card_text_v3(b, card, "cue")
cue4 = V.card_text_v3(b, b.shown_card(rid, "C4"), "cue")
check("words card has no number, numbers card has no direction word",
      "(impact" not in w and "increases" not in n and "decreases" not in n)
check("matched card: signs are the negative of the registered card",
      [x.split("(impact ")[1][0] for x in m.split("\n")[1:]] ==
      [{"+": "-", "-": "+"}[x.split("(impact ")[1][0]] for x in b.card_text(card).split("\n")[1:]])
check("cue flips between C2 and C4", ("increase the risk" in cue2) != ("increase the risk" in cue4))
try:
    V.prompt_v3(b, rid, "C2", "alarm", "matched")
    check("matched card refused in the alarm framing", False)
except ValueError:
    check("matched card refused in the alarm framing", True)
# the mock reads every format with the right direction
mk = V.MockV3([s["q_reversed"] for s in L.SPEC.values()])
for fmt in V.CHANGED:
    fr = "reversed" if fmt == "matched" else "alarm"
    f2, s2 = mk.evidence(V.prompt_v3(b, rid, "C2", fr, fmt)[0])
    f4, s4 = mk.evidence(V.prompt_v3(b, rid, "C4", fr, fmt)[0])
    full2 = sum(it["shap"] for it in card)
    check(f"mock reads the {fmt} card", f2 == fmt and f4 == fmt and np.sign(s2) == np.sign(full2) == -np.sign(s4))

# ------------------------------------------------------------------------------------------------ 2. estimators
def rec(path):
    return pd.read_csv(path, low_memory=False, dtype={"id": str})


Q = rec(os.path.join(RUNS, "main_qwen-instruct", "records_qwen-instruct.csv.gz"))
O3 = rec(os.path.join(RUNS, "v2_olmo3-instruct", "records_olmo3-instruct.csv.gz"))
worst = 0.0
for df in (Q, O3):
    for ds in L.DATASETS:
        for fr in L.FRAMINGS:
            w_ = L._wide(V.slice_v3(df[df["dataset"] == ds], fr, "full"))
            x = L.asym_arrays(w_)
            one = np.ones((1, len(w_)))
            a_ref = L.adjusted_fit(x["l0"], x["la"], x["ls"], x["t"], x["z"], x["y"])[0]
            ap_ref = L.reading_adjusted_asymmetry(x["l0"], x["la"], x["ls"], x["t"])
            worst = max(worst, abs(V.stat_a(x, one)[0] - a_ref), abs(V.stat_ap(x, one)[0] - ap_ref))
check(f"vectorised a, a' equal ler_main on real records (max diff {worst:.1e})", worst < 1e-8)
rng = np.random.default_rng(1)
ds_ = Q[(Q["dataset"] == "physionet2012")]
w_ = L._wide(V.slice_v3(ds_, "alarm"))
x = L.asym_arrays(w_)
worst = 0.0
for _ in range(5):
    idx = rng.integers(0, len(w_), len(w_))
    W = np.bincount(idx, minlength=len(w_)).astype(float)[None, :]
    a_ref = L.adjusted_fit(*(x[k][idx] for k in ("l0", "la", "ls", "t", "z", "y")))[0]
    ap_ref = L.reading_adjusted_asymmetry(*(x[k][idx] for k in ("l0", "la", "ls", "t")))
    worst = max(worst, abs(V.stat_a(x, W)[0] - a_ref), abs(V.stat_ap(x, W)[0] - ap_ref))
check(f"weighted fit with resample counts = refit on resampled rows (max diff {worst:.1e})", worst < 1e-8)
est, dr = V.bootstrap(50, {"m": lambda W: W.sum(1) / 50}, n_boot=2500)
check("bootstrap: 2,500 resamples in chunks, weights sum to n", len(dr["m"]) == 2500 and np.allclose(dr["m"], 1.0))

# ------------------------------------------------------------------------------------------------ 3. the paper's split
split = json.load(open(SPLIT))["split"]
worst = 0.0
for tag, df in (("qwen-instruct", Q), ("olmo3-instruct", O3)):
    for ds in L.DATASETS:
        r = V.compare_formats(df[df["dataset"] == ds], ("full",), n_boot=200)["formats"]["full"]
        p = split[f"{tag}|{ds}"]
        worst = max(worst, abs(r["O"]["a"]["est"] - p["O_a"]), abs(r["O"]["a_prime"]["est"] - p["O_ap"]),
                    abs(r["W"]["a"]["est"] - p["W_a"]), abs(r["W"]["a_prime"]["est"] - p["W_ap"]))
check(f"split of compare_formats = the paper's split (max diff {worst:.1e})", worst < 1e-8)
check("pre-listed cells match the paper's conclusive cells and signs",
      all(split[k]["valid"] and split[k][part]["conclusive"] and split[k][part]["sign"] == s
          for part in ("W", "O") for k, s in V.PRELISTED[part].items()))
others = [k for k, v in split.items() if k.split("|")[0] in ("qwen-instruct", "llama-instruct", "olmo3-instruct")
          and v["valid"]]
check("no other conclusive cell of the three earlier models is left out of the pre-listed cells",
      all((k in V.PRELISTED[p]) == bool(split[k][p]["conclusive"]) for k in others for p in ("W", "O")))


# ------------------------------------------------------------------------------------------------ 4. decision rules
def part(est_a, est_ap, p):
    sign = int(np.sign(est_a)) if np.sign(est_a) == np.sign(est_ap) else 0
    return {"a": {"est": est_a}, "a_prime": {"est": est_ap}, "p": p, "sign": sign, "conclusive": sign != 0 and p < 0.05}


def cell(O, W, valid=True, fmts=None, choice=None, fvalid=True):
    c = {"validity": {"passes": valid, "formats": {f: {"passes": valid and fvalid} for f in V.CHANGED}},
         "split": {"formats": {"full": {"O": O, "W": W}}}, "formats": {}, "choice": choice}
    for f, (Of, Wf, dO, dW) in (fmts or {}).items():
        c["formats"][f] = {"formats": {f: {"O": Of, "W": Wf, "change_from_full": {"O": dO, "W": dW}}}}
    return c


new = ["n1", "n2", "n3"]
cells = {}
for t in new:
    for ds in L.DATASETS:
        cells[f"{t}|{ds}"] = cell(part(0.1, 0.1, 0.5), part(-1.0, -0.9, 0.0001))
cells["n3|physionet2012"]["split"]["formats"]["full"]["W"] = part(0.8, 0.9, 0.001)   # opposite direction
r = V.r4(cells, new)
check("R4: two models show the pull, n3 has an opposite cell -> supported", r["supported"]
      and r["per_model"]["n3"]["opposite_cells"] == 1 and not r["per_model"]["n3"]["shows_pull"]
      and r["models_showing_pull"] == 2)
for ds in L.DATASETS:
    cells[f"n2|{ds}"]["validity"]["passes"] = False
r = V.r4(cells, new)
check("R4: n2 invalid everywhere, n3 opposite -> not supported but testable",
      not r["supported"] and r["testable"] and r["models_tested"] == 2)
for ds in L.DATASETS:
    cells[f"n1|{ds}"]["validity"]["passes"] = False
check("R4: one model with valid cells -> not testable", not V.r4(cells, new)["testable"])
# R5 agreement
ch = lambda s: {"validity": {"passes": True}, "joint": {"conclusive": True, "sign": s}}
cells = {f"{t}|{ds}": cell(part(-0.5, -0.4, 0.0001), part(-1, -1, 0.0001), choice=ch(-1)) for t in new for ds in L.DATASETS}
check("R5: all agree -> supported", V.r5(cells, new)["supported"] and V.r5(cells, new)["n_compared"] == 9)
cells["n1|diabetes130"]["choice"] = ch(1)
check("R5: one disagreement -> not supported", not V.r5(cells, new)["supported"] and V.r5(cells, new)["n_agree"] == 8)
# R6 / R7 on the pre-listed cells only (no new model): each card keeps or loses the part
cells = {}
for k in set(V.PRELISTED["W"]) | set(V.PRELISTED["O"]):
    sw = V.PRELISTED["W"].get(k, -1)
    so = V.PRELISTED["O"].get(k, 1)
    f = {}
    for fmt in V.CHANGED:
        Of = part(so * 0.5, so * 0.4, 0.0001) if fmt in ("numbers", "matched") else part(-0.01 * so, 0.02 * so, 0.9)
        dO = part(-so * 0.01, -so * 0.02, 0.5) if fmt in ("numbers", "matched") else part(-so * 0.5, -so * 0.4, 0.001)
        Wf = part(sw * 1.0, sw * 0.8, 0.0001)
        dW = part(0.0, 0.0, 1.0)
        f[fmt] = (Of, Wf, dO, dW)
    cells[k] = cell(part(so * 0.5, so * 0.5, 0.0001), part(sw * 1.0, sw * 1.0, 0.0001), fmts=f)
conf = V.confirmatory_v3(cells, [])
check("R6: W kept with every card -> supported", conf["R6"]["supported"]
      and all(conf["R6"]["formats"][f]["n_kept"] == 6 for f in V.CHANGED))
check("R7: O kept with the numbers card only -> 'carried by the numbers'",
      conf["R7"]["interpretation"].startswith("carried by the numbers") and conf["R7"]["cue"].startswith("not shown")
      and conf["R7"]["formats"]["words"]["n_reduced"] == 4)
for k in cells:      # the words card fails validity everywhere
    cells[k]["validity"]["formats"]["words"]["passes"] = False
conf = V.confirmatory_v3(cells, [])
check("R6: words card invalid -> W not kept with it -> not supported", not conf["R6"]["supported"]
      and conf["R6"]["formats"]["words"]["n_kept"] == 0)
check("R6 eligible = the six pre-listed cells with their signs", conf["R6"]["eligible"].keys() == V.PRELISTED["W"].keys()
      and conf["R6"]["eligible"]["qwen-instruct|physionet2012"]["sign"] == 1)

# replay check (Q8a): identical readings give zero differences; a shifted copy is detected
r0 = V.replay_check(O3, O3)
sh_ = O3.copy()
sh_.loc[sh_["cond"] == "C2", "lo_alarm"] += 0.5
r1 = V.replay_check(O3, sh_)
check("replay check: identical runs agree, a shift of 0.5 in C2 is found",
      r0["n_compared"] == 600 * 3 * 6 and r0["max_abs_diff"] == 0 and r0["share_same_answer"] == 1
      and abs(r1["max_abs_diff"] - 0.5) < 1e-9 and abs(r1["share_within_0_01"] - 2 / 3) < 0.01)

# result lines of the check and smoke processes: found even after other output on the same line
r_ = V.parse_result('INFO loading\nnoise without newline SMOKE_V3_RESULT {"ok": true, "x": [1, 2]}\nstderr line\n',
                    "SMOKE_V3_RESULT")
check("parse_result finds the result after other output on the same line", r_ == {"ok": True, "x": [1, 2]}
      and V.parse_result("nothing here", "SMOKE_V3_RESULT") is None
      and V.parse_result("SMOKE_V3_RESULT {broken", "SMOKE_V3_RESULT") is None)
check("registered models: three earlier and the three pinned new models with 40-character revisions",
      V.models_v3("earlier") == ["qwen-instruct", "llama-instruct", "olmo3-instruct"]
      and V.models_v3("new") == ["falcon3-7b", "yi15-9b", "phi4-mini"]
      and all(len(m["revision"]) == 40 for m in V.REGISTERED_MODELS_V3.values()))

# ------------------------------------------------------------------------------------------------ 5. model window
with tempfile.TemporaryDirectory() as d:
    def win(cfg):
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
        return V.model_window(d)
    check("model window: 16,384 cap, own shorter window, longrope original window",
          win({"max_position_embeddings": 131072, "rope_scaling": {"rope_type": "llama3", "factor": 8}}) == 16384
          and win({"max_position_embeddings": 8192}) == 8192
          and win({"max_position_embeddings": 131072, "original_max_position_embeddings": 4096,
                   "rope_scaling": {"type": "longrope", "long_factor": [1], "short_factor": [1]}}) == 4096
          and win({"max_position_embeddings": 32768}) == 16384)

n_fail = sum(not x for _, x in ok)
print(f"\n{len(ok) - n_fail} of {len(ok)} checks passed")
sys.exit(1 if n_fail else 0)
