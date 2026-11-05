# %% [markdown]
# # Registered analysis, fifth study (`prereg-v5`)
#
# **What this notebook does.** It reads the `v5_<model>.zip` files of the fifth study, checks that every run saw the
# same fresh patients and prompts and used the registered code, models, protocol and engineering-check decisions, and
# runs the analysis registered in `PREREGISTRATION_V5.md`: R15 (the placebo card and the answer prior), R16 (W against
# the placebo card), R17 (generated decisions) and the registered questions Q18 to Q22, including the imputation bound
# of Deviation D5.
#
# **How to run.** It needs no GPU and no internet. On Kaggle: upload the zip files as one dataset (Add Input → Upload
# → New Dataset) and Save & Run All; Kaggle unpacks zip files, and the first code cell accepts either the zip files or
# the unpacked folders. Locally or on Colab: put the zip files next to the notebook and run all cells. Runtime: about
# 1 to 2 hours on a CPU. Download `analysis_v5_results.zip` at the end.

# %%
import os, sys, json, glob, gzip, zipfile, hashlib, time
import numpy as np
import pandas as pd

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
IN_DIR = os.path.join(WORK, "analysis_v5_inputs")
OUT = os.path.join(WORK, "analysis_v5_results")
SKIP = [os.path.realpath(IN_DIR), os.path.realpath(OUT)]
CODE_FILES = ("ler_main.py", "ler_v3.py", "ler_v4.py", "ler_v5.py")


def _skipped(p):
    rp = os.path.realpath(p)
    return any(rp == s or rp.startswith(s + os.sep) for s in SKIP)


def _csv_bytes(raw, name):
    return gzip.decompress(raw) if name.endswith(".gz") else raw


SOURCES, DUPES = {}, {}


def _add(where, files):
    log = json.loads(next(v for k, v in files.items() if k.startswith("run_log_")))
    if log.get("study") != "v5":
        return
    key = log["model_tag"]
    sig = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    if key in SOURCES:
        if SOURCES[key][2] != sig:
            DUPES.setdefault(key, [SOURCES[key][0]]).append(where)
        return
    SOURCES[key] = (where, files, sig)


def _files(tag, read):
    rec = next((n for n in (f"records_{tag}.csv.gz", f"records_{tag}.csv") if read(n) is not None), None)
    gen = next((n for n in (f"generated_{tag}.csv.gz", f"generated_{tag}.csv") if read(n) is not None), None)
    if rec is None or gen is None or read(f"run_log_{tag}.json") is None or any(read(n) is None for n in CODE_FILES):
        return None
    out = {n: read(n) for n in CODE_FILES}
    out.update({f"run_log_{tag}.json": read(f"run_log_{tag}.json"), f"records_{tag}.csv": _csv_bytes(read(rec), rec),
                f"generated_{tag}.csv": _csv_bytes(read(gen), gen)})
    return out


for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        if _skipped(p) or not os.path.basename(p).startswith("v5_"):
            continue
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
            logs = [n for n in names if n.startswith("run_log_") and n.endswith(".json")]
            if len(logs) != 1:
                continue
            tag = logs[0][len("run_log_"):-len(".json")]
            f = _files(tag, lambda n: z.read(n) if n in names else None)
            if f:
                _add(p, f)
for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "run_log_*.json"), recursive=True)):
        if _skipped(p):
            continue
        folder, tag = os.path.dirname(p), os.path.basename(p)[len("run_log_"):-len(".json")]
        f = _files(tag, lambda n: open(os.path.join(folder, n), "rb").read()
                   if os.path.exists(os.path.join(folder, n)) else None)
        if f:
            _add(folder, f)
if DUPES:
    raise RuntimeError(f"Different files for the same run: {DUPES}. Attach only the final output of each run.")
if not SOURCES:
    raise RuntimeError("No fifth-study run found. Attach the v5_<model>.zip files (Kaggle: Add Input -> Upload).")
os.makedirs(OUT, exist_ok=True)
for tag, (where, files, sig) in sorted(SOURCES.items()):
    d = os.path.join(IN_DIR, tag)
    os.makedirs(d, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    print(f"{tag}: from {where}")
    for name in sorted(sig):
        print(f"    {name:34s} sha256 {sig[name][:16]}")

# %% [markdown]
# ## Load the runs and check that they are comparable
# Every run used the registered code (`ler_main.py`, `ler_v3.py` and `ler_v4.py` unchanged, one version of
# `ler_v5.py`), the same protocol text and engineering-check decisions, the registered model revision and grids, the
# same fresh patients and cards, and every prompt that two runs share has the same text. Each run is complete.

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


RUNS_TAGS = sorted(SOURCES)
first = RUNS_TAGS[0]
code = {tag: tuple(sha(os.path.join(IN_DIR, tag, n)) for n in CODE_FILES) for tag in RUNS_TAGS}
if len(set(code.values())) != 1:
    raise RuntimeError(f"The runs used different code: {code}")
sys.path.insert(0, os.path.join(IN_DIR, first))
for _m in ("ler_v5", "ler_v4", "ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_v3 as V
import ler_v4 as W4
import ler_v5 as F

RUNS, RECS, GENS = {}, {}, {}
for tag in RUNS_TAGS:
    d = os.path.join(IN_DIR, tag)
    RUNS[tag] = json.load(open(os.path.join(d, f"run_log_{tag}.json")))
    RECS[tag] = pd.read_csv(os.path.join(d, f"records_{tag}.csv"), dtype={"id": str}, low_memory=False)
    GENS[tag] = pd.read_csv(os.path.join(d, f"generated_{tag}.csv"), dtype={"id": str}, low_memory=False)
    GENS[tag]["parsed"] = GENS[tag]["parsed"].where(GENS[tag]["parsed"].notna(), None)
MOCK = any(RUNS[t].get("mock") for t in RUNS_TAGS)
problems = []
if code[first][:3] != (F.MAIN_SHA256, F.V3_SHA256, F.V4_SHA256):
    problems.append("ler_main.py, ler_v3.py or ler_v4.py is not the registered version")
CHECK = {"code_sha256": dict(zip(CODE_FILES, code[first])), "v5_version": F.V5_VERSION, "mock": MOCK}
psha = {t: json.dumps(RUNS[t]["protocol"]["sha256"], sort_keys=True) for t in RUNS_TAGS}
if len(set(psha.values())) != 1:
    problems.append(f"the runs checked different protocol texts: {psha}")
CHECK["protocol_sha256"] = RUNS[first]["protocol"]["sha256"]
if not MOCK and any(RUNS[t]["protocol"]["strings_missing"]["v5"] for t in RUNS_TAGS):
    problems.append("a run's protocol check reported missing strings")
DECISIONS = F.ENGCHECK_DECISIONS
dec_runs = {t: json.dumps(RUNS[t].get("decisions"), sort_keys=True) for t in RUNS_TAGS}
if len(set(dec_runs.values())) != 1:
    problems.append(f"the runs used different engineering-check decisions: {dec_runs}")
if DECISIONS is None and MOCK:
    DECISIONS = RUNS[first]["decisions"]
if DECISIONS is None:
    problems.append("the engineering-check decisions are not recorded in ler_v5.py")
elif json.dumps(DECISIONS, sort_keys=True) != dec_runs[first]:
    problems.append("the runs' decisions differ from those registered in ler_v5.py")
CHECK["decisions"] = DECISIONS
EXPECTED = [F.MODELS_V5[m]["tag"] for m in F.MODELS_V5 if m not in (DECISIONS or {}).get("not_run", [])]
for tag in RUNS_TAGS:
    r = RUNS[tag]
    if tag not in F.TAG2ID:
        problems.append(f"{tag} is not a registered model of the fifth study")
        continue
    mid = F.TAG2ID[tag]
    if not MOCK and r.get("model_revision") != F.MODELS_V5[mid]["revision"]:
        problems.append(f"{tag}: revision {r.get('model_revision')} is not the registered {F.MODELS_V5[mid]['revision']}")
    if r.get("mock") != MOCK:
        problems.append("mock and real runs are mixed")
    if DECISIONS is not None and tuple(r["grids"]) != F.model_grids(mid, DECISIONS):
        problems.append(f"{tag}: grids {r['grids']} are not the registered {F.model_grids(mid, DECISIONS)}")
fps = {t: json.dumps({ds: f["fingerprint"] for ds, f in RUNS[t]["fresh"].items()}, sort_keys=True) for t in RUNS_TAGS}
if len(set(fps.values())) != 1:
    problems.append("the runs used different fresh patients or cards (fingerprints differ)")
if not MOCK and json.loads(fps[first]) != F.FRESH_FP:
    problems.append("the fresh samples differ from the fingerprints recorded at registration")
allp = pd.concat([RECS[t][["key", "text_sha"]] for t in RUNS_TAGS]).drop_duplicates()
bad_keys = allp["key"][allp["key"].duplicated()].tolist()
if bad_keys:
    problems.append(f"{len(bad_keys)} prompts have different texts in different runs, e.g. {bad_keys[:3]}")
COMPLETE = {}
for tag in RUNS_TAGS:
    df, g = RECS[tag], GENS[tag]
    n_pat = df.groupby("dataset")["id"].nunique().to_dict()
    want = sum(n_pat.values()) * len(F.grid_v5(tuple(RUNS[tag]["grids"])))
    COMPLETE[tag] = {"records": int(len(df)), "expected": int(want), "patients": n_pat, "generated": int(len(g)),
                     "complete": len(df) == want and len(g) > 0 and not df["key"].duplicated().any()}
    if not COMPLETE[tag]["complete"]:
        problems.append(f"{tag}: {len(df)} records (expected {want}), {len(g)} generated answers")
    if not MOCK and any(n_pat.get(ds) != RUNS[tag]["fresh"][ds]["n"] for ds in F.DATASETS_V5):
        problems.append(f"{tag}: not every fresh patient was scored")
CHECK.update(runs=RUNS_TAGS, complete=COMPLETE, missing_registered_runs=[t for t in EXPECTED if t not in RUNS_TAGS])
if problems:
    raise RuntimeError("The runs are not comparable:\n" + "\n".join(problems))
N_BOOT = int(os.environ.get("LER_TEST_BOOT", F.N_BOOT))
N_BOOT_BOUND = int(os.environ.get("LER_TEST_BOOT", F.N_BOOT_BOUND))
if (N_BOOT != F.N_BOOT or N_BOOT_BOUND != F.N_BOOT_BOUND) and not MOCK:
    raise RuntimeError("LER_TEST_BOOT is for offline tests only")
CHECK["n_boot"], CHECK["n_boot_bound"] = N_BOOT, N_BOOT_BOUND
print(json.dumps(CHECK, indent=1, default=str))
print("All checks passed." + (" (MOCK RUNS: OFFLINE TEST ONLY)" if MOCK else ""))

# %% [markdown]
# ## Every cell
# Validity; O, W and the static lean of every pair against C0 and against the placebo card, at the threshold and at the
# neutral point; the change of the lean with the placebo card; the numbers card at three scales; the probability; the
# generated decisions (paired bootstrap over patients, 10,000 resamples, seed 2026).

# %%
t0 = time.time()


def analyse_all(recs, gens, n_boot, only=None):
    cells = {}
    for tag in RUNS_TAGS:
        df, g = recs[tag], gens[tag]
        for ds in F.DATASETS_V5:
            k = f"{tag}|{ds}"
            if only is not None and k not in only:
                continue
            cells[k] = F.analyse_cell_v5(df[df["dataset"] == ds], g[g["dataset"] == ds], n_boot=n_boot)
        print(f"  {tag}: done ({time.time() - t0:.0f} s)", flush=True)
    return cells


CELLS = analyse_all(RECS, GENS, N_BOOT)
CONF = F.confirmatory_v5(CELLS)
Q22 = {"R16_with_template_variance": F.r16_template(CELLS), "counts": F.template_counts(CELLS)}
Q21 = {"earlier": F.earlier_agreement(CELLS), "descriptives": {}, "generated": {}}
for tag in RUNS_TAGS:
    for ds in F.DATASETS_V5:
        k = f"{tag}|{ds}"
        df, g = RECS[tag], GENS[tag]
        Q21["descriptives"][k] = F.descriptives_v5(df[df["dataset"] == ds])
        Q21["generated"][k] = F.generated_summary_v5(g[g["dataset"] == ds], df[df["dataset"] == ds], N_BOOT)
print(f"analysis: {time.time() - t0:.0f} s")

# %% [markdown]
# ## The imputation bound of Deviation D5 (Q21f)
# For every run with imputed readings: in each such record, the probability of the less likely answer word is halved,
# and R15 and R16 are computed again (2,000 resamples). It decides nothing.

# %%
t0 = time.time()
IMPUTED = {t: float((RECS[t]["n_imputed"] > 0).mean()) for t in RUNS_TAGS}
BOUND = {"imputed_share": IMPUTED}
_conf_cells = set(F.R15_CELLS) | set(F.R16_CELLS)
if any(IMPUTED[k.split("|")[0]] > 0 for k in _conf_cells if k.split("|")[0] in IMPUTED):
    BREC = {t: (F.halve_less_likely_v5(d) if IMPUTED[t] > 0 else d) for t, d in RECS.items()}
    BCELLS = analyse_all(BREC, GENS, N_BOOT_BOUND, only=_conf_cells)
    BCONF = F.confirmatory_v5(BCELLS)
    BOUND["R15"], BOUND["R16"] = BCONF["R15"], BCONF["R16"]
    BOUND["changed"] = [f"{r} {k}: supported {CONF[r]['cells'][k]['supported']} -> {BCONF[r]['cells'][k]['supported']}"
                        for r in ("R15", "R16") for k in CONF[r]["cells"]
                        if CONF[r]["cells"][k]["supported"] != BCONF[r]["cells"][k]["supported"]]
print("imputed share per run:", {k: round(v, 3) for k, v in IMPUTED.items()}, f"| {time.time() - t0:.0f} s")

# %% [markdown]
# ## Results

# %%
def f2(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def pf(p):
    return "–" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |" for r in rows]
    return "\n".join(out)


yn = lambda b: "yes" if b else "no"
sgn = lambda r: ("toward “" + ("Yes" if r["sign"] > 0 else "No") + "”") if r and r.get("conclusive") else "not conclusive"
sgo = lambda r: ("toward " + ("alarm" if r["sign"] > 0 else "safe")) if r and r.get("conclusive") else "not conclusive"


def part_cells(r):
    if not r:
        return ["–"] * 4
    return [f"{f2(r['a'])} [{f2(r['a_ci'][0])}, {f2(r['a_ci'][1])}]", f"{f2(r['ap'])} [{f2(r['ap_ci'][0])}, {f2(r['ap_ci'][1])}]",
            pf(r["p"]), yn(r["conclusive"])]


REPORT = [f"# Fifth study: results (`prereg-v5`)\n", f"Runs: {', '.join(RUNS_TAGS)}. Missing registered runs: "
          f"{CHECK['missing_registered_runs'] or 'none'}. Bootstrap resamples: {N_BOOT}."
          + (" **MOCK RUNS: OFFLINE TEST ONLY.**" if MOCK else "") + "\n"]

rows = []
for k, c in CELLS.items():
    v = c["validity"]
    rows.append([k, f"{v['yn']['alarm_median_mass']:.3f}", f"{v['yn']['reversed_median_mass']:.3f}",
                 f"{v['yn']['c0_accuracy']:.3f}", yn(v["yn"]["passes"])] + [yn(v[p]["passes"]) for p in ("pa", "tf", "ab", "dose", "prob")])
REPORT += ["## Validity\n", md_table(["cell", "mass alarm", "mass reversed", "C0 acc.", "valid", "pa", "tf", "ab", "dose", "prob"],
                                     rows), ""]

R15 = CONF["R15"]
rows = []
for k, e in R15["cells"].items():
    lc = CELLS.get(k, {}).get("lean_change") or {}
    rows.append([k, yn(e["testable"]), f2(lc.get("lean_c0")), f2(lc.get("lean_cp")), f2(e.get("delta")),
                 f"[{f2((lc.get('delta_ci') or [None, None])[0])}, {f2((lc.get('delta_ci') or [None, None])[1])}]",
                 pf(e.get("p")), pf(e["p_holm"]), yn(e["supported"])])
REPORT += ["## R15. The placebo card and the answer prior (Qwen3-8B)\n",
           md_table(["cell", "valid", "λ(C0)", "λ(CP)", "Δλ", "95% interval", "p", "Holm p", "supported"], rows), "",
           f"**R15: {'supported' if R15['supported'] else 'not supported'}** (prediction: Δλ > 0 in both cells).\n"]
R16 = CONF["R16"]
rows = []
for k, e in R16["cells"].items():
    w = (((CELLS.get(k) or {}).get("pairs") or {}).get("yn") or {}).get("CP") or {}
    rows.append([k, "+" if e["predicted"] > 0 else "−", yn(e["testable"]), *part_cells(L._get(w, "neutral", "W"))[:2],
                 pf(e.get("p")), pf(e["p_holm"]), yn(e["supported"]), yn(e["opposite"])])
REPORT += ["## R16. W against the placebo card, at its neutral point\n",
           md_table(["cell", "predicted", "valid", "W (a)", "W (a′)", "p", "Holm p", "supported", "opposite"], rows), "",
           f"**R16: {'supported' if R16['supported'] else 'not supported'}** ({R16['n_supported']} of 4 cells supported; "
           f"needs at least 3 and none opposite).\n"]
R17 = CONF["R17"]
rows = [[k, "+" if e["predicted"] > 0 else "−", yn(e["testable"]), e.get("n", "–"), f2(e.get("E_W"), 3),
         f"[{f2((e.get('ci') or [None, None])[0], 3)}, {f2((e.get('ci') or [None, None])[1], 3)}]", yn(e["supported"])]
        for k, e in R17["cells"].items()]
REPORT += ["## R17. Generated decisions\n",
           md_table(["cell", "predicted", "testable", "patients", "E_W", "95% interval", "supported"], rows), "",
           f"**R17: {'supported' if R17['supported'] else 'not supported'}** ({R17['n_supported']} of 4 cells; needs at least 3).\n"]

rows = []
for k, c in CELLS.items():
    for pair, by in c["pairs"].items():
        for base, res in by.items():
            if not res:
                continue
            for rd in ("threshold", "neutral"):
                w, o = res[rd]["W"], res[rd]["O"]
                rows.append([k, pair, base, rd, res["n"], f2(res["lean"]), f2(w["a"]), f2(w["ap"]), pf(w["p"]), sgn(w),
                             f2(o["a"]), f2(o["ap"]), pf(o["p"]), res["support"][rd],
                             f2(res[rd]["delta_star"]) if res[rd]["delta_star"] is not None else "–", yn(res["readable"][rd])])
REPORT += ["## Q18 and Q21c. O and W of every pair, against no card (C0) and against the placebo card (CP)\n",
           "W < 0 is a pull toward the second answer word (“No”, “False”, “B”). p is the intersection–union p over "
           "patients (no correction). near: patients within 1 log-odds of the reading point in both framings; δ*: the "
           "share by which a card would have to change the prior to produce W; readable: conclusive, at least 30 near, "
           "δ* ≥ 0.5.\n",
           md_table(["cell", "pair", "baseline", "reading", "n", "λ", "W (a)", "W (a′)", "p", "W", "O (a)", "O (a′)", "p",
                     "near", "δ*", "readable"], rows), ""]
rows = []
for k, c in CELLS.items():
    d = c.get("dose")
    if d:
        rows.append([k, d["n"]] + [f2(d["movement"][s]["est"]) for s in ("A", "Anum05", "Anum1", "Anum2")] +
                    [f"{f2(d['dm']['est'])} [{f2(d['dm']['ci'][0])}, {f2(d['dm']['ci'][1])}]", pf(d["dm"]["p"])])
REPORT += ["## Q19. The size of the impacts\n", "Mean movement m = mean(la − ls) in the alarm framing; Δm = m(×2) − m(×0.5).\n",
           md_table(["cell", "n", "registered card", "numbers ×0.5", "×1", "×2", "Δm", "p"], rows) if rows else "No valid cell.", ""]
rows = []
for k, c in CELLS.items():
    pr = c.get("prob")
    if pr:
        for base in ("C0", "CP"):
            res = pr.get(base)
            if res:
                for rd in ("threshold", "neutral"):
                    rows.append([k, base, rd, res["n"], f2(res[rd]["W"]["a"]), f2(res[rd]["W"]["ap"]), sgn(res[rd]["W"]),
                                 f2(res[rd]["O"]["a"]), f2(res[rd]["O"]["ap"]), sgo(res[rd]["O"])])
REPORT += ["## Q20. A probability instead of a card\n",
           md_table(["cell", "baseline", "reading", "n", "W (a)", "W (a′)", "W", "O (a)", "O (a′)", "O"], rows) if rows else "No valid cell.", ""]

rows = []
for k, c in CELLS.items():
    lc = c.get("lean_change") or {}
    rows.append([k, f2(lc.get("lean_c0")), f2(lc.get("lean_cp")), f2(lc.get("delta")), pf(lc.get("p")),
                 f2(lc.get("placebo_outcome_shift")), f"[{f2((lc.get('placebo_outcome_shift_ci') or [None, None])[0])}, "
                 f"{f2((lc.get('placebo_outcome_shift_ci') or [None, None])[1])}]"])
REPORT += ["## Q21. Diagnostics\n", "**a. The placebo card's effect on the static lean and on the outcome:**\n",
           md_table(["cell", "λ(C0)", "λ(CP)", "Δλ", "p", "CP − C0 (alarm log-odds)", "95% interval"], rows), ""]
rows = [[k] + [f"{v[f'{rd}|{p}']['earlier']:+d} → {v[f'{rd}|{p}']['fresh']:+d}" for rd in ("threshold", "neutral") for p in ("W", "O")]
        for k, v in Q21["earlier"]["cells"].items() if v["compared"]]
REPORT += ["**b. The Yes/No pair against C0: earlier patients → fresh patients** (+1 / −1 conclusive sign, 0 not conclusive):\n",
           md_table(["cell", "W threshold", "O threshold", "W neutral", "O neutral"], rows), "",
           md_table(["reading", "earlier conclusive", "same sign", "opposite", "not conclusive now"],
                    [[k, t["earlier_conclusive"], t["same"], t["opposite"], t["not_conclusive"]]
                     for k, t in Q21["earlier"]["tally"].items()]), ""]
rows = [[k, g["n"], f"{g['parsed_share']:.3f}", f"{g['agreement']:.3f}", g["n_full_patients"],
         f2(g["E_W_generated"]["E_W"], 3), f2(g["E_W_first_token"]["E_W"], 3)] for k, g in Q21["generated"].items()]
REPORT += ["**d. Generated answers** (agreement with the first-token reading among parsed answers; E_W from the generated "
           "answers and from the first-token reading of the same patients):\n",
           md_table(["cell", "answers", "parsed", "agreement", "patients with 6 parsed", "E_W generated", "E_W first token"], rows), ""]
rows = []
for k, d in Q21["descriptives"].items():
    for fs, x in d.items():
        rows.append([k, fs, x["n"], x["invalid"], f"{x['median_mass']:.3f}", f"{x['imputed_share']:.3f}",
                     ", ".join(f"{c} {v:.2f}" for c, v in x["accuracy"].items()), ", ".join(f"{c} {v:.2f}" for c, v in x["follow"].items())])
REPORT += ["**e. Per framing and set:** invalid records, median answer-word mass, imputed share, accuracy and the share of "
           "answers that follow the evidence, by condition:\n",
           md_table(["cell", "framing | set", "n", "invalid", "mass", "imputed", "accuracy", "follows the evidence"], rows), ""]
REPORT += ["**f. Imputation bound.** Share of records with an imputed variant: " +
           ", ".join(f"{k} {v:.3f}" for k, v in IMPUTED.items()) + "."]
if "R15" in BOUND:
    REPORT += [f"Under the bound: R15 {'supported' if BOUND['R15']['supported'] else 'not supported'}, R16 "
               f"{'supported' if BOUND['R16']['supported'] else 'not supported'} ({BOUND['R16']['n_supported']} of 4). "
               "Cells whose decision changes: " + ("; ".join(BOUND["changed"]) or "none") + ".\n"]
T = Q22["R16_with_template_variance"]
REPORT += ["## Q22. With the template variance of the fourth study\n",
           md_table(["R16 cell", "predicted", "p (template)", "Holm p", "supported", "opposite"],
                    [[k, "+" if e["predicted"] > 0 else "−", pf(e.get("p")), pf(e["p_holm"]), yn(e["supported"]), yn(e["opposite"])]
                     for k, e in T["cells"].items()]), "",
           f"R16 with this test: {T['n_supported']} of 4 cells supported; it would be "
           f"{'supported' if T['would_be_supported'] else 'not supported'} (no decision rule).\n",
           md_table(["pair | baseline | reading | part", "conclusive", "stay conclusive"],
                    [[k, v["conclusive"], v["stay_conclusive"]] for k, v in Q22["counts"].items() if v["conclusive"]]), ""]
REPORT[2:2] = ["## Decisions\n",
               f"- **R15** (the placebo card weakens Qwen3-8B's preference for “No”): {'supported' if R15['supported'] else 'not supported'}.",
               f"- **R16** (W against the placebo card keeps the family-specific direction): {'supported' if R16['supported'] else 'not supported'} "
               f"({R16['n_supported']} of 4 cells).",
               f"- **R17** (the effect appears in generated decisions): {'supported' if R17['supported'] else 'not supported'} "
               f"({R17['n_supported']} of 4 cells).", ""]
REPORT_MD = "\n".join(REPORT)
print(REPORT_MD)

# %% [markdown]
# ## Save and zip

# %%
RESULTS = {"check": CHECK, "cells": CELLS, "confirmatory": CONF, "q21": Q21, "q22": Q22, "bound": BOUND,
           "minutes": round((time.time() - T0) / 60, 1)}
with open(os.path.join(OUT, "results_v5.json"), "w") as f:
    json.dump(RESULTS, f, indent=1, default=float)
with open(os.path.join(OUT, "report_v5.md"), "w", encoding="utf-8") as f:
    f.write(REPORT_MD + "\n")
ZIP = os.path.join(WORK, "analysis_v5_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("results_v5.json", "report_v5.md"):
        z.write(os.path.join(OUT, name), name)
    z.write(os.path.join(IN_DIR, first, "ler_v5.py"), "ler_v5.py")
print("Saved:", ZIP, f"| {(time.time() - T0) / 60:.1f} min")
