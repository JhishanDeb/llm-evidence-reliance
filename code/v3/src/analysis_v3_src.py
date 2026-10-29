# %% [markdown]
# # Registered analysis, third study (`prereg-v3`)
#
# **What this notebook does.** It reads the six `v3_<model>.zip` files of the third study, checks that every run saw
# the same patients and prompts and used the registered code, models and protocol, and runs the analysis registered
# in [`PREREGISTRATION_V3.md`](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v3):
# R4 (the answer-word part in new families), R5 (the outcome part in new families, checked against the answer
# options), R6 (the answer-word part with each changed card), R7 (the outcome part with each reduced card), and the
# descriptive questions Q8. If the earlier runs of the three earlier models are attached too (`main_qwen-instruct.zip`,
# `main_llama-instruct.zip`, `v2_olmo3-instruct.zip` from the repository's `results/`), it also compares their
# readings with this study's readings of the same prompts (Q8a); no decision depends on them.
#
# **How to run.** It needs no GPU and no internet. On Kaggle: upload the zip files as one dataset (Add Input →
# Upload → New Dataset) and Save & Run All; Kaggle unpacks zip files, and the first code cell accepts either the zip
# files or the unpacked folders. Locally or on Colab: put the zip files next to the notebook and run all cells.
# Runtime: about 15 to 30 minutes on a CPU. Download `analysis_v3_results.zip` at the end.

# %%
import os, sys, json, glob, gzip, zipfile, hashlib, time
import numpy as np
import pandas as pd

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
IN_DIR = os.path.join(WORK, "analysis_v3_inputs")
OUT = os.path.join(WORK, "analysis_v3_results")
SKIP = [os.path.realpath(IN_DIR), os.path.realpath(OUT)]


def _skipped(p):
    rp = os.path.realpath(p)
    return any(rp == s or rp.startswith(s + os.sep) for s in SKIP)


def _csv_bytes(raw, name):
    return gzip.decompress(raw) if name.endswith(".gz") else raw


# A run arrives as a zip file or as an unpacked folder. Each source is reduced to the files the analysis reads; the
# study of a run comes from its run log ('v3', 'v2' or the first study).
SOURCES, DUPES = {}, {}


def _add(where, files):
    log = json.loads(next(v for k, v in files.items() if k.startswith("run_log_")))
    key = (log.get("study") or "v1", log["model_tag"])
    sig = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    if key in SOURCES:
        if SOURCES[key][2] != sig:
            DUPES.setdefault(key, [SOURCES[key][0]]).append(where)
        return
    SOURCES[key] = (where, files, sig)


def _files(tag, read):
    rec = next((n for n in (f"records_{tag}.csv.gz", f"records_{tag}.csv") if read(n) is not None), None)
    if rec is None or read("ler_main.py") is None or read(f"run_log_{tag}.json") is None:
        return None
    out = {"ler_main.py": read("ler_main.py"), f"run_log_{tag}.json": read(f"run_log_{tag}.json"),
           f"records_{tag}.csv": _csv_bytes(read(rec), rec)}
    if read("ler_v3.py") is not None:
        out["ler_v3.py"] = read("ler_v3.py")
    return out


for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        if _skipped(p) or not os.path.basename(p).startswith(("v3_", "v2_", "main_")):
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
if not any(s == "v3" for s, _ in SOURCES):
    raise RuntimeError("No third-study run found. Attach the v3_<model>.zip files (Kaggle: Add Input -> Upload).")
os.makedirs(OUT, exist_ok=True)
for (study, tag), (where, files, sig) in sorted(SOURCES.items()):
    d = os.path.join(IN_DIR, study, tag)
    os.makedirs(d, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    print(f"{study} {tag}: from {where}")
    for name in sorted(sig):
        print(f"    {name:34s} sha256 {sig[name][:16]}")

# %% [markdown]
# ## Load the runs and check that they are comparable
# Every third-study run used the registered code (`ler_main.py` main-2.0 unchanged, one version of `ler_v3.py`), the
# same protocol text, the registered model revision, the same patients, cards and solved examples, and every prompt
# that two runs share has the same text. Each run is complete.

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


V3RUNS = sorted(t for s, t in SOURCES if s == "v3")
first = V3RUNS[0]
code = {tag: (sha(os.path.join(IN_DIR, "v3", tag, "ler_main.py")), sha(os.path.join(IN_DIR, "v3", tag, "ler_v3.py")))
        for tag in V3RUNS}
if len(set(code.values())) != 1:
    raise RuntimeError(f"The third-study runs used different code: {code}")
sys.path.insert(0, os.path.join(IN_DIR, "v3", first))
for _m in ("ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_v3 as V

RUNS, RECS = {}, {}
for (study, tag) in SOURCES:
    d = os.path.join(IN_DIR, study, tag)
    RUNS[(study, tag)] = json.load(open(os.path.join(d, f"run_log_{tag}.json")))
    RECS[(study, tag)] = pd.read_csv(os.path.join(d, f"records_{tag}.csv"), dtype={"id": str}, low_memory=False)
MOCK = any(RUNS[("v3", t)].get("mock") for t in V3RUNS)
problems = []
if code[first][0] != V.MAIN_SHA256 or L.CODE_VERSION != "main-2.0":
    problems.append("ler_main.py is not the registered main-2.0")
CHECK = {"ler_main_sha256": code[first][0], "ler_v3_sha256": code[first][1], "v3_version": V.V3_VERSION,
         "mock": MOCK}
psha = {t: RUNS[("v3", t)]["protocol"]["sha256"] for t in V3RUNS}
if len(set(psha.values())) != 1:
    problems.append(f"the runs checked different protocol texts: {psha}")
CHECK["protocol_sha256"] = psha[first]
TAG2ID = {m["tag"]: mid for mid, m in V.REGISTERED_MODELS_V3.items()}
if MOCK:
    TAG2ID.update({c["tag"]: mid for mid, c in V.CANDIDATES_V3.items()})
REGISTERED_TAGS = [m["tag"] for m in V.REGISTERED_MODELS_V3.values()]
for tag in V3RUNS:
    r = RUNS[("v3", tag)]
    if tag not in TAG2ID:
        problems.append(f"{tag} is not a registered model of the third study")
        continue
    mid = TAG2ID[tag]
    reg = (V.REGISTERED_MODELS_V3.get(mid) or {}).get("revision")
    if not MOCK and r.get("model_revision") != reg:
        problems.append(f"{tag}: revision {r.get('model_revision')} is not the registered {reg}")
    if r.get("mock") != MOCK:
        problems.append("mock and real runs are mixed")
fps = {t: json.dumps(RUNS[("v3", t)]["fingerprints"], sort_keys=True) for t in V3RUNS}
if len(set(fps.values())) != 1:
    problems.append("the runs used different patients, cards or solved examples (fingerprints differ)")
if not MOCK:
    for ds, exp in L.DEV_RUN["fingerprints"].items():
        got = RUNS[("v3", first)]["fingerprints"][ds]
        if any(got[k] != v for k, v in exp.items()):
            problems.append(f"{ds}: fingerprints differ from the dev phase")
for st, tag in SOURCES:          # earlier runs: same test patients and cards as this study
    if st != "v3":
        f_old, f_new = RUNS[(st, tag)]["fingerprints"], RUNS[("v3", first)]["fingerprints"]
        if any(f_old[ds].get(k) != f_new[ds].get(k) for ds in L.DATASETS for k in ("test_ids", "test_cards")):
            problems.append(f"earlier run {st} {tag} used other test patients or cards")
allp = pd.concat([RECS[("v3", t)][["key", "text_sha"]] for t in V3RUNS]).drop_duplicates()
bad_keys = allp["key"][allp["key"].duplicated()].tolist()
if bad_keys:
    problems.append(f"{len(bad_keys)} prompts have different texts in different runs, e.g. {bad_keys[:3]}")
COMPLETE = {}
for tag in V3RUNS:
    df = RECS[("v3", tag)]
    grids = (V.REGISTERED_MODELS_V3.get(TAG2ID.get(tag)) or {}).get("grids") or \
            (V.NEW_GRIDS if tag in [c["tag"] for c in V.CANDIDATES_V3.values()] else V.EARLIER_GRIDS)
    n_pat = df.groupby("dataset")["id"].nunique().to_dict()
    want = sum(n_pat.values()) * len(V.grid_v3(grids))
    COMPLETE[tag] = {"records": int(len(df)), "expected": int(want), "patients": n_pat, "complete": len(df) == want}
    if len(df) != want:
        problems.append(f"{tag}: {len(df)} records, expected {want}")
CHECK.update(runs=V3RUNS, earlier_runs=sorted(f"{s} {t}" for s, t in SOURCES if s != "v3"), complete=COMPLETE,
             missing_registered_runs=[t for t in REGISTERED_TAGS if t not in V3RUNS])
if problems:
    raise RuntimeError("The runs are not comparable:\n" + "\n".join(problems))
N_BOOT = int(os.environ.get("LER_TEST_BOOT", V.N_BOOT))
if N_BOOT != V.N_BOOT and not MOCK:
    raise RuntimeError("LER_TEST_BOOT is for offline tests only")
CHECK["n_boot"] = N_BOOT
print(json.dumps(CHECK, indent=1, default=str))
print("All checks passed." + (" (MOCK RUNS: OFFLINE TEST ONLY)" if MOCK else ""))

# %% [markdown]
# ## Every model and dataset
# Validity, the split with the registered card, each changed card against the registered card (paired bootstrap over
# patients, 10,000 resamples, seed 2026), descriptives, and the answer options of the new families.

# %%
t0 = time.time()
CELLS = {}
for tag in V3RUNS:
    df = RECS[("v3", tag)]
    for ds in L.DATASETS:
        CELLS[f"{tag}|{ds}"] = V.analyse_cell_v3(df[df["dataset"] == ds], n_boot=N_BOOT)
    print(f"{tag}: done ({time.time() - t0:.0f} s)", flush=True)
ROLE = {t: ("new" if t in [c["tag"] for c in V.CANDIDATES_V3.values()] else "earlier") for t in V3RUNS}
NEW_TAGS = [m["tag"] for m in V.REGISTERED_MODELS_V3.values() if m["role"] == "new"]
if MOCK and not NEW_TAGS:
    NEW_TAGS = [t for t in V3RUNS if ROLE[t] == "new"]
CONF = V.confirmatory_v3(CELLS, NEW_TAGS)
REPLAY = {}
for tag in V3RUNS:
    for st in ("v1", "v2"):
        if (st, tag) in RECS and ROLE[tag] == "earlier":
            old = RECS[(st, tag)]
            if (old["shot_set"] == "A").any() and (old["framing"] == "reversed").any():
                REPLAY[tag] = dict(V.replay_check(RECS[("v3", tag)], old), earlier_study=st)
print(f"analysis: {time.time() - t0:.0f} s")

# %% [markdown]
# ## Results
# Below: validity; the two parts with the registered card; R4 to R7 as registered; the descriptive questions.

# %%
def f2(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def pf(p):
    return "–" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def part_row(r):
    if not r:
        return ["–"] * 3
    return [f"{f2(r['a']['est'])} [{f2(r['a']['ci'][0])}, {f2(r['a']['ci'][1])}]",
            f"{f2(r['a_prime']['est'])} [{f2(r['a_prime']['ci'][0])}, {f2(r['a_prime']['ci'][1])}]", pf(r["p"])]


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


REPORT = [f"# Third study: results (`prereg-v3`)\n", f"Runs: {', '.join(V3RUNS)}. Missing registered runs: "
          f"{CHECK['missing_registered_runs'] or 'none'}. Bootstrap resamples: {N_BOOT}."
          + (" **MOCK RUNS: OFFLINE TEST ONLY.**" if MOCK else "") + "\n"]
rows = []
for k, c in CELLS.items():
    v = c["validity"]
    rows.append([k, f"{v['alarm_median_mass']:.3f}", f"{v['reversed_median_mass']:.3f}", f"{v['c0_accuracy']:.3f}",
                 "yes" if v["passes"] else "no"] +
                [f"{v['formats'][f]['median_mass']:.3f}" if v["formats"][f]["n_records"] else "–" for f in V.CHANGED] +
                [("yes" if c["choice"]["validity"]["passes"] else "no") if c.get("choice") else "–"])
REPORT += ["## Validity\n", md_table(["cell", "mass alarm", "mass reversed", "C0 acc.", "valid", "mass words",
                                      "mass numbers", "mass cue", "mass matched", "answer options valid"], rows), ""]
rows = []
for k, c in CELLS.items():
    s = L._get(c, "split", "formats", "full") or {}
    rows.append([k, "yes" if c["validity"]["passes"] else "no", *part_row(s.get("O")), *part_row(s.get("W")),
                 f2(L._get(s, "movement", "est"))])
REPORT += ["## The two parts with the registered card\n",
           "O = (s_alarm + s_reversed) / 2 and W = (s_alarm − s_reversed) / 2, for s = a and s = a′ (95% intervals); "
           "W < 0 is a pull toward the answer “No”. p is the intersection–union p (no correction).\n",
           md_table(["cell", "valid", "O (a)", "O (a′)", "p", "W (a)", "W (a′)", "p", "movement"], rows), ""]

R4, R5, R6, R7 = CONF["R4"], CONF["R5"], CONF["R6"], CONF["R7"]
rows = [[k, "yes" if e["included"] else "no", f2(e["W_a"]), f2(e["W_a_prime"]), pf(e["p_used"]), pf(e["p_holm"]),
         "yes" if e["supported"] else "no"] for k, e in R4["cells"].items()]
REPORT += ["## R4. The answer-word part in the new families\n", md_table(["cell", "valid", "W (a)", "W (a′)", "p",
                                                                         "Holm p", "supported"], rows), "",
           "Per model: " + "; ".join(f"{t}: {v['supported_cells']} of {v['valid_cells']} valid cells, "
                                     f"{v['opposite_cells']} in the other direction, shows the pull: "
                                     f"{'yes' if v['shows_pull'] else 'no'}" for t, v in R4["per_model"].items()) + ".\n",
           f"**R4: {'supported' if R4['supported'] else ('not testable' if not R4['testable'] else 'not supported')}** "
           f"({R4['models_showing_pull']} of {len(NEW_TAGS)} new models show the pull toward “No”).\n"]
rows = [[k, "yes" if e["included"] else "no", f2(e["O_a"]), f2(e["O_a_prime"]), pf(e["p_used"]), pf(e["p_holm"]),
         {1: "toward alarm", -1: "toward safe", 0: "–"}[e["sign"]] if e["supported"] else "not shown"]
        for k, e in R5["cells"].items()]
REPORT += ["## R5. The outcome part in the new families, and the answer options\n",
           md_table(["cell", "valid", "O (a)", "O (a′)", "p", "Holm p", "direction"], rows), "",
           md_table(["cell", "sign of O", "sign with answer options", "agree"],
                    [[k, v["O_sign"], v["answer_option_sign"], "yes" if v["agree"] else "no"]
                     for k, v in R5["agreement"].items()]) if R5["agreement"] else "No cell with both conclusive.", "",
           f"**R5: {'supported' if R5['supported'] else ('not testable' if not R5['testable'] else 'not supported')}** "
           f"({R5['n_agree']} of {R5['n_compared']} cells agree).\n"]
for name, R, fmts in (("R6. The answer-word part W with each changed card", R6, V.CHANGED),
                      ("R7. The outcome part O with each reduced card", R7, V.CHANGED)):
    REPORT += [f"## {name}\n", "Eligible cells: " + ", ".join(f"{k} ({'+' if e['sign'] > 0 else '−'}; {e['source']})"
                                                             for k, e in R["eligible"].items()) + ".\n"]
    for f in fmts:
        F = R["formats"][f]
        rows = [[k, "yes" if e["included"] else "no", f2(e["a"]), f2(e["a_prime"]), pf(e["p_used"]), pf(e["p_holm"]),
                 "yes" if e["supported"] else "no", f2(e["change_a"]), f2(e["change_a_prime"]),
                 "yes" if e["reduced"] else "no"] for k, e in F["cells"].items()]
        REPORT += [f"**Card: {f}.** Kept in {F['n_kept']} of {F['n_eligible']} eligible cells (kept with this card: "
                   f"{'yes' if F['kept'] else 'no'}); smaller than with the registered card in {F['n_reduced']}.\n",
                   md_table(["cell", "valid", "a", "a′", "p", "Holm p", "kept", "change (a)", "change (a′)", "smaller"],
                            rows), ""]
REPORT += [f"**R6: {'supported' if R6['supported'] else ('not testable' if not R6['testable'] else 'not supported')}** "
           "(the answer-word part is kept with all four changed cards).\n",
           f"**R7:** {R7['interpretation']}; cue: {R7['cue']}.\n"]
S = CONF["sensitivity_prelisted_only"]
REPORT += ["Sensitivity (pre-listed cells only): R6 " + ("supported" if S["R6"]["supported"] else "not supported") +
           f"; R7 {S['R7']['interpretation']}; cue: {S['R7']['cue']}.\n"]
rows = []
for k, c in CELLS.items():
    for f, comp in c["formats"].items():
        if not comp:
            continue
        r = comp["formats"][f]
        rows.append([k, f, *part_row(r["alarm"])[:2], f2(r["movement"]["est"]),
                     f2(L._get(r, "relative_O", "a", "est")), f2(L._get(r, "relative_W", "a", "est"))])
REPORT += ["## Q8. Descriptive\n", "**Q8b–c.** The alarm-framing statistic of the first study, the movement "
           "mean(la − ls), and the parts relative to the movement, per card:\n",
           md_table(["cell", "card", "alarm a", "alarm a′", "movement", "O / movement (a)", "W / movement (a)"], rows), ""]
if REPLAY:
    REPORT += ["**Q8a.** Registered-card prompts scored again, against the earlier study:\n",
               md_table(["model", "earlier study", "prompts", "text differs", "max |Δ|", "median |Δ|", "within 0.01",
                         "same answer"], [[t, r["earlier_study"], r["n_compared"], r["n_text_differs"],
                                           f"{r['max_abs_diff']:.4f}", f"{r['median_abs_diff']:.4f}",
                                           f"{r['share_within_0_01']:.3f}", f"{r['share_same_answer']:.3f}"]
                                          for t, r in REPLAY.items()]), ""]
else:
    REPORT += ["**Q8a.** The earlier runs were not attached; the replay comparison was not made.\n"]
rows = []
for tag in NEW_TAGS:
    ms = (RUNS.get(("v3", tag)) or {}).get("memcheck_summary") or {}
    for ds in L.DATASETS:
        m = ms.get(ds) or {}
        rows.append([tag, ds, m.get("file"), m.get("header_matched_chars"), m.get("row_exact_matches"),
                     "yes" if m.get("flag_verbatim_rows") else "no"])
REPORT += ["**Q8e.** Memorization check of the new models (tabmemcheck; flag = at least one row completed verbatim):\n",
           md_table(["model", "dataset", "file", "header characters matched", "rows completed verbatim", "flag"], rows), ""]
_dec = lambda R: "supported" if R["supported"] else ("not testable" if not R["testable"] else "not supported")
REPORT[2:2] = ["## Decisions\n",
               f"- **R4** (answer-word part in the new families): {_dec(R4)}.",
               f"- **R5** (outcome part agrees with the answer options): {_dec(R5)}.",
               f"- **R6** (answer-word part kept with every changed card): {_dec(R6)}; kept with: " +
               ", ".join(f for f in V.CHANGED if R6["formats"][f]["kept"]) + ".",
               f"- **R7** (outcome part): {R7['interpretation']}; cue: {R7['cue']}.", ""]
REPORT_MD = "\n".join(REPORT)
print(REPORT_MD)

# %% [markdown]
# ## Save and zip

# %%
RESULTS = {"check": CHECK, "cells": CELLS, "confirmatory": CONF, "replay": REPLAY, "new_models": NEW_TAGS,
           "roles": ROLE, "minutes": round((time.time() - T0) / 60, 1)}
with open(os.path.join(OUT, "results_v3.json"), "w") as f:
    json.dump(RESULTS, f, indent=1, default=float)
with open(os.path.join(OUT, "report_v3.md"), "w", encoding="utf-8") as f:
    f.write(REPORT_MD + "\n")
ZIP = os.path.join(WORK, "analysis_v3_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("results_v3.json", "report_v3.md"):
        z.write(os.path.join(OUT, name), name)
    z.write(os.path.join(IN_DIR, "v3", first, "ler_v3.py"), "ler_v3.py")
print("Saved:", ZIP, f"| {(time.time() - T0) / 60:.1f} min")
