# %% [markdown]
# # Registered analysis, second study (`prereg-v2`)
#
# **What this notebook does.** It reads the eight `v2_<model>.zip` files of the second study and the two first-study
# instruct runs (`main_qwen-instruct.zip`, `main_llama-instruct.zip`, whose alarm-framing readings enter the R2
# framing comparison), checks that every run saw the same patients and prompts, and runs the analysis registered in
# [`PREREGISTRATION_V2.md`](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v2):
# R1 (replication), R2 (outcome or answer word), R3 (instruction tuning in Olmo 3), the Olmo 3 stage trace, and the
# labelled secondary analyses.
#
# **How to run.** It needs no GPU and no internet. On Kaggle: upload the ten zip files as one dataset (Add Input →
# Upload → New Dataset) and Save & Run All; Kaggle unpacks zip files, and the first code cell accepts either the zip
# files or the unpacked folders. Locally or on Colab: put the ten zip files next to the notebook and run all cells.
# Runtime: about 1 to 1.5 hours on a CPU (10,000 bootstrap resamples per test). Download
# `analysis_v2_results.zip` at the end.

# %%
import os, sys, json, glob, gzip, zipfile, hashlib, time
import numpy as np
import pandas as pd

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
IN_DIR = os.path.join(WORK, "analysis_v2_inputs")
OUT = os.path.join(WORK, "analysis_v2_results")
SKIP = [os.path.realpath(IN_DIR), os.path.realpath(OUT)]


def _skipped(p):
    rp = os.path.realpath(p)
    return any(rp == s or rp.startswith(s + os.sep) for s in SKIP)


def _csv_bytes(raw, name):
    return gzip.decompress(raw) if name.endswith(".gz") else raw


# A run arrives as a zip file or as an unpacked folder (Kaggle unpacks zip files when it builds a dataset). Each source
# is reduced to the three files the analysis reads. The study of a run (first or second) comes from its run log.
SOURCES, DUPES = {}, {}


def _add(where, files):
    log = json.loads(next(v for k, v in files.items() if k.startswith("run_log_")))
    key = ("v2" if log.get("study") == "v2" else "v1", log["model_tag"])
    sig = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    if key in SOURCES:
        if SOURCES[key][2] != sig:
            DUPES.setdefault(key, [SOURCES[key][0]]).append(where)
        return
    SOURCES[key] = (where, files, sig)


def _files(tag, read):
    rec = next((n for n in (f"records_{tag}.csv.gz", f"records_{tag}.csv") if read(n) is not None), None)
    if rec is None or read("ler_main.py") is None:
        return None
    return {"ler_main.py": read("ler_main.py"), f"run_log_{tag}.json": read(f"run_log_{tag}.json"),
            f"records_{tag}.csv": _csv_bytes(read(rec), rec)}


for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        if _skipped(p) or not os.path.basename(p).startswith(("v2_", "main_")):
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
    raise RuntimeError("No run found. Attach the zip files (Kaggle: Add Input -> Upload) or put them next to this notebook.")
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
# Every second-study run used the same code and the same protocol text; every run of both studies used the same
# patients, cards and partners, and every prompt shared by two runs has the same text.

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


V2 = sorted(t for s, t in SOURCES if s == "v2")
V1 = sorted(t for s, t in SOURCES if s == "v1")
code_sha = {tag: sha(os.path.join(IN_DIR, "v2", tag, "ler_main.py")) for tag in V2}
if len(set(code_sha.values())) != 1:
    raise RuntimeError(f"The second-study runs used different versions of ler_main.py: {code_sha}")
first = V2[0]
sys.path.insert(0, os.path.join(IN_DIR, "v2", first))
if "ler_main" in sys.modules:
    del sys.modules["ler_main"]
import ler_main as L

RUNS, RECS = {}, {}
for study, tags in (("v2", V2), ("v1", V1)):
    for tag in tags:
        d = os.path.join(IN_DIR, study, tag)
        RUNS[(study, tag)] = json.load(open(os.path.join(d, f"run_log_{tag}.json")))
        RECS[(study, tag)] = pd.read_csv(os.path.join(d, f"records_{tag}.csv"), dtype={"id": str})
# the two first-study instruct runs are needed only for the R2 framing comparison
V1_NEEDED = [L.REGISTERED_MODELS_V2[m]["tag"] for m in L.REGISTERED_MODELS_V2 if L.REGISTERED_MODELS_V2[m]["role"] == "v1"]
for tag in V1:
    if tag not in V1_NEEDED:
        del RUNS[("v1", tag)], RECS[("v1", tag)]

CHECK = {"code_version": L.CODE_VERSION, "code_sha256": code_sha[first]}
problems = []
if L.CODE_VERSION != "main-2.0":
    problems.append(f"unexpected code version {L.CODE_VERSION}")
for (study, tag), r in RUNS.items():
    if study == "v1" and r.get("code_sha256", "")[:16] != "10557671c645bb2e":
        problems.append(f"first-study run {tag} did not use main-1.0")
    if (study == "v2" and r.get("registered_revision") and not (r.get("mock") or r.get("local_model"))
            and r.get("model_revision") != r.get("registered_revision")):
        problems.append(f"{tag}: revision {r.get('model_revision')} differs from the registered one")
if len({r["protocol"]["sha256"] for (s, t), r in RUNS.items() if s == "v2"}) != 1:
    problems.append("the second-study runs checked different protocol texts")
fps = {k: json.dumps(r["fingerprints"], sort_keys=True) for k, r in RUNS.items()}
if len(set(fps.values())) != 1:
    problems.append("different patients, features, cards or partners")
_sha_by_key = {}
for k, df in RECS.items():
    for key, ts in zip(df["key"], df["text_sha"]):
        if _sha_by_key.setdefault(key, ts) != ts:
            problems.append(f"prompt {key} has different texts in different runs")
            break
if len({r.get("read_mode") for r in RUNS.values()}) > 1:
    problems.append("WARNING: the runs did not all use the same reading mode (see the Runs table)")
if any(r.get("mock") or r.get("local_model") for r in RUNS.values()):
    problems.append("TEST RUN: at least one input comes from a mock or local test model")
CHECK["problems"] = problems
CHECK["models"] = {f"{s}|{t}": {"model_id": r["model_id"], "revision": r["model_revision"],
                                "backend": r["backend_info"].get("backend"), "read_mode": r.get("read_mode"),
                                "grids": r.get("grids", ["yesno (first study)"]), "n_records": r["n_records"],
                                "n_invalid": int((RECS[(s, t)]["valid"] == 0).sum()),
                                "n_imputed": int((RECS[(s, t)]["n_imputed"] > 0).sum())}
                   for (s, t), r in RUNS.items()}
CHECK["missing_runs"] = [m["tag"] for m in L.REGISTERED_MODELS_V2.values() if m["tag"] not in V2] + \
                        [f"first-study {t}" for t in V1_NEEDED if ("v1", t) not in RUNS]
print(json.dumps(CHECK, indent=1, default=str))
_missing_v1 = [t for t in V1_NEEDED if ("v1", t) not in RUNS]
if _missing_v1:
    raise RuntimeError(f"The first-study runs {_missing_v1} are needed (main_<model>.zip from results/main/). Attach them.")
if [p for p in problems if not p.startswith(("TEST RUN", "WARNING"))]:
    raise RuntimeError("The runs are not comparable: " + "; ".join(problems))

# %% [markdown]
# ## Every measure for every model × dataset cell
# Yes/No grid: the first study's measures (alarm framing, example set A unless stated). Answer-option grid: the
# averaged reading of the two option orders. 10,000 bootstrap resamples per test.

# %%
t0 = time.time()
TAGINFO = {m["tag"]: dict(m, model_id=mid) for mid, m in L.REGISTERED_MODELS_V2.items()}
CELLS, CHOICE_CELLS = {}, {}


def combined(tag):
    """Second-study records of a model, plus its first-study records for the two first-study models (their alarm
    Yes/No readings enter the framing and example-order comparisons)."""
    df = RECS[("v2", tag)]
    return pd.concat([RECS[("v1", tag)], df], ignore_index=True) if TAGINFO[tag]["role"] == "v1" else df


for tag in V2:
    df = RECS[("v2", tag)]
    if "yesno" in TAGINFO[tag]["grids"]:
        for ds in L.DATASETS:
            CELLS[f"{tag}|{ds}"] = L.analyse_cell(df[df["dataset"] == ds], trust=False)
            v = CELLS[f"{tag}|{ds}"]["validity"]
            print(f"  {tag:20s} {ds:14s} Yes/No validity: mass {v['median_yn_mass']:.3f}, C0 accuracy "
                  f"{v['c0_accuracy']:.3f} -> {'passes' if v['passes'] else 'EXCLUDED'} | {time.time() - t0:.0f}s")
    if "choice" in TAGINFO[tag]["grids"]:
        both = combined(tag)
        for ds in L.DATASETS:
            CHOICE_CELLS[f"{tag}|{ds}"] = L.analyse_choice_cell(both[both["dataset"] == ds])
            v = CHOICE_CELLS[f"{tag}|{ds}"]["validity"]
            print(f"  {tag:20s} {ds:14s} A/B validity: mass {v['median_ab_mass']:.3f}, C0 accuracy "
                  f"{v['c0_accuracy']:.3f} -> {'passes' if v['passes'] else 'EXCLUDED'} | {time.time() - t0:.0f}s")

# R3 and the stage trace (Olmo 3)
R3, TRACE, TRACE_VALID, BASEFORMAT = {}, {}, {}, {}
_rec = lambda tag, ds: RECS[("v2", tag)][RECS[("v2", tag)]["dataset"] == ds] if ("v2", tag) in RECS else None
for ds in L.DATASETS:
    TRACE_VALID[ds] = {t: L.validity_trace(CELLS.get(f"{t}|{ds}")) for t in L.OLMO3_STAGES}
    fi, fb = _rec("olmo3-instruct", ds), _rec("olmo3-base", ds)
    ok = TRACE_VALID[ds]["olmo3-instruct"]["passes"] and TRACE_VALID[ds]["olmo3-base"]["passes"]
    R3[ds] = L.adjusted_difference(fi, fb) if (ok and fi is not None and fb is not None) else None
    steps = {}
    for later, earlier in zip(L.OLMO3_STAGES[1:], L.OLMO3_STAGES[:-1]):
        a, b = _rec(later, ds), _rec(earlier, ds)
        steps[f"{later} - {earlier}"] = L.stage_difference(a, b) if (a is not None and b is not None) else None
    TRACE[ds] = steps
    # base-format control (labelled): the final instruct model prompted in the base model's plain-text format
    if fi is not None and len(L.slice_records(fi, "alarm", "Abase")):
        BASEFORMAT[ds] = {"asymmetry": L.asymmetry(L.slice_records(fi, "alarm", "Abase")),
                          "final_plain_minus_base": L.stage_difference(fi, fb, "Abase", "A") if fb is not None else None,
                          "final_chat_minus_final_plain": L.stage_difference(fi, fi, "A", "Abase")}
CONF = L.confirmatory_v2(CELLS, CHOICE_CELLS, R3)
# example-order control (labelled): examples in the order safe first, the five instruct models
ORDER = {}
for tag in V2:
    if "order" in TAGINFO[tag]["grids"]:
        both = combined(tag)
        for ds in L.DATASETS:
            sub = both[both["dataset"] == ds]
            ORDER[f"{tag}|{ds}"] = {"asymmetry": L.asymmetry(L.slice_records(sub, "alarm", "Asf")),
                                    "difference": L.order_difference(sub)}
# the chat templates of the Olmo 3 SFT, DPO and final checkpoints (the stage trace assumes the same format)
TEMPLATES = {t: RUNS[("v2", t)].get("template_system_text") for t in L.OLMO3_STAGES if ("v2", t) in RUNS}
print(f"analysis done in {time.time() - t0:.0f}s")


# Deviation D5 of the first study, labelled sensitivity analysis: in records with an imputed variant, the probability of
# the less likely answer word is halved, which bounds the effect of the imputation; a and a' are recomputed.
def halve_less_likely(df):
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


t0 = time.time()
SENS_IMPUTED = {}
_boot = L.BOOT_RESAMPLES
L.BOOT_RESAMPLES = 2000          # point estimates and rough intervals are enough for a bounding analysis
try:
    for tag in V2:
        df = RECS[("v2", tag)]
        if int((df["n_imputed"] > 0).sum()) == 0:
            continue
        adj = halve_less_likely(df)
        for ds in L.DATASETS:
            sub = adj[adj["dataset"] == ds]
            ent = {"share_imputed": float((df[df["dataset"] == ds]["n_imputed"] > 0).mean())}
            if "yesno" in TAGINFO[tag]["grids"]:
                r = L.asymmetry(L.slice_records(sub))
                ent.update(a=L._get(r, "adjusted", "a"), a_ci=L._get(r, "adjusted", "ci"),
                           a_prime=L._get(r, "reading_adjusted", "a"), a_prime_ci=L._get(r, "reading_adjusted", "ci"))
            if "choice" in TAGINFO[tag]["grids"]:
                pooled = L.choice_pooled(sub)
                r = L.asymmetry(pooled) if pooled is not None else None
                ent.update(a_choice=L._get(r, "adjusted", "a"), a_prime_choice=L._get(r, "reading_adjusted", "a"))
            SENS_IMPUTED[f"{tag}|{ds}"] = ent
    # per-order answer-option asymmetry (labelled sensitivity): each option order on its own
    PER_ORDER = {}
    for k in CHOICE_CELLS:
        tag, ds = k.split("|")
        sub = RECS[("v2", tag)][RECS[("v2", tag)]["dataset"] == ds]
        PER_ORDER[k] = {fr: L.asymmetry(L.slice_records(sub, fr, "A")) for fr in L.CHOICE_FRAMINGS}
finally:
    L.BOOT_RESAMPLES = _boot
print(f"imputation and per-order sensitivity done in {time.time() - t0:.0f}s ({len(SENS_IMPUTED)} cells)")

# %% [markdown]
# ## Results tables

# %%
f2 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"
f3 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
fp = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.2g}" if x >= 0.001 else "<0.001")
ci = lambda c: "–" if not c or any(v is None or np.isnan(v) for v in c) else f"[{c[0]:+.2f}, {c[1]:+.2f}]"
est = lambda r, k: f"{f2(L._get(r, k, 'a'))} {ci(L._get(r, k, 'ci'))} ({fp(L._get(r, k, 'p_boot'))})"
sup = lambda e: f"{'**yes**' if e['supported'] else 'no'}{'' if e['included'] else ' (not included)'}"
md = ["# Registered analysis, second study: results", "",
      f"Protocol `{L.PROTOCOL_V2_TAG}`; analysis code {L.CODE_VERSION} (sha256 {CHECK['code_sha256'][:16]}).", ""]
if problems:
    md += [f"**Warning:** {'; '.join(problems)}", ""]
if CHECK["missing_runs"]:
    md += [f"**Missing runs:** {', '.join(CHECK['missing_runs'])} (registered rule: their cells enter with p = 1 and "
           "count as not supporting; tests that need them are reported as not testable).", ""]
md += ["## Runs", "", "| study | model | revision | grids | read | records | invalid | imputed |",
       "|---|---|---|---|---|---|---|---|"]
for k, m in CHECK["models"].items():
    md.append(f"| {k.split('|')[0]} | {m['model_id']} | {str(m['revision'])[:12]} | {', '.join(m['grids'])} | "
              f"{m['read_mode']} | {m['n_records']} | {m['n_invalid']} | {m['n_imputed']} |")
md += ["", "## Validity", "", "| model | dataset | Yes/No: median mass | Yes/No: C0 accuracy | Yes/No status | "
       "A/B: median mass | A/B: C0 accuracy | A/B status |", "|---|---|---|---|---|---|---|---|"]
for tag in V2:
    for ds in L.DATASETS:
        v = L._get(CELLS.get(f"{tag}|{ds}"), "validity") or {}
        w = L._get(CHOICE_CELLS.get(f"{tag}|{ds}"), "validity") or {}
        st = lambda x: "–" if not x else ("passes" if x["passes"] else "**excluded**")
        md.append(f"| {tag} | {ds} | {f3(v.get('median_yn_mass'))} | {f3(v.get('c0_accuracy'))} | {st(v)} | "
                  f"{f3(w.get('median_ab_mass'))} | {f3(w.get('c0_accuracy'))} | {st(w)} |")
md += ["", "## Memorization check", "",
       "| model | dataset | file | header: chars matched | rows completed verbatim | flagged |", "|---|---|---|---|---|---|"]
for (s, tag), r in RUNS.items():
    if s != "v2":
        continue
    for ds, m in (r.get("memcheck_summary") or {}).items():
        md.append(f"| {tag} | {ds} | {m.get('file')} | {m.get('header_matched_chars')} | {m.get('row_exact_matches')} | "
                  f"{'**yes**' if m.get('flag_verbatim_rows') else 'no'} |")

md += ["", "## R1. Replication: a < 0 and a' < 0 in the new instruct models (alarm framing, example set A)", "",
       "Intersection-union test, Holm over the 9 cells.", "",
       "| model | dataset | a [95% CI] (p) | a' [95% CI] (p) | test p | Holm p | supported | registered Δ = A − S [95% CI] |",
       "|---|---|---|---|---|---|---|---|"]
for k, e in CONF["R1"]["cells"].items():
    c = CELLS.get(k)
    tag, ds = k.split("|")
    h = L._get(c, "H3")
    md.append(f"| {tag} | {ds} | {est(h, 'adjusted')} | {est(h, 'reading_adjusted')} | {fp(e['p'])} | {fp(e['p_holm'])} | "
              f"{sup(e)} | {f2(L._get(h, 'delta', 'delta'))} {ci(L._get(h, 'delta', 'ci'))} |")
ov = CONF["R1"]["overall"]
md += ["", f"**R1 replicated: {'YES' if ov['supported_overall'] else 'NO'}** ("
       + "; ".join(f"{t}: {v['datasets_supported']}/3 datasets" for t, v in ov["per_model"].items())
       + f"; rule: at least 2 of 3 datasets for at least {ov['models_needed']} of the 3 new models)", "",
       "Where a, a' and the registered Δ do not agree in sign, the cell is described as not robust (Deviation D1 of "
       "the first study).", ""]

md += ["## R2. Outcome or Yes/No question format: a < 0 and a' < 0 in the answer-option framing", "",
       "Mean of the two option orders. Eligible models: the two first-study models and each new model that meets the "
       f"R1 per-model criterion ({', '.join(CONF['R2']['overall']['eligible_models'])}). Intersection-union test, Holm "
       "over their cells.", "",
       "| model | dataset | a [95% CI] (p) | a' [95% CI] (p) | test p | Holm p | supported |", "|---|---|---|---|---|---|---|"]
for k, e in CONF["R2"]["cells"].items():
    c = CHOICE_CELLS.get(k)
    tag, ds = k.split("|")
    h = L._get(c, "R2")
    md.append(f"| {tag} | {ds} | {est(h, 'adjusted')} | {est(h, 'reading_adjusted')} | {fp(e['p'])} | {fp(e['p_holm'])} | "
              f"{sup(e)} |")
ov = CONF["R2"]["overall"]
md += ["", f"**R2 supported: {'YES' if ov['supported_overall'] else 'NO'}** ("
       + "; ".join(f"{t}: {v['datasets_supported']}/3 datasets" for t, v in ov["per_model"].items())
       + f"; rule: at least 2 of 3 datasets for at least {ov['models_needed']} of the {len(ov['eligible_models'])} "
       "eligible models)", "",
       "### Secondary comparison: δ = a(answer option) − a(alarm Yes/No), same patients", "",
       "When a < 0 with Yes/No, δ > 0 means the asymmetry is weaker with answer options. A cell counts for the "
       "interpretation only if it is valid in both framings and a and a' are both below 0 with Yes/No.", "",
       "| model | dataset | a, a' with Yes/No | δ for a [95% CI] (p) | δ for a' [95% CI] (p) | test p | counts |",
       "|---|---|---|---|---|---|---|"]
for tag in L.models_v2(("new", "v1"), "instruct"):
    for ds in L.DATASETS:
        c = CHOICE_CELLS.get(f"{tag}|{ds}") or {}
        d = c.get("framing_difference")
        cnt = bool(d and L._get(c, "validity", "passes") and L._get(c, "validity_yesno", "passes")
                   and (d.get("a_y") or 0) < 0 and (L._get(d, "reading_adjusted_diff", "a_y") or 0) < 0)
        md.append(f"| {tag} | {ds} | {f2(L._get(d, 'a_y'))}, {f2(L._get(d, 'reading_adjusted_diff', 'a_y'))} | "
                  f"{f2(L._get(d, 'diff'))} {ci(L._get(d, 'ci'))} ({fp(L._get(d, 'p_boot'))}) | "
                  f"{f2(L._get(d, 'reading_adjusted_diff', 'diff'))} {ci(L._get(d, 'reading_adjusted_diff', 'ci'))} "
                  f"({fp(L._get(d, 'reading_adjusted_diff', 'p_boot'))}) | {fp(L._get(d, 'test', 'p'))} | "
                  f"{'yes' if cnt and tag in ov['eligible_models'] else 'no'} |")
md += ["", f"**R2 interpretation (Section 6 rule): {CONF['R2']['interpretation']}.**", ""]

md += ["## R3. Instruction tuning in Olmo 3: a(final) − a(base) < 0 and the same for a'", "",
       "Validity for R3 and the stage trace: median Yes+No mass ≥ 0.5 and C0 AUROC ≥ 0.55.", "",
       "| dataset | a difference [95% CI] | a' difference [95% CI] | test p | Holm p | supported |",
       "|---|---|---|---|---|---|"]
for k, e in CONF["R3"]["cells"].items():
    ds = k.split("|")[1]
    r = R3.get(ds)
    md.append(f"| {ds} | {f2(L._get(r, 'diff'))} {ci(L._get(r, 'ci'))} | "
              f"{f2(L._get(r, 'reading_adjusted_diff', 'diff'))} {ci(L._get(r, 'reading_adjusted_diff', 'ci'))} | "
              f"{fp(e['p'])} | {fp(e['p_holm'])} | {'**yes**' if e['supported'] else 'no'}"
              f"{'' if e['included'] else (' (not tested: a model fails validity)' if not e.get('both_valid') else ' (not estimable)')} |")
md += ["", f"**R3 supported: {'YES' if CONF['R3']['overall']['supported_overall'] else 'NO'}** "
       f"({CONF['R3']['overall']['datasets_supported']}/3 datasets; rule: at least 2 of 3)", ""]

md += ["## Olmo 3 stage trace (described, no decision rule)", "",
       "Stages that fail the R3 validity rule are shown and marked. The base model is prompted as plain text; the "
       "SFT, DPO and final models through their chat template. Default system text of the templates: "
       + "; ".join(f"{t}: {v!r}" for t, v in TEMPLATES.items()) + ".", "",
       "| dataset | stage | valid (mass; C0 AUROC) | a [95% CI] | a' [95% CI] |", "|---|---|---|---|---|"]
for ds in L.DATASETS:
    for st in L.OLMO3_STAGES:
        c = CELLS.get(f"{st}|{ds}")
        tv = TRACE_VALID[ds][st]
        md.append(f"| {ds} | {st} | {'yes' if tv['passes'] else '**no**'} ({f3(tv['median_yn_mass'])}; {f3(tv['c0_auroc'])}) | "
                  f"{f2(L._get(c, 'H3', 'adjusted', 'a'))} {ci(L._get(c, 'H3', 'adjusted', 'ci'))} | "
                  f"{f2(L._get(c, 'H3', 'reading_adjusted', 'a'))} {ci(L._get(c, 'H3', 'reading_adjusted', 'ci'))} |")
md += ["", "| dataset | step | a change [95% CI] | a' change [95% CI] |", "|---|---|---|---|"]
for ds, steps in TRACE.items():
    for step, r in steps.items():
        md.append(f"| {ds} | {step} | {f2(L._get(r, 'diff'))} {ci(L._get(r, 'ci'))} | "
                  f"{f2(L._get(r, 'reading_adjusted_diff', 'diff'))} {ci(L._get(r, 'reading_adjusted_diff', 'ci'))} |")

md += ["", "### Base-format control (labelled): Olmo-3-7B-Instruct prompted as plain text, like the base model", "",
       "| dataset | a, final in plain text [95% CI] | final (plain) − base: a [95% CI] | final (chat) − final (plain): a [95% CI] |",
       "|---|---|---|---|"]
for ds, b in BASEFORMAT.items():
    md.append(f"| {ds} | {f2(L._get(b, 'asymmetry', 'adjusted', 'a'))} {ci(L._get(b, 'asymmetry', 'adjusted', 'ci'))} | "
              f"{f2(L._get(b, 'final_plain_minus_base', 'diff'))} {ci(L._get(b, 'final_plain_minus_base', 'ci'))} | "
              f"{f2(L._get(b, 'final_chat_minus_final_plain', 'diff'))} {ci(L._get(b, 'final_chat_minus_final_plain', 'ci'))} |")
md += ["", "## Example-order control (labelled): the four solved examples in the order safe first", "",
       "In the registered order the last solved example always has the safe outcome. Here it has the alarming one.", "",
       "| model | dataset | a, safe first [95% CI] | a', safe first [95% CI] | change in a [95% CI] | change in a' [95% CI] |",
       "|---|---|---|---|---|---|"]
for k, o in ORDER.items():
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | {f2(L._get(o, 'asymmetry', 'adjusted', 'a'))} {ci(L._get(o, 'asymmetry', 'adjusted', 'ci'))} | "
              f"{f2(L._get(o, 'asymmetry', 'reading_adjusted', 'a'))} {ci(L._get(o, 'asymmetry', 'reading_adjusted', 'ci'))} | "
              f"{f2(L._get(o, 'difference', 'diff'))} {ci(L._get(o, 'difference', 'ci'))} | "
              f"{f2(L._get(o, 'difference', 'reading_adjusted_diff', 'diff'))} "
              f"{ci(L._get(o, 'difference', 'reading_adjusted_diff', 'ci'))} |")
md += ["", "## Secondary (labelled; as in the first study)", "",
       "### H1 and H2: accuracy with true (C2) and flipped (C4) evidence against none (C0), exact McNemar, no correction", "",
       "| model | dataset | C0 | C2 − C0 (p) | C4 − C0 (p) |", "|---|---|---|---|---|"]
for k, c in CELLS.items():
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | {f3(L._get(c, 'by_condition', 'C0', 'accuracy'))} | {f2(L._get(c, 'H1', 'diff'))} "
              f"({fp(L._get(c, 'H1', 'p'))}) | {f2(L._get(c, 'H2', 'diff'))} ({fp(L._get(c, 'H2', 'p'))}) |")
md += ["", "### Q3 with the reversed Yes/No question (the first study's four-way table)", "",
       "| model | dataset | a: alarm / reversed [95% CI] | a': alarm / reversed [95% CI] | interpretation |",
       "|---|---|---|---|---|"]
for k, c in CELLS.items():
    tag, ds = k.split("|")
    cols = [f"{f2(L._get(c, 'H3', key, 'a'))} {ci(L._get(c, 'H3', key, 'ci'))} / "
            f"{f2(L._get(c, 'Q3_reversed', key, 'a'))} {ci(L._get(c, 'Q3_reversed', key, 'ci'))}"
            for key in ("adjusted", "reading_adjusted")]
    md.append(f"| {tag} | {ds} | {cols[0]} | {cols[1]} | {c['Q3_interpretation']}"
              f"{'' if c['validity']['passes'] else ' (cell excluded)'} |")
md += ["", "### Robustness and sensitivity of a (Yes/No grid)", "",
       "| model | dataset | a with set A / B / C | a, d winsorized [95% CI] | share of expected baselines within 1 SD of 0 | "
       "first-stage R² | log-odds, unadjusted [95% CI] | follow alarm / safe card |", "|---|---|---|---|---|---|---|---|"]
for k, c in CELLS.items():
    tag, ds = k.split("|")
    a = c.get("H3") or {}
    rob = L._get(c, "exploratory", "robustness_example_sets") or {}
    sets = [L._get(a, "adjusted", "a")] + [L._get(rob.get(s), "H3", "adjusted", "a") for s in ("B", "C")]
    md.append(f"| {tag} | {ds} | {' / '.join(f2(x) for x in sets)} | {f2(L._get(a, 'sensitivity_winsorized', 'a'))} "
              f"{ci(L._get(a, 'sensitivity_winsorized', 'ci'))} | {f3(L._get(a, 'adjusted', 'share_e_within_1sd_of_0'))} | "
              f"{f3(L._get(a, 'adjusted', 'coef', 'first_stage_r2'))} | {f2(L._get(a, 'logodds_unadjusted', 'value'))} "
              f"{ci(L._get(a, 'logodds_unadjusted', 'ci'))} | {f3(a.get('follow_alarm_card'))} / {f3(a.get('follow_safe_card'))} |")
md += ["", "### Answer-option framing: accuracy and preference for the letter A", "",
       "| model | dataset | C0 | C2 | C4 | mean log P(A) − log P(B) |", "|---|---|---|---|---|---|"]
for k, c in CHOICE_CELLS.items():
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | " + " | ".join(f3(L._get(c, 'by_condition', cc, 'accuracy')) for cc in ("C0", "C2", "C4"))
              + f" | {f2(c.get('letter_A_logodds_mean'))} |")
md += ["", "### Answer-option framing, each option order on its own (labelled; 2,000 resamples)", "",
       "| model | dataset | a, alarming outcome as A | a', alarming as A | a, safe outcome as A | a', safe as A |",
       "|---|---|---|---|---|---|"]
for k, po in PER_ORDER.items():
    tag, ds = k.split("|")
    g = lambda fr, key: L._get(po.get(fr), key, "a")
    md.append(f"| {tag} | {ds} | {f2(g('choice_alarm_A', 'adjusted'))} | {f2(g('choice_alarm_A', 'reading_adjusted'))} | "
              f"{f2(g('choice_safe_A', 'adjusted'))} | {f2(g('choice_safe_A', 'reading_adjusted'))} |")
if SENS_IMPUTED:
    md += ["", "### Sensitivity to the top-1,000 reading (D5 bound; 2,000 resamples)", "",
           "| model | dataset | records with an imputed variant | a, bounded | a', bounded | a (A/B), bounded | a' (A/B), bounded |",
           "|---|---|---|---|---|---|---|"]
    for k, v in SENS_IMPUTED.items():
        tag, ds = k.split("|")
        md.append(f"| {tag} | {ds} | {v['share_imputed']:.1%} | {f2(v.get('a'))} | {f2(v.get('a_prime'))} | "
                  f"{f2(v.get('a_choice'))} | {f2(v.get('a_prime_choice'))} |")
REPORT = "\n".join(md) + "\n"
print(REPORT)

# %% [markdown]
# ## Save

# %%
with open(os.path.join(OUT, "report_v2.md"), "w", encoding="utf-8") as f:
    f.write(REPORT)
with open(os.path.join(OUT, "results_v2.json"), "w") as f:
    json.dump({"checks": CHECK, "confirmatory": CONF, "cells_yesno": CELLS, "cells_choice": CHOICE_CELLS, "R3": R3,
               "stage_trace": TRACE, "stage_validity": TRACE_VALID, "templates": TEMPLATES,
               "base_format_control": BASEFORMAT, "example_order_control": ORDER, "per_order": PER_ORDER,
               "sensitivity_imputation_D5": SENS_IMPUTED}, f, indent=1, default=str)
rows = []
for tag in V2:
    for ds in L.DATASETS:
        c, h = CELLS.get(f"{tag}|{ds}"), CHOICE_CELLS.get(f"{tag}|{ds}")
        rows.append({"model": tag, "role": TAGINFO[tag]["role"], "dataset": ds,
                     "yesno_valid": L._get(c, "validity", "passes"), "acc_C0": L._get(c, "by_condition", "C0", "accuracy"),
                     "a": L._get(c, "H3", "adjusted", "a"), "a_prime": L._get(c, "H3", "reading_adjusted", "a"),
                     "h3_test_p": L._get(c, "H3", "h3_test", "p"), "delta": L._get(c, "H3", "delta", "delta"),
                     "ab_valid": L._get(h, "validity", "passes"), "a_ab": L._get(h, "R2", "adjusted", "a"),
                     "a_prime_ab": L._get(h, "R2", "reading_adjusted", "a"), "r2_test_p": L._get(h, "R2", "h3_test", "p"),
                     "framing_delta_a": L._get(h, "framing_difference", "diff"),
                     "framing_delta_a_prime": L._get(h, "framing_difference", "reading_adjusted_diff", "diff")})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "cells_summary_v2.csv"), index=False)
ZIP = os.path.join(WORK, "analysis_v2_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("report_v2.md", "results_v2.json", "cells_summary_v2.csv"):
        z.write(os.path.join(OUT, name), name)
print("Saved:", ZIP, f"| total {(time.time() - T0) / 60:.1f} min")
