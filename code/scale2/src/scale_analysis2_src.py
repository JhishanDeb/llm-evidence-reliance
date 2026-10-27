# %% [markdown]
# # Exploratory scale check, second plan: analysis (two-step reading)
#
# **What this notebook does.** It reads the runs of the second scale plan (`scale2_<model>-2step.zip`), the first
# plan's 8B run (`scale_qwen3-8b-awq.zip`, the bridge between the two readings) and 4B run
# (`scale_qwen3-4b-awq.zip`, reused as the plan says), and, if attached, the second study's Qwen3-8B run
# (`v2_qwen3-instruct.zip`). It checks that every run used the registered code and saw the same patients and prompts,
# and computes the measures and reading rules fixed in
# [`SCALE_CHECK_2.md`, tag `scale-plan-2`](https://github.com/JhishanDeb/llm-evidence-reliance/blob/scale-plan-2/SCALE_CHECK_2.md).
# Every measure comes from the registered code of the second study (`ler_main.py`, main-2.0). Exploratory: not part
# of either pre-registration; it changes no registered result and no result of the first scale plan.
#
# **How to run.** No GPU and no internet. On Kaggle: upload the zip files as one dataset (Add Input → Upload → New
# Dataset) and Save & Run All. The first code cell accepts zip files or unpacked folders. Locally or on Colab: put the
# zip files next to the notebook and run all cells. Runtime: about 20 to 40 minutes on a CPU. Download
# `scale2_analysis_results.zip` at the end.

# %%
import os, sys, json, glob, gzip, zipfile, hashlib, time
import numpy as np
import pandas as pd

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
IN_DIR = os.path.join(WORK, "scale2_analysis_inputs")
OUT = os.path.join(WORK, "scale2_analysis_results")
SKIP = [os.path.realpath(IN_DIR), os.path.realpath(OUT)]
REF_TAG = "qwen3-instruct"


def _skipped(p):
    rp = os.path.realpath(p)
    return any(rp == s or rp.startswith(s + os.sep) for s in SKIP)


SOURCES, DUPES = {}, {}
WANT = ("run_log_{t}.json", "modelcheck_{t}.json", "records_{t}.csv", "example_prompts_{t}.md", "memcheck_{t}.json")


def _add(where, read, tag):
    log_raw = read(f"run_log_{tag}.json")
    if log_raw is None or read("ler_main.py") is None:
        return
    log = json.loads(log_raw)
    study = log.get("study")
    if not (study in ("scale", "scale2") or (study == "v2" and tag == REF_TAG)):
        return
    files = {"ler_main.py": read("ler_main.py")}
    for name in ("ler_scale.py", "ler_scale2.py"):
        if read(name) is not None and (name != "ler_scale2.py" or study == "scale2"):
            files[name] = read(name)
    for pat in WANT:
        name = pat.format(t=tag)
        data = read(name)
        if data is None and name.endswith(".csv"):
            data = read(name + ".gz")
            data = gzip.decompress(data) if data is not None else None
        if data is not None:
            files[name] = data
    sig = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    if tag in SOURCES:
        if SOURCES[tag][3] != sig:
            DUPES.setdefault(tag, [SOURCES[tag][1]]).append(where)
        return
    SOURCES[tag] = (study, where, files, sig)


for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        if _skipped(p) or not os.path.basename(p).startswith(("scale_", "scale2_", "v2_")):
            continue
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
            logs = [n for n in names if n.startswith("run_log_") and n.endswith(".json")]
            if len(logs) == 1:
                _add(p, lambda n: z.read(n) if n in names else None, logs[0][len("run_log_"):-len(".json")])
for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "run_log_*.json"), recursive=True)):
        if _skipped(p):
            continue
        folder, tag = os.path.dirname(p), os.path.basename(p)[len("run_log_"):-len(".json")]
        _add(folder, lambda n: open(os.path.join(folder, n), "rb").read()
             if os.path.exists(os.path.join(folder, n)) else None, tag)
if DUPES:
    raise RuntimeError(f"Different files for the same run: {DUPES}. Attach only the final output of each run.")
os.makedirs(OUT, exist_ok=True)
for tag, (study, where, files, sig) in sorted(SOURCES.items()):
    d = os.path.join(IN_DIR, tag)
    os.makedirs(d, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    print(f"{study:7s} {tag}: from {where}")
    for name in sorted(sig):
        print(f"    {name:42s} sha256 {sig[name][:16]}")

# %% [markdown]
# ## Load the runs and check that they are comparable

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


P2_TAGS = sorted(t for t, v in SOURCES.items() if v[0] == "scale2")
P1_TAGS = sorted(t for t, v in SOURCES.items() if v[0] == "scale")
if not P2_TAGS:
    raise RuntimeError("No scale2_<model>-2step.zip found. Attach the runs of the second plan.")
sys.path.insert(0, os.path.join(IN_DIR, P2_TAGS[0]))
for _m in ("ler_scale2", "ler_scale", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_scale as S
import ler_scale2 as S2

if os.environ.get("LER_TEST_BOOT"):          # offline tests only: fewer bootstrap resamples
    L.BOOT_RESAMPLES = int(os.environ["LER_TEST_BOOT"])
BRIDGE1 = S2.BRIDGE                          # plan-1 tag of the 8B model
REUSED = list(S2.REUSED_FROM_PLAN1)          # plan-1 tag(s) reused under plan 2 (the 4B model)
missing_inputs = [t for t in [BRIDGE1] + REUSED if t not in SOURCES]
if missing_inputs:
    raise RuntimeError(f"The first plan's runs {missing_inputs} are needed (results/scale/ in the repository). Attach them.")
problems, warnings = [], []
for t in SOURCES:
    if sha(os.path.join(IN_DIR, t, "ler_main.py")) != S.REGISTERED_CODE_SHA256:
        problems.append(f"{t}: not the registered ler_main.py main-2.0")
_s1 = {t: sha(os.path.join(IN_DIR, t, "ler_scale.py")) for t in P1_TAGS + P2_TAGS}
if len(set(_s1.values())) != 1:
    problems.append("the runs used different versions of ler_scale.py")
_s2 = {t: sha(os.path.join(IN_DIR, t, "ler_scale2.py")) for t in P2_TAGS}
if len(set(_s2.values())) != 1:
    problems.append("the second-plan runs used different versions of ler_scale2.py")

RUNS, RECS, MODELCHECKS = {}, {}, {}
for tag in SOURCES:
    d = os.path.join(IN_DIR, tag)
    RUNS[tag] = json.load(open(os.path.join(d, f"run_log_{tag}.json")))
    if os.path.exists(os.path.join(d, f"records_{tag}.csv")):
        RECS[tag] = pd.read_csv(os.path.join(d, f"records_{tag}.csv"), dtype={"id": str})
    if os.path.exists(os.path.join(d, f"modelcheck_{tag}.json")):
        MODELCHECKS[tag] = json.load(open(os.path.join(d, f"modelcheck_{tag}.json")))
ID_OF = {m["tag"]: mid for mid, m in S.SCALE_MODELS.items()}
base = lambda t: t[:-len("-2step")] if t.endswith("-2step") else t
SIZE = {t: S.SCALE_MODELS[ID_OF[base(t)]]["size_b"] for t in SOURCES if base(t) in ID_OF}
SIZE[REF_TAG] = 8
for t in P1_TAGS + P2_TAGS:
    r = RUNS[t]
    if base(t) not in ID_OF:
        problems.append(f"{t} is not a scale-check model")
    elif not r.get("mock") and r.get("model_revision") != S.SCALE_MODELS[ID_OF[base(t)]]["revision"]:
        problems.append(f"{t}: revision differs from the pinned one")
if len({RUNS[t]["plan"]["sha256"] for t in P2_TAGS}) != 1:
    problems.append("the second-plan runs checked different plan texts")
if any(RUNS[t]["plan"].get("tag") != S2.PLAN2_TAG for t in P2_TAGS):
    problems.append("a second-plan run did not check SCALE_CHECK_2.md")
RAN2 = [t for t in P2_TAGS if not RUNS[t].get("cannot_be_run") and t in RECS]
CANNOT2 = {t: RUNS[t].get("model_check", {}).get("problems") for t in P2_TAGS if RUNS[t].get("cannot_be_run")}
NOT_ATTACHED = [m["tag"] + "-2step" for m in S2.SCALE2_MODELS.values() if m["tag"] + "-2step" not in SOURCES]
WITH_RECS = [t for t in RECS]
fps = {t: json.dumps(RUNS[t]["fingerprints"], sort_keys=True) for t in WITH_RECS}
if len(set(fps.values())) != 1:
    problems.append("different patients, features, cards or partners")
_sha_by_key = {}
for t, df in RECS.items():
    for key, ts in zip(df["key"], df["text_sha"]):
        if _sha_by_key.setdefault(key, ts) != ts:
            problems.append(f"prompt {key} has different texts in different runs")
            break
if len({RUNS[t].get("read_mode") for t in WITH_RECS}) > 1:
    problems.append("the runs did not all use the top-1,000 reading")


def example_blocks(tag):
    p = os.path.join(IN_DIR, tag, f"example_prompts_{tag}.md")
    if not os.path.exists(p):
        return {}
    out = {}
    for part in open(p, encoding="utf-8").read().split("## ")[1:]:
        head, _, body = part.partition("\n")
        out[head.strip()] = body.strip()
    return out


RENDER = {}
for t in RAN2:
    b, ref_t = example_blocks(t), (base(t) if base(t) in SOURCES else REF_TAG)
    rb = example_blocks(ref_t)
    common = sorted(set(b) & set(rb))
    diff = [h for h in common if b[h] != rb[h]]
    RENDER[t] = {"compared_with": ref_t, "compared": len(common), "identical": len(common) - len(diff)}
    if not common or diff:
        warnings.append(f"{t}: {len(diff)} of {len(common)} rendered example prompts differ from {ref_t}")
if any(RUNS[t].get("mock") for t in RUNS) or os.environ.get("LER_TEST_BOOT"):
    warnings.append("TEST RUN: mock inputs or a reduced number of bootstrap resamples")
CHECK = {"code_sha256": S.REGISTERED_CODE_SHA256, "ler_scale_sha256": sorted(set(_s1.values())),
         "ler_scale2_sha256": sorted(set(_s2.values())),
         "plan2_sha256": sorted({RUNS[t]["plan"]["sha256"] for t in P2_TAGS}), "ran": RAN2, "cannot_be_run": CANNOT2,
         "not_attached": NOT_ATTACHED, "reused_from_plan1": REUSED, "reference_attached": REF_TAG in RECS,
         "rendered_example_prompts": RENDER, "problems": problems, "warnings": warnings,
         "boot_resamples": L.BOOT_RESAMPLES}
print(json.dumps(CHECK, indent=1, default=str))
if problems:
    raise RuntimeError("The runs are not comparable: " + "; ".join(problems))

# %% [markdown]
# ## Every measure for every model × dataset cell
# The models of the size comparison: the 4B model (first-plan records, reused) and the 8B, 14B and 32B models under
# the two-step reading. Also the 8B model under the first reading (bridge) and, if attached, the second study's
# Qwen3-8B in float16. Every run is restricted to the same 13 prompts per patient.

# %%
t0 = time.time()
GRID_KEYS = {f"{c}|{f}|{s}" for c, f, s in S.scale_grid()}
_cfs = lambda df: df["cond"] + "|" + df["framing"] + "|" + df["shot_set"]
for t in list(RECS):
    RECS[t] = RECS[t][_cfs(RECS[t]).isin(GRID_KEYS)].reset_index(drop=True)
SIZE_MODELS = sorted([t for t in REUSED if t in RECS] + RAN2, key=lambda t: (SIZE[t], t))
EXTRA = [BRIDGE1] + ([REF_TAG] if REF_TAG in RECS else [])
MODELS = SIZE_MODELS + EXTRA
NAME = lambda t: ID_OF[base(t)].split("/")[1] if base(t) in ID_OF else t
LABEL = {t: NAME(t) for t in RAN2}
LABEL.update({t: f"{NAME(t)} (first reading, reused)" for t in REUSED})
LABEL.update({BRIDGE1: f"{NAME(BRIDGE1)} (first reading: bridge)", REF_TAG: "Qwen3-8B float16 (second study, first reading)"})
BRIDGE2 = BRIDGE1 + "-2step"
CELLS, CHOICE = {}, {}
for tag in MODELS:
    df = RECS[tag]
    for ds in L.DATASETS:
        sub = df[df["dataset"] == ds]
        CELLS[f"{tag}|{ds}"] = L.analyse_cell(sub, exploratory=False, trust=False)
        CHOICE[f"{tag}|{ds}"] = L.analyse_choice_cell(sub)
        v, w = CELLS[f"{tag}|{ds}"]["validity"], CHOICE[f"{tag}|{ds}"]["validity"]
        print(f"  {tag:20s} {ds:14s} Yes/No {'passes' if v['passes'] else 'EXCLUDED'} (mass {v['median_yn_mass']:.3f}) | "
              f"A/B {'passes' if w['passes'] else 'EXCLUDED'} (mass {w['median_ab_mass']:.3f}) | {time.time() - t0:.0f}s")


def _sub(tag, ds):
    return RECS[tag][RECS[tag]["dataset"] == ds]


def choice_difference(dx, dy):
    px, py = L.choice_pooled(dx), L.choice_pooled(dy)
    if px is None or py is None:
        return None
    return L._paired_asym_difference(L._wide(px), L._wide(py))


PAIRS = []
if BRIDGE2 in RAN2:
    PAIRS.append(("bridge", BRIDGE2, BRIDGE1))
    PAIRS += [("size", t, BRIDGE2) for t in SIZE_MODELS if t != BRIDGE2]
DIFFS = {}
for kind, x, y in PAIRS:
    for ds in L.DATASETS:
        DIFFS[f"{x}-{y}|{ds}"] = {"kind": kind, "x": x, "y": y, "dataset": ds,
                                  "yesno": L.stage_difference(_sub(x, ds), _sub(y, ds)),
                                  "choice": choice_difference(_sub(x, ds), _sub(y, ds)),
                                  "valid_yesno": bool(CELLS[f"{x}|{ds}"]["validity"]["passes"]
                                                      and CELLS[f"{y}|{ds}"]["validity"]["passes"]),
                                  "valid_choice": bool(CHOICE[f"{x}|{ds}"]["validity"]["passes"]
                                                       and CHOICE[f"{y}|{ds}"]["validity"]["passes"])}
BOLD = {}
for t in RAN2:
    r = RECS[t]
    for ds in L.DATASETS:
        s = r[r["dataset"] == ds]
        ch = s["framing"].map(L.is_choice)
        BOLD[f"{t}|{ds}"] = {"yesno": float(s[~ch]["p_bold"].median()), "choice": float(s[ch]["p_bold"].median()),
                             "mass_first_yesno": float(s[(~ch) & (s["framing"] == "alarm")]["mass_first"].median()),
                             "mass_two_step_yesno": float(s[(~ch) & (s["framing"] == "alarm")]["yn_mass"].median())}
print(f"analysis done in {time.time() - t0:.0f}s ({L.BOOT_RESAMPLES} bootstrap resamples per estimate)")

# %% [markdown]
# ## The reading rules of the second plan

# %%
def cell_pattern(asym, valid):
    a, a0, p = L._get(asym, "adjusted", "a"), L._get(asym, "reading_adjusted", "a"), L._get(asym, "h3_test", "p")
    return bool(valid and S.finite(a) and S.finite(a0) and S.finite(p) and a < 0 and a0 < 0 and p < 0.05)


PATTERN = {}
for tag in MODELS:
    yn = {ds: cell_pattern(CELLS[f"{tag}|{ds}"].get("H3"), CELLS[f"{tag}|{ds}"]["validity"]["passes"]) for ds in L.DATASETS}
    ab = {ds: cell_pattern(CHOICE[f"{tag}|{ds}"].get("R2"), CHOICE[f"{tag}|{ds}"]["validity"]["passes"]) for ds in L.DATASETS}
    PATTERN[tag] = {"yesno_cells": yn, "yesno_model": sum(yn.values()) >= 2,
                    "choice_cells": ab, "choice_model": sum(ab.values()) >= 2}
counts = lambda fmt, tags: ", ".join(f"{LABEL[t]} {sum(PATTERN[t][fmt + '_cells'].values())}/3" for t in tags)


def bridge_statement(fmt):
    if BRIDGE2 not in RAN2:
        return "the 8B model was not run under the two-step reading"
    ok, old = PATTERN[BRIDGE2][f"{fmt}_model"], PATTERN[BRIDGE1][f"{fmt}_model"]
    head = ("the two-step reading is taken not to remove the pattern (Qwen3-8B-AWQ shows it)" if ok else
            "Qwen3-8B-AWQ does not show the pattern under the two-step reading"
            + (" although it did under the first reading" if old else ", nor under the first reading"))
    return head + f" ({counts(fmt, [BRIDGE2, BRIDGE1])})"


def size_statement(fmt):
    key = f"{fmt}_model"
    if BRIDGE2 not in RAN2:
        return "not interpretable: the 8B model was not run under the two-step reading"
    if not PATTERN[BRIDGE2][key]:
        return f"reported but not interpreted: Qwen3-8B-AWQ does not show the pattern ({counts(fmt, SIZE_MODELS)})"
    larger = [t for t in RAN2 if SIZE[t] > 8]
    missing = [ID_OF[m["tag"]].split("/")[1] for m in S2.SCALE2_MODELS.values()
               if m["size_b"] > 8 and m["tag"] + "-2step" not in RAN2]
    note = f"; {', '.join(missing)} not run" if missing else ""
    if not larger:
        return f"no larger model could be run{note} ({counts(fmt, SIZE_MODELS)})"
    if all(PATTERN[t][key] for t in larger):
        return f"persists with size{note} ({counts(fmt, SIZE_MODELS)})"
    return (f"weakens with size: {', '.join(LABEL[t] for t in larger if not PATTERN[t][key])} does not show it{note} "
            f"({counts(fmt, SIZE_MODELS)})")


SUMMARY = {"bridge_yesno": bridge_statement("yesno"), "size_yesno": size_statement("yesno"),
           "bridge_choice_described": bridge_statement("choice"), "size_choice_described": size_statement("choice"),
           "per_model": {t: {"yesno": f"{sum(PATTERN[t]['yesno_cells'].values())}/3",
                             "choice": f"{sum(PATTERN[t]['choice_cells'].values())}/3"} for t in MODELS}}
print(json.dumps(SUMMARY, indent=1))

# %% [markdown]
# ## Figure: a and a′ by model size (two-step reading)

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DS_NAME = {"physionet2012": "PhysioNet 2012", "physionet2019": "PhysioNet 2019", "diabetes130": "Diabetes 130"}
COL_AP, COL_A, COL_OFF, COL_BR = "#2a6fdb", "#8a8f98", "#c9ccd1", "#d1882a"
xs = {t: np.log2(SIZE[t]) for t in SIZE_MODELS}
xs[BRIDGE1] = np.log2(8) + 0.3
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
for row, (fmt, getc, key) in enumerate((("Yes/No question", CELLS, "H3"), ("answer options (A/B)", CHOICE, "R2"))):
    for col, ds in enumerate(L.DATASETS):
        ax = axes[row, col]
        ax.axhline(0, color="#b0b4ba", lw=1)
        for stat, dx, colr, fill in (("reading_adjusted", -0.07, COL_AP, True), ("adjusted", 0.07, COL_A, False)):
            lx, ly = [], []
            for t in SIZE_MODELS + [BRIDGE1]:
                c = getc[f"{t}|{ds}"]
                est, ci = L._get(c, key, stat, "a"), L._get(c, key, stat, "ci")
                if not S.finite(est):
                    continue
                valid = c["validity"]["passes"]
                cc = (COL_BR if t == BRIDGE1 else colr) if valid else COL_OFF
                x = xs[t] + dx
                if ci and all(S.finite(v) for v in ci):
                    ax.plot([x, x], ci, color=cc, lw=1.4)
                ax.plot([x], [est], marker="D" if t == BRIDGE1 else "o", ms=6.5, color=cc,
                        markerfacecolor=cc if fill else "white", markeredgewidth=1.5, zorder=3)
                if t != BRIDGE1 and valid:
                    lx.append(x)
                    ly.append(est)
            if len(lx) > 1:
                o = np.argsort(lx)
                ax.plot(np.array(lx)[o], np.array(ly)[o], color=colr, lw=1, alpha=0.5, ls="-" if fill else "--")
        if row == 0:
            ax.set_title(DS_NAME[ds], fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{fmt}\n← safe evidence moves it more", fontsize=9)
        ax.set_xticks([np.log2(s) for s in (4, 8, 14, 32)])
        ax.set_xticklabels(["4B", "8B", "14B", "32B"])
        ax.spines[["top", "right"]].set_visible(False)
fig.legend(handles=[Line2D([], [], marker="o", color=COL_AP, lw=0, label="a′ (two-step; 4B: first reading)"),
                    Line2D([], [], marker="o", color=COL_A, markerfacecolor="white", lw=0, label="a"),
                    Line2D([], [], marker="D", color=COL_BR, lw=0, label="8B under the first reading (bridge)"),
                    Line2D([], [], marker="o", color=COL_OFF, lw=0, label="cell excluded by validity")],
           loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Scale check, second plan (exploratory): asymmetry by Qwen3 model size, 4-bit, 95% intervals", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
FIG = os.path.join(OUT, "scale2_figure.png")
fig.savefig(FIG, dpi=160)
print("saved", FIG)

# %% [markdown]
# ## Report

# %%
f2 = lambda x: "–" if not S.finite(x) else f"{x:+.2f}"
f3 = lambda x: "–" if not S.finite(x) else f"{x:.3f}"
fp = lambda x: "–" if not S.finite(x) else (f"{x:.2g}" if x >= 0.001 else "<0.001")
ci = lambda c: "–" if not c or not all(S.finite(v) for v in c) else f"[{c[0]:+.2f}, {c[1]:+.2f}]"
est = lambda r, k: f"{f2(L._get(r, k, 'a'))} {ci(L._get(r, k, 'ci'))}"
yes = lambda b: "**yes**" if b else "no"

md = ["# Exploratory scale check, second plan: results (two-step reading)", "",
      f"Plan `{S2.PLAN2_TAG}` (SCALE_CHECK_2.md sha256 {CHECK['plan2_sha256'][0][:16]}); registered code main-2.0 "
      f"(sha256 {S.REGISTERED_CODE_SHA256[:16]}); ler_scale {S.SCALE_VERSION}; ler_scale2 {S2.SCALE2_VERSION}. "
      "Exploratory: not part of either pre-registration; no correction for multiple tests. The 4B model's records "
      "come from the first plan (" + S2.REUSED_FROM_PLAN1["qwen3-4b-awq"] + ").", ""]
if warnings:
    md += ["**Warnings:** " + "; ".join(warnings), ""]
md += ["## Runs", "", "| run | revision | status | records | imputed | model check: two-step mass Yes/No (first token "
       "alone), A/B | s per prompt (dev) | memorization flagged |", "|---|---|---|---|---|---|---|---|"]
for t in sorted(SOURCES, key=lambda t: (SOURCES[t][0], SIZE.get(t, 0), t)):
    r, mc = RUNS[t], MODELCHECKS.get(t, {})
    status = {"scale": "first plan", "scale2": "second plan", "v2": "second study"}[SOURCES[t][0]]
    if t in CANNOT2:
        status += ": cannot be run: " + "; ".join(CANNOT2[t])
    flagged = [ds for ds, v in (r.get("memcheck_summary") or {}).items() if v.get("flag_verbatim_rows")]
    mfirst = L._get(mc, "reading_yesno", "mass_first_median")
    md.append(f"| {t} | {str(r.get('model_revision'))[:12]} | {status} | {len(RECS[t]) if t in RECS else 0} | "
              f"{int((RECS[t]['n_imputed'] > 0).sum()) if t in RECS else '–'} | "
              f"{f3(L._get(mc, 'reading_yesno', 'mass_median'))}{' (' + f3(mfirst) + ')' if S.finite(mfirst) else ''}, "
              f"{f3(L._get(mc, 'reading_choice', 'mass_median'))} | {f3(L._get(mc, 'reading_yesno', 'seconds_per_prompt'))} | "
              f"{', '.join(flagged) or ('none' if r.get('memcheck_summary') else 'not run under this plan')} |")
for t in NOT_ATTACHED:
    md.append(f"| {t} | – | not attached | – | – | – | – | – |")
md += ["", "Rendered example prompts: " + "; ".join(f"{t} {v['identical']}/{v['compared']} identical to {v['compared_with']}"
                                                   for t, v in RENDER.items()), ""]
md += ["## Probability of \" **\" after \"Answer:\" (median over test prompts)", "",
       "| model | dataset | Yes/No | A/B | Yes/No mass, first token alone | Yes/No mass, two-step |", "|---|---|---|---|---|---|"]
for k, b in BOLD.items():
    t, ds = k.split("|")
    md.append(f"| {LABEL[t]} | {ds} | {f3(b['yesno'])} | {f3(b['choice'])} | {f3(b['mass_first_yesno'])} | "
              f"{f3(b['mass_two_step_yesno'])} |")
md += ["", "## Validity", "", "| model | dataset | Yes/No: mass | Yes/No: C0 accuracy | Yes/No | A/B: mass | A/B: C0 accuracy | A/B |",
       "|---|---|---|---|---|---|---|---|"]
for t in MODELS:
    for ds in L.DATASETS:
        v, w = CELLS[f"{t}|{ds}"]["validity"], CHOICE[f"{t}|{ds}"]["validity"]
        md.append(f"| {LABEL[t]} | {ds} | {f3(v['median_yn_mass'])} | {f3(v['c0_accuracy'])} | "
                  f"{'passes' if v['passes'] else '**excluded**'} | {f3(w['median_ab_mass'])} | {f3(w['c0_accuracy'])} | "
                  f"{'passes' if w['passes'] else '**excluded**'} |")
for title, getc, key, pk in (("Yes/No question: a and a′ (alarm framing, example set A)", CELLS, "H3", "yesno"),
                             ("Answer options: a and a′ on the averaged reading of both option orders (described)",
                              CHOICE, "R2", "choice")):
    md += ["", f"## {title}", "", "| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |",
           "|---|---|---|---|---|---|"]
    for t in MODELS:
        for ds in L.DATASETS:
            h = getc[f"{t}|{ds}"].get(key)
            valid = getc[f"{t}|{ds}"]["validity"]["passes"]
            md.append(f"| {LABEL[t]} | {ds} | {est(h, 'adjusted')} | {est(h, 'reading_adjusted')} | "
                      f"{fp(L._get(h, 'h3_test', 'p'))} | {yes(PATTERN[t][pk + '_cells'][ds])}{'' if valid else ' (excluded)'} |")
    md += ["", "| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |", "|---|---|---|"]
    for t in MODELS:
        md.append(f"| {LABEL[t]} | {sum(PATTERN[t][pk + '_cells'].values())}/3 | {yes(PATTERN[t][pk + '_model'])} |")
md += ["", "## Paired differences over the same patients", "",
       "x − y; below 0 means the asymmetry toward safe-pointing evidence is larger in x.", "",
       "| comparison | dataset | format | both cells valid | a difference [95% CI] | a′ difference [95% CI] | p |",
       "|---|---|---|---|---|---|---|"]
for k, d in DIFFS.items():
    for fmt in ("yesno", "choice"):
        r = d[fmt]
        md.append(f"| {d['kind']}: {LABEL[d['x']]} − {LABEL[d['y']]} | {d['dataset']} | {'Yes/No' if fmt == 'yesno' else 'A/B'} | "
                  f"{yes(d['valid_' + fmt])} | {f2(L._get(r, 'diff'))} {ci(L._get(r, 'ci'))} | "
                  f"{f2(L._get(r, 'reading_adjusted_diff', 'diff'))} {ci(L._get(r, 'reading_adjusted_diff', 'ci'))} | "
                  f"{fp(L._get(r, 'test', 'p'))} |")
md += ["", "## Q3 with the reversed Yes/No question", "",
       "| model | dataset | a: alarm / reversed | a′: alarm / reversed | interpretation |", "|---|---|---|---|---|"]
for t in MODELS:
    for ds in L.DATASETS:
        c = CELLS[f"{t}|{ds}"]
        md.append(f"| {LABEL[t]} | {ds} | {f2(L._get(c, 'H3', 'adjusted', 'a'))} / {f2(L._get(c, 'Q3_reversed', 'adjusted', 'a'))} | "
                  f"{f2(L._get(c, 'H3', 'reading_adjusted', 'a'))} / {f2(L._get(c, 'Q3_reversed', 'reading_adjusted', 'a'))} | "
                  f"{c.get('Q3_interpretation')}{'' if c['validity']['passes'] else ' (cell excluded)'} |")
md += ["", "## Accuracy (Yes/No, alarm framing; exact McNemar, uncorrected)", "",
       "| model | dataset | C0 | C2 − C0 (p) | C4 − C0 (p) |", "|---|---|---|---|---|"]
for t in MODELS:
    for ds in L.DATASETS:
        c = CELLS[f"{t}|{ds}"]
        md.append(f"| {LABEL[t]} | {ds} | {f3(L._get(c, 'by_condition', 'C0', 'accuracy'))} | "
                  f"{f2(L._get(c, 'H1', 'diff'))} ({fp(L._get(c, 'H1', 'p'))}) | {f2(L._get(c, 'H2', 'diff'))} ({fp(L._get(c, 'H2', 'p'))}) |")
md += ["", "## Reading rules of the second plan", "",
       f"- Bridge, Yes/No question: {SUMMARY['bridge_yesno']}.",
       f"- Size, Yes/No question: **{SUMMARY['size_yesno']}**.",
       f"- Bridge, answer options (described): {SUMMARY['bridge_choice_described']}.",
       f"- Size, answer options (described): {SUMMARY['size_choice_described']}.",
       "- Nothing here changes R1, R2, R3, any result of the first study, or the results of the first scale plan.", ""]
REPORT = "\n".join(md)
open(os.path.join(OUT, "report_scale2.md"), "w", encoding="utf-8").write(REPORT)
print(REPORT)

# %% [markdown]
# ## Save everything and make one zip to download

# %%
rows = []
for t in MODELS:
    for ds in L.DATASETS:
        c, h = CELLS[f"{t}|{ds}"], CHOICE[f"{t}|{ds}"]
        rows.append({"model": t, "label": LABEL[t], "size_b": SIZE.get(t), "dataset": ds,
                     "yesno_valid": c["validity"]["passes"], "acc_C0": L._get(c, "by_condition", "C0", "accuracy"),
                     "a": L._get(c, "H3", "adjusted", "a"), "a_prime": L._get(c, "H3", "reading_adjusted", "a"),
                     "iu_p": L._get(c, "H3", "h3_test", "p"), "pattern_yesno": PATTERN[t]["yesno_cells"][ds],
                     "ab_valid": h["validity"]["passes"], "a_ab": L._get(h, "R2", "adjusted", "a"),
                     "a_prime_ab": L._get(h, "R2", "reading_adjusted", "a"), "iu_p_ab": L._get(h, "R2", "h3_test", "p"),
                     "pattern_ab": PATTERN[t]["choice_cells"][ds], "q3": c.get("Q3_interpretation")})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "cells_scale2.csv"), index=False)
RESULTS = {"check": CHECK, "summary": SUMMARY, "pattern": PATTERN, "bold": BOLD, "cells": CELLS, "choice_cells": CHOICE,
           "differences": DIFFS,
           "runs": {t: {k: RUNS[t].get(k) for k in ("study", "model_id", "model_revision", "size_b", "cannot_be_run",
                                                    "read_mode", "backend_info", "timings_seconds", "memcheck_summary")}
                    for t in RUNS},
           "model_checks": {t: {k: v for k, v in m.items() if k != "traceback"} for t, m in MODELCHECKS.items()},
           "inputs": {t: {"from": v[1], "sha256": v[3]} for t, v in SOURCES.items()}}
json.dump(RESULTS, open(os.path.join(OUT, "results_scale2.json"), "w"), indent=1, default=str)
ZIP = os.path.join(WORK, "scale2_analysis_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("report_scale2.md", "results_scale2.json", "cells_scale2.csv", "scale2_figure.png"):
        z.write(os.path.join(OUT, name), name)
print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB) | total {(time.time() - T0) / 60:.1f} min")
