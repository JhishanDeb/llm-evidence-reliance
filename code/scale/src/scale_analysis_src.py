# %% [markdown]
# # Exploratory scale check: analysis (Qwen3 4B to 32B)
#
# **What this notebook does.** It reads the `scale_<model>.zip` files of the scale check and the second study's
# Qwen3-8B run (`v2_qwen3-instruct.zip`, the float16 reference), checks that every run used the registered code and
# saw the same patients and prompts, and computes the measures and reading rules fixed in
# [`SCALE_CHECK.md`, tag `scale-plan`](https://github.com/JhishanDeb/llm-evidence-reliance/blob/scale-plan/SCALE_CHECK.md).
# Every measure comes from the registered code of the second study (`ler_main.py`, main-2.0). This analysis is
# exploratory: it is not part of either pre-registration and changes no registered result.
#
# **How to run.** No GPU and no internet are needed. On Kaggle: upload the `scale_*.zip` files and
# `v2_qwen3-instruct.zip` (from `results/v2/` in the repository) as one dataset (Add Input → Upload → New Dataset)
# and Save & Run All. Kaggle unpacks zip files; the first code cell accepts zip files or unpacked folders. Locally or
# on Colab: put the zip files next to the notebook and run all cells. Runtime: about 20 to 40 minutes on a CPU.
# Download `scale_analysis_results.zip` at the end.

# %%
import os, sys, json, glob, gzip, zipfile, hashlib, time
import numpy as np
import pandas as pd

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
IN_DIR = os.path.join(WORK, "scale_analysis_inputs")
OUT = os.path.join(WORK, "scale_analysis_results")
SKIP = [os.path.realpath(IN_DIR), os.path.realpath(OUT)]
REF_TAG = "qwen3-instruct"


def _skipped(p):
    rp = os.path.realpath(p)
    return any(rp == s or rp.startswith(s + os.sep) for s in SKIP)


# A run arrives as a zip file or as an unpacked folder. Each source is reduced to the files the analysis reads.
SOURCES, DUPES = {}, {}
WANT = ("run_log_{t}.json", "modelcheck_{t}.json", "records_{t}.csv", "example_prompts_{t}.md", "memcheck_{t}.json")


def _add(where, read, tag):
    log_raw = read(f"run_log_{tag}.json")
    if log_raw is None or read("ler_main.py") is None:
        return
    log = json.loads(log_raw)
    if log.get("study") == "scale":
        kind = "scale"
    elif log.get("study") == "v2" and tag == REF_TAG:
        kind = "reference"
    else:
        return
    files = {"ler_main.py": read("ler_main.py")}
    if read("ler_scale.py") is not None:
        files["ler_scale.py"] = read("ler_scale.py")
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
    SOURCES[tag] = (kind, where, files, sig)


for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        if _skipped(p) or not os.path.basename(p).startswith(("scale_", "v2_")):
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
if REF_TAG not in SOURCES:
    raise RuntimeError("v2_qwen3-instruct.zip (the second study's Qwen3-8B run, results/v2/ in the repository) is "
                       "needed as the float16 reference. Attach it.")
if not any(k == "scale" for k, *_ in SOURCES.values()):
    raise RuntimeError("No scale_<model>.zip found. Attach the scale-check runs.")
os.makedirs(OUT, exist_ok=True)
for tag, (kind, where, files, sig) in sorted(SOURCES.items()):
    d = os.path.join(IN_DIR, tag)
    os.makedirs(d, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    print(f"{kind:9s} {tag}: from {where}")
    for name in sorted(sig):
        print(f"    {name:36s} sha256 {sig[name][:16]}")

# %% [markdown]
# ## Load the runs and check that they are comparable
# Every run used the registered `ler_main.py` (main-2.0); every scale run used the same `ler_scale.py` and the same
# plan; all runs saw the same patients, cards and partners; a prompt shared by two runs has the same text; and the
# rendered example prompts (what the model sees, through its chat template) are compared with the reference.

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


SCALE_TAGS = sorted(t for t, v in SOURCES.items() if v[0] == "scale")
sys.path.insert(0, os.path.join(IN_DIR, SCALE_TAGS[0]))
for _m in ("ler_scale", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_scale as S

if os.environ.get("LER_TEST_BOOT"):          # offline tests only: fewer bootstrap resamples
    L.BOOT_RESAMPLES = int(os.environ["LER_TEST_BOOT"])
problems, warnings = [], []
code_sha = {t: sha(os.path.join(IN_DIR, t, "ler_main.py")) for t in SOURCES}
bad = {t: s[:16] for t, s in code_sha.items() if s != S.REGISTERED_CODE_SHA256}
if bad:
    problems.append(f"runs not made with the registered ler_main.py main-2.0: {bad}")
scale_sha = {t: sha(os.path.join(IN_DIR, t, "ler_scale.py")) for t in SCALE_TAGS}
if len(set(scale_sha.values())) != 1:
    problems.append(f"the scale runs used different versions of ler_scale.py: { {t: s[:16] for t, s in scale_sha.items()} }")

RUNS, RECS, MODELCHECKS = {}, {}, {}
for tag in SOURCES:
    d = os.path.join(IN_DIR, tag)
    RUNS[tag] = json.load(open(os.path.join(d, f"run_log_{tag}.json")))
    if os.path.exists(os.path.join(d, f"records_{tag}.csv")):
        RECS[tag] = pd.read_csv(os.path.join(d, f"records_{tag}.csv"), dtype={"id": str})
    if os.path.exists(os.path.join(d, f"modelcheck_{tag}.json")):
        MODELCHECKS[tag] = json.load(open(os.path.join(d, f"modelcheck_{tag}.json")))
TAG_OF = {m["tag"]: mid for mid, m in S.SCALE_MODELS.items()}
SIZE = {m["tag"]: m["size_b"] for m in S.SCALE_MODELS.values()}
RAN = [t for t in SCALE_TAGS if not RUNS[t].get("cannot_be_run") and t in RECS]
CANNOT = {t: RUNS[t].get("model_check", {}).get("problems") for t in SCALE_TAGS if RUNS[t].get("cannot_be_run")}
NOT_ATTACHED = [m["tag"] for m in S.SCALE_MODELS.values() if m["tag"] not in SOURCES]
for t in SCALE_TAGS:
    r = RUNS[t]
    if t not in TAG_OF:
        problems.append(f"{t} is not a scale-check model")
    elif not r.get("mock") and r.get("model_revision") != S.SCALE_MODELS[TAG_OF[t]]["revision"]:
        problems.append(f"{t}: revision {r.get('model_revision')} differs from the pinned one")
if len({RUNS[t]["plan"]["sha256"] for t in SCALE_TAGS}) != 1:
    problems.append("the scale runs checked different plan texts")
ref = RUNS[REF_TAG]
if not ref.get("mock") and ref.get("model_revision") != S.REFERENCE["revision"]:
    problems.append("the reference run is not Qwen3-8B at the registered revision")
fps = {t: json.dumps(RUNS[t]["fingerprints"], sort_keys=True) for t in [REF_TAG] + RAN}
if len(set(fps.values())) != 1:
    problems.append("different patients, features, cards or partners")
_sha_by_key = {}
for t, df in RECS.items():
    for key, ts in zip(df["key"], df["text_sha"]):
        if _sha_by_key.setdefault(key, ts) != ts:
            problems.append(f"prompt {key} has different texts in different runs")
            break
if len({RUNS[t].get("read_mode") for t in [REF_TAG] + RAN}) > 1:
    problems.append("the runs did not all use the same reading mode")


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
_ref_blocks = example_blocks(REF_TAG)
for t in RAN:
    b = example_blocks(t)
    common = sorted(set(b) & set(_ref_blocks))
    diff = [h for h in common if b[h] != _ref_blocks[h]]
    RENDER[t] = {"compared": len(common), "identical": len(common) - len(diff), "first_difference": diff[:1]}
    if not common or diff:
        warnings.append(f"{t}: {len(diff)} of {len(common)} rendered example prompts differ from the reference "
                        "(the chat template of the 4-bit checkpoint renders them differently)")
if any(RUNS[t].get("mock") or RUNS[t].get("local_model") for t in RUNS) or os.environ.get("LER_TEST_BOOT"):
    warnings.append("TEST RUN: mock inputs or a reduced number of bootstrap resamples")
CHECK = {"code_sha256": S.REGISTERED_CODE_SHA256, "ler_scale_sha256": sorted(set(scale_sha.values())),
         "plan_sha256": sorted({RUNS[t]["plan"]["sha256"] for t in SCALE_TAGS}), "ran": RAN, "cannot_be_run": CANNOT,
         "not_attached": NOT_ATTACHED, "rendered_example_prompts": RENDER, "problems": problems, "warnings": warnings,
         "boot_resamples": L.BOOT_RESAMPLES}
print(json.dumps(CHECK, indent=1, default=str))
if problems:
    raise RuntimeError("The runs are not comparable: " + "; ".join(problems))

# %% [markdown]
# ## Every measure for every model × dataset cell
# The reference is restricted to the prompts of the scale grid (the same 13 per patient). Yes/No: alarm framing,
# example set A, plus the reversed framing for Q3. Answer options: the averaged reading of both option orders.

# %%
t0 = time.time()
GRID_KEYS = {f"{c}|{f}|{s}" for c, f, s in S.scale_grid()}
_cfs = lambda df: df["cond"] + "|" + df["framing"] + "|" + df["shot_set"]
RECS[REF_TAG] = RECS[REF_TAG][_cfs(RECS[REF_TAG]).isin(GRID_KEYS)].reset_index(drop=True)
MODELS = [REF_TAG] + sorted(RAN, key=lambda t: (SIZE[t], t))
LABEL = {REF_TAG: "Qwen3-8B (float16, second study)"} | {t: f"{TAG_OF[t].split('/')[1]}" for t in SCALE_TAGS if t in TAG_OF}
CELLS, CHOICE = {}, {}
for tag in MODELS:
    df = RECS[tag]
    for ds in L.DATASETS:
        sub = df[df["dataset"] == ds]
        CELLS[f"{tag}|{ds}"] = L.analyse_cell(sub, exploratory=False, trust=False)
        CHOICE[f"{tag}|{ds}"] = L.analyse_choice_cell(sub)
        v, w = CELLS[f"{tag}|{ds}"]["validity"], CHOICE[f"{tag}|{ds}"]["validity"]
        print(f"  {tag:15s} {ds:14s} Yes/No {'passes' if v['passes'] else 'EXCLUDED'} (mass {v['median_yn_mass']:.3f}, "
              f"C0 acc {v['c0_accuracy']:.3f}) | A/B {'passes' if w['passes'] else 'EXCLUDED'} (mass "
              f"{w['median_ab_mass']:.3f}, C0 acc {w['c0_accuracy']:.3f}) | {time.time() - t0:.0f}s")


def _sub(tag, ds):
    return RECS[tag][RECS[tag]["dataset"] == ds]


def choice_difference(dx, dy):
    """a(x) - a(y) and a'(x) - a'(y) on the averaged answer-option reading, paired over the same patients."""
    px, py = L.choice_pooled(dx), L.choice_pooled(dy)
    if px is None or py is None:
        return None
    return L._paired_asym_difference(L._wide(px), L._wide(py))


# paired differences: quantization (8B 4-bit minus 8B float16) and size (each other 4-bit model minus 8B 4-bit)
PAIRS = []
if S.QUANT_CONTROL in RAN:
    PAIRS.append(("quantization", S.QUANT_CONTROL, REF_TAG))
    PAIRS += [("size", t, S.QUANT_CONTROL) for t in MODELS if t not in (REF_TAG, S.QUANT_CONTROL)]
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
print(f"analysis done in {time.time() - t0:.0f}s ({L.BOOT_RESAMPLES} bootstrap resamples per estimate)")

# %% [markdown]
# ## The reading rules of the plan
# A cell shows the pattern if it is valid, a < 0 and a′ < 0, and the intersection–union p < 0.05 (no correction).
# A model shows the pattern if at least 2 of the 3 datasets do.

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


def counts(fmt):
    """'model k/3' for every model, in order of size."""
    return ", ".join(f"{LABEL[t]} {sum(PATTERN[t][fmt + '_cells'].values())}/3" for t in MODELS)


def size_statement(fmt):
    """The size rule of the plan (Yes/No question). For answer options the same rule is applied and only described."""
    key = f"{fmt}_model"
    if S.QUANT_CONTROL not in RAN:
        return "not interpretable: the 8B 4-bit control was not run"
    if not PATTERN[S.QUANT_CONTROL][key]:
        return ("reported but not interpreted: Qwen3-8B-AWQ does not show the pattern"
                + (", so 4-bit weights may change it" if PATTERN[REF_TAG][key] else
                   ", and neither does Qwen3-8B in float16") + f" (datasets with the pattern: {counts(fmt)})")
    larger = [t for t in RAN if SIZE[t] > 8]
    if not larger:
        return "no larger model was run"
    missing = [TAG_OF[m["tag"]].split("/")[1] for m in S.SCALE_MODELS.values() if m["size_b"] > 8 and m["tag"] not in RAN]
    note = f" ({', '.join(missing)} not run)" if missing else ""
    if all(PATTERN[t][key] for t in larger):
        return "persists with size" + note + f" (datasets with the pattern: {counts(fmt)})"
    return ("weakens with size: " + ", ".join(LABEL[t] for t in larger if not PATTERN[t][key]) + " does not show it"
            + note + f" (datasets with the pattern: {counts(fmt)})")


def quant_statement(fmt):
    key = f"{fmt}_model"
    if S.QUANT_CONTROL not in RAN:
        return "not run"
    q, r = PATTERN[S.QUANT_CONTROL][key], PATTERN[REF_TAG][key]
    head = ("4-bit weights are taken not to remove the pattern (Qwen3-8B-AWQ shows it)" if q else
            "Qwen3-8B-AWQ does not show the pattern" + (" although Qwen3-8B in float16 does" if r else
                                                         ", nor does Qwen3-8B in float16"))
    return head + (f" ({LABEL[S.QUANT_CONTROL]} {sum(PATTERN[S.QUANT_CONTROL][fmt + '_cells'].values())}/3, "
                   f"float16 {sum(PATTERN[REF_TAG][fmt + '_cells'].values())}/3)")


SUMMARY = {
    "quantization_yesno": quant_statement("yesno"),
    "size_yesno": size_statement("yesno"),
    "quantization_choice_described": quant_statement("choice"),
    "size_choice_described": size_statement("choice"),
    "per_model": {t: {"yesno": f"{sum(PATTERN[t]['yesno_cells'].values())}/3",
                      "choice": f"{sum(PATTERN[t]['choice_cells'].values())}/3"} for t in MODELS},
}
print(json.dumps(SUMMARY, indent=1))

# %% [markdown]
# ## Figure: a and a′ by model size

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS_NAME = {"physionet2012": "PhysioNet 2012", "physionet2019": "PhysioNet 2019", "diabetes130": "Diabetes 130"}
COL_AP, COL_A, COL_OFF = "#2a6fdb", "#8a8f98", "#c9ccd1"
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
xs = {t: np.log2(SIZE[t]) for t in RAN}
xs[REF_TAG] = np.log2(8) + 0.28
for row, (fmt, getc, key) in enumerate((("Yes/No question", CELLS, "H3"), ("answer options (A/B)", CHOICE, "R2"))):
    for col, ds in enumerate(L.DATASETS):
        ax = axes[row, col]
        ax.axhline(0, color="#b0b4ba", lw=1)
        for stat, dx, colr, fill in (("reading_adjusted", -0.07, COL_AP, True), ("adjusted", 0.07, COL_A, False)):
            line_x, line_y = [], []
            for t in MODELS:
                c = getc[f"{t}|{ds}"]
                est, ci = L._get(c, key, stat, "a"), L._get(c, key, stat, "ci")
                if not S.finite(est):
                    continue
                valid = c["validity"]["passes"]
                cc = colr if valid else COL_OFF
                x = xs[t] + dx
                if ci and all(S.finite(v) for v in ci):
                    ax.plot([x, x], ci, color=cc, lw=1.4)
                ax.plot([x], [est], marker="D" if t == REF_TAG else "o", ms=6.5, color=cc,
                        markerfacecolor=cc if fill else "white", markeredgewidth=1.5, zorder=3)
                if t != REF_TAG and valid:
                    line_x.append(x)
                    line_y.append(est)
            if len(line_x) > 1:
                o = np.argsort(line_x)
                ax.plot(np.array(line_x)[o], np.array(line_y)[o], color=colr, lw=1, alpha=0.5,
                        ls="-" if fill else "--")
        if row == 0:
            ax.set_title(DS_NAME[ds], fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{fmt}\n← safe evidence moves it more", fontsize=9)
        ax.set_xticks([np.log2(s) for s in (4, 8, 14, 32)])
        ax.set_xticklabels(["4B", "8B", "14B", "32B"])
        ax.spines[["top", "right"]].set_visible(False)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", color=COL_AP, lw=0, label="a′ (4-bit)"),
                    Line2D([], [], marker="o", color=COL_A, markerfacecolor="white", lw=0, label="a (4-bit)"),
                    Line2D([], [], marker="D", color=COL_AP, lw=0, label="Qwen3-8B float16 (second study)"),
                    Line2D([], [], marker="o", color=COL_OFF, lw=0, label="cell excluded by validity")],
           loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Scale check (exploratory): asymmetry by Qwen3 model size, 95% intervals", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
FIG = os.path.join(OUT, "scale_figure.png")
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

md = ["# Exploratory scale check: results", "",
      f"Plan `{S.PLAN_TAG}` (SCALE_CHECK.md sha256 {CHECK['plan_sha256'][0][:16]}); registered code main-2.0 "
      f"(sha256 {S.REGISTERED_CODE_SHA256[:16]}); ler_scale {S.SCALE_VERSION}. Exploratory: not part of either "
      "pre-registration; no correction for multiple tests.", ""]
if warnings:
    md += ["**Warnings:** " + "; ".join(warnings), ""]
md += ["## Runs", "", "| model | revision | status | records | imputed | dev mass Yes/No, A/B | s per prompt (dev) | "
       "memorization flagged |", "|---|---|---|---|---|---|---|---|"]
for t in [REF_TAG] + sorted(SCALE_TAGS, key=lambda t: (SIZE.get(t, 0), t)):
    r, mc = RUNS[t], MODELCHECKS.get(t, {})
    status = "reference" if t == REF_TAG else ("cannot be run: " + "; ".join(CANNOT[t]) if t in CANNOT else "ran")
    flagged = [ds for ds, v in (r.get("memcheck_summary") or {}).items() if v.get("flag_verbatim_rows")]
    md.append(f"| {LABEL.get(t, t)} | {str(r.get('model_revision'))[:12]} | {status} | "
              f"{len(RECS[t]) if t in RECS else 0} | {int((RECS[t]['n_imputed'] > 0).sum()) if t in RECS else '–'} | "
              f"{f3(L._get(mc, 'reading_yesno', 'mass_median'))}, {f3(L._get(mc, 'reading_choice', 'mass_median'))} | "
              f"{f3(L._get(mc, 'reading_yesno', 'seconds_per_prompt'))} | {', '.join(flagged) or 'none'} |")
for t in NOT_ATTACHED:
    md.append(f"| {TAG_OF[t].split('/')[1]} | – | not attached | – | – | – | – | – |")
md += ["", "Rendered example prompts compared with the reference: " +
       "; ".join(f"{t} {v['identical']}/{v['compared']} identical" for t, v in RENDER.items()), ""]

md += ["## Validity", "", "| model | dataset | Yes/No: mass | Yes/No: C0 accuracy | Yes/No | A/B: mass | A/B: C0 accuracy | A/B |",
       "|---|---|---|---|---|---|---|---|"]
for t in MODELS:
    for ds in L.DATASETS:
        v, w = CELLS[f"{t}|{ds}"]["validity"], CHOICE[f"{t}|{ds}"]["validity"]
        md.append(f"| {LABEL[t]} | {ds} | {f3(v['median_yn_mass'])} | {f3(v['c0_accuracy'])} | "
                  f"{'passes' if v['passes'] else '**excluded**'} | {f3(w['median_ab_mass'])} | {f3(w['c0_accuracy'])} | "
                  f"{'passes' if w['passes'] else '**excluded**'} |")

md += ["", "## Yes/No question: a and a′ (alarm framing, example set A)", "",
       "| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |", "|---|---|---|---|---|---|"]
for t in MODELS:
    for ds in L.DATASETS:
        h = CELLS[f"{t}|{ds}"].get("H3")
        valid = CELLS[f"{t}|{ds}"]["validity"]["passes"]
        md.append(f"| {LABEL[t]} | {ds} | {est(h, 'adjusted')} | {est(h, 'reading_adjusted')} | "
                  f"{fp(L._get(h, 'h3_test', 'p'))} | {yes(PATTERN[t]['yesno_cells'][ds])}{'' if valid else ' (excluded)'} |")
md += ["", "| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |", "|---|---|---|"]
for t in MODELS:
    md.append(f"| {LABEL[t]} | {sum(PATTERN[t]['yesno_cells'].values())}/3 | {yes(PATTERN[t]['yesno_model'])} |")

md += ["", "## Answer options: a and a′ on the averaged reading of both option orders (described)", "",
       "| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |", "|---|---|---|---|---|---|"]
for t in MODELS:
    for ds in L.DATASETS:
        h = CHOICE[f"{t}|{ds}"].get("R2")
        valid = CHOICE[f"{t}|{ds}"]["validity"]["passes"]
        md.append(f"| {LABEL[t]} | {ds} | {est(h, 'adjusted')} | {est(h, 'reading_adjusted')} | "
                  f"{fp(L._get(h, 'h3_test', 'p'))} | {yes(PATTERN[t]['choice_cells'][ds])}{'' if valid else ' (excluded)'} |")
md += ["", "| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |", "|---|---|---|"]
for t in MODELS:
    md.append(f"| {LABEL[t]} | {sum(PATTERN[t]['choice_cells'].values())}/3 | {yes(PATTERN[t]['choice_model'])} |")

md += ["", "## Paired differences over the same patients", "",
       "x − y; below 0 means the asymmetry toward safe-pointing evidence is larger in x.", "",
       "| comparison | dataset | format | both cells valid | a difference [95% CI] | a′ difference [95% CI] | p |",
       "|---|---|---|---|---|---|---|"]
for k, d in DIFFS.items():
    for fmt in ("yesno", "choice"):
        r = d[fmt]
        md.append(f"| {LABEL[d['x']]} − {LABEL[d['y']]} | {d['dataset']} | {'Yes/No' if fmt == 'yesno' else 'A/B'} | "
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
                  f"{f2(L._get(c, 'H1', 'diff'))} ({fp(L._get(c, 'H1', 'p'))}) | "
                  f"{f2(L._get(c, 'H2', 'diff'))} ({fp(L._get(c, 'H2', 'p'))}) |")

md += ["", "## Reading rules of the plan", "",
       f"- Quantization, Yes/No question: {SUMMARY['quantization_yesno']}.",
       f"- Size, Yes/No question: **{SUMMARY['size_yesno']}**.",
       f"- Quantization, answer options (described): {SUMMARY['quantization_choice_described']}.",
       f"- Size, answer options (described): {SUMMARY['size_choice_described']}.",
       "- Nothing here changes R1, R2, R3 or any result of the first study.", ""]
REPORT = "\n".join(md)
open(os.path.join(OUT, "report_scale.md"), "w", encoding="utf-8").write(REPORT)
print(REPORT)

# %% [markdown]
# ## Save everything and make one zip to download

# %%
rows = []
for t in MODELS:
    for ds in L.DATASETS:
        c, h = CELLS[f"{t}|{ds}"], CHOICE[f"{t}|{ds}"]
        rows.append({"model": t, "size_b": SIZE.get(t, 8), "float16": t == REF_TAG, "dataset": ds,
                     "yesno_valid": c["validity"]["passes"], "acc_C0": L._get(c, "by_condition", "C0", "accuracy"),
                     "a": L._get(c, "H3", "adjusted", "a"), "a_prime": L._get(c, "H3", "reading_adjusted", "a"),
                     "iu_p": L._get(c, "H3", "h3_test", "p"), "pattern_yesno": PATTERN[t]["yesno_cells"][ds],
                     "ab_valid": h["validity"]["passes"], "a_ab": L._get(h, "R2", "adjusted", "a"),
                     "a_prime_ab": L._get(h, "R2", "reading_adjusted", "a"), "iu_p_ab": L._get(h, "R2", "h3_test", "p"),
                     "pattern_ab": PATTERN[t]["choice_cells"][ds], "q3": c.get("Q3_interpretation")})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "cells_scale.csv"), index=False)
RESULTS = {"check": CHECK, "summary": SUMMARY, "pattern": PATTERN, "cells": CELLS, "choice_cells": CHOICE,
           "differences": DIFFS, "runs": {t: {k: RUNS[t].get(k) for k in ("model_id", "model_revision", "size_b",
                                                                          "cannot_be_run", "read_mode", "backend_info",
                                                                          "timings_seconds", "memcheck_summary")}
                                          for t in RUNS},
           "model_checks": {t: {k: v for k, v in m.items() if k != "traceback"} for t, m in MODELCHECKS.items()},
           "inputs": {t: {"from": v[1], "sha256": v[3]} for t, v in SOURCES.items()}}
json.dump(RESULTS, open(os.path.join(OUT, "results_scale.json"), "w"), indent=1, default=str)
ZIP = os.path.join(WORK, "scale_analysis_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("report_scale.md", "results_scale.json", "cells_scale.csv", "scale_figure.png"):
        z.write(os.path.join(OUT, name), name)
print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB) | total {(time.time() - T0) / 60:.1f} min")
