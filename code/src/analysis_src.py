# %% [markdown]
# # Registered analysis: LLM evidence reliance
#
# **What this notebook does.** It reads the four `main_<model>.zip` files from the main runs, checks that all
# four models saw exactly the same patients and prompts, and runs the analysis registered in
# [`PREREGISTRATION.md` (prereg-v1.1)](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v1.1):
# validity checks, H1–H3 with Holm over the 6 primary cells and the supported-overall rule, the Q3 framing table,
# the secondary hypotheses H4–H6, the secondary asymmetry measures, and the exploratory analyses (labelled).
#
# **How to run.** It needs no GPU. On Kaggle: upload the four zip files as one dataset (Datasets → New Dataset),
# add it to this notebook (Add Input), then **Save & Run All**. Locally or on Colab: put the four zip files next
# to the notebook and run all cells. Runtime: about 15–30 minutes on a CPU (10,000 bootstrap resamples per test).
# Download `analysis_results.zip` at the end.

# %%
import os, sys, json, glob, zipfile, hashlib, time
import numpy as np
import pandas as pd


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
ZIPS, DUPES = {}, {}
for root in SEARCH:
    for p in glob.glob(os.path.join(root, "**", "main_*.zip"), recursive=True):
        tag = os.path.basename(p)[len("main_"):-len(".zip")]
        rp = os.path.realpath(p)
        if tag in ZIPS and os.path.realpath(ZIPS[tag]) != rp and sha_file(ZIPS[tag]) != sha_file(p):
            DUPES.setdefault(tag, [ZIPS[tag]]).append(p)
        ZIPS.setdefault(tag, p)
if DUPES:
    raise RuntimeError(f"More than one different zip for the same model: {DUPES}. Attach only the final zip of each run.")
print("found:", json.dumps(ZIPS, indent=1))
if not ZIPS:
    raise RuntimeError("No main_<model>.zip found. Attach the four zip files (Kaggle: Add Input) or put them next to "
                       "this notebook.")
IN_DIR = os.path.join(WORK, "analysis_inputs")
OUT = os.path.join(WORK, "analysis_results")
os.makedirs(OUT, exist_ok=True)
for tag, p in ZIPS.items():
    with zipfile.ZipFile(p) as z:
        z.extractall(os.path.join(IN_DIR, tag))

# %% [markdown]
# ## Load the runs and check that they are comparable
# Same analysis code, same protocol text, same test patients, cards and partners, and the same prompt text for
# every record, in every run.

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


code_sha = {tag: sha(os.path.join(IN_DIR, tag, "ler_main.py")) for tag in ZIPS}
if len(set(code_sha.values())) != 1:
    raise RuntimeError(f"The runs used different versions of ler_main.py: {code_sha}")
first = sorted(ZIPS)[0]
sys.path.insert(0, os.path.join(IN_DIR, first))
if "ler_main" in sys.modules:
    del sys.modules["ler_main"]
import ler_main as L

RUNS, RECS = {}, {}
for tag in ZIPS:
    RUNS[tag] = json.load(open(os.path.join(IN_DIR, tag, f"run_log_{tag}.json")))
    RECS[tag] = pd.read_csv(os.path.join(IN_DIR, tag, f"records_{tag}.csv.gz"), dtype={"id": str})
    assert RUNS[tag]["model_tag"] == tag, (tag, RUNS[tag]["model_tag"])

CHECK = {"code_version": L.CODE_VERSION, "code_sha256": code_sha[first]}
problems = []
if len({r["protocol"]["sha256"] for r in RUNS.values()}) != 1:
    problems.append("different protocol texts")
fps = {tag: json.dumps(r["fingerprints"], sort_keys=True) for tag, r in RUNS.items()}
if len(set(fps.values())) != 1:
    problems.append("different patients, features, cards or partners")
keysets = {tag: tuple(sorted(df["key"] + "|" + df["text_sha"])) for tag, df in RECS.items()}
if len(set(keysets.values())) != 1:
    problems.append("different prompt sets or prompt texts")
if len({r.get("read_mode") for r in RUNS.values()}) > 1 or len({r["backend_info"].get("backend") for r in RUNS.values()}) > 1:
    problems.append("WARNING: the runs did not all use the same backend and reading mode (see the Runs table); "
                    "H4 comparisons across them are marked as affected")
if len({r["n_records"] for r in RUNS.values()}) > 1:
    problems.append("different numbers of records")
if any(r.get("mock") or r.get("local_model") for r in RUNS.values()):
    problems.append("TEST RUN: at least one input comes from a mock or local test model")
CHECK["problems"] = problems
CHECK["models"] = {tag: {"model_id": r["model_id"], "revision": r["model_revision"], "backend": r["backend_info"].get("backend"),
                         "read_mode": r.get("read_mode"), "n_records": r["n_records"],
                         "n_invalid": int((RECS[tag]["valid"] == 0).sum()),
                         "n_imputed": int((RECS[tag]["n_imputed"] > 0).sum())} for tag, r in RUNS.items()}
CHECK["missing_models"] = [t for t in L.MODEL_TAGS if t not in RUNS]
print(json.dumps(CHECK, indent=1))
if [p for p in problems if not p.startswith(("TEST RUN", "WARNING"))]:
    raise RuntimeError("The runs are not comparable: " + "; ".join(problems))

# %% [markdown]
# ## Every measure for every model × dataset cell
# (alarm framing, example set A unless stated; 10,000 bootstrap resamples per test)

# %%
t0 = time.time()
CELLS = {}
for tag, df in RECS.items():
    for ds in L.DATASETS:
        CELLS[f"{tag}|{ds}"] = L.analyse_cell(df[df["dataset"] == ds])
        v = CELLS[f"{tag}|{ds}"]["validity"]
        print(f"  {tag:15s} {ds:14s} validity: median yn_mass {v['median_yn_mass']:.3f}, C0 accuracy "
              f"{v['c0_accuracy']:.3f} -> {'passes' if v['passes'] else 'EXCLUDED'} | {time.time() - t0:.0f}s")

H4 = {}
for fam in ("qwen", "llama"):
    ti, tb = f"{fam}-instruct", f"{fam}-base"
    for ds in L.DATASETS:
        ci, cb = CELLS.get(f"{ti}|{ds}"), CELLS.get(f"{tb}|{ds}")
        if ci and cb and ci["validity"]["passes"] and cb["validity"]["passes"]:
            H4[f"{fam}|{ds}"] = L.adjusted_difference(RECS[ti][RECS[ti]["dataset"] == ds],
                                                      RECS[tb][RECS[tb]["dataset"] == ds])
        else:
            H4[f"{fam}|{ds}"] = None
CONF = L.confirmatory(CELLS, H4)
print(f"analysis done in {time.time() - t0:.0f}s")


# Deviation D5, labelled sensitivity analysis: in records where one variant was outside the top 1,000 (imputed), the
# probability of the less likely word is halved, which bounds the effect of the imputation; a and a' are recomputed.
def halve_less_likely(df):
    df = df.copy()
    m = df["n_imputed"] > 0
    lower_yes = df["lp_yes"] < df["lp_no"]
    df.loc[m & lower_yes, "lp_yes"] -= np.log(2)
    df.loc[m & ~lower_yes, "lp_no"] -= np.log(2)
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
    for tag, df in RECS.items():
        if int((df["n_imputed"] > 0).sum()) == 0:
            continue
        adj = halve_less_likely(df)
        for ds in L.DATASETS:
            r = L.asymmetry(L.slice_records(adj[adj["dataset"] == ds]))
            SENS_IMPUTED[f"{tag}|{ds}"] = {"a": L._get(r, "adjusted", "a"), "a_ci": L._get(r, "adjusted", "ci"),
                                           "a_prime": L._get(r, "reading_adjusted", "a"),
                                           "a_prime_ci": L._get(r, "reading_adjusted", "ci"),
                                           "share_imputed": float((df[df["dataset"] == ds]["n_imputed"] > 0).mean())}
finally:
    L.BOOT_RESAMPLES = _boot
print(f"imputation sensitivity done in {time.time() - t0:.0f}s ({len(SENS_IMPUTED)} cells)")

# %% [markdown]
# ## Results tables

# %%
f2 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"
f3 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
fp = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.2g}" if x >= 0.001 else "<0.001")
ci = lambda c: "–" if not c or any(v is None or np.isnan(v) for v in c) else f"[{c[0]:+.2f}, {c[1]:+.2f}]"
INSTRUCT = [t for t in ("qwen-instruct", "llama-instruct")]
md = ["# Registered analysis: results", "",
      f"Protocol `{L.PROTOCOL_TAG}`; analysis code {L.CODE_VERSION} (sha256 {CHECK['code_sha256'][:16]}).", ""]
if problems:
    md += [f"**Warning:** {'; '.join(problems)}", ""]
if CHECK["missing_models"]:
    md += [f"**Missing runs:** {', '.join(CHECK['missing_models'])} (their cells count as not supporting).", ""]
md += ["## Runs", "", "| model | revision | backend | read | records | invalid | imputed |", "|---|---|---|---|---|---|---|"]
for tag, m in CHECK["models"].items():
    md.append(f"| {m['model_id']} | {str(m['revision'])[:12]} | {m['backend']} | {m['read_mode']} | {m['n_records']} | "
              f"{m['n_invalid']} | {m['n_imputed']} |")
md += ["", "## Validity (Section 5, D3.3)", "", "| model | dataset | median Yes+No mass | C0 accuracy | status |",
       "|---|---|---|---|---|"]
for k, c in CELLS.items():
    tag, ds = k.split("|")
    v = c["validity"]
    md.append(f"| {tag} | {ds} | {f3(v['median_yn_mass'])} | {f3(v['c0_accuracy'])} | "
              f"{'passes' if v['passes'] else '**excluded**'} |")
md += ["", "## Memorization check (Section 3, D3.7)", "",
       "| model | dataset | file | header: chars matched | rows completed verbatim | flagged |", "|---|---|---|---|---|---|"]
for tag, r in RUNS.items():
    for ds, m in (r.get("memcheck_summary") or {}).items():
        md.append(f"| {tag} | {ds} | {m.get('file')} | {m.get('header_matched_chars')} | {m.get('row_exact_matches')} | "
                  f"{'**yes**' if m.get('flag_verbatim_rows') else 'no'} |")

md += ["", "## Primary hypotheses (instruct models; Holm over the 6 cells)", ""]
for h, label in (("H1", "True evidence helps: accuracy C2 > C0 (exact McNemar)"),
                 ("H2", "Flipped evidence hurts: accuracy C4 < C0 (exact McNemar)"),
                 ("H3", "Asymmetry toward alarm: a > 0 and a' > 0 (D1; intersection-union test, p = max(p_a, p_a'))")):
    fam = CONF["primary"][h]
    if h == "H3":
        md += [f"### {h}. {label}", "",
               "| model | dataset | a [95% CI] (p) | a' [95% CI] (p) | test p | Holm p | supported | registered Δ = A − S [95% CI] |",
               "|---|---|---|---|---|---|---|---|"]
    else:
        md += [f"### {h}. {label}", "", "| model | dataset | accuracy difference | p | Holm p | supported |",
               "|---|---|---|---|---|---|"]
    for k, e in fam["cells"].items():
        c = CELLS.get(k)
        tag, ds = k.split("|")
        sup = f"{'**yes**' if e['supported'] else 'no'}{'' if e['included'] else ' (not included)'}"
        if h == "H3":
            g = lambda *ks: L._get(c, "H3", *ks)
            md.append(f"| {tag} | {ds} | {f2(g('adjusted', 'a'))} {ci(g('adjusted', 'ci'))} ({fp(g('adjusted', 'p_boot'))}) | "
                      f"{f2(g('reading_adjusted', 'a'))} {ci(g('reading_adjusted', 'ci'))} ({fp(g('reading_adjusted', 'p_boot'))}) | "
                      f"{fp(e['p'])} | {fp(e['p_holm'])} | {sup} | {f2(g('delta', 'delta'))} {ci(g('delta', 'ci'))} |")
        else:
            md.append(f"| {tag} | {ds} | {f2(L._get(c, h, 'diff'))} | {fp(e['p'])} | {fp(e['p_holm'])} | {sup} |")
    ov = fam["overall"]
    per = "; ".join(f"{t}: {v['datasets_supported']}/3 datasets" for t, v in ov["per_model"].items())
    md += ["", f"**{h} supported overall: {'YES' if ov['supported_overall'] else 'NO'}** ({per}; rule: at least 2 of "
           "3 datasets for each instruct model)", ""]
    if h == "H3":
        md += ["Where a, a' and the registered Δ do not agree in sign, the cell is described as not robust in the paper "
               "(Deviation D1).", ""]

md += ["## Q3. Does the asymmetry follow the outcome or the answer word? (D1 four-way table)", "",
       "| model | dataset | a: alarm / reversed framing [95% CI] | a': alarm / reversed framing [95% CI] | interpretation |",
       "|---|---|---|---|---|"]
for tag in INSTRUCT:
    for ds in L.DATASETS:
        c = CELLS.get(f"{tag}|{ds}")
        if not c:
            continue
        cols = []
        for key in ("adjusted", "reading_adjusted"):
            cols.append(f"{f2(L._get(c, 'H3', key, 'a'))} {ci(L._get(c, 'H3', key, 'ci'))} / "
                        f"{f2(L._get(c, 'Q3_reversed', key, 'a'))} {ci(L._get(c, 'Q3_reversed', key, 'ci'))}")
        md.append(f"| {tag} | {ds} | {cols[0]} | {cols[1]} | "
                  f"{c['Q3_interpretation']}{'' if c['validity']['passes'] else ' (cell excluded)'} |")

md += ["", "## Secondary hypotheses (Holm within each over 6)", ""]
md += ["### H4. Instruction tuning increases the asymmetry: a(instruct) − a(base) > 0 and the same for a' "
       "(intersection-union test)", "",
       "| family / dataset | a difference [95% CI] | a' difference [95% CI] | test p | Holm p | supported |",
       "|---|---|---|---|---|---|"]
for k, e in CONF["secondary"]["H4"]["cells"].items():
    r = H4.get(k)
    fam_ = k.split("|")[0]
    modes = {RUNS[t].get("read_mode") for t in (f"{fam_}-instruct", f"{fam_}-base") if t in RUNS}
    note = " (affected: the two runs used different reading modes)" if len(modes) > 1 else ""
    md.append(f"| {k.replace('|', ' / ')}{note} | {f2(L._get(r, 'diff'))} {ci(L._get(r, 'ci'))} | "
              f"{f2(L._get(r, 'reading_adjusted_diff', 'diff'))} {ci(L._get(r, 'reading_adjusted_diff', 'ci'))} | "
              f"{fp(e['p'])} | {fp(e['p_holm'])} | {'**yes**' if e['supported'] else 'no'}"
              f"{'' if e['included'] else ' (not tested)'} |")
md.append("")
sec = [("H5", "Agreement-based trust over-trusts on flipped evidence: over-trust > 0",
        lambda k: (L._get(CELLS.get(k), "H5", "overtrust"), L._get(CELLS.get(k), "H5", "ci"))),
       ("H6", "The flip-consistency flag improves the trust AUROC (mixed setting; DeLong)",
        lambda k: (L._get(CELLS.get(k), "H6", "diff"), None))]
for h, label, get in sec:
    md += [f"### {h}. {label}", "", "| cell | estimate | 95% CI | p | Holm p | supported |", "|---|---|---|---|---|---|"]
    for k, e in CONF["secondary"][h]["cells"].items():
        est, cc = get(k)
        md.append(f"| {k.replace('|', ' / ')} | {f2(est)} | {ci(cc)} | {fp(e['p'])} | {fp(e['p_holm'])} | "
                  f"{'**yes**' if e['supported'] else 'no'}{'' if e['included'] else ' (not tested)'} |")
    md.append("")

md += ["## Sensitivity analyses for the asymmetry (labelled; D1)", "",
       "| model | dataset | a, d winsorized at 1%/99% [95% CI] | share of patients with expected baseline within 1 SD of 0 | first-stage R² |",
       "|---|---|---|---|---|"]
for k, c in CELLS.items():
    a_ = c.get("H3")
    if not a_:
        continue
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | {f2(L._get(a_, 'sensitivity_winsorized', 'a'))} {ci(L._get(a_, 'sensitivity_winsorized', 'ci'))} | "
              f"{f3(L._get(a_, 'adjusted', 'share_e_within_1sd_of_0'))} | {f3(L._get(a_, 'adjusted', 'coef', 'first_stage_r2'))} |")
md.append("")

md += ["## Secondary asymmetry measures (Section 6; not used for H3, Q3 or H4)", "",
       "| model | dataset | n | A | S | Δ = A − S [95% CI] | log-odds, unadjusted [95% CI] | follow alarm card | follow safe card | true card points to alarm | C0 share alarm |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, c in CELLS.items():
    a = c.get("H3")
    if not a:
        continue
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | {a['n']} | {f3(a['delta']['A'])} | {f3(a['delta']['S'])} | {f2(a['delta']['delta'])} "
              f"{ci(a['delta']['ci'])} | {f2(a['logodds_unadjusted']['value'])} {ci(a['logodds_unadjusted']['ci'])} | "
              f"{f3(a['follow_alarm_card'])} | {f3(a['follow_safe_card'])} | {f3(a['share_true_card_alarm'])} | "
              f"{f3(a['c0_share_alarm'])} |")

if SENS_IMPUTED:
    md += ["## Sensitivity to the top-1,000 reading (labelled; D5)", "",
           "In records where one variant of a word was outside the top 1,000, the less likely word's probability is "
           "halved, which bounds the effect of the imputation. 2,000 bootstrap resamples.", "",
           "| model | dataset | records with an imputed variant | a [95% CI] (main) | a, bounded | a' [95% CI] (main) | a', bounded |",
           "|---|---|---|---|---|---|---|"]
    for k, v in SENS_IMPUTED.items():
        c = CELLS.get(k)
        tag, ds = k.split("|")
        md.append(f"| {tag} | {ds} | {v['share_imputed']:.1%} | {f2(L._get(c, 'H3', 'adjusted', 'a'))} "
                  f"{ci(L._get(c, 'H3', 'adjusted', 'ci'))} | {f2(v['a'])} {ci(v['a_ci'])} | "
                  f"{f2(L._get(c, 'H3', 'reading_adjusted', 'a'))} {ci(L._get(c, 'H3', 'reading_adjusted', 'ci'))} | "
                  f"{f2(v['a_prime'])} {ci(v['a_prime_ci'])} |")
    md.append("")

md += ["", "## Exploratory (labelled; not used for conclusions)", "",
       "### Base models on H1–H3 (no multiplicity correction)", "",
       "| model | dataset | H1 diff (p) | H2 diff (p) | a [95% CI] (p) |", "|---|---|---|---|---|"]
for tag in ("qwen-base", "llama-base"):
    for ds in L.DATASETS:
        c = CELLS.get(f"{tag}|{ds}")
        if not c:
            continue
        md.append(f"| {tag} | {ds} | {f2(L._get(c, 'H1', 'diff'))} ({fp(L._get(c, 'H1', 'p'))}) | "
                  f"{f2(L._get(c, 'H2', 'diff'))} ({fp(L._get(c, 'H2', 'p'))}) | {f2(L._get(c, 'H3', 'adjusted', 'a'))} "
                  f"{ci(L._get(c, 'H3', 'adjusted', 'ci'))} ({fp(L._get(c, 'H3', 'adjusted', 'p_boot'))}) |")
md += ["", "### C6 (another patient's card) versus C4 (flipped card), and robustness across example sets", "",
       "| model | dataset | accuracy C4 / C6 | follow rate C2 / C4 / C6 | a with set A / B / C |", "|---|---|---|---|---|"]
for k, c in CELLS.items():
    ex = c.get("exploratory", {})
    acc = c["by_condition"]
    fr = ex.get("follow_rate", {})
    rob = ex.get("robustness_example_sets", {})
    a_sets = [L._get(c, "H3", "adjusted", "a")] + [L._get(rob.get(s), "H3", "adjusted", "a") for s in ("B", "C")]
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | {f3(L._get(acc, 'C4', 'accuracy'))} / {f3(L._get(acc, 'C6', 'accuracy'))} | "
              f"{f3(fr.get('C2'))} / {f3(fr.get('C4'))} / {f3(fr.get('C6'))} | {' / '.join(f2(x) for x in a_sets)} |")
md += ["", "### Accuracy by condition (alarm framing, example set A)", "",
       "| model | dataset | C0 | C2 | C4 | C6 | AUROC C0 |", "|---|---|---|---|---|---|---|"]
for k, c in CELLS.items():
    acc = c["by_condition"]
    tag, ds = k.split("|")
    md.append(f"| {tag} | {ds} | " + " | ".join(f3(L._get(acc, cc, 'accuracy')) for cc in L.CONDITIONS) +
              f" | {f3(L._get(acc, 'C0', 'auroc'))} |")
REPORT = "\n".join(md) + "\n"
print(REPORT)

# %% [markdown]
# ## Save

# %%
with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
    f.write(REPORT)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump({"checks": CHECK, "confirmatory": CONF, "cells": CELLS, "H4": H4,
               "sensitivity_imputation_D5": SENS_IMPUTED}, f, indent=1, default=str)
rows = []
for k, c in CELLS.items():
    tag, ds = k.split("|")
    a = c.get("H3") or {}
    rows.append({"model": tag, "dataset": ds, "passes_validity": c["validity"]["passes"],
                 "acc_C0": L._get(c, "by_condition", "C0", "accuracy"), "acc_C2": L._get(c, "by_condition", "C2", "accuracy"),
                 "acc_C4": L._get(c, "by_condition", "C4", "accuracy"), "H1_p": L._get(c, "H1", "p"), "H2_p": L._get(c, "H2", "p"),
                 "a": L._get(a, "adjusted", "a"), "a_ci_low": (L._get(a, "adjusted", "ci") or [None])[0],
                 "a_ci_high": (L._get(a, "adjusted", "ci") or [None, None])[1], "a_p": L._get(a, "adjusted", "p_boot"),
                 "a_prime": L._get(a, "reading_adjusted", "a"), "a_prime_p": L._get(a, "reading_adjusted", "p_boot"),
                 "h3_test_p": L._get(a, "h3_test", "p"),
                 "a_reversed": L._get(c, "Q3_reversed", "adjusted", "a"), "Q3": c["Q3_interpretation"],
                 "delta": L._get(a, "delta", "delta"), "logodds_unadjusted": L._get(a, "logodds_unadjusted", "value"),
                 "overtrust": L._get(c, "H5", "overtrust"), "H6_diff": L._get(c, "H6", "diff")})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "cells_summary.csv"), index=False)
ZIP = os.path.join(WORK, "analysis_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("report.md", "results.json", "cells_summary.csv"):
        z.write(os.path.join(OUT, name), name)
print("Saved:", ZIP, f"| total {(time.time() - T0) / 60:.1f} min")
