# %% [markdown]
# # Registered analysis, fourth study (`prereg-v4`)
#
# **What this notebook does.** It reads the `v4_<model>.zip` files of the fourth study, checks that every run saw the
# same patients and prompts and used the registered code, models, protocol and engineering-check decisions, and runs the
# analysis registered in
# [`PREREGISTRATION_V4.md`](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v4): R8 (W with
# True/False answers), R9 (W with the negated question), their joint reading, R10 (post-training and the chat
# template), R11 (example order), R12 (paraphrased prompt), R13 (direction words and numbers in one sentence) and the
# descriptive questions Q14, including the imputation bound of Deviation D5. If earlier runs are attached too (the
# `v3_*.zip`, `v2_*.zip` and `main_*.zip` files of the repository's `results/`), it compares their readings with this
# study's readings of the same prompts (Q14a); no decision depends on them.
#
# **How to run.** It needs no GPU and no internet. On Kaggle: upload the zip files as one dataset (Add Input → Upload
# → New Dataset) and Save & Run All; Kaggle unpacks zip files, and the first code cell accepts either the zip files or
# the unpacked folders. Locally or on Colab: put the zip files next to the notebook and run all cells. Runtime: about
# 30 to 60 minutes on a CPU. Download `analysis_v4_results.zip` at the end.

# %%
import os, sys, json, glob, gzip, zipfile, hashlib, time
import numpy as np
import pandas as pd

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SEARCH = [d for d in os.environ.get("LER_INPUT_DIRS", "").split(os.pathsep) if d] + \
         (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + [WORK]
IN_DIR = os.path.join(WORK, "analysis_v4_inputs")
OUT = os.path.join(WORK, "analysis_v4_results")
SKIP = [os.path.realpath(IN_DIR), os.path.realpath(OUT)]


def _skipped(p):
    rp = os.path.realpath(p)
    return any(rp == s or rp.startswith(s + os.sep) for s in SKIP)


def _csv_bytes(raw, name):
    return gzip.decompress(raw) if name.endswith(".gz") else raw


# A run arrives as a zip file or as an unpacked folder. Each source is reduced to the files the analysis reads; the
# study of a run comes from its run log ('v4', 'v3', 'v2' or the first study).
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
    for n in ("ler_v3.py", "ler_v4.py"):
        if read(n) is not None:
            out[n] = read(n)
    gen = next((n for n in (f"generated_{tag}.csv.gz", f"generated_{tag}.csv") if read(n) is not None), None)
    if gen:
        out[f"generated_{tag}.csv"] = _csv_bytes(read(gen), gen)
    return out


for root in SEARCH:
    for p in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        if _skipped(p) or not os.path.basename(p).startswith(("v4_", "v3_", "v2_", "main_")):
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
if not any(s == "v4" for s, _ in SOURCES):
    raise RuntimeError("No fourth-study run found. Attach the v4_<model>.zip files (Kaggle: Add Input -> Upload).")
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
# Every fourth-study run used the registered code (`ler_main.py` main-2.0 and `ler_v3.py` v3-1.1 unchanged, one version
# of `ler_v4.py`), the same protocol text and engineering-check decisions, the registered model revision, the same
# patients, cards and solved examples, and every prompt that two runs share has the same text. Each run is complete.

# %%
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


V4RUNS = sorted(t for s, t in SOURCES if s == "v4")
first = V4RUNS[0]
code = {tag: tuple(sha(os.path.join(IN_DIR, "v4", tag, n)) for n in ("ler_main.py", "ler_v3.py", "ler_v4.py"))
        for tag in V4RUNS}
if len(set(code.values())) != 1:
    raise RuntimeError(f"The fourth-study runs used different code: {code}")
sys.path.insert(0, os.path.join(IN_DIR, "v4", first))
for _m in ("ler_v4", "ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_v3 as V
import ler_v4 as W

RUNS, RECS, GENS = {}, {}, {}
for (study, tag) in SOURCES:
    d = os.path.join(IN_DIR, study, tag)
    RUNS[(study, tag)] = json.load(open(os.path.join(d, f"run_log_{tag}.json")))
    RECS[(study, tag)] = pd.read_csv(os.path.join(d, f"records_{tag}.csv"), dtype={"id": str}, low_memory=False)
    if os.path.exists(os.path.join(d, f"generated_{tag}.csv")):
        GENS[(study, tag)] = pd.read_csv(os.path.join(d, f"generated_{tag}.csv"), low_memory=False)
MOCK = any(RUNS[("v4", t)].get("mock") for t in V4RUNS)
problems = []
if code[first][0] != W.MAIN_SHA256 or L.CODE_VERSION != "main-2.0":
    problems.append("ler_main.py is not the registered main-2.0")
if code[first][1] != W.V3_SHA256 or V.V3_VERSION != "v3-1.1":
    problems.append("ler_v3.py is not the registered v3-1.1")
CHECK = {"code_sha256": dict(zip(("ler_main.py", "ler_v3.py", "ler_v4.py"), code[first])), "v4_version": W.V4_VERSION,
         "mock": MOCK}
psha = {t: RUNS[("v4", t)]["protocol"]["sha256"] for t in V4RUNS}
if len(set(psha.values())) != 1:
    problems.append(f"the runs checked different protocol texts: {psha}")
CHECK["protocol_sha256"] = psha[first]
DECISIONS = W.ENGCHECK_DECISIONS
dec_runs = {t: json.dumps(RUNS[("v4", t)].get("decisions"), sort_keys=True) for t in V4RUNS}
if len(set(dec_runs.values())) != 1:
    problems.append(f"the runs used different engineering-check decisions: {dec_runs}")
if DECISIONS is None and MOCK:
    DECISIONS = RUNS[("v4", first)]["decisions"]
if DECISIONS is None:
    problems.append("the engineering-check decisions are not recorded in ler_v4.py")
elif json.dumps(DECISIONS, sort_keys=True) != dec_runs[first]:
    problems.append("the runs' decisions differ from those registered in ler_v4.py")
CHECK["decisions"] = DECISIONS
FAMILIES = list((DECISIONS or {}).get("r10_families", []))
EXPECTED = W.INSTRUCT_TAGS + [W.family_tags(f)[0] for f in FAMILIES]
for tag in V4RUNS:
    r = RUNS[("v4", tag)]
    if tag not in W.TAG2ID:
        problems.append(f"{tag} is not a registered model of the fourth study")
        continue
    mid = W.TAG2ID[tag]
    if not MOCK and r.get("model_revision") != W.MODELS_V4[mid]["revision"]:
        problems.append(f"{tag}: revision {r.get('model_revision')} is not the registered {W.MODELS_V4[mid]['revision']}")
    if r.get("mock") != MOCK:
        problems.append("mock and real runs are mixed")
fps = {t: json.dumps(RUNS[("v4", t)]["fingerprints"], sort_keys=True) for t in V4RUNS}
if len(set(fps.values())) != 1:
    problems.append("the runs used different patients, cards or solved examples (fingerprints differ)")
if not MOCK:
    for ds, exp in L.DEV_RUN["fingerprints"].items():
        got = RUNS[("v4", first)]["fingerprints"][ds]
        if any(got[k] != v for k, v in exp.items()):
            problems.append(f"{ds}: fingerprints differ from the dev phase")
for st, tag in SOURCES:          # earlier runs: same test patients and cards as this study
    if st != "v4":
        f_old, f_new = RUNS[(st, tag)]["fingerprints"], RUNS[("v4", first)]["fingerprints"]
        if any(f_old[ds].get(k) != f_new[ds].get(k) for ds in L.DATASETS for k in ("test_ids", "test_cards")):
            problems.append(f"earlier run {st} {tag} used other test patients or cards")
allp = pd.concat([RECS[("v4", t)][["key", "text_sha"]] for t in V4RUNS]).drop_duplicates()
bad_keys = allp["key"][allp["key"].duplicated()].tolist()
if bad_keys:
    problems.append(f"{len(bad_keys)} prompts have different texts in different runs, e.g. {bad_keys[:3]}")
COMPLETE = {}
for tag in V4RUNS:
    df = RECS[("v4", tag)]
    grids = tuple(RUNS[("v4", tag)]["grids"])
    if DECISIONS is not None and tag in W.TAG2ID and grids != W.model_grids(W.TAG2ID[tag], DECISIONS):
        problems.append(f"{tag}: grids {grids} are not the registered {W.model_grids(W.TAG2ID[tag], DECISIONS)}")
    n_pat = df.groupby("dataset")["id"].nunique().to_dict()
    want = sum(n_pat.values()) * len(W.grid_v4(grids))
    COMPLETE[tag] = {"records": int(len(df)), "expected": int(want), "patients": n_pat, "complete": len(df) == want}
    if len(df) != want:
        problems.append(f"{tag}: {len(df)} records, expected {want}")
CHECK.update(runs=V4RUNS, earlier_runs=sorted(f"{s} {t}" for s, t in SOURCES if s != "v4"), complete=COMPLETE,
             missing_registered_runs=[t for t in EXPECTED if t not in V4RUNS])
if problems:
    raise RuntimeError("The runs are not comparable:\n" + "\n".join(problems))
N_BOOT = int(os.environ.get("LER_TEST_BOOT", W.N_BOOT))
N_BOOT_BOUND = int(os.environ.get("LER_TEST_BOOT", W.N_BOOT_BOUND))
if (N_BOOT != W.N_BOOT or N_BOOT_BOUND != W.N_BOOT_BOUND) and not MOCK:
    raise RuntimeError("LER_TEST_BOOT is for offline tests only")
CHECK["n_boot"], CHECK["n_boot_bound"] = N_BOOT, N_BOOT_BOUND
print(json.dumps(CHECK, indent=1, default=str))
print("All checks passed." + (" (MOCK RUNS: OFFLINE TEST ONLY)" if MOCK else ""))

# %% [markdown]
# ## Every cell and every R10 family
# Validity, the registered pair, each new pair against the registered pair, the cue and the net-number card, the static
# leans and descriptives (paired bootstrap over patients, 10,000 resamples, seed 2026); for each R10 family and dataset,
# the cells B·plain, B·chat, I·chat (and I·plain if run) with the contrasts P and F.

# %%
t0 = time.time()
INST = [t for t in W.INSTRUCT_TAGS if t in V4RUNS]


def analyse_all(recs, n_boot):
    cells, fam = {}, {}
    for tag in INST:
        df = recs[tag]
        for ds in L.DATASETS:
            cells[f"{tag}|{ds}"] = W.analyse_cell_v4(df[df["dataset"] == ds], n_boot=n_boot)
        print(f"  {tag}: done ({time.time() - t0:.0f} s)", flush=True)
    for f in FAMILIES:
        bt, it = W.family_tags(f)
        if bt in recs and it in recs:
            for ds in W.R10_ELIGIBLE[f]:
                db, di = recs[bt], recs[it]
                fam[f"{f}|{ds}"] = W.analyse_2x2(db[db["dataset"] == ds], di[di["dataset"] == ds], n_boot)
        print(f"  family {f}: done ({time.time() - t0:.0f} s)", flush=True)
    return cells, fam


R4RECS = {t: RECS[("v4", t)] for t in V4RUNS}
CELLS, FAM = analyse_all(R4RECS, N_BOOT)
CONF = W.confirmatory_v4(CELLS, FAM, FAMILIES, W.INSTRUCT_TAGS)   # a missing model enters R13 with p = 1
LEAN = W.lean_vs_w(CELLS)
GREEDY = {t: W.greedy_summary(GENS[("v4", t)], RECS[("v4", t)]) for t in INST if ("v4", t) in GENS}
REPLAY = {}
for tag in V4RUNS:
    for st in ("v3", "v2", "v1"):
        if (st, tag) not in RECS:
            continue
        old = RECS[(st, tag)]
        r = W.replay_v4(RECS[("v4", tag)], old, ("A",))
        if r["n_shared_prompts"]:
            REPLAY[f"{tag} vs {st}, set A"] = r
        r = W.replay_v4(RECS[("v4", tag)], old, ("Asf",), ("alarm",))
        if r["n_shared_prompts"]:
            REPLAY[f"{tag} vs {st}, Asf alarm"] = r
print(f"analysis: {time.time() - t0:.0f} s")

# %% [markdown]
# ## The imputation bound of Deviation D5 (Q14f)
# For every run with imputed readings: in each such record, the probability of the less likely answer word is halved,
# and every registered result is computed again (2,000 resamples). It decides nothing.

# %%
t0 = time.time()
IMPUTED = {t: float((R4RECS[t]["n_imputed"] > 0).mean()) for t in V4RUNS}
BOUND = {"imputed_share": IMPUTED}
if any(v > 0 for v in IMPUTED.values()):
    BREC = {t: (W.halve_less_likely_v4(d) if IMPUTED[t] > 0 else d) for t, d in R4RECS.items()}
    BCELLS, BFAM = analyse_all(BREC, N_BOOT_BOUND)
    BCONF = W.confirmatory_v4(BCELLS, BFAM, FAMILIES, W.INSTRUCT_TAGS)
    BOUND["confirmatory"] = BCONF
    chg = []
    for name in ("R8", "R9", "R11", "R12"):
        for part in ("W", "O"):
            for k, e in CONF[name][part]["cells"].items():
                be = BCONF[name][part]["cells"][k]
                if e["supported"] != be["supported"] or e["reduced"] != be["reduced"]:
                    chg.append(f"{name} {part} {k}: supported {e['supported']} -> {be['supported']}, "
                               f"reduced {e['reduced']} -> {be['reduced']}")
    BOUND["changed_cells"] = chg
print("imputed share per run:", {k: round(v, 3) for k, v in IMPUTED.items()}, f"| {time.time() - t0:.0f} s")

# %% [markdown]
# ## Results

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


yn = lambda b: "yes" if b else "no"
REPORT = [f"# Fourth study: results (`prereg-v4`)\n", f"Runs: {', '.join(V4RUNS)}. Missing registered runs: "
          f"{CHECK['missing_registered_runs'] or 'none'}. R10 families: {', '.join(FAMILIES) or 'none'}. Bootstrap "
          f"resamples: {N_BOOT}." + (" **MOCK RUNS: OFFLINE TEST ONLY.**" if MOCK else "") + "\n"]

rows = []
for k, c in CELLS.items():
    v = c["validity"]
    rows.append([k, f"{v['split']['alarm_median_mass']:.3f}", f"{v['split']['reversed_median_mass']:.3f}",
                 f"{v['split']['c0_accuracy']:.3f}", yn(v["split"]["passes"])] +
                [yn(L._get(v, "pairs", p, "passes")) if p in v["pairs"] else "–" for p in W.TESTED_PAIRS + ("aggregate", "plain")])
REPORT += ["## Validity\n", md_table(["cell", "mass alarm", "mass reversed", "C0 acc.", "valid"] +
                                     [f"{p} valid" for p in W.TESTED_PAIRS + ("aggregate", "plain")], rows), ""]
rows = []
for k, c in CELLS.items():
    for f_, x in c["validity"]["framings"].items():
        rows.append([k, f_, f"{x.get('median_mass', x.get('alarm_median_mass', float('nan'))):.3f}",
                     f2(x.get("movement")), f2(x.get("c0_spearman_with_alarm")), yn(x["passes"])])
REPORT += ["**Validity of each new framing** (mass, movement, C0 Spearman correlation with the registered alarm question):\n",
           md_table(["cell", "framing / set", "mass", "movement", "C0 Spearman", "valid"], rows), ""]

rows = []
for k, c in CELLS.items():
    s = L._get(c, "registered", "pairs", "registered") or {}
    rows.append([k, yn(c["validity"]["split"]["passes"]), *part_row(s.get("O")), *part_row(s.get("W")),
                 f2(L._get(s, "movement", "est"))])
REPORT += ["## O and W with the registered pair\n",
           "O = (s_first + s_second) / 2 and W = (s_first − s_second) / 2, for s = a and s = a′ (95% intervals); W < 0 "
           "is a pull toward the second answer word. p is the intersection–union p (no correction).\n",
           md_table(["cell", "valid", "O (a)", "O (a′)", "p", "W (a)", "W (a′)", "p", "movement"], rows), ""]


def kept_section(title, res, part):
    rows = [[k, "+" if e["sign_registered"] > 0 else "−", yn(e["included"]), f2(e["a"]), f2(e["a_prime"]),
             pf(e["p_used"]), pf(e["p_holm"]), yn(e["supported"]), f2(e["change_a"]), f2(e["change_a_prime"]),
             pf(e["change_p_holm"]), yn(e["reduced"]), yn(e["opposite"])] for k, e in res["cells"].items()]
    return [f"**{title}, {part}.** Supported in {res['n_supported']} of {res['n_eligible']} eligible cells, reduced in "
            f"{res['n_reduced']}, opposite in {res['n_opposite']}: **{res['result']}**.\n",
            md_table(["cell", "sign", "valid", part + " (a)", part + " (a′)", "p", "Holm p", "supported", "change (a)",
                      "change (a′)", "Holm p", "reduced", "opposite"], rows), ""]


for name, title in (("R8", "R8. True/False answers"), ("R9", "R9. The negated question"),
                    ("R11", "R11. Alarming-last example order"), ("R12", "R12. Paraphrased prompt")):
    REPORT += [f"## {title}\n"]
    for part in ("W", "O"):
        REPORT += kept_section(title.split(". ")[1], CONF[name][part], part)
REPORT += ["## Joint reading of R8 and R9\n", f"W with True/False: {CONF['R8']['W']['result']}; W with the negated "
           f"question: {CONF['R9']['W']['result']}. **Reading: {CONF['joint_reading']}.**\n"]
rows = []
for k, c in CELLS.items():
    d = L._get(c, "pairs", "neg", "pairs", "wording", "W")
    if d:
        rows.append([k, *part_row(d)])
REPORT += ["**Wording contrast** D = (s_reversed − s_negated) / 2 (both framings have “Yes” for the safe outcome):\n",
           md_table(["cell", "D (a)", "D (a′)", "p"], rows), ""]

R10 = CONF["R10"]
rows = []
for k in R10["P"]:
    c = FAM.get(k) or {}
    pr = c.get("primary") or {}
    rows.append([k] + [part_row(L._get(pr, "pairs", x, "W"))[1] for x in ("B_plain", "B_chat", "I_chat")] +
                [f"{f2(R10['P'][k]['a'])} / {f2(R10['P'][k]['a_prime'])}", pf(R10["P"][k]["p_holm"]),
                 yn(R10["P"][k]["supported"]), f"{f2(R10['F'][k]['a'])} / {f2(R10['F'][k]['a_prime'])}",
                 pf(R10["F"][k]["p_holm"]), yn(R10["F"][k]["supported"])])
REPORT += ["## R10. Post-training and the chat template\n",
           md_table(["cell", "W B·plain (a′)", "W B·chat (a′)", "W I·chat (a′)", "P (a / a′)", "Holm p", "P supported",
                     "F (a / a′)", "Holm p", "F in direction of W"], rows), "",
           f"P supported in {R10['n_P_supported']} of {R10['n_eligible']} eligible cells; F in the direction of W in "
           f"{R10['n_F_in_direction']}. **R10: {'supported' if R10['supported'] else 'not supported'}; "
           f"{R10['interpretation']}.**\n"]
rows = []
for k, c in FAM.items():
    sec = c.get("secondary")
    if sec:
        rows.append([k, yn(L._get(c, "validity", "I_plain", "passes")), part_row(L._get(sec, "pairs", "I_plain", "W"))[1],
                     *part_row(L._get(sec, "contrasts", "P2")),
                     *part_row(L._get(sec, "contrasts", "F2"))])
REPORT += ["**Q14g.** With I·plain: W(I·plain), P′ = W(I·plain) − W(B·plain) and F′ = W(I·chat) − W(I·plain):\n",
           md_table(["cell", "I·plain valid", "W I·plain (a′)", "P′ (a)", "P′ (a′)", "p", "F′ (a)", "F′ (a′)", "p"], rows)
           if rows else "No I·plain cell was run.", ""]
R13 = CONF["R13"]
rows = [[k, yn(e["included"]), f2(e["m_full"]), f2(e["m_cue"]), f2(e["m_net"]), f2(e["dm"]), pf(e["p_used"]),
         pf(e["p_holm"]), yn(e["supported"])] for k, e in R13["cells"].items()]
REPORT += ["## R13. Direction words and numbers in one sentence\n",
           md_table(["cell", "valid", "movement, card", "cue", "net number", "Δm", "p", "Holm p", "supported"], rows), "",
           f"**R13: {'supported' if R13['supported'] else 'not supported'}** ({R13['n_supported']} of {R13['n_valid']} "
           "valid cells).\n"]

REPORT += ["## Q14. Descriptive\n", "**Q14c. Static answer lean and W:**\n",
           md_table(["pair", "valid cells", "conclusive W", "sign of W = sign of lean", "Spearman (lean, W a′)"],
                    [[p, v["n_valid"], v["n_conclusive"], v["n_sign_agrees"], f2(v["spearman_lean_W_a_prime"])]
                     for p, v in LEAN.items()]), ""]
rows = []
for t, g in GREEDY.items():
    for mode, r in g.items():
        rows.append([t, mode, r["n"], f"{r['parsed_share']:.3f}", f"{r['agreement']:.3f}"] +
                    [f"{f2(r['by_dataset'][ds]['E_W_generated'])} / {f2(r['by_dataset'][ds]['E_W_first_token'])}"
                     for ds in L.DATASETS])
REPORT += ["**Q14d. Generated answers** (agreement with the first-token reading among parsed answers; E_W from the "
           "generated answers / from the first-token reading, same prompts):\n",
           md_table(["model", "mode", "answers", "parsed", "agreement"] + [f"E_W {ds}" for ds in L.DATASETS], rows), ""]
if REPLAY:
    REPORT += ["**Q14a. Replay** of prompts shared with earlier runs:\n",
               md_table(["comparison", "prompts", "text differs", "max |Δ|", "median |Δ|", "same answer"],
                        [[k, r["n_compared"], r["n_text_differs"], f"{r['max_abs_diff']:.4f}", f"{r['median_abs_diff']:.4f}",
                          f"{r['share_same_answer']:.3f}"] for k, r in REPLAY.items()]), ""]
else:
    REPORT += ["**Q14a.** The earlier runs were not attached; the replay comparison was not made.\n"]
REPORT += ["**Q14f. Imputation bound.** Share of records with an imputed variant: " +
           ", ".join(f"{k} {v:.3f}" for k, v in IMPUTED.items()) + "."]
if "confirmatory" in BOUND:
    B = BOUND["confirmatory"]
    REPORT += [f"Under the bound: R8 W {B['R8']['W']['result']}, R9 W {B['R9']['W']['result']} (joint reading: "
               f"{B['joint_reading']}); R10 {B['R10']['interpretation']}; R11 W {B['R11']['W']['result']}, O "
               f"{B['R11']['O']['result']}; R12 W {B['R12']['W']['result']}, O {B['R12']['O']['result']}; R13 "
               f"{'supported' if B['R13']['supported'] else 'not supported'}. Cells whose support or reduction changes: "
               + ("; ".join(BOUND["changed_cells"]) or "none") + ".\n"]
_res = lambda r: r["result"]
REPORT[2:2] = ["## Decisions\n",
               f"- **R8** (W with True/False): {_res(CONF['R8']['W'])} (prediction: kept); O {_res(CONF['R8']['O'])}.",
               f"- **R9** (W with the negated question): {_res(CONF['R9']['W'])} (prediction: kept); O {_res(CONF['R9']['O'])}.",
               f"- **Joint reading:** {CONF['joint_reading']}.",
               f"- **R10** (post-training with the format fixed): {'supported' if R10['supported'] else 'not supported'}; "
               f"{R10['interpretation']}.",
               f"- **R11** (example order): W {_res(CONF['R11']['W'])}, O {_res(CONF['R11']['O'])}; {CONF['R11']['O_reading']}.",
               f"- **R12** (paraphrase): W {_res(CONF['R12']['W'])}, O {_res(CONF['R12']['O'])}.",
               f"- **R13** (words over numbers in one sentence): {'supported' if R13['supported'] else 'not supported'}.", ""]
REPORT_MD = "\n".join(REPORT)
print(REPORT_MD)

# %% [markdown]
# ## Save and zip

# %%
RESULTS = {"check": CHECK, "cells": CELLS, "families": FAM, "confirmatory": CONF, "lean": LEAN, "generated": GREEDY,
           "replay": REPLAY, "bound": BOUND, "minutes": round((time.time() - T0) / 60, 1)}
with open(os.path.join(OUT, "results_v4.json"), "w") as f:
    json.dump(RESULTS, f, indent=1, default=float)
with open(os.path.join(OUT, "report_v4.md"), "w", encoding="utf-8") as f:
    f.write(REPORT_MD + "\n")
ZIP = os.path.join(WORK, "analysis_v4_results.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("results_v4.json", "report_v4.md"):
        z.write(os.path.join(OUT, name), name)
    z.write(os.path.join(IN_DIR, "v4", first, "ler_v4.py"), "ler_v4.py")
print("Saved:", ZIP, f"| {(time.time() - T0) / 60:.1f} min")
