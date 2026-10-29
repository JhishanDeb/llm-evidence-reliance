# %% [markdown]
# # Engineering check of the models for the third study (before registration)
#
# **What this notebook does.** Before `PREREGISTRATION_V3.md` is tagged, it checks that each candidate model can be
# run as the third study will register it: download, loading in float16 with vLLM on Kaggle's two T4 GPUs (with the
# model's window), the chat template and its default system text, the single-token answer words (Yes/No and A/B), the
# exact-reading check (Deviation D2), finite readings, the answer-word mass with every card format and with the answer
# options, the share of imputed readings under the top-1,000 reading (Deviation D5), prompt lengths and the time per
# prompt. It also asks four questions with an obvious answer, to see that the answer words are read the right way
# round. The three earlier models are checked with the new card formats too; for them the check decides nothing.
#
# **What it does not do.** It computes no accuracy, no answer direction and no asymmetry. It uses dev-sample patients
# only (never test patients). Its output decides which new models are registered, so it stays purely technical.
#
# **How to run on Kaggle**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**.
# 3. Add-ons → Secrets: `HF_TOKEN` (the same token as before). For `CohereLabs/aya-expanse-8b` (and Llama), accept the
#    terms on Hugging Face first; if you do not, that one model is reported as not downloadable and the rest run.
# 4. **Save Version → Save & Run All (Commit).** Expected time: about 2 to 2.5 hours (nine models, one after another).
# 5. Download `engcheck_v3.zip` from the Output tab and send it to me.

# %%
# ============ settings ============
SKIP_MODELS = []            # model ids to leave out, e.g. ["CohereLabs/aya-expanse-8b"]
RUN_MEMCHECK = False        # not part of the engineering check
BACKEND = "vllm"

import os, sys, time, json, subprocess, platform, shutil
T_START = time.time()
MOCK = os.environ.get("LER_MOCK") == "1"            # offline test of the notebook; never set on Kaggle
LOCAL_MODEL = None


def sh(cmd, timeout=None, cwd=None, env=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout,
                       cwd=cwd, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


_probe = ("import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
          "'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
code, out = sh([sys.executable, "-c", _probe])
ENV = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    ENV.update(json.loads(out.strip().splitlines()[-1]))
except Exception:
    ENV["torch_probe_error"] = out[-500:]
ENV["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")[1].strip()
print("Python :", ENV["python"], "| torch:", ENV.get("torch"), "| GPUs:", ENV.get("gpus"))
if not MOCK and not ENV.get("gpus"):
    raise RuntimeError("No GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again.")

# %% [markdown]
# ## Paths and access token

# %%
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC_DIR = os.path.abspath(".")
DATA_DIR = os.environ.get("LER_DATA_DIR", "/tmp/ler_data" if os.path.isdir("/kaggle/working") else
                          os.path.join(WORK, "ler_data"))
EXTRA_ROOTS = (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + \
              [d for d in os.environ.get("LER_EXTRA_ROOTS", "").split(os.pathsep) if d]
CACHE = "/tmp/hf_cache"
OUT_DIR = os.path.join(WORK, "engcheck_v3")
for _d in (DATA_DIR, CACHE, OUT_DIR):
    os.makedirs(_d, exist_ok=True)
os.environ["HF_HOME"] = CACHE
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
print("HF_TOKEN available:", bool(HF_TOKEN))
TIMINGS = {}

# %% [markdown]
# ## Install vLLM, the pinned xgboost and tabmemcheck
# vLLM is matched to Kaggle's preinstalled torch (the procedure that worked in the dev phase). xgboost is pinned
# to the dev-phase version so that the features and cards are rebuilt exactly; the notebook checks this with the
# dev fingerprints below. tabmemcheck runs the memorization check.

# %%
t0 = time.time()
BACKEND_NOTE = {"requested": BACKEND}


def _pkg_versions(names):
    code = "import importlib.metadata as m, json\nout={}\n" + "".join(
        f"\ntry: out['{n}']=m.version('{n}')\nexcept Exception: pass" for n in names) + "\nprint(json.dumps(out))"
    return json.loads(sh([sys.executable, "-c", code])[1].strip().splitlines()[-1])


def _wheel_exists(url):
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        return True
    except Exception:
        return False


VLLM_PIN = "0.26.0"     # the dev-phase version; the same in every run of all three studies


def install_vllm():
    import re
    have = _pkg_versions(["vllm"])
    if "vllm" in have and have["vllm"].split("+")[0] == VLLM_PIN:
        return True, f"vllm {have['vllm']} already installed"
    pins = _pkg_versions(["torch"])
    if "torch" not in pins:
        return False, "torch not installed"
    torch_pub, _, torch_cuda = pins["torch"].partition("+")
    with open("/tmp/vllm_constraints.txt", "w") as f:
        f.write(f"torch=={torch_pub}\n")
    base = [sys.executable, "-m", "pip", "install", "--prefer-binary", "-c", "/tmp/vllm_constraints.txt"]
    code, out = sh(base + ["--only-binary", "vllm", "--dry-run", f"vllm=={VLLM_PIN}"], timeout=1200)
    BACKEND_NOTE["vllm_dry_run_tail"] = out[-1500:]
    m = re.search(r"Would install .*?\bvllm-(\d+\.\d+\.\d+(?:\.post\d+)?)\b", out, re.S)
    if code != 0 or not m:
        return False, f"vLLM {VLLM_PIN} has no build for this torch version (dry run failed)"
    ver = m.group(1)
    target = [f"vllm=={VLLM_PIN}"]
    if torch_cuda.startswith("cu12"):
        tags = []
        for t in (torch_cuda, "cu129", "cu128", "cu126"):
            if t not in tags:
                tags.append(t)
        for t in tags:
            url = (f"https://github.com/vllm-project/vllm/releases/download/v{ver}/"
                   f"vllm-{ver}+{t}-cp38-abi3-manylinux_2_28_x86_64.whl")
            if _wheel_exists(url):
                target = [url]
                break
    BACKEND_NOTE["vllm_wheel"] = target[0]
    code, out = sh(base + ["-q"] + target, timeout=2400)
    BACKEND_NOTE["vllm_install_tail"] = out[-1500:]
    after = _pkg_versions(["vllm", "torch"])
    if code != 0 or "vllm" not in after:
        return False, "pip install vllm failed"
    if after.get("torch") != pins.get("torch"):
        return False, f"vLLM install changed torch {pins.get('torch')} -> {after.get('torch')}"
    if "torchcodec" in _pkg_versions(["torchcodec"]) and sh([sys.executable, "-c", "import torchcodec"])[0] != 0:
        sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchcodec"])
        BACKEND_NOTE["removed_broken_torchcodec"] = True
    code, out = sh([sys.executable, "-c", "from vllm import LLM, SamplingParams"], timeout=900)
    if code != 0:
        BACKEND_NOTE["vllm_import_error"] = out[-3000:]
        return False, "vLLM installed but cannot be imported: " + out.strip().splitlines()[-1][:300]
    return True, f"installed vllm {after['vllm']} ({target[0]}) for torch {after['torch']}"


USE_VLLM = False
NUMPY_BEFORE = _pkg_versions(["numpy"]).get("numpy")
if MOCK or LOCAL_MODEL:
    BACKEND_NOTE["install"] = "offline test: vLLM not installed"
elif BACKEND in ("auto", "vllm"):
    try:
        USE_VLLM, msg = install_vllm()
    except Exception as e:
        USE_VLLM, msg = False, f"vLLM install error: {e!r}"
    BACKEND_NOTE["install"] = msg
    print(msg)
    if not USE_VLLM and BACKEND == "vllm":
        raise RuntimeError("BACKEND='vllm' requested but vLLM is unavailable: " + msg)
# vLLM can bring its own numpy; make sure pandas / scikit-learn / xgboost still import, else restore numpy
NUMPY_AFTER = _pkg_versions(["numpy"]).get("numpy")
BACKEND_NOTE["numpy"] = {"before": NUMPY_BEFORE, "after": NUMPY_AFTER}
if NUMPY_AFTER != NUMPY_BEFORE:
    _ok = sh([sys.executable, "-c", "import numpy, pandas, sklearn, scipy, xgboost"])[0] == 0
    BACKEND_NOTE["numpy_changed_imports_ok"] = _ok
    if not _ok:
        print(f"vLLM changed numpy {NUMPY_BEFORE} -> {NUMPY_AFTER} and broke other packages: restoring numpy, "
              "using transformers instead of vLLM")
        sh([sys.executable, "-m", "pip", "install", "-q", f"numpy=={NUMPY_BEFORE}"])
        USE_VLLM = False
        BACKEND_NOTE["install"] += " | numpy restored, vLLM not used"

# pinned xgboost (dev-phase version) and tabmemcheck, without touching torch / numpy / pandas
XGB_PIN = "3.4.1"
pins = _pkg_versions(["torch", "numpy", "pandas", "xgboost", "tabmemcheck"])
with open("/tmp/ler_constraints.txt", "w") as f:
    for k in ("torch", "numpy", "pandas"):
        if k in pins:
            f.write(f"{k}=={pins[k].split('+')[0]}\n")
want = []
if pins.get("xgboost") != XGB_PIN and not (MOCK or LOCAL_MODEL):
    want.append(f"xgboost=={XGB_PIN}")
if RUN_MEMCHECK and pins.get("tabmemcheck") != "0.1.6":
    want.append("tabmemcheck==0.1.6")
if want:
    code, out = sh([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "-c", "/tmp/ler_constraints.txt",
                    *want], timeout=1800)
    BACKEND_NOTE["extra_install"] = {"packages": want, "ok": code == 0, "tail": out[-1500:]}
    print("installed:", want, "| ok:", code == 0)
    if code != 0:
        print(out[-1500:])
after = _pkg_versions(["xgboost", "tabmemcheck"])
if not (MOCK or LOCAL_MODEL) and after.get("xgboost") != XGB_PIN:
    raise RuntimeError(f"xgboost {XGB_PIN} (the dev-phase version) could not be installed; found "
                       f"{after.get('xgboost')}. The features and cards could differ from the dev phase.")
if RUN_MEMCHECK and sh([sys.executable, "-c", "import tabmemcheck"])[0] != 0:
    raise RuntimeError("tabmemcheck could not be installed or imported. The memorization check is registered "
                       "(Section 3), so the run stops here. See BACKEND_NOTE['extra_install'].")
TIMINGS["install"] = time.time() - t0
print({k: v for k, v in BACKEND_NOTE.items() if not k.endswith("_tail") and k != "extra_install"})

# %% [markdown]
# ## Shared code
# `ler_main.py` (main-2.0, unchanged) and `ler_v3.py` (card formats, models, model window, engineering check).

# %% [writefile ler_main.py]

# %% [writefile ler_v3.py]

# %% [markdown]
# ## Data, splits, XGBoost and the dev sample, checked against the dev phase

# %%
import importlib, hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
for _m in ("ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_v3 as V
CODE_SHA = {n: L.file_sha256(os.path.join(SRC_DIR, n)) for n in ("ler_main.py", "ler_v3.py")}
if CODE_SHA["ler_main.py"] != V.MAIN_SHA256:
    raise RuntimeError("ler_main.py is not the registered main-2.0")
print("code: ler_main", L.CODE_VERSION, CODE_SHA["ler_main.py"][:16], "| ler_v3", V.V3_VERSION, CODE_SHA["ler_v3.py"][:16])
try:
    _ids_txt = (open(os.environ["LER_PILOT_IDS_PATH"]).read() if os.environ.get("LER_PILOT_IDS_PATH")
                else L.http_get(L.PILOT_IDS_URL).decode("utf-8"))
    PILOT_IDS = [ln.strip() for ln in _ids_txt.splitlines() if ln.strip() and not ln.startswith("#")]
except Exception:
    PILOT_IDS = [str(i) for i in L.PILOT_IDS_FALLBACK]
t0 = time.time()
BUNDLES, FP_CHECK = {}, {}
for ds in L.DATASETS:
    print(f"[{ds}]")
    if ds == "physionet2019":
        df_, feats_, cats_, info_ = L.load_physionet2019(DATA_DIR, EXTRA_ROOTS, log=print,
                                                          workers=max(1, min(4, os.cpu_count() or 1)))
    else:
        df_, feats_, cats_, info_ = L.LOADERS[ds](DATA_DIR, EXTRA_ROOTS, log=print)
    b = L.Bundle(ds, df_, feats_, cats_, info_, pilot_ids=PILOT_IDS if ds == "physionet2012" else (), sample="dev")
    BUNDLES[ds] = b
    FP_CHECK[ds] = {k: b.info["fingerprint"][k] == v for k, v in L.DEV_RUN["fingerprints"][ds].items()}
    print("   dev fingerprints match:", FP_CHECK[ds])
TIMINGS["data"] = time.time() - t0
if not MOCK:
    bad = {ds: [k for k, ok in v.items() if not ok] for ds, v in FP_CHECK.items() if not all(v.values())}
    if bad:
        raise RuntimeError(f"The rebuilt data differ from the dev phase: {bad}. Send me this output.")
print("All dev fingerprints match.")

# %% [markdown]
# ## Dev prompts for the check
# Check set, per dataset: 6 dev patients × C0, C2, C4 × both framings with the registered card (36); the same 6
# patients × C2, C4 × both framings with each reduced card (72) and × C2, C4 in the reversed framing with the matched
# card (12); 3 dev patients × C0, C2, C4 × both option orders (18). Timing set: every dev patient, C2, alarm framing,
# registered card (100 per dataset), and 50 per dataset × C2 × both option orders (100).

# %%
ITEMS = []
for ds, b in BUNDLES.items():
    for rid in b.ids[:6]:
        for cond, fr, shot in V.grid_v3(("split", "formats", "matched")):
            fmt = V.SET_FMT[shot]
            ITEMS.append({"set": "check", "fmt": V.fmt_label(fr, shot), "framing": fr,
                          "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": V.prompt_v3(b, rid, cond, fr, fmt)[0]})
    for rid in b.ids[:3]:
        for cond in ("C0", "C2", "C4"):
            for fr in L.CHOICE_FRAMINGS:
                ITEMS.append({"set": "check", "fmt": "choice", "framing": fr, "key": f"{ds}|{rid}|{cond}|{fr}|A",
                              "text": b.prompt(rid, cond, fr, "A")[0]})
    for rid in b.ids:
        ITEMS.append({"set": "timing", "fmt": "yesno_full", "framing": "alarm", "key": f"{ds}|{rid}|C2|alarm|A",
                      "text": b.prompt(rid, "C2", "alarm", "A")[0]})
    for rid in b.ids[:50]:
        for fr in L.CHOICE_FRAMINGS:
            ITEMS.append({"set": "timing", "fmt": "choice", "framing": fr, "key": f"{ds}|{rid}|C2|{fr}|A",
                          "text": b.prompt(rid, "C2", fr, "A")[0]})
PROMPTS_PATH = os.path.join(OUT_DIR, "dev_prompts.json")
json.dump(ITEMS, open(PROMPTS_PATH, "w"))
print(pd.Series([f"{i['set']} {i['fmt']}" for i in ITEMS]).value_counts().sort_index())
for _f in ("yesno_words", "yesno_numbers", "yesno_cue", "yesno_matched"):
    _it = [i for i in ITEMS if i["fmt"] == _f and "|C2|" in i["key"]][0]
    print(f"\n{_f} (dev patient, {_it['key']}):\n" + _it["text"].split(L.PATIENT_HEADER)[1].split("\n\n")[1])

# %% [markdown]
# ## Check each model (one separate process per model, so the GPU memory is freed in between)
# New-family candidates in the registered order, then the three earlier models (with their pinned revisions).

# %%
CANDIDATES = [(m, c["kind"], None, V.REQUIRED_NEW) for m, c in V.CANDIDATES_V3.items()] + \
             [(m, c["kind"], c["revision"], V.REQUIRED_EARLIER) for m, c in V.EXISTING_V3.items()]
CANDIDATES = [c for c in CANDIDATES if c[0] not in SKIP_MODELS]
RESULTS = {}
for mid, kind, rev, formats in CANDIDATES:
    t0 = time.time()
    print(f"\n=== {mid} ({kind}) ===", flush=True)
    if MOCK:
        RESULTS[mid] = {"model_id": mid, "kind": kind, "ok": False, "problems": ["offline test: not run"]}
        continue
    try:
        code, out = sh([sys.executable, "-m", "ler_v3", "engcheck", mid, kind, PROMPTS_PATH,
                        os.path.join(CACHE, "hub"), str(rev), ",".join(formats)], timeout=3600, cwd=SRC_DIR,
                       env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired:
        code, out = 1, "engineering check timed out after 60 minutes"
    r = V.parse_result(out, "ENGCHECK_V3_RESULT") or {"model_id": mid, "kind": kind, "ok": False,
                                                       "problems": ["the check process failed"]}
    r["process_tail"] = out[-4000:]
    RESULTS[mid] = r
    shutil.rmtree(os.path.join(CACHE, "hub", "models--" + mid.replace("/", "--")), ignore_errors=True)
    print(f"ok: {r['ok']} | {round(time.time() - t0)} s | problems: {r.get('problems')}")
    json.dump(RESULTS, open(os.path.join(OUT_DIR, "engcheck_v3.json"), "w"), indent=1, default=str)

# %% [markdown]
# ## Summary
# `ok` uses the formats the model will be run with (`formats_required`). The first three eligible new candidates, in
# the registered order, are the new models of the third study (PREREGISTRATION_V3.md, Section 10). Projected scoring
# time: 36,000 Yes/No prompts for every model, and 10,800 answer-option prompts for a new model (1,800 patients).

# %%
N_YESNO, N_CHOICE = 1800 * 20, 1800 * 6
rows = []
for mid, r in RESULTS.items():
    g = lambda k, f: (r.get(k) or {}).get(f)
    role = "new" if mid in V.CANDIDATES_V3 else "earlier"
    spp_yn, spp_ab = g("reading_timing_yesno_full", "seconds_per_prompt"), g("reading_timing_choice", "seconds_per_prompt")
    proj = (spp_yn or 0) * N_YESNO + ((spp_ab or 0) * N_CHOICE if role == "new" else 0)
    row = {"model": mid, "role": role, "ok": r.get("ok"), "revision": (r.get("revision") or "")[:12],
           "dtype": (r.get("config") or {}).get("dtype"), "quant": (r.get("config") or {}).get("quantization"),
           "window": r.get("max_model_len"), "load_s": r.get("load_seconds"),
           "tokens_med": (r.get("prompt_tokens") or {}).get("median"), "tokens_max": (r.get("prompt_tokens") or {}).get("max"),
           "exact_yn": g("exact_check_yesno_full", "max_abs_diff"), "exact_ab": g("exact_check_choice", "max_abs_diff")}
    for f in V.FMT_LABELS:
        row[f"mass_{f.replace('yesno_', '')}"] = g(f"reading_check_{f}", "mass_median")
    row.update(mass_timing_yn=g("reading_timing_yesno_full", "mass_median"), mass_timing_ab=g("reading_timing_choice", "mass_median"),
               imputed_yn=g("reading_timing_yesno_full", "imputed_share"), imputed_ab=g("reading_timing_choice", "imputed_share"),
               sanity="".join("y" if v else "n" for v in (r.get("sanity_right_word_more_likely") or {}).values()),
               s_per_prompt_yn=spp_yn, s_per_prompt_ab=spp_ab, projected_scoring_h=round(proj / 3600, 2),
               system_text=(r.get("template_system_text") or "")[:100], problems="; ".join(r.get("problems", [])))
    rows.append(row)
SUMMARY = pd.DataFrame(rows)
SELECTED = [m for m in V.CANDIDATES_V3 if RESULTS.get(m, {}).get("ok")][:3]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
if len(SUMMARY):
    print(SUMMARY.drop(columns=["system_text", "problems"]).round(4).to_string(index=False))
    print()
    for _, x in SUMMARY.iterrows():
        print(f"{x['model']}: system text {x['system_text']!r}\n    problems: {x['problems'] or 'none'}")
print("\nFirst three eligible new candidates (registered order):", SELECTED)

# %% [markdown]
# ## Save and zip

# %%
import zipfile
with open(os.path.join(OUT_DIR, "engcheck_v3.md"), "w", encoding="utf-8") as f:
    try:
        _table = SUMMARY.drop(columns=["problems"]).round(4).to_markdown(index=False)
    except ImportError:
        _table = "```\n" + SUMMARY.drop(columns=["problems"]).round(4).to_string(index=False) + "\n```"
    f.write("# Engineering check, third study\n\n" + _table + "\n\n## Problems\n\n" +
            "\n".join(f"- {m}: {p or 'none'}" for m, p in zip(SUMMARY.get("model", []), SUMMARY.get("problems", []))) +
            f"\n\n## First three eligible new candidates\n\n{SELECTED}\n")
LOG = {"code_version": L.CODE_VERSION, "v3_version": V.V3_VERSION, "code_sha256": CODE_SHA,
       "env": ENV, "backend": BACKEND_NOTE, "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()},
       "dev_fingerprint_check": FP_CHECK, "n_prompts": len(ITEMS), "selected": SELECTED,
       "prompt_keys_sha256": hashlib.sha256("".join(sorted(i["key"] for i in ITEMS)).encode()).hexdigest(),
       "total_minutes": round((time.time() - T_START) / 60, 1)}
json.dump(LOG, open(os.path.join(OUT_DIR, "engcheck_log.json"), "w"), indent=1, default=str)
ZIP = os.path.join(WORK, "engcheck_v3.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("engcheck_v3.json", "engcheck_v3.md", "engcheck_log.json"):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            z.write(os.path.join(OUT_DIR, name), name)
    z.write(os.path.join(SRC_DIR, "ler_main.py"), "ler_main.py")
    z.write(os.path.join(SRC_DIR, "ler_v3.py"), "ler_v3.py")
print("Saved:", ZIP, f"| total {(time.time() - T_START) / 60:.1f} min")
print("DONE. Download engcheck_v3.zip from the Output tab.")
