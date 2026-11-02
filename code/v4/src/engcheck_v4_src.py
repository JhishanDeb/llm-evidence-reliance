# %% [markdown]
# # Engineering check of the models for the fourth study (before registration)
#
# **What this notebook does.** Before `PREREGISTRATION_V4.md` is tagged, it checks that each model of the fourth study
# can be run as registered, and in which formats: download, loading in float16 with vLLM on Kaggle's two T4 GPUs (with
# the model's window), the chat template and its default system text, single-token answer words (Yes/No and True/False),
# the exact-reading check (Deviation D2), finite readings, the answer-word mass in every framing and format, the share of
# imputed readings under the top-1,000 reading (Deviation D5), prompt lengths, the time per prompt, questions with an
# obvious answer, and for the instruction-tuned models the time and parse share of generated answers. For each base
# model it also checks that its tokenizer gives exactly the token ids of the instruction-tuned tokenizer in the chat
# format. At the end it applies the rules of Section 10 and prints the decisions to record in the protocol and the code.
#
# **What it does not do.** It computes no accuracy, no answer direction, no agreement of generated answers with the
# reading and no asymmetry. It uses dev-sample patients only (never test patients).
#
# **How to run on Kaggle**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**.
# 3. Add-ons → Secrets: `HF_TOKEN` (the same token as before; Llama needs the accepted licence).
# 4. **Save Version → Save & Run All (Commit).** Expected time: about 1.5 to 2 hours for all eight models; about 35
#    minutes for the second check (three models, set in the first code cell).
# 5. Download `engcheck_v4.zip` (or `engcheck_v4_<RUN_LABEL>.zip`) from the Output tab and send it to me.

# %%
# ============ settings ============
SKIP_MODELS = []            # model ids to leave out. The second check (2026-11-02, RUN_LABEL "run2") left out
                            # "Qwen/Qwen2.5-7B", "allenai/Olmo-3-7B-Instruct", "tiiuae/Falcon3-7B-Instruct",
                            # "01-ai/Yi-1.5-9B-Chat-16K" and "allenai/Olmo-3-1025-7B" (PREREGISTRATION_V4.md, Section 10)
RUN_LABEL = ""              # the output is engcheck_v4_<RUN_LABEL>.zip, or engcheck_v4.zip if empty
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
OUT_DIR = os.path.join(WORK, "engcheck_v4")
ZIP_NAME = f"engcheck_v4_{RUN_LABEL}.zip" if RUN_LABEL else "engcheck_v4.zip"
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
# ## Install vLLM and the pinned xgboost
# vLLM is matched to Kaggle's preinstalled torch (the procedure that worked in the dev phase). xgboost is pinned
# to the dev-phase version so that the features and cards are rebuilt exactly; the notebook checks this with the
# dev fingerprints below. No memorization check runs in this study (every model was checked in an earlier study).

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


VLLM_PIN = "0.26.0"     # the dev-phase version; the same in every run of all four studies


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

# pinned xgboost (dev-phase version), without touching torch / numpy / pandas
XGB_PIN = "3.4.1"
pins = _pkg_versions(["torch", "numpy", "pandas", "xgboost"])
with open("/tmp/ler_constraints.txt", "w") as f:
    for k in ("torch", "numpy", "pandas"):
        if k in pins:
            f.write(f"{k}=={pins[k].split('+')[0]}\n")
want = []
if pins.get("xgboost") != XGB_PIN and not (MOCK or LOCAL_MODEL):
    want.append(f"xgboost=={XGB_PIN}")
if want:
    code, out = sh([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "-c", "/tmp/ler_constraints.txt",
                    *want], timeout=1800)
    BACKEND_NOTE["extra_install"] = {"packages": want, "ok": code == 0, "tail": out[-1500:]}
    print("installed:", want, "| ok:", code == 0)
    if code != 0:
        print(out[-1500:])
after = _pkg_versions(["xgboost"])
if not (MOCK or LOCAL_MODEL) and after.get("xgboost") != XGB_PIN:
    raise RuntimeError(f"xgboost {XGB_PIN} (the dev-phase version) could not be installed; found "
                       f"{after.get('xgboost')}. The features and cards could differ from the dev phase.")
TIMINGS["install"] = time.time() - t0
print({k: v for k, v in BACKEND_NOTE.items() if not k.endswith("_tail") and k != "extra_install"})

# %% [markdown]
# ## Shared code
# `ler_main.py` (main-2.0) and `ler_v3.py` (v3-1.1), unchanged, and `ler_v4.py` (new formats, models, engineering check).

# %% [writefile ler_main.py]

# %% [writefile ler_v3.py]

# %% [writefile ler_v4.py]

# %% [markdown]
# ## Data, splits, XGBoost and the dev sample, checked against the dev phase

# %%
import importlib, hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
for _m in ("ler_v4", "ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_v3 as V
import ler_v4 as W
CODE_SHA = {n: L.file_sha256(os.path.join(SRC_DIR, n)) for n in ("ler_main.py", "ler_v3.py", "ler_v4.py")}
if CODE_SHA["ler_main.py"] != W.MAIN_SHA256 or CODE_SHA["ler_v3.py"] != W.V3_SHA256:
    raise RuntimeError("ler_main.py or ler_v3.py is not the registered version")
print("code:", {k: v[:16] for k, v in CODE_SHA.items()}, "| ler_v4", W.V4_VERSION)
# gated models: stop now if Hugging Face refuses the download (before the data step and the model checks)
_GATED = [m for m in W.MODELS_V4 if m.startswith("meta-llama/") and m not in SKIP_MODELS]
if _GATED and not MOCK:
    _bad = W.access_problems(_GATED, HF_TOKEN, os.path.join(CACHE, "access_check"))
    if _bad:
        raise RuntimeError("Hugging Face refuses the download of " + "; ".join(f"{m} ({p})" for m, p in _bad.items())
                           + ". " + W.ACCESS_HELP)
    print("access to the gated models confirmed:", _GATED)
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
# Check set, per dataset: 6 dev patients × every prompt of the grids (37 for an instruction-tuned model with the plain
# format, 12 for a base model). Timing set: every dev patient, C2, alarm framing, registered prompt (100 per dataset).
# Generated answers: 2 dev patients × C0, C2, C4 × both framings (12 per dataset, in each mode). Every prompt is checked
# against the registered prompt as in the run notebook.

# %%
ITEMS = []
for ds, b in BUNDLES.items():
    for rid in b.ids[:6]:
        for g in W.INSTRUCT_GRIDS + ("plain", "chat"):
            for cond, fr, shot in W.GRID_DEF[g]:
                text, cdir = W.prompt_v4(b, rid, cond, fr, shot)
                W.check_prompt_v4(b, rid, cond, fr, shot, text, cdir)
                ITEMS.append({"set": "check", "grid": g, "framing": fr, "shot": shot,
                              "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": text})
    for rid in b.ids:
        ITEMS.append({"set": "timing", "grid": "split", "framing": "alarm", "shot": "A",
                      "key": f"{ds}|{rid}|C2|alarm|A", "text": b.prompt(rid, "C2", "alarm", "A")[0]})
    for rid in b.ids[:2]:
        for cond, fr, shot in W.GREEDY_GRID:
            ITEMS.append({"set": "greedy", "grid": "split", "framing": fr, "shot": shot,
                          "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": b.prompt(rid, cond, fr, shot)[0]})
PROMPTS_PATH = os.path.join(OUT_DIR, "dev_prompts.json")
json.dump(ITEMS, open(PROMPTS_PATH, "w"))
print(pd.Series([f"{i['set']} {i['grid']}" for i in ITEMS]).value_counts().sort_index())
_b = BUNDLES[L.DATASETS[0]]
for _fr, _shot in (("tf_alarm", "A"), ("negated", "A"), ("para_alarm", "A"), ("alarm", "Anet")):
    _t = W.prompt_v4(_b, _b.ids[0], "C2", _fr, _shot)[0]
    print(f"\n{_fr} / {_shot} (dev patient), last lines:\n" + "\n".join(_t.splitlines()[-4:]))

# %% [markdown]
# ## Check each model (one separate process per model, so the GPU memory is freed in between)
# The five instruction-tuned models, then the three base models in the registered order (Llama 3.1, Qwen2.5, Olmo 3),
# each with the tokenizer of its family's instruction-tuned model for the chat format.

# %%
CANDIDATES = [m for m in W.MODELS_V4 if W.MODELS_V4[m]["kind"] == "instruct"] + \
             [m for f in W.R10_FAMILY_ORDER for m, c in W.MODELS_V4.items() if c["kind"] == "base" and c["family"] == f]
CANDIDATES = [m for m in CANDIDATES if m not in SKIP_MODELS]
RESULTS = {}
for mid in CANDIDATES:
    c = W.MODELS_V4[mid]
    t0 = time.time()
    print(f"\n=== {mid} ({c['kind']}) ===", flush=True)
    if MOCK:
        RESULTS[mid] = {"model_id": mid, "kind": c["kind"], "ok": False, "problems": ["offline test: not run"]}
        continue
    chat = c.get("chat_model")
    args = [sys.executable, "-m", "ler_v4", "engcheck", mid, c["kind"], PROMPTS_PATH, os.path.join(CACHE, "hub"),
            c["revision"], str(chat), str(W.MODELS_V4[chat]["revision"] if chat else None)]
    try:
        code, out = sh(args, timeout=3600, cwd=SRC_DIR, env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired:
        code, out = 1, "engineering check timed out after 60 minutes"
    r = W.parse_result(out, "ENGCHECK_V4_RESULT") or {"model_id": mid, "kind": c["kind"], "ok": False,
                                                       "problems": ["the check process failed"]}
    r["process_tail"] = out[-4000:]
    RESULTS[mid] = r
    shutil.rmtree(os.path.join(CACHE, "hub", "models--" + mid.replace("/", "--")), ignore_errors=True)
    print(f"ok: {r['ok']} | {round(time.time() - t0)} s | formats ok: {r.get('formats_ok')} | problems: {r.get('problems')}")
    json.dump(RESULTS, open(os.path.join(OUT_DIR, "engcheck_v4.json"), "w"), indent=1, default=str)

# %% [markdown]
# ## Summary and the decisions of Section 10
# Median answer-word mass on the check prompts of every framing and format, imputed share and time per prompt on the
# timing prompts, the exact-reading checks, the obvious questions, generated answers, and the token identity of the
# base models' chat format. The decisions follow the rules of Section 10 and go into `ler_v4.py` (ENGCHECK_DECISIONS)
# and Section 10 of the protocol before the tag.

# %%
rows = []
N_PER_MODEL = {"instruct": 1800 * 31, "base": 1800 * 12}
for mid, r in RESULTS.items():
    rd = r.get("readings") or {}
    spp = (rd.get("timing|split|alarm|A") or {}).get("seconds_per_prompt")
    row = {"model": mid, "kind": r.get("kind"), "ok": r.get("ok"), "revision": (r.get("revision") or "")[:12],
           "window": r.get("max_model_len"), "tokens_max": (r.get("prompt_tokens") or {}).get("max"),
           "exact_yn": (r.get("exact_check_Yes/No") or {}).get("max_abs_diff"),
           "exact_tf": (r.get("exact_check_True/False") or {}).get("max_abs_diff"),
           "imputed_timing": (rd.get("timing|split|alarm|A") or {}).get("imputed_share"),
           "s_per_prompt": spp, "projected_scoring_h": round((spp or 0) * N_PER_MODEL.get(r.get("kind"), 0) / 3600, 2),
           "sanity": "".join("y" if v else "n" for v in (r.get("sanity_right_word_more_likely") or {}).values()),
           "parsed_prefill": (r.get("generated_prefill") or {}).get("parsed_share"),
           "parsed_free": (r.get("generated_free") or {}).get("parsed_share"),
           "chat_tokens_identical": (r.get("chat_token_identity") or {}).get("identical"),
           "formats_ok": ",".join(k for k, v in (r.get("formats_ok") or {}).items() if v),
           "formats_failed": ",".join(k for k, v in (r.get("formats_ok") or {}).items() if not v),
           "system_text": (r.get("template_system_text") or "")[:100], "problems": "; ".join(r.get("problems", []))}
    for k, v in rd.items():
        if k.startswith("check|"):
            row["mass " + k.split("|", 2)[2]] = v.get("mass_median")
    rows.append(row)
SUMMARY = pd.DataFrame(rows)
DECISIONS = W.decisions_from_check(RESULTS)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
if len(SUMMARY):
    print(SUMMARY.drop(columns=["system_text", "problems"]).round(4).T.to_string())
    print()
    for _, x in SUMMARY.iterrows():
        print(f"{x['model']}: system text {x['system_text']!r}\n    problems: {x['problems'] or 'none'}")
print("\nDecisions (rules of Section 10), to record in ler_v4.py and PREREGISTRATION_V4.md:")
print("ENGCHECK_DECISIONS =", json.dumps(DECISIONS))

# %% [markdown]
# ## Save and zip

# %%
import zipfile
with open(os.path.join(OUT_DIR, "engcheck_v4.md"), "w", encoding="utf-8") as f:
    try:
        _table = SUMMARY.drop(columns=["problems"]).round(4).T.to_markdown()
    except ImportError:
        _table = "```\n" + SUMMARY.drop(columns=["problems"]).round(4).T.to_string() + "\n```"
    f.write("# Engineering check, fourth study\n\n" + _table + "\n\n## Problems\n\n" +
            "\n".join(f"- {m}: {p or 'none'}" for m, p in zip(SUMMARY.get("model", []), SUMMARY.get("problems", []))) +
            f"\n\n## Decisions\n\n{json.dumps(DECISIONS)}\n")
LOG = {"code_version": L.CODE_VERSION, "v3_version": V.V3_VERSION, "v4_version": W.V4_VERSION, "code_sha256": CODE_SHA,
       "env": ENV, "backend": BACKEND_NOTE, "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()},
       "dev_fingerprint_check": FP_CHECK, "n_prompts": len(ITEMS), "decisions": DECISIONS, "skip_models": SKIP_MODELS,
       "run_label": RUN_LABEL,
       "prompt_keys_sha256": hashlib.sha256("".join(sorted(i["key"] + i["set"] for i in ITEMS)).encode()).hexdigest(),
       "total_minutes": round((time.time() - T_START) / 60, 1)}
json.dump(LOG, open(os.path.join(OUT_DIR, "engcheck_log.json"), "w"), indent=1, default=str)
ZIP = os.path.join(WORK, ZIP_NAME)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("engcheck_v4.json", "engcheck_v4.md", "engcheck_log.json"):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            z.write(os.path.join(OUT_DIR, name), name)
    for name in ("ler_main.py", "ler_v3.py", "ler_v4.py"):
        z.write(os.path.join(SRC_DIR, name), name)
print("Saved:", ZIP, f"| total {(time.time() - T_START) / 60:.1f} min")
print(f"DONE. Download {ZIP_NAME} from the Output tab.")
