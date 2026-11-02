# %% [markdown]
# # Fourth study: one run per model (pre-registered, `prereg-v4`)
#
# **What this notebook does.** It runs one model of the fourth study
# ([`PREREGISTRATION_V4.md`, release `prereg-v4`](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v4)).
# For an instruction-tuned model: every registered test prompt (the registered Yes/No prompts, True/False statements,
# the negated question, the alarming-last example order in both framings, the paraphrased prompt, the cue and the
# net-number card, and the plain-text format if the engineering check selected it), then greedy generated answers for a
# subsample. For a base model: the registered prompts in plain text and through the chat template of the
# instruction-tuned model of its family. It tests no hypothesis; the analysis notebook does that after all runs.
#
# **Before the first run:** `PREREGISTRATION_V4.md` with the result of the engineering check must be committed and the
# `prereg-v4` release created. The notebook checks this and stops if it is missing.
#
# **How to run on Kaggle (once per model, seven runs)**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**. Add-ons → Secrets: `HF_TOKEN`.
# 3. In the first code cell set `MODEL_ID` to one of the registered models.
# 4. **Save Version → Save & Run All (Commit).** Expected time: about 3 to 4 hours for Llama-3.1-8B-Instruct,
#    Qwen2.5-7B-Instruct, Olmo-3-7B-Instruct and Falcon3-7B-Instruct, about 5.5 hours for Yi-1.5-9B-Chat-16K, and about
#    1.5 hours for each base model. Kaggle allows 12 hours per session.
# 5. Download `v4_<model>.zip` from the Output tab. Keep all of them.
#
# **If a run stops part-way:** add the failed version's output as an input (Add Input → Your Work → this
# notebook) and run again. The scoring and generation cells pick up the saved checkpoints.

# %%
# ============ settings (the only cell you change) ============
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# The registered models of the fourth study, one per run, in this order (PREREGISTRATION_V4.md, Sections 4 and 9):
#   "meta-llama/Llama-3.1-8B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"  "allenai/Olmo-3-7B-Instruct"
#   "tiiuae/Falcon3-7B-Instruct"        "01-ai/Yi-1.5-9B-Chat-16K"
#   then the base models of the two R10 families that the engineering check selected:
#   "meta-llama/Llama-3.1-8B"  "Qwen/Qwen2.5-7B"
BACKEND = "vllm"            # vLLM only (a model that vLLM cannot run counts as "cannot be run")
READ_OVERRIDE = "topk"      # Deviation D5 of the first study, kept: the top 1,000 log-probabilities (the check still runs)

import os, sys, time, json, subprocess, platform
T_START = time.time()
MOCK = os.environ.get("LER_MOCK") == "1"            # CPU self-test with a fake model; never set on Kaggle
LOCAL_MODEL = os.environ.get("LER_LOCAL_MODEL")      # offline test with a small local model (transformers)
if os.environ.get("LER_MODEL_ID"):
    MODEL_ID = os.environ["LER_MODEL_ID"]
if LOCAL_MODEL:
    BACKEND = "hf"


def sh(cmd, timeout=None, cwd=None, env=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout,
                       cwd=cwd, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


# torch is probed in a subprocess so the kernel does not import it before vLLM is installed
_probe = ("import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
          "'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
code, out = sh([sys.executable, "-c", _probe])
ENV = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    ENV.update(json.loads(out.strip().splitlines()[-1]))
except Exception:
    ENV["torch_probe_error"] = out[-500:]
ENV["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")[1].strip()
print("Python :", ENV["python"])
print("torch  :", ENV.get("torch"), "| CUDA:", ENV.get("cuda"))
print("GPUs   :", ENV.get("gpus"), "|", ENV["nvidia_smi"][:200])
print("MODEL_ID =", MODEL_ID, "| BACKEND =", BACKEND, "| MOCK =", MOCK, "| LOCAL_MODEL =", LOCAL_MODEL)
if not MOCK and not LOCAL_MODEL and not ENV.get("gpus"):
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
for _d in (DATA_DIR, CACHE):
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
if MODEL_ID.startswith("meta-llama/") and not (HF_TOKEN or MOCK or LOCAL_MODEL):
    raise RuntimeError("This model is gated: add your Hugging Face token as the Kaggle secret HF_TOKEN "
                       "(Add-ons -> Secrets) after your access request was approved.")
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
# `ler_main.py` (main-2.0) and `ler_v3.py` (v3-1.1) are the registered code of the earlier studies, unchanged: data,
# splits, XGBoost, cards, the registered prompts, the reading, the statistics and the cue. `ler_v4.py` adds the new
# prompt formats, the models of the fourth study, the protocol check, generated answers and the fourth study's analysis.

# %% [writefile ler_main.py]

# %% [writefile ler_v3.py]

# %% [writefile ler_v4.py]

# %% [markdown]
# ## Check the code and the protocol: prompt strings, models and revisions
# `ler_main.py` and `ler_v3.py` must be byte-identical to main-2.0 and v3-1.1. The notebook downloads the four protocol
# files from the `prereg-v4` release: every fixed prompt string, model id and pinned revision in the code must appear
# there word for word, and the first study's deviations must be present. Otherwise the run stops.

# %%
import importlib
import hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
for _m in ("ler_v4", "ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]          # cells re-run by hand -> use the freshly written files
import ler_main as L
import ler_v3 as V
import ler_v4 as W

CODE_SHA = {n: L.file_sha256(os.path.join(SRC_DIR, n)) for n in ("ler_main.py", "ler_v3.py", "ler_v4.py")}
if CODE_SHA["ler_main.py"] != W.MAIN_SHA256 or L.CODE_VERSION != "main-2.0":
    raise RuntimeError(f"ler_main.py is not the registered main-2.0 (sha256 {CODE_SHA['ler_main.py']})")
if CODE_SHA["ler_v3.py"] != W.V3_SHA256 or V.V3_VERSION != "v3-1.1":
    raise RuntimeError(f"ler_v3.py is not the registered v3-1.1 (sha256 {CODE_SHA['ler_v3.py']})")
if MODEL_ID not in W.MODELS_V4:
    raise RuntimeError(f"{MODEL_ID} is not one of the registered models of the fourth study: {list(W.MODELS_V4)}")
DECISIONS = W.ENGCHECK_DECISIONS
if DECISIONS is None and (MOCK or LOCAL_MODEL):      # offline tests: decisions given by the test, or all formats
    DECISIONS = json.loads(os.environ.get("LER_MOCK_DECISIONS", "null")) or {
        "formats_not_run": {}, "plain_run": ["llama-instruct", "qwen-instruct"], "r10_families": ["llama", "qwen"]}
if DECISIONS is None:
    raise RuntimeError("The result of the engineering check is not recorded in ler_v4.py (ENGCHECK_DECISIONS). It must be "
                       "recorded in PREREGISTRATION_V4.md and in the code before the runs.")
_M = W.MODELS_V4[MODEL_ID]
MODEL_KIND, MODEL_FAMILY, MODEL_TAG, MODEL_REVISION = _M["kind"], _M["family"], _M["tag"], _M["revision"]
GRIDS = W.model_grids(MODEL_ID, DECISIONS)
CHAT_MODEL = _M.get("chat_model")
OUT_DIR = os.path.join(WORK, f"v4_outputs_{MODEL_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"model {MODEL_ID}: {MODEL_KIND}, family {MODEL_FAMILY}, grids {GRIDS}, revision {MODEL_REVISION}"
      + (f", chat template of {CHAT_MODEL}" if CHAT_MODEL else "") + f", outputs in {OUT_DIR}")
print("code:", {k: v[:16] for k, v in CODE_SHA.items()}, "| ler_v4", W.V4_VERSION)
print("decisions of the engineering check:", json.dumps(DECISIONS))
_GATED = [m for m in (MODEL_ID, CHAT_MODEL) if m and m.startswith("meta-llama/")]
if _GATED and not (MOCK or LOCAL_MODEL):
    _bad = W.access_problems(_GATED, HF_TOKEN, os.path.join(CACHE, "access_check"))
    if _bad:
        raise RuntimeError("Hugging Face refuses the download of " + "; ".join(f"{m} ({p})" for m, p in _bad.items())
                           + ". " + W.ACCESS_HELP)
    print("access to the gated models confirmed:", _GATED)

PROTOCOL = {"tag": W.PROTOCOL_V4_TAG, "urls": dict(W.PROTOCOL_AT_V4_URLS, **{"PREREGISTRATION_V4.md": W.PROTOCOL_V4_URL})}
try:
    _rd = lambda env, url: (open(os.environ[env], encoding="utf-8").read() if os.environ.get(env)
                            else L.http_get(url).decode("utf-8"))
    _t1 = _rd("LER_PROTOCOL_PATH", W.PROTOCOL_AT_V4_URLS["PREREGISTRATION.md"])
    _t2 = _rd("LER_PROTOCOL_V2_PATH", W.PROTOCOL_AT_V4_URLS["PREREGISTRATION_V2.md"])
    _t3 = _rd("LER_PROTOCOL_V3_PATH", W.PROTOCOL_AT_V4_URLS["PREREGISTRATION_V3.md"])
    _t4 = _rd("LER_PROTOCOL_V4_PATH", W.PROTOCOL_V4_URL)
except Exception as e:
    raise RuntimeError(f"Could not download the protocol files at the tag {W.PROTOCOL_V4_TAG} ({e!r}). Commit "
                       "PREREGISTRATION_V4.md and create the prereg-v4 release on GitHub before the runs.") from e
for _k, _t in (("sha256_v1_file", _t1), ("sha256_v2_file", _t2), ("sha256_v3_file", _t3), ("sha256", _t4)):
    PROTOCOL[_k] = hashlib.sha256(_t.encode("utf-8")).hexdigest()
_missing = {"v1": L.check_against_protocol(_t1) + L.check_deviations(_t1), "v2": L.check_against_protocol_v2(_t2),
            "v3": V.check_against_protocol_v3(_t3), "v4": W.check_against_protocol_v4(_t4)}
PROTOCOL["strings_missing"] = _missing
for _k in ("v1", "v2", "v3"):
    if _missing[_k]:
        raise RuntimeError(f"The protocol file of study {_k} at prereg-v4 lacks these strings:\n" + "\n".join(_missing[_k]))
if _missing["v4"] and not (MOCK or LOCAL_MODEL):
    raise RuntimeError("PREREGISTRATION_V4.md at prereg-v4 lacks these strings, model ids, revisions or sections:\n"
                       + "\n".join(_missing["v4"]))
if _missing["v4"]:
    print("OFFLINE TEST: PREREGISTRATION_V4.md lacks", _missing["v4"], "(a real run would stop here)")
print(f"Protocol check passed ({W.PROTOCOL_V4_TAG}, PREREGISTRATION_V4.md sha256 {PROTOCOL['sha256'][:16]}): prompt "
      "strings, models and revisions match.")

try:
    _ids_txt = (open(os.environ["LER_PILOT_IDS_PATH"]).read() if os.environ.get("LER_PILOT_IDS_PATH")
                else L.http_get(L.PILOT_IDS_URL).decode("utf-8"))
    PILOT_IDS = [ln.strip() for ln in _ids_txt.splitlines() if ln.strip() and not ln.startswith("#")]
    if sorted(PILOT_IDS) != sorted(str(i) for i in L.PILOT_IDS_FALLBACK):
        raise RuntimeError("pilot_record_ids.txt on GitHub differs from the copy in the code")
    PROTOCOL["pilot_ids_source"] = "github prereg-v1 (matches built-in copy)"
except RuntimeError:
    raise
except Exception as e:
    PILOT_IDS = [str(i) for i in L.PILOT_IDS_FALLBACK]
    PROTOCOL["pilot_ids_source"] = f"built-in copy (GitHub not reachable: {e!r})"
print("pilot patients excluded from dev/test pools:", len(PILOT_IDS), "|", PROTOCOL["pilot_ids_source"])

# %% [markdown]
# ## Data: three public datasets
# PhysioNet 2012 (ODC-By 1.0), PhysioNet 2019 (CC BY 4.0) and UCI Diabetes 130 (CC BY 4.0), exactly as in the
# earlier studies.

# %%
t0 = time.time()
TABLES, DATA_INFO = {}, {}
for ds in L.DATASETS:
    t1 = time.time()
    print(f"[{ds}]")
    if ds == "physionet2019":
        df_, feats_, cats_, info_ = L.load_physionet2019(DATA_DIR, EXTRA_ROOTS, log=print,
                                                          workers=max(1, min(4, os.cpu_count() or 1)))
    else:
        df_, feats_, cats_, info_ = L.LOADERS[ds](DATA_DIR, EXTRA_ROOTS, log=print)
    info_["licence"] = L.SPEC[ds]["licence"]
    info_["seconds"] = round(time.time() - t1, 1)
    TABLES[ds], DATA_INFO[ds] = (df_, feats_, cats_), info_
    print("  ", json.dumps({k: v for k, v in info_.items() if k != "id_mapping_tables"}))
    if info_["positives"] < 50 or len(df_) < 500:
        raise RuntimeError(f"{ds}: too few patients or positives; check the data step")
TIMINGS["data"] = time.time() - t0

# %% [markdown]
# ## Splits, XGBoost, cards and the test set, checked against the dev phase
# The splits, the 16 features, the dev patients' cards and the solved examples must be identical to the dev run
# (fingerprints). The test set is the one of the earlier studies: 300 patients per class per dataset (seed 2027).

# %%
t0 = time.time()
BUNDLES = {}
FP_CHECK = {}
for ds in L.DATASETS:
    df_, feats_, cats_ = TABLES[ds]
    b = L.Bundle(ds, df_, feats_, cats_, DATA_INFO[ds], pilot_ids=PILOT_IDS if ds == "physionet2012" else (),
                 sample="test")
    BUNDLES[ds] = b
    fp = b.info["fingerprint"]
    expected = L.DEV_RUN["fingerprints"][ds]
    FP_CHECK[ds] = {k: fp[k] == v for k, v in expected.items()}
    print("   test sample:", json.dumps(b.info["test_sample"]), "| cards:", json.dumps(b.info["test_cards"]))
    print("   dev fingerprints match:", FP_CHECK[ds])
TIMINGS["bundles"] = time.time() - t0
if not (MOCK or LOCAL_MODEL):
    bad = {ds: [k for k, ok in v.items() if not ok] for ds, v in FP_CHECK.items() if not all(v.values())}
    if bad:
        raise RuntimeError(f"The rebuilt data differ from the dev phase: {bad}. Do not run the study on data "
                           "that differ from the dev run; send me this output.")
    print("All dev fingerprints match: same splits, features, dev cards and solved examples as the dev run.")

# %% [markdown]
# ## Build every test prompt and check them
# Every prompt of the model's grids (Section 6). Each new prompt is checked against the registered prompt of the same
# patient, condition and example set: it may differ only where Section 5 of the protocol says.

# %%
RECORDS = []
GRID = W.grid_v4(GRIDS)
_max_p = int(os.environ.get("LER_TEST_MAX_PATIENTS", 0)) if (MOCK or LOCAL_MODEL) else 0   # offline tests only
for ds, b in BUNDLES.items():
    b.use_sample("test")
    for rid in (b.ids[:_max_p] if _max_p else b.ids):
        for cond, framing, shot in GRID:
            text, cdir = W.prompt_v4(b, rid, cond, framing, shot)
            RECORDS.append({"dataset": ds, "id": rid, "cond": cond, "framing": framing, "shot_set": shot,
                            "y": b.y[rid], "card_dir": cdir, "xgb_p": float(b.xgb_p[rid]),
                            "key": f"{ds}|{rid}|{cond}|{framing}|{shot}", "text": text,
                            "text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]})
PROMPTS_DF = pd.DataFrame(RECORDS)
_byk = {r["key"]: r for r in RECORDS}
_strip = lambda t: "\n".join(ln for ln in t.splitlines() if not ln.startswith(
    ("Question:", "Statement:", "Answer", "Is this statement true or false?", "Reply with")))
_TWIN = {"reversed": "alarm", "tf_reversed": "tf_alarm", "para_reversed": "para_alarm", "negated": "alarm"}
for r in RECORDS:
    b = BUNDLES[r["dataset"]]
    W.check_prompt_v4(b, r["id"], r["cond"], r["framing"], r["shot_set"], r["text"], r["card_dir"])
    tw = _TWIN.get(r["framing"])
    if tw:
        twin = _byk[r["key"].replace(f"|{r['framing']}|", f"|{tw}|")]
        assert _strip(r["text"]) == _strip(twin["text"]), "framings differ outside the question, statement and answer lines"
        assert r["card_dir"] == twin["card_dir"]
    if r["cond"] == "C4":
        c2 = _byk[r["key"].replace("|C4|", "|C2|")]
        assert r["card_dir"] == 1 - c2["card_dir"], "C4 card does not point opposite to C2"
_nz = sum(1 for ds, b in BUNDLES.items() for rid in b.ids if abs(sum(it["shap"] for it in b.cards[rid])) < 0.005)
CHECKS = {"prompts_checked": len(RECORDS), "grids": list(GRIDS),
          "prompts_per_dataset": PROMPTS_DF.groupby("dataset").size().to_dict(),
          "prompts_per_framing_and_set": {f"{k[0]}/{k[1]}": int(v) for k, v in
                                          PROMPTS_DF.groupby(["framing", "shot_set"]).size().items()},
          "patients_with_net_impact_printed_as_zero": _nz,
          "prompt_set_sha256": hashlib.sha256("".join(sorted(PROMPTS_DF["key"] + PROMPTS_DF["text_sha"])).encode())
          .hexdigest()}
print(f"{len(RECORDS)} test prompts built and checked: {CHECKS['prompts_per_dataset']}")
print(json.dumps(CHECKS["prompts_per_framing_and_set"]))
print("patients whose net impact prints as +0.00 or -0.00:", _nz)
_rid = BUNDLES[L.DATASETS[0]].ids[0]
for _fr, _shot in (("tf_alarm", "A"), ("negated", "A"), ("para_alarm", "A"), ("alarm", "Anet")):
    _show = f"{L.DATASETS[0]}|{_rid}|C2|{_fr}|{_shot}"
    if _show in _byk:
        print(f"\nEnd of one prompt ({_show}):\n...")
        print(_byk[_show]["text"][-700:])

# DEV prompts for the vLLM exact-reading check (Deviation D2); no test output is used for the check
_dev = {"yesno": [], "tf": [], "chat": []}
for ds, b in BUNDLES.items():
    b.use_sample("dev")
    for rid in b.ids[:6]:
        for cond in W.SPLIT_CONDS:
            _dev["yesno"].append(b.prompt(rid, cond, "alarm", "A")[0])
            if "tf" in GRIDS and rid in b.ids[:3]:
                _dev["tf"].append(W.prompt_v4(b, rid, cond, "tf_alarm", "A")[0])
    b.use_sample("test")

# %% [markdown]
# ## Model: download, vLLM smoke test with the exact-reading check, load
# The smoke test runs in a separate process (a failed engine start cannot leave this notebook holding GPU memory). The
# model window is 16,384 tokens or the model's own window if shorter. A base model's chat format uses the tokenizer
# and chat template of the instruction-tuned model of its family, at its registered revision (tokenizer files only).

# %%
t0 = time.time()
if MOCK:
    MODEL_PATH, MODEL_REV = None, "mock"
elif LOCAL_MODEL:
    MODEL_PATH, MODEL_REV = LOCAL_MODEL, "local-test-model"
else:
    MODEL_PATH, MODEL_REV = L.download_model(MODEL_ID, os.path.join(CACHE, "hub"), MODEL_REVISION, HF_TOKEN)
    print("model revision:", MODEL_REV)
    if MODEL_REV != MODEL_REVISION:
        raise RuntimeError(f"downloaded revision {MODEL_REV} differs from the registered {MODEL_REVISION}")
CHAT_TOK, CHAT_REV = None, None
if CHAT_MODEL and not MOCK:
    from transformers import AutoTokenizer
    _cpath, CHAT_REV = W.download_tokenizer(CHAT_MODEL, os.path.join(CACHE, "hub"), W.MODELS_V4[CHAT_MODEL]["revision"],
                                            HF_TOKEN)
    if CHAT_REV != W.MODELS_V4[CHAT_MODEL]["revision"] and not LOCAL_MODEL:
        raise RuntimeError(f"chat tokenizer revision {CHAT_REV} differs from the registered one")
    CHAT_TOK = AutoTokenizer.from_pretrained(_cpath)
TIMINGS["model_download"] = time.time() - t0
WINDOW = V.model_window(MODEL_PATH) if MODEL_PATH else L.MAX_MODEL_LEN
print("model window (max_model_len):", WINDOW)

READ_MODE = "exact"
SMOKE_PROMPTS_PATH = os.path.join(OUT_DIR, "smoke_prompts.json")
if USE_VLLM:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    _sp = {"yesno": [L.render(t, MODEL_KIND, _tok) for t in _dev["yesno"]],
           "tf": [L.render(t, MODEL_KIND, _tok) for t in _dev["tf"]]}
    if CHAT_TOK is not None:
        _sp["yesno"] += [W.render_v4(t, MODEL_KIND, "Achat", _tok, CHAT_TOK) for t in _dev["yesno"][:9]]
    json.dump(_sp, open(SMOKE_PROMPTS_PATH, "w"))
    t0 = time.time()
    try:
        code, out = sh([sys.executable, "-m", "ler_v4", "smoke", MODEL_PATH, MODEL_KIND, SMOKE_PROMPTS_PATH],
                       timeout=1800, cwd=SRC_DIR, env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired:
        code, out = 1, "smoke test timed out"
    SMOKE = W.parse_result(out, "SMOKE_V4_RESULT") if code == 0 else None
    TIMINGS["smoke_test"] = time.time() - t0
    if SMOKE:
        BACKEND_NOTE["vllm_smoke_test"] = SMOKE
        READ_MODE = SMOKE["read"]
        print("smoke test ok | read mode from the check:", READ_MODE, "| exact check:", json.dumps(SMOKE["exact_check"]))
        if SMOKE.get("exact_check_tf"):
            print("exact check, True/False:", json.dumps(SMOKE["exact_check_tf"]))
        if READ_OVERRIDE and READ_OVERRIDE != READ_MODE:
            BACKEND_NOTE["read_override"] = f"{READ_MODE} -> {READ_OVERRIDE} (Deviation D5)"
            READ_MODE = READ_OVERRIDE
        print("read mode used:", READ_MODE, "(Deviation D5 of the first study: the same reading in every run)")
        print("sky prompt:", SMOKE["sky_prompt"])
    else:
        BACKEND_NOTE["vllm_smoke_test"] = out[-3000:]
        print("vLLM smoke test FAILED\n", out[-3000:])
        raise RuntimeError("vLLM smoke test failed: under PREREGISTRATION_V4.md this model 'cannot be run'")

t0 = time.time()
N_GPU = len(ENV.get("gpus") or [])
if MOCK:
    _mock_cfg = json.loads(os.environ.get("LER_MOCK_CFG", "{}"))
    _truth = {"\n" + b._block(b.values[r]): b.y[r] for b in BUNDLES.values() for r in b.ids}
    BACKEND_OBJ = W.MockV4(truth=_truth, **_mock_cfg)
    if CHAT_MODEL:
        CHAT_TOK, CHAT_REV = BACKEND_OBJ.tokenizer, "mock"
elif USE_VLLM:
    BACKEND_OBJ = L.VLLMBackend(MODEL_PATH, tensor_parallel_size=2 if N_GPU >= 2 else 1, max_model_len=WINDOW,
                                read=READ_MODE)
else:
    BACKEND_NOTE["switched_to_hf"] = BACKEND != "hf"
    BACKEND_OBJ = L.HFBackend(MODEL_PATH)
TIMINGS["model_load"] = time.time() - t0
BACKEND_NOTE["used"] = BACKEND_OBJ.name
TOK = BACKEND_OBJ.tokenizer
SYSTEM_TEXT = (L.template_system_text(TOK) if MODEL_KIND == "instruct" else
               L.template_system_text(CHAT_TOK) if CHAT_TOK is not None else None)   # base: the chat format's text
print("backend:", BACKEND_OBJ.name, "| read:", BACKEND_OBJ.read, "|", json.dumps(BACKEND_OBJ.info(), default=str)[:500])
print("chat template adds its own system text:", repr(SYSTEM_TEXT))

# %% [markdown]
# ## Rendered prompts and their lengths
# For a base model in the chat format, the base tokenizer must give exactly the token ids of the instruction-tuned
# tokenizer for every rendered prompt; otherwise the run stops.

# %%
t0 = time.time()
PROMPTS_DF["rendered"] = [W.render_v4(t, MODEL_KIND, s, TOK, CHAT_TOK) for t, s in
                          zip(PROMPTS_DF["text"], PROMPTS_DF["shot_set"])]
PROMPTS_DF["n_tokens"] = [len(TOK.encode(t, add_special_tokens=False)) for t in PROMPTS_DF["rendered"]]
if CHAT_TOK is not None and not MOCK:
    CHECKS["chat_token_identity"] = W.same_token_ids(PROMPTS_DF.loc[PROMPTS_DF["shot_set"] == "Achat", "rendered"],
                                                     CHAT_TOK, TOK)
    print("chat format, token ids identical to the instruction-tuned tokenizer:", CHECKS["chat_token_identity"])
    if not CHECKS["chat_token_identity"]["identical"]:
        raise RuntimeError("the base tokenizer does not give the same token ids as the instruction-tuned tokenizer")
TIMINGS["render"] = time.time() - t0
print("prompt tokens: median", int(PROMPTS_DF["n_tokens"].median()), "| max", int(PROMPTS_DF["n_tokens"].max()),
      "| window", WINDOW)
if PROMPTS_DF["n_tokens"].max() >= WINDOW - 1:
    raise RuntimeError("a prompt is longer than the model window")
for _s in PROMPTS_DF["shot_set"].unique():
    print(f"\nEnd of one rendered prompt, set {_s} (what the model sees last):\n...",
          PROMPTS_DF.loc[PROMPTS_DF["shot_set"] == _s, "rendered"].iloc[0][-250:])

# %% [markdown]
# ## Score every test prompt: the answer-word log-probabilities after "Answer:"
# "Yes"/"No" or "True"/"False" (stored in the lp_yes / lp_no columns, with the column answer_words saying which). Saved
# after every chunk. If an earlier run of this model stopped part-way and its output is attached as an input, the scored
# prompts are picked up from there.

# %%
CKPT_NAME = f"v4_checkpoint_{MODEL_TAG}.jsonl"
CKPT = os.path.join(OUT_DIR, CKPT_NAME)
_systext = hashlib.sha256(str(SYSTEM_TEXT).encode()).hexdigest()[:12]
SIG = (f"{MODEL_ID}|{MODEL_REV}|{MODEL_KIND}|{BACKEND_OBJ.name}|{BACKEND_OBJ.read}|{L.CODE_VERSION}|{V.V3_VERSION}|"
       f"{W.V4_VERSION}|{WINDOW}|{_systext}|{CHAT_REV}")
done = {}


def _read_ckpt(path, store, valid_key):
    n = 0
    for ln in open(path):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("sig") == SIG and valid_key(d):
            store[d["key"] if "mode" not in d else d["gkey"]] = d
            n += 1
    return n


_okrec = lambda d: d.get("key") in _byk and d.get("text_sha") == _byk[d["key"]]["text_sha"]
for root in EXTRA_ROOTS:
    for dirpath, _, files in os.walk(root):
        if CKPT_NAME in files and os.path.join(dirpath, CKPT_NAME) != CKPT:
            print(f"resuming from {os.path.join(dirpath, CKPT_NAME)}: "
                  f"{_read_ckpt(os.path.join(dirpath, CKPT_NAME), done, _okrec)} records")
if os.path.exists(CKPT):
    _read_ckpt(CKPT, done, _okrec)
with open(CKPT, "w") as f:                      # rewrite the local checkpoint with everything recovered so far
    for d in done.values():
        f.write(json.dumps(d, default=float) + "\n")
todo = PROMPTS_DF[~PROMPTS_DF["key"].isin(done)]
print(f"{len(done)} already scored, {len(todo)} to go")
CHUNK = 512 if BACKEND_OBJ.name == "vllm" else 64
TIME_META = os.path.join(OUT_DIR, f"v4_timing_{MODEL_TAG}.json")
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {}
_tm = _tm if _tm.get("sig") == SIG else {"sig": SIG, "seconds": 0.0, "scored": 0}
_stop_at = int(os.environ.get("LER_TEST_STOP_AT_CHUNK", 0)) if MOCK else 0    # offline test of the resume
t0 = time.time()
n_scored, n_invalid = 0, 0
_aw = todo["framing"].map(lambda f: "/".join(W.answer_words(f)))
_parts = [g.iloc[i:i + CHUNK] for _, g in todo.groupby(_aw, sort=True) for i in range(0, len(g), CHUNK)]  # one word pair per chunk
with open(CKPT, "a") as fout:
    for j, part in enumerate(_parts):
        if _stop_at and j == _stop_at:
            raise RuntimeError(f"OFFLINE TEST: stopped at chunk {j}")
        res = W.read_v4(BACKEND_OBJ, part["rendered"].tolist(), part["framing"].iloc[0])
        for (_, r), rr in zip(part.iterrows(), res):
            rec = {k: r[k] for k in ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir", "xgb_p",
                                     "n_tokens", "text_sha")}
            rec["card_dir"] = None if pd.isna(rec["card_dir"]) else int(rec["card_dir"])
            rec = W.finish_record_v4(rec, rr, r["framing"])
            rec["sig"] = SIG
            n_invalid += rec["valid"] == 0
            done[rec["key"]] = rec
            fout.write(json.dumps(rec, default=float) + "\n")
        fout.flush()
        n_scored += len(part)
        if n_scored >= 256 and n_invalid > 0.2 * n_scored:
            raise RuntimeError(f"{n_invalid} of the first {n_scored} readings are invalid (no usable answer-word "
                               "probability). Something is wrong with the model or backend; stopping to save GPU time. "
                               "Under PREREGISTRATION_V4.md this model counts as 'cannot be run'.")
        el = time.time() - t0
        json.dump({"sig": SIG, "seconds": _tm["seconds"] + el, "scored": _tm["scored"] + n_scored}, open(TIME_META, "w"))
        if j == len(_parts) - 1 or j % max(1, len(_parts) // 10) == 0:
            print(f"  {n_scored}/{len(todo)} scored | {el / 60:.1f} min | {el / n_scored:.3f} s/prompt")
TIMINGS["scoring"] = time.time() - t0
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {"seconds": 0.0, "scored": 0}
RES = pd.DataFrame([done[k] for k in PROMPTS_DF["key"]])
for c in ("y", "pred_alarm", "pred_yes", "correct", "valid", "n_imputed"):
    RES[c] = RES[c].astype(int)
print("scored:", len(RES), "| total scoring time across runs:", round(_tm.get("seconds", 0) / 3600, 2), "h")

# %% [markdown]
# ## Generated answers (instruction-tuned models; Section 5.7)
# Greedy decoding for 50 test patients per class per dataset (seed 2031), the registered prompts (C0, C2, C4 × both
# framings), in two modes: after the "Answer:" prefill (at most 8 new tokens) and through the chat template without it
# (at most 64 new tokens). The answer is read from the text by the registered rule. No test uses these answers.

# %%
GEN = pd.DataFrame()
if MODEL_KIND == "instruct":
    t0 = time.time()
    GEN_NAME = f"v4_generated_{MODEL_TAG}.jsonl"
    GEN_CKPT = os.path.join(OUT_DIR, GEN_NAME)
    gitems = []
    for ds, b in BUNDLES.items():
        pts = set(W.greedy_patients(b))
        for rid in [r for r in (b.ids[:_max_p] if _max_p else b.ids) if r in pts]:
            for cond, fr, shot in W.GREEDY_GRID:
                k = f"{ds}|{rid}|{cond}|{fr}|{shot}"
                for mode in ("prefill", "free"):
                    gitems.append({"gkey": f"{k}|{mode}", "key": k, "mode": mode})
    gdone = {}
    _okgen = lambda d: d.get("key") in _byk
    for root in EXTRA_ROOTS:
        for dirpath, _, files in os.walk(root):
            if GEN_NAME in files and os.path.join(dirpath, GEN_NAME) != GEN_CKPT:
                _read_ckpt(os.path.join(dirpath, GEN_NAME), gdone, _okgen)
    if os.path.exists(GEN_CKPT):
        _read_ckpt(GEN_CKPT, gdone, _okgen)
    with open(GEN_CKPT, "w") as f:
        for d in gdone.values():
            f.write(json.dumps(d) + "\n")
    gtodo = [g for g in gitems if g["gkey"] not in gdone]
    print(f"generated answers: {len(gdone)} done, {len(gtodo)} to go")
    _pk = PROMPTS_DF.set_index("key")
    with open(GEN_CKPT, "a") as fout:
        for mode in ("prefill", "free"):
            items = [g for g in gtodo if g["mode"] == mode]
            for i in range(0, len(items), 256):
                part = items[i:i + 256]
                ps = [_pk.at[g["key"], "rendered"] if mode == "prefill" else W.render_free(_pk.at[g["key"], "text"], TOK)
                      for g in part]
                outs = W.generate(BACKEND_OBJ, ps, W.GREEDY_MAX_TOKENS[mode])
                for g, o in zip(part, outs):
                    d = dict(g, text=o, parsed=W.parse_generated(o), sig=SIG)
                    gdone[g["gkey"]] = d
                    fout.write(json.dumps(d) + "\n")
                fout.flush()
    GEN = pd.DataFrame([gdone[g["gkey"]] for g in gitems])[["key", "mode", "text", "parsed"]]
    TIMINGS["generation"] = time.time() - t0
    for mode, g in GEN.groupby("mode"):
        print(f"  {mode}: {len(g)} answers, parsed {g['parsed'].notna().mean():.3f}; first three: "
              f"{[t[:60] for t in g['text'].head(3)]}")

# %% [markdown]
# ## The validity checks
# The registered rules (Section 8) decide which cells and formats enter the analysis. Nothing else is analysed here.

# %%
rows = []
for ds in L.DATASETS:
    r = RES[RES["dataset"] == ds]
    v = W.validity_v4(r)
    row = {"dataset": ds, "n_prompts": len(r), "n_invalid": int((r["valid"] == 0).sum()),
           "n_imputed": int((r["n_imputed"] > 0).sum()), "tokens_median": int(r["n_tokens"].median()),
           "tokens_max": int(r["n_tokens"].max()), "mass_alarm": v["split"]["alarm_median_mass"],
           "mass_reversed": v["split"]["reversed_median_mass"], "c0_accuracy": v["split"]["c0_accuracy"],
           "valid_split": v["split"]["passes"]}
    for k, x in v["framings"].items():
        row[f"mass {k}"] = x.get("median_mass", x.get("alarm_median_mass"))
        row[f"valid {k}"] = x["passes"]
    rows.append(row)
VALIDITY = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(VALIDITY.round(3).T.to_string())
CHECKS["validity"] = VALIDITY.to_dict(orient="records")
CHECKS["read_mode"] = BACKEND_OBJ.read

# %% [markdown]
# ## Save everything and make one zip to download

# %%
import zipfile
REC_NAME = f"records_{MODEL_TAG}.csv.gz"
RES.drop(columns=["sig"], errors="ignore").to_csv(os.path.join(OUT_DIR, REC_NAME), index=False)
GEN_OUT = f"generated_{MODEL_TAG}.csv.gz"
if len(GEN):
    GEN.to_csv(os.path.join(OUT_DIR, GEN_OUT), index=False)
_pk = PROMPTS_DF.set_index("key")
with open(os.path.join(OUT_DIR, f"example_prompts_{MODEL_TAG}.md"), "w", encoding="utf-8") as f:
    for ds, b in BUNDLES.items():
        rid = b.ids[0]
        for cond, fr, shot in GRID:
            k = f"{ds}|{rid}|{cond}|{fr}|{shot}"
            if k in _pk.index and cond == "C2":
                f.write(f"## {ds} | patient {rid} | {cond} | {fr} | set {shot}\n\n```\n{_pk.at[k, 'rendered']}\n```\n\n")
RUN_LOG = {"study": "v4", "grids": list(GRIDS), "registered_revision": MODEL_REVISION, "decisions": DECISIONS,
           "protocol": PROTOCOL, "model_id": MODEL_ID, "model_tag": MODEL_TAG, "model_kind": MODEL_KIND,
           "model_family": MODEL_FAMILY, "model_revision": MODEL_REV, "chat_model": CHAT_MODEL, "chat_revision": CHAT_REV,
           "window": WINDOW, "backend": BACKEND_NOTE, "backend_info": BACKEND_OBJ.info(), "read_mode": BACKEND_OBJ.read,
           "template_system_text": SYSTEM_TEXT, "env": ENV, "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()},
           "scoring_seconds_all_runs": _tm.get("seconds"), "code_version": L.CODE_VERSION, "v3_version": V.V3_VERSION,
           "v4_version": W.V4_VERSION, "code_sha256": CODE_SHA,
           "library_versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "scipy", "xgboost")},
           "fingerprints": {ds: b.info["fingerprint"] for ds, b in BUNDLES.items()}, "dev_fingerprint_check": FP_CHECK,
           "data": DATA_INFO, "bundles": {ds: b.info for ds, b in BUNDLES.items()}, "checks": CHECKS,
           "n_records": len(RES), "n_generated": len(GEN), "mock": MOCK, "local_model": bool(LOCAL_MODEL)}
with open(os.path.join(OUT_DIR, f"run_log_{MODEL_TAG}.json"), "w") as f:
    json.dump(RUN_LOG, f, indent=1, default=str)
ZIP = os.path.join(WORK, f"v4_{MODEL_TAG}.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in (REC_NAME, GEN_OUT, f"run_log_{MODEL_TAG}.json", f"example_prompts_{MODEL_TAG}.md"):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            z.write(os.path.join(OUT_DIR, name), name)
    for name in ("ler_main.py", "ler_v3.py", "ler_v4.py"):
        z.write(os.path.join(SRC_DIR, name), name)
print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB)")
print(f"Total time: {(time.time() - T_START) / 60:.1f} min")
print(f"\nDONE ({MODEL_ID}). Download v4_{MODEL_TAG}.zip from the Output tab.")
