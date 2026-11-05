# %% [markdown]
# # Fifth study: one run per model (pre-registered, `prereg-v5`)
#
# **What this notebook does.** It runs one model of the fifth study (`PREREGISTRATION_V5.md`, registered on OSF and
# released as `prereg-v5`) on the fresh patients: PhysioNet 2012 set C and the PhysioNet 2019 patients that no earlier
# study used. Every registered prompt of the model's grids (the Yes/No, paraphrased, True/False and A/B pairs with no
# evidence, the placebo card, the true card and the flipped card; the numbers card at three scales; the risk probability),
# then greedy generated answers for the generation subsample. It tests no hypothesis; the analysis notebook does that
# after all runs.
#
# **Before the first run:** `PREREGISTRATION_V5.md` with the result of the engineering check must be committed, the
# `prereg-v5` release created and the OSF registration submitted. The notebook checks the release and stops if the
# protocol is missing.
#
# **How to run on Kaggle (once per model)**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**. Add-ons → Secrets: `HF_TOKEN`.
# 3. In the first code cell set `MODEL_ID` to the next model in the registered order.
# 4. **Save Version → Save & Run All (Commit).** Expected time: about 4 to 8 hours (Granite 3.3 and Yi-1.5 are the
#    slowest). Kaggle allows 12 hours per session.
# 5. Download `v5_<model>.zip` from the Output tab. Keep all of them.
#
# **If a run stops part-way:** add the failed version's output as an input (Add Input → Your Work → this notebook) and
# run again. The scoring and generation cells pick up the saved checkpoints. For Granite 3.3, whose chat template adds
# the current date, a run that is resumed on another day starts the scoring again, so that one run never mixes dates.

# %%
# ============ settings (the only cell you change) ============
MODEL_ID = "Qwen/Qwen3-8B"
# The registered models of the fifth study, one per run, in this order (PREREGISTRATION_V5.md, Sections 4 and 9):
#   "Qwen/Qwen3-8B"  "tiiuae/Falcon3-7B-Instruct"  "01-ai/Yi-1.5-9B-Chat-16K"  "ibm-granite/granite-3.3-8b-instruct"
#   "meta-llama/Llama-3.1-8B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"  "allenai/Olmo-3-7B-Instruct"
BACKEND = "vllm"            # vLLM only (a model that vLLM cannot run counts as "cannot be run")
READ_OVERRIDE = "topk"      # Deviation D5 of the first study, kept: the top 1,000 log-probabilities (the check still runs)

import os, sys, time, json, subprocess, platform, shutil
T_START = time.time()
MOCK = os.environ.get("LER_MOCK") == "1"            # CPU self-test with a fake model; never set on Kaggle
if os.environ.get("LER_MODEL_ID"):
    MODEL_ID = os.environ["LER_MODEL_ID"]
LOCAL_MODEL = None


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
print("MODEL_ID =", MODEL_ID, "| BACKEND =", BACKEND, "| MOCK =", MOCK)
if not MOCK and not ENV.get("gpus"):
    raise RuntimeError("No GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again.")

# %% [markdown]
# ## Paths and access token
# The model files go to the first location where a test file can be written (the first engineering check lost one
# model to a read-only /tmp).

# %%
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC_DIR = os.path.abspath(".")
DATA_DIR = os.environ.get("LER_DATA_DIR", "/tmp/ler_data" if os.path.isdir("/kaggle/working") else
                          os.path.join(WORK, "ler_data"))
EXTRA_ROOTS = (["/kaggle/input"] if os.path.isdir("/kaggle/input") else []) + \
              [d for d in os.environ.get("LER_EXTRA_ROOTS", "").split(os.pathsep) if d]


def _cache_writable(d):
    try:
        os.makedirs(d, exist_ok=True)
        _p = os.path.join(d, ".ler_write_probe")
        with open(_p, "wb") as _f:
            _f.write(b"\0" * 1048576)
        os.remove(_p)
        return True
    except OSError as _e:
        print(f"model cache {d} is not writable: {_e!r}")
        return False


CACHE_CANDIDATES = ["/tmp/hf_cache", os.path.expanduser("~/.cache/ler_hf")]
CACHE = next((d for d in CACHE_CANDIDATES if _cache_writable(d)), CACHE_CANDIDATES[0])
os.makedirs(DATA_DIR, exist_ok=True)
try:
    _u = shutil.disk_usage(CACHE)
    DISK = {"model_cache": CACHE, "free_gb": round(_u.free / 1e9, 1), "total_gb": round(_u.total / 1e9, 1)}
except OSError as _e:
    DISK = {"model_cache": CACHE, "error": repr(_e)}
print("disk:", DISK)
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
if MODEL_ID.startswith("meta-llama/") and not (HF_TOKEN or MOCK):
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


VLLM_PIN = "0.26.0"     # the dev-phase version; the same in every run of all four earlier studies


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
# `ler_main.py` (main-2.0), `ler_v3.py` (v3-1.1) and `ler_v4.py` (v4-1.2) are the registered code of the earlier
# studies, unchanged. `ler_v5.py` adds the placebo card, the scaled numbers cards, the risk probability, the fresh
# patients, the models and decisions of the fifth study, the protocol check, and the fifth study's analysis.

# %% [writefile ler_main.py]

# %% [writefile ler_v3.py]

# %% [writefile ler_v4.py]

# %% [writefile ler_v5.py]

# %% [markdown]
# ## Check the code and the protocol
# `ler_main.py`, `ler_v3.py` and `ler_v4.py` must be byte-identical to the registered versions, and the SHA-256 of
# `ler_v5.py` must be listed in `PREREGISTRATION_V5.md`. The notebook downloads the five protocol files from the
# `prereg-v5` release: every fixed prompt string, model id, pinned revision, hash and fingerprint in the code must appear
# there word for word. Otherwise the run stops.

# %%
import importlib
import hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
for _m in ("ler_v5", "ler_v4", "ler_v3", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]          # cells re-run by hand -> use the freshly written files
import ler_main as L
import ler_v3 as V
import ler_v4 as W4
import ler_v5 as F
if MOCK:
    L.TEST_PER_CLASS = int(os.environ.get("LER_TEST_PER_CLASS", L.TEST_PER_CLASS))    # small fake data only

CODE_SHA = {n: L.file_sha256(os.path.join(SRC_DIR, n)) for n in ("ler_main.py", "ler_v3.py", "ler_v4.py", "ler_v5.py")}
if (CODE_SHA["ler_main.py"], CODE_SHA["ler_v3.py"], CODE_SHA["ler_v4.py"]) != (F.MAIN_SHA256, F.V3_SHA256, F.V4_SHA256):
    raise RuntimeError("ler_main.py, ler_v3.py or ler_v4.py is not the registered version")
if MODEL_ID not in F.MODELS_V5:
    raise RuntimeError(f"{MODEL_ID} is not one of the registered models of the fifth study: {list(F.MODELS_V5)}")
DECISIONS = F.ENGCHECK_DECISIONS
if DECISIONS is None and MOCK:          # offline tests: decisions given by the test, or every grid
    DECISIONS = json.loads(os.environ.get("LER_MOCK_DECISIONS", "null")) or {"not_run": [], "grids_not_run": {},
                                                                             "gen_per_class": F.GEN_PER_CLASS}
if DECISIONS is None:
    raise RuntimeError("The result of the engineering check is not recorded in ler_v5.py (ENGCHECK_DECISIONS). It must be "
                       "recorded in PREREGISTRATION_V5.md and in the code before the runs.")
_M = F.MODELS_V5[MODEL_ID]
MODEL_FAMILY, MODEL_TAG, MODEL_REVISION = _M["family"], _M["tag"], _M["revision"]
GRIDS = F.model_grids(MODEL_ID, DECISIONS)
GEN_PER_CLASS = int(DECISIONS["gen_per_class"])
OUT_DIR = os.path.join(WORK, f"v5_outputs_{MODEL_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"model {MODEL_ID}: family {MODEL_FAMILY}, grids {GRIDS}, revision {MODEL_REVISION}, generation subsample "
      f"{GEN_PER_CLASS} per class, outputs in {OUT_DIR}")
print("code:", {k: v[:16] for k, v in CODE_SHA.items()}, "| ler_v5", F.V5_VERSION)
print("decisions of the engineering check:", json.dumps(DECISIONS))
if MODEL_ID.startswith("meta-llama/") and not MOCK:
    _bad = F.access_problems([MODEL_ID], HF_TOKEN, os.path.join(CACHE, "access_check"))
    if _bad:
        raise RuntimeError("Hugging Face refuses the download of " + "; ".join(f"{m} ({p})" for m, p in _bad.items())
                           + ". " + F.ACCESS_HELP)
    print("access to the gated model confirmed")

PROTOCOL = {"tag": F.PROTOCOL_V5_TAG}
_ENV_PATH = {"PREREGISTRATION.md": "LER_PROTOCOL_PATH", "PREREGISTRATION_V2.md": "LER_PROTOCOL_V2_PATH",
             "PREREGISTRATION_V3.md": "LER_PROTOCOL_V3_PATH", "PREREGISTRATION_V4.md": "LER_PROTOCOL_V4_PATH",
             "PREREGISTRATION_V5.md": "LER_PROTOCOL_V5_PATH"}
_URLS = dict(F.PROTOCOL_AT_V5_URLS, **{"PREREGISTRATION_V5.md": F.PROTOCOL_V5_URL})
_T = {}
try:
    for _name, _url in _URLS.items():
        _env = _ENV_PATH[_name]
        _T[_name] = (open(os.environ[_env], encoding="utf-8").read() if os.environ.get(_env)
                     else L.http_get(_url).decode("utf-8"))
except Exception as e:
    raise RuntimeError(f"Could not download the protocol files at the tag {F.PROTOCOL_V5_TAG} ({e!r}). Commit "
                       "PREREGISTRATION_V5.md and create the prereg-v5 release on GitHub before the runs.") from e
PROTOCOL["sha256"] = {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in _T.items()}
_missing = {"v1": L.check_against_protocol(_T["PREREGISTRATION.md"]) + L.check_deviations(_T["PREREGISTRATION.md"]),
            "v2": L.check_against_protocol_v2(_T["PREREGISTRATION_V2.md"]),
            "v3": V.check_against_protocol_v3(_T["PREREGISTRATION_V3.md"]),
            "v4": W4.check_against_protocol_v4(_T["PREREGISTRATION_V4.md"]),
            "v5": F.check_against_protocol_v5(_T["PREREGISTRATION_V5.md"])}
if CODE_SHA["ler_v5.py"] not in _T["PREREGISTRATION_V5.md"]:
    _missing["v5"].append(f"the SHA-256 of ler_v5.py ({CODE_SHA['ler_v5.py']})")
PROTOCOL["strings_missing"] = _missing
for _k in ("v1", "v2", "v3", "v4"):
    if _missing[_k]:
        raise RuntimeError(f"The protocol file of study {_k} at prereg-v5 lacks these strings:\n" + "\n".join(_missing[_k]))
if _missing["v5"] and not MOCK:
    raise RuntimeError("PREREGISTRATION_V5.md at prereg-v5 lacks these strings, model ids, revisions, hashes or "
                       "sections:\n" + "\n".join(_missing["v5"]))
if _missing["v5"]:
    print("OFFLINE TEST: PREREGISTRATION_V5.md lacks", _missing["v5"], "(a real run would stop here)")
print(f"Protocol check passed ({F.PROTOCOL_V5_TAG}, PREREGISTRATION_V5.md sha256 "
      f"{PROTOCOL['sha256']['PREREGISTRATION_V5.md'][:16]}): prompt strings, models, revisions and hashes match.")

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
print("pilot patients excluded:", len(PILOT_IDS), "|", PROTOCOL["pilot_ids_source"])

# %% [markdown]
# ## Data, splits, XGBoost and cards, checked against the earlier studies; the fresh patients
# PhysioNet 2012 (ODC-By 1.0, sets A and B for the model, set C for the fresh patients) and PhysioNet 2019 (CC BY 4.0).
# The splits, features, dev cards, solved examples and the earlier test samples must be identical to the earlier runs
# (fingerprints). The fresh samples are then built and must match the fingerprints recorded at registration.

# %%
t0 = time.time()
BUNDLES, DATA_INFO, FP_CHECK, FRESH = {}, {}, {}, {}
for ds in F.DATASETS_V5:
    t1 = time.time()
    print(f"[{ds}]")
    if ds == "physionet2019":
        df_, feats_, cats_, info_ = L.load_physionet2019(DATA_DIR, EXTRA_ROOTS, log=print,
                                                          workers=max(1, min(4, os.cpu_count() or 1)))
    else:
        df_, feats_, cats_, info_ = L.LOADERS[ds](DATA_DIR, EXTRA_ROOTS, log=print)
    info_["licence"] = L.SPEC[ds]["licence"]
    b = L.Bundle(ds, df_, feats_, cats_, info_, pilot_ids=PILOT_IDS if ds == "physionet2012" else (), sample="test")
    FP_CHECK[ds] = {k: b.info["fingerprint"][k] == v for k, v in L.DEV_RUN["fingerprints"][ds].items()}
    FP_CHECK[ds].update({k: b.info["fingerprint"][k] == v for k, v in F.REGISTERED_TEST_FP[ds].items()})
    print("   dev and earlier test fingerprints match:", FP_CHECK[ds])
    set_c = F.load_p12_set_c(DATA_DIR, EXTRA_ROOTS, log=print) if ds == "physionet2012" else None
    FRESH[ds] = F.build_fresh(b, set_c, PILOT_IDS, log=print)
    info_["seconds"] = round(time.time() - t1, 1)
    DATA_INFO[ds] = {k: v for k, v in info_.items() if k != "id_mapping_tables"}
    BUNDLES[ds] = b
TIMINGS["data"] = time.time() - t0
if not MOCK:
    bad = {ds: [k for k, ok in v.items() if not ok] for ds, v in FP_CHECK.items() if not all(v.values())}
    if bad:
        raise RuntimeError(f"The rebuilt data differ from the earlier runs: {bad}. Send me this output.")
    if F.FRESH_FP is None or any(FRESH[ds]["fingerprint"] != F.FRESH_FP[ds] for ds in F.DATASETS_V5):
        raise RuntimeError("The fresh samples differ from those recorded at registration (FRESH_FP): "
                           f"{ {ds: f['fingerprint'] for ds, f in FRESH.items()} }. Send me this output.")
print("All fingerprints match." + (" (offline test: not checked)" if MOCK else ""))
print(json.dumps({ds: {k: v for k, v in f.items() if k != "fingerprint"} for ds, f in FRESH.items()}))

# %% [markdown]
# ## Build every prompt and check them
# Every prompt of the model's grids (Section 5.5). Each prompt is checked against the registered prompt of the same
# patient: it may differ only in the evidence block, as Section 5 of the protocol says.

# %%
RECORDS = []
GRID = F.grid_v5(GRIDS)
_max_p = int(os.environ.get("LER_TEST_MAX_PATIENTS", 0)) if MOCK else 0      # offline tests only
for ds, b in BUNDLES.items():
    for rid in (b.ids[:_max_p] if _max_p else b.ids):
        for cond, framing, shot in GRID:
            text, cdir = F.prompt_v5(b, rid, cond, framing, shot)
            F.check_prompt_v5(b, rid, cond, framing, shot, text, cdir)
            RECORDS.append({"dataset": ds, "id": rid, "cond": cond, "framing": framing, "shot_set": shot,
                            "y": b.y[rid], "card_dir": cdir, "xgb_p": float(b.xgb_p[rid]),
                            "key": f"{ds}|{rid}|{cond}|{framing}|{shot}", "text": text,
                            "text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]})
PROMPTS_DF = pd.DataFrame(RECORDS)
_byk = {r["key"]: r for r in RECORDS}
_ev = lambda t: [blk for blk in t.split("\n\n") if blk.startswith("Evidence from a machine-learning model")]
_cp_block = {}
for r in RECORDS:
    if r["cond"] in ("C4", "CQ4"):
        twin = _byk[r["key"].replace(f"|{r['cond']}|", f"|{r['cond'][:-1]}2|")]
        if twin["card_dir"] is None:
            assert r["card_dir"] is None
        else:
            assert r["card_dir"] == 1 - twin["card_dir"], "the second card does not point opposite to the first"
    if r["cond"] == "CP":       # the placebo block is the same in every framing of a patient
        blk = _ev(r["text"])
        assert len(blk) == 1
        assert _cp_block.setdefault((r["dataset"], r["id"]), blk[0]) == blk[0], "placebo block differs between framings"
CHECKS = {"prompts_checked": len(RECORDS), "grids": list(GRIDS),
          "prompts_per_dataset": PROMPTS_DF.groupby("dataset").size().to_dict(),
          "prompts_per_framing_set_cond": {f"{k[0]}/{k[1]}/{k[2]}": int(v) for k, v in
                                           PROMPTS_DF.groupby(["framing", "shot_set", "cond"]).size().items()},
          "prob_exactly_50": {ds: int(sum(F.prob_percent(b, r) == 50 for r in b.ids)) for ds, b in BUNDLES.items()},
          "prompt_set_sha256": hashlib.sha256("".join(sorted(PROMPTS_DF["key"] + PROMPTS_DF["text_sha"])).encode())
          .hexdigest()}
print(f"{len(RECORDS)} prompts built and checked: {CHECKS['prompts_per_dataset']}")
_rid = BUNDLES[F.DATASETS_V5[0]].ids[0]
for _c, _fr, _shot in (("CP", "alarm", "A"), ("CQ2", "alarm", "A"), ("C2", "alarm", "Anum05"), ("C2", "choice_alarm_A", "A")):
    _show = f"{F.DATASETS_V5[0]}|{_rid}|{_c}|{_fr}|{_shot}"
    if _show in _byk:
        print(f"\nEnd of one prompt ({_show}):\n...")
        print(_byk[_show]["text"][-800:])

# DEV prompts for the exact-reading check of Deviation D2 (recorded; the run reads the top 1,000 as registered)
_dev = {"yesno": [], "tf": []}
for ds, b in BUNDLES.items():
    b.use_sample("dev")
    for rid in b.ids[:6]:
        for cond in ("C0", "C2", "C4"):
            _dev["yesno"].append(b.prompt(rid, cond, "alarm", "A")[0])
            if "tf" in GRIDS and rid in b.ids[:3]:
                _dev["tf"].append(W4.prompt_v4(b, rid, cond, "tf_alarm", "A")[0])
    b.use_sample("fresh")

# %% [markdown]
# ## Model: download, vLLM smoke test with the exact-reading check, load
# The smoke test runs in a separate process (a failed engine start cannot leave this notebook holding GPU memory). The
# model window is 16,384 tokens or the model's own window if shorter.

# %%
t0 = time.time()
if MOCK:
    MODEL_PATH, MODEL_REV = None, "mock"
else:
    MODEL_PATH, MODEL_REV = L.download_model(MODEL_ID, os.path.join(CACHE, "hub"), MODEL_REVISION, HF_TOKEN)
    print("model revision:", MODEL_REV)
    if MODEL_REV != MODEL_REVISION:
        raise RuntimeError(f"downloaded revision {MODEL_REV} differs from the registered {MODEL_REVISION}")
TIMINGS["model_download"] = time.time() - t0
WINDOW = V.model_window(MODEL_PATH) if MODEL_PATH else L.MAX_MODEL_LEN
print("model window (max_model_len):", WINDOW)

READ_MODE = "exact"
SMOKE_PROMPTS_PATH = os.path.join(OUT_DIR, "smoke_prompts.json")
if USE_VLLM:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    json.dump({"yesno": [L.render(t, "instruct", _tok) for t in _dev["yesno"]],
               "tf": [L.render(t, "instruct", _tok) for t in _dev["tf"]]}, open(SMOKE_PROMPTS_PATH, "w"))
    t0 = time.time()
    try:
        code, out = sh([sys.executable, "-m", "ler_v4", "smoke", MODEL_PATH, "instruct", SMOKE_PROMPTS_PATH],
                       timeout=1800, cwd=SRC_DIR, env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired:
        code, out = 1, "smoke test timed out"
    SMOKE = W4.parse_result(out, "SMOKE_V4_RESULT") if code == 0 else None
    TIMINGS["smoke_test"] = time.time() - t0
    if SMOKE:
        BACKEND_NOTE["vllm_smoke_test"] = SMOKE
        READ_MODE = SMOKE["read"]
        print("smoke test ok | read mode from the check:", READ_MODE, "| exact check:", json.dumps(SMOKE["exact_check"]))
        if READ_OVERRIDE and READ_OVERRIDE != READ_MODE:
            BACKEND_NOTE["read_override"] = f"{READ_MODE} -> {READ_OVERRIDE} (Deviation D5)"
            READ_MODE = READ_OVERRIDE
        print("read mode used:", READ_MODE, "(Deviation D5 of the first study: the same reading in every run)")
    else:
        BACKEND_NOTE["vllm_smoke_test"] = out[-3000:]
        print("vLLM smoke test FAILED\n", out[-3000:])
        raise RuntimeError("vLLM smoke test failed: under PREREGISTRATION_V5.md this model 'cannot be run'")

t0 = time.time()
N_GPU = len(ENV.get("gpus") or [])
if MOCK:
    _mock_cfg = json.loads(os.environ.get("LER_MOCK_CFG", "{}"))
    _truth = {"\n" + b._block(b.values[r]): b.y[r] for b in BUNDLES.values() for r in b.ids}
    BACKEND_OBJ = F.MockV5(truth=_truth, **_mock_cfg)
else:
    BACKEND_OBJ = L.VLLMBackend(MODEL_PATH, tensor_parallel_size=2 if N_GPU >= 2 else 1, max_model_len=WINDOW,
                                read=READ_MODE)
TIMINGS["model_load"] = time.time() - t0
BACKEND_NOTE["used"] = BACKEND_OBJ.name
TOK = BACKEND_OBJ.tokenizer
SYSTEM_TEXT = L.template_system_text(TOK)
print("backend:", BACKEND_OBJ.name, "| read:", BACKEND_OBJ.read, "|", json.dumps(BACKEND_OBJ.info(), default=str)[:500])
print("chat template adds its own system text:", repr(SYSTEM_TEXT))

# %% [markdown]
# ## Rendered prompts and their lengths

# %%
t0 = time.time()
PROMPTS_DF["rendered"] = [F.render_v5(t, TOK) for t in PROMPTS_DF["text"]]
PROMPTS_DF["n_tokens"] = [len(TOK.encode(t, add_special_tokens=False)) for t in PROMPTS_DF["rendered"]]
TIMINGS["render"] = time.time() - t0
print("prompt tokens: median", int(PROMPTS_DF["n_tokens"].median()), "| max", int(PROMPTS_DF["n_tokens"].max()),
      "| window", WINDOW)
if PROMPTS_DF["n_tokens"].max() >= WINDOW - 1:
    raise RuntimeError("a prompt is longer than the model window")
print("\nEnd of one rendered prompt (what the model sees last):\n...", PROMPTS_DF["rendered"].iloc[0][-250:])

# %% [markdown]
# ## Score every prompt: the answer-word log-probabilities after "Answer:"
# "Yes"/"No", "True"/"False" or "A"/"B" (stored in the lp_yes / lp_no columns, with the column answer_words saying
# which). Saved after every chunk. If an earlier run of this model stopped part-way and its output is attached as an
# input, the scored prompts are picked up from there.

# %%
CKPT_NAME = f"v5_checkpoint_{MODEL_TAG}.jsonl"
CKPT = os.path.join(OUT_DIR, CKPT_NAME)
_systext = hashlib.sha256(str(SYSTEM_TEXT).encode()).hexdigest()[:12]
SIG = (f"{MODEL_ID}|{MODEL_REV}|{BACKEND_OBJ.name}|{BACKEND_OBJ.read}|{L.CODE_VERSION}|{V.V3_VERSION}|{W4.V4_VERSION}|"
       f"{F.V5_VERSION}|{WINDOW}|{_systext}")
done = {}


def _read_ckpt(path, store, valid_key, keyname="key"):
    n = 0
    for ln in open(path):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("sig") == SIG and valid_key(d):
            store[d[keyname]] = d
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
TIME_META = os.path.join(OUT_DIR, f"v5_timing_{MODEL_TAG}.json")
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {}
_tm = _tm if _tm.get("sig") == SIG else {"sig": SIG, "seconds": 0.0, "scored": 0}
_stop_at = int(os.environ.get("LER_TEST_STOP_AT_CHUNK", 0)) if MOCK else 0    # offline test of the resume
t0 = time.time()
n_scored, n_invalid = 0, 0
_aw = todo["framing"].map(lambda f: "/".join(F.answer_words(f)))
_parts = [g.iloc[i:i + CHUNK] for _, g in todo.groupby(_aw, sort=True) for i in range(0, len(g), CHUNK)]  # one word pair per chunk
with open(CKPT, "a") as fout:
    for j, part in enumerate(_parts):
        if _stop_at and j == _stop_at:
            raise RuntimeError(f"OFFLINE TEST: stopped at chunk {j}")
        res = F.read_v5(BACKEND_OBJ, part["rendered"].tolist(), part["framing"].iloc[0])
        for (_, r), rr in zip(part.iterrows(), res):
            rec = {k: r[k] for k in ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir", "xgb_p",
                                     "n_tokens", "text_sha")}
            rec["card_dir"] = None if pd.isna(rec["card_dir"]) else int(rec["card_dir"])
            rec = F.finish_record_v5(rec, rr, r["framing"])
            rec["sig"] = SIG
            n_invalid += rec["valid"] == 0
            done[rec["key"]] = rec
            fout.write(json.dumps(rec, default=float) + "\n")
        fout.flush()
        n_scored += len(part)
        if n_scored >= 256 and n_invalid > 0.2 * n_scored:
            raise RuntimeError(f"{n_invalid} of the first {n_scored} readings are invalid (no usable answer-word "
                               "probability). Something is wrong with the model or backend; stopping to save GPU time. "
                               "Under PREREGISTRATION_V5.md this model counts as 'cannot be run'.")
        el = time.time() - t0
        json.dump({"sig": SIG, "seconds": _tm["seconds"] + el, "scored": _tm["scored"] + n_scored}, open(TIME_META, "w"))
        if j == len(_parts) - 1 or j % max(1, len(_parts) // 10) == 0:
            print(f"  {n_scored}/{len(todo)} scored | {el / 60:.1f} min | {el / n_scored:.3f} s/prompt", flush=True)
TIMINGS["scoring"] = time.time() - t0
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {"seconds": 0.0, "scored": 0}
RES = pd.DataFrame([done[k] for k in PROMPTS_DF["key"]])
for c in ("y", "pred_alarm", "pred_yes", "correct", "valid", "n_imputed"):
    RES[c] = RES[c].astype(int)
print("scored:", len(RES), "| total scoring time across runs:", round(_tm.get("seconds", 0) / 3600, 2), "h")

# %% [markdown]
# ## Generated answers (Section 5.6)
# Greedy decoding for the generation subsample (Section 3: per class and dataset, seed 2033), C0, C2 and C4 in both
# Yes/No framings, through the chat template without the "Answer:" prefill, at most 512 new tokens. The answer is read
# from the text by the registered rule.

# %%
t0 = time.time()
GEN_NAME = f"v5_generated_{MODEL_TAG}.jsonl"
GEN_CKPT = os.path.join(OUT_DIR, GEN_NAME)
gitems = []
for ds, b in BUNDLES.items():
    pts = set(F.gen_patients(b, GEN_PER_CLASS))
    for rid in [r for r in (b.ids[:_max_p] if _max_p else b.ids) if r in pts]:
        for cond, fr, shot in F.GEN_GRID:
            k = f"{ds}|{rid}|{cond}|{fr}|{shot}"
            gitems.append({"key": k, "dataset": ds, "id": rid, "cond": cond, "framing": fr, "y": b.y[rid],
                           "card_dir": _byk[k]["card_dir"]})
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
gtodo = [g for g in gitems if g["key"] not in gdone]
print(f"generated answers: {len(gdone)} done, {len(gtodo)} to go")
with open(GEN_CKPT, "a") as fout:
    for i in range(0, len(gtodo), 256):
        part = gtodo[i:i + 256]
        outs = F.generate(BACKEND_OBJ, [F.render_free(_byk[g["key"]]["text"], TOK) for g in part], F.GEN_MAX_TOKENS)
        for g, o in zip(part, outs):
            d = dict(g, text=o, parsed=F.parse_generated(o), sig=SIG)
            gdone[g["key"]] = d
            fout.write(json.dumps(d) + "\n")
        fout.flush()
GEN = pd.DataFrame([gdone[g["key"]] for g in gitems]).drop(columns=["sig"], errors="ignore")
TIMINGS["generation"] = time.time() - t0
print(f"  {len(GEN)} answers, parsed {GEN['parsed'].notna().mean():.3f}; first three: "
      f"{[t[:60] for t in GEN['text'].head(3)]}")

# %% [markdown]
# ## The validity checks
# The registered rules (Section 8) decide which cells and pairs enter the analysis. Nothing else is analysed here.

# %%
rows = []
for ds in F.DATASETS_V5:
    r = RES[RES["dataset"] == ds]
    v = F.validity_v5(r)
    row = {"dataset": ds, "n_prompts": len(r), "n_invalid": int((r["valid"] == 0).sum()),
           "n_imputed": int((r["n_imputed"] > 0).sum()), "tokens_median": int(r["n_tokens"].median()),
           "tokens_max": int(r["n_tokens"].max()), "mass_alarm": v["yn"]["alarm_median_mass"],
           "mass_reversed": v["yn"]["reversed_median_mass"], "c0_accuracy": v["yn"]["c0_accuracy"],
           "valid_yn": v["yn"]["passes"]}
    for k in ("pa", "tf", "ab", "dose", "prob"):
        if k in GRIDS:
            row[f"valid {k}"] = v[k]["passes"]
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
GEN.to_csv(os.path.join(OUT_DIR, GEN_OUT), index=False)
_pk = PROMPTS_DF.set_index("key")
with open(os.path.join(OUT_DIR, f"example_prompts_{MODEL_TAG}.md"), "w", encoding="utf-8") as f:
    for ds, b in BUNDLES.items():
        rid = b.ids[0]
        for cond, fr, shot in GRID:
            k = f"{ds}|{rid}|{cond}|{fr}|{shot}"
            if k in _pk.index and cond in ("CP", "C2", "CQ2"):
                f.write(f"## {ds} | patient {rid} | {cond} | {fr} | set {shot}\n\n```\n{_pk.at[k, 'rendered']}\n```\n\n")
RUN_LOG = {"study": "v5", "grids": list(GRIDS), "registered_revision": MODEL_REVISION, "decisions": DECISIONS,
           "protocol": PROTOCOL, "model_id": MODEL_ID, "model_tag": MODEL_TAG, "model_family": MODEL_FAMILY,
           "model_revision": MODEL_REV, "window": WINDOW, "backend": BACKEND_NOTE, "backend_info": BACKEND_OBJ.info(),
           "read_mode": BACKEND_OBJ.read, "template_system_text": SYSTEM_TEXT, "env": ENV, "disk": DISK,
           "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()}, "scoring_seconds_all_runs": _tm.get("seconds"),
           "code_version": L.CODE_VERSION, "v3_version": V.V3_VERSION, "v4_version": W4.V4_VERSION,
           "v5_version": F.V5_VERSION, "code_sha256": CODE_SHA,
           "library_versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "scipy", "xgboost")},
           "fingerprints": {ds: b.info["fingerprint"] for ds, b in BUNDLES.items()}, "fingerprint_check": FP_CHECK,
           "fresh": FRESH, "data": DATA_INFO, "checks": CHECKS, "gen_per_class": GEN_PER_CLASS,
           "n_records": len(RES), "n_generated": len(GEN), "mock": MOCK}
with open(os.path.join(OUT_DIR, f"run_log_{MODEL_TAG}.json"), "w") as f:
    json.dump(RUN_LOG, f, indent=1, default=str)
ZIP = os.path.join(WORK, f"v5_{MODEL_TAG}.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in (REC_NAME, GEN_OUT, f"run_log_{MODEL_TAG}.json", f"example_prompts_{MODEL_TAG}.md"):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            z.write(os.path.join(OUT_DIR, name), name)
    for name in ("ler_main.py", "ler_v3.py", "ler_v4.py", "ler_v5.py"):
        z.write(os.path.join(SRC_DIR, name), name)
print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB)")
print(f"Total time: {(time.time() - T_START) / 60:.1f} min")
print(f"\nDONE ({MODEL_ID}). Download v5_{MODEL_TAG}.zip from the Output tab.")
