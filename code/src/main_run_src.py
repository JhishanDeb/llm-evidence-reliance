# %% [markdown]
# # Main run: LLM evidence reliance (pre-registered study)
#
# **What this notebook does.** It runs the main run of the protocol
# ([`PREREGISTRATION.md`, release `prereg-v1.1`](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v1.1))
# for **one model**: the memorization check, then every test prompt (600 patients per dataset × 14 prompts =
# 25,200 prompts). Run it four times, once per registered model. It does not test hypotheses; the separate
# analysis notebook does that after all four runs, exactly as registered.
#
# **Before the first run:** the Deviations entry must be committed and the `prereg-v1.1` release created. The
# notebook checks this and stops if it is missing.
#
# **How to run on Kaggle (once per model)**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**.
# 3. For the two Llama models: request access on Hugging Face first, then Add-ons → Secrets → `HF_TOKEN`.
# 4. In the first code cell set `MODEL_ID` to one of the four registered models.
# 5. **Save Version → Save & Run All (Commit).** Expected time: about 2–2.5 hours per model with vLLM. If vLLM
#    fails and the notebook falls back to transformers, a run can take much longer; see the next paragraph.
# 6. Download `main_<model>.zip` from the Output tab. Keep all four.
#
# **If a run stops part-way:** open the failed version's Output, add it to the notebook as an input
# (Add Input → Your Work → this notebook), and run again. The scoring cell picks up the saved checkpoint.

# %%
# ============ settings (the only cell you change) ============
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# The four registered models, one per run:
#   "Qwen/Qwen2.5-7B-Instruct"   "Qwen/Qwen2.5-7B"
#   "meta-llama/Llama-3.1-8B-Instruct"   "meta-llama/Llama-3.1-8B"   (these two need HF_TOKEN)
BACKEND = "auto"            # "auto" = vLLM, falling back to transformers | "vllm" | "hf"
MODEL_REVISION = None       # None = latest commit (resolved and logged)
RUN_MEMCHECK = True         # protocol Section 3 / Deviation D3.7; set False only if you log it as a deviation
READ_OVERRIDE = "topk"      # Deviation D5: all four runs read the top 1,000 log-probabilities (the check still runs)

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
    raise RuntimeError("Llama models are gated: add your Hugging Face token as the Kaggle secret HF_TOKEN "
                       "(Add-ons -> Secrets) after your access request was approved.")
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


VLLM_PIN = "0.26.0"     # the dev-phase version; the same for all four runs


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
# This cell writes `ler_main.py`: the dev-phase code for data, splits, XGBoost, cards and prompts (unchanged),
# plus the test sample, the exact Yes/No reading, the memorization check and the registered analysis.

# %% [writefile ler_main.py]

# %% [markdown]
# ## Check the protocol: prompt strings and the committed deviations
# Downloads `PREREGISTRATION.md` from the `prereg-v1.1` release. Every fixed prompt string in the code must
# appear there word for word, and the Deviations entry (D1–D4) must be present. Otherwise the run stops.

# %%
import importlib
import hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
if "ler_main" in sys.modules:
    del sys.modules["ler_main"]          # cells re-run by hand -> use the freshly written file
import ler_main as L

if MODEL_ID not in L.REGISTERED_MODELS:
    raise RuntimeError(f"{MODEL_ID} is not one of the four registered models: {list(L.REGISTERED_MODELS)}")
MODEL_KIND, MODEL_FAMILY, MODEL_TAG = L.REGISTERED_MODELS[MODEL_ID]
OUT_DIR = os.path.join(WORK, f"main_outputs_{MODEL_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"model {MODEL_ID}: {MODEL_KIND}, family {MODEL_FAMILY}, outputs in {OUT_DIR}")

PROTOCOL = {"tag": L.PROTOCOL_TAG, "url": L.PROTOCOL_URL}
_local = os.environ.get("LER_PROTOCOL_PATH")
try:
    _text = open(_local, encoding="utf-8").read() if _local else L.http_get(L.PROTOCOL_URL).decode("utf-8")
except Exception as e:
    raise RuntimeError(f"Could not download PREREGISTRATION.md at the tag {L.PROTOCOL_TAG} ({e!r}). Commit the "
                       "Deviations entry and create the prereg-v1.1 release on GitHub before the main run.") from e
PROTOCOL["sha256"] = hashlib.sha256(_text.encode("utf-8")).hexdigest()
_missing = L.check_against_protocol(_text)
_missing_dev = L.check_deviations(_text)
PROTOCOL["strings_missing_from_protocol"], PROTOCOL["deviation_markers_missing"] = _missing, _missing_dev
if _missing:
    raise RuntimeError("These prompt strings in the code do not appear verbatim in the protocol:\n" + "\n".join(_missing))
if _missing_dev:
    raise RuntimeError("The protocol at prereg-v1.1 lacks the committed deviations (missing: " + "; ".join(_missing_dev)
                       + "). Commit the Deviations entry and tag it prereg-v1.1 first.")
print(f"Protocol check passed ({L.PROTOCOL_TAG}, sha256 {PROTOCOL['sha256'][:16]}): prompt strings match and "
      "Deviations D1-D4 are committed.")

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
# dev phase.

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
# (fingerprints). Then the test set is drawn: 300 patients per class from the eligible test pool (seed 2027).

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
        raise RuntimeError(f"The rebuilt data differ from the dev phase: {bad}. Do not run the main study on data "
                           "that differ from the dev run; send me this output.")
    print("All dev fingerprints match: same splits, features, dev cards and solved examples as the dev run.")

# %% [markdown]
# ## Build every test prompt and check them
# Main grid (C0, C2, C4, C6 × alarm and reversed framing × example set A) and robustness grid (C0, C2, C4 ×
# alarm framing × example sets B and C): 14 prompts per patient.

# %%
RECORDS = []
_max_p = int(os.environ.get("LER_TEST_MAX_PATIENTS", 0)) if (MOCK or LOCAL_MODEL) else 0   # offline tests only
for ds, b in BUNDLES.items():
    b.use_sample("test")
    for rid in (b.ids[:_max_p] if _max_p else b.ids):
        grid = [(c, f, "A") for c in L.CONDITIONS for f in L.FRAMINGS] + \
               [(c, "alarm", s) for c in L.ROBUST_GRID["conditions"] for s in L.ROBUST_GRID["shot_sets"]]
        for cond, framing, shot in grid:
            text, cdir = b.prompt(rid, cond, framing, shot)
            if text is None:
                continue
            RECORDS.append({"dataset": ds, "id": rid, "cond": cond, "framing": framing, "shot_set": shot,
                            "y": b.y[rid], "card_dir": cdir, "xgb_p": float(b.xgb_p[rid]),
                            "key": f"{ds}|{rid}|{cond}|{framing}|{shot}", "text": text,
                            "text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]})
PROMPTS_DF = pd.DataFrame(RECORDS)

_strip = lambda t: "\n".join(ln for ln in t.splitlines() if not ln.startswith(("Question:", "Answer:")))
_byk = {r["key"]: r for r in RECORDS}
for r in RECORDS:
    if r["cond"] == "C0":
        assert L.EVIDENCE_HEADER not in r["text"], "C0 prompt contains evidence"
    else:
        assert r["text"].count(L.EVIDENCE_HEADER) == 1, "evidence block missing or repeated"
    if r["framing"] == "reversed":
        twin = _byk[r["key"].replace("|reversed|", "|alarm|")]
        assert _strip(r["text"]) == _strip(twin["text"]), "framings differ outside the Question/Answer lines"
        assert r["card_dir"] == twin["card_dir"]
    if r["cond"] in ("C4", "C6") and r["shot_set"] == "A" and r["framing"] == "alarm":
        c2 = _byk[f"{r['dataset']}|{r['id']}|C2|alarm|A"]
        assert r["card_dir"] == 1 - c2["card_dir"], f"{r['cond']} card does not point opposite to C2"
CHECKS = {"prompts_checked": len(RECORDS),
          "prompts_per_dataset": PROMPTS_DF.groupby("dataset").size().to_dict(),
          "c6_without_partner": int(sum(v is None for b in BUNDLES.values() for v in b.partner.values())),
          "prompt_set_sha256": hashlib.sha256("".join(sorted(PROMPTS_DF["key"] + PROMPTS_DF["text_sha"])).encode())
          .hexdigest()}
print(f"{len(RECORDS)} test prompts built and checked: {CHECKS['prompts_per_dataset']}")
_rid = BUNDLES[L.DATASETS[0]].ids[0]
print("\nOne full prompt (C2, alarm framing, example set A):\n")
print(_byk[f"{L.DATASETS[0]}|{_rid}|C2|alarm|A"]["text"])

# a few DEV prompts for the vLLM exact-reading check (Deviation D2); no test output is used for the check
SMOKE_PROMPTS_PATH = os.path.join(OUT_DIR, "smoke_prompts.json")
_dev_texts = []
for ds, b in BUNDLES.items():
    b.use_sample("dev")
    for rid in b.ids[:6]:
        for cond in ("C0", "C2", "C4"):
            _dev_texts.append(b.prompt(rid, cond, "alarm", "A")[0])
    b.use_sample("test")

# %% [markdown]
# ## Model: download, vLLM smoke test with the exact-reading check, load
# The smoke test runs in a separate process (a failed engine start cannot leave this notebook holding GPU
# memory). It also compares the exact log-probabilities (`logprob_token_ids`) with the ordinary top-20 values on
# 54 dev prompts; if they disagree, the run reads the top 1,000 instead (Deviation D2).

# %%
t0 = time.time()
if MOCK:
    MODEL_PATH, MODEL_REV = None, "mock"
elif LOCAL_MODEL:
    MODEL_PATH, MODEL_REV = LOCAL_MODEL, "local-test-model"
else:
    MODEL_PATH, MODEL_REV = L.download_model(MODEL_ID, os.path.join(CACHE, "hub"), MODEL_REVISION, HF_TOKEN)
    print("model revision:", MODEL_REV)
TIMINGS["model_download"] = time.time() - t0

READ_MODE = "exact"
if USE_VLLM:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    json.dump([L.render(t, MODEL_KIND, _tok) for t in _dev_texts], open(SMOKE_PROMPTS_PATH, "w"))
    t0 = time.time()
    try:
        code, out = sh([sys.executable, "-m", "ler_main", "smoke", MODEL_PATH, MODEL_KIND, SMOKE_PROMPTS_PATH],
                       timeout=1800, cwd=SRC_DIR, env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired:
        code, out = 1, "smoke test timed out"
    ok = code == 0 and "SMOKE_RESULT" in out
    TIMINGS["smoke_test"] = time.time() - t0
    if ok:
        SMOKE = json.loads(out[out.index("SMOKE_RESULT") + len("SMOKE_RESULT"):].strip().splitlines()[0])
        BACKEND_NOTE["vllm_smoke_test"] = SMOKE
        READ_MODE = SMOKE["read"]
        print("smoke test ok | read mode from the check:", READ_MODE, "| exact check:", json.dumps(SMOKE["exact_check"]))
        if READ_OVERRIDE and READ_OVERRIDE != READ_MODE:
            BACKEND_NOTE["read_override"] = f"{READ_MODE} -> {READ_OVERRIDE} (Deviation D5)"
            READ_MODE = READ_OVERRIDE
        print("read mode used:", READ_MODE, "(Deviation D5: the same reading for all four runs)")
        print("sky prompt:", SMOKE["sky_prompt"])
    else:
        BACKEND_NOTE["vllm_smoke_test"] = out[-3000:]
        print("vLLM smoke test FAILED -> switching to transformers (exact full softmax)\n", out[-3000:])
        if BACKEND == "vllm":
            raise RuntimeError("vLLM smoke test failed")
        USE_VLLM = False

t0 = time.time()
N_GPU = len(ENV.get("gpus") or [])
if MOCK:
    _mock_cfg = json.loads(os.environ.get("LER_MOCK_CFG", "{}"))
    _truth = {"\n" + b._block(b.values[r]): b.y[r] for b in BUNDLES.values() for r in b.ids}
    BACKEND_OBJ = L.MockBackend([s["q_reversed"] for s in L.SPEC.values()], truth=_truth, **_mock_cfg)
elif USE_VLLM:
    BACKEND_OBJ = L.VLLMBackend(MODEL_PATH, tensor_parallel_size=2 if N_GPU >= 2 else 1,
                                max_model_len=L.MAX_MODEL_LEN, read=READ_MODE)
else:
    BACKEND_NOTE["switched_to_hf"] = BACKEND != "hf"
    BACKEND_OBJ = L.HFBackend(MODEL_PATH)
TIMINGS["model_load"] = time.time() - t0
BACKEND_NOTE["used"] = BACKEND_OBJ.name
TOK = BACKEND_OBJ.tokenizer
SYSTEM_TEXT = L.template_system_text(TOK) if MODEL_KIND == "instruct" else None
print("backend:", BACKEND_OBJ.name, "| read:", BACKEND_OBJ.read, "|", json.dumps(BACKEND_OBJ.info(), default=str)[:500])
print("chat template adds its own system text:", repr(SYSTEM_TEXT))

# %% [markdown]
# ## Memorization check (protocol Section 3, Deviation D3.7)
# tabmemcheck header test and row completion test on the public files that contain the outcomes. Runs before
# any test prompt is scored.

# %%
t0 = time.time()
MEMCHECK = {"run": RUN_MEMCHECK}
if RUN_MEMCHECK:
    _files = {ds: DATA_INFO[ds].get("memcheck_file") for ds in L.DATASETS}
    _llm = L.TabmemLLM(BACKEND_OBJ, MODEL_KIND, L.MAX_MODEL_LEN)
    MEMCHECK["results"] = L.run_memcheck(_files, _llm, log=print)
    MEMCHECK["calls"], MEMCHECK["max_prompt_tokens"] = _llm.calls, _llm.max_prompt_tokens
    MEMCHECK["settings"] = {k: v for k, v in L.MEMCHECK.items()}
    import tabmemcheck as _tm
    MEMCHECK["tabmemcheck_version"] = _tm.__version__
    with open(os.path.join(OUT_DIR, f"memcheck_{MODEL_TAG}.json"), "w") as f:
        json.dump(MEMCHECK, f, indent=1, default=str)
    print("flagged datasets:", [ds for ds, r in MEMCHECK["results"].items() if r.get("flag_verbatim_rows")])
else:
    print("Memorization check SKIPPED (RUN_MEMCHECK = False): log this under Deviations.")
TIMINGS["memcheck"] = time.time() - t0

# %% [markdown]
# ## Rendered prompts and their lengths

# %%
PROMPTS_DF["rendered"] = [L.render(t, MODEL_KIND, TOK) for t in PROMPTS_DF["text"]]
PROMPTS_DF["n_tokens"] = [len(TOK.encode(t, add_special_tokens=False)) for t in PROMPTS_DF["rendered"]]
print("prompt tokens: median", int(PROMPTS_DF["n_tokens"].median()), "| max", int(PROMPTS_DF["n_tokens"].max()),
      "| limit", L.MAX_MODEL_LEN)
if PROMPTS_DF["n_tokens"].max() >= L.MAX_MODEL_LEN - 1:
    raise RuntimeError("a prompt is longer than the model window")
print("\nEnd of one rendered prompt (what the model sees last):\n...", PROMPTS_DF["rendered"].iloc[0][-300:])

# %% [markdown]
# ## Score every test prompt: log P(Yes) and log P(No) after "Answer:"
# Saved after every chunk. If an earlier run of this model stopped part-way and its output is attached as an
# input, the scored prompts are picked up from there.

# %%
CKPT_NAME = f"main_checkpoint_{MODEL_TAG}.jsonl"
CKPT = os.path.join(OUT_DIR, CKPT_NAME)
SIG = f"{MODEL_ID}|{MODEL_REV}|{MODEL_KIND}|{BACKEND_OBJ.name}|{BACKEND_OBJ.read}|{L.CODE_VERSION}"
done = {}


def _read_ckpt(path):
    n = 0
    for ln in open(path):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("sig") == SIG and d.get("key") in _byk and d.get("text_sha") == _byk[d["key"]]["text_sha"]:
            done[d["key"]] = d
            n += 1
    return n


for root in EXTRA_ROOTS:
    for dirpath, _, files in os.walk(root):
        if CKPT_NAME in files and os.path.join(dirpath, CKPT_NAME) != CKPT:
            print(f"resuming from {os.path.join(dirpath, CKPT_NAME)}: {_read_ckpt(os.path.join(dirpath, CKPT_NAME))} records")
if os.path.exists(CKPT):
    _read_ckpt(CKPT)
with open(CKPT, "w") as f:                      # rewrite the local checkpoint with everything recovered so far
    for d in done.values():
        f.write(json.dumps(d, default=float) + "\n")
todo = PROMPTS_DF[~PROMPTS_DF["key"].isin(done)]
print(f"{len(done)} already scored, {len(todo)} to go")
CHUNK = 512 if BACKEND_OBJ.name == "vllm" else 64
TIME_META = os.path.join(OUT_DIR, f"timing_{MODEL_TAG}.json")
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {}
_tm = _tm if _tm.get("sig") == SIG else {"sig": SIG, "seconds": 0.0, "scored": 0}
t0 = time.time()
n_scored, n_invalid = 0, 0
with open(CKPT, "a") as fout:
    for s in range(0, len(todo), CHUNK):
        part = todo.iloc[s:s + CHUNK]
        res = BACKEND_OBJ.yes_no(part["rendered"].tolist())
        for (_, r), rr in zip(part.iterrows(), res):
            rec = {k: r[k] for k in ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir",
                                     "xgb_p", "n_tokens", "text_sha")}
            rec["card_dir"] = None if pd.isna(rec["card_dir"]) else int(rec["card_dir"])
            rec = L.finish_record(rec, rr, r["framing"])
            rec["sig"] = SIG
            n_invalid += rec["valid"] == 0
            done[rec["key"]] = rec
            fout.write(json.dumps(rec, default=float) + "\n")
        fout.flush()
        n_scored += len(part)
        if n_scored >= 256 and n_invalid > 0.2 * n_scored:
            raise RuntimeError(f"{n_invalid} of the first {n_scored} readings are invalid (no usable Yes/No "
                               "probability). Something is wrong with the model or backend; stopping to save GPU time.")
        el = time.time() - t0
        json.dump({"sig": SIG, "seconds": _tm["seconds"] + el, "scored": _tm["scored"] + n_scored}, open(TIME_META, "w"))
        if n_scored == len(todo) or (n_scored // CHUNK) % max(1, len(todo) // CHUNK // 10) == 0:
            print(f"  {n_scored}/{len(todo)} scored | {el / 60:.1f} min | {el / n_scored:.3f} s/prompt")
TIMINGS["scoring"] = time.time() - t0
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {"seconds": 0.0, "scored": 0}
RES = pd.DataFrame([done[k] for k in PROMPTS_DF["key"]])
for c in ("y", "pred_alarm", "pred_yes", "correct", "valid", "n_imputed"):
    RES[c] = RES[c].astype(int)
print("scored:", len(RES), "| total scoring time across runs:", round(_tm.get("seconds", 0) / 3600, 2), "h")

# %% [markdown]
# ## Engineering checks and the validity checks
# The registered validity rule (median Yes+No mass ≥ 0.5 and C0 accuracy ≥ 0.55, alarm framing, example set A)
# decides which cells enter the confirmatory analysis. Nothing else is analysed here.

# %%
rows = []
for ds in L.DATASETS:
    r = RES[RES["dataset"] == ds]
    v = L.validity(r)
    rows.append({"dataset": ds, "n_prompts": len(r), "n_invalid": int((r["valid"] == 0).sum()),
                 "n_imputed": int((r["n_imputed"] > 0).sum()), "median_yn_mass": v["median_yn_mass"],
                 "c0_accuracy": v["c0_accuracy"], "passes_validity": v["passes"],
                 "tokens_median": int(r["n_tokens"].median()), "tokens_max": int(r["n_tokens"].max())})
VALIDITY = pd.DataFrame(rows)
print(VALIDITY.round(3).to_string(index=False))
CHECKS["validity"] = VALIDITY.to_dict(orient="records")
CHECKS["read_mode"] = BACKEND_OBJ.read

# %% [markdown]
# ## Save everything and make one zip to download

# %%
import zipfile
REC_NAME = f"records_{MODEL_TAG}.csv.gz"
RES.drop(columns=["sig"], errors="ignore").to_csv(os.path.join(OUT_DIR, REC_NAME), index=False)
with open(os.path.join(OUT_DIR, f"example_prompts_{MODEL_TAG}.md"), "w", encoding="utf-8") as f:
    for ds, b in BUNDLES.items():
        rid = b.ids[0]
        for cond in L.CONDITIONS:
            for fr in L.FRAMINGS:
                k = f"{ds}|{rid}|{cond}|{fr}|A"
                if k in _byk:
                    f.write(f"## {ds} | patient {rid} | {cond} | {fr} framing | example set A\n\n```\n"
                            f"{PROMPTS_DF.set_index('key').at[k, 'rendered']}\n```\n\n")
RUN_LOG = {"protocol": PROTOCOL, "model_id": MODEL_ID, "model_tag": MODEL_TAG, "model_kind": MODEL_KIND,
           "model_family": MODEL_FAMILY, "model_revision": MODEL_REV, "backend": BACKEND_NOTE,
           "backend_info": BACKEND_OBJ.info(), "read_mode": BACKEND_OBJ.read, "template_system_text": SYSTEM_TEXT,
           "env": ENV, "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()},
           "scoring_seconds_all_runs": _tm.get("seconds"), "code_version": L.CODE_VERSION,
           "code_sha256": L.file_sha256(os.path.join(SRC_DIR, "ler_main.py")),
           "library_versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "scipy", "xgboost")},
           "fingerprints": {ds: b.info["fingerprint"] for ds, b in BUNDLES.items()}, "dev_fingerprint_check": FP_CHECK,
           "data": DATA_INFO, "bundles": {ds: b.info for ds, b in BUNDLES.items()}, "checks": CHECKS,
           "memcheck_summary": {ds: {k: r.get(k) for k in ("file", "flag_verbatim_rows")} | {
               "row_exact_matches": r.get("row", {}).get("exact_matches"),
               "header_matched_chars": r.get("header", {}).get("matched_chars")}
               for ds, r in MEMCHECK.get("results", {}).items()},
           "n_records": len(RES), "mock": MOCK, "local_model": bool(LOCAL_MODEL)}
with open(os.path.join(OUT_DIR, f"run_log_{MODEL_TAG}.json"), "w") as f:
    json.dump(RUN_LOG, f, indent=1, default=str)
ZIP = os.path.join(WORK, f"main_{MODEL_TAG}.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in (REC_NAME, f"run_log_{MODEL_TAG}.json", f"example_prompts_{MODEL_TAG}.md",
                 f"memcheck_{MODEL_TAG}.json"):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            z.write(os.path.join(OUT_DIR, name), name)
    z.write(os.path.join(SRC_DIR, "ler_main.py"), "ler_main.py")
print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB)")
print(f"Total time: {(time.time() - T_START) / 60:.1f} min")
print(f"\nDONE ({MODEL_ID}). Download main_{MODEL_TAG}.zip from the Output tab.")
