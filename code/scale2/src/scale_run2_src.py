# %% [markdown]
# # Exploratory scale check, second plan: one run per model with the two-step reading
#
# **What this notebook does.** It runs one model of the second scale plan
# ([`SCALE_CHECK_2.md`, tag `scale-plan-2`](https://github.com/JhishanDeb/llm-evidence-reliance/blob/scale-plan-2/SCALE_CHECK_2.md)).
# Under the first plan, the 14B and 32B models could not be read: after "Answer:" they mostly start markdown bold
# (" **") and write the answer in bold. The two-step reading adds that path: P(word) = P1(word) + P1(" **") × P2(word).
# Prompts, the record and every measure come from the registered code of the second study (`ler_main.py`, main-2.0),
# unchanged; the notebook checks its hash. It tests nothing; the analysis notebook of the second plan does that.
#
# **This is not part of either pre-registration.** The plan was committed before the first run.
#
# **How to run on Kaggle (once per model)**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**.
# 3. In the first code cell set `MODEL_ID`. Order: `Qwen/Qwen3-8B-AWQ`, `Qwen/Qwen3-14B-AWQ`, `Qwen/Qwen3-32B-AWQ`.
# 4. **Save Version → Save & Run All (Commit).** Expected time: about 3 hours for 8B, 5 to 6 hours for 14B, 10 to 13
#    hours for 32B (it will probably need a second session, see below). Kaggle allows 12 hours per session.
# 5. Download `scale2_<model>-2step.zip` from the Output tab. If the model check fails, the notebook still writes the zip
#    (with the reason) and then stops; keep that zip too.
#
# **If a run stops part-way (or says PARTIAL RUN):** add that version's output as an input (Add Input → Your Work →
# this notebook) and run again. The scoring cell picks up the saved checkpoint, and the memorization check is not
# repeated.

# %%
# ============ settings (the only cell you change) ============
MODEL_ID = "Qwen/Qwen3-8B-AWQ"
# The three models of the second scale plan, one per run, in this order:
#   "Qwen/Qwen3-8B-AWQ"   "Qwen/Qwen3-14B-AWQ"   "Qwen/Qwen3-32B-AWQ"
READ_OVERRIDE = "topk"      # Deviation D5 of the first study, kept in both studies: the top 1,000 log-probabilities
TIME_BUDGET_HOURS = 11.0    # scoring stops after this many hours, so the session ends cleanly before Kaggle's limit

import os, sys, time, json, subprocess, platform
T_START = time.time()
BACKEND = "vllm"
MOCK = os.environ.get("LER_MOCK") == "1"            # CPU self-test with a fake model; never set on Kaggle
if os.environ.get("LER_MODEL_ID"):
    MODEL_ID = os.environ["LER_MODEL_ID"]


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
print("MODEL_ID =", MODEL_ID, "| MOCK =", MOCK)
if not MOCK and len(ENV.get("gpus") or []) < 2:
    raise RuntimeError("Two GPUs are needed. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again.")

# %% [markdown]
# ## Paths

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
TIMINGS = {}

# %% [markdown]
# ## Install vLLM, the pinned xgboost and tabmemcheck
# The same procedure and versions as the second study: vLLM 0.26.0 matched to Kaggle's torch, xgboost 3.4.1 (so that
# the features and cards are rebuilt exactly; checked with the dev fingerprints below), tabmemcheck 0.1.6.

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


VLLM_PIN = "0.26.0"     # the version of both studies


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
if MOCK:
    BACKEND_NOTE["install"] = "offline test: vLLM not installed"
else:
    try:
        USE_VLLM, msg = install_vllm()
    except Exception as e:
        USE_VLLM, msg = False, f"vLLM install error: {e!r}"
    BACKEND_NOTE["install"] = msg
    print(msg)
    if not USE_VLLM:
        raise RuntimeError("vLLM is unavailable: " + msg)
NUMPY_AFTER = _pkg_versions(["numpy"]).get("numpy")
BACKEND_NOTE["numpy"] = {"before": NUMPY_BEFORE, "after": NUMPY_AFTER}
if NUMPY_AFTER != NUMPY_BEFORE:
    _ok = sh([sys.executable, "-c", "import numpy, pandas, sklearn, scipy, xgboost"])[0] == 0
    BACKEND_NOTE["numpy_changed_imports_ok"] = _ok
    if not _ok:
        raise RuntimeError(f"vLLM changed numpy {NUMPY_BEFORE} -> {NUMPY_AFTER} and broke other packages. Send me "
                           "this output.")

XGB_PIN = "3.4.1"
pins = _pkg_versions(["torch", "numpy", "pandas", "xgboost", "tabmemcheck"])
with open("/tmp/ler_constraints.txt", "w") as f:
    for k in ("torch", "numpy", "pandas"):
        if k in pins:
            f.write(f"{k}=={pins[k].split('+')[0]}\n")
want = []
if pins.get("xgboost") != XGB_PIN and not MOCK:
    want.append(f"xgboost=={XGB_PIN}")
if pins.get("tabmemcheck") != "0.1.6":
    want.append("tabmemcheck==0.1.6")
if want:
    code, out = sh([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "-c", "/tmp/ler_constraints.txt",
                    *want], timeout=1800)
    BACKEND_NOTE["extra_install"] = {"packages": want, "ok": code == 0, "tail": out[-1500:]}
    print("installed:", want, "| ok:", code == 0)
    if code != 0:
        print(out[-1500:])
after = _pkg_versions(["xgboost", "tabmemcheck"])
if not MOCK and after.get("xgboost") != XGB_PIN:
    raise RuntimeError(f"xgboost {XGB_PIN} could not be installed; found {after.get('xgboost')}.")
if sh([sys.executable, "-c", "import tabmemcheck"])[0] != 0:
    raise RuntimeError("tabmemcheck could not be installed or imported. See BACKEND_NOTE['extra_install'].")
TIMINGS["install"] = time.time() - t0
print({k: v for k, v in BACKEND_NOTE.items() if not k.endswith("_tail") and k != "extra_install"})

# %% [markdown]
# ## Shared code
# `ler_main.py` is the registered code of the second study (main-2.0), byte for byte; `ler_scale.py` is the first
# scale plan's module (scale-1.0), as committed at `scale-plan`; `ler_scale2.py` adds the two-step reading.

# %% [writefile ler_main.py]

# %% [writefile ler_scale.py]

# %% [writefile ler_scale2.py]

# %% [markdown]
# ## Check the code and the plan
# `ler_main.py` must be the registered main-2.0 (SHA-256). `SCALE_CHECK_2.md` is downloaded from the `scale-plan-2`
# tag; every model id and revision of this notebook, and the code hash, must appear in it. Otherwise the run stops.

# %%
import importlib
import hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
for _m in ("ler_scale2", "ler_scale", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]              # cells re-run by hand -> use the freshly written files
import ler_main as L
import ler_scale as S
import ler_scale2 as S2

CODE_SHA = L.file_sha256(os.path.join(SRC_DIR, "ler_main.py"))
if CODE_SHA != S.REGISTERED_CODE_SHA256:
    raise RuntimeError(f"ler_main.py is not the registered main-2.0 (sha256 {CODE_SHA}).")
print("ler_main.py is the registered main-2.0:", CODE_SHA[:16], "| ler_scale", S.SCALE_VERSION, "| ler_scale2",
      S2.SCALE2_VERSION)
if MODEL_ID not in S2.SCALE2_MODELS:
    raise RuntimeError(f"{MODEL_ID} is not one of the second plan's models: {list(S2.SCALE2_MODELS)}")
_M = S2.SCALE2_MODELS[MODEL_ID]
MODEL_KIND, MODEL_TAG, MODEL_REVISION = S.KIND, _M["tag"] + "-2step", _M["revision"]   # file names differ from plan 1
RUN_MEMCHECK = _M["memcheck"]
OUT_DIR = os.path.join(WORK, f"scale2_outputs_{MODEL_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"model {MODEL_ID}: tag {MODEL_TAG}, {_M['size_b']}B, revision {MODEL_REVISION}, vLLM memory share "
      f"{_M['gpu_memory_utilization']}, eager {_M['enforce_eager']}, memorization check {RUN_MEMCHECK}, outputs in {OUT_DIR}")

PLAN = {"tag": S2.PLAN2_TAG, "url": S2.PLAN2_URL}
try:
    _plan = (open(os.environ["LER_SCALE_PLAN2_PATH"], encoding="utf-8").read() if os.environ.get("LER_SCALE_PLAN2_PATH")
             else L.http_get(S2.PLAN2_URL).decode("utf-8"))
except Exception as e:
    raise RuntimeError(f"Could not download SCALE_CHECK_2.md at the tag {S2.PLAN2_TAG} ({e!r}). Commit the plan and "
                       "create the tag scale-plan-2 before the runs.") from e
PLAN["sha256"] = hashlib.sha256(_plan.encode("utf-8")).hexdigest()
PLAN["missing"] = S2.check_against_plan2(_plan)
if PLAN["missing"]:
    raise RuntimeError("SCALE_CHECK_2.md at scale-plan-2 lacks these model ids, revisions or sections:\n"
                       + "\n".join(PLAN["missing"]))
print(f"Plan check passed ({S2.PLAN2_TAG}, SCALE_CHECK_2.md sha256 {PLAN['sha256'][:16]}).")

try:
    _ids_txt = (open(os.environ["LER_PILOT_IDS_PATH"]).read() if os.environ.get("LER_PILOT_IDS_PATH")
                else L.http_get(L.PILOT_IDS_URL).decode("utf-8"))
    PILOT_IDS = [ln.strip() for ln in _ids_txt.splitlines() if ln.strip() and not ln.startswith("#")]
    if sorted(PILOT_IDS) != sorted(str(i) for i in L.PILOT_IDS_FALLBACK):
        raise RuntimeError("pilot_record_ids.txt on GitHub differs from the copy in the code")
    PLAN["pilot_ids_source"] = "github prereg-v1 (matches built-in copy)"
except RuntimeError:
    raise
except Exception as e:
    PILOT_IDS = [str(i) for i in L.PILOT_IDS_FALLBACK]
    PLAN["pilot_ids_source"] = f"built-in copy (GitHub not reachable: {e!r})"
print("pilot patients excluded from dev/test pools:", len(PILOT_IDS), "|", PLAN["pilot_ids_source"])

# %% [markdown]
# ## Data: three public datasets
# PhysioNet 2012 (ODC-By 1.0), PhysioNet 2019 (CC BY 4.0) and UCI Diabetes 130 (CC BY 4.0), as in both studies.

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
if not MOCK:
    bad = {ds: [k for k, ok in v.items() if not ok] for ds, v in FP_CHECK.items() if not all(v.values())}
    if bad:
        raise RuntimeError(f"The rebuilt data differ from the dev phase: {bad}. Send me this output.")
    print("All dev fingerprints match: same splits, features, dev cards and solved examples as both studies.")

# %% [markdown]
# ## Build every test prompt and check them
# 13 prompts per patient (example set A): Yes/No alarm framing C0, C2, C4, C6; Yes/No reversed framing C0, C2, C4;
# answer options C0, C2, C4 in both option orders.

# %%
RECORDS = []
GRID = S.scale_grid()
_max_p = int(os.environ.get("LER_TEST_MAX_PATIENTS", 0)) if MOCK else 0   # offline tests only
for ds, b in BUNDLES.items():
    b.use_sample("test")
    for rid in (b.ids[:_max_p] if _max_p else b.ids):
        for cond, framing, shot in GRID:
            text, cdir = b.prompt(rid, cond, framing, shot)
            if text is None:
                continue
            RECORDS.append({"dataset": ds, "id": rid, "cond": cond, "framing": framing, "shot_set": shot,
                            "y": b.y[rid], "card_dir": cdir, "xgb_p": float(b.xgb_p[rid]),
                            "key": f"{ds}|{rid}|{cond}|{framing}|{shot}", "text": text,
                            "text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]})
PROMPTS_DF = pd.DataFrame(RECORDS)

_strip = lambda t: "\n".join(ln for ln in t.splitlines() if not ln.startswith(("Question:", "Answer", "A: ", "B: ")))
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
    if L.is_choice(r["framing"]):
        twin = _byk[f"{r['dataset']}|{r['id']}|{r['cond']}|alarm|A"]
        assert _strip(r["text"]) == _strip(twin["text"]), "answer-option prompt differs outside the Question/Answer lines"
        assert r["card_dir"] == twin["card_dir"] and r["text"].endswith(L.CHOICE_INSTRUCTION)
    if r["cond"] in ("C4", "C6") and r["framing"] == "alarm":
        c2 = _byk[f"{r['dataset']}|{r['id']}|C2|alarm|A"]
        assert r["card_dir"] == 1 - c2["card_dir"], f"{r['cond']} card does not point opposite to C2"
CHECKS = {"prompts_checked": len(RECORDS), "grid": [list(g) for g in GRID],
          "prompts_per_dataset": PROMPTS_DF.groupby("dataset").size().to_dict(),
          "prompts_per_framing": PROMPTS_DF.groupby("framing").size().to_dict(),
          "prompt_set_sha256": hashlib.sha256("".join(sorted(PROMPTS_DF["key"] + PROMPTS_DF["text_sha"])).encode())
          .hexdigest()}
print(f"{len(RECORDS)} test prompts built and checked: {CHECKS['prompts_per_dataset']} | {CHECKS['prompts_per_framing']}")
_show = f"{L.DATASETS[0]}|{BUNDLES[L.DATASETS[0]].ids[0]}|C2|alarm|A"
print(f"\nOne full prompt ({_show}):\n")
print(_byk[_show]["text"])

# dev prompts (never test patients) for the model check
_dev_texts = {"yesno": [], "choice": []}
for ds, b in BUNDLES.items():
    b.use_sample("dev")
    for rid in b.ids[:6]:
        for cond in ("C0", "C2", "C4"):
            _dev_texts["yesno"].append(b.prompt(rid, cond, "alarm", "A")[0])
            if rid in b.ids[:3]:
                for fr in L.CHOICE_FRAMINGS:
                    _dev_texts["choice"].append(b.prompt(rid, cond, fr, "A")[0])
    b.use_sample("test")

# %% [markdown]
# ## Model: download, the model check, load
# The model check runs in a separate process (a failed engine start cannot leave this notebook holding GPU memory):
# load with float16 activations on both GPUs, the two-step reading of 54 Yes/No and 54 answer-option dev prompts, and
# the four sanity questions. If the model cannot be run, the notebook saves the reason in `scale2_<model>.zip` and
# stops.

# %%
import zipfile
t0 = time.time()
ZIP = os.path.join(WORK, f"scale2_{MODEL_TAG}.zip")
MODELCHECK_NAME = f"modelcheck_{MODEL_TAG}.json"


def save_zip(names):
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if os.path.exists(os.path.join(OUT_DIR, name)):
                z.write(os.path.join(OUT_DIR, name), name)
        z.write(os.path.join(SRC_DIR, "ler_main.py"), "ler_main.py")
        for _f in ("ler_scale.py", "ler_scale2.py"):
            z.write(os.path.join(SRC_DIR, _f), _f)
    print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB)")


if MOCK:
    MODEL_PATH, MODEL_REV = None, "mock"
else:
    MODEL_PATH, MODEL_REV = L.download_model(MODEL_ID, os.path.join(CACHE, "hub"), MODEL_REVISION, HF_TOKEN)
    print("model revision:", MODEL_REV)
    if MODEL_REV != MODEL_REVISION:
        raise RuntimeError(f"downloaded revision {MODEL_REV} differs from the pinned {MODEL_REVISION}")
TIMINGS["model_download"] = time.time() - t0

t0 = time.time()
if MOCK:
    MODELCHECK = {"ok": os.environ.get("LER_SCALE_FAIL_CHECK") != "1", "mock": True,
                  "problems": [] if os.environ.get("LER_SCALE_FAIL_CHECK") != "1" else ["simulated failure (test)"]}
else:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    _pp = os.path.join(OUT_DIR, "modelcheck_prompts.json")
    json.dump({k: [L.render(t, MODEL_KIND, _tok) for t in v] for k, v in _dev_texts.items()}, open(_pp, "w"))
    try:
        code, out = sh([sys.executable, "-m", "ler_scale2", "check", MODEL_PATH, _pp, str(_M["gpu_memory_utilization"]),
                        "1" if _M["enforce_eager"] else "0"], timeout=3600, cwd=SRC_DIR,
                       env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired as e:
        code, out = 1, f"model check timed out: {e!r}"
    MODELCHECK = S2.parse_model_check2(out)
MODELCHECK.update(model_id=MODEL_ID, model_tag=MODEL_TAG, revision=MODEL_REV, size_b=_M["size_b"])
json.dump(MODELCHECK, open(os.path.join(OUT_DIR, MODELCHECK_NAME), "w"), indent=1, default=str)
TIMINGS["model_check"] = time.time() - t0
print(json.dumps({k: v for k, v in MODELCHECK.items() if k not in ("traceback", "output_tail", "variants")},
                 default=str)[:3000])
if not MODELCHECK["ok"]:
    json.dump({"study": "scale2", "model_id": MODEL_ID, "model_tag": MODEL_TAG, "size_b": _M["size_b"],
               "model_revision": MODEL_REV, "pinned_revision": MODEL_REVISION, "cannot_be_run": True,
               "model_check": MODELCHECK, "plan": PLAN, "env": ENV, "backend": BACKEND_NOTE, "mock": MOCK,
               "code_sha256": CODE_SHA, "scale_version": S.SCALE_VERSION, "scale2_version": S2.SCALE2_VERSION},
              open(os.path.join(OUT_DIR, f"run_log_{MODEL_TAG}.json"), "w"), indent=1, default=str)
    save_zip([MODELCHECK_NAME, f"run_log_{MODEL_TAG}.json"])
    raise RuntimeError(f"{MODEL_ID} cannot be run as planned: {MODELCHECK.get('problems')}. Download "
                       f"scale2_{MODEL_TAG}.zip (it records the reason) and send it to me.")
print(f"model check passed | two-step Yes/No dev mass {L._get(MODELCHECK, 'reading_yesno', 'mass_median')} "
      f"(first token alone {L._get(MODELCHECK, 'reading_yesno', 'mass_first_median')}, P(' **') "
      f"{L._get(MODELCHECK, 'reading_yesno', 'p_bold_median')}) | "
      f"A/B dev mass {L._get(MODELCHECK, 'reading_choice', 'mass_median')} | "
      f"{L._get(MODELCHECK, 'reading_yesno', 'seconds_per_prompt')} s per dev prompt")

t0 = time.time()
if MOCK:
    _mock_cfg = json.loads(os.environ.get("LER_MOCK_CFG", "{}"))
    _truth = {"\n" + b._block(b.values[r]): b.y[r] for b in BUNDLES.values() for r in b.ids}
    BACKEND_OBJ = L.MockBackend([s["q_reversed"] for s in L.SPEC.values()], truth=_truth, **_mock_cfg)
else:
    BACKEND_OBJ = S2.TwoStepVLLMBackend(MODEL_PATH, tensor_parallel_size=2, max_model_len=L.MAX_MODEL_LEN,
                                        read=READ_OVERRIDE, gpu_memory_utilization=_M["gpu_memory_utilization"],
                                        enforce_eager=_M["enforce_eager"])
TIMINGS["model_load"] = time.time() - t0
BACKEND_NOTE["used"] = BACKEND_OBJ.name
TOK = BACKEND_OBJ.tokenizer
SYSTEM_TEXT = L.template_system_text(TOK)
print("backend:", BACKEND_OBJ.name, "| read:", BACKEND_OBJ.read, "|", json.dumps(BACKEND_OBJ.info(), default=str)[:600])
print("chat template adds its own system text:", repr(SYSTEM_TEXT))

# %% [markdown]
# ## Memorization check (14B and 32B; the 8B model was checked under the first plan)
# tabmemcheck header test and row completion test on the public files that contain the outcomes. If an earlier
# session of this model already ran it (its output attached as an input), that result is reused.

# %%
t0 = time.time()
MEMCHECK_NAME = f"memcheck_{MODEL_TAG}.json"
MEMCHECK = None if RUN_MEMCHECK else {"run": False, "model_revision": MODEL_REV, "results": {},
                                      "note": "checked under the first scale plan (results/scale/), nothing flagged"}
for root in (EXTRA_ROOTS if RUN_MEMCHECK else []):
    for dirpath, _, files in os.walk(root):
        if MEMCHECK_NAME in files:
            _m = json.load(open(os.path.join(dirpath, MEMCHECK_NAME)))
            if _m.get("model_revision") == MODEL_REV and _m.get("results"):
                MEMCHECK = dict(_m, reused_from=os.path.join(dirpath, MEMCHECK_NAME))
if MEMCHECK is None:
    MEMCHECK = {"run": True, "model_revision": MODEL_REV}
    _files = {ds: DATA_INFO[ds].get("memcheck_file") for ds in L.DATASETS}
    _llm = L.TabmemLLM(BACKEND_OBJ, MODEL_KIND, L.MAX_MODEL_LEN)
    MEMCHECK["results"] = L.run_memcheck(_files, _llm, log=print)
    MEMCHECK["calls"], MEMCHECK["max_prompt_tokens"] = _llm.calls, _llm.max_prompt_tokens
    MEMCHECK["settings"] = {k: v for k, v in L.MEMCHECK.items()}
    import tabmemcheck as _tm
    MEMCHECK["tabmemcheck_version"] = _tm.__version__
elif MEMCHECK.get("reused_from"):
    print("memorization check reused from", MEMCHECK["reused_from"])
else:
    print("memorization check not run:", MEMCHECK.get("note"))
with open(os.path.join(OUT_DIR, MEMCHECK_NAME), "w") as f:
    json.dump(MEMCHECK, f, indent=1, default=str)
print("flagged datasets:", [ds for ds, r in MEMCHECK["results"].items() if r.get("flag_verbatim_rows")])
TIMINGS["memcheck"] = time.time() - t0

# %% [markdown]
# ## Rendered prompts and their lengths

# %%
PROMPTS_DF["rendered"] = [L.render(t, MODEL_KIND, TOK) for t in PROMPTS_DF["text"]]
PROMPTS_DF["n_tokens"] = [len(TOK.encode(t, add_special_tokens=False)) for t in PROMPTS_DF["rendered"]]
PROMPTS_DF["rendered_sha"] = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in PROMPTS_DF["rendered"]]
print("prompt tokens: median", int(PROMPTS_DF["n_tokens"].median()), "| max", int(PROMPTS_DF["n_tokens"].max()),
      "| limit", L.MAX_MODEL_LEN)
if PROMPTS_DF["n_tokens"].max() >= L.MAX_MODEL_LEN - 1:
    raise RuntimeError("a prompt is longer than the model window")
print("\nEnd of one rendered prompt (what the model sees last):\n...", PROMPTS_DF["rendered"].iloc[0][-300:])

# %% [markdown]
# ## Score every test prompt: the answer-word log-probabilities after "Answer:"
# Saved after every chunk. If an earlier session of this model stopped part-way and its output is attached as an
# input, the scored prompts are picked up from there.

# %%
CKPT_NAME = f"scale2_checkpoint_{MODEL_TAG}.jsonl"
CKPT = os.path.join(OUT_DIR, CKPT_NAME)
_systext = hashlib.sha256(str(SYSTEM_TEXT).encode()).hexdigest()[:12]
SIG = f"{MODEL_ID}|{MODEL_REV}|{MODEL_KIND}|{BACKEND_OBJ.name}|{BACKEND_OBJ.read}|{L.CODE_VERSION}|{S.SCALE_VERSION}|{S2.SCALE2_VERSION}|{_systext}"
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
with open(CKPT, "w") as f:
    for d in done.values():
        f.write(json.dumps(d, default=float) + "\n")
todo = PROMPTS_DF[~PROMPTS_DF["key"].isin(done)]
print(f"{len(done)} already scored, {len(todo)} to go")
CHUNK = 512 if BACKEND_OBJ.name == "vllm" else 64
TIME_META = os.path.join(OUT_DIR, f"scale2_timing_{MODEL_TAG}.json")
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {}
for root in EXTRA_ROOTS:                       # carry the scoring time of an earlier session
    for dirpath, _, files in os.walk(root):
        if os.path.basename(TIME_META) in files and not _tm:
            _tm = json.load(open(os.path.join(dirpath, os.path.basename(TIME_META))))
_tm = _tm if _tm.get("sig") == SIG else {"sig": SIG, "seconds": 0.0, "scored": 0}
t0 = time.time()
n_scored, n_invalid = 0, 0
_is_ch = todo["framing"].map(L.is_choice)
_parts = [g.iloc[i:i + CHUNK] for g in (todo[~_is_ch], todo[_is_ch]) for i in range(0, len(g), CHUNK)]
_budget = float(os.environ.get("LER_TIME_BUDGET_HOURS", TIME_BUDGET_HOURS)) * 3600
with open(CKPT, "a") as fout:
    for j, part in enumerate(_parts):
        if time.time() - T_START > _budget or (MOCK and j == int(os.environ.get("LER_TEST_STOP_AT_CHUNK", -1))):
            print(f"time budget of {_budget / 3600:.2f} h reached: stopping after {n_scored} prompts in this session")
            break
        res = BACKEND_OBJ.read_framing(part["rendered"].tolist(), part["framing"].iloc[0])
        for (_, r), rr in zip(part.iterrows(), res):
            rec = {k: r[k] for k in ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir",
                                     "xgb_p", "n_tokens", "text_sha", "rendered_sha")}
            rec["card_dir"] = None if pd.isna(rec["card_dir"]) else int(rec["card_dir"])
            rec = L.finish_record(rec, rr, r["framing"])
            rec["p_bold"] = float(rr.get("p_bold", 0.0 if MOCK else float("nan")))
            rec["mass_first"] = float(rr.get("mass_first", rec["yn_mass"] if MOCK else float("nan")))
            rec["sig"] = SIG
            n_invalid += rec["valid"] == 0
            done[rec["key"]] = rec
            fout.write(json.dumps(rec, default=float) + "\n")
        fout.flush()
        n_scored += len(part)
        if n_scored >= 256 and n_invalid > 0.2 * n_scored:
            raise RuntimeError(f"{n_invalid} of the first {n_scored} readings are invalid (no usable answer-word "
                               "probability). Stopping to save GPU time; send me this output.")
        el = time.time() - t0
        json.dump({"sig": SIG, "seconds": _tm["seconds"] + el, "scored": _tm["scored"] + n_scored}, open(TIME_META, "w"))
        if j == len(_parts) - 1 or j % max(1, len(_parts) // 10) == 0:
            print(f"  {n_scored}/{len(todo)} scored | {el / 60:.1f} min | {el / n_scored:.3f} s/prompt")
TIMINGS["scoring"] = time.time() - t0
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {"seconds": 0.0, "scored": 0}
PARTIAL = len(done) < len(PROMPTS_DF)
if PARTIAL:
    print(f"\nPARTIAL RUN: {len(done)} of {len(PROMPTS_DF)} prompts scored so far. The checkpoint is saved in the "
          "Output tab. Run this notebook again with this version's output attached (Add Input -> Your Work -> this "
          "notebook); it continues where it stopped. The cells below are skipped in this session.")
else:
    RES = pd.DataFrame([done[k] for k in PROMPTS_DF["key"]])
    for c in ("y", "pred_alarm", "pred_yes", "correct", "valid", "n_imputed"):
        RES[c] = RES[c].astype(int)
    print("scored:", len(RES), "| total scoring time across sessions:", round(_tm.get("seconds", 0) / 3600, 2), "h")

# %% [markdown]
# ## Validity (the registered rules; nothing else is analysed here)

# %%
if not PARTIAL:
    rows = []
    for ds in L.DATASETS:
        r = RES[RES["dataset"] == ds]
        v, w = L.validity(r), L.validity_choice(r)
        rows.append({"dataset": ds, "n_prompts": len(r), "n_invalid": int((r["valid"] == 0).sum()),
                     "n_imputed": int((r["n_imputed"] > 0).sum()), "tokens_median": int(r["n_tokens"].median()),
                     "tokens_max": int(r["n_tokens"].max()), "median_yn_mass": v["median_yn_mass"],
                     "c0_accuracy": v["c0_accuracy"], "passes_validity": v["passes"],
                     "median_ab_mass": w["median_ab_mass"], "c0_accuracy_ab": w["c0_accuracy"],
                     "passes_validity_ab": w["passes"],
                     "p_bold_median_yesno": float(r[~r["framing"].map(L.is_choice)]["p_bold"].median()),
                     "p_bold_median_ab": float(r[r["framing"].map(L.is_choice)]["p_bold"].median())})
    VALIDITY = pd.DataFrame(rows)
    print(VALIDITY.round(3).to_string(index=False))
    CHECKS["validity"] = VALIDITY.to_dict(orient="records")
    CHECKS["read_mode"] = BACKEND_OBJ.read
else:
    print("skipped: partial run (see above)")

# %% [markdown]
# ## Save everything and make one zip to download

# %%
def save_all():
    REC_NAME = f"records_{MODEL_TAG}.csv.gz"
    RES.drop(columns=["sig"], errors="ignore").to_csv(os.path.join(OUT_DIR, REC_NAME), index=False)
    with open(os.path.join(OUT_DIR, f"example_prompts_{MODEL_TAG}.md"), "w", encoding="utf-8") as f:
        for ds, b in BUNDLES.items():
            rid = b.ids[0]
            for cond in L.CONDITIONS:
                for fr in L.FRAMINGS + L.CHOICE_FRAMINGS:
                    k = f"{ds}|{rid}|{cond}|{fr}|A"
                    if k in _byk:
                        f.write(f"## {ds} | patient {rid} | {cond} | {fr} framing | example set A\n\n```\n"
                                f"{PROMPTS_DF.set_index('key').at[k, 'rendered']}\n```\n\n")
    RUN_LOG = {"study": "scale2", "scale_version": S.SCALE_VERSION, "scale2_version": S2.SCALE2_VERSION, "plan": PLAN, "model_id": MODEL_ID, "model_tag": MODEL_TAG,
               "model_kind": MODEL_KIND, "size_b": _M["size_b"], "pinned_revision": MODEL_REVISION, "model_revision": MODEL_REV,
               "cannot_be_run": False, "model_check": MODELCHECK, "backend": BACKEND_NOTE, "backend_info": BACKEND_OBJ.info(),
               "read_mode": BACKEND_OBJ.read, "template_system_text": SYSTEM_TEXT, "env": ENV,
               "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()},
               "scoring_seconds_all_sessions": _tm.get("seconds"), "code_version": L.CODE_VERSION, "code_sha256": CODE_SHA,
               "library_versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "scipy", "xgboost")},
               "fingerprints": {ds: b.info["fingerprint"] for ds, b in BUNDLES.items()}, "dev_fingerprint_check": FP_CHECK,
               "data": DATA_INFO, "bundles": {ds: b.info for ds, b in BUNDLES.items()}, "checks": CHECKS,
               "memcheck_summary": {ds: {k: r.get(k) for k in ("file", "flag_verbatim_rows")} | {
                   "row_exact_matches": r.get("row", {}).get("exact_matches"),
                   "header_matched_chars": r.get("header", {}).get("matched_chars")}
                   for ds, r in MEMCHECK.get("results", {}).items()},
               "n_records": len(RES), "mock": MOCK}
    with open(os.path.join(OUT_DIR, f"run_log_{MODEL_TAG}.json"), "w") as f:
        json.dump(RUN_LOG, f, indent=1, default=str)
    save_zip([REC_NAME, f"run_log_{MODEL_TAG}.json", f"example_prompts_{MODEL_TAG}.md", MEMCHECK_NAME, MODELCHECK_NAME])
    print(f"Total time: {(time.time() - T_START) / 60:.1f} min")
    print(f"\nDONE ({MODEL_ID}). Download scale2_{MODEL_TAG}.zip from the Output tab.")


if PARTIAL:
    print(f"skipped: partial run. Run again with this version's output attached to finish {MODEL_ID}.")
else:
    save_all()
