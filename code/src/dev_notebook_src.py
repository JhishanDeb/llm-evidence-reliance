# %% [markdown]
# # Dev phase: LLM evidence reliance (pre-registered study)
#
# **What this notebook does.** It runs the *dev phase* described in Section 8 of the protocol
# ([`PREREGISTRATION.md`, release `prereg-v1`](https://github.com/JhishanDeb/llm-evidence-reliance/releases/tag/prereg-v1)):
# 100 dev patients per dataset, one model (Qwen2.5-7B-Instruct), every condition and framing. It checks the code,
# the Yes/No reading, the prompt lengths and the runtime, and projects how long the full study will take.
#
# **What it does not do.** It never touches the test patients' LLM outputs. Dev results may change only
# engineering settings (batch size, backend). Any change to prompts, measures or hypotheses must be logged under
# "Deviations" in the protocol *before* the main run.
#
# **How to run on Kaggle**
# 1. File → Import Notebook → upload this file.
# 2. Settings: Accelerator **GPU T4 x2**, Internet **On**.
# 3. (Not needed for Qwen.) For later Llama runs: Add-ons → Secrets → `HF_TOKEN`.
# 4. **Save Version → Save & Run All (Commit).** Expected time: about 30–60 minutes, most of it the PhysioNet 2019
#    download (~40,000 small files) and the model download.
# 5. Download `dev_bundle.zip` from the Output tab and send it to me.

# %%
# ============ settings (the only cell you normally change) ============
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # dev phase: one instruct model (protocol Section 8)
MODEL_KIND = "instruct"                 # "instruct": chat template + 'Answer:' prefill | "base": text + '\nAnswer:'
BACKEND = "auto"                        # "auto" = vLLM, falling back to transformers | "vllm" | "hf"
MODEL_REVISION = None                   # None = latest commit (resolved and logged)
GPU_HOURS_BUDGET = 30.0                 # your weekly Kaggle GPU quota, used only for the runtime projection
MAX_MODEL_LEN = 4096

import os, sys, time, json, subprocess, platform
T_START = time.time()
MOCK = os.environ.get("LER_MOCK") == "1"            # CPU self-test with a fake model; never set on Kaggle
LOCAL_MODEL = os.environ.get("LER_LOCAL_MODEL")      # offline test with a small local model (transformers)
MAX_MODEL_LEN = int(os.environ.get("LER_MAX_MODEL_LEN", MAX_MODEL_LEN))
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
print("MODEL_ID =", MODEL_ID, "| KIND =", MODEL_KIND, "| BACKEND =", BACKEND, "| MOCK =", MOCK,
      "| LOCAL_MODEL =", LOCAL_MODEL)
if not MOCK and not LOCAL_MODEL and not ENV.get("gpus"):
    raise RuntimeError("No GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again.")

# %% [markdown]
# ## Paths and access token

# %%
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC_DIR = os.path.abspath(".")
DATA_DIR = os.environ.get("LER_DATA_DIR", "/tmp/ler_data" if os.path.isdir("/kaggle/working") else
                          os.path.join(WORK, "ler_data"))
EXTRA_ROOTS = ["/kaggle/input"] if os.path.isdir("/kaggle/input") else []
OUT_DIR = os.path.join(WORK, "dev_outputs")
CACHE = "/tmp/hf_cache"
for _d in (DATA_DIR, OUT_DIR, CACHE):
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
print("HF_TOKEN available:", bool(HF_TOKEN), "(not needed for Qwen)")
print("data:", DATA_DIR, "| outputs:", OUT_DIR)
TIMINGS = {}

# %% [markdown]
# ## Install vLLM (falls back to transformers if anything fails)
# Same procedure that worked in the MedQA project: vLLM is matched to Kaggle's preinstalled torch, the CUDA 12
# build is used when torch is a CUDA 12 build, and a broken optional `torchcodec` is removed.

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


def install_vllm():
    import re
    have = _pkg_versions(["vllm"])
    if "vllm" in have:
        return True, f"vllm {have['vllm']} already installed"
    pins = _pkg_versions(["torch"])
    if "torch" not in pins:
        return False, "torch not installed"
    torch_pub, _, torch_cuda = pins["torch"].partition("+")
    with open("/tmp/vllm_constraints.txt", "w") as f:
        f.write(f"torch=={torch_pub}\n")
    base = [sys.executable, "-m", "pip", "install", "--prefer-binary", "-c", "/tmp/vllm_constraints.txt"]
    code, out = sh(base + ["--only-binary", "vllm", "--dry-run", "vllm"], timeout=1200)
    BACKEND_NOTE["vllm_dry_run_tail"] = out[-1500:]
    m = re.search(r"Would install .*?\bvllm-(\d+\.\d+\.\d+(?:\.post\d+)?)\b", out, re.S)
    if code != 0 or not m:
        return False, "no vLLM release built for this torch version (dry run failed)"
    ver = m.group(1)
    target = ["vllm"]
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
    BACKEND_NOTE["install"] = "offline test: nothing installed"
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
for _pkg in ("xgboost", "sklearn"):
    if sh([sys.executable, "-c", f"import {_pkg}"])[0] != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "xgboost" if _pkg == "xgboost" else "scikit-learn"])
TIMINGS["install"] = time.time() - t0
print({k: v for k, v in BACKEND_NOTE.items() if not k.endswith("_tail")})

# %% [markdown]
# ## Shared code
# This cell writes `ler_dev.py`: every rule the protocol fixes (prompt text, seeds, inclusion rules, splits,
# XGBoost settings) lives here, so the main notebook will reuse exactly the same code.

# %% [writefile ler_dev.py]

# %% [markdown]
# ## Check the code against the registered protocol
# Downloads `PREREGISTRATION.md` and `pilot_record_ids.txt` from the `prereg-v1` release and checks that every
# fixed prompt string in the code appears there word for word.

# %%
import importlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
if "ler_dev" in sys.modules:
    del sys.modules["ler_dev"]          # cells re-run by hand -> use the freshly written file
import ler_dev as L

PROTOCOL = {"tag": "prereg-v1", "url": L.PROTOCOL_URL}
_local = os.environ.get("LER_PROTOCOL_PATH")
try:
    _text = open(_local, encoding="utf-8").read() if _local else L.http_get(L.PROTOCOL_URL).decode("utf-8")
    PROTOCOL["sha256"] = __import__("hashlib").sha256(_text.encode("utf-8")).hexdigest()
    _missing = L.check_against_protocol(_text)
    PROTOCOL["strings_missing_from_protocol"] = _missing
    if _missing:
        raise RuntimeError("These prompt strings in the code do not appear verbatim in the protocol:\n"
                           + "\n".join(_missing))
    print("Protocol check passed: every fixed prompt string matches PREREGISTRATION.md (prereg-v1).")
except RuntimeError:
    raise
except Exception as e:
    PROTOCOL["check_error"] = repr(e)
    print("WARNING: could not download the protocol to check the strings:", e)

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
# PhysioNet 2012 (ODC-By 1.0), PhysioNet 2019 (CC BY 4.0) and UCI Diabetes 130 (CC BY 4.0). Each is downloaded
# once (or taken from an attached Kaggle dataset) and turned into one row per patient using the protocol's rules.

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
# ## Splits, XGBoost, the 16 features and the evidence cards
# Per dataset: 60% trains XGBoost, 10% is the dev pool, 30% the test pool (stratified, seed 2027). Only the dev
# pool is used here. For the test pool we only count how many patients are eligible, to see whether 300 per class
# is possible (otherwise the protocol's fallback rule applies).

# %%
t0 = time.time()
BUNDLES = {}
for ds in L.DATASETS:
    df_, feats_, cats_ = TABLES[ds]
    BUNDLES[ds] = L.Bundle(ds, df_, feats_, cats_, DATA_INFO[ds], pilot_ids=PILOT_IDS if ds == "physionet2012" else ())
    b = BUNDLES[ds]
    print("   features:", ", ".join(L.display_name(ds, c) for c in b.features))
    print("   cards:", json.dumps(b.info["cards"]))
TIMINGS["bundles"] = time.time() - t0

# %% [markdown]
# ## Build every dev prompt and check them
# Grid on the dev patients: the main grid (C0, C2, C4, C6 × alarm and reversed framing × example set A) plus the
# robustness grid (C0, C2, C4 × alarm framing × example sets B and C), exactly as the full run will do it.

# %%
RECORDS = []
for ds, b in BUNDLES.items():
    for rid in b.ids:
        grid = [(c, f, "A") for c in L.CONDITIONS for f in L.FRAMINGS] + \
               [(c, "alarm", s) for c in L.ROBUST_GRID["conditions"] for s in L.ROBUST_GRID["shot_sets"]]
        for cond, framing, shot in grid:
            text, cdir = b.prompt(rid, cond, framing, shot)
            if text is None:
                continue
            RECORDS.append({"dataset": ds, "id": rid, "cond": cond, "framing": framing, "shot_set": shot,
                            "y": b.y[rid], "card_dir": cdir, "xgb_p": float(b.xgb_p[rid]),
                            "key": f"{ds}|{rid}|{cond}|{framing}|{shot}", "text": text})
PROMPTS_DF = pd.DataFrame(RECORDS)

# --- automatic checks on the prompts ---
_strip = lambda t: "\n".join(ln for ln in t.splitlines() if not ln.startswith(("Question:", "Answer:")))
_checks = {}
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
_checks["prompts_checked"] = len(RECORDS)
print(f"{len(RECORDS)} dev prompts built and checked "
      f"({PROMPTS_DF.groupby('dataset').size().to_dict()}); C6 skipped for "
      f"{sum(v is None for b in BUNDLES.values() for v in b.partner.values())} patients without a partner.")
for ds, b in BUNDLES.items():
    rid = b.ids[0]
    print("\n" + "=" * 34, ds, "| full prompt: C2, alarm framing, example set A", "=" * 34)
    print(_byk[f"{ds}|{rid}|C2|alarm|A"]["text"])
    for c in ("C4", "C6"):
        k = f"{ds}|{rid}|{c}|alarm|A"
        if k in _byk:
            print(f"\n--- evidence block, {c} ---")
            print(_byk[k]["text"].split(L.EVIDENCE_HEADER)[1].split("\n\nQuestion:")[0].strip())
    print("\n--- question lines, reversed framing ---")
    print(_byk[f"{ds}|{rid}|C0|reversed|A"]["text"].split("\n\nQuestion:")[-1].strip())

# %% [markdown]
# ## Model: download the exact revision, test vLLM in a subprocess, load
# The vLLM smoke test runs in a separate process first, so a failed start cannot leave this notebook holding GPU
# memory; if it fails, the notebook switches to the transformers backend and records why.

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

if USE_VLLM:
    t0 = time.time()
    try:
        code, out = sh([sys.executable, "-m", "ler_dev", "smoke", MODEL_PATH, MODEL_KIND], timeout=1800,
                       cwd=SRC_DIR, env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired:
        code, out = 1, "smoke test timed out"
    ok = code == 0 and "SMOKE_RESULT" in out
    BACKEND_NOTE["vllm_smoke_test"] = "ok" if ok else out[-3000:]
    TIMINGS["smoke_test"] = time.time() - t0
    if ok:
        print(out[out.index("SMOKE_RESULT"):][:1200])
    else:
        print("vLLM smoke test FAILED -> switching to transformers\n", out[-3000:])
        if BACKEND == "vllm":
            raise RuntimeError("vLLM smoke test failed")
        USE_VLLM = False

t0 = time.time()
N_GPU = len(ENV.get("gpus") or [])
if MOCK:
    BACKEND_OBJ = L.MockBackend([s["q_reversed"] for s in L.SPEC.values()])
elif USE_VLLM:
    BACKEND_OBJ = L.VLLMBackend(MODEL_PATH, tensor_parallel_size=2 if N_GPU >= 2 else 1, max_model_len=MAX_MODEL_LEN)
else:
    BACKEND_NOTE["switched_to_hf"] = BACKEND != "hf"
    BACKEND_OBJ = L.HFBackend(MODEL_PATH)
TIMINGS["model_load"] = time.time() - t0
BACKEND_NOTE["used"] = BACKEND_OBJ.name
TOK = BACKEND_OBJ.tokenizer
SYSTEM_TEXT = L.template_system_text(TOK) if MODEL_KIND == "instruct" else None
print("backend:", BACKEND_OBJ.name, "|", json.dumps(BACKEND_OBJ.info(), default=str)[:600])
print("chat template adds its own system text:", repr(SYSTEM_TEXT))

# rendered prompts and their lengths
PROMPTS_DF["rendered"] = [L.render(t, MODEL_KIND, TOK) for t in PROMPTS_DF["text"]]
PROMPTS_DF["n_tokens"] = [len(TOK.encode(t, add_special_tokens=False)) for t in PROMPTS_DF["rendered"]]
print("prompt tokens: median", int(PROMPTS_DF["n_tokens"].median()), "| max", int(PROMPTS_DF["n_tokens"].max()),
      "| limit", MAX_MODEL_LEN)
if PROMPTS_DF["n_tokens"].max() >= MAX_MODEL_LEN - 1:
    raise RuntimeError("a prompt is longer than the model window; raise MAX_MODEL_LEN")
print("\nEnd of one rendered prompt (what the model sees last):\n...", PROMPTS_DF["rendered"].iloc[0][-300:])

# %% [markdown]
# ## Score every dev prompt: P(Yes) and P(No) after "Answer:"
# Results are saved every chunk, so a re-run of this cell continues where it stopped.

# %%
CKPT = os.path.join(OUT_DIR, "dev_checkpoint.jsonl")
SIG = f"{MODEL_ID}|{MODEL_REV}|{MODEL_KIND}|{BACKEND_OBJ.name}|{L.CODE_VERSION}"
done = {}
if os.path.exists(CKPT):
    for ln in open(CKPT):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("sig") == SIG:
            done[d["key"]] = d
todo = PROMPTS_DF[~PROMPTS_DF["key"].isin(done)]
print(f"{len(done)} already scored, {len(todo)} to go")
CHUNK = 512 if BACKEND_OBJ.name == "vllm" else 64
TIME_META = os.path.join(OUT_DIR, "dev_timing.json")
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {}
_tm = _tm if _tm.get("sig") == SIG else {"sig": SIG, "seconds": 0.0, "scored": 0}
t0 = time.time()
n_scored = 0
with open(CKPT, "a") as fout:
    for s in range(0, len(todo), CHUNK):
        part = todo.iloc[s:s + CHUNK]
        res = BACKEND_OBJ.yes_no(part["rendered"].tolist())
        for (_, r), rr in zip(part.iterrows(), res):
            rec = {k: r[k] for k in ("key", "dataset", "id", "cond", "framing", "shot_set", "y", "card_dir",
                                     "xgb_p", "n_tokens")}
            rec["card_dir"] = None if pd.isna(rec["card_dir"]) else int(rec["card_dir"])
            rec = L.finish_record(rec, rr, r["framing"])
            rec["sig"] = SIG
            done[rec["key"]] = rec
            fout.write(json.dumps(rec, default=float) + "\n")
        fout.flush()
        n_scored += len(part)
        el = time.time() - t0
        json.dump({"sig": SIG, "seconds": _tm["seconds"] + el, "scored": _tm["scored"] + n_scored}, open(TIME_META, "w"))
        if n_scored == len(todo) or (n_scored // CHUNK) % max(1, len(todo) // CHUNK // 8) == 0:
            print(f"  {n_scored}/{len(todo)} scored | {el:.0f}s | {el / n_scored:.3f} s/prompt")
SCORE_SECONDS = time.time() - t0
TIMINGS["scoring"] = SCORE_SECONDS
_tm = json.load(open(TIME_META)) if os.path.exists(TIME_META) else {"seconds": 0.0, "scored": 0}
TOTAL_SCORE_SECONDS, TOTAL_SCORED = _tm.get("seconds", 0.0), _tm.get("scored", 0)   # across resumed runs
RES = pd.DataFrame([done[k] for k in PROMPTS_DF["key"]])
for c in ("y", "pred_alarm", "pred_yes", "correct", "valid"):
    RES[c] = RES[c].astype(int)
print("scored:", len(RES))

# %% [markdown]
# ## Engineering checks and the runtime projection
# These are the checks the dev phase exists for. The validity thresholds are the protocol's (median Yes+No
# probability ≥ 0.5 and C0 accuracy ≥ 0.55, alarm framing, example set A); here they are only an early warning,
# because the protocol applies them to the test patients.

# %%
CHECKS = {"prompts": _checks}
rows = []
for ds in L.DATASETS:
    r = RES[RES["dataset"] == ds]
    a = r[(r["framing"] == "alarm") & (r["shot_set"] == "A")]
    c0 = a[a["cond"] == "C0"]
    c0v = c0[c0["valid"] == 1]
    rows.append({"dataset": ds, "n_prompts": len(r), "yn_mass_median_alarm_A": a["yn_mass"].median(),
                 "yn_mass_min": r["yn_mass"].min(), "n_yn_mass_below_0.01": int((r["yn_mass"] < 0.01).sum()),
                 "n_invalid_zero_mass": int((r["valid"] == 0).sum()),
                 "C0_accuracy": c0v["correct"].mean(), "C0_share_alarm": c0v["pred_alarm"].mean(),
                 "valid_yn_mass": bool(a["yn_mass"].median() >= 0.5),
                 "valid_C0_acc": bool(c0v["correct"].mean() >= 0.55),
                 "tokens_median": int(r["n_tokens"].median()), "tokens_max": int(r["n_tokens"].max())})
VALIDITY = pd.DataFrame(rows)
print(VALIDITY.round(3).to_string(index=False))
CHECKS["validity"] = VALIDITY.to_dict(orient="records")

# runtime projection for the full study (4 models, main + robustness grids on the test patients)
sec_per_prompt = TOTAL_SCORE_SECONDS / TOTAL_SCORED if TOTAL_SCORED else float("nan")
planned = {ds: 2 * BUNDLES[ds].info["capacity"]["test_per_class_planned"] for ds in L.DATASETS}
per_patient = len(L.MAIN_GRID["conditions"]) * len(L.MAIN_GRID["framings"]) + \
              len(L.ROBUST_GRID["conditions"]) * len(L.ROBUST_GRID["shot_sets"])
proj = []
for label, n_by_ds in (("planned (300 per class, or fallback)", planned),
                       ("compute fallback (200 per class)", {ds: min(400, n) for ds, n in planned.items()})):
    prompts_per_model = sum(n_by_ds.values()) * per_patient
    # one Kaggle session per model: install + data + XGBoost + model download/load are paid every session
    overhead_h = sum(TIMINGS.get(k, 0) for k in ("install", "data", "bundles", "model_download", "model_load",
                                                  "smoke_test")) / 3600
    h_model = prompts_per_model * sec_per_prompt / 3600 + overhead_h
    proj.append({"setting": label, "patients": n_by_ds, "prompts_per_model": prompts_per_model,
                 "session_overhead_hours": round(overhead_h, 2), "hours_per_model": round(h_model, 2),
                 "hours_4_models": round(4 * h_model, 2), "within_budget": bool(4 * h_model <= GPU_HOURS_BUDGET),
                 "fits_one_12h_session": bool(h_model <= 11.0)})
PROJECTION = {"seconds_per_prompt": sec_per_prompt, "backend": BACKEND_OBJ.name, "rows": proj,
              "note": "Measured on Qwen2.5-7B-Instruct; Llama-3.1-8B will be somewhat slower. Each model is assumed "
                      "to run in its own Kaggle session, so install, data, XGBoost and model loading are counted "
                      "once per model. The tabmemcheck memorization tests (protocol Section 3) are not included."}
print(f"\nmeasured: {sec_per_prompt:.3f} s per prompt ({BACKEND_OBJ.name})")
for p in proj:
    print(f"  {p['setting']}: {p['prompts_per_model']} prompts/model -> {p['hours_per_model']} h per model "
          f"(incl. {p['session_overhead_hours']} h setup), {p['hours_4_models']} h for 4 models | within "
          f"{GPU_HOURS_BUDGET:.0f} h budget: {p['within_budget']} | fits one 12 h session: {p['fits_one_12h_session']}")

# things to record under "Deviations" (clarifications) before the main run
TO_LOG = [f"Eligibility operationalised as at least {L.MIN_MEASURED} of {L.N_FEATURES} selected features measured "
          f"(>= 60%); the pilot code used int(0.6 x 16) = 9.",
          "Diabetes 130: the first encounter per patient is taken first, then excluded if it ended in death or "
          "hospice (so such patients are dropped, not replaced by a later encounter).",
          "Validity check: median Yes+No mass over all conditions under the alarm framing with example set A; "
          "C0 accuracy under the alarm framing with example set A.",
          "Solved examples: drawn at random from all training-split patients with the registered seeds, shown in the "
          "order alarming, safe, alarming, safe.",
          "A prompt with no probability on any Yes/No token (Yes+No mass = 0) is counted and left out of the "
          "analysis; it is not read as 'No'."]
if SYSTEM_TEXT:
    TO_LOG.append(f"The {MODEL_ID} chat template adds its own default system text when no system message is "
                  f"given: {SYSTEM_TEXT[:160]!r}. It is used as is (no system message is passed).")
for ds in L.DATASETS:
    cap = BUNDLES[ds].info["capacity"]
    if cap["test_fallback_rule_applies"]:
        TO_LOG.append(f"{ds}: only {cap['test_pool']['eligible_pos']} eligible positives in the test pool, so the "
                      f"pre-registered fallback applies ({cap['test_per_class_planned']} per class).")
if not proj[0]["fits_one_12h_session"]:
    TO_LOG.append("One model does not fit in a single 12 h Kaggle session: the main run must be split by dataset "
                  "(resume from the checkpoint).")
if not proj[0]["within_budget"] and TOTAL_SCORED:
    TO_LOG.append("Projected GPU time exceeds the budget: decide on the pre-registered compute fallback "
                  "(N = 400 per dataset) before the first test run.")
for v in rows:
    if not (v["valid_yn_mass"] and v["valid_C0_acc"]):
        TO_LOG.append(f"{v['dataset']}: dev validity warning (median Yes+No mass {v['yn_mass_median_alarm_A']:.2f}, C0 "
                      f"accuracy {v['C0_accuracy']:.2f}); look at this before the main run.")
print("\nTo record under 'Deviations' before the main run:")
for t in TO_LOG:
    print(" -", t)

# %% [markdown]
# ## Every protocol analysis, run on the dev patients (code check only)
# **Do not use these numbers to change prompts, measures or hypotheses.** This runs each pre-registered measure
# (accuracy with Wilson CIs, AUROC, H1/H2 McNemar, H3 asymmetry with its bootstrap p-value, the reversed-framing
# question, the log-odds asymmetry, H5 over-trust and H6 flip-consistency with DeLong) so that the code is tested
# before the main run. H4 needs a base model, so it is only tested on synthetic data. Holm here runs across the
# 3 datasets of this one model; in the main run it runs across the 6 primary cells. With 100 patients per dataset
# the intervals are wide.

# %%
EXPLORE = {ds: L.analyse_cell(RES[RES["dataset"] == ds]) for ds in L.DATASETS}
EXPLORE = L.holm_across(EXPLORE)
fmt = lambda x: "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"
for ds, e in EXPLORE.items():
    acc = {c: round(v["accuracy"], 3) for c, v in e["by_condition"].items()}
    print(f"[{ds}] accuracy (alarm, set A): {acc} | invalid readings: {e['n_invalid']}")
    for h in ("H1", "H2"):
        if e.get(h):
            r = e[h]
            print(f"   {h}: accuracy diff {fmt(r['diff'])} | McNemar p {r['p']:.3g} | Holm p {r['p_holm']:.3g}")
    if e.get("H3"):
        r = e["H3"]
        print(f"   H3: A={r['A']:.2f} S={r['S']:.2f} delta={fmt(r['delta'])} [{fmt(r['delta_ci'][0])}, "
              f"{fmt(r['delta_ci'][1])}] p_boot {r['p_boot']:.3g} | log-odds asymmetry {fmt(r['logodds_asym'])} | "
              f"flip C2/C4 {r['flip_rate_C2_C4']:.2f}")
    if e.get("Q3_reversed"):
        r = e["Q3_reversed"]
        print(f"   Q3 (reversed framing): delta={fmt(r['delta'])} [{fmt(r['delta_ci'][0])}, {fmt(r['delta_ci'][1])}]"
              f" -> {e.get('Q3_interpretation')}")
    if e.get("H5"):
        r = e["H5"]
        print(f"   H5: over-trust on flipped evidence {fmt(r['overtrust'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]")
    if e.get("H6"):
        r = e["H6"]
        print(f"   H6: AUROC with flag {r['auroc_with_flag']:.3f} vs without {r['auroc_without_flag']:.3f} | "
              f"DeLong p {r['p_delong']:.3g}")

# %% [markdown]
# ## Save everything and make one zip to download

# %%
import zipfile
RES.drop(columns=["sig"], errors="ignore").to_csv(os.path.join(OUT_DIR, "dev_records.csv"), index=False)
with open(os.path.join(OUT_DIR, "example_prompts.md"), "w", encoding="utf-8") as f:
    for ds, b in BUNDLES.items():
        rid = b.ids[0]
        for cond in L.CONDITIONS:
            for fr in L.FRAMINGS:
                k = f"{ds}|{rid}|{cond}|{fr}|A"
                if k in _byk:
                    f.write(f"## {ds} | patient {rid} | {cond} | {fr} framing | example set A\n\n```\n"
                            f"{PROMPTS_DF.set_index('key').at[k, 'rendered']}\n```\n\n")
RUN_LOG = {"protocol": PROTOCOL, "model_id": MODEL_ID, "model_revision": MODEL_REV, "model_kind": MODEL_KIND,
           "backend": BACKEND_NOTE, "backend_info": BACKEND_OBJ.info(), "template_system_text": SYSTEM_TEXT,
           "env": ENV, "timings_seconds": {k: round(v, 1) for k, v in TIMINGS.items()},
           "code_version": L.CODE_VERSION, "code_sha256": L.file_sha256(os.path.join(SRC_DIR, "ler_dev.py")),
           "library_versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "scipy", "xgboost")},
           "fingerprints": {ds: b.info["fingerprint"] for ds, b in BUNDLES.items()},
           "data": DATA_INFO, "bundles": {ds: b.info for ds, b in BUNDLES.items()},
           "checks": CHECKS, "projection": PROJECTION, "to_log_before_main_run": TO_LOG,
           "exploratory_dev_results": EXPLORE, "n_records": len(RES)}
with open(os.path.join(OUT_DIR, "run_log.json"), "w") as f:
    json.dump(RUN_LOG, f, indent=1, default=str)
md = ["# Dev-phase report", "", f"Model: {MODEL_ID} (revision {MODEL_REV}, {MODEL_KIND}), backend {BACKEND_OBJ.name}",
      f"Protocol: prereg-v1, sha256 {PROTOCOL.get('sha256', 'not checked')}", "",
      "## Data", ""]
for ds in L.DATASETS:
    b = BUNDLES[ds]
    md += [f"- **{ds}**: {json.dumps(DATA_INFO[ds], default=str)}",
           f"  - split {json.dumps(b.info['split'])}", f"  - XGBoost AUROC on the dev pool (16 features): "
           f"{b.info['xgb']['auroc_dev_pool_16']:.3f}", f"  - test-pool capacity {json.dumps(b.info['capacity'])}",
           f"  - cards {json.dumps(b.info['cards'])}"]
md += ["", "## Validity (dev, early warning)", "", "```", VALIDITY.round(3).to_string(index=False), "```", "",
       "## Runtime projection", "", f"{sec_per_prompt:.3f} s per prompt ({BACKEND_OBJ.name})", ""]
md += [f"- {p['setting']}: {p['prompts_per_model']} prompts per model, {p['hours_per_model']} h per model, "
       f"{p['hours_4_models']} h for 4 models (budget {GPU_HOURS_BUDGET:.0f} h)" for p in proj]
md += ["", "## To record under Deviations before the main run", ""] + [f"- {t}" for t in TO_LOG]
md += ["", "## Exploratory dev results (not for decisions)", "", "```", json.dumps(EXPLORE, indent=1, default=str), "```"]
with open(os.path.join(OUT_DIR, "dev_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
ZIP = os.path.join(WORK, "dev_bundle.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("dev_report.md", "run_log.json", "dev_records.csv", "example_prompts.md"):
        z.write(os.path.join(OUT_DIR, name), name)
    z.write(os.path.join(SRC_DIR, "ler_dev.py"), "ler_dev.py")
print("Saved:", ZIP, f"({os.path.getsize(ZIP) / 1e6:.1f} MB)")
print(f"Total time: {(time.time() - T_START) / 60:.1f} min")
print("\nDONE. Download dev_bundle.zip from the Output tab and send it to me.")
