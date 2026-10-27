# %% [markdown]
# # Scale check: diagnostic of the models that could not be read (engineering only)
#
# **Why.** Qwen3-14B-AWQ and Qwen3-32B-AWQ failed the model check of the scale-check plan (`SCALE_CHECK.md`, tag
# `scale-plan`): after "Answer:" they put a median of only about 0.17 of their probability on "Yes"/"No" (dev
# prompts). Qwen3-8B-AWQ passed (0.95). This notebook shows what the three models put their probability on instead.
#
# **What it does.** On the same 54 Yes/No and 54 answer-option dev prompts as the model check (dev patients only, never
# test patients), and on the four sanity questions, it records the 20 most likely next tokens, their mean probability,
# the answer-word mass, and the greedy continuation of a few prompts. It computes no accuracy and no asymmetry, and it
# changes no rule of the plan: the two models stay "cannot be run".
#
# **How to run on Kaggle.** File → Import Notebook → upload this file. Settings: Accelerator **GPU T4 x2**,
# Internet **On**. Save Version → Save & Run All. About 1 hour. Download `scale_diagnostic.zip` from the Output tab.

# %%
# ============ settings ============
DIAG_MODELS = ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ", "Qwen/Qwen3-32B-AWQ"]   # the 8B passed: the contrast

import os, sys, time, json, subprocess, platform, shutil
T_START = time.time()
BACKEND = "vllm"
MOCK = os.environ.get("LER_MOCK") == "1"            # CPU self-test; never set on Kaggle


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
print("torch:", ENV.get("torch"), "| GPUs:", ENV.get("gpus"))
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
# `ler_main.py` (the registered main-2.0, byte for byte), `ler_scale.py` (the scale check, as committed at `scale-plan`)
# and `scale_diag.py` (this diagnostic).

# %% [writefile ler_main.py]

# %% [writefile ler_scale.py]

# %% [writefile scale_diag.py]


# %% [markdown]
# ## Check the code and the plan
# The same checks as the run notebook: `ler_main.py` is the registered main-2.0 and the plan at `scale-plan` lists every
# model and revision.

# %%
import importlib
import hashlib
import numpy as np
import pandas as pd
sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()
for _m in ("scale_diag", "ler_scale", "ler_main"):
    if _m in sys.modules:
        del sys.modules[_m]
import ler_main as L
import ler_scale as S
import scale_diag as D

CODE_SHA = L.file_sha256(os.path.join(SRC_DIR, "ler_main.py"))
if CODE_SHA != S.REGISTERED_CODE_SHA256:
    raise RuntimeError(f"ler_main.py is not the registered main-2.0 (sha256 {CODE_SHA}).")
_plan = (open(os.environ["LER_SCALE_PLAN_PATH"], encoding="utf-8").read() if os.environ.get("LER_SCALE_PLAN_PATH")
         else L.http_get(S.PLAN_URL).decode("utf-8"))
PLAN = {"tag": S.PLAN_TAG, "sha256": hashlib.sha256(_plan.encode("utf-8")).hexdigest(), "missing": S.check_against_plan(_plan)}
if PLAN["missing"]:
    raise RuntimeError(f"SCALE_CHECK.md at scale-plan lacks: {PLAN['missing']}")
print("code", CODE_SHA[:16], "| plan", PLAN["sha256"][:16])
OUT_DIR = os.path.join(WORK, "scale_diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)
try:
    _ids_txt = (open(os.environ["LER_PILOT_IDS_PATH"]).read() if os.environ.get("LER_PILOT_IDS_PATH")
                else L.http_get(L.PILOT_IDS_URL).decode("utf-8"))
    PILOT_IDS = [ln.strip() for ln in _ids_txt.splitlines() if ln.strip() and not ln.startswith("#")]
except Exception:
    PILOT_IDS = [str(i) for i in L.PILOT_IDS_FALLBACK]
assert sorted(PILOT_IDS) == sorted(str(i) for i in L.PILOT_IDS_FALLBACK)

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
# ## The dev prompts of the model check, and the sanity questions

# %%
_dev_texts = {"yesno": [], "choice": []}
for ds, b in BUNDLES.items():
    b.use_sample("dev")
    for rid in b.ids[:6]:
        for cond in ("C0", "C2", "C4"):
            _dev_texts["yesno"].append(b.prompt(rid, cond, "alarm", "A")[0])
            if rid in b.ids[:3]:
                for fr in L.CHOICE_FRAMINGS:
                    _dev_texts["choice"].append(b.prompt(rid, cond, fr, "A")[0])
_dev_texts["sanity"] = [t for t, _, _ in L.SANITY_PROMPTS.values()]
print({k: len(v) for k, v in _dev_texts.items()})

# %% [markdown]
# ## Each model in its own process: download, read the top 20 next tokens, greedy continuations

# %%
RESULTS = {}
for mid in DIAG_MODELS:
    m = S.SCALE_MODELS[mid]
    t0 = time.time()
    if MOCK:
        RESULTS[mid] = {"ok": True, "mock": True}
        continue
    path, rev = L.download_model(mid, os.path.join(CACHE, "hub"), m["revision"], HF_TOKEN)
    if rev != m["revision"]:
        raise RuntimeError(f"{mid}: downloaded revision {rev} differs from the pinned {m['revision']}")
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(path)
    pp = os.path.join(OUT_DIR, "prompts.json")
    json.dump({k: [L.render(t, S.KIND, _tok) for t in v] for k, v in _dev_texts.items()}, open(pp, "w"))
    try:
        code, out = sh([sys.executable, "-m", "scale_diag", path, pp, str(m["gpu_memory_utilization"]),
                        "1" if m["enforce_eager"] else "0"], timeout=3600, cwd=SRC_DIR,
                       env=dict(os.environ, PYTHONPATH=SRC_DIR))
    except subprocess.TimeoutExpired as e:
        out = f"timed out: {e!r}"
    r = D.parse(out)
    r.update(model_id=mid, revision=rev, size_b=m["size_b"], seconds_total=round(time.time() - t0, 1))
    RESULTS[mid] = r
    json.dump(r, open(os.path.join(OUT_DIR, f"diag_{m['tag']}.json"), "w"), indent=1, default=str)
    print(f"\n===== {mid} ({r.get('seconds_total')} s) ok={r.get('ok')} {r.get('error', '')}")
    for fmt in ("yesno", "choice", "sanity"):
        x = r.get(fmt)
        if not x:
            continue
        print(f"--- {fmt}: answer-word mass (median) {x['answer_mass_median']:.3f}")
        print("    most likely next tokens (mean probability):",
              ", ".join(f"{k} {v:.3f}" for k, v in x["top_tokens_mean_probability"][:12]))
        for g in x["greedy_continuations"][:4]:
            print("    greedy continuation:", repr(g))
    shutil.rmtree(path, ignore_errors=True)        # free the disk before the next model

# %% [markdown]
# ## Save

# %%
import zipfile
md = ["# Scale check: diagnostic of the next token after \"Answer:\" (dev prompts only)", "",
      f"Code main-2.0 ({CODE_SHA[:16]}), plan {PLAN['tag']} ({PLAN['sha256'][:16]}). Engineering only: no accuracy, "
      "no asymmetry, no rule changed.", ""]
for mid, r in RESULTS.items():
    md += [f"## {mid}", ""]
    for fmt in ("yesno", "choice", "sanity"):
        x = r.get(fmt)
        if not x:
            continue
        md += [f"**{fmt}** ({x['n_prompts']} prompts): answer-word mass median {x['answer_mass_median']:.3f}", "",
               "| next token | mean probability |", "|---|---|"]
        md += [f"| `{k}` | {v:.3f} |" for k, v in x["top_tokens_mean_probability"][:15]]
        md += ["", "Greedy continuations after \"Answer:\":", ""] + [f"- `{g!r}`" for g in x["greedy_continuations"]] + [""]
open(os.path.join(OUT_DIR, "diagnostic.md"), "w", encoding="utf-8").write("\n".join(md))
json.dump({"env": ENV, "plan": PLAN, "code_sha256": CODE_SHA, "results": RESULTS}, open(os.path.join(OUT_DIR, "diagnostic.json"), "w"),
          indent=1, default=str)
ZIP = os.path.join(WORK, "scale_diagnostic.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(OUT_DIR)):
        if name != "prompts.json":
            z.write(os.path.join(OUT_DIR, name), name)
    for name in ("ler_main.py", "ler_scale.py", "scale_diag.py"):
        z.write(os.path.join(SRC_DIR, name), name)
print("Saved:", ZIP, f"| total {(time.time() - T_START) / 60:.1f} min")
