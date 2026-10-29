"""Shared code for the MAIN RUN of the pre-registered study
"Do LLMs reason with model explanations, or defer to them one way?"

Protocol: PREREGISTRATION.md, release prereg-v1.1 of https://github.com/JhishanDeb/llm-evidence-reliance
(prereg-v1 plus the Deviations entry D1-D4 of 2026-10-17, committed before any test patient was scored).
The data, split, XGBoost, card and prompt code is the dev-phase code (ler_dev.py, dev-1.1) unchanged, so the
dev fingerprints can be checked. New here: the test sample, the exact Yes/No reading (D2), the memorization
check (D3.7) and the analysis with the adjusted asymmetry statistic (D1).

Version main-2.0 (second study, PREREGISTRATION_V2.md, release prereg-v2) adds, without changing any prompt,
reading or statistic of main-1.0: the answer-option framing ("A" / "B", both orders), the reading of the A/B
log-probabilities, the models registered for the second study, and the analysis of the second study.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODE_VERSION = "main-2.0"

# =============================================================================================================
# 1. Protocol constants (prereg-v1, with the prereg-v1.1 deviations)
# =============================================================================================================
PROTOCOL_TAG = "prereg-v1.1"
PROTOCOL_URL = f"https://raw.githubusercontent.com/JhishanDeb/llm-evidence-reliance/{PROTOCOL_TAG}/PREREGISTRATION.md"
# phrases that must appear in the Deviations section of the tagged protocol before the main run may start
DEVIATION_MARKERS = ["#### D1. Primary asymmetry statistic", "d = a + g1·e + g3·(e/s)³ + c·t",
                     "The test is an intersection–union test",
                     "#### D2. Reading P(Yes) and P(No) exactly", "`logprob_token_ids`",
                     "#### D3. Clarifications", "#### D4. Sample size"]
SPLIT_SEED = 2027            # splits, XGBoost, C6 partners, dev sample
BOOT_SEED = 2026             # paired bootstrap
BOOT_RESAMPLES = 10_000
SHOT_SEEDS = {"A": 2027, "B": 2028, "C": 2029}
N_FEATURES = 16
TOP_K = 5
MIN_MEASURED = math.ceil(0.6 * N_FEATURES)   # "at least 60% of the 16 selected features measured" -> 10 of 16
TEST_PER_CLASS = 300
DEV_PER_CLASS = 50
CONDITIONS = ("C0", "C2", "C4", "C6")
FRAMINGS = ("alarm", "reversed")
DATASETS = ("physionet2012", "physionet2019", "diabetes130")
# full-run grid per model (used for the runtime projection)
MAIN_GRID = dict(conditions=CONDITIONS, framings=FRAMINGS, shot_sets=("A",))
ROBUST_GRID = dict(conditions=("C0", "C2", "C4"), framings=("alarm",), shot_sets=("B", "C"))
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=2, tree_method="hist", random_state=SPLIT_SEED, n_jobs=-1)

SPEC = {
    "physionet2012": dict(
        task="predict in-hospital mortality for an ICU patient from measurements taken in the first 48 hours "
             "of the ICU stay.",
        outcome="death",
        q_alarm="Will this patient die during the current hospital stay?",
        q_reversed="Will this patient survive the current hospital stay?",
        licence="Open Data Commons Attribution License v1.0"),
    "physionet2019": dict(
        task="predict whether an ICU patient will develop sepsis after the first 24 hours of the ICU stay, "
             "from measurements taken in those first 24 hours.",
        outcome="sepsis",
        q_alarm="Will this patient develop sepsis later in this ICU stay?",
        q_reversed="Will this patient stay free of sepsis for the rest of this ICU stay?",
        licence="Creative Commons Attribution 4.0 International (CC BY 4.0)"),
    "diabetes130": dict(
        task="predict whether a hospitalized patient with diabetes will be readmitted within 30 days of "
             "discharge, from information recorded during the hospital stay.",
        outcome="readmission within 30 days",
        q_alarm="Will this patient be readmitted within 30 days of discharge?",
        q_reversed="Will this patient avoid readmission within 30 days of discharge?",
        licence="Creative Commons Attribution 4.0 International (CC BY 4.0)"),
}
EXAMPLES_HEADER = "Here are 4 example patients with known outcomes:"
PATIENT_HEADER = "Now the patient to predict:"
ANSWER_INSTRUCTION = "Answer with Yes or No."
EVIDENCE_HEADER = ("Evidence from a machine-learning model (XGBoost) trained on other patients. These are the "
                   "5 factors that most influenced its risk estimate for this patient:")
PREFILL = "Answer:"

# Second study (PREREGISTRATION_V2.md): the answer-option framing. The two outcomes are offered as options A and B,
# in both orders, so that neither outcome is tied to the word "Yes" or "No" and position is balanced.
CHOICE_QUESTION = "Which of these two outcomes is more likely for this patient?"
CHOICE_INSTRUCTION = "Answer with A or B."
CHOICE_OPTIONS = {
    "physionet2012": {"alarm": "The patient dies during the current hospital stay.",
                      "safe": "The patient survives the current hospital stay."},
    "physionet2019": {"alarm": "The patient develops sepsis later in this ICU stay.",
                      "safe": "The patient stays free of sepsis for the rest of this ICU stay."},
    "diabetes130": {"alarm": "The patient is readmitted within 30 days of discharge.",
                    "safe": "The patient avoids readmission within 30 days of discharge."},
}
CHOICE_FRAMINGS = ("choice_alarm_A", "choice_safe_A")       # option A = the alarming / the safe outcome
CHOICE_GRID = dict(conditions=("C0", "C2", "C4"), framings=CHOICE_FRAMINGS, shot_sets=("A",))
ANSWER_WORDS = {"yesno": ("Yes", "No"), "choice": ("A", "B")}


def is_choice(framing: str) -> bool:
    return framing in CHOICE_FRAMINGS


def answer_pair(framing: str) -> tuple:
    """The two answer words of a framing: ('Yes', 'No') or ('A', 'B'); the first is the 'positive' word."""
    return ANSWER_WORDS["choice" if is_choice(framing) else "yesno"]


def positive_means_alarm(framing: str) -> bool:
    """True if the first answer word ('Yes' or 'A') stands for the alarming outcome in this framing."""
    if framing in ("alarm", "choice_alarm_A"):
        return True
    if framing in ("reversed", "choice_safe_A"):
        return False
    raise ValueError(framing)


PILOT_IDS_URL = "https://raw.githubusercontent.com/JhishanDeb/llm-evidence-reliance/prereg-v1/pilot_record_ids.txt"
# copy of pilot_record_ids.txt (prereg-v1), used if GitHub cannot be reached; checked against it when it can
PILOT_IDS_FALLBACK = [
    132556, 132893, 132996, 133079, 133327, 133863, 133905, 133989, 134010, 134063, 134314, 134419,
    134477, 134498, 134594, 134692, 134823, 134970, 135044, 135176, 135210, 135267, 135268, 135293,
    135342, 135388, 135453, 135879, 135950, 135964, 136060, 136197, 136215, 136240, 136256, 136289,
    136322, 136325, 136462, 136472, 136633, 136640, 136663, 137020, 137333, 137447, 137542, 137563,
    137577, 137583, 137587, 137950, 137960, 138038, 138062, 138127, 138172, 138349, 138367, 138390,
    138511, 138625, 138788, 138847, 138964, 139182, 139502, 139536, 139741, 139889, 139899, 140009,
    140235, 140307, 140740, 140767, 140799, 140813, 140953, 141051, 141147, 141212, 141482, 141561,
    141663, 141665, 141752, 141866, 141887, 142133, 142160, 142195, 142242, 142484, 142542, 142673,
    142688, 142832, 142938, 142987, 143200, 143379, 143699, 143726, 143727, 143799, 143864, 143865,
    143896, 143919, 144049, 144227, 144247, 144344, 144402, 144464, 144579, 144616, 144728, 144735,
    144764, 144766, 144830, 144899, 145183, 145229, 145252, 145302, 145335, 145620, 145713, 145963,
    146009, 146256, 146269, 146513, 146529, 146536, 146899, 146977, 147116, 147338, 147365, 147386,
    147753, 147843, 147960, 148144, 148184, 148206, 148284, 148303, 148473, 148586, 148663, 148681,
    148696, 148712, 148733, 148943, 148945, 148973, 149237, 149279, 149335, 149491, 149526, 149770,
    150012, 150135, 150391, 150400, 150412, 150536, 150562, 150600, 150704, 150848, 150987, 150995,
    151000, 151006, 151108, 151136, 151180, 151255, 151352, 151400, 151441, 151473, 151902, 151920,
    151981, 152040, 152153, 152204, 152335, 152589, 152705, 152829,
]


def protocol_strings():
    """Every fixed string the prompts use, for the check against the registered protocol text."""
    out = [EXAMPLES_HEADER, PATIENT_HEADER, ANSWER_INSTRUCTION, EVIDENCE_HEADER, "Task: {TASK}",
           "Question: {QUESTION}", "Answer: {Yes or No}",
           "1. {feature} = {value}: {increases|decreases} the risk of {OUTCOME} (impact {+0.42})"]
    for s in SPEC.values():
        out += [s["task"], s["q_alarm"], s["q_reversed"], f"| {s['outcome']} |"]
    return out


def check_against_protocol(text: str) -> list[str]:
    """Strings used by the code that do not appear verbatim in the registered protocol (should be empty)."""
    flat = re.sub(r"\s+", " ", text)
    return [s for s in protocol_strings() if re.sub(r"\s+", " ", s) not in flat]


def check_deviations(text: str) -> list[str]:
    """Deviation markers missing from the tagged protocol (should be empty)."""
    dev = text.split("## Deviations", 1)[-1] if "## Deviations" in text else ""
    return [m for m in DEVIATION_MARKERS if m not in dev]


# The four registered models (protocol Section 5): id -> (kind, family, short tag)
REGISTERED_MODELS = {
    "Qwen/Qwen2.5-7B": ("base", "qwen", "qwen-base"),
    "Qwen/Qwen2.5-7B-Instruct": ("instruct", "qwen", "qwen-instruct"),
    "meta-llama/Llama-3.1-8B": ("base", "llama", "llama-base"),
    "meta-llama/Llama-3.1-8B-Instruct": ("instruct", "llama", "llama-instruct"),
}
MODEL_TAGS = [v[2] for v in REGISTERED_MODELS.values()]

# ------------------------------------------------------------------------------- second study (prereg-v2)
PROTOCOL_V2_TAG = "prereg-v2"
_RAW = "https://raw.githubusercontent.com/JhishanDeb/llm-evidence-reliance"
PROTOCOL_V2_URL = f"{_RAW}/{PROTOCOL_V2_TAG}/PREREGISTRATION_V2.md"
PROTOCOL_V1_AT_V2_URL = f"{_RAW}/{PROTOCOL_V2_TAG}/PREREGISTRATION.md"
# section markers that must appear in PREREGISTRATION_V2.md at the tag before any second-study run may start
V2_MARKERS = ["### R1.", "### R2.", "### R3.", "## 4. Models", "## 5. Answer-option framing", "## 7. Analysis"]

# Models of the second study: id -> role, kind, family, tag, prompt grids, pinned revision.
#   role "new":   instruction-tuned models not used in the first study (R1, R2)
#   role "trace": the Olmo 3 post-training stages before the final model (R3 and the stage trace)
#   role "v1":    the two instruct models of the first study, answer-option framing only (R2)
# Revisions of the new models are pinned in PREREGISTRATION_V2.md after the engineering check; None = not yet pinned.
REGISTERED_MODELS_V2 = {
    "allenai/Olmo-3-7B-Instruct": dict(role="new", kind="instruct", family="olmo3", tag="olmo3-instruct",
                                       grids=("yesno", "choice", "order", "baseformat"),
                                       revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
    "Qwen/Qwen3-8B": dict(role="new", kind="instruct", family="qwen3", tag="qwen3-instruct",
                          grids=("yesno", "choice", "order"), revision="b968826d9c46dd6066d109eabc6255188de91218"),
    # third new model: Ministral-3-8B-Instruct-2512 (BF16 and main repositories) could not be loaded by vLLM 0.26.0 in
    # the engineering check, so the first eligible backup takes its place (PREREGISTRATION_V2.md, Section 9)
    "ibm-granite/granite-3.3-8b-instruct": dict(role="new", kind="instruct", family="granite3", tag="granite3-instruct",
                                                grids=("yesno", "choice", "order"),
                                                revision="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"),
    "allenai/Olmo-3-1025-7B": dict(role="trace", kind="base", family="olmo3", tag="olmo3-base",
                                   grids=("yesno",), revision="a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    "allenai/Olmo-3-7B-Instruct-SFT": dict(role="trace", kind="instruct", family="olmo3", tag="olmo3-sft",
                                           grids=("yesno",), revision="e1452fc572d51966ff4aaeb25118b891eb93e549"),
    "allenai/Olmo-3-7B-Instruct-DPO": dict(role="trace", kind="instruct", family="olmo3", tag="olmo3-dpo",
                                           grids=("yesno",), revision="b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"),
    "Qwen/Qwen2.5-7B-Instruct": dict(role="v1", kind="instruct", family="qwen", tag="qwen-instruct",
                                     grids=("choice", "order"), revision="a09a35458c702b33eeacc393d103063234e8bc28"),
    "meta-llama/Llama-3.1-8B-Instruct": dict(role="v1", kind="instruct", family="llama", tag="llama-instruct",
                                             grids=("choice", "order"), revision="0e9e39f249a16976918f6564b8830bc894c89659"),
}
OLMO3_STAGES = ("olmo3-base", "olmo3-sft", "olmo3-dpo", "olmo3-instruct")
# models checked before registration (engineering only): (id, kind, answer formats it must pass). Ministral 3 is
# checked in its BF16 repository and its main repository (which may hold FP8 weights, unusable on T4 GPUs); the last
# two models are the backups for a new model that cannot be run
ENGCHECK_CANDIDATES = [
    ("allenai/Olmo-3-7B-Instruct", "instruct", ("yesno", "choice")),
    ("allenai/Olmo-3-7B-Instruct-DPO", "instruct", ("yesno",)),
    ("allenai/Olmo-3-7B-Instruct-SFT", "instruct", ("yesno",)),
    ("allenai/Olmo-3-1025-7B", "base", ("yesno",)),
    ("Qwen/Qwen3-8B", "instruct", ("yesno", "choice")),
    ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "instruct", ("yesno", "choice")),
    ("mistralai/Ministral-3-8B-Instruct-2512", "instruct", ("yesno", "choice")),
    ("ibm-granite/granite-3.3-8b-instruct", "instruct", ("yesno", "choice")),
    ("mistralai/Mistral-7B-Instruct-v0.3", "instruct", ("yesno", "choice")),
]
BACKUP_MODELS = ["ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-7B-Instruct-v0.3"]   # order of the check
ENGCHECK_ZIP_SHA256 = "068c0351ae00111c03d7d8e95a3a1a73c716cfe0c50f713c0d0b6e4c1fb82a4d"      # results/v2/engcheck_v2.zip


def grid_v2(grids) -> list:
    """(condition, framing, example set) triples of a second-study run:
    'yesno'      = the 14 prompts per patient of the first study (main grid + robustness grid);
    'choice'     = C0, C2, C4 x both option orders x example set A (6);
    'order'      = C0, C2, C4 x alarm framing x example set A in the order safe first ('Asf', 3);
    'baseformat' = C0, C2, C4 x alarm framing x example set A rendered as plain text ('Abase', 3)."""
    out = []
    if "yesno" in grids:
        out += [(c, f, "A") for c in CONDITIONS for f in FRAMINGS]
        out += [(c, "alarm", s) for c in ROBUST_GRID["conditions"] for s in ROBUST_GRID["shot_sets"]]
    if "choice" in grids:
        out += [(c, f, s) for c in CHOICE_GRID["conditions"] for f in CHOICE_GRID["framings"]
                for s in CHOICE_GRID["shot_sets"]]
    if "order" in grids:
        out += [(c, "alarm", "Asf") for c in ("C0", "C2", "C4")]
    if "baseformat" in grids:
        out += [(c, "alarm", "Abase") for c in ("C0", "C2", "C4")]
    return out


def render_kind(model_kind: str, shot_set: str) -> str:
    """How a prompt is rendered: the model's own format, except the base-format control ('Abase'), which is rendered
    as plain text followed by 'Answer:' (the format of the base model)."""
    return "base" if shot_set == "Abase" else model_kind


def protocol_strings_v2():
    """Every fixed string the second study adds, and every model id and pinned revision of the second study, for the
    check against PREREGISTRATION_V2.md."""
    out = [CHOICE_QUESTION, CHOICE_INSTRUCTION, "Answer: {A or B}"]
    for o in CHOICE_OPTIONS.values():
        out += [o["alarm"], o["safe"]]
    for mid, m in REGISTERED_MODELS_V2.items():
        out.append(mid)
        if m["revision"]:
            out.append(m["revision"])
    return out


def check_against_protocol_v2(text: str) -> list[str]:
    """Strings of the second study missing from PREREGISTRATION_V2.md, and missing section markers (should be empty)."""
    flat = re.sub(r"\s+", " ", text)
    miss = [s for s in protocol_strings_v2() if re.sub(r"\s+", " ", s) not in flat]
    return miss + [m for m in V2_MARKERS if m not in text]

# What the Kaggle dev run (dev-1.1, 2026-10-17) produced. The main run must rebuild exactly the same splits,
# features, dev cards and solved examples, so it pins the same xgboost and checks these fingerprints.
DEV_RUN = {
    "xgboost": "3.4.1",
    "fingerprints": {
        "physionet2012": {"dev_pool": "666188bd08c26e4f", "test_pool": "b4b0dcdbcac1bd57",
                          "features": "6de558459e12c44d", "cards": "d15b0e6bf2a50bda", "shots": "961186a3c7999705"},
        "physionet2019": {"dev_pool": "93faf7b255c2c0a8", "test_pool": "9630a2581cc23843",
                          "features": "40bfef7bcc1dced9", "cards": "3ab29fe6e0eff7d8", "shots": "db197d273ab5cb44"},
        "diabetes130": {"dev_pool": "6a8ae23d135b5daa", "test_pool": "bab9cbb714d08b09",
                        "features": "4f96bc315b1615f3", "cards": "dd242cffc99f30ee", "shots": "db9222ebf02a22d4"},
    },
}

# Deviation D2: exact reading, its check, and the fallback
LOGPROB_CHECK_TOL = 0.05     # max |difference| in log-probability between logprob_token_ids and the top-20 values
TOPK_FALLBACK = 1000         # top-k read if logprob_token_ids is unavailable or fails the check
MAX_MODEL_LEN = int(os.environ.get("LER_MAX_MODEL_LEN", 16384))   # env override only for offline tests

# Deviation D3.7: memorization check
MEMCHECK = {"tabmemcheck": "0.1.6", "seed": 2027, "max_tokens": 300, "few_shot_steps": (7, 5, 3, 1),
            "files": {"physionet2012": "Outcomes-a.txt", "physionet2019": "training_setA/p000001.psv",
                      "diabetes130": "diabetic_data.csv"}}


# =============================================================================================================
# 2. Downloads
# =============================================================================================================
UA = {"User-Agent": "Mozilla/5.0 (llm-evidence-reliance research notebook)"}


def http_get(url: str, timeout: int = 120, tries: int = 4) -> bytes:
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccups: retry
            last = e
            time.sleep(min(20, 2 * attempt))
    raise RuntimeError(f"could not fetch {url}: {last}")


def download(url: str, dest, tries: int = 4) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if tmp.stat().st_size == 0:
                raise IOError("empty file")
            tmp.replace(dest)
            return dest
        except Exception as e:
            last = e
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"could not download {url}: {last}")


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def parse_s3_listing(xml_bytes: bytes):
    """ListObjectsV2 XML -> (keys, next_continuation_token or None)."""
    root = ET.fromstring(xml_bytes)
    keys, token, truncated = [], None, False
    for el in root.iter():
        t = _strip_ns(el.tag)
        if t == "Key":
            keys.append(el.text)
        elif t == "IsTruncated":
            truncated = (el.text or "").strip().lower() == "true"
        elif t == "NextContinuationToken":
            token = el.text
    return keys, (token if truncated else None)


def s3_list(bucket_url: str, prefix: str, max_pages: int = 1000) -> list[str]:
    keys, token = [], None
    for _ in range(max_pages):
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        page, token = parse_s3_listing(http_get(bucket_url.rstrip("/") + "/?" + urllib.parse.urlencode(q)))
        keys += page
        if not token:
            break
    return keys


def parallel_download(pairs, workers: int = 32, progress_every: int = 5000) -> int:
    """pairs = [(url, dest)]; skips files that already exist. Returns the number downloaded."""
    todo = [(u, Path(d)) for u, d in pairs if not (Path(d).exists() and Path(d).stat().st_size > 0)]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(lambda p: download(p[0], p[1]), todo), 1):
            done = i
            if i % progress_every == 0:
                print(f"    downloaded {i}/{len(todo)}")
    return done


def find_dirs_with(roots, pattern: str, min_count: int):
    """Folders under roots holding at least min_count files matching pattern (e.g. '*.psv')."""
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
            n = sum(1 for _ in d.glob(pattern))
            if n >= min_count:
                found.append((d, n))
    return found


# =============================================================================================================
# 3. Datasets -> one table each: columns id, y, features...
# =============================================================================================================
# ---------------------------------------------------------------------------------------- PhysioNet 2012
P12_BASES = ["https://physionet.org/files/challenge-2012/1.0.0",
             "https://physionet-open.s3.amazonaws.com/challenge-2012/1.0.0"]
P12_STATIC = ("Age", "Gender", "Height", "ICUType")
P12_PLAUSIBLE = {"pH": (6.5, 8.0), "Temp": (25, 45), "HR": (1, 300), "RespRate": (1, 100), "Weight": (20, 350),
                 "Height": (100, 250), "Age": (0, 120), "FiO2": (0.21, 1.0), "GCS": (3, 15), "SaO2": (20, 100)}


def _p12_clean(param, v):
    if v < 0:
        return None
    if param == "pH" and v > 14:
        v = v / 100.0
    if param == "FiO2" and 1.0 < v <= 100.0:
        v = v / 100.0
    r = P12_PLAUSIBLE.get(param)
    if r and not (r[0] <= v <= r[1]):
        return None
    return v


def p12_read_record(path):
    rid, static, series = None, {}, defaultdict(list)
    with open(path) as f:
        f.readline()  # Time,Parameter,Value
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            _, param, value = parts
            try:
                value = float(value)
            except ValueError:
                continue
            if param == "RecordID":
                rid = int(value)
                continue
            value = _p12_clean(param, value)
            if value is None:
                continue
            if param in P12_STATIC:
                static[param] = value
            else:
                series[param].append(value)
    feats = {"id": str(rid if rid is not None else int(Path(path).stem))}
    for p in P12_STATIC:
        feats[p] = static.get(p, np.nan)
    for p, vals in series.items():
        if p == "MechVent":
            continue
        if p == "Urine":
            feats["Urine_total"] = float(np.sum(vals))
            continue
        feats[f"{p}_min"] = float(np.min(vals))
        feats[f"{p}_max"] = float(np.max(vals))
        feats[f"{p}_last"] = float(vals[-1])
    feats["MechVent_any"] = 1.0 if len(series.get("MechVent", [])) > 0 else 0.0
    return feats


def _p12_records(folder):
    folder = Path(folder)
    return sorted(p for p in folder.glob("*.txt") if p.stem.isdigit()) if folder.is_dir() else []


def _p12_find_set(letter, roots):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for outc in root.rglob(f"Outcomes-{letter}.txt"):
            cands = [outc.parent / f"set-{letter}"] + [p for p in outc.parent.rglob(f"set-{letter}") if p.is_dir()]
            for cand in cands:
                if len(_p12_records(cand)) > 100:
                    return cand, outc
    return None


def _p12_get_set(letter, data_dir: Path, roots, log):
    found = _p12_find_set(letter, roots)
    if found:
        log(f"  set-{letter}: found {len(_p12_records(found[0]))} records in {found[0]}")
        return found
    tgz, outc = data_dir / f"set-{letter}.tar.gz", data_dir / f"Outcomes-{letter}.txt"
    for name, dest in ((f"set-{letter}.tar.gz", tgz), (f"Outcomes-{letter}.txt", outc)):
        if dest.exists():
            continue
        err = None
        for base in P12_BASES:
            try:
                log(f"  downloading {base}/{name}")
                download(f"{base}/{name}", dest)
                err = None
                break
            except Exception as e:
                err = e
        if err:
            raise RuntimeError(f"PhysioNet 2012 download failed ({err}). On Kaggle turn Internet on, or attach "
                               "the files (set-a, set-b folders + Outcomes-a/b.txt) as a dataset.")
    target = data_dir / f"extract-{letter}"
    with tarfile.open(tgz, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.name.endswith(".txt") and not m.name.startswith("/") and ".." not in m.name]
        try:
            tf.extractall(target, members=members, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(target, members=members)
    best, best_n = None, 0
    for d in [target, *[p for p in target.rglob("*") if p.is_dir()]]:
        n = len(_p12_records(d))
        if n > best_n:
            best, best_n = d, n
    if best_n < 100:
        raise RuntimeError(f"extraction of set-{letter} looks wrong: {best_n} records in {target}")
    log(f"  set-{letter}: {best_n} records extracted")
    return best, outc


def load_physionet2012(data_dir, extra_roots=(), log=print):
    data_dir = Path(data_dir) / "physionet2012"
    data_dir.mkdir(parents=True, exist_ok=True)
    roots = [data_dir, *extra_roots]
    rows, outs = [], []
    for letter in ("a", "b"):
        rec_dir, outc = _p12_get_set(letter, data_dir, roots, log)
        if letter == "a":
            memfile = outc       # Outcomes-a.txt: the public file with the outcomes (memorization check)
        recs = _p12_records(rec_dir)
        if len(recs) < P12_MIN_PER_SET:
            raise RuntimeError(f"PhysioNet 2012 set-{letter}: {len(recs)} records found, expected 4,000")
        rows += [p12_read_record(fp) for fp in recs]
        o = pd.read_csv(outc)
        o.columns = [c.strip() for c in o.columns]
        o = o[["RecordID", "In-hospital_death"]].rename(columns={"RecordID": "id", "In-hospital_death": "y"})
        o["id"] = o["id"].astype(int).astype(str)
        outs.append(o)
    feats = pd.DataFrame(rows)
    df = feats.merge(pd.concat(outs, ignore_index=True), on="id", how="inner")
    df = df[df["y"].isin([0, 1])].reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    feat_cols = [c for c in df.columns if c not in ("id", "y")]
    df[feat_cols] = df[feat_cols].astype(float)
    info = {"records_parsed": len(feats), "with_outcome": len(df), "positives": int(df["y"].sum()),
            "memcheck_file": str(memfile)}
    return df[["id", "y", *feat_cols]], feat_cols, [], info


# ---------------------------------------------------------------------------------------- PhysioNet 2019
P19_S3 = "https://physionet-open.s3.amazonaws.com"
P19_PREFIX = "challenge-2019/1.0.0/training/"
P19_WEB = "https://physionet.org/files/challenge-2019/1.0.0/training"
P19_VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
            "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
            "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
P19_STATIC = ["Age", "Gender", "HospAdmTime"]
WINDOW_H = 24


def p19_features(path):
    """One patient file -> (features or None, status). Status: pos | neg | excl_label_le24 | excl_short | excl_empty."""
    d = pd.read_csv(path, sep="|")
    if len(d) == 0:
        return None, "excl_empty"
    hrs = d["ICULOS"].to_numpy(dtype=float)
    lab = d["SepsisLabel"].to_numpy(dtype=float)
    win = hrs <= WINDOW_H
    if (lab[win] == 1).any():
        return None, "excl_label_le24"
    pos = bool((lab[~win] == 1).any())
    if not pos and np.nanmax(hrs) < WINDOW_H + 1:
        return None, "excl_short"
    w = d.loc[win]
    if len(w) == 0:
        return None, "excl_empty"
    f = {"id": Path(path).stem}
    for v in P19_VARS:
        col = w[v].to_numpy(dtype=float) if v in w else np.array([np.nan])
        m = ~np.isnan(col)
        if m.any():
            f[f"{v}_min"], f[f"{v}_max"], f[f"{v}_last"] = float(col[m].min()), float(col[m].max()), float(col[m][-1])
        else:
            f[f"{v}_min"] = f[f"{v}_max"] = f[f"{v}_last"] = np.nan
    first = d.iloc[0]
    for v in P19_STATIC:
        f[v] = float(first[v]) if v in d and pd.notna(first[v]) else np.nan
    f["y"] = int(pos)
    return f, ("pos" if pos else "neg")


P19_MIN_PER_SET = int(os.environ.get("LER_P19_MIN_PER_SET", 20000))   # set A has 20,336 files, set B 20,000
P12_MIN_PER_SET = int(os.environ.get("LER_P12_MIN_PER_SET", 4000))    # sets A and B have 4,000 records each


def _p19_files(roots):
    """Complete local copies only: {set_name: folder} for folders with >= P19_MIN_PER_SET .psv files
    (a partial download is never used)."""
    out = {}
    for d, n in find_dirs_with(roots, "*.psv", P19_MIN_PER_SET):
        name = "training_setB" if "setB" in str(d) else ("training_setA" if "setA" in str(d) else str(d))
        out.setdefault(name, d)
    return out


def _p19_download(data_dir: Path, log):
    pairs = []
    try:
        keys = s3_list(P19_S3, P19_PREFIX)
        keys = [k for k in keys if k.endswith(".psv")]
        log(f"  S3 listing: {len(keys)} .psv files")
        for k in keys:
            rel = k[len(P19_PREFIX):]
            pairs.append((f"{P19_S3}/{urllib.parse.quote(k)}", data_dir / rel))
    except Exception as e:
        log(f"  S3 listing failed ({e}); trying the PhysioNet web listing")
        for s in ("training_setA", "training_setB"):
            html = http_get(f"{P19_WEB}/{s}/").decode("utf-8", "replace")
            names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
            log(f"  {s}: {len(names)} files listed")
            pairs += [(f"{P19_WEB}/{s}/{n}", data_dir / s / n) for n in names]
    if len(pairs) < 30000:
        raise RuntimeError(f"PhysioNet 2019 listing found only {len(pairs)} files (expected ~40,336).")
    t0 = time.time()
    try:
        n = parallel_download(pairs)
    except Exception as e:
        raise RuntimeError(f"PhysioNet 2019 download stopped ({e}). Run this cell again: files already "
                           "downloaded are kept and only the missing ones are fetched.") from e
    missing = [d for _, d in pairs if not (Path(d).exists() and Path(d).stat().st_size > 0)]
    if missing:
        raise RuntimeError(f"{len(missing)} PhysioNet 2019 files are missing after the download; run the cell again.")
    log(f"  downloaded {n} files in {time.time() - t0:.0f}s ({len(pairs)} in total)")


def load_physionet2019(data_dir, extra_roots=(), log=print, workers: int = 4):
    data_dir = Path(data_dir) / "physionet2019"
    data_dir.mkdir(parents=True, exist_ok=True)
    roots = [data_dir, *extra_roots]
    sets = _p19_files(roots)
    if len(sets) < 2:   # partial copies are ignored, so a broken download is completed, never used
        log("  PhysioNet 2019 not found locally: downloading the two training sets (~40k small files)")
        _p19_download(data_dir, log)
        sets = _p19_files([data_dir])
    files = sorted(p for d in sets.values() for p in d.glob("*.psv"))
    log(f"  {len(files)} patient files in {len(sets)} folders")
    rows, status = [], defaultdict(int)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(p19_features, files, chunksize=256))
    else:
        results = [p19_features(p) for p in files]
    for f, st in results:
        status[st] += 1
        if f is not None:
            rows.append(f)
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in ("id", "y")]
    df[feat_cols] = df[feat_cols].astype(float)
    set_a = sets.get("training_setA")
    memfile = None
    if set_a is not None:
        memfile = Path(set_a) / "p000001.psv"
        if not memfile.exists():
            memfile = sorted(Path(set_a).glob("*.psv"))[0]
    info = {"files": len(files), **{f"status_{k}": v for k, v in sorted(status.items())},
            "included": len(df), "positives": int(df["y"].sum()), "memcheck_file": str(memfile)}
    return df[["id", "y", *feat_cols]], feat_cols, [], info


# ------------------------------------------------------------------------------------------- Diabetes 130
D130_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
D130_EXCLUDE_DISPOSITION = {"11", "13", "14", "19", "20", "21"}   # death or hospice
D130_NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                "number_emergency", "number_inpatient", "number_diagnoses"]
D130_ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
D130_DROP = ["encounter_id", "patient_nbr", "readmitted"]


def parse_ids_mapping(text: str) -> dict:
    """IDS_mapping.csv holds three stacked tables ('<name>_id,description' headers, ',' separator lines)."""
    maps, cur = {}, None
    for row in csv.reader(io.StringIO(text)):
        row = [c.strip() for c in row]
        if not row or all(c == "" for c in row):
            continue
        if row[0].endswith("_id") and len(row) > 1 and row[1].lower() == "description":
            cur = row[0]
            maps[cur] = {}
            continue
        if cur is None or row[0] == "":
            continue
        maps[cur][row[0]] = ",".join(row[1:]).strip()
    return maps


def _ci_find(roots, name):
    low = name.lower()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name.lower() == low:
                return p
    return None


def load_diabetes130(data_dir, extra_roots=(), log=print):
    data_dir = Path(data_dir) / "diabetes130"
    data_dir.mkdir(parents=True, exist_ok=True)
    roots = [data_dir, *extra_roots]
    csv_path, map_path = _ci_find(roots, "diabetic_data.csv"), _ci_find(roots, "IDS_mapping.csv")
    if csv_path is None or map_path is None:
        zpath = data_dir / "diabetes130.zip"
        if not zpath.exists():
            log(f"  downloading {D130_URL}")
            download(D130_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                if ".." in m or m.startswith("/"):
                    continue
                if m.lower().endswith(("diabetic_data.csv", "ids_mapping.csv")):
                    z.extract(m, data_dir)
        csv_path, map_path = _ci_find([data_dir], "diabetic_data.csv"), _ci_find([data_dir], "IDS_mapping.csv")
        if csv_path is None or map_path is None:
            raise RuntimeError("diabetic_data.csv / IDS_mapping.csv not found after extracting the UCI zip")
    log(f"  using {csv_path}")
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=["?", ""])
    maps = parse_ids_mapping(map_path.read_text(encoding="utf-8", errors="replace"))
    n_raw = len(raw)
    raw["_enc"] = pd.to_numeric(raw["encounter_id"])
    raw = raw.sort_values("_enc", kind="mergesort")
    first = raw.drop_duplicates("patient_nbr", keep="first")
    keep = ~first["discharge_disposition_id"].astype(str).str.strip().isin(D130_EXCLUDE_DISPOSITION)
    df = first[keep].copy()
    df["y"] = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)
    df["id"] = df["encounter_id"].astype(str)
    feat_cols = [c for c in raw.columns if c not in D130_DROP and c != "_enc"]
    cat_cols = [c for c in feat_cols if c not in D130_NUMERIC]
    for c in D130_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat_cols:
        vals = df[c].astype(object).where(df[c].notna(), None)
        df[c] = pd.Categorical([None if v is None else str(v).strip() for v in vals])
    info = {"encounters": n_raw, "first_encounters": len(first),
            "excluded_death_or_hospice": int((~keep).sum()), "included": len(df), "positives": int(df["y"].sum()),
            "id_mapping_tables": {k: len(v) for k, v in maps.items()}, "memcheck_file": str(csv_path)}
    out = df[["id", "y", *feat_cols]].reset_index(drop=True)
    out.attrs["id_maps"] = maps
    return out, feat_cols, cat_cols, info


LOADERS = {"physionet2012": load_physionet2012, "physionet2019": load_physionet2019,
           "diabetes130": load_diabetes130}


# =============================================================================================================
# 4. Display names and values (what the LLM reads)
# =============================================================================================================
P12_INFO = {
    "Albumin": ("Albumin", "g/dL"), "ALP": ("Alkaline phosphatase", "IU/L"), "ALT": ("ALT liver enzyme", "IU/L"),
    "AST": ("AST liver enzyme", "IU/L"), "Bilirubin": ("Bilirubin", "mg/dL"), "BUN": ("Blood urea nitrogen", "mg/dL"),
    "Cholesterol": ("Cholesterol", "mg/dL"), "Creatinine": ("Creatinine", "mg/dL"),
    "DiasABP": ("Diastolic arterial blood pressure (invasive)", "mmHg"),
    "FiO2": ("Inspired oxygen fraction FiO2", "0-1"), "GCS": ("Glasgow Coma Scale", "3-15"),
    "Glucose": ("Glucose", "mg/dL"), "HCO3": ("Bicarbonate", "mmol/L"), "HCT": ("Hematocrit", "%"),
    "HR": ("Heart rate", "beats/min"), "K": ("Potassium", "mEq/L"), "Lactate": ("Lactate", "mmol/L"),
    "Mg": ("Magnesium", "mmol/L"), "MAP": ("Mean arterial pressure (invasive)", "mmHg"), "Na": ("Sodium", "mEq/L"),
    "NIDiasABP": ("Diastolic blood pressure (cuff)", "mmHg"), "NIMAP": ("Mean blood pressure (cuff)", "mmHg"),
    "NISysABP": ("Systolic blood pressure (cuff)", "mmHg"), "PaCO2": ("Arterial CO2 PaCO2", "mmHg"),
    "PaO2": ("Arterial oxygen PaO2", "mmHg"), "pH": ("Arterial pH", ""), "Platelets": ("Platelets", "x1000/uL"),
    "RespRate": ("Respiratory rate", "breaths/min"), "SaO2": ("Oxygen saturation SaO2", "%"),
    "SysABP": ("Systolic arterial blood pressure (invasive)", "mmHg"), "Temp": ("Temperature", "deg C"),
    "TroponinI": ("Troponin I", "ug/L"), "TroponinT": ("Troponin T", "ug/L"),
    "WBC": ("White blood cell count", "x1000/uL"), "Weight": ("Weight", "kg"),
}
P19_INFO = {
    "HR": ("Heart rate", "beats/min"), "O2Sat": ("Pulse oximetry O2 saturation", "%"), "Temp": ("Temperature", "deg C"),
    "SBP": ("Systolic blood pressure", "mmHg"), "MAP": ("Mean arterial pressure", "mmHg"),
    "DBP": ("Diastolic blood pressure", "mmHg"), "Resp": ("Respiratory rate", "breaths/min"),
    "EtCO2": ("End-tidal CO2", "mmHg"), "BaseExcess": ("Base excess", "mmol/L"), "HCO3": ("Bicarbonate", "mmol/L"),
    "FiO2": ("Inspired oxygen fraction FiO2", "0-1"), "pH": ("Arterial pH", ""), "PaCO2": ("Arterial CO2 PaCO2", "mmHg"),
    "SaO2": ("Arterial oxygen saturation SaO2", "%"), "AST": ("AST liver enzyme", "IU/L"),
    "BUN": ("Blood urea nitrogen", "mg/dL"), "Alkalinephos": ("Alkaline phosphatase", "IU/L"),
    "Calcium": ("Calcium", "mg/dL"), "Chloride": ("Chloride", "mmol/L"), "Creatinine": ("Creatinine", "mg/dL"),
    "Bilirubin_direct": ("Direct bilirubin", "mg/dL"), "Glucose": ("Glucose", "mg/dL"), "Lactate": ("Lactate", ""),
    "Magnesium": ("Magnesium", ""), "Phosphate": ("Phosphate", "mg/dL"), "Potassium": ("Potassium", "mmol/L"),
    "Bilirubin_total": ("Total bilirubin", "mg/dL"), "TroponinI": ("Troponin I", "ng/mL"), "Hct": ("Hematocrit", "%"),
    "Hgb": ("Hemoglobin", "g/dL"), "PTT": ("Partial thromboplastin time", "seconds"),
    "WBC": ("White blood cell count", "x1000/uL"), "Fibrinogen": ("Fibrinogen", "mg/dL"),
    "Platelets": ("Platelets", "x1000/uL"),
}
STAT_WORD = {"min": "lowest", "max": "highest", "last": "last value"}
P12_ICU = {1: "coronary care unit", 2: "cardiac surgery recovery unit", 3: "medical ICU", 4: "surgical ICU"}
D130_NAMES = {
    "race": "Race", "gender": "Sex", "age": "Age group", "weight": "Weight (lb)",
    "admission_type_id": "Admission type", "discharge_disposition_id": "Discharge disposition",
    "admission_source_id": "Admission source", "time_in_hospital": "Days in hospital", "payer_code": "Payer code",
    "medical_specialty": "Admitting physician specialty", "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures (other than lab tests)",
    "num_medications": "Number of distinct medications", "number_outpatient": "Outpatient visits in the prior year",
    "number_emergency": "Emergency visits in the prior year", "number_inpatient": "Inpatient admissions in the prior year",
    "diag_1": "Primary diagnosis (ICD-9 code)", "diag_2": "Secondary diagnosis (ICD-9 code)",
    "diag_3": "Additional secondary diagnosis (ICD-9 code)", "number_diagnoses": "Number of diagnoses",
    "max_glu_serum": "Glucose serum test result", "A1Cresult": "HbA1c test result",
    "change": "Change in diabetes medications", "diabetesMed": "Diabetes medication prescribed",
}
D130_DRUG_VALUES = {"No": "not prescribed", "Steady": "prescribed, dose unchanged", "Up": "dose increased",
                    "Down": "dose decreased"}


def _num(v) -> str:
    v = float(v)
    a = abs(v)
    s = f"{v:.0f}" if a >= 100 else (f"{v:.1f}" if a >= 10 else f"{v:.2f}")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _isna(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (not isinstance(v, str) and pd.isna(v))


def display_name(ds: str, col: str) -> str:
    if ds == "physionet2012":
        special = {"Age": "Age (years)", "Gender": "Sex", "Height": "Height (cm)", "ICUType": "ICU type",
                   "Urine_total": "Urine output, total over 48 h (mL)", "MechVent_any": "Mechanical ventilation"}
        if col in special:
            return special[col]
        base, _, stat = col.rpartition("_")
        name, unit = P12_INFO.get(base, (base, ""))
        return f"{name}, {STAT_WORD.get(stat, stat)}" + (f" ({unit})" if unit else "")
    if ds == "physionet2019":
        special = {"Age": "Age (years)", "Gender": "Sex",
                   "HospAdmTime": "Hospital admission time relative to ICU admission (hours; negative = before)"}
        if col in special:
            return special[col]
        base, _, stat = col.rpartition("_")
        name, unit = P19_INFO.get(base, (base, ""))
        return f"{name}, {STAT_WORD.get(stat, stat)} in the first 24 h" + (f" ({unit})" if unit else "")
    if ds == "diabetes130":
        if col in D130_NAMES:
            return D130_NAMES[col]
        return f"{col.replace('-', ' and ').capitalize()} (diabetes drug)"
    raise ValueError(ds)


def display_value(ds: str, col: str, v, id_maps=None) -> str:
    if ds in ("physionet2012", "physionet2019"):
        if _isna(v):
            return "not measured"
        v = float(v)
        if col == "Gender":
            return "male" if v >= 0.5 else "female"
        if ds == "physionet2012" and col == "ICUType":
            return P12_ICU.get(int(round(v)), _num(v))
        if ds == "physionet2012" and col == "MechVent_any":
            return "yes" if v >= 0.5 else "no"
        return _num(v)
    if ds == "diabetes130":
        if _isna(v):
            return "not recorded"
        s = str(v).strip()
        if col in D130_NUMERIC:
            return _num(float(s))
        if col in D130_ID_COLS:
            desc = (id_maps or {}).get(col, {}).get(s)
            return desc if desc and desc.upper() not in ("NULL", "NAN") else ("not recorded" if desc else s)
        if col in ("max_glu_serum", "A1Cresult"):
            return {"None": "not measured", "Norm": "normal"}.get(s, s)
        if col == "age":
            m = re.match(r"\[(\d+)-(\d+)\)", s)
            return f"{m.group(1)}-{m.group(2)} years" if m else s
        if col == "change":
            return {"Ch": "yes", "No": "no"}.get(s, s)
        if col in D130_NAMES:
            return s
        return D130_DRUG_VALUES.get(s, s)
    raise ValueError(ds)


# =============================================================================================================
# 5. Splits, XGBoost, SHAP cards, examples
# =============================================================================================================
def make_xgb(spw: float, categorical: bool):
    import xgboost as xgb
    kw = dict(XGB_PARAMS, scale_pos_weight=spw, eval_metric="auc")
    if categorical:
        kw["enable_categorical"] = True
    return xgb.XGBClassifier(**kw)


def shap_values(model, X: pd.DataFrame, categorical: bool):
    """Exact TreeSHAP from XGBoost (log-odds units): one value per column."""
    import xgboost as xgb
    dm = xgb.DMatrix(X, feature_names=list(X.columns), enable_categorical=categorical)
    contrib = model.get_booster().predict(dm, pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]


def n_measured(X: pd.DataFrame) -> np.ndarray:
    return X.notna().sum(axis=1).to_numpy()


def _fp(v) -> str:
    return hashlib.sha256(json.dumps(v, default=str, sort_keys=True).encode()).hexdigest()[:16]


class Bundle:
    """Everything for one dataset: splits, the 16-feature XGBoost, the dev sample (always built, for the dev
    fingerprints) and, with sample="test", the test sample with its cards and C6 partners."""

    def __init__(self, name, df, feat_cols, cat_cols, data_info, pilot_ids=(), log=print, sample="dev"):
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        self.name, self.spec = name, SPEC[name]
        self.id_maps = df.attrs.get("id_maps", {})
        self.categorical = len(cat_cols) > 0
        self.info = {"data": data_info}
        # fixed row order (sorted by id) so the dev and main runs reproduce the same splits
        df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
        y = df["y"].to_numpy()
        idx = np.arange(len(df))
        tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=SPLIT_SEED)
        dev, test = train_test_split(rest, test_size=0.75, stratify=y[rest], random_state=SPLIT_SEED)
        pilot = set(str(p) for p in pilot_ids)
        n_dev0, n_test0 = len(dev), len(test)
        dev = np.array([i for i in dev if df.at[i, "id"] not in pilot], dtype=int)
        test = np.array([i for i in test if df.at[i, "id"] not in pilot], dtype=int)
        self.info["split"] = {"train": len(tr), "dev_pool": len(dev), "test_pool": len(test),
                              "pilot_removed_dev": n_dev0 - len(dev), "pilot_removed_test": n_test0 - len(test),
                              "train_positives": int(y[tr].sum())}
        self.df, self.tr, self.dev_pool, self.test_pool = df, tr, dev, test
        X = df[feat_cols]
        spw = float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum()))
        full = make_xgb(spw, self.categorical).fit(X.iloc[tr], y[tr])
        sv_tr, _ = shap_values(full, X.iloc[tr], self.categorical)
        imp = pd.Series(np.abs(sv_tr).mean(axis=0), index=feat_cols).sort_values(ascending=False, kind="mergesort")
        self.features = list(imp.index[:N_FEATURES])
        names = [display_name(name, c) for c in self.features]
        if len(set(names)) != len(names):
            raise RuntimeError(f"{name}: two selected features share a display name: {names}")
        self.model = make_xgb(spw, self.categorical).fit(X.iloc[tr][self.features], y[tr])
        p_dev = self.model.predict_proba(X.iloc[dev][self.features])[:, 1]
        self.info["xgb"] = {"auroc_dev_pool_16": float(roc_auc_score(y[dev], p_dev)),
                            "auroc_dev_pool_all": float(roc_auc_score(y[dev], full.predict_proba(X.iloc[dev])[:, 1])),
                            "scale_pos_weight": spw, "features": self.features,
                            "mean_abs_shap": {c: float(imp[c]) for c in self.features}}
        # eligibility (>= 10 of the 16 selected features measured)
        meas = n_measured(X[self.features])
        self.eligible = meas >= MIN_MEASURED
        cap = {}
        for pool_name, pool in (("dev_pool", dev), ("test_pool", test)):
            el = pool[self.eligible[pool]]
            cap[pool_name] = {"eligible": len(el), "eligible_pos": int(y[el].sum()),
                              "eligible_neg": int((y[el] == 0).sum())}
        npos = cap["test_pool"]["eligible_pos"]
        cap["test_per_class_planned"] = TEST_PER_CLASS if npos >= TEST_PER_CLASS else npos
        cap["test_fallback_rule_applies"] = npos < TEST_PER_CLASS
        self.info["capacity"] = cap
        # dev sample: DEV_PER_CLASS per class from eligible dev-pool patients
        el = dev[self.eligible[dev]]
        pos, neg = el[y[el] == 1], el[y[el] == 0]
        k = min(DEV_PER_CLASS, len(pos), len(neg))
        if k < 10:
            raise RuntimeError(f"{name}: only {k} eligible dev patients per class; check the data step")
        rs = np.random.RandomState(SPLIT_SEED)
        pick = np.concatenate([rs.choice(pos, k, replace=False), rs.choice(neg, k, replace=False)])
        pick = pick[rs.permutation(len(pick))]
        self.patients = pick
        self.info["dev_sample"] = {"per_class": int(k), "n": int(len(pick))}
        self._build_sample(pick)
        self.info["cards"] = self._card_info()
        # solved examples: 2 alarming + 2 safe from the training split, per example set
        self.shots, self.shot_ids = {}, {}
        tr_sorted = np.sort(tr)
        for s, seed in SHOT_SEEDS.items():
            r = np.random.RandomState(seed)
            a = r.choice(tr_sorted[y[tr_sorted] == 1], 2, replace=False)
            b = r.choice(tr_sorted[y[tr_sorted] == 0], 2, replace=False)
            rows = [a[0], b[0], a[1], b[1]]   # alternating, alarming outcome first
            self.shots[s] = [({c: X.at[i, c] for c in self.features}, int(y[i])) for i in rows]
            self.shot_ids[s] = [df.at[i, "id"] for i in rows]
        # second study (not part of the fingerprints): "Asf" = the examples of set A in the order safe, alarming,
        # safe, alarming (example-order control); "Abase" = set A unchanged, for the base-format control (the run
        # renders these prompts as plain text, as for a base model)
        self.shots["Asf"] = [self.shots["A"][i] for i in (1, 0, 3, 2)]
        self.shots["Abase"] = list(self.shots["A"])
        fp = {"dev_pool": sorted(df["id"].iloc[dev]), "test_pool": sorted(df["id"].iloc[test]),
              "features": self.features,
              "cards": {r: [(c["col"], round(c["shap"], 6)) for c in self.cards[r]] for r in sorted(self.ids)},
              "shots": self.shot_ids}
        self.info["fingerprint"] = {k: _fp(v) for k, v in fp.items()}
        self.dev_ids = list(self.ids)
        self.sample = "dev"
        self._samples = {"dev": self._state()}
        if sample == "test":
            # test set (Section 3): 300 per class from the eligible test pool, or all eligible positives and as
            # many negatives if there are fewer than 300 (the fallback rule), seed 2027
            el = test[self.eligible[test]]
            pos, neg = el[y[el] == 1], el[y[el] == 0]
            k = min(TEST_PER_CLASS, len(pos), len(neg))
            rs = np.random.RandomState(SPLIT_SEED)
            pick = np.concatenate([rs.choice(pos, k, replace=False), rs.choice(neg, k, replace=False)])
            pick = pick[rs.permutation(len(pick))]
            self.patients = pick
            self.info["test_sample"] = {"per_class": int(k), "n": int(len(pick)),
                                        "fallback_rule_applied": bool(k < TEST_PER_CLASS)}
            self._build_sample(pick)
            self.info["test_cards"] = self._card_info()
            self.info["fingerprint"]["test_ids"] = _fp(sorted(self.ids))
            self.info["fingerprint"]["test_cards"] = _fp(
                {r: [(c["col"], round(c["shap"], 6)) for c in self.cards[r]] for r in sorted(self.ids)})
            self.info["fingerprint"]["test_partners"] = _fp({r: self.partner[r] for r in sorted(self.ids)})
            self.sample = "test"
            self._samples["test"] = self._state()
        log(f"  {name}: train {len(tr)}, dev pool {len(dev)}, test pool {len(test)} | XGBoost AUROC (dev pool, "
            f"16 features) {self.info['xgb']['auroc_dev_pool_16']:.3f} | {self.sample} sample {len(self.ids)} | "
            f"eligible test positives {npos}{' -> FALLBACK RULE' if npos < TEST_PER_CLASS else ''}")

    def _build_sample(self, pick):
        """Cards, values, outcomes and C6 partners for the sampled patients (dev or test)."""
        df, X, y = self.df, self.df[self.features], self.df["y"].to_numpy()
        Xp = X.iloc[pick]
        sv, _ = shap_values(self.model, Xp, self.categorical)
        self.xgb_p = dict(zip(df["id"].iloc[pick], self.model.predict_proba(Xp)[:, 1]))
        self.cards, self.card_dir, self.values = {}, {}, {}
        for i, row_i in enumerate(pick):
            rid = df.at[row_i, "id"]
            order = np.argsort(-np.abs(sv[i]), kind="mergesort")[:TOP_K]
            self.cards[rid] = [{"col": self.features[j], "value": Xp.iloc[i, j], "shap": float(sv[i, j])}
                               for j in order]
            self.card_dir[rid] = int(sum(c["shap"] for c in self.cards[rid]) > 0)
            self.values[rid] = {c: Xp.iloc[i][c] for c in self.features}
        self.ids = [df.at[i, "id"] for i in pick]
        self.y = {df.at[i, "id"]: int(y[i]) for i in pick}
        # C6 partners: a random patient of the same sample whose card points the other way (seed 2027)
        prs = np.random.RandomState(SPLIT_SEED)
        self.partner = {}
        for rid in sorted(self.ids):
            opp = sorted(o for o in self.ids if self.card_dir[o] != self.card_dir[rid])
            self.partner[rid] = opp[prs.randint(len(opp))] if opp else None

    _STATE = ("patients", "cards", "card_dir", "values", "ids", "y", "xgb_p", "partner")

    def _state(self):
        return {k: getattr(self, k) for k in self._STATE}

    def use_sample(self, which):
        """Switch prompts, cards and partners between the dev and the test sample."""
        for k, v in self._samples[which].items():
            setattr(self, k, v)
        self.sample = which

    def _card_info(self):
        agree = np.mean([self.card_dir[r] == int(self.xgb_p[r] >= 0.5) for r in self.ids])
        return {"card_points_alarm_share": float(np.mean(list(self.card_dir.values()))),
                "card_agrees_with_xgb_label": float(agree),
                "partners_missing": int(sum(v is None for v in self.partner.values()))}

    # ------------------------------------------------------------------------------------------- prompts
    def _block(self, values: dict) -> str:
        return "\n".join(f"- {display_name(self.name, c)}: {display_value(self.name, c, values[c], self.id_maps)}"
                         for c in self.features)

    def question(self, framing: str) -> str:
        if is_choice(framing):          # second study: the two outcomes as options A and B
            o = CHOICE_OPTIONS[self.name]
            first, second = (o["alarm"], o["safe"]) if positive_means_alarm(framing) else (o["safe"], o["alarm"])
            return f"{CHOICE_QUESTION}\nA: {first}\nB: {second}"
        return self.spec["q_alarm"] if framing == "alarm" else self.spec["q_reversed"]

    @staticmethod
    def yes_means_alarm(framing: str) -> bool:
        return framing == "alarm"

    def card_text(self, card) -> str:
        lines = []
        for k, it in enumerate(card, 1):
            word = "increases" if it["shap"] > 0 else "decreases"
            imp = f"{abs(it['shap']):.2f}"
            imp = ("+" if it["shap"] > 0 else "-") + imp      # the printed sign always matches the word
            lines.append(f"{k}. {display_name(self.name, it['col'])} = "
                         f"{display_value(self.name, it['col'], it['value'], self.id_maps)}: {word} the risk of "
                         f"{self.spec['outcome']} (impact {imp})")
        return EVIDENCE_HEADER + "\n" + "\n".join(lines)

    def shown_card(self, rid, cond):
        if cond == "C0":
            return None
        if cond == "C2":
            return self.cards[rid]
        if cond == "C4":
            return [dict(it, shap=-it["shap"]) for it in self.cards[rid]]
        if cond == "C6":
            other = self.partner[rid]
            return None if other is None else self.cards[other]
        raise ValueError(cond)

    def prompt(self, rid, cond, framing, shot_set):
        """-> (text, card_dir) ; card_dir = 1 if the shown card points to the alarming outcome, None in C0.
        Returns (None, None) when C6 has no partner."""
        q = self.question(framing)
        pos_word, neg_word = answer_pair(framing)          # 'Yes'/'No' (first study) or 'A'/'B' (second study)
        ex = []
        for k, (vals, yy) in enumerate(self.shots[shot_set], 1):
            pos = (yy == 1) if positive_means_alarm(framing) else (yy == 0)
            ex.append(f"Example {k}\n{self._block(vals)}\nQuestion: {q}\nAnswer: {pos_word if pos else neg_word}")
        parts = [f"Task: {self.spec['task']}", EXAMPLES_HEADER, *ex, PATIENT_HEADER + "\n" + self._block(self.values[rid])]
        card = self.shown_card(rid, cond)
        if cond == "C6" and card is None:
            return None, None
        if card is not None:
            parts.append(self.card_text(card))
        parts.append(f"Question: {q}\n{CHOICE_INSTRUCTION if is_choice(framing) else ANSWER_INSTRUCTION}")
        cdir = None if card is None else int(sum(it["shap"] for it in card) > 0)
        return "\n\n".join(parts), cdir


def render(text: str, kind: str, tokenizer=None) -> str:
    """Instruct: whole text as one user message through the chat template + 'Answer:' prefill.
    Base: text + '\\nAnswer:'."""
    if kind == "base":
        bos = getattr(tokenizer, "bos_token", None) if tokenizer is not None else None
        return (bos or "") + text + "\n" + PREFILL
    msgs = [{"role": "user", "content": text}]
    try:
        r = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        r = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return r + PREFILL


def template_system_text(tokenizer) -> str | None:
    """Text the chat template adds before the user turn when no system message is given (e.g. Qwen2.5's
    default system prompt), or None. Logged because the protocol says 'no system message'."""
    try:
        r = tokenizer.apply_chat_template([{"role": "user", "content": "XYZ_USER_XYZ"}], tokenize=False,
                                          add_generation_prompt=False)
    except Exception:
        return None
    head = r.split("XYZ_USER_XYZ")[0]
    head = re.sub(r"<\|[^|>]*\|>", "\n", head)
    lines = [ln.strip() for ln in head.splitlines()]
    lines = [ln for ln in lines if ln and ln.lower() not in ("system", "user", "assistant")]
    return " ".join(lines) if lines else None


# =============================================================================================================
# 6. Model backends: next-token Yes/No probabilities after "Answer:"
# =============================================================================================================
class NonFiniteLogprobError(RuntimeError):
    pass


def find_word_variants(tokenizer, words=("Yes", "No")):
    """Single-token ids for 'W' and ' W' (byte-level 'ĠW' or sentencepiece '▁W')."""
    vocab = tokenizer.get_vocab()
    variants, strings = {}, {}
    for w in words:
        ids = []
        for cand in (w, "Ġ" + w, "▁" + w):
            tid = vocab.get(cand)
            if tid is None or tid in ids:
                continue
            if tokenizer.decode([tid]).strip() == w:
                ids.append(int(tid))
                strings[int(tid)] = tokenizer.decode([tid])
        if not ids:
            raise RuntimeError(f"no single-token variant of '{w}' in this tokenizer")
        variants[w] = ids
    return variants, strings


def eos_token_ids(model_path, tokenizer):
    ids = []
    try:
        from transformers import GenerationConfig
        e = GenerationConfig.from_pretrained(model_path).eos_token_id
        ids += list(e) if isinstance(e, (list, tuple)) else ([e] if e is not None else [])
    except Exception:
        pass
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(int(i) for i in ids))


def download_model(model_id, cache_dir, revision=None, token=None):
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi(token=token).model_info(model_id, revision=revision).sha
    path = snapshot_download(model_id, revision=sha, cache_dir=cache_dir, token=token,
                             allow_patterns=["*.json", "model*.safetensors", "*.model", "tokenizer*", "*.txt",
                                             "*.jinja", "*.tiktoken"])
    return path, sha


def logsumexp(vals) -> float:
    """log(sum(exp(v))) over the given log-probabilities; nan if any value is nan or +inf."""
    vals = [float(v) for v in vals]
    if not vals or any(math.isnan(v) or v == math.inf for v in vals):
        return float("nan")
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals))


class Backend:
    """Reads the next-token log-probabilities of the answer-word variant tokens after 'Answer:' (Deviation D2).
    The first study reads 'Yes'/'No'; the second study also reads 'A'/'B' (answer-option framing). Each reading
    requests only the variant ids of the words it needs, so a Yes/No reading is the same as in main-1.0."""
    name = "base"
    read = "exact"

    def ensure_words(self, words):
        """Adds the single-token variants of words not read so far (e.g. 'A', 'B')."""
        missing = [w for w in words if w not in self.variants]
        if missing:
            v, s = find_word_variants(self.tokenizer, tuple(missing))
            self.variants.update(v)
            self.strings.update(s)

    def wanted_ids(self, words=("Yes", "No")):
        self.ensure_words(words)
        return sorted({t for w in words for t in self.variants[w]})

    def word_logprobs(self, prompts, words=("Yes", "No")):
        """-> list of (dict token_id -> log-probability for every variant id of `words`, number of imputed variants)."""
        raise NotImplementedError

    def answer_logprobs(self, prompts, pair=("Yes", "No")):
        """-> list of dicts {lp_pos, lp_neg, n_imputed}: log of the summed variant probabilities of the first and the
        second answer word of `pair`."""
        out = []
        for d, n_imp in self.word_logprobs(prompts, words=tuple(pair)):
            out.append({"lp_pos": logsumexp([d[t] for t in self.variants[pair[0]]]),
                        "lp_neg": logsumexp([d[t] for t in self.variants[pair[1]]]), "n_imputed": int(n_imp)})
        return out

    def yes_no(self, prompts):
        """-> list of dicts {lp_yes, lp_no, n_imputed}: log of the summed variant probabilities (first study)."""
        return [{"lp_yes": r["lp_pos"], "lp_no": r["lp_neg"], "n_imputed": r["n_imputed"]}
                for r in self.answer_logprobs(prompts, ("Yes", "No"))]

    def read_framing(self, prompts, framing):
        """Reading for prompts of one framing -> dicts {lp_yes, lp_no, n_imputed}; in the answer-option framings
        lp_yes / lp_no hold the log-probabilities of 'A' / 'B' (the record says which words were read)."""
        return [{"lp_yes": r["lp_pos"], "lp_no": r["lp_neg"], "n_imputed": r["n_imputed"]}
                for r in self.answer_logprobs(prompts, answer_pair(framing))]

    def generate_text(self, token_id_lists, max_tokens):
        raise NotImplementedError

    def info(self):
        return {"backend": self.name, "read": self.read,
                "variants": {w: [[i, self.strings[i]] for i in ids] for w, ids in self.variants.items()}}


class VLLMBackend(Backend):
    """vLLM. read="token_ids": exact log-probabilities of the variant ids (SamplingParams.logprob_token_ids,
    vLLM >= 0.2x). read="topk": the top TOPK_FALLBACK log-probabilities; a variant outside them gets the
    log-probability of the last returned token (an upper bound) and is counted as imputed."""
    name = "vllm"

    def __init__(self, model_path, tensor_parallel_size=1, max_model_len=MAX_MODEL_LEN, seed=SPLIT_SEED,
                 read="token_ids", topk=TOPK_FALLBACK, gpu_memory_utilization=0.80, max_num_batched_tokens=4096):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.variants, self.strings = find_word_variants(self.tokenizer)
        self.read, self.topk, self.max_model_len = read, topk, max_model_len
        kw = dict(model=model_path, tokenizer=model_path, tensor_parallel_size=tensor_parallel_size, dtype="float16",
                  max_model_len=max_model_len, seed=seed, gpu_memory_utilization=gpu_memory_utilization,
                  max_num_batched_tokens=max_num_batched_tokens, disable_custom_all_reduce=True,
                  max_logprobs=max(20, topk), generation_config="vllm")
        try:
            self.llm = LLM(**kw)
        except TypeError as e:  # older vLLM without `generation_config`
            if "generation_config" not in str(e):
                raise
            kw.pop("generation_config")
            self.llm = LLM(**kw)

    def _ids(self, prompts):
        return [{"prompt_token_ids": self.tokenizer.encode(p, add_special_tokens=False)} for p in prompts]

    def _raw(self, prompts, read, wanted=None):
        from vllm import SamplingParams
        wanted = sorted({t for v in self.variants.values() for t in v}) if wanted is None else list(wanted)
        if read == "token_ids":
            sp = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=wanted)
        else:
            sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20 if read == "top20" else self.topk)
        outs = self.llm.generate(self._ids(prompts), sp, use_tqdm=False)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            if not hasattr(lp, "items"):
                raise RuntimeError(f"unexpected vLLM logprobs container {type(lp)}; unset VLLM_FLAT_LOGPROBS")
            res.append({int(t): float(getattr(v, "logprob", v)) for t, v in lp.items()})
        return res, wanted

    def word_logprobs(self, prompts, words=("Yes", "No"), read=None):
        read = read or self.read
        raw, wanted = self._raw(prompts, read, self.wanted_ids(words))
        out = []
        for vals in raw:
            if read == "token_ids":
                missing = [t for t in wanted if t not in vals]
                if missing:
                    raise RuntimeError(f"vLLM did not return the requested token ids {missing}")
                out.append(({t: vals[t] for t in wanted}, 0))
            else:   # top-k: impute the last returned log-probability for absent variants (an upper bound)
                finite = [v for v in vals.values() if math.isfinite(v)]
                if len(finite) < len(vals) or not finite:
                    out.append(({t: float("nan") for t in wanted}, 0))
                    continue
                floor = min(finite)
                d = {t: vals.get(t, floor) for t in wanted}
                out.append((d, sum(t not in vals for t in wanted)))
        return out

    def check_exact(self, prompts, words=None):
        """Deviation D2: compare logprob_token_ids with the ordinary top-20 log-probabilities on the same prompts
        (all variant ids read so far, or those of `words`)."""
        ids = None if words is None else self.wanted_ids(words)
        try:
            exact, wanted = self._raw(prompts, "token_ids", ids)
        except Exception as e:
            return {"ok": False, "error": repr(e)[:500]}
        top, _ = self._raw(prompts, "top20", ids)
        diffs = []
        for e, t in zip(exact, top):
            for tid in wanted:
                if tid not in e:
                    return {"ok": False, "error": f"token id {tid} missing from the logprob_token_ids output"}
                if tid in t:
                    diffs.append(abs(e[tid] - t[tid]))
        mx = max(diffs) if diffs else float("nan")
        ok = bool(diffs) and all(math.isfinite(x) for x in diffs) and mx <= LOGPROB_CHECK_TOL
        return {"ok": ok, "n_prompts": len(prompts), "n_compared": len(diffs), "max_abs_diff": mx,
                "tolerance": LOGPROB_CHECK_TOL}

    def generate_text(self, token_id_lists, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=int(max_tokens), temperature=0.0)
        outs = self.llm.generate([{"prompt_token_ids": list(ids)} for ids in token_id_lists], sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def info(self):
        import vllm
        d = super().info()
        d["vllm_version"] = vllm.__version__
        if self.read == "topk":
            d["topk"] = self.topk
        return d


class HFBackend(Backend):
    """transformers fallback: one forward pass per prompt, full next-token distribution (exact)."""
    name = "hf"

    def __init__(self, model_path, batch_size=8, device_map="auto"):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.variants, self.strings = find_word_variants(self.tokenizer)
        self.eos = eos_token_ids(model_path, self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos[0]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map if torch.cuda.is_available() else None, **{dkey: dtype})
        self.model.eval()
        self.batch_size = batch_size

    def word_logprobs(self, prompts, words=("Yes", "No")):
        t = self.torch
        wanted = self.wanted_ids(words)
        dev = self.model.get_input_embeddings().weight.device
        res = []
        for b in range(0, len(prompts), self.batch_size):
            enc = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts[b:b + self.batch_size]]
            L = max(len(e) for e in enc)
            ids = t.tensor([[self.pad_id] * (L - len(e)) + e for e in enc], device=dev)
            att = t.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=dev)
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            with t.no_grad():
                out = self.model.base_model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False)
                logits = self.model.get_output_embeddings()(out.last_hidden_state[:, -1, :]).float()
            lp = t.log_softmax(logits, dim=-1)
            finite = t.isfinite(logits).all(dim=-1).cpu().tolist()
            rows = lp[:, wanted].cpu().tolist()
            for ok, row in zip(finite, rows):
                res.append(({k: (v if ok else float("nan")) for k, v in zip(wanted, row)}, 0))
        return res

    def generate_text(self, token_id_lists, max_tokens):
        t = self.torch
        dev = self.model.get_input_embeddings().weight.device
        out = []
        for ids in token_id_lists:
            x = t.tensor([list(ids)], device=dev)
            from transformers import GenerationConfig
            gc = GenerationConfig(max_new_tokens=int(max_tokens), do_sample=False, repetition_penalty=1.0,
                                  pad_token_id=self.pad_id, eos_token_id=self.eos)     # plain greedy decoding
            with t.no_grad():
                g = self.model.generate(input_ids=x, attention_mask=t.ones_like(x), generation_config=gc)
            out.append(self.tokenizer.decode(g[0, x.shape[1]:], skip_special_tokens=True))
        return out

    def info(self):
        import transformers
        d = super().info()
        d["transformers_version"] = transformers.__version__
        return d


class _MockTok:
    bos_token = None
    bos_token_id = None

    def __init__(self):
        self.vocab = {"Yes": 1, "ĠYes": 2, "No": 3, "ĠNo": 4, "A": 5, "ĠA": 6, "B": 7, "ĠB": 8}

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids, **kw):
        inv = {v: k.replace("Ġ", " ") for k, v in self.vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4 + 1))

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class MockBackend(Backend):
    """Deterministic fake model for CPU tests. It reads the evidence impacts in the prompt. `alarm_gain` and
    `safe_gain` set how strongly alarm-pointing and safe-pointing evidence move it, `lean` shifts its baseline,
    and `scale` stretches its log-odds (large values give very confident answers)."""
    name = "mock"
    _IMPACT = re.compile(r"\(impact ([+-]\d+\.\d+)\)")

    def __init__(self, reversed_questions, alarm_gain=1.6, safe_gain=0.6, lean=0.0, scale=1.0, seed="m",
                 truth=None, signal=1.5, choice_alarm_gain=None, choice_safe_gain=None, letter_bias=0.0):
        self.tokenizer = _MockTok()
        self.variants, self.strings = find_word_variants(self.tokenizer)
        self.reversed_q = set(reversed_questions)
        self.alarm_gain, self.safe_gain, self.lean, self.scale, self.seed = alarm_gain, safe_gain, lean, scale, seed
        self.truth, self.signal = truth or {}, signal       # patient block -> outcome, so the mock beats chance
        # answer-option framing: its own gains (default: the Yes/No gains) and a pull toward the letter 'A'
        self.c_alarm_gain = alarm_gain if choice_alarm_gain is None else choice_alarm_gain
        self.c_safe_gain = safe_gain if choice_safe_gain is None else choice_safe_gain
        self.letter_bias = letter_bias
        self.alarm_options = {o["alarm"] for o in CHOICE_OPTIONS.values()}

    def word_logprobs(self, prompts, words=("Yes", "No")):
        wanted = self.wanted_ids(words)
        res = []
        for p in prompts:
            key = p.split(PATIENT_HEADER)[-1].split("\n\n")[0]
            base = 3.0 * (_u(key, self.seed) - 0.5) + self.lean
            if key in self.truth:
                base += self.signal * (2 * self.truth[key] - 1)
            s = sum(float(x) for x in self._IMPACT.findall(p.split(PATIENT_HEADER)[-1]))
            choice = CHOICE_INSTRUCTION in p
            ga, gs = (self.c_alarm_gain, self.c_safe_gain) if choice else (self.alarm_gain, self.safe_gain)
            z = self.scale * (base + (ga * s if s > 0 else gs * s))      # log-odds of the alarming outcome
            if choice:
                opt_a = re.findall(r"\nA: (.*)\nB: ", p)[-1]
                zp = (z if opt_a in self.alarm_options else -z) + self.letter_bias     # log-odds of "A"
                ids = (6, 5, 8, 7)
            else:
                q = re.findall(r"Question: (.*)\n" + re.escape(ANSWER_INSTRUCTION), p)
                reversed_ = bool(q) and q[-1] in self.reversed_q
                zp = -z if reversed_ else z                       # log-odds of "Yes"
                ids = (2, 1, 4, 3)
            mass = 0.80 + 0.19 * _u(key, "m")
            lmy = math.log(mass) - math.log1p(math.exp(-zp)) if zp > -30 else math.log(mass) + zp
            lmn = math.log(mass) - math.log1p(math.exp(zp)) if zp < 30 else math.log(mass) - zp
            full = {ids[0]: lmy + math.log(0.9), ids[1]: lmy + math.log(0.1), ids[2]: lmn + math.log(0.9),
                    ids[3]: lmn + math.log(0.1)}
            other = math.log(1 - mass) - math.log(4) if mass < 1 else -50.0      # the words not offered here
            res.append(({t: full.get(t, other) for t in wanted}, 0))
        return res

    def generate_text(self, token_id_lists, max_tokens):
        return ["1,2,3" for _ in token_id_lists]


def smoke_test(model_path, kind="instruct", prompts_path=None):
    """Run in a subprocess before vLLM is loaded in the notebook kernel (a failed engine start then cannot leave
    the kernel holding GPU memory). Also runs the Deviation D2 check of logprob_token_ids against the top-20
    log-probabilities on the given prompts."""
    import torch
    tp = 2 if torch.cuda.device_count() >= 2 else 1
    t0 = time.time()
    be = VLLMBackend(model_path, tensor_parallel_size=tp, read="token_ids")
    text = "Is the sky usually blue on a clear day?\n" + ANSWER_INSTRUCTION
    prompts = [render(text, kind, be.tokenizer)]
    choice = []
    if prompts_path:
        data = json.load(open(prompts_path))
        if isinstance(data, dict):        # second study: {"yesno": [...], "choice": [...]} (rendered dev prompts)
            prompts += data.get("yesno", [])
            choice = data.get("choice", [])
        else:                             # first study: a list of rendered Yes/No dev prompts
            prompts += data
    check = be.check_exact(prompts)
    check_choice = be.check_exact(choice, ("A", "B")) if choice else None
    read = "token_ids" if check.get("ok") and (check_choice is None or check_choice.get("ok")) else "topk"
    be.read = read
    r = be.yes_no(prompts[:1])
    res = {"ok": True, "load_seconds": time.time() - t0, "tensor_parallel_size": tp, "read": read,
           "exact_check": check, "sky_prompt": r[0], "info": be.info()}
    if check_choice is not None:
        res["exact_check_choice"] = check_choice
    print("SMOKE_RESULT " + json.dumps(res, default=str))
    return res


SANITY_PROMPTS = {   # engineering sanity checks: (text, answer words, expected first word?)
    "sky_yes": ("Is the sky usually blue on a clear day?\n" + ANSWER_INSTRUCTION, ("Yes", "No"), True),
    "fire_no": ("Is fire usually cold to the touch?\n" + ANSWER_INSTRUCTION, ("Yes", "No"), False),
    "color_A": ("Which of these two is a color?\nA: Blue\nB: Seven\n" + CHOICE_INSTRUCTION, ("A", "B"), True),
    "color_B": ("Which of these two is a color?\nA: Seven\nB: Blue\n" + CHOICE_INSTRUCTION, ("A", "B"), False),
}


def _mass(rows):
    m = [math.exp(r["lp_pos"]) + math.exp(r["lp_neg"]) if math.isfinite(r["lp_pos"]) and math.isfinite(r["lp_neg"])
         else float("nan") for r in rows]
    f = [x for x in m if math.isfinite(x)]
    return {"n": len(m), "finite_share": len(f) / len(m) if m else float("nan"),
            "mass_median": float(np.median(f)) if f else float("nan"), "mass_min": float(min(f)) if f else float("nan"),
            "imputed_share": float(np.mean([r["n_imputed"] > 0 for r in rows])) if rows else float("nan")}


def engineering_check(model_id, kind, prompts_path, cache_dir, revision=None, formats=None):
    """Second study, BEFORE registration (PREREGISTRATION_V2.md): can this model be run as registered? Engineering
    only: download, the vLLM float16 load on the Kaggle GPUs, the chat template, the answer-word tokens, the exact
    reading check (D2), finiteness, the answer-word mass, the reading under top-1,000 (D5), prompt lengths and
    the time per prompt. It computes no accuracy, no answer direction and no asymmetry. The prompts are dev-sample
    prompts (never test patients). `formats` = the answer formats the model will be run with ('yesno', 'choice');
    only these decide eligibility ('ok'), the others are reported. Prints one ENGCHECK_RESULT line."""
    formats = tuple(formats or (("yesno", "choice") if kind == "instruct" else ("yesno",)))
    res = {"model_id": model_id, "kind": kind, "formats_required": list(formats), "ok": False, "problems": []}
    t0 = time.time()
    try:
        path, sha = download_model(model_id, cache_dir, revision, os.environ.get("HF_TOKEN"))
        res.update(revision=sha, download_seconds=round(time.time() - t0, 1))
        cfg = json.load(open(os.path.join(path, "config.json")))
        tcfg = cfg.get("text_config") or {}
        res["config"] = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                         "dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tcfg.get("torch_dtype"),
                         "quantization": (cfg.get("quantization_config") or {}).get("quant_method"),
                         "max_position_embeddings": cfg.get("max_position_embeddings") or tcfg.get("max_position_embeddings")}
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        t1 = time.time()
        be = VLLMBackend(path, tensor_parallel_size=tp, read="token_ids")
        res.update(load_seconds=round(time.time() - t1, 1), tensor_parallel_size=tp)
        items = json.load(open(prompts_path))
        tok = be.tokenizer
        res["template_system_text"] = template_system_text(tok) if kind == "instruct" else None
        pairs = [("Yes", "No")] + ([("A", "B")] if kind == "instruct" else [])
        for pair in pairs:
            try:
                be.ensure_words(pair)
            except Exception as e:
                res["problems"].append(f"no single-token variant for {pair}: {e!r}"[:300])
        res["variants"] = {w: [[i, be.strings[i]] for i in ids] for w, ids in be.variants.items()}
        groups = {}
        for it in items:
            if is_choice(it["framing"]) and kind != "instruct":
                continue
            groups.setdefault((it["set"], "choice" if is_choice(it["framing"]) else "yesno"), []).append(it)
        rendered = {k: [render(it["text"], kind, tok) for it in v] for k, v in groups.items()}
        ntok = [len(tok.encode(r, add_special_tokens=False)) for v in rendered.values() for r in v]
        res["prompt_tokens"] = {"median": int(np.median(ntok)), "max": int(max(ntok))}
        if max(ntok) >= MAX_MODEL_LEN - 1:
            res["problems"].append("a prompt is longer than the model window")
        for fmt in ("yesno", "choice"):
            if ("check", fmt) in rendered:
                res[f"rendered_tail_{fmt}"] = rendered[("check", fmt)][0][-500:]
        # D2 exact-reading check per answer format, on the check prompts
        for fmt, pair in (("yesno", ("Yes", "No")), ("choice", ("A", "B"))):
            if ("check", fmt) in rendered and pair[0] in be.variants:
                res[f"exact_check_{fmt}"] = be.check_exact(rendered[("check", fmt)], pair)
        # the registered reading (top 1,000, D5) on every prompt: finiteness, mass, imputation, time
        be.read = "topk"
        for (sset, fmt), rs in sorted(rendered.items()):
            pair = ("A", "B") if fmt == "choice" else ("Yes", "No")
            if pair[0] not in be.variants:
                continue
            t1 = time.time()
            rows = be.answer_logprobs(rs, pair)
            el = time.time() - t1
            res[f"reading_{sset}_{fmt}"] = dict(_mass(rows), seconds=round(el, 1),
                                                 seconds_per_prompt=round(el / max(1, len(rs)), 4))
        # sanity prompts (the answer is obvious): only whether the right word is more likely
        san = {}
        for name, (text, pair, first) in SANITY_PROMPTS.items():
            if pair[0] not in be.variants:
                continue
            r = be.answer_logprobs([render(text, kind, tok)], pair)[0]
            san[name] = bool((r["lp_pos"] > r["lp_neg"]) == first)
        res["sanity_right_word_more_likely"] = san
        fmt_of = lambda name: "choice" if name.endswith("_choice") or name.startswith("color_") else "yesno"
        for k, v in res.items():
            if k.startswith("reading_") and fmt_of(k) in formats:
                if v["finite_share"] < 1:
                    res["problems"].append(f"{k}: non-finite readings ({v['finite_share']:.3f} finite)")
                if not v["mass_median"] >= VALIDITY_MIN_MASS:
                    res["problems"].append(f"{k}: median answer-word mass {v['mass_median']:.3f} < {VALIDITY_MIN_MASS}")
        failed = [k for k, v in san.items() if not v and fmt_of(k) in formats]
        if failed:
            res["problems"].append(f"sanity prompts failed: {failed}")
        if res["config"].get("quantization"):
            res["problems"].append(f"quantized weights ({res['config']['quantization']}); the study runs float16 weights")
        res["ok"] = not res["problems"]
    except Exception as e:
        import traceback
        res["problems"].append(f"error: {e!r}"[:500])
        res["traceback"] = traceback.format_exc()[-3000:]
    res["total_seconds"] = round(time.time() - t0, 1)
    print("ENGCHECK_RESULT " + json.dumps(res, default=str))
    return res


# =============================================================================================================
# 7. Scoring records and the analysis
# =============================================================================================================
def finish_record(rec: dict, r: dict, framing: str) -> dict:
    """Adds the reading of one prompt (Section 5 with Deviation D2). lp_yes / lp_no are the log of the summed
    probabilities of the Yes / No variant tokens. The log-odds of the alarming outcome comes straight from the
    log-probabilities, with no clipping. A record with zero Yes/No mass or a non-finite log-odds is marked
    valid = 0 and left out of every analysis; it is never turned into a 'No' (Deviation D3, item 5)."""
    lpy, lpn = float(r["lp_yes"]), float(r["lp_no"])
    rec["answer_words"] = "/".join(answer_pair(framing))
    py = math.exp(lpy) if math.isfinite(lpy) else (0.0 if lpy == -math.inf else float("nan"))
    pn = math.exp(lpn) if math.isfinite(lpn) else (0.0 if lpn == -math.inf else float("nan"))
    mass = py + pn
    lo_yes = lpy - lpn if (math.isfinite(lpy) and math.isfinite(lpn)) else float("nan")
    rec.update(lp_yes=lpy, lp_no=lpn, p_yes_raw=py, p_no_raw=pn, yn_mass=mass, n_imputed=int(r.get("n_imputed", 0)))
    if not (mass > 0 and math.isfinite(lo_yes)):
        rec.update(valid=0, p_yes=float("nan"), pred_yes=-1, p_alarm=float("nan"), lo_alarm=float("nan"),
                   pred_alarm=-1, correct=-1)
        return rec
    p_yes = 1.0 / (1.0 + math.exp(-lo_yes)) if lo_yes > -700 else 0.0
    yes = lo_yes > 0
    # first study: 'Yes' means the alarming outcome in the alarm framing; second study: in the answer-option
    # framings the 'yes' fields hold option A, which is the alarming outcome in choice_alarm_A
    alarm = positive_means_alarm(framing)
    rec.update(valid=1, p_yes=p_yes, pred_yes=int(yes), p_alarm=p_yes if alarm else 1 - p_yes,
               lo_alarm=lo_yes if alarm else -lo_yes, pred_alarm=int(yes) if alarm else int(not yes))
    rec["correct"] = int(rec["pred_alarm"] == rec["y"])
    return rec


# ----------------------------------------------------------------------------------- statistics helpers
def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact two-sided McNemar test on paired 0/1 correctness (a vs b)."""
    from scipy.stats import binom
    a, b = np.asarray(correct_a, int), np.asarray(correct_b, int)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(only_a, only_b), n, 0.5)))
    return {"acc_a": float(a.mean()), "acc_b": float(b.mean()), "diff": float(a.mean() - b.mean()),
            "a_only_correct": only_a, "b_only_correct": only_b, "p": p}


def holm(pvals) -> list:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def _fast_delong(preds_sorted, m):
    from scipy.stats import rankdata
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx = np.vstack([rankdata(preds_sorted[r, :m]) for r in range(k)])
    ty = np.vstack([rankdata(preds_sorted[r, m:]) for r in range(k)])
    tz = np.vstack([rankdata(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n


def delong_paired(y, s1, s2) -> dict:
    """Two-sided paired DeLong test of AUROC(s1) - AUROC(s2) (fast DeLong, Sun & Xu 2014)."""
    from scipy.stats import norm
    y = np.asarray(y)
    m = int((y == 1).sum())
    if m == 0 or m == len(y):
        return {"auc1": float("nan"), "auc2": float("nan"), "diff": float("nan"), "p": float("nan")}
    order = np.concatenate([np.flatnonzero(y == 1), np.flatnonzero(y == 0)])
    aucs, cov = _fast_delong(np.vstack([np.asarray(s1, float)[order], np.asarray(s2, float)[order]]), m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    p = (1.0 if abs(diff) < 1e-12 else 0.0) if var <= 1e-15 else float(2 * norm.sf(abs(diff) / math.sqrt(var)))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff, "p": p}


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X, y, iters: int = 50, ridge: float = 1e-6):
    """Plain (unpenalised, apart from a 1e-6 ridge for numerical stability) logistic regression by Newton steps.
    Returns weights with the intercept last."""
    X = np.column_stack([np.asarray(X, float), np.ones(len(X))])
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(w, X):
    X = np.column_stack([np.asarray(X, float), np.ones(len(X))])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


# ----------------------------------------------------------------------------------- protocol measures
VALIDITY_MIN_MASS = 0.5
VALIDITY_MIN_C0_ACC = 0.55


def slice_records(df: pd.DataFrame, framing: str = "alarm", shot_set: str = "A") -> pd.DataFrame:
    return df[(df["framing"] == framing) & (df["shot_set"] == shot_set)]


def validity(df: pd.DataFrame) -> dict:
    """Section 5 validity checks with Deviation D3, item 3: median Yes+No mass over every alarm-framing, set-A
    record of the cell (C0, C2, C4, C6; a record without a usable reading counts as mass 0) and C0 accuracy on
    the alarm-framing, set-A records."""
    a = slice_records(df)
    mass = a["yn_mass"].astype(float).fillna(0.0)
    c0 = a[(a["cond"] == "C0") & (a["valid"] == 1)]
    med = float(mass.median()) if len(mass) else float("nan")
    acc = float(c0["correct"].mean()) if len(c0) else float("nan")
    return {"median_yn_mass": med, "c0_accuracy": acc, "n_alarm_A_records": int(len(a)), "n_c0_valid": int(len(c0)),
            "passes": bool(med >= VALIDITY_MIN_MASS and acc >= VALIDITY_MIN_C0_ACC)}


def _wide(df: pd.DataFrame, need_conds=("C0", "C2", "C4")):
    """Per patient (valid readings only): predictions, P(alarm), log-odds and card directions per condition, plus
    the XGBoost probability. Patients without valid readings in `need_conds` are dropped (Deviation D3, item 5)."""
    d = df[df["valid"] == 1]
    if len(d) == 0:
        return None
    w = d.pivot_table(index="id", columns="cond", values=["pred_alarm", "p_alarm", "lo_alarm", "card_dir", "y",
                                                           "correct", "p_yes", "xgb_p"], aggfunc="first")
    need = [(v, c) for v in ("pred_alarm", "lo_alarm") for c in need_conds] + [("card_dir", "C2")]
    if not all(k in w.columns for k in need):
        return None
    return w.dropna(subset=need).sort_index()


def xgb_logodds(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def asym_arrays(w) -> dict:
    cd = w[("card_dir", "C2")].to_numpy().astype(int)            # direction of the patient's TRUE card
    c0 = w[("pred_alarm", "C0")].to_numpy().astype(int)
    c2, c4 = w[("pred_alarm", "C2")].to_numpy().astype(int), w[("pred_alarm", "C4")].to_numpy().astype(int)
    l0 = w[("lo_alarm", "C0")].to_numpy(float)
    l2, l4 = w[("lo_alarm", "C2")].to_numpy(float), w[("lo_alarm", "C4")].to_numpy(float)
    return {"c0": c0, "c2": c2, "c4": c4,
            "pa": np.where(cd == 1, c2, c4), "ps": np.where(cd == 1, c4, c2),          # alarm / safe-pointing card
            "l0": l0, "la": np.where(cd == 1, l2, l4), "ls": np.where(cd == 1, l4, l2),
            "t": np.where(cd == 1, 1.0, -1.0),
            "z": xgb_logodds(w[("xgb_p", "C0")].to_numpy(float)), "y": w[("y", "C0")].to_numpy(float)}


def predicted_baseline(l0, z, y):
    """Deviation D1, first stage: OLS of l0 on [1, z, z^2, z^3, y] (z standardized); returns the fitted values e."""
    zs = (z - z.mean()) / z.std()
    Z = np.column_stack([np.ones_like(zs), zs, zs ** 2, zs ** 3, y])
    return Z @ np.linalg.lstsq(Z, l0, rcond=None)[0]


def adjusted_fit(l0, la, ls, t, z, y):
    """Deviation D1: e = expected baseline from the first stage; OLS of d = (la - l0) + (ls - l0) on
    [1, e, (e/s)^3, t] with s = SD of e. Returns (a, details) with a = the intercept, or (nan, None) when the
    regression cannot be fitted (too few patients, constant z, or a rank-deficient design)."""
    l0, la, ls, t, z, y = (np.asarray(v, float) for v in (l0, la, ls, t, z, y))
    if len(l0) < 10 or not np.isfinite(z.std()) or z.std() == 0:
        return float("nan"), None
    e = predicted_baseline(l0, z, y)
    d = (la - l0) + (ls - l0)
    s = e.std()
    if not np.isfinite(s) or s == 0:
        return float("nan"), None
    X = np.column_stack([np.ones_like(e), e, (e / s) ** 3, t])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan"), None
    coef = np.linalg.lstsq(X, d, rcond=None)[0]
    r2 = 1 - np.var(l0 - e) / np.var(l0) if np.var(l0) > 0 else float("nan")
    return float(coef[0]), {"a": float(coef[0]), "g1": float(coef[1]), "g3_per_s3": float(coef[2]),
                            "c_card_truth": float(coef[3]), "s": float(s), "first_stage_r2": float(r2),
                            "e_range": [float(e.min()), float(e.max())]}


def adjusted_asymmetry(l0, la, ls, t, z, y):
    """The primary asymmetry statistic a (Deviation D1). Same estimate as analysis/simulate_asymmetry_measures.py."""
    return adjusted_fit(l0, la, ls, t, z, y)[0]


def boot_p(boots) -> float:
    """Two-sided bootstrap p-value of Section 7: 2 x min(share <= 0, share >= 0), capped at 1."""
    b = np.asarray(boots, float)
    b = b[~np.isnan(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))) if len(b) else float("nan")


def _pct(b):
    b = np.asarray(b, float)
    b = b[~np.isnan(b)]
    return [float(x) for x in np.percentile(b, [2.5, 97.5])] if len(b) else [float("nan"), float("nan")]


def _asym_stats(c0, pa, ps):
    n_safe, n_alarm = int((c0 == 0).sum()), int((c0 == 1).sum())
    A = ((pa == 1) & (c0 == 0)).sum() / n_safe if n_safe else np.nan
    S = ((ps == 0) & (c0 == 1)).sum() / n_alarm if n_alarm else np.nan
    return A, S, A - S


def reading_adjusted_asymmetry(l0, la, ls, t):
    """a' (Deviation D1): the same second stage with the patient's own no-evidence reading l0 as the baseline,
    OLS of d on [1, l0, (l0/s)^3, t]; the intercept. Same estimate as the 'l0-adjusted' column of the simulation."""
    l0, la, ls, t = (np.asarray(v, float) for v in (l0, la, ls, t))
    s = l0.std()
    if len(l0) < 10 or not np.isfinite(s) or s == 0:
        return float("nan")
    X = np.column_stack([np.ones_like(l0), l0, (l0 / s) ** 3, t])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(X, (la - l0) + (ls - l0), rcond=None)[0][0])


def _winsor(x, lo=1, hi=99):
    q1, q2 = np.percentile(x, [lo, hi])
    return np.clip(x, q1, q2)


def winsorized_asymmetry(l0, la, ls, t, z, y):
    """Labelled sensitivity analysis (D1): a with d winsorized at its 1st and 99th percentiles."""
    l0, la, ls, t, z, y = (np.asarray(v, float) for v in (l0, la, ls, t, z, y))
    if len(l0) < 10 or not np.isfinite(z.std()) or z.std() == 0:
        return float("nan")
    e = predicted_baseline(l0, z, y)
    d = _winsor((la - l0) + (ls - l0))
    s = e.std()
    if not np.isfinite(s) or s == 0:
        return float("nan")
    X = np.column_stack([np.ones_like(e), e, (e / s) ** 3, t])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(X, d, rcond=None)[0][0])


def asymmetry(df: pd.DataFrame) -> dict | None:
    """One framing x example-set slice (Deviation D1). Primary: the adjusted asymmetry a and the reading-adjusted
    a', each with a paired bootstrap over patients (10,000 resamples, seed 2026, every regression refitted in every
    resample) and the two-sided p of Section 7; the H3 test is the intersection-union test p = max(p_a, p_a').
    Sensitivity: a with d winsorized. Secondary (Section 6): A, S, Delta = A - S; the unadjusted log-odds
    asymmetry; follow rates."""
    w = _wide(df)
    if w is None or len(w) < 10:
        return None
    z = asym_arrays(w)
    n = len(w)
    a_hat, coef = adjusted_fit(z["l0"], z["la"], z["ls"], z["t"], z["z"], z["y"])
    a0_hat = reading_adjusted_asymmetry(z["l0"], z["la"], z["ls"], z["t"])
    aw_hat = winsorized_asymmetry(z["l0"], z["la"], z["ls"], z["t"], z["z"], z["y"])
    A, S, D = _asym_stats(z["c0"], z["pa"], z["ps"])
    lo_un = float(np.mean(z["la"] - z["l0"]) - np.mean(z["l0"] - z["ls"]))
    rng = np.random.default_rng(BOOT_SEED)
    adj_b, a0_b, aw_b, d_b, lo_b = (np.empty(BOOT_RESAMPLES) for _ in range(5))
    for i in range(BOOT_RESAMPLES):
        b = rng.integers(0, n, n)
        args = (z["l0"][b], z["la"][b], z["ls"][b], z["t"][b])
        adj_b[i] = adjusted_asymmetry(*args, z["z"][b], z["y"][b])
        a0_b[i] = reading_adjusted_asymmetry(*args)
        aw_b[i] = winsorized_asymmetry(*args, z["z"][b], z["y"][b])
        d_b[i] = _asym_stats(z["c0"][b], z["pa"][b], z["ps"][b])[2]
        lo_b[i] = np.mean(z["la"][b] - z["l0"][b]) - np.mean(z["l0"][b] - z["ls"][b])
    p_a, p_a0 = boot_p(adj_b), boot_p(a0_b)
    est_ok = bool(np.isfinite(a_hat) and np.isfinite(a0_hat))
    e = predicted_baseline(z["l0"], z["z"], z["y"]) if coef else None
    return {"n": int(n),
            "adjusted": {"a": a_hat, "ci": _pct(adj_b), "p_boot": p_a,
                         "n_boot_failed": int(np.isnan(adj_b).sum()), "coef": coef,
                         "share_e_within_1sd_of_0": float(np.mean(np.abs(e) < e.std())) if e is not None else None},
            "reading_adjusted": {"a": a0_hat, "ci": _pct(a0_b), "p_boot": p_a0,
                                 "n_boot_failed": int(np.isnan(a0_b).sum())},
            "h3_test": {"p": max(p_a, p_a0) if est_ok else float("nan"),
                        "both_positive": bool(est_ok and a_hat > 0 and a0_hat > 0),
                        "agree_in_sign": bool(est_ok and np.sign(a_hat) == np.sign(a0_hat))},
            "sensitivity_winsorized": {"a": aw_hat, "ci": _pct(aw_b), "p_boot": boot_p(aw_b)},
            "delta": {"A": float(A), "S": float(S), "delta": float(D), "ci": _pct(d_b), "p_boot": boot_p(d_b)},
            "logodds_unadjusted": {"value": lo_un, "ci": _pct(lo_b), "p_boot": boot_p(lo_b)},
            "follow_alarm_card": float(np.mean(z["pa"] == 1)), "follow_safe_card": float(np.mean(z["ps"] == 0)),
            "flip_rate_C2_C4": float(np.mean(z["c2"] != z["c4"])),
            "share_true_card_alarm": float(np.mean(z["t"] > 0)), "c0_share_alarm": float(np.mean(z["c0"] == 1)),
            "c0_logodds_median": float(np.median(z["l0"]))}


def q3_interpretation(ci_alarm, ci_reversed) -> str:
    """Deviation D1: the four-way framing table on the adjusted asymmetry (95% CIs)."""
    vals = list(ci_alarm or [np.nan, np.nan]) + list(ci_reversed or [np.nan, np.nan])
    if any(v is None or np.isnan(v) for v in vals):
        return "inconclusive"
    up_a, dn_a = ci_alarm[0] > 0, ci_alarm[1] < 0
    up_r, dn_r = ci_reversed[0] > 0, ci_reversed[1] < 0
    if up_a and up_r:
        return "follows the outcome, toward alarm (alarm aversion)"
    if dn_a and dn_r:
        return "follows the outcome, toward safe"
    if up_a and dn_r:
        return 'follows the answer word "Yes"'
    if dn_a and up_r:
        return 'follows the answer word "No"'
    return "inconclusive"


def adjusted_difference(df_instruct: pd.DataFrame, df_base: pd.DataFrame) -> dict | None:
    """H4 (Deviation D1): a(instruct) - a(base) and a'(instruct) - a'(base) for the same family and dataset, on the
    alarm framing with example set A; paired bootstrap over the patients valid in both models (every regression
    refitted in each resample); intersection-union test p = max of the two p-values. Delta(instruct) - Delta(base)
    is reported as a secondary value."""
    wi, wb = _wide(slice_records(df_instruct)), _wide(slice_records(df_base))
    if wi is None or wb is None:
        return None
    ids = sorted(set(wi.index) & set(wb.index))
    if len(ids) < 10:
        return None
    zi, zb = asym_arrays(wi.loc[ids]), asym_arrays(wb.loc[ids])
    adj = lambda z, b: adjusted_asymmetry(z["l0"][b], z["la"][b], z["ls"][b], z["t"][b], z["z"][b], z["y"][b])
    rdj = lambda z, b: reading_adjusted_asymmetry(z["l0"][b], z["la"][b], z["ls"][b], z["t"][b])
    dlt = lambda z, b: _asym_stats(z["c0"][b], z["pa"][b], z["ps"][b])[2]
    n = len(ids)
    full = np.arange(n)
    est, est0, est_d = (f(zi, full) - f(zb, full) for f in (adj, rdj, dlt))
    rng = np.random.default_rng(BOOT_SEED)
    boots, boots0, boots_d = np.empty(BOOT_RESAMPLES), np.empty(BOOT_RESAMPLES), np.empty(BOOT_RESAMPLES)
    for i in range(BOOT_RESAMPLES):
        b = rng.integers(0, n, n)
        boots[i], boots0[i], boots_d[i] = (f(zi, b) - f(zb, b) for f in (adj, rdj, dlt))
    p, p0 = boot_p(boots), boot_p(boots0)
    ok = bool(np.isfinite(est) and np.isfinite(est0))
    return {"n": n, "a_instruct": adj(zi, full), "a_base": adj(zb, full), "diff": float(est), "ci": _pct(boots),
            "p_boot": p, "n_boot_failed": int(np.isnan(boots).sum()),
            "reading_adjusted_diff": {"diff": float(est0), "ci": _pct(boots0), "p_boot": p0},
            "h4_test": {"p": max(p, p0) if ok else float("nan"), "both_positive": bool(ok and est > 0 and est0 > 0)},
            "secondary_delta_diff": {"diff": float(est_d), "ci": _pct(boots_d), "p_boot": boot_p(boots_d)}}


def _trust_features(conf, agree, flag=None):
    cols = [np.asarray(conf, float), np.asarray(agree, float)]
    if flag is not None:
        cols.append(np.asarray(flag, float))
    return np.column_stack(cols)


def overtrust(df: pd.DataFrame) -> dict | None:
    """H5: logistic regression of correctness on [confidence = max(P(Yes), P(No)), agrees-with-card] fitted on
    C2 records, applied to C4 records. Over-trust = mean predicted P(correct) - observed accuracy on C4. Paired
    bootstrap over patients (the regression is refitted in every resample). Needs valid C2 and C4 readings."""
    w = _wide(df, need_conds=("C2", "C4"))
    if w is None or len(w) < 10:
        return None
    feats = {}
    for c in ("C2", "C4"):
        py = w[("p_yes", c)].to_numpy(float)
        conf = np.maximum(py, 1 - py)
        agree = (w[("pred_alarm", c)].to_numpy() == w[("card_dir", c)].to_numpy()).astype(float)
        feats[c] = (_trust_features(conf, agree), w[("correct", c)].to_numpy(float))
    (X2, y2), (X4, y4) = feats["C2"], feats["C4"]

    def stat(idx):
        wts = fit_logistic(X2[idx], y2[idx])
        return float(predict_logistic(wts, X4[idx]).mean() - y4[idx].mean())

    n = len(y2)
    est = stat(np.arange(n))
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(BOOT_RESAMPLES)])
    return {"n": n, "overtrust": est, "ci": _pct(boots), "p_boot": boot_p(boots),
            "acc_C4": float(y4.mean()), "mean_predicted_C4": float(est + y4.mean())}


def flip_flag_mixed(df: pd.DataFrame, folds: int = 5) -> dict | None:
    """H6: mixed setting. Each patient gets the true (C2) or flipped (C4) card at random (p = 0.5, seed 2027).
    Trust regression with vs without the flip-consistency flag (prediction changes between C2 and C4);
    5-fold cross-fitted predictions; AUROC for correctness; paired DeLong test. Needs valid C2 and C4 readings."""
    w = _wide(df, need_conds=("C2", "C4"))
    if w is None or len(w) < 20:
        return None
    n = len(w)
    pick = np.where(np.random.default_rng(SPLIT_SEED).random(n) < 0.5, "C2", "C4")
    flag = (w[("pred_alarm", "C2")].to_numpy() != w[("pred_alarm", "C4")].to_numpy()).astype(float)
    py = np.array([w[("p_yes", c)].iloc[i] for i, c in enumerate(pick)], float)
    pred = np.array([w[("pred_alarm", c)].iloc[i] for i, c in enumerate(pick)])
    card = np.array([w[("card_dir", c)].iloc[i] for i, c in enumerate(pick)])
    y = np.array([w[("correct", c)].iloc[i] for i, c in enumerate(pick)], float)
    conf, agree = np.maximum(py, 1 - py), (pred == card).astype(float)
    X0, X1 = _trust_features(conf, agree), _trust_features(conf, agree, flag)
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    s0, s1 = np.empty(n), np.empty(n)
    for test in np.array_split(perm, folds):
        train = np.setdiff1d(perm, test)
        if len(np.unique(y[train])) < 2:
            s0[test] = s1[test] = y[train].mean()
            continue
        s0[test] = predict_logistic(fit_logistic(X0[train], y[train]), X0[test])
        s1[test] = predict_logistic(fit_logistic(X1[train], y[train]), X1[test])
    r = delong_paired(y, s1, s0)
    return {"n": n, "auroc_with_flag": r["auc1"], "auroc_without_flag": r["auc2"], "diff": r["diff"],
            "p_delong": r["p"], "share_flagged": float(flag.mean())}


def _accuracy_table(a: pd.DataFrame, conds) -> dict:
    out = {}
    for c in conds:
        x = a[(a["cond"] == c) & (a["valid"] == 1)]
        if len(x):
            out[c] = {"n": int(len(x)), "accuracy": float(x["correct"].mean()),
                      "wilson": wilson(int(x["correct"].sum()), int(len(x))), "auroc": auroc(x["y"], x["p_alarm"])}
    return out


def _paired_correct(a: pd.DataFrame):
    v = a[a["valid"] == 1]
    return v.pivot_table(index="id", columns="cond", values="correct", aggfunc="first")


def analyse_cell(df: pd.DataFrame, exploratory: bool = True, trust: bool = True) -> dict:
    """Every protocol measure for one model x dataset cell (valid readings only). trust=False leaves out H5 and H6
    (first study only; not part of the second study)."""
    a = slice_records(df)
    out = {"n_records": int(len(df)), "n_invalid": int((df["valid"] == 0).sum()),
           "n_records_imputed": int((df.get("n_imputed", pd.Series(0, index=df.index)) > 0).sum()),
           "validity": validity(df), "by_condition": _accuracy_table(a, CONDITIONS)}
    w = _paired_correct(a)
    if {"C0", "C2"} <= set(w.columns):
        x = w[["C0", "C2"]].dropna()
        out["H1"] = mcnemar_exact(x["C2"], x["C0"])           # predicted: C2 > C0
    if {"C0", "C4"} <= set(w.columns):
        x = w[["C0", "C4"]].dropna()
        out["H2"] = mcnemar_exact(x["C4"], x["C0"])           # predicted: C4 < C0
    out["H3"] = asymmetry(a)
    out["Q3_reversed"] = asymmetry(slice_records(df, "reversed", "A"))
    q = {}
    for key in ("adjusted", "reading_adjusted"):
        ci_a = out["H3"][key]["ci"] if out["H3"] else None
        ci_r = out["Q3_reversed"][key]["ci"] if out["Q3_reversed"] else None
        q[key] = q3_interpretation(ci_a, ci_r)
    out["Q3_by_statistic"] = q
    # D1: the framing answer is stated only when a and a' give the same answer
    out["Q3_interpretation"] = (q["adjusted"] if q["adjusted"] == q["reading_adjusted"]
                                else "inconclusive (a and a' disagree)")
    if trust:
        out["H5"] = overtrust(a)
        out["H6"] = flip_flag_mixed(a)
    if exploratory:
        ex = {}
        if {"C4", "C6"} <= set(w.columns):
            x = w[["C4", "C6"]].dropna()
            ex["C6_vs_C4_accuracy"] = mcnemar_exact(x["C6"], x["C4"])
            v = a[a["valid"] == 1]
            fol = lambda c: float((v[v["cond"] == c]["pred_alarm"] == v[v["cond"] == c]["card_dir"]).mean())
            ex["follow_rate"] = {"C2": fol("C2"), "C4": fol("C4"), "C6": fol("C6")}
        rob = {}
        for s in ROBUST_GRID["shot_sets"]:
            r = slice_records(df, "alarm", s)
            if len(r) == 0:
                continue
            wr = _paired_correct(r)
            rob[s] = {"by_condition": _accuracy_table(r, ROBUST_GRID["conditions"]), "H3": asymmetry(r)}
            if {"C0", "C2", "C4"} <= set(wr.columns):
                x = wr[["C0", "C2", "C4"]].dropna()
                rob[s]["H1"] = mcnemar_exact(x["C2"], x["C0"])
                rob[s]["H2"] = mcnemar_exact(x["C4"], x["C0"])
        ex["robustness_example_sets"] = rob
        out["exploratory"] = ex
    return out


# ----------------------------------------------------------------------------------- hypothesis families
def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# hypothesis -> (p-value, estimate in the predicted direction?)
HYP_GETTERS = {
    "H1": lambda c: (_get(c, "H1", "p"), (_get(c, "H1", "diff") or 0) > 0),
    "H2": lambda c: (_get(c, "H2", "p"), (_get(c, "H2", "diff") or 0) < 0),
    # a cell where a cannot be estimated gets no p-value (it then enters Holm with p = 1 and is not supported)
    "H3": lambda c: (_get(c, "H3", "h3_test", "p") if _finite(_get(c, "H3", "h3_test", "p")) else None,
                     bool(_get(c, "H3", "h3_test", "both_positive"))),
    "H5": lambda c: (_get(c, "H5", "p_boot"), (_get(c, "H5", "overtrust") or 0) > 0),
    "H6": lambda c: (_get(c, "H6", "p_delong"), (_get(c, "H6", "diff") or 0) > 0),
}


def holm_family(entries: dict) -> dict:
    """entries: key -> {"p": p or None, "direction_ok": bool, "included": bool}. Holm over all keys; a key that is
    not included (failed validity, not estimable or not tested) enters with p = 1 (Deviation D3, item 3).
    Adds p_holm and supported (Holm p < 0.05 in the predicted direction)."""
    keys = list(entries)
    ps = []
    for k in keys:
        e = entries[k]
        p = e.get("p")
        ps.append(1.0 if (not e.get("included") or p is None or (isinstance(p, float) and np.isnan(p))) else float(p))
    adj = holm(ps)
    out = {}
    for k, p_used, pa in zip(keys, ps, adj):
        e = dict(entries[k])
        e.update(p_used=p_used, p_holm=float(pa),
                 supported=bool(e.get("included") and pa < 0.05 and e.get("direction_ok")))
        out[k] = e
    return out


def supported_overall(family: dict, instruct_tags, datasets=DATASETS) -> dict:
    """Section 7: supported overall if supported in at least 2 of 3 datasets for each instruct model."""
    per_model = {}
    for tag in instruct_tags:
        k = sum(bool(family.get(f"{tag}|{ds}", {}).get("supported")) for ds in datasets)
        per_model[tag] = {"datasets_supported": int(k), "ok": k >= 2}
    return {"per_model": per_model, "supported_overall": bool(per_model) and all(v["ok"] for v in per_model.values())}


def confirmatory(cells: dict, h4: dict) -> dict:
    """cells: 'tag|dataset' -> analyse_cell result for every scored model; h4: 'family|dataset' -> result of
    adjusted_difference (or None if not tested). Returns the primary and secondary families."""
    instruct = [REGISTERED_MODELS[m][2] for m in REGISTERED_MODELS if REGISTERED_MODELS[m][0] == "instruct"]
    out = {"primary": {}, "secondary": {}}
    for h in ("H1", "H2", "H3", "H5", "H6"):
        ent = {}
        for tag in instruct:
            for ds in DATASETS:
                c = cells.get(f"{tag}|{ds}")
                p, ok = HYP_GETTERS[h](c) if c else (None, False)
                ent[f"{tag}|{ds}"] = {"p": p, "direction_ok": bool(ok),
                                      "included": bool(c and c["validity"]["passes"] and p is not None)}
        fam = holm_family(ent)
        if h in ("H1", "H2", "H3"):
            out["primary"][h] = {"cells": fam, "overall": supported_overall(fam, instruct)}
        else:
            out["secondary"][h] = {"cells": fam}
    ent = {}
    for fam_name in ("qwen", "llama"):
        for ds in DATASETS:
            r = h4.get(f"{fam_name}|{ds}")
            ok = bool(r) and _finite(_get(r, "h4_test", "p"))
            ent[f"{fam_name}|{ds}"] = {"p": r["h4_test"]["p"] if ok else None,
                                       "direction_ok": bool(ok and r["h4_test"]["both_positive"]), "included": ok}
    out["secondary"]["H4"] = {"cells": holm_family(ent)}
    return out


# ----------------------------------------------------------------------------------- second study (prereg-v2)
def _sigmoid(x):
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def choice_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["framing"].isin(CHOICE_FRAMINGS) & (df["shot_set"] == "A")]


def choice_pooled(df: pd.DataFrame) -> pd.DataFrame | None:
    """Answer-option framing (PREREGISTRATION_V2.md, Section 5): per patient and condition, the mean over the two
    option orders of the log-odds of the alarming outcome. A patient and condition with an invalid reading in either
    order is left out. Returns records in the first study's format (framing 'choice', example set A)."""
    c = choice_records(df)
    if len(c) == 0:
        return None
    v = c[c["valid"] == 1]
    if len(v) == 0:
        return None
    g = v.groupby(["id", "cond"], sort=True)
    agg = g.agg(lo_alarm=("lo_alarm", "mean"), n_orders=("framing", "nunique"), y=("y", "first"),
                card_dir=("card_dir", "first"), xgb_p=("xgb_p", "first")).reset_index()
    agg = agg[agg["n_orders"] == 2].drop(columns="n_orders")
    p = _sigmoid(agg["lo_alarm"].to_numpy(float))
    agg["p_alarm"] = p
    agg["p_yes"] = p                      # only so that _wide finds the column; not used for the answer-option framing
    agg["pred_alarm"] = (agg["lo_alarm"].to_numpy(float) > 0).astype(int)
    agg["correct"] = (agg["pred_alarm"] == agg["y"].astype(int)).astype(int)
    agg["valid"] = 1
    agg["framing"], agg["shot_set"] = "choice", "A"
    return agg


def validity_choice(df: pd.DataFrame) -> dict:
    """Validity of an answer-option cell (PREREGISTRATION_V2.md, Section 5): median A+B mass over every answer-option
    record of the cell (an unusable reading counts as 0) >= 0.5, and accuracy of the averaged C0 reading >= 0.55."""
    c = choice_records(df)
    mass = c["yn_mass"].astype(float).fillna(0.0)
    pooled = choice_pooled(df)
    c0 = pooled[pooled["cond"] == "C0"] if pooled is not None else pd.DataFrame(columns=["correct"])
    med = float(mass.median()) if len(mass) else float("nan")
    acc = float(c0["correct"].mean()) if len(c0) else float("nan")
    return {"median_ab_mass": med, "c0_accuracy": acc, "n_choice_records": int(len(c)), "n_c0_pooled": int(len(c0)),
            "passes": bool(med >= VALIDITY_MIN_MASS and acc >= VALIDITY_MIN_C0_ACC)}


def _paired_asym_difference(wx, wy) -> dict | None:
    """a(x) - a(y) and a'(x) - a'(y) on the patients present in both wide tables, with a paired bootstrap over those
    patients (10,000 resamples, seed 2026; every regression refitted in every resample). Intersection-union p = max
    of the two p-values. Also Delta(x) - Delta(y)."""
    if wx is None or wy is None:
        return None
    ids = sorted(set(wx.index) & set(wy.index))
    if len(ids) < 10:
        return None
    zx, zy = asym_arrays(wx.loc[ids]), asym_arrays(wy.loc[ids])
    adj = lambda z, b: adjusted_asymmetry(z["l0"][b], z["la"][b], z["ls"][b], z["t"][b], z["z"][b], z["y"][b])
    rdj = lambda z, b: reading_adjusted_asymmetry(z["l0"][b], z["la"][b], z["ls"][b], z["t"][b])
    dlt = lambda z, b: _asym_stats(z["c0"][b], z["pa"][b], z["ps"][b])[2]
    n = len(ids)
    full = np.arange(n)
    est, est0, est_d = (f(zx, full) - f(zy, full) for f in (adj, rdj, dlt))
    rng = np.random.default_rng(BOOT_SEED)
    boots, boots0, boots_d = np.empty(BOOT_RESAMPLES), np.empty(BOOT_RESAMPLES), np.empty(BOOT_RESAMPLES)
    for i in range(BOOT_RESAMPLES):
        b = rng.integers(0, n, n)
        boots[i], boots0[i], boots_d[i] = (f(zx, b) - f(zy, b) for f in (adj, rdj, dlt))
    p, p0 = boot_p(boots), boot_p(boots0)
    ok = bool(np.isfinite(est) and np.isfinite(est0))
    return {"n": n, "a_x": adj(zx, full), "a_y": adj(zy, full), "diff": float(est), "ci": _pct(boots), "p_boot": p,
            "reading_adjusted_diff": {"a_x": rdj(zx, full), "a_y": rdj(zy, full), "diff": float(est0),
                                      "ci": _pct(boots0), "p_boot": p0},
            "test": {"p": max(p, p0) if ok else float("nan"), "both_negative": bool(ok and est < 0 and est0 < 0),
                     "both_positive": bool(ok and est > 0 and est0 > 0)},
            "delta_diff": {"diff": float(est_d), "ci": _pct(boots_d), "p_boot": boot_p(boots_d)}}


def framing_difference(df: pd.DataFrame) -> dict | None:
    """R2, secondary comparison: delta = a(answer option) - a(alarm Yes/No), and the same for a', same model and
    dataset, paired over the patients valid in both framings. delta > 0: weaker without the words Yes and No."""
    pooled = choice_pooled(df)
    if pooled is None:
        return None
    return _paired_asym_difference(_wide(pooled), _wide(slice_records(df)))


def order_difference(df: pd.DataFrame) -> dict | None:
    """Example-order control: a(examples safe first, 'Asf') - a(registered order, set A), alarm framing, same model and
    dataset, paired over the patients valid in both."""
    return _paired_asym_difference(_wide(slice_records(df, "alarm", "Asf")), _wide(slice_records(df)))


def stage_difference(df_later: pd.DataFrame, df_earlier: pd.DataFrame,
                     later_set: str = "A", earlier_set: str = "A") -> dict | None:
    """Stage trace (R3): a(later stage) - a(earlier stage) and the same for a', alarm framing, example set A (or the
    base-format set 'Abase' for the base-format control)."""
    return _paired_asym_difference(_wide(slice_records(df_later, "alarm", later_set)),
                                   _wide(slice_records(df_earlier, "alarm", earlier_set)))


VALIDITY_MIN_C0_AUROC = 0.55


def validity_trace(cell: dict | None) -> dict:
    """Validity for R3 and the stage trace (PREREGISTRATION_V2.md): median Yes+No mass >= 0.5 and C0 AUROC >= 0.55
    (alarm framing, example set A). AUROC is used instead of accuracy because a strong lean of a base model lowers
    accuracy without making its readings uninformative, and the statistic a is built to absorb a lean."""
    mass = _get(cell, "validity", "median_yn_mass")
    au = _get(cell, "by_condition", "C0", "auroc")
    ok = bool(_finite(mass) and _finite(au) and mass >= VALIDITY_MIN_MASS and au >= VALIDITY_MIN_C0_AUROC)
    return {"median_yn_mass": mass, "c0_auroc": au, "passes": ok}


def analyse_choice_cell(df: pd.DataFrame) -> dict:
    """R2 for one model x dataset: validity, accuracy by condition and the asymmetry on the averaged answer-option
    reading; the mean preference for the letter A (log P(A) - log P(B), all valid answer-option records); and the
    change between framings when the model also has the Yes/No grid."""
    c = choice_records(df)
    pooled = choice_pooled(df)
    out = {"n_records": int(len(c)), "n_invalid": int((c["valid"] == 0).sum()),
           "n_records_imputed": int((c.get("n_imputed", pd.Series(0, index=c.index)) > 0).sum()),
           "validity": validity_choice(df)}
    v = c[c["valid"] == 1]
    out["letter_A_logodds_mean"] = float((v["lp_yes"] - v["lp_no"]).mean()) if len(v) else float("nan")
    if pooled is not None:
        out["by_condition"] = _accuracy_table(pooled, ("C0", "C2", "C4"))
        out["R2"] = asymmetry(pooled)
    else:
        out["by_condition"], out["R2"] = {}, None
    has_yesno = len(slice_records(df)) > 0
    out["validity_yesno"] = validity(df) if has_yesno else None
    out["framing_difference"] = framing_difference(df) if has_yesno else None
    return out


def _both_negative(asym: dict | None) -> bool:
    a, a0 = _get(asym, "adjusted", "a"), _get(asym, "reading_adjusted", "a")
    return bool(_finite(a) and _finite(a0) and a < 0 and a0 < 0)


def _iut_p(asym: dict | None):
    p = _get(asym, "h3_test", "p")
    return p if _finite(p) else None


def models_v2(role=None, kind=None) -> list:
    return [m["tag"] for m in REGISTERED_MODELS_V2.values()
            if (role is None or m["role"] in ((role,) if isinstance(role, str) else role))
            and (kind is None or m["kind"] == kind)]


def _count_rule(family: dict, tags, need_datasets: int, need_models: int, datasets=DATASETS) -> dict:
    per = {}
    for tag in tags:
        k = sum(bool(family.get(f"{tag}|{ds}", {}).get("supported")) for ds in datasets)
        per[tag] = {"datasets_supported": int(k), "ok": k >= need_datasets}
    n_ok = sum(v["ok"] for v in per.values())
    return {"per_model": per, "models_ok": int(n_ok), "models_needed": need_models,
            "supported_overall": bool(n_ok >= need_models)}


def confirmatory_v2(cells: dict, choice_cells: dict, r3: dict) -> dict:
    """PREREGISTRATION_V2.md, Section 6.
    cells: 'tag|dataset' -> analyse_cell result (Yes/No grid); choice_cells: 'tag|dataset' -> analyse_choice_cell
    result; r3: dataset -> adjusted_difference(Olmo 3 final, Olmo 3 base) or None.
    R1: new instruct models, a < 0 and a' < 0 (alarm framing, set A); Holm over 9 cells; replicated if >= 2 of 3
        datasets for >= 2 of the 3 new models.
    R2: eligible models (the two first-study models + new models meeting the R1 per-model criterion), answer-option
        framing; Holm over their cells; supported if >= 2 of 3 datasets for a majority of the eligible models.
    R3: Olmo 3 final - base, both differences < 0; validity by validity_trace; Holm over 3 datasets; supported if >= 2
        of 3 datasets."""
    out = {}
    new = models_v2("new")
    ent = {}
    for tag in new:
        for ds in DATASETS:
            c = cells.get(f"{tag}|{ds}")
            h = c.get("H3") if c else None
            p = _iut_p(h)
            ent[f"{tag}|{ds}"] = {"p": p, "direction_ok": _both_negative(h),
                                  "included": bool(c and c["validity"]["passes"] and p is not None),
                                  "a": _get(h, "adjusted", "a"), "a_prime": _get(h, "reading_adjusted", "a"),
                                  "delta": _get(h, "delta", "delta")}
    fam = holm_family(ent)
    out["R1"] = {"cells": fam, "overall": _count_rule(fam, new, 2, 2)}
    # R2 is evaluated on the models whose Yes/No asymmetry is established: the two first-study models and each new
    # model that meets the R1 per-model criterion (at least 2 of 3 datasets)
    eligible = models_v2("v1", "instruct") + [t for t in new if out["R1"]["overall"]["per_model"][t]["ok"]]
    five = eligible
    ent = {}
    for tag in five:
        for ds in DATASETS:
            c = choice_cells.get(f"{tag}|{ds}")
            h = c.get("R2") if c else None
            p = _iut_p(h)
            ent[f"{tag}|{ds}"] = {"p": p, "direction_ok": _both_negative(h),
                                  "included": bool(c and c["validity"]["passes"] and p is not None),
                                  "a": _get(h, "adjusted", "a"), "a_prime": _get(h, "reading_adjusted", "a")}
    fam = holm_family(ent)
    need = len(five) // 2 + 1                     # a majority of the eligible models
    overall = _count_rule(fam, five, 2, need)
    overall["eligible_models"] = list(five)
    # interpretation rule for R2 (Section 6). A cell counts only if it is valid in both framings and a and a' are both
    # below 0 in the alarm Yes/No framing; per model, the majority is over its counted cells (at least one); a cell
    # 'is weaker with A/B' if delta > 0 for both statistics with the unadjusted intersection-union p < 0.05.
    word = {}
    for tag in five:
        diffs = []
        for ds in DATASETS:
            c = choice_cells.get(f"{tag}|{ds}") or {}
            d = c.get("framing_difference")
            if (d and _finite(_get(d, "test", "p")) and _get(c, "validity", "passes")
                    and _get(c, "validity_yesno", "passes") and _finite(d.get("a_y"))
                    and _finite(_get(d, "reading_adjusted_diff", "a_y"))
                    and d["a_y"] < 0 and d["reading_adjusted_diff"]["a_y"] < 0):
                diffs.append(d)
        hit = sum(bool(d["test"]["both_positive"] and d["test"]["p"] < 0.05) for d in diffs)
        word[tag] = {"cells_counted": len(diffs), "cells_weaker_with_answer_options": int(hit),
                     "majority": bool(diffs) and hit > len(diffs) / 2}
    n_word = sum(v["majority"] for v in word.values())
    if overall["supported_overall"]:
        interp = ("follows the outcome, and is larger with the Yes/No question" if n_word >= need
                  else "follows the outcome")
    elif n_word >= need:
        interp = "depends on the Yes/No question format"
    else:
        interp = "unresolved"
    out["R2"] = {"cells": fam, "overall": overall, "framing_comparison": word, "interpretation": interp}
    ent = {}
    for ds in DATASETS:
        r = r3.get(ds)
        p = _get(r, "h4_test", "p")
        valid = all(validity_trace(cells.get(f"{t}|{ds}"))["passes"] for t in ("olmo3-instruct", "olmo3-base"))
        ok = bool(r) and _finite(p) and bool(valid)
        neg = bool(ok and r["diff"] < 0 and r["reading_adjusted_diff"]["diff"] < 0)
        ent[f"olmo3|{ds}"] = {"p": p if ok else None, "direction_ok": neg, "included": ok, "both_valid": bool(valid),
                              "diff_a": _get(r, "diff"), "diff_a_prime": _get(r, "reading_adjusted_diff", "diff")}
    fam = holm_family(ent)
    k = sum(v["supported"] for v in fam.values())
    out["R3"] = {"cells": fam, "overall": {"datasets_supported": int(k), "supported_overall": bool(k >= 2)}}
    return out


# ----------------------------------------------------------------------------------- memorization check (D3.7)
class PromptTooLong(RuntimeError):
    pass


class TabmemLLM:
    """tabmemcheck's LLM interface on top of a loaded backend: greedy decoding, chat mode for instruct models
    (through the model's chat template), completion mode for base models (with BOS when the model has one)."""

    def __init__(self, backend, kind, max_model_len=MAX_MODEL_LEN):
        self.be, self.kind, self.max_model_len = backend, kind, max_model_len
        self.chat_mode = kind == "instruct"
        self.calls, self.max_prompt_tokens = 0, 0

    def _gen(self, ids, max_tokens):
        if len(ids) + int(max_tokens) > self.max_model_len:      # never shorten a response: report instead
            raise PromptTooLong(f"{len(ids)} prompt tokens + {int(max_tokens)} new tokens > {self.max_model_len}")
        self.calls += 1
        self.max_prompt_tokens = max(self.max_prompt_tokens, len(ids))
        return self.be.generate_text([ids], int(max_tokens))[0]

    def completion(self, prompt, temperature, max_tokens):
        tok = self.be.tokenizer
        ids = tok.encode(prompt, add_special_tokens=False)
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None:
            ids = [bos] + ids
        return self._gen(ids, max_tokens)

    def chat_completion(self, messages, temperature, max_tokens):
        tok = self.be.tokenizer
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._gen(tok.encode(text, add_special_tokens=False), max_tokens)

    def __repr__(self):
        return f"TabmemLLM({self.be.name}, {self.kind})"


def _csv_rows(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def run_memcheck(files: dict, llm, log=print) -> dict:
    """tabmemcheck header test and row completion test per dataset (Deviation D3, item 7). Flag: at least one
    row completed verbatim. The library's console output is captured and returned in 'stdout'."""
    import contextlib
    import tabmemcheck as tm
    tm.config.max_tokens = MEMCHECK["max_tokens"]
    tm.config.temperature = 0
    tm.config.print_prompts = tm.config.print_responses = False
    out = {}
    for ds, path in files.items():
        res = {"file": os.path.basename(str(path)) if path else None}
        if not path or not os.path.exists(str(path)):
            res["status"] = "file not found"
            out[ds] = res
            log(f"  [{ds}] memorization check skipped: file not found")
            continue
        path = str(path)
        rows = _csv_rows(path)
        res["n_rows"] = len(rows)
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            # header test: run separately at each default split row, skip rows whose prompt does not fit, keep the best
            best, tried = None, []
            for row in (2, 4, 6, 8):
                try:
                    hp, hc, hr = tm.header_test(path, llm, split_rows=[row], rng=np.random.default_rng(MEMCHECK["seed"]),
                                                verbose=False)
                    m = 0
                    for c, r in zip(hc or "", hr or ""):
                        if c != r:
                            break
                        m += 1
                    tried.append({"split_row": row, "status": "ok", "matched_chars": m})
                    if best is None or m > best["matched_chars"]:
                        best = {"status": "ok", "split_row": row, "matched_chars": m, "true_completion": (hc or "")[:500],
                                "response": (hr or "")[:500], "prompt_tail": (hp or "")[-300:]}
                except PromptTooLong as e:
                    tried.append({"split_row": row, "status": f"prompt too long ({e})"})
                except Exception as e:                       # e.g. tabmemcheck errors on empty responses
                    tried.append({"split_row": row, "status": f"error: {e!r}"[:300]})
            res["header"] = dict(best or {"status": "not run: " + "; ".join(t["status"] for t in tried)}, split_rows=tried)
            nq = min(25, len(rows) - 11)
            res["row"] = {"status": "not run"}
            steps = MEMCHECK["few_shot_steps"] if llm.chat_mode else (None,)
            for fs in steps:
                try:
                    kw = dict(num_queries=nq, rng=np.random.default_rng(MEMCHECK["seed"]), print_levenshtein=False)
                    if fs is not None:
                        kw["few_shot"] = fs
                    suffixes, responses = tm.row_completion_test(path, llm, **kw)
                    exact = [bool(s.strip() and s.strip() in r.strip()) for s, r in zip(suffixes, responses)]
                    res["row"] = {"status": "ok", "few_shot": fs, "num_queries": nq, "exact_matches": int(sum(exact)),
                                  "duplicate_row_share": float(1 - len(set(rows)) / len(rows)),
                                  "matched_rows": [s[:300] for s, x in zip(suffixes, exact) if x][:10],
                                  "examples": [{"true": s[:300], "response": r[:300]}
                                               for s, r in list(zip(suffixes, responses))[:3]]}
                    break
                except PromptTooLong as e:
                    res["row"] = {"status": f"prompt too long with few_shot={fs} ({e})"}
                except Exception as e:
                    res["row"] = {"status": f"error: {e!r}"[:300]}
                    break
        res["flag_verbatim_rows"] = bool(res["row"].get("exact_matches", 0) > 0)
        res["seconds"] = round(time.time() - t0, 1)
        res["stdout"] = buf.getvalue()[-20000:]
        out[ds] = res
        hdr = (f"{res['header']['matched_chars']} chars matched" if "matched_chars" in res["header"]
               else res["header"]["status"])
        log(f"  [{ds}] header: {hdr} | rows: "
            f"{res['row'].get('exact_matches', res['row']['status'])}/{res['row'].get('num_queries', '-')} verbatim"
            f"{' (few-shot ' + str(res['row'].get('few_shot')) + ')' if res['row'].get('few_shot') else ''} | "
            f"flagged: {res['flag_verbatim_rows']} | {res['seconds']}s")
    return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "smoke":
        smoke_test(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "instruct",
                   sys.argv[4] if len(sys.argv) > 4 else None)
    elif len(sys.argv) >= 6 and sys.argv[1] == "engcheck":
        engineering_check(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                          sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "None" else None,
                          tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else None)
