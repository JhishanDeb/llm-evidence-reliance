"""Shared code for the DEV PHASE of the pre-registered study
"Do LLMs reason with model explanations, or defer to them one way?"

Protocol: PREREGISTRATION.md, release prereg-v1 of https://github.com/JhishanDeb/llm-evidence-reliance
Everything the protocol fixes (prompt text, seeds, sizes, inclusion rules) is a constant or function in this
file, so the dev notebook and the later main notebook use exactly the same code.
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

CODE_VERSION = "dev-1.1"

# =============================================================================================================
# 1. Protocol constants (prereg-v1)
# =============================================================================================================
PROTOCOL_URL = "https://raw.githubusercontent.com/JhishanDeb/llm-evidence-reliance/prereg-v1/PREREGISTRATION.md"
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
    info = {"records_parsed": len(feats), "with_outcome": len(df), "positives": int(df["y"].sum())}
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
    info = {"files": len(files), **{f"status_{k}": v for k, v in sorted(status.items())},
            "included": len(df), "positives": int(df["y"].sum())}
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
            "id_mapping_tables": {k: len(v) for k, v in maps.items()}}
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


class Bundle:
    """Everything for one dataset: splits, the 16-feature XGBoost, dev patients, cards, partners, examples."""

    def __init__(self, name, df, feat_cols, cat_cols, data_info, pilot_ids=(), log=print):
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
        # cards for the dev patients
        Xp = X.iloc[pick][self.features]
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
        # C6 partners: a random dev patient whose card points the other way
        prs = np.random.RandomState(SPLIT_SEED)
        self.partner = {}
        for rid in sorted(self.ids):
            opp = sorted(o for o in self.ids if self.card_dir[o] != self.card_dir[rid])
            self.partner[rid] = opp[prs.randint(len(opp))] if opp else None
        agree = np.mean([self.card_dir[r] == int(self.xgb_p[r] >= 0.5) for r in self.ids])
        self.info["cards"] = {"card_points_alarm_share": float(np.mean(list(self.card_dir.values()))),
                              "card_agrees_with_xgb_label": float(agree),
                              "partners_missing": int(sum(v is None for v in self.partner.values()))}
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
        fp = {"dev_pool": sorted(df["id"].iloc[dev]), "test_pool": sorted(df["id"].iloc[test]),
              "features": self.features,
              "cards": {r: [(c["col"], round(c["shap"], 6)) for c in self.cards[r]] for r in sorted(self.ids)},
              "shots": self.shot_ids}
        self.info["fingerprint"] = {k: hashlib.sha256(json.dumps(v, default=str, sort_keys=True).encode()).hexdigest()[:16]
                                    for k, v in fp.items()}
        log(f"  {name}: train {len(tr)}, dev pool {len(dev)}, test pool {len(test)} | XGBoost AUROC (dev pool, "
            f"16 features) {self.info['xgb']['auroc_dev_pool_16']:.3f} | dev sample {len(pick)} | eligible test "
            f"positives {npos}{' -> FALLBACK RULE' if npos < TEST_PER_CLASS else ''}")

    # ------------------------------------------------------------------------------------------- prompts
    def _block(self, values: dict) -> str:
        return "\n".join(f"- {display_name(self.name, c)}: {display_value(self.name, c, values[c], self.id_maps)}"
                         for c in self.features)

    def question(self, framing: str) -> str:
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
        ex = []
        for k, (vals, yy) in enumerate(self.shots[shot_set], 1):
            yes = (yy == 1) if self.yes_means_alarm(framing) else (yy == 0)
            ex.append(f"Example {k}\n{self._block(vals)}\nQuestion: {q}\nAnswer: {'Yes' if yes else 'No'}")
        parts = [f"Task: {self.spec['task']}", EXAMPLES_HEADER, *ex, PATIENT_HEADER + "\n" + self._block(self.values[rid])]
        card = self.shown_card(rid, cond)
        if cond == "C6" and card is None:
            return None, None
        if card is not None:
            parts.append(self.card_text(card))
        parts.append(f"Question: {q}\n{ANSWER_INSTRUCTION}")
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


class Backend:
    name = "base"
    topk = None

    def yes_no(self, prompts):
        """-> list of dicts {p_yes_raw, p_no_raw} (probabilities summed over the token variants)."""
        out = []
        for d in self.word_logprobs(prompts):
            vals = list(d.values())
            if any(not math.isfinite(v) for v in vals):
                raise NonFiniteLogprobError("non-finite logprob")
            py = sum(math.exp(d[t]) for t in self.variants["Yes"] if t in d)
            pn = sum(math.exp(d[t]) for t in self.variants["No"] if t in d)
            out.append({"p_yes_raw": py, "p_no_raw": pn})
        return out

    def info(self):
        return {"backend": self.name, "topk": self.topk,
                "variants": {w: [[i, self.strings[i]] for i in ids] for w, ids in self.variants.items()}}


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model_path, tensor_parallel_size=1, max_model_len=4096, seed=SPLIT_SEED, topk=20,
                 gpu_memory_utilization=0.80, max_num_batched_tokens=4096):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.variants, self.strings = find_word_variants(self.tokenizer)
        self.topk = topk
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

    def word_logprobs(self, prompts):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.topk)
        ids = [{"prompt_token_ids": self.tokenizer.encode(p, add_special_tokens=False)} for p in prompts]
        outs = self.llm.generate(ids, sp, use_tqdm=False)
        wanted = {t for v in self.variants.values() for t in v}
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            if not hasattr(lp, "items"):
                raise RuntimeError(f"unexpected vLLM logprobs container {type(lp)}; unset VLLM_FLAT_LOGPROBS")
            vals = {int(t): float(getattr(v, "logprob", v)) for t, v in lp.items()}
            if any(not math.isfinite(x) for x in vals.values()):   # checked BEFORE keeping only Yes/No ids
                raise NonFiniteLogprobError("non-finite logprob in the vLLM top-k")
            res.append({t: v for t, v in vals.items() if t in wanted})
        return res

    def info(self):
        import vllm
        d = super().info()
        d["vllm_version"] = vllm.__version__
        return d


class HFBackend(Backend):
    """transformers fallback: one forward pass per prompt, full next-token distribution."""
    name = "hf"

    def __init__(self, model_path, batch_size=8, device_map="auto"):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.variants, self.strings = find_word_variants(self.tokenizer)
        eos = eos_token_ids(model_path, self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos[0]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map if torch.cuda.is_available() else None, **{dkey: dtype})
        self.model.eval()
        self.batch_size = batch_size

    def word_logprobs(self, prompts):
        t = self.torch
        wanted = [i for v in self.variants.values() for i in v]
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
            if not bool(t.isfinite(logits).all()):
                raise NonFiniteLogprobError("non-finite logits")
            lp = t.log_softmax(logits, dim=-1)[:, wanted].cpu().tolist()
            res += [dict(zip(wanted, row)) for row in lp]
        return res

    def info(self):
        import transformers
        d = super().info()
        d["transformers_version"] = transformers.__version__
        return d


class _MockTok:
    def __init__(self):
        self.vocab = {"Yes": 1, "ĠYes": 2, "No": 3, "ĠNo": 4}

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids, **kw):
        inv = {v: k.replace("Ġ", " ") for k, v in self.vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4 + 1))

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class MockBackend(Backend):
    """Deterministic fake model for CPU tests. It reads the evidence impacts in the prompt and leans toward the
    alarming outcome more strongly than toward the safe one, so every analysis path has something to measure."""
    name = "mock"
    _IMPACT = re.compile(r"\(impact ([+-]\d+\.\d+)\)")

    def __init__(self, reversed_questions):
        self.tokenizer = _MockTok()
        self.variants, self.strings = find_word_variants(self.tokenizer)
        self.reversed_q = set(reversed_questions)

    def word_logprobs(self, prompts):
        res = []
        for p in prompts:
            key = p.split(PATIENT_HEADER)[-1].split("\n\n")[0]
            base = 3.0 * (_u(key) - 0.5)
            s = sum(float(x) for x in self._IMPACT.findall(p.split(PATIENT_HEADER)[-1]))
            z = base + (1.6 * s if s > 0 else 0.6 * s)
            p_alarm = 1 / (1 + math.exp(-z))
            q = re.findall(r"Question: (.*)\n" + re.escape(ANSWER_INSTRUCTION), p)
            reversed_ = bool(q) and q[-1] in self.reversed_q
            p_yes = 1 - p_alarm if reversed_ else p_alarm
            mass = 0.80 + 0.19 * _u(key, "m")
            res.append({2: math.log(mass * p_yes * 0.9 + 1e-12), 1: math.log(mass * p_yes * 0.1 + 1e-12),
                        4: math.log(mass * (1 - p_yes) * 0.9 + 1e-12), 3: math.log(mass * (1 - p_yes) * 0.1 + 1e-12)})
        return res


def smoke_test(model_path, kind="instruct"):
    """Run in a subprocess before loading vLLM in the notebook kernel: a failed engine start cannot then leave
    the kernel holding GPU memory."""
    import torch
    tp = 2 if torch.cuda.device_count() >= 2 else 1
    t0 = time.time()
    be = VLLMBackend(model_path, tensor_parallel_size=tp)
    text = "Is the sky usually blue on a clear day?\n" + ANSWER_INSTRUCTION
    r = be.yes_no([render(text, kind, be.tokenizer)])[0]
    res = {"ok": True, "load_seconds": time.time() - t0, "tensor_parallel_size": tp, **r, "info": be.info()}
    print("SMOKE_RESULT " + json.dumps(res, default=str))
    return res


# =============================================================================================================
# 7. Scoring records and dev analysis
# =============================================================================================================
def finish_record(rec: dict, r: dict, framing: str) -> dict:
    """Adds the reading of one prompt. A prompt with no probability on any Yes/No token (yn_mass = 0) is
    marked valid = 0 and left out of every analysis; it is never turned into a 'No'."""
    py, pn = r["p_yes_raw"], r["p_no_raw"]
    mass = py + pn
    rec.update(p_yes_raw=py, p_no_raw=pn, yn_mass=mass)
    if not mass > 0:
        rec.update(valid=0, p_yes=float("nan"), pred_yes=-1, p_alarm=float("nan"), pred_alarm=-1, correct=-1)
        return rec
    p_yes = py / mass
    yes = p_yes > 0.5
    rec.update(valid=1, p_yes=p_yes, pred_yes=int(yes),
               p_alarm=p_yes if framing == "alarm" else 1 - p_yes,
               pred_alarm=int(yes) if framing == "alarm" else int(not yes))
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
def _wide(df: pd.DataFrame):
    """Per patient: predictions, P(alarm) and card directions for C0, C2, C4 (valid readings only)."""
    d = df[df["valid"] == 1]
    w = d.pivot_table(index="id", columns="cond", values=["pred_alarm", "p_alarm", "card_dir", "y", "correct",
                                                           "p_yes"], aggfunc="first")
    need = [("pred_alarm", c) for c in ("C0", "C2", "C4")]
    if not all(k in w.columns for k in need):
        return None
    return w.dropna(subset=need)


def _asym_stats(c0, pa, ps):
    n_safe, n_alarm = int((c0 == 0).sum()), int((c0 == 1).sum())
    A = ((pa == 1) & (c0 == 0)).sum() / n_safe if n_safe else np.nan
    S = ((ps == 0) & (c0 == 1)).sum() / n_alarm if n_alarm else np.nan
    return A, S, A - S


def asymmetry(df: pd.DataFrame):
    """Protocol Section 6, on one framing x example-set slice: A, S, Delta = A - S with a paired bootstrap
    (10,000 resamples of patients, seed 2026) and the two-sided bootstrap p-value of Section 7 (H3).
    Also the secondary log-odds asymmetry and the follow rates."""
    w = _wide(df)
    if w is None or len(w) == 0:
        return None
    cd = w[("card_dir", "C2")].to_numpy()
    c0 = w[("pred_alarm", "C0")].to_numpy().astype(int)
    c2, c4 = w[("pred_alarm", "C2")].to_numpy().astype(int), w[("pred_alarm", "C4")].to_numpy().astype(int)
    pa, ps = np.where(cd == 1, c2, c4), np.where(cd == 1, c4, c2)
    l0 = logit(w[("p_alarm", "C0")])
    l2, l4 = logit(w[("p_alarm", "C2")]), logit(w[("p_alarm", "C4")])
    la, ls = np.where(cd == 1, l2, l4), np.where(cd == 1, l4, l2)
    A, S, D = _asym_stats(c0, pa, ps)
    lo_asym = float(np.mean(la - l0) - np.mean(l0 - ls))
    rng = np.random.default_rng(BOOT_SEED)
    n = len(c0)
    boots, lboots = [], []
    for _ in range(BOOT_RESAMPLES):
        b = rng.integers(0, n, n)
        d = _asym_stats(c0[b], pa[b], ps[b])[2]
        if not np.isnan(d):
            boots.append(d)
        lboots.append(np.mean(la[b] - l0[b]) - np.mean(l0[b] - ls[b]))
    boots = np.asarray(boots)
    ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) else [np.nan, np.nan]
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))) if len(boots) else float("nan")
    return {"n": int(n), "A": float(A), "S": float(S), "delta": float(D), "delta_ci": [float(x) for x in ci],
            "p_boot": p, "logodds_asym": lo_asym,
            "logodds_asym_ci": [float(x) for x in np.percentile(lboots, [2.5, 97.5])],
            "follow_alarm_card": float(np.mean(pa == 1)), "follow_safe_card": float(np.mean(ps == 0)),
            "flip_rate_C2_C4": float(np.mean(c2 != c4))}


def delta_difference(df_instruct: pd.DataFrame, df_base: pd.DataFrame) -> dict:
    """H4: Delta(instruct) - Delta(base) for the same model family, paired bootstrap over the same patients."""
    wi, wb = _wide(df_instruct), _wide(df_base)
    ids = sorted(set(wi.index) & set(wb.index))
    arrs = []
    for w in (wi.loc[ids], wb.loc[ids]):
        cd = w[("card_dir", "C2")].to_numpy()
        c0 = w[("pred_alarm", "C0")].to_numpy().astype(int)
        c2, c4 = w[("pred_alarm", "C2")].to_numpy().astype(int), w[("pred_alarm", "C4")].to_numpy().astype(int)
        arrs.append((c0, np.where(cd == 1, c2, c4), np.where(cd == 1, c4, c2)))
    diff = _asym_stats(*arrs[0])[2] - _asym_stats(*arrs[1])[2]
    rng = np.random.default_rng(BOOT_SEED)
    n = len(ids)
    boots = []
    for _ in range(BOOT_RESAMPLES):
        b = rng.integers(0, n, n)
        d = _asym_stats(*(a[b] for a in arrs[0]))[2] - _asym_stats(*(a[b] for a in arrs[1]))[2]
        if not np.isnan(d):
            boots.append(d)
    boots = np.asarray(boots)
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))) if len(boots) else float("nan")
    return {"n": n, "diff": float(diff), "ci": np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) else None,
            "p_boot": p}


def _trust_features(conf, agree, flag=None):
    cols = [np.asarray(conf, float), np.asarray(agree, float)]
    if flag is not None:
        cols.append(np.asarray(flag, float))
    return np.column_stack(cols)


def overtrust(df: pd.DataFrame) -> dict | None:
    """H5: logistic regression of correctness on [confidence, agrees-with-card] fitted on C2 records, applied to
    C4 records. Over-trust = mean predicted P(correct) - observed accuracy on C4. Paired bootstrap over patients
    (the regression is refitted in every resample)."""
    w = _wide(df)
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
    return {"n": n, "overtrust": est, "ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "acc_C4": float(y4.mean()), "mean_predicted_C4": float(est + y4.mean())}


def flip_flag_mixed(df: pd.DataFrame, folds: int = 5) -> dict | None:
    """H6: mixed setting. Each patient gets the true (C2) or flipped (C4) card at random (p = 0.5, seed 2027).
    Trust regression with vs without the flip-consistency flag (prediction changes between C2 and C4);
    5-fold cross-fitted predictions; AUROC for correctness; paired DeLong test."""
    w = _wide(df)
    if w is None or len(w) < 20:
        return None
    w = w.sort_index()
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


def analyse_cell(df: pd.DataFrame) -> dict:
    """Every protocol measure for one model x dataset cell (valid readings only)."""
    d = df[df["valid"] == 1]
    a = d[(d["framing"] == "alarm") & (d["shot_set"] == "A")]
    out = {"n_records": int(len(df)), "n_invalid": int((df["valid"] == 0).sum())}
    acc = {}
    for c in CONDITIONS:
        x = a[a["cond"] == c]
        if len(x):
            acc[c] = {"n": int(len(x)), "accuracy": float(x["correct"].mean()),
                      "wilson": wilson(int(x["correct"].sum()), int(len(x))), "auroc": auroc(x["y"], x["p_alarm"])}
    out["by_condition"] = acc
    w = a.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna()
    if {"C0", "C2", "C4"} <= set(w.columns):
        out["H1"] = mcnemar_exact(w["C2"], w["C0"])   # predicted: C2 > C0
        out["H2"] = mcnemar_exact(w["C4"], w["C0"])   # predicted: C4 < C0
    out["H3"] = asymmetry(a)
    out["Q3_reversed"] = asymmetry(d[(d["framing"] == "reversed") & (d["shot_set"] == "A")])
    q = out["Q3_reversed"]
    if q:
        lo, hi = q["delta_ci"]
        out["Q3_interpretation"] = ("follows the outcome (alarm aversion)" if lo > 0 else
                                    "follows the answer word 'Yes'" if hi < 0 else "inconclusive")
    out["H5"] = overtrust(a)
    out["H6"] = flip_flag_mixed(a)
    return out


def holm_across(cells: dict) -> dict:
    """Holm correction within each hypothesis family across the given cells; adds p_holm and 'supported'
    (Holm-adjusted p < 0.05 in the predicted direction)."""
    fam = {"H1": lambda r: (r["H1"]["p"], r["H1"]["diff"] > 0) if r.get("H1") else (np.nan, False),
           "H2": lambda r: (r["H2"]["p"], r["H2"]["diff"] < 0) if r.get("H2") else (np.nan, False),
           "H3": lambda r: (r["H3"]["p_boot"], r["H3"]["delta"] > 0) if r.get("H3") else (np.nan, False)}
    keys = list(cells)
    for h, get in fam.items():
        ps = [get(cells[k])[0] for k in keys]
        adj = holm(ps)
        for k, pa in zip(keys, adj):
            if cells[k].get(h):
                cells[k][h]["p_holm"] = pa
                cells[k][h]["supported"] = bool(pa < 0.05 and get(cells[k])[1])
    return cells


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "smoke":
        smoke_test(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "instruct")
