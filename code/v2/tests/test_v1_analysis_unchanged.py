"""main-2.0's first-study analysis functions give the same numbers as main-1.0 on the real first-study records
(reduced bootstrap for speed). usage: python test_v1_analysis_unchanged.py V1_FROZEN V2_MODULE RUN_DIR"""
import importlib.util, json, sys
import numpy as np, pandas as pd
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
L1, L2 = load("ler_v1", sys.argv[1]), load("ler_v2", sys.argv[2])
L1.BOOT_RESAMPLES = L2.BOOT_RESAMPLES = 300
run = sys.argv[3]
def same(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a: same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), path
    else:
        assert a == b, (path, a, b)
for tag in ("qwen-instruct", "llama-base"):
    df = pd.read_csv(f"{run}/{tag}/records_{tag}.csv.gz", dtype={"id": str})
    for ds in ("physionet2012", "diabetes130"):
        d = df[df["dataset"] == ds]
        same(L1.analyse_cell(d), L2.analyse_cell(d))
        print(f"{tag} {ds}: analyse_cell identical")
fi = pd.read_csv(f"{run}/qwen-instruct/records_qwen-instruct.csv.gz", dtype={"id": str})
fb = pd.read_csv(f"{run}/qwen-base/records_qwen-base.csv.gz", dtype={"id": str})
d1 = L1.adjusted_difference(fi[fi.dataset == "physionet2012"], fb[fb.dataset == "physionet2012"])
d2 = L2.adjusted_difference(fi[fi.dataset == "physionet2012"], fb[fb.dataset == "physionet2012"])
same(d1, d2); print("adjusted_difference identical")
# the R3 helper on the same data equals the H4 function (same bootstrap, same numbers)
sd = L2.stage_difference(fi[fi.dataset == "physionet2012"], fb[fb.dataset == "physionet2012"])
assert sd["diff"] == d2["diff"] and sd["ci"] == d2["ci"] and sd["test"]["p"] == d2["h4_test"]["p"]
print("stage_difference equals adjusted_difference on the same pair")
print("FIRST-STUDY ANALYSIS UNCHANGED")
