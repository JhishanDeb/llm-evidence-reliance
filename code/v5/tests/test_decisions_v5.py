"""Checks that the decisions, fingerprints and hashes recorded in ler_v5.py follow from the outputs of the two
engineering checks by the rules of Section 10, and that the protocol contains everything the run notebook checks.
usage: python test_decisions_v5.py CODE_DIR ENGCHECK_ZIP ENGCHECK_RUN2_ZIP PREREGISTRATION_V5.md"""
import hashlib
import json
import sys
import zipfile

CODE, Z1, Z2, PROT = sys.argv[1:5]
sys.path.insert(0, CODE)
import ler_v5 as F  # noqa: E402

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def load(z):
    with zipfile.ZipFile(z) as f:
        return (json.loads(f.read("engcheck_v5.json")), json.loads(f.read("engcheck_log.json")),
                hashlib.sha256(f.read("ler_v5.py")).hexdigest())


r1, log1, code1 = load(Z1)
r2, log2, code2 = load(Z2)
merged = F.merge_checks(r1, r2)
check("merge: the second check replaces the first only for the models it ran",
      all(merged[m] is (r2[m] if m in r2 else r1[m]) for m in F.MODELS_V5) and set(r2) <= set(F.MODELS_V5))
n_fresh = sum(f["n"] for f in log1["fresh"].values())
n_gen = sum(f["generation_subsample"]["150"] for f in log1["fresh"].values())
dec = F.decisions_from_check(merged, n_fresh, n_gen)
check(f"recorded decisions = rules (i) to (iii) on both checks: {json.dumps(dec)}", F.ENGCHECK_DECISIONS == dec)
fp1 = {ds: f["fingerprint"] for ds, f in log1["fresh"].items()}
fp2 = {ds: f["fingerprint"] for ds, f in log2["fresh"].items()}
check("both checks built the same fresh samples, and FRESH_FP records them", fp1 == fp2 == F.FRESH_FP)
check("the earlier dev and test fingerprints matched in both checks",
      all(all(v.values()) for v in log1["fingerprint_check"].values()) and all(all(v.values()) for v in log2["fingerprint_check"].values()))
check("ENGCHECK_ZIP_SHA256 holds the SHA-256 of both outputs",
      tuple(F.ENGCHECK_ZIP_SHA256) == tuple(hashlib.sha256(open(z, "rb").read()).hexdigest() for z in (Z1, Z2)))
check("ENGCHECK_CODE_SHA256 holds the ler_v5.py of both checks (the same v5-0.1)",
      tuple(F.ENGCHECK_CODE_SHA256) == (code1,) and code1 == code2)
check("the registered code is v5-1.0", F.V5_VERSION == "v5-1.0")
text = open(PROT, encoding="utf-8").read()
miss = F.check_against_protocol_v5(text)
check(f"the protocol contains every string the run notebook checks (missing: {miss})", not miss)
sha5 = hashlib.sha256(open(f"{CODE}/ler_v5.py", "rb").read()).hexdigest()
check("the protocol lists the SHA-256 of this ler_v5.py", sha5 in text)
print(f"\n{sum(ok)} of {len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
