"""CPU tests of ler_v5.py on synthetic data with fake models (never real patients or models).
usage: LER_P12_MIN_PER_SET=1500 LER_P19_MIN_PER_SET=1000 python unit_tests_v5.py CODE_DIR FAKEDATA_DIR [N_PER_DATASET]

Checks: the earlier modules are the registered versions; the fresh samples are disjoint from every earlier sample; every
prompt of every grid differs from a registered prompt only where Section 5 says; the fake models' built-in effects are
recovered by the registered analysis (a card that weakens the answer prior shows up in the lean change and in W against
no card, but not in W against the placebo card; an asymmetry in updating shows up against both baselines; the scaled
numbers cards and the probability evidence are read; the decision-level effect of generated answers has the sign of W);
and the decision rules of Section 7 and the engineering-check rules of Section 10 give the expected decisions."""
import hashlib
import json
import sys
import time
from pathlib import Path

CODE, FAKE = Path(sys.argv[1]), Path(sys.argv[2])
NPD = int(sys.argv[3]) if len(sys.argv) > 3 else 160
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
import ler_main as L
import ler_v5 as F

# the fake PhysioNet 2019 data have few positives; a smaller registered test sample leaves some for the fresh sample
# (the real test pool keeps 166 eligible positives and the dev pool 105 after the earlier samples)
L.TEST_PER_CLASS = 15

ok = []


def chk(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
chk("ler_main, ler_v3 and ler_v4 are the registered versions",
    (sha(CODE / "ler_main.py"), sha(CODE / "ler_v3.py"), sha(CODE / "ler_v4.py")) == (F.MAIN_SHA256, F.V3_SHA256, F.V4_SHA256))
chk("42 scored prompts per patient, 6 generated", F.SCORED_PER_PATIENT == 42 and len(F.GEN_GRID) == 6)

# ---------------------------------------------------------------- data, registered samples, fresh samples
B = {}
t0 = time.time()
for ds in F.DATASETS_V5:
    df, feats, cats, info = L.LOADERS[ds](FAKE, (), log=lambda *a: None)
    b = L.Bundle(ds, df, feats, cats, info, pilot_ids=L.PILOT_IDS_FALLBACK if ds == "physionet2012" else (),
                 sample="test", log=lambda *a: None)
    set_c = F.load_p12_set_c(FAKE, (), log=lambda *a: None) if ds == "physionet2012" else None
    inf = F.build_fresh(b, set_c, L.PILOT_IDS_FALLBACK, log=print)
    earlier = set(b._samples["test"]["ids"]) | set(b._samples["dev"]["ids"])
    chk(f"{ds}: fresh sample disjoint from the earlier test and dev samples",
        not (set(b.ids) & earlier) and inf["overlap_with_earlier_samples"] == 0)
    if ds == "physionet2012":
        chk("physionet2012: every fresh patient comes from set C", set(b.ids) <= set(set_c["id"]))
    else:
        pool = set(b.df["id"].iloc[np.concatenate([b.test_pool, b.dev_pool])])
        chk("physionet2019: every fresh patient comes from the test or dev pool", set(b.ids) <= pool)
    chk(f"{ds}: classes balanced, fallback rule as stated",
        sum(b.y.values()) * 2 == len(b.ids) and inf["fallback_rule_applied"] == (inf["per_class"] < F.FRESH_PER_CLASS))
    b2 = L.Bundle(ds, df, feats, cats, info, pilot_ids=L.PILOT_IDS_FALLBACK if ds == "physionet2012" else (),
                  sample="test", log=lambda *a: None)
    inf2 = F.build_fresh(b2, set_c, L.PILOT_IDS_FALLBACK, log=lambda *a: None)
    chk(f"{ds}: the fresh sample is reproducible (fingerprints)", inf2["fingerprint"] == inf["fingerprint"])
    g = F.gen_patients(b)
    chk(f"{ds}: generation subsample within the fresh sample, balanced", set(g) <= set(b.ids)
        and sum(b.y[i] for i in g) * 2 == len(g))
    B[ds] = b
print(f"data and samples: {time.time() - t0:.0f} s")

# ---------------------------------------------------------------- prompts
b = B["physionet2012"]
rid = b.ids[0]
n_checked = 0
for ds, bb in B.items():
    for r in bb.ids[:8]:
        for g in F.GRIDS:
            for cond, fr, shot in F.GRID_DEF[g]:
                text, cdir = F.prompt_v5(bb, r, cond, fr, shot)
                F.check_prompt_v5(bb, r, cond, fr, shot, text, cdir)
                n_checked += 1
chk(f"every prompt of every grid passes its check ({n_checked} prompts)", n_checked == 2 * 8 * 42)
tp, _ = F.prompt_v5(b, rid, "CP", "alarm")
t2, _ = F.prompt_v5(b, rid, "C2", "alarm")
chk("placebo: same header and factors as the true card, no direction words, impacts 0.00",
    tp.count("(impact 0.00)") == 5 and tp.split(L.EVIDENCE_HEADER)[0] == t2.split(L.EVIDENCE_HEADER)[0]
    and [ln.split(":")[0] for ln in tp.split(L.EVIDENCE_HEADER)[1].strip().splitlines()[:5]] ==
    [ln.split(":")[0] for ln in t2.split(L.EVIDENCE_HEADER)[1].strip().splitlines()[:5]])
tq2, d2 = F.prompt_v5(b, rid, "CQ2", "reversed")
tq4, d4 = F.prompt_v5(b, rid, "CQ4", "reversed")
p = F.prob_percent(b, rid)
chk("probability: CQ2 shows the model's percentage, CQ4 its complement, directions opposite",
    f"is {p}%." in tq2 and f"is {100 - p}%." in tq4 and (p == 50 or d2 == 1 - d4))
t05, _ = F.prompt_v5(b, rid, "C2", "alarm", "Anum05")
t20, _ = F.prompt_v5(b, rid, "C2", "alarm", "Anum2")
imp = lambda t: [float(x) for x in __import__("re").findall(r"\(impact ([+-]\d+\.\d+)\)", t)]
chk("dose: the x2 impacts are four times the x0.5 impacts (to rounding)",
    all(abs(a2 - 4 * a05) <= 0.031 for a05, a2 in zip(imp(t05), imp(t20))) and len(imp(t05)) == 5)
for name, t in (("placebo", tp), ("probability", tq2), ("numbers x0.5", t05)):
    print(f"\n--- {name} (fake patient), evidence block ---")
    print([blk for blk in t.split("\n\n") if blk.startswith("Evidence")][0])


# ---------------------------------------------------------------- fake runs and the registered analysis
def run_mock(be, bb, ds, ids):
    rec, gen = [], []
    for r in ids:
        for g in F.GRIDS:
            for cond, fr, shot in F.GRID_DEF[g]:
                text, cdir = F.prompt_v5(bb, r, cond, fr, shot)
                rd = F.read_v5(be, [F.render_v5(text, be.tokenizer)], fr)[0]
                x = {"key": f"{ds}|{r}|{cond}|{fr}|{shot}", "dataset": ds, "id": r, "cond": cond, "framing": fr,
                     "shot_set": shot, "y": bb.y[r], "card_dir": cdir, "xgb_p": float(bb.xgb_p[r])}
                rec.append(F.finish_record_v5(x, rd, fr))
        for cond, fr, shot in F.GEN_GRID:
            text, cdir = F.prompt_v5(bb, r, cond, fr, shot)
            out = F.generate(be, [F.render_free(text, be.tokenizer)], F.GEN_MAX_TOKENS)[0]
            gen.append({"key": f"{ds}|{r}|{cond}|{fr}|{shot}", "dataset": ds, "id": r, "cond": cond, "framing": fr,
                        "y": bb.y[r], "card_dir": cdir, "text": out, "parsed": F.parse_generated(out)})
    return pd.DataFrame(rec), pd.DataFrame(gen)


CFG = {"qwen3-instruct": dict(prior=3.0, card_prior_factor=0.6, pull=0.0, seed="q3", signal=5.0),  # card weakens a 'No' prior
       "falcon3-7b": dict(prior=0.1, pull=-0.5, seed="f3"),                                 # updating toward 'Yes'
       "granite3-instruct": dict(prior=0.2, pull=0.5, seed="g3", dose_power=0.0),          # toward 'No'; ignores sizes
       "yi15-9b": dict(prior=0.3, pull=-0.4, seed="y1", card_prior_factor=1.0)}
CELLS = {}
t0 = time.time()
for tag, cfg in CFG.items():
    for ds, bb in B.items():
        truth = {"\n" + bb._block(bb.values[r]): bb.y[r] for r in bb.ids}
        be = F.MockV5(alarm_gain=1.0, safe_gain=1.0, scale=1.0, truth=truth, **({"signal": 1.5} | cfg))
        rec, gen = run_mock(be, bb, ds, sorted(bb.ids)[:NPD])
        CELLS[f"{tag}|{ds}"] = F.analyse_cell_v5(rec, gen, n_boot=300)
print(f"fake runs and analysis: {time.time() - t0:.0f} s")
q = CELLS["qwen3-instruct|physionet2012"]
lc = q["lean_change"]
chk(f"card that weakens a prior of 3 by 40%: lean -3 without a card, -1.8 with the placebo, change +1.2 "
    f"(got {lc['lean_c0']:.2f}, {lc['lean_cp']:.2f}, {lc['delta']:+.2f})",
    abs(lc["lean_c0"] + 3) < 0.15 and abs(lc["lean_cp"] + 1.8) < 0.15 and abs(lc["delta"] - 1.2) < 0.15)
w0 = q["pairs"]["yn"]["C0"]["neutral"]["W"]
wp = q["pairs"]["yn"]["CP"]["neutral"]["W"]
chk(f"... W against no card at the neutral point about +2.4 (got {w0['ap']:+.2f}), against the placebo about 0 "
    f"(got {wp['ap']:+.2f}, not conclusive)", abs(w0["ap"] - 2.4) < 0.4 and abs(wp["ap"]) < 0.3 and not wp["conclusive"])
f = CELLS["falcon3-7b|physionet2012"]
for base in ("C0", "CP"):
    w = f["pairs"]["yn"][base]["neutral"]["W"]
    chk(f"updating toward 'Yes' (pull -0.5): W about +1 against {base} (got {w['ap']:+.2f}), conclusive",
        abs(w["ap"] - 1.0) < 0.3 and w["sign"] == 1 and w["conclusive"])
chk("... the decision-level effect of generated answers has the sign of W",
    f["generated"]["E_W"] > 0 and f["generated"]["parsed_share"] > 0.8)
gq = CELLS["granite3-instruct|physionet2012"]
chk("updating toward 'No' (pull +0.5): W about -1 against the placebo",
    abs(gq["pairs"]["yn"]["CP"]["neutral"]["W"]["ap"] + 1.0) < 0.3)
dm = CELLS["falcon3-7b|physionet2012"]["dose"]
dg = gq["dose"]
chk(f"dose: a model that reads the sizes moves more with x2 than with x0.5 (dm {dm['dm']['est']:+.2f}); one that "
    f"ignores them does not (dm {dg['dm']['est']:+.2f})", dm["dm"]["est"] > 0.5 and abs(dg["dm"]["est"]) < 0.1)
pr = f["prob"]["CP"]["neutral"]["W"]
chk(f"probability evidence is read: W about +1 for the 'Yes' model (got {pr['ap']:+.2f})", abs(pr["ap"] - 1.0) < 0.4)
chk("letters carry no polarity pull in the fake models: W with A/B about 0",
    abs(f["pairs"]["ab"]["CP"]["neutral"]["W"]["ap"]) < 0.3)
chk("support and delta* are reported", all(k in q["pairs"]["yn"]["C0"] for k in ("support", "readable"))
    and q["pairs"]["yn"]["C0"]["neutral"]["delta_star"] is not None)
# the fake PhysioNet 2019 sample is small (62 patients), so the decision rules are checked on PhysioNet 2012 analyses
CELLS_R = dict(CELLS)
for tag in ("falcon3-7b", "yi15-9b", "qwen3-instruct"):
    CELLS_R[f"{tag}|physionet2019"] = CELLS[f"{tag}|physionet2012"]
conf = F.confirmatory_v5(CELLS_R)
chk(f"R15 supported for the fake Qwen3 (both datasets): {conf['R15']['supported']}", conf["R15"]["supported"])
chk(f"R16: 4 of 4 fake cells supported ({conf['R16']['n_supported']})", conf["R16"]["supported"] and conf["R16"]["n_supported"] == 4)
chk(f"R17: generated answers in 4 of 4 fake cells ({conf['R17']['n_supported']})", conf["R17"]["supported"])
CELLS2 = dict(CELLS_R)
CELLS2["granite3-instruct|physionet2012"] = CELLS["falcon3-7b|physionet2012"]
conf2 = F.confirmatory_v5(CELLS2)
chk("R16 fails when a cell is conclusive in the opposite direction", not conf2["R16"]["supported"]
    and conf2["R16"]["cells"]["granite3-instruct|physionet2012"]["opposite"])

# ---------------------------------------------------------------- engineering-check rules
fake = lambda yn=True, others=True, spp=0.05, gpp=1.0: {
    "grids_ok": {"yn": yn, "pa": True, "tf": others, "ab": others, "dose": True, "prob": True},
    "single_token_words": {"Yes/No": True}, "readings": {"timing|yn|alarm|A": {"seconds_per_prompt": spp}},
    "generated_free": {"seconds_per_prompt": gpp}}
res = {m: fake(gpp=1.5) for m in F.MODELS_V5}
res["Qwen/Qwen3-8B"] = fake(others=False, gpp=1.5)
res["ibm-granite/granite-3.3-8b-instruct"] = fake(yn=False, gpp=1.5)
d = F.decisions_from_check(res, 1100, 600)
chk("rules (i) and (ii): a model without Yes/No is not run; failed grids are dropped",
    d["not_run"] == ["ibm-granite/granite-3.3-8b-instruct"] and d["grids_not_run"] == {"qwen3-instruct": ["tf", "ab"]})
chk(f"rule (iii): 6 models at 1.5 s per generated answer exceed 8 hours ({d['projected_generation_h_at_150']} h) -> 100 per class",
    d["gen_per_class"] == 100)
d2 = F.decisions_from_check({m: fake(gpp=0.3) for m in F.MODELS_V5}, 1100, 600)
chk("rule (iii): at 0.3 s per answer the subsample stays at 150 per class", d2["gen_per_class"] == 150)
bad = [n for n, c in ok if not c]
print(f"\n{len(ok) - len(bad)} of {len(ok)} checks passed")
for n in bad:
    print("FAILED:", n)
sys.exit(1 if bad else 0)
