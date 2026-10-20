"""Main module vs dev module: same data tables, same dev fingerprints, same dev prompts; test sample sane;
the D1 estimator equals the one in the committed simulation script."""
import sys, os, importlib.util, json
import numpy as np
src_dev, src_main, data_dir, sim_path = sys.argv[1:5]
os.environ.setdefault("LER_P19_MIN_PER_SET", "1000"); os.environ.setdefault("LER_P12_MIN_PER_SET", "4000")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
D = load("ler_dev", src_dev); M = load("ler_main", src_main); SIM = load("simasym", sim_path)
pilot = [str(i) for i in D.PILOT_IDS_FALLBACK]
for ds in D.DATASETS:
    kw = dict(workers=1) if ds == "physionet2019" else {}
    t1 = D.LOADERS[ds](data_dir, (), log=lambda *a: None, **kw)
    t2 = M.LOADERS[ds](data_dir, (), log=lambda *a: None, **kw)
    assert t1[0].equals(t2[0]) and t1[1] == t2[1] and t1[2] == t2[2], ds
    assert os.path.exists(t2[3]["memcheck_file"]), t2[3]["memcheck_file"]
    pid = pilot if ds == "physionet2012" else ()
    bd = D.Bundle(ds, t1[0], t1[1], t1[2], t1[3], pilot_ids=pid, log=lambda *a: None)
    bm = M.Bundle(ds, t2[0], t2[1], t2[2], t2[3], pilot_ids=pid, log=print, sample="test")
    fm = {k: v for k, v in bm.info["fingerprint"].items() if k not in ("test_ids", "test_cards", "test_partners")}
    assert fm == bd.info["fingerprint"], (ds, fm, bd.info["fingerprint"])
    assert bm.dev_ids == bd.ids
    # dev prompts identical when the main bundle is switched back to the dev sample
    bm2 = M.Bundle(ds, t2[0], t2[1], t2[2], t2[3], pilot_ids=pid, log=lambda *a: None, sample="dev")
    for rid in bd.ids[:20]:
        for cond in M.CONDITIONS:
            for fr in M.FRAMINGS:
                assert bd.prompt(rid, cond, fr, "A") == bm2.prompt(rid, cond, fr, "A")
    # test sample
    k = bm.info["test_sample"]["per_class"]
    ys = [bm.y[r] for r in bm.ids]
    assert sum(ys) == k and len(ys) == 2 * k, (ds, sum(ys), len(ys))
    assert not set(bm.ids) & set(bm.dev_ids)
    assert set(bm.ids) <= set(bm.df["id"].iloc[bm.test_pool])
    assert not set(bm.ids) & set(pilot)
    assert all(bm.card_dir[bm.partner[r]] != bm.card_dir[r] for r in bm.ids if bm.partner[r])
    elig = dict(zip(bm.df["id"], bm.eligible))
    assert all(elig[r] for r in bm.ids)
    print(f"{ds}: dev fingerprints identical; test sample {len(bm.ids)} ({k}/class, fallback "
          f"{bm.info['test_sample']['fallback_rule_applied']}); cards {json.dumps(bm.info['test_cards'])}")
# estimator identical to the committed simulation
rng = np.random.default_rng(5)
for _ in range(50):
    n = 300; y = (np.arange(n) % 2).astype(float); z = y + rng.normal(0, 1, n)
    l0 = 3 * z + rng.normal(0, 6, n); la = l0 + rng.normal(2, 1, n); ls = l0 - rng.normal(1, 1, n)
    t = np.where(rng.random(n) < 0.3, 1.0, -1.0)
    assert M.adjusted_asymmetry(l0, la, ls, t, z, y) == SIM.adjusted_asymmetry(l0, la, ls, t, z, y)
assert np.isnan(M.adjusted_asymmetry(l0, la, ls, np.ones(n), z, y))      # all true cards one way -> not estimable
assert np.isnan(M.adjusted_asymmetry(l0, la, ls, t, np.ones(n), y))      # constant XGBoost log-odds -> not estimable
print("adjusted_asymmetry identical to the simulation script; rank-deficient case -> nan")
