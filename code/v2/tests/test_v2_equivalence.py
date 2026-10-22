"""main-2.0 against main-1.0 (the code of the four first-study runs), on the fake data:
- every first-study prompt is byte-identical, and so are the protocol strings and the analysis constants;
- the mock reading and finish_record of first-study prompts are identical (apart from the new 'answer_words' field);
- the answer-option prompts differ from the alarm prompt only in the Question / Answer lines, the solved examples
  are answered correctly for both option orders, and averaging the two orders removes a letter bias.
usage: python test_v2_equivalence.py ler_main_v1_frozen.py ler_main.py FAKE_DATA_DIR"""
import importlib.util
import math
import os
import sys

v1_path, v2_path, data_dir = sys.argv[1:4]
os.environ.setdefault("LER_P19_MIN_PER_SET", "1000")
os.environ.setdefault("LER_P12_MIN_PER_SET", "4000")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


L1, L2 = load("ler_v1", v1_path), load("ler_v2", v2_path)
assert L2.CODE_VERSION == "main-2.0" and L1.CODE_VERSION == "main-1.0"
assert L1.protocol_strings() == L2.protocol_strings()
for k in ("SPLIT_SEED", "BOOT_SEED", "BOOT_RESAMPLES", "SHOT_SEEDS", "N_FEATURES", "TOP_K", "MIN_MEASURED",
          "TEST_PER_CLASS", "DEV_PER_CLASS", "CONDITIONS", "FRAMINGS", "DATASETS", "MAIN_GRID", "ROBUST_GRID",
          "XGB_PARAMS", "SPEC", "EXAMPLES_HEADER", "PATIENT_HEADER", "ANSWER_INSTRUCTION", "EVIDENCE_HEADER",
          "PREFILL", "DEV_RUN", "LOGPROB_CHECK_TOL", "TOPK_FALLBACK", "MEMCHECK", "REGISTERED_MODELS",
          "VALIDITY_MIN_MASS", "VALIDITY_MIN_C0_ACC", "PROTOCOL_TAG", "DEVIATION_MARKERS"):
    assert getattr(L1, k) == getattr(L2, k), k
v1_grid = [(c, f, "A") for c in L1.CONDITIONS for f in L1.FRAMINGS] + \
          [(c, "alarm", s) for c in L1.ROBUST_GRID["conditions"] for s in L1.ROBUST_GRID["shot_sets"]]
assert L2.grid_v2(("yesno",)) == v1_grid and len(v1_grid) == 14
assert len(L2.grid_v2(("choice",))) == 6 and len(L2.grid_v2(("yesno", "choice"))) == 20

pilot = [str(i) for i in L1.PILOT_IDS_FALLBACK]
quiet = lambda *a: None
n_prompts, n_choice = 0, 0
for ds in L1.DATASETS:
    kw = dict(workers=1) if ds == "physionet2019" else {}
    t = L1.LOADERS[ds](data_dir, (), log=quiet, **kw)
    t2 = L2.LOADERS[ds](data_dir, (), log=quiet, **kw)
    assert t[0].equals(t2[0]) and t[1:3] == t2[1:3], ds
    pid = pilot if ds == "physionet2012" else ()
    b1 = L1.Bundle(ds, *t[:4], pilot_ids=pid, log=quiet, sample="test")
    b2 = L2.Bundle(ds, *t2[:4], pilot_ids=pid, log=quiet, sample="test")
    assert b1.info["fingerprint"] == b2.info["fingerprint"], ds
    opts = L2.CHOICE_OPTIONS[ds]
    for sample in ("dev", "test"):
        b1.use_sample(sample)
        b2.use_sample(sample)
        for rid in b2.ids:
            for cond, fr, s in v1_grid:
                assert b1.prompt(rid, cond, fr, s) == b2.prompt(rid, cond, fr, s), (ds, rid, cond, fr, s)
                n_prompts += 1
            for cond in ("C0", "C2", "C4"):
                ref, cdir = b2.prompt(rid, cond, "alarm", "A")
                for fr in L2.CHOICE_FRAMINGS:
                    text, cd = b2.prompt(rid, cond, fr, "A")
                    n_choice += 1
                    assert cd == cdir
                    strip = lambda x: [ln for ln in x.splitlines()
                                       if not ln.startswith(("Question:", "Answer", "A: ", "B: "))]
                    assert strip(text) == strip(ref), (ds, rid, cond, fr)
                    first = opts["alarm"] if fr == "choice_alarm_A" else opts["safe"]
                    second = opts["safe"] if fr == "choice_alarm_A" else opts["alarm"]
                    assert text.count(f"\nA: {first}\nB: {second}\n") == 5, (ds, fr)
                    assert text.endswith(f"Question: {L2.CHOICE_QUESTION}\nA: {first}\nB: {second}\n"
                                         f"{L2.CHOICE_INSTRUCTION}")
                    assert L2.ANSWER_INSTRUCTION not in text and "Answer: Yes" not in text and "Answer: No" not in text
                    # solved examples: the letter of the true outcome
                    answers = [ln.split(": ", 1)[1] for ln in text.splitlines() if ln.startswith("Answer: ")]
                    truth = [yy for _, yy in b2.shots["A"]]
                    exp = [("A" if (yy == 1) == (fr == "choice_alarm_A") else "B") for yy in truth]
                    assert answers == exp, (answers, exp)
    print(f"{ds}: first-study prompts identical; answer-option prompts checked")
print(f"{n_prompts} first-study prompts identical to main-1.0; {n_choice} answer-option prompts checked")

# reading: mock backend and finish_record
b2.use_sample("test")
rq = [s["q_reversed"] for s in L1.SPEC.values()]
m1 = L1.MockBackend(rq, alarm_gain=1.7, safe_gain=0.5, lean=0.4, scale=2.0, seed="eq")
m2 = L2.MockBackend(rq, alarm_gain=1.7, safe_gain=0.5, lean=0.4, scale=2.0, seed="eq", letter_bias=0.8)
rid = b2.ids[0]
texts = [b2.prompt(r, c, f, s)[0] for r in b2.ids[:15] for c, f, s in v1_grid if b2.prompt(r, c, f, s)[0]]
r1, r2 = m1.yes_no(texts), m2.yes_no(texts)
assert r1 == r2
r3 = m2.read_framing(texts, "alarm")
assert r3 == r2
for (c, f, s), a, b in zip([(c, f, s) for r in b2.ids[:15] for c, f, s in v1_grid if b2.prompt(r, c, f, s)[0]], r1, r2):
    x = L1.finish_record({"y": 1}, a, f)
    y = L2.finish_record({"y": 1}, b, f)
    assert y.pop("answer_words") == "Yes/No"
    assert x.keys() == y.keys() and all((x[k] == y[k]) or (isinstance(x[k], float) and math.isnan(x[k]) and math.isnan(y[k]))
                                        for k in x), (x, y)
print("mock reading and finish_record of first-study prompts identical")
for r in b2.ids[:15]:
    for c in ("C0", "C2", "C4"):
        lo_alarm = L2.finish_record({"y": 1}, m2.read_framing([b2.prompt(r, c, "alarm", "A")[0]], "alarm")[0], "alarm")
        ch = {}
        for fr in L2.CHOICE_FRAMINGS:
            rec = L2.finish_record({"y": 1}, m2.read_framing([b2.prompt(r, c, fr, "A")[0]], fr)[0], fr)
            assert rec["answer_words"] == "A/B"
            ch[fr] = rec["lo_alarm"]
        # the mock's letter bias (+0.8 toward 'A') moves the two orders apart; their mean is the alarm-framing value
        assert abs(ch["choice_alarm_A"] - ch["choice_safe_A"] - 1.6) < 1e-9
        assert abs((ch["choice_alarm_A"] + ch["choice_safe_A"]) / 2 - lo_alarm["lo_alarm"]) < 1e-9
print("answer-option reading: A/B read, lo_alarm by option order, the mean of the two orders removes the letter bias")
print("ALL V2 EQUIVALENCE TESTS PASSED")
