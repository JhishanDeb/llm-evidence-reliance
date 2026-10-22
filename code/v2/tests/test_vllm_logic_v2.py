"""VLLMBackend (main-2.0) reading logic against the stand-in vllm: the Yes/No reading is unchanged from main-1.0
(same ids requested, same values, same imputation count), the A/B reading requests only the A/B ids, the D2 check
runs per answer format, and the engineering check runs end to end on the tiny model.
usage: python test_vllm_logic_v2.py FAKEVLLM_DIR V2_DIR TINY_MODEL_DIR V1_FROZEN_PATH"""
import importlib.util
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import vllm  # noqa: E402  (the stand-in)
import ler_main as L  # noqa: E402

spec = importlib.util.spec_from_file_location("ler_v1", sys.argv[4])
L1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L1)

tiny = sys.argv[3]
prompts = [f"Patient {i}: Answer with Yes or No.\nAnswer:" for i in range(40)]
ref_of = lambda be, p: vllm._dist(be.tokenizer.encode(p, add_special_tokens=False))

for read in ("token_ids", "topk"):
    b1 = L1.VLLMBackend(tiny, read=read, topk=30)
    b2 = L.VLLMBackend(tiny, read=read, topk=30)
    assert b1.yes_no(prompts) == b2.yes_no(prompts), read
    assert [x for x in b1.word_logprobs(prompts)] == [x for x in b2.word_logprobs(prompts)], read
    # reading A/B first must not change a later Yes/No reading (only the requested ids count)
    b2.answer_logprobs(prompts, ("A", "B"))
    assert b1.yes_no(prompts) == b2.yes_no(prompts), read
    print(f"read={read}: Yes/No reading identical to main-1.0, also after an A/B reading")

be = L.VLLMBackend(tiny, read="token_ids")
ab = be.wanted_ids(("A", "B"))
assert ab == sorted(be.variants["A"] + be.variants["B"])
for p, (d, n) in zip(prompts, be.word_logprobs(prompts, words=("A", "B"))):
    ref = ref_of(be, p)
    assert set(d) == set(ab) and n == 0 and all(abs(d[t] - ref[t]) < 1e-12 for t in ab)
r = be.answer_logprobs(prompts[:1], ("A", "B"))[0]
ref = ref_of(be, prompts[0])
assert abs(r["lp_pos"] - math.log(sum(math.exp(ref[t]) for t in be.variants["A"]))) < 1e-12
c = be.check_exact(prompts, ("A", "B"))
assert c["ok"] and c["n_compared"] > 0, c
be.read, be.topk = "topk", 30
n_imp = sum(n for _, n in be.word_logprobs(prompts, words=("A", "B")))
assert n_imp <= len(ab) * len(prompts)
print("A/B reading: exact ids, D2 check per format, top-k imputation counted over A/B ids only")

# engineering check end to end (download is skipped by pointing the cache at the local folder)
items = [{"set": "check", "framing": "alarm", "text": "Question: Is it cold?\nAnswer with Yes or No."},
         {"set": "check", "framing": "choice_alarm_A", "text": "Question: Which?\nA: x\nB: y\nAnswer with A or B."},
         {"set": "timing", "framing": "alarm", "text": "Question: Is it hot?\nAnswer with Yes or No."}]
tmp = tempfile.mkdtemp()
pp = os.path.join(tmp, "p.json")
json.dump(items, open(pp, "w"))
L.download_model = lambda mid, cache, rev=None, tok=None: (tiny, "localsha")
res = L.engineering_check("tiny/test", "instruct", pp, tmp)
assert res["revision"] == "localsha" and "exact_check_yesno" in res and "exact_check_choice" in res, res
assert "reading_check_yesno" in res and "reading_check_choice" in res and "reading_timing_yesno" in res, res
assert set(res["sanity_right_word_more_likely"]) == set(L.SANITY_PROMPTS)
print("engineering check ran end to end; problems found on the random tiny model:", res["problems"])
print("VLLMBackend v2 logic ok")
# eligibility uses only the formats the model will be run with
res2 = L.engineering_check("tiny/test", "instruct", pp, tmp, formats=("yesno",))
assert res2["formats_required"] == ["yesno"] and not any("choice" in p or "color_" in p for p in res2["problems"]), res2["problems"]
assert "reading_check_choice" in res2          # still reported
print("engineering check: formats=('yesno',) ignores A/B problems but still reports them")
