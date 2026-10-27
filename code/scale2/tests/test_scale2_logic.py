"""ler_scale2 against the stand-in vllm and the tiny test model: the two-step reading equals
P1(w) + P1(' **') * P2(w) computed by hand from the stand-in's distributions (exact when every token is returned,
with the D5 imputation when not), the record is built by the registered finish_record, the model check runs end to
end, and the plan check finds every model id and revision.
usage: python test_scale2_logic.py FAKEVLLM_DIR V2_DIR SCALE_DIR SCALE2_DIR TINY_MODEL_DIR PLAN2_PATH"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile

fake, v2, scale, scale2, tiny, plan2 = sys.argv[1:7]
sys.path[:0] = [fake, v2, scale, scale2]
import vllm  # noqa: E402  (the stand-in)
import ler_main as L  # noqa: E402
import ler_scale as S1  # noqa: E402
import ler_scale2 as S2  # noqa: E402

# plan check
text = open(plan2, encoding="utf-8").read()
assert S2.check_against_plan2(text) == [], S2.check_against_plan2(text)
r14 = S2.SCALE2_MODELS["Qwen/Qwen3-14B-AWQ"]["revision"]
assert S2.check_against_plan2(text.replace(r14, "y" * 40)) == [r14]
for bad in ("doesn't", "isn't", "can't", "won't", "didn't", "it's"):
    assert bad not in text, bad
assert list(S2.SCALE2_MODELS) == ["Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ", "Qwen/Qwen3-32B-AWQ"]
assert [m["memcheck"] for m in S2.SCALE2_MODELS.values()] == [False, True, True]
for mid, m in S2.SCALE2_MODELS.items():
    assert m["revision"] == S1.SCALE_MODELS[mid]["revision"] and m["revision"] in text
print("plan 2 check: models, revisions, code hash, sections found; a missing revision is reported")


# bold token: found in a byte-level vocabulary, refused when it is not one token
class Tok:
    def __init__(self, vocab):
        self.v = vocab

    def get_vocab(self):
        return self.v

    def decode(self, ids):
        inv = {i: k.replace("Ġ", " ") for k, i in self.v.items()}
        return "".join(inv[i] for i in ids)

    def encode(self, s, add_special_tokens=False):
        return [self.v["Ġ"], self.v["*"], self.v["*"]] if s == " **" else []


assert S2.bold_token_id(Tok({"Ġ**": 7, "Ġ": 1, "*": 2})) == 7
try:
    S2.bold_token_id(Tok({"Ġ": 1, "*": 2}))
    raise AssertionError("should have refused")
except RuntimeError:
    pass
print("bold token: 'Ġ**' found; a tokenizer without a single ' **' token is refused")

# the two-step reading against hand computation
BOLD_ID = 17
S2.bold_token_id = lambda tok: BOLD_ID          # the tiny tokenizer has no ' **'; any id works with the stand-in
prompts = [f"Patient {i}: Answer with Yes or No.\nAnswer:" for i in range(25)]
e = math.exp
for topk in (1000, 30):
    be = S2.TwoStepVLLMBackend(tiny, topk=topk, gpu_memory_utilization=0.9, enforce_eager=True)
    assert be.read == "topk" and be.bold_id == BOLD_ID
    for pair in (("Yes", "No"), ("A", "B")):
        got = be.two_step(prompts, pair)
        pos, neg = be.variants[pair[0]], be.variants[pair[1]]
        for p, g in zip(prompts, got):
            x = be.tokenizer.encode(p, add_special_tokens=False)
            d1, d2 = vllm._dist(x), vllm._dist(x + [BOLD_ID])
            top1 = sorted(range(vllm.V), key=lambda i: -d1[i])[:topk]
            top2 = sorted(range(vllm.V), key=lambda i: -d2[i])[:topk]
            f1, f2 = min(d1[i] for i in top1), min(d2[i] for i in top2)
            v1 = lambda t: d1[t] if t in top1 else f1
            v2 = lambda t: d2[t] if t in top2 else f2
            pb = e(v1(BOLD_ID))
            pp = sum(e(v1(t)) for t in pos) + pb * sum(e(v2(t)) for t in pos)
            pn = sum(e(v1(t)) for t in neg) + pb * sum(e(v2(t)) for t in neg)
            assert abs(g["lp_pos"] - math.log(pp)) < 1e-9 and abs(g["lp_neg"] - math.log(pn)) < 1e-9, (g, pp, pn)
            assert abs(g["p_bold"] - pb) < 1e-12
            n_imp = sum(t not in top1 for t in pos + neg + [BOLD_ID]) + sum(t not in top2 for t in pos + neg)
            assert g["n_imputed"] == n_imp
            one = be.answer_logprobs([p], pair)[0]          # inherited one-step reading of the same prompt
            assert g["lp_pos"] >= one["lp_pos"] - 1e-12 and g["lp_neg"] >= one["lp_neg"] - 1e-12
    if topk == 1000:
        assert all(g["n_imputed"] == 0 for g in be.two_step(prompts, ("Yes", "No")))
    # read_framing feeds the registered record unchanged
    for fr in ("alarm", "reversed") + L.CHOICE_FRAMINGS:
        rr = be.read_framing(prompts[:3], fr)
        for r in rr:
            rec = L.finish_record({"y": 1}, r, fr)
            assert rec["valid"] == 1 and abs(rec["yn_mass"] - (e(r["lp_yes"]) + e(r["lp_no"]))) < 1e-12
    assert be.info()["reading"].startswith("two-step") and be.llm.kw["enforce_eager"] is True
print("two-step reading: equals P1(w) + P1(' **') * P2(w) by hand, exact and with top-k imputation; never below the "
      "one-step reading; records built by the registered finish_record")

# the second step really is the same prompt plus the bold token, sent right after the first
seen = []
orig = vllm.LLM.generate


def spy(self, reqs, sp, use_tqdm=False):
    seen.extend(r["prompt_token_ids"] for r in reqs)
    return orig(self, reqs, sp, use_tqdm)


vllm.LLM.generate = spy
be.two_step(prompts[:4], ("Yes", "No"))
vllm.LLM.generate = orig
assert len(seen) == 8 and all(seen[2 * i + 1] == seen[2 * i] + [BOLD_ID] for i in range(4))
print("requests: each prompt is followed by the same prompt plus ' **'")

# model check end to end
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(tiny)
yn = [L.render(f"Question: Is patient {i} ill?\n" + L.ANSWER_INSTRUCTION, "instruct", tok) for i in range(6)]
ab = [L.render(f"Question: Which?\nA: x{i}\nB: y\n" + L.CHOICE_INSTRUCTION, "instruct", tok) for i in range(6)]
pp = os.path.join(tempfile.mkdtemp(), "p.json")
json.dump({"yesno": yn, "choice": ab}, open(pp, "w"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = S2.model_check2(tiny, pp, 0.92, True)
parsed = S2.parse_model_check2(buf.getvalue())
assert parsed == json.loads(json.dumps(res, default=str))
for k in ("reading_yesno", "reading_choice", "sanity_right_word_more_likely", "engine", "bold_token"):
    assert k in res, (k, res)
assert {"p_bold_median", "mass_first_median", "mass_median"} <= set(res["reading_yesno"])
assert res["reading_yesno"]["mass_median"] >= res["reading_yesno"]["mass_first_median"]
assert res["ok"] == (not res["problems"])
print(f"model check: ok={res['ok']} problems={res['problems']}")
assert S2.parse_model_check2("crashed")["ok"] is False
print("ALL SCALE-2 LOGIC TESTS PASSED")
