"""ler_scale against the stand-in vllm and the tiny test model: the 4-bit backend reads exactly like the registered
VLLMBackend (only engine settings differ), the model check runs end to end and reports failures, the plan check
finds every model id and revision, and the grid is the planned 13 prompts per patient.
usage: python test_scale_logic.py FAKEVLLM_DIR V2_DIR SCALE_DIR TINY_MODEL_DIR PLAN_PATH"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

fake, v2, scale, tiny, plan = sys.argv[1:6]
sys.path[:0] = [fake, v2, scale]
import vllm  # noqa: E402  (the stand-in)
import ler_main as L  # noqa: E402
import ler_scale as S  # noqa: E402

# the registered code
assert hashlib.sha256(open(os.path.join(v2, "ler_main.py"), "rb").read()).hexdigest() == S.REGISTERED_CODE_SHA256
assert L.CODE_VERSION == "main-2.0"

# grid: 13 prompts per patient, as in the plan
g = S.scale_grid()
assert len(g) == 13 and len(set(g)) == 13, g
assert {(c, "alarm", "A") for c in ("C0", "C2", "C4", "C6")} <= set(g)
assert {(c, "reversed", "A") for c in ("C0", "C2", "C4")} <= set(g)
assert {(c, f, "A") for c in ("C0", "C2", "C4") for f in L.CHOICE_FRAMINGS} <= set(g)
assert all(s == "A" for _, _, s in g)
print("grid: 13 prompts per patient as planned")

# plan check
text = open(plan, encoding="utf-8").read()
assert S.check_against_plan(text) == [], S.check_against_plan(text)
rev = S.SCALE_MODELS["Qwen/Qwen3-14B-AWQ"]["revision"]
assert S.check_against_plan(text.replace(rev, "x" * 40)) == [rev]
assert S.check_against_plan(text.replace("## Reading rules", "## Rules")) == ["## Reading rules"]
assert "doesn't" not in text and "isn't" not in text and "can't" not in text and "won't" not in text
print("plan check: every model id, revision, the code hash and the sections found; a missing one is reported")

# the 4-bit backend reads exactly like the registered backend
prompts = [f"Patient {i}: Answer with Yes or No.\nAnswer:" for i in range(40)]
for read in ("token_ids", "topk"):
    b0 = L.VLLMBackend(tiny, read=read, topk=30)
    b1 = S.ScaleVLLMBackend(tiny, read=read, topk=30, gpu_memory_utilization=0.92, enforce_eager=True)
    assert b0.yes_no(prompts) == b1.yes_no(prompts), read
    assert b0.answer_logprobs(prompts, ("A", "B")) == b1.answer_logprobs(prompts, ("A", "B")), read
    for fr in ("alarm", "reversed") + L.CHOICE_FRAMINGS:
        assert b0.read_framing(prompts, fr) == b1.read_framing(prompts, fr), (read, fr)
    k0, k1 = dict(b0.llm.kw), dict(b1.llm.kw)
    assert k1.pop("enforce_eager") is True and k1.pop("gpu_memory_utilization") == 0.92
    k0.pop("gpu_memory_utilization")
    assert k0 == k1, (k0, k1)
    assert b1.info()["engine_settings"]["enforce_eager"] is True
print("ScaleVLLMBackend: readings identical to the registered VLLMBackend; only memory share and eager mode differ")

# model check end to end (tiny random model: the sanity questions may fail, and that must be reported)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(tiny)
yn = [L.render(f"Question: Is patient {i} ill?\n" + L.ANSWER_INSTRUCTION, "instruct", tok) for i in range(6)]
ab = [L.render(f"Question: Which?\nA: x{i}\nB: y\n" + L.CHOICE_INSTRUCTION, "instruct", tok) for i in range(6)]
pp = os.path.join(tempfile.mkdtemp(), "p.json")
json.dump({"yesno": yn, "choice": ab}, open(pp, "w"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = S.model_check(tiny, pp, 0.92, True)
parsed = S.parse_model_check(buf.getvalue())
assert parsed == json.loads(json.dumps(res, default=str)), "result line does not round-trip"
for k in ("exact_check_yesno", "exact_check_choice", "reading_yesno", "reading_choice", "sanity_right_word_more_likely",
          "engine", "variants"):
    assert k in res, (k, res)
assert res["engine"]["gpu_memory_utilization"] == 0.92 and res["engine"]["enforce_eager"] is True
assert res["reading_yesno"]["n"] == 6 and res["reading_choice"]["n"] == 6
assert res["ok"] == (not res["problems"])
san_fail = [k for k, v in res["sanity_right_word_more_likely"].items() if not v]
assert bool(san_fail) == any("sanity" in p for p in res["problems"]), res["problems"]
print(f"model check: ok={res['ok']} problems={res['problems']} | Yes/No dev mass {res['reading_yesno']['mass_median']:.4f}")

# a model whose engine cannot start: the failure is caught and reported, not raised
orig = vllm.LLM.__init__
vllm.LLM.__init__ = lambda self, **kw: (_ for _ in ()).throw(RuntimeError("CUDA out of memory (simulated)"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = S.model_check(tiny, pp, 0.92, True)
vllm.LLM.__init__ = orig
assert res["ok"] is False and "out of memory" in res["problems"][0] and "traceback" in res
assert S.parse_model_check(buf.getvalue())["ok"] is False
assert S.parse_model_check("crashed before printing")["ok"] is False
print("model check: an engine that cannot start is reported as 'cannot be run'")
print("ALL SCALE LOGIC TESTS PASSED")
