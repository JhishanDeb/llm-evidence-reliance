"""Exploratory scale check (SCALE_CHECK.md at the tag 'scale-plan'): Qwen3 from 4B to 32B, 4-bit AWQ checkpoints.

Prompts, the reading and every measure come from the registered code of the second study (ler_main.py, main-2.0),
which is not changed. This module adds only what the scale check needs on top of it: the model list, the prompt
grid, the check against the plan, a vLLM backend whose memory settings can be set per model, and the check of a
model before its run (load, sanity questions, answer-word mass, time per prompt).
"""
import json
import math
import os
import sys
import time

import numpy as np

import ler_main as L

SCALE_VERSION = "scale-1.0"
REGISTERED_CODE_SHA256 = "0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e"   # ler_main.py main-2.0
PLAN_TAG = "scale-plan"
PLAN_URL = f"{L._RAW}/{PLAN_TAG}/SCALE_CHECK.md"
PLAN_MARKERS = ["## Models", "## Prompts and reading", "## Measures", "## Reading rules", "Each patient gets 13 prompts"]
KIND = "instruct"

# id -> short tag, size in billions of parameters, pinned revision, vLLM memory settings, order of the runs
SCALE_MODELS = {
    "Qwen/Qwen3-8B-AWQ": dict(tag="qwen3-8b-awq", size_b=8, revision="4da05a8edb55c6046cce958586c33b61da07bb79",
                              gpu_memory_utilization=0.80, enforce_eager=False, order=1),
    "Qwen/Qwen3-14B-AWQ": dict(tag="qwen3-14b-awq", size_b=14, revision="31c69efc29464b6bb0aee1398b5a7b50a99340c3",
                               gpu_memory_utilization=0.80, enforce_eager=False, order=2),
    "Qwen/Qwen3-32B-AWQ": dict(tag="qwen3-32b-awq", size_b=32, revision="0499c3ac83fdef8810b907a23894ba91e95eddd8",
                               gpu_memory_utilization=0.92, enforce_eager=True, order=3),
    "Qwen/Qwen3-4B-AWQ": dict(tag="qwen3-4b-awq", size_b=4, revision="74d4bd2bd4bff9cafc9345221320bffb08b406a3",
                              gpu_memory_utilization=0.80, enforce_eager=False, order=4),
}
# the second-study run whose records are reused as the float16 reference
REFERENCE = {"model_id": "Qwen/Qwen3-8B", "tag": "qwen3-instruct", "size_b": 8,
             "revision": "b968826d9c46dd6066d109eabc6255188de91218"}
QUANT_CONTROL = "qwen3-8b-awq"


def scale_grid() -> list:
    """(condition, framing, example set) of a scale-check run: 13 prompts per patient, all with example set A.
    Yes/No alarm framing C0, C2, C4, C6 (the slice of the registered validity rule and of a, a'); Yes/No reversed
    framing C0, C2, C4 (Q3); answer options C0, C2, C4 x both option orders (as in the second study)."""
    out = [(c, "alarm", "A") for c in L.CONDITIONS]
    out += [(c, "reversed", "A") for c in ("C0", "C2", "C4")]
    out += [(c, f, "A") for c in L.CHOICE_GRID["conditions"] for f in L.CHOICE_FRAMINGS]
    return out


def plan_strings() -> list:
    out = list(PLAN_MARKERS) + [REGISTERED_CODE_SHA256, REFERENCE["model_id"], REFERENCE["revision"]]
    for mid, m in SCALE_MODELS.items():
        out += [mid, m["revision"]]
    return out


def check_against_plan(text: str) -> list:
    """Model ids, revisions, the code hash and section markers missing from SCALE_CHECK.md (should be empty)."""
    return [s for s in plan_strings() if s not in text]


class ScaleVLLMBackend(L.VLLMBackend):
    """The registered VLLMBackend with two engine settings that can be set per model (the share of GPU memory vLLM
    may use and eager mode, which saves the memory of CUDA graphs). The reading (word_logprobs, check_exact, the
    top-1,000 imputation) is inherited unchanged. The weights are 4-bit AWQ; vLLM reads that from the model config."""

    def __init__(self, model_path, tensor_parallel_size=1, max_model_len=L.MAX_MODEL_LEN, seed=L.SPLIT_SEED,
                 read="token_ids", topk=L.TOPK_FALLBACK, gpu_memory_utilization=0.80, max_num_batched_tokens=4096,
                 enforce_eager=False):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.variants, self.strings = L.find_word_variants(self.tokenizer)
        self.read, self.topk, self.max_model_len = read, topk, max_model_len
        kw = dict(model=model_path, tokenizer=model_path, tensor_parallel_size=tensor_parallel_size, dtype="float16",
                  max_model_len=max_model_len, seed=seed, gpu_memory_utilization=gpu_memory_utilization,
                  max_num_batched_tokens=max_num_batched_tokens, disable_custom_all_reduce=True,
                  max_logprobs=max(20, topk), generation_config="vllm", enforce_eager=bool(enforce_eager))
        self.engine_settings = {k: kw[k] for k in ("tensor_parallel_size", "dtype", "max_model_len",
                                                   "gpu_memory_utilization", "max_num_batched_tokens", "enforce_eager")}
        self.llm = LLM(**kw)

    def info(self):
        d = super().info()
        d["engine_settings"] = self.engine_settings
        return d


def model_check(model_path, prompts_path, gpu_memory_utilization=0.80, enforce_eager=False):
    """Before a scale-check run, in a subprocess (a failed engine start cannot leave the notebook holding GPU memory).
    Loads the model, runs the exact-reading check (Deviation D2; recorded only, the run reads the top 1,000 as in both
    studies), reads the dev prompts (never test patients) with the top 1,000, and asks the four sanity questions of
    the second study's engineering check. The model can be run if it loads, every sanity question gets the right
    word and the dev Yes/No readings are finite with a median Yes+No mass of at least 0.5. Prints one
    SCALE_MODELCHECK_RESULT line. Computes no accuracy and no asymmetry."""
    res = {"ok": False, "problems": []}
    t0 = time.time()
    try:
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        be = ScaleVLLMBackend(model_path, tensor_parallel_size=tp, read="token_ids",
                              gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager)
        res.update(load_seconds=round(time.time() - t0, 1), engine=be.engine_settings)
        data = json.load(open(prompts_path))
        be.ensure_words(("A", "B"))
        res["variants"] = {w: [[i, be.strings[i]] for i in ids] for w, ids in be.variants.items()}
        for fmt, pair in (("yesno", ("Yes", "No")), ("choice", ("A", "B"))):
            if data.get(fmt):
                res[f"exact_check_{fmt}"] = be.check_exact(data[fmt], pair)
        be.read = "topk"
        for fmt, pair in (("yesno", ("Yes", "No")), ("choice", ("A", "B"))):
            if data.get(fmt):
                t1 = time.time()
                rows = be.answer_logprobs(data[fmt], pair)
                el = time.time() - t1
                res[f"reading_{fmt}"] = dict(L._mass(rows), seconds=round(el, 1),
                                             seconds_per_prompt=round(el / max(1, len(rows)), 4))
        san = {}
        for name, (text, pair, first) in L.SANITY_PROMPTS.items():
            r = be.answer_logprobs([L.render(text, KIND, be.tokenizer)], pair)[0]
            san[name] = bool((r["lp_pos"] > r["lp_neg"]) == first)
        res["sanity_right_word_more_likely"] = san
        failed = [k for k, v in san.items() if not v]
        if failed:
            res["problems"].append(f"sanity questions failed: {failed}")
        yn = res.get("reading_yesno") or {}
        if not yn:
            res["problems"].append("no dev Yes/No prompts were read")
        elif yn["finite_share"] < 1:
            res["problems"].append(f"non-finite Yes/No readings ({yn['finite_share']:.3f} finite)")
        elif not yn["mass_median"] >= L.VALIDITY_MIN_MASS:
            res["problems"].append(f"median Yes+No mass on the dev prompts {yn['mass_median']:.3f} < {L.VALIDITY_MIN_MASS}")
        res["ok"] = not res["problems"]
    except Exception as e:
        import traceback
        res["problems"].append(f"error: {e!r}"[:500])
        res["traceback"] = traceback.format_exc()[-3000:]
    res["total_seconds"] = round(time.time() - t0, 1)
    print("SCALE_MODELCHECK_RESULT " + json.dumps(res, default=str))
    return res


def parse_model_check(output: str) -> dict:
    """The result line of model_check from a subprocess output, or a failure record if there is none."""
    key = "SCALE_MODELCHECK_RESULT "
    for ln in reversed(output.splitlines()):
        if ln.startswith(key):
            return json.loads(ln[len(key):])
    return {"ok": False, "problems": ["the model check printed no result (the process failed before the end)"],
            "output_tail": output[-3000:]}


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "check":
        model_check(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.80,
                    len(sys.argv) > 5 and sys.argv[5] == "1")
