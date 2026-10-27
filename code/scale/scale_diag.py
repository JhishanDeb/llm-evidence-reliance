"""Diagnostic for the scale check (engineering only, dev patients only): what do the Qwen3 4-bit models put their
next-token probability on after 'Answer:'? Qwen3-14B-AWQ and Qwen3-32B-AWQ failed the model check of the plan
(median answer-word mass below 0.5 on the dev prompts); Qwen3-8B-AWQ passed and is the contrast. This changes no rule
of SCALE_CHECK.md and computes no accuracy and no asymmetry.

For each model and answer format: the 20 most likely next tokens of every dev prompt (the ordinary top-20 reading of
vLLM), their mean probability over the prompts, the answer-word mass, and the greedy continuation of a few prompts
(what the model would write after 'Answer:')."""
import json
import math
import sys
import time
from collections import defaultdict

import ler_main as L
import ler_scale as S


def diagnose(model_path, prompts_path, gpu_memory_utilization=0.80, enforce_eager=False, n_generate=8, max_tokens=16):
    res = {"ok": False}
    t0 = time.time()
    try:
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        be = S.ScaleVLLMBackend(model_path, tensor_parallel_size=tp, read="topk",
                                gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager)
        tok = be.tokenizer
        show = lambda t: repr(tok.decode([t]))
        data = json.load(open(prompts_path))
        be.ensure_words(("A", "B"))
        for fmt, pair in (("yesno", ("Yes", "No")), ("choice", ("A", "B")), ("sanity", None)):
            prompts = data.get(fmt) or []
            if not prompts:
                continue
            raw, _ = be._raw(prompts, "top20", be.wanted_ids(pair or ("Yes", "No")))
            mean, per, mass = defaultdict(float), [], []
            ans = {t for w in (pair or ("Yes", "No", "A", "B")) for t in be.variants[w]}
            for d in raw:
                top = sorted(d.items(), key=lambda kv: -kv[1])[:20]
                for t, lp in top:
                    mean[show(t)] += math.exp(lp) / len(raw)
                per.append([[show(t), round(math.exp(lp), 4)] for t, lp in top[:8]])
                mass.append(sum(math.exp(lp) for t, lp in top if t in ans))
            gens = be.generate_text([tok.encode(p, add_special_tokens=False) for p in prompts[:n_generate]], max_tokens)
            mass_sorted = sorted(mass)
            res[fmt] = {"n_prompts": len(prompts),
                        "answer_mass_median": mass_sorted[len(mass) // 2],
                        "top_tokens_mean_probability": [[k, round(v, 4)] for k, v in
                                                         sorted(mean.items(), key=lambda kv: -kv[1])[:25]],
                        "first_prompts_top8": per[:6],
                        "greedy_continuations": gens}
        res["ok"] = True
    except Exception as e:
        import traceback
        res["error"] = repr(e)[:500]
        res["traceback"] = traceback.format_exc()[-3000:]
    res["seconds"] = round(time.time() - t0, 1)
    print("SCALE_DIAG_RESULT " + json.dumps(res, default=str))
    return res


def parse(output: str) -> dict:
    key = "SCALE_DIAG_RESULT "
    for ln in reversed(output.splitlines()):
        if ln.startswith(key):
            return json.loads(ln[len(key):])
    return {"ok": False, "error": "no result line", "output_tail": output[-3000:]}


if __name__ == "__main__":
    diagnose(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1")
