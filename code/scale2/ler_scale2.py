"""Exploratory scale check, second plan (SCALE_CHECK_2.md at the tag 'scale-plan-2'): the two-step reading.

Under the first plan (scale-plan), Qwen3-14B-AWQ and Qwen3-32B-AWQ could not be run: after "Answer:" they put most
of their probability on the token " **" (markdown bold) and then write the answer in bold. The two-step reading adds
that path: for each answer word w,

    P(w) = P1(w) + P1(" **") * P2(w)

where P1 is read after "Answer:" and P2 after "Answer: **" (the same prompt with the token " **" added), each summed
over the single-token variants of w ("Yes" and " Yes", ...). Both steps read the top 1,000 log-probabilities, as in
both studies (Deviation D5); a variant outside them gets the last returned log-probability (an upper bound) and is
counted as imputed. Everything else (prompts, the record, every measure) comes from the registered ler_main.py
(main-2.0) and from ler_scale.py (scale-1.0), both unchanged.
"""
import json
import math
import sys
import time

import ler_main as L
import ler_scale as S1

SCALE2_VERSION = "scale-2.0"
PLAN2_TAG = "scale-plan-2"
PLAN2_URL = f"{L._RAW}/{PLAN2_TAG}/SCALE_CHECK_2.md"
PLAN2_MARKERS = ["## The two-step reading", "## Models and order", "## Measures", "## Reading rules",
                 "P1(word) + P1(\" **\") × P2(word)"]
BOLD = " **"
# runs under this plan, in this order; the 4B model is not run again (see SCALE_CHECK_2.md)
SCALE2_MODELS = {mid: dict(S1.SCALE_MODELS[mid], memcheck=mc) for mid, mc in
                 (("Qwen/Qwen3-8B-AWQ", False), ("Qwen/Qwen3-14B-AWQ", True), ("Qwen/Qwen3-32B-AWQ", True))}
REUSED_FROM_PLAN1 = {"qwen3-4b-awq": "every reading of the first-plan run has answer-word mass >= 0.99"}
BRIDGE = "qwen3-8b-awq"


def plan2_strings() -> list:
    out = list(PLAN2_MARKERS) + [S1.REGISTERED_CODE_SHA256, S1.PLAN_TAG]
    for mid, m in SCALE2_MODELS.items():
        out += [mid, m["revision"]]
    return out


def check_against_plan2(text: str) -> list:
    return [s for s in plan2_strings() if s not in text]


def bold_token_id(tokenizer) -> int:
    """The single token ' **' (byte-level 'Ġ**'); stops if the tokenizer has none."""
    tid = tokenizer.get_vocab().get("Ġ**")
    if tid is None or tokenizer.decode([tid]) != BOLD:
        ids = tokenizer.encode(BOLD, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != BOLD:
            raise RuntimeError(f"' **' is not a single token in this tokenizer ({ids})")
        tid = ids[0]
    return int(tid)


class TwoStepVLLMBackend(S1.ScaleVLLMBackend):
    """ScaleVLLMBackend (engine settings per model) with the two-step reading. read_framing returns, per prompt,
    lp_yes / lp_no (log P(w) of the two answer words of the framing), n_imputed, p_bold = P1(' **') and the
    first-step answer-word mass. The two steps of a prompt are sent one after the other in the same batch, so the
    second can reuse the cached prefix of the first."""
    name = "vllm"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.read = "topk"
        self.bold_id = bold_token_id(self.tokenizer)

    def _raw_ids(self, id_lists):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.topk)
        outs = self.llm.generate([{"prompt_token_ids": list(x)} for x in id_lists], sp, use_tqdm=False)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            if not hasattr(lp, "items"):
                raise RuntimeError(f"unexpected vLLM logprobs container {type(lp)}; unset VLLM_FLAT_LOGPROBS")
            res.append({int(t): float(getattr(v, "logprob", v)) for t, v in lp.items()})
        return res

    @staticmethod
    def _pick(vals, ids):
        """log-probabilities of `ids` from one top-k reading, imputing the last returned value for absent ids."""
        finite = [v for v in vals.values() if math.isfinite(v)]
        if len(finite) < len(vals) or not finite:
            return None, 0
        floor = min(finite)
        return {t: vals.get(t, floor) for t in ids}, sum(t not in vals for t in ids)

    def two_step(self, prompts, pair):
        self.ensure_words(pair)
        pos, neg = self.variants[pair[0]], self.variants[pair[1]]
        ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        reqs = []
        for x in ids:
            reqs += [x, x + [self.bold_id]]
        raw = self._raw_ids(reqs)
        out = []
        for i in range(len(prompts)):
            r1, r2 = raw[2 * i], raw[2 * i + 1]
            d1, n1 = self._pick(r1, pos + neg + [self.bold_id])
            d2, n2 = self._pick(r2, pos + neg)
            if d1 is None or d2 is None:
                out.append({"lp_pos": float("nan"), "lp_neg": float("nan"), "n_imputed": 0,
                            "p_bold": float("nan"), "mass_first": float("nan")})
                continue
            e = math.exp
            pb = e(d1[self.bold_id])
            p1p, p1n = sum(e(d1[t]) for t in pos), sum(e(d1[t]) for t in neg)
            p2p, p2n = sum(e(d2[t]) for t in pos), sum(e(d2[t]) for t in neg)
            pp, pn = p1p + pb * p2p, p1n + pb * p2n
            out.append({"lp_pos": math.log(pp) if pp > 0 else -math.inf,
                        "lp_neg": math.log(pn) if pn > 0 else -math.inf,
                        "n_imputed": int(n1 + n2), "p_bold": pb, "mass_first": p1p + p1n})
        return out

    def read_framing(self, prompts, framing):
        return [{"lp_yes": r["lp_pos"], "lp_no": r["lp_neg"], "n_imputed": r["n_imputed"], "p_bold": r["p_bold"],
                 "mass_first": r["mass_first"]} for r in self.two_step(prompts, L.answer_pair(framing))]

    def info(self):
        d = super().info()
        d.update(reading="two-step (scale-2.0)", bold_token=[self.bold_id, self.tokenizer.decode([self.bold_id])])
        return d


def _mass2(rows):
    m = sorted(math.exp(r["lp_pos"]) + math.exp(r["lp_neg"]) for r in rows
               if math.isfinite(r["lp_pos"]) and math.isfinite(r["lp_neg"]))
    b = sorted(r["p_bold"] for r in rows if math.isfinite(r["p_bold"]))
    f = sorted(r["mass_first"] for r in rows if math.isfinite(r["mass_first"]))
    med = lambda v: v[len(v) // 2] if v else float("nan")
    return {"n": len(rows), "finite_share": len(m) / len(rows) if rows else float("nan"), "mass_median": med(m),
            "mass_min": m[0] if m else float("nan"), "p_bold_median": med(b), "mass_first_median": med(f),
            "imputed_share": sum(r["n_imputed"] > 0 for r in rows) / len(rows) if rows else float("nan")}


def model_check2(model_path, prompts_path, gpu_memory_utilization=0.80, enforce_eager=False):
    """The model check of the first plan with the two-step reading: the model loads, the four sanity questions get
    the right word, and the dev Yes/No readings are finite with a median two-step answer-word mass of at least 0.5.
    Dev prompts only. Prints one SCALE2_MODELCHECK_RESULT line."""
    res = {"ok": False, "problems": []}
    t0 = time.time()
    try:
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        be = TwoStepVLLMBackend(model_path, tensor_parallel_size=tp, gpu_memory_utilization=gpu_memory_utilization,
                                enforce_eager=enforce_eager)
        res.update(load_seconds=round(time.time() - t0, 1), engine=be.engine_settings,
                   bold_token=[be.bold_id, be.tokenizer.decode([be.bold_id])])
        data = json.load(open(prompts_path))
        be.ensure_words(("A", "B"))
        res["variants"] = {w: [[i, be.strings[i]] for i in ids] for w, ids in be.variants.items()}
        for fmt, pair in (("yesno", ("Yes", "No")), ("choice", ("A", "B"))):
            if data.get(fmt):
                t1 = time.time()
                rows = be.two_step(data[fmt], pair)
                el = time.time() - t1
                res[f"reading_{fmt}"] = dict(_mass2(rows), seconds=round(el, 1),
                                             seconds_per_prompt=round(el / max(1, len(rows)), 4))
        san = {}
        for name, (text, pair, first) in L.SANITY_PROMPTS.items():
            r = be.two_step([L.render(text, S1.KIND, be.tokenizer)], pair)[0]
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
            res["problems"].append(f"median two-step Yes+No mass on the dev prompts {yn['mass_median']:.3f} < "
                                   f"{L.VALIDITY_MIN_MASS}")
        res["ok"] = not res["problems"]
    except Exception as e:
        import traceback
        res["problems"].append(f"error: {e!r}"[:500])
        res["traceback"] = traceback.format_exc()[-3000:]
    res["total_seconds"] = round(time.time() - t0, 1)
    print("SCALE2_MODELCHECK_RESULT " + json.dumps(res, default=str))
    return res


def parse_model_check2(output: str) -> dict:
    key = "SCALE2_MODELCHECK_RESULT "
    for ln in reversed(output.splitlines()):
        if ln.startswith(key):
            return json.loads(ln[len(key):])
    return {"ok": False, "problems": ["the model check printed no result (the process failed before the end)"],
            "output_tail": output[-3000:]}


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "check":
        model_check2(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.80,
                     len(sys.argv) > 5 and sys.argv[5] == "1")
