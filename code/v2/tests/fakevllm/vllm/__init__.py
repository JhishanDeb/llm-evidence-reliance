"""Minimal stand-in for vllm (CPU tests of VLLMBackend's reading logic). The 'model' gives each prompt a
deterministic next-token distribution over a vocabulary of size V; output containers mimic vLLM 0.26."""
import hashlib
import math
from dataclasses import dataclass

__version__ = "0.26.0-fake"
V = 412
MODE = {"broken_token_ids": False}


@dataclass
class Logprob:
    logprob: float
    rank: int = None
    decoded_token: str = None


class SamplingParams:
    def __init__(self, max_tokens=16, temperature=1.0, logprobs=None, logprob_token_ids=None):
        self.max_tokens, self.temperature = max_tokens, temperature
        self.logprobs, self.logprob_token_ids = logprobs, logprob_token_ids


def _dist(ids):
    h = hashlib.sha256(str(list(ids)).encode()).digest()
    logits = [((h[i % 32] * (i + 7)) % 97) / 9.0 for i in range(V)]
    m = max(logits)
    z = sum(math.exp(x - m) for x in logits)
    return [x - m - math.log(z) for x in logits]


class _O:
    def __init__(self, logprobs, text=""):
        self.logprobs, self.text = logprobs, text


class _R:
    def __init__(self, o):
        self.outputs = [o]


class LLM:
    def __init__(self, **kw):
        self.kw = kw

    def generate(self, prompts, sp, use_tqdm=False):
        out = []
        for p in prompts:
            lp = _dist(p["prompt_token_ids"])
            order = sorted(range(V), key=lambda i: -lp[i])
            sampled = order[0]
            if sp.logprob_token_ids is not None:
                ids = list(sp.logprob_token_ids)
                vals = lp
                if MODE["broken_token_ids"]:     # e.g. renormalised over the requested ids only
                    zz = math.log(sum(math.exp(lp[i]) for i in ids))
                    vals = [x - zz for x in lp]
                d = {i: Logprob(vals[i]) for i in ids}
                d[sampled] = Logprob(lp[sampled], 1)
            elif sp.logprobs is not None:
                d = {i: Logprob(lp[i], r + 1) for r, i in enumerate(order[:sp.logprobs])}
            else:
                d = None
            out.append(_R(_O([d] if d is not None else None, text="1,2,3")))
        return out
