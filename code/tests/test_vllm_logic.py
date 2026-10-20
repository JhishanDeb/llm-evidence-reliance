"""VLLMBackend reading logic against the stand-in vllm: exact token-id reading, the D2 check (passes when
values agree, fails when the token-id values are renormalised), and the top-k fallback with imputation."""
import math
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import vllm  # noqa: E402  (the stand-in)
import ler_main as L  # noqa: E402

tiny = sys.argv[3]
be = L.VLLMBackend(tiny, read="token_ids")
prompts = [f"Patient {i}: Answer with Yes or No.\nAnswer:" for i in range(40)]
wanted = sorted({t for v in be.variants.values() for t in v})
ref_of = lambda p: vllm._dist(be.tokenizer.encode(p, add_special_tokens=False))

for p, (d, n) in zip(prompts, be.word_logprobs(prompts)):
    ref = ref_of(p)
    assert n == 0 and all(abs(d[t] - ref[t]) < 1e-12 for t in wanted)
r = be.yes_no(prompts[:1])[0]
ref = ref_of(prompts[0])
assert abs(r["lp_yes"] - math.log(sum(math.exp(ref[t]) for t in be.variants["Yes"]))) < 1e-12
c = be.check_exact(prompts)
print("exact check:", c)
assert c["ok"] and c["n_compared"] > 0
vllm.MODE["broken_token_ids"] = True
c = be.check_exact(prompts)
print("check with renormalised token-id values:", c)
assert not c["ok"]
vllm.MODE["broken_token_ids"] = False

be.read, be.topk = "topk", 30
n_imp = 0
for p, (d, n) in zip(prompts, be.word_logprobs(prompts)):
    ref = ref_of(p)
    floor = sorted(ref, reverse=True)[29]
    n_imp += n
    for t in wanted:
        assert abs(d[t] - (ref[t] if ref[t] >= floor else floor)) < 1e-12
print("top-k fallback: imputed variants", n_imp, "of", 4 * len(prompts))
assert n_imp > 0
print("generate_text:", be.generate_text([[1, 2, 3]], 5))
print("VLLMBackend logic ok")
