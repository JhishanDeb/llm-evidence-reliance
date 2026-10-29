"""CPU test of engineering_check_v3() and smoke_v3() with a stand-in for vLLM (fakevllm/) and for torch
(faketorch/), a tiny tokenizer that has single tokens for Yes, No, A and B, and dev prompts built from the fake data.
It checks that every registered quantity is recorded for every card format and that eligibility is computed from
the required formats; the fake model's numbers mean nothing.
usage: PYTHONPATH=tests/fakevllm:tests/faketorch python test_engcheck_v3.py CODE_DIR FAKE_DATA_DIR WORK_DIR"""
import contextlib
import io
import json
import os
import sys

CODE, DATA, WORK = sys.argv[1:4]
sys.path.insert(0, CODE)
import ler_main as L  # noqa: E402
import ler_v3 as V  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

os.makedirs(WORK, exist_ok=True)
tiny = os.path.join(WORK, "tiny")
corpus = ["Answer: Yes Answer: No Answer: A Answer: B Yes No A B", "Answer with Yes or No. Answer with A or B."] * 80
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
tok.chat_template = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n'}}"
                     "{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}")
tok.save_pretrained(tiny)
json.dump({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "torch_dtype": "bfloat16",
           "max_position_embeddings": 8192}, open(os.path.join(tiny, "config.json"), "w"))
L.download_model = lambda model_id, cache_dir, revision=None, token=None: (tiny, "fake-revision")

items, smoke = [], {"yesno": [], "choice": []}
for ds in L.DATASETS:
    if ds == "physionet2019":
        df, f, c, i = L.load_physionet2019(DATA, (), log=lambda *a: None, workers=4)
    else:
        df, f, c, i = L.LOADERS[ds](DATA, (), log=lambda *a: None)
    b = L.Bundle(ds, df, f, c, i, sample="dev", log=lambda *a: None)
    for rid in b.ids[:2]:
        for cond, fr, shot in V.grid_v3(("split", "formats", "matched")):
            items.append({"set": "check", "fmt": V.fmt_label(fr, shot), "framing": fr, "key": f"{ds}|{rid}|{cond}|{fr}|{shot}",
                          "text": V.prompt_v3(b, rid, cond, fr, V.SET_FMT[shot])[0]})
        for cond in ("C0", "C2", "C4"):
            for fr in L.CHOICE_FRAMINGS:
                items.append({"set": "check", "fmt": "choice", "framing": fr, "key": f"{ds}|{rid}|{cond}|{fr}|A",
                              "text": b.prompt(rid, cond, fr, "A")[0]})
            smoke["yesno"].append(L.render(b.prompt(rid, cond, "alarm", "A")[0], "instruct", tok))
    for rid in b.ids[:5]:
        items.append({"set": "timing", "fmt": "yesno_full", "framing": "alarm", "key": f"{ds}|{rid}|C2|alarm|A",
                      "text": b.prompt(rid, "C2", "alarm", "A")[0]})
pp = os.path.join(WORK, "dev_prompts.json")
json.dump(items, open(pp, "w"))
sp = os.path.join(WORK, "smoke_prompts.json")
json.dump(smoke, open(sp, "w"))

ok = []
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = V.engineering_check_v3("fake/model", "instruct", pp, os.path.join(WORK, "cache"), None, V.REQUIRED_NEW)
parsed = V.parse_result(buf.getvalue(), "ENGCHECK_V3_RESULT")
ok.append(("result line printed and parsed", parsed is not None and parsed["model_id"] == "fake/model"))
ok.append(("no error in the check", not any(p.startswith("error") for p in r["problems"])))
ok.append(("model window from config (8192)", r.get("max_model_len") == 8192))
ok.append(("readings for every format", all(f"reading_check_{f}" in r for f in V.FMT_LABELS)
           and "reading_timing_yesno_full" in r))
ok.append(("exact-reading checks recorded", "exact_check_yesno_full" in r and "exact_check_choice" in r))
ok.append(("four sanity questions", set(r["sanity_right_word_more_likely"]) == set(L.SANITY_PROMPTS)))
ok.append(("ok == no problems", r["ok"] == (not r["problems"])))
r2 = V.engineering_check_v3("fake/model", "instruct", pp, os.path.join(WORK, "cache"), None, ("yesno_full",))
probs = [p for p in r2["problems"] if "choice" in p or "words" in p or "cue" in p]
ok.append(("only the required formats decide eligibility", not probs))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    V.smoke_v3(tiny, "instruct", sp)
s = V.parse_result(buf.getvalue(), "SMOKE_V3_RESULT")
ok.append(("smoke test result with window and read mode", s is not None and s["max_model_len"] == 8192
           and s["read"] in ("token_ids", "topk")))
for name, x in ok:
    print(("PASS " if x else "FAIL ") + name)
print("problems listed by the fake check (expected with a fake model):", r["problems"][:4])
sys.exit(0 if all(x for _, x in ok) else 1)
