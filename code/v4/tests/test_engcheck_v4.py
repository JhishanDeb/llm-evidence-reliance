"""CPU test of engineering_check_v4() and smoke_v4() with a stand-in for vLLM (fakevllm/) and for torch (faketorch/),
a tiny tokenizer with single tokens for Yes, No, True and False, and dev prompts built from the fake data. It checks
that every registered quantity is recorded for every grid and framing, for an instruction-tuned and a base model (with
the chat format and its token-identity check), and that the decisions follow the rules of Section 10; the fake model's
numbers mean nothing.
usage: PYTHONPATH=tests/fakevllm:tests/faketorch python test_engcheck_v4.py CODE_DIR FAKE_DATA_DIR WORK_DIR"""
import contextlib
import io
import json
import os
import sys

CODE, DATA, WORK = sys.argv[1:4]
sys.path.insert(0, CODE)
import ler_main as L  # noqa: E402
import ler_v4 as W  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

os.makedirs(WORK, exist_ok=True)
tiny = os.path.join(WORK, "tiny")
corpus = ["Answer: Yes Answer: No Answer: True Answer: False Yes No True False",
          "Answer with Yes or No. Answer with True or False."] * 80
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=420, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
tok.chat_template = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n'}}"
                     "{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}")
tok.save_pretrained(tiny)
json.dump({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "torch_dtype": "bfloat16",
           "max_position_embeddings": 8192}, open(os.path.join(tiny, "config.json"), "w"))
L.download_model = lambda model_id, cache_dir, revision=None, token=None: (tiny, "fake-revision")
W.download_tokenizer = lambda model_id, cache_dir, revision=None, token=None: (tiny, "fake-chat-revision")

items, smoke = [], {"yesno": [], "tf": []}
for ds in L.DATASETS:
    if ds == "physionet2019":
        df, f, c, i = L.load_physionet2019(DATA, (), log=lambda *a: None, workers=4)
    else:
        df, f, c, i = L.LOADERS[ds](DATA, (), log=lambda *a: None)
    b = L.Bundle(ds, df, f, c, i, sample="dev", log=lambda *a: None)
    for rid in b.ids[:2]:
        for g in W.INSTRUCT_GRIDS + ("plain", "chat"):
            for cond, fr, shot in W.GRID_DEF[g]:
                text = W.prompt_v4(b, rid, cond, fr, shot)[0]
                items.append({"set": "check", "grid": g, "framing": fr, "shot": shot,
                              "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": text})
        for cond in W.SPLIT_CONDS:
            smoke["yesno"].append(L.render(b.prompt(rid, cond, "alarm", "A")[0], "instruct", tok))
            smoke["tf"].append(L.render(W.prompt_v4(b, rid, cond, "tf_alarm", "A")[0], "instruct", tok))
    for rid in b.ids[:5]:
        items.append({"set": "timing", "grid": "split", "framing": "alarm", "shot": "A", "key": f"{ds}|{rid}|C2|alarm|A",
                      "text": b.prompt(rid, "C2", "alarm", "A")[0]})
    for rid in b.ids[:1]:
        for cond, fr, shot in W.GREEDY_GRID:
            items.append({"set": "greedy", "grid": "split", "framing": fr, "shot": shot,
                          "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": b.prompt(rid, cond, fr, shot)[0]})
pp = os.path.join(WORK, "dev_prompts.json")
json.dump(items, open(pp, "w"))
sp = os.path.join(WORK, "smoke_prompts.json")
json.dump(smoke, open(sp, "w"))

ok = []
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = W.engineering_check_v4("meta-llama/Llama-3.1-8B-Instruct", "instruct", pp, os.path.join(WORK, "cache"))
parsed = W.parse_result(buf.getvalue(), "ENGCHECK_V4_RESULT")
ok.append(("result line printed and parsed", parsed is not None and parsed["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"))
ok.append(("no error in the check", not any(p.startswith("error") for p in r["problems"]) and "traceback" not in r))
ok.append(("model window from config (8192)", r.get("max_model_len") == 8192))
want = {f"check|{g}|{fr}|{s}" for g in W.INSTRUCT_GRIDS + ("plain",) for _, fr, s in W.GRID_DEF[g]}
ok.append(("readings for every grid, framing and set, and the timing set",
           want <= set(r["readings"]) and "timing|split|alarm|A" in r["readings"]
           and not any(k.startswith("check|chat|") for k in r["readings"])))
ok.append(("single-token words and both exact-reading checks recorded",
           r["single_token_words"] == {"Yes/No": True, "True/False": True}
           and "exact_check_Yes/No" in r and "exact_check_True/False" in r))
ok.append(("six obvious questions (four Yes/No, two True/False)", set(r["sanity_right_word_more_likely"]) ==
           {"sky_yes", "fire_no", "sky_true", "fire_false"}))
ok.append(("generated answers in both modes, with time and parse share",
           all(r[f"generated_{m}"]["n"] == 18 and "parsed_share" in r[f"generated_{m}"] for m in ("prefill", "free"))))
ok.append(("formats_ok for every grid of the run and the plain format", set(r["formats_ok"]) == set(W.INSTRUCT_GRIDS + ("plain",))))
ok.append(("ok == no problems", r["ok"] == (not r["problems"])))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rb = W.engineering_check_v4("Qwen/Qwen2.5-7B", "base", pp, os.path.join(WORK, "cache"), None,
                                "Qwen/Qwen2.5-7B-Instruct", None)
ok.append(("base model: readings in plain and chat format only", {k.split("|")[1] for k in rb["readings"] if k.startswith("check")}
           == {"split", "chat"} and "generated_free" not in rb))
ok.append(("base model: chat tokenizer recorded and token identity checked",
           rb.get("chat_model") == "Qwen/Qwen2.5-7B-Instruct" and rb["chat_token_identity"]["identical"]
           and rb["chat_token_identity"]["n"] == 36))
ok.append(("base model: r10_eligible from mass, finiteness and token identity", rb["r10_eligible"] ==
           bool(rb["formats_ok"]["split"] and rb["formats_ok"]["chat"])))
tail = rb.get("rendered_tail_chat_alarm", "")
ok.append(("base model chat format uses the chat template", "<|im_start|>assistant\nAnswer:" in tail))
res = {m: (r if c["kind"] == "instruct" else rb) for m, c in W.MODELS_V4.items()}
dec = W.decisions_from_check(res)
ok.append(("decisions computed from the results", set(dec) == {"formats_not_run", "plain_run", "r10_families"}
           and len(dec["r10_families"]) <= 2))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    W.smoke_v4(tiny, "instruct", sp)
s = W.parse_result(buf.getvalue(), "SMOKE_V4_RESULT")
ok.append(("smoke test with both exact-reading checks, window and read mode",
           s is not None and s["max_model_len"] == 8192 and s["read"] in ("token_ids", "topk")
           and s["exact_check"]["n_prompts"] == 1 + len(smoke["yesno"]) and s["exact_check_tf"]["n_prompts"] == len(smoke["tf"])))
for name, x in ok:
    print(("PASS " if x else "FAIL ") + name)
print("problems listed by the fake check (expected with a fake model):", r["problems"][:4], rb["problems"][:2])
print("decisions with the fake model:", dec)
sys.exit(0 if all(x for _, x in ok) else 1)
