"""CPU test of engineering_check_v5() with a stand-in for vLLM (fakevllm/) and torch (faketorch/), a tiny tokenizer with
single tokens for Yes, No, True, False, A and B, and dev prompts built from the fake data. It checks that every quantity
of Section 10 is recorded for every grid and framing and that the decisions follow the rules; the fake model's numbers
mean nothing.
usage: PYTHONPATH=tests/fakevllm:tests/faketorch python test_engcheck_v5.py CODE_DIR FAKE_DATA_DIR WORK_DIR"""
import contextlib
import io
import json
import os
import sys

CODE, DATA, WORK = sys.argv[1:4]
sys.path.insert(0, CODE)
import ler_main as L  # noqa: E402
import ler_v5 as F  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

os.makedirs(WORK, exist_ok=True)
tiny = os.path.join(WORK, "tiny")
corpus = ["Answer: Yes Answer: No Answer: True Answer: False Answer: A Answer: B Yes No True False A B",
          "Answer with Yes or No. Answer with True or False. Answer with A or B."] * 80
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

items = []
for ds in F.DATASETS_V5:
    if ds == "physionet2019":
        df, f, c, i = L.load_physionet2019(DATA, (), log=lambda *a: None, workers=4)
    else:
        df, f, c, i = L.LOADERS[ds](DATA, (), log=lambda *a: None)
    b = L.Bundle(ds, df, f, c, i, sample="dev", log=lambda *a: None)
    for rid in b.ids[:2]:
        for g in F.GRIDS:
            for cond, fr, shot in F.GRID_DEF[g]:
                text, cdir = F.prompt_v5(b, rid, cond, fr, shot)
                F.check_prompt_v5(b, rid, cond, fr, shot, text, cdir)
                items.append({"set": "check", "grid": g, "framing": fr, "shot": shot, "cond": cond,
                              "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": text})
    for rid in b.ids[:5]:
        items.append({"set": "timing", "grid": "yn", "framing": "alarm", "shot": "A", "cond": "C2",
                      "key": f"{ds}|{rid}|C2|alarm|A", "text": b.prompt(rid, "C2", "alarm", "A")[0]})
    for rid in b.ids[:1]:
        for cond, fr, shot in F.GEN_GRID:
            items.append({"set": "gen", "grid": "gen", "framing": fr, "shot": shot, "cond": cond,
                          "key": f"{ds}|{rid}|{cond}|{fr}|{shot}", "text": F.prompt_v5(b, rid, cond, fr, shot)[0]})
pp = os.path.join(WORK, "dev_prompts.json")
json.dump(items, open(pp, "w"))

ok = []
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = F.engineering_check_v5("Qwen/Qwen3-8B", pp, os.path.join(WORK, "cache"))
parsed = F.parse_result(buf.getvalue(), "ENGCHECK_V5_RESULT")
ok.append(("result line printed and parsed", parsed is not None and parsed["model_id"] == "Qwen/Qwen3-8B"))
ok.append(("no error in the check", not any(p.startswith("error") for p in r["problems"]) and "traceback" not in r))
ok.append(("model window from config (8192)", r.get("max_model_len") == 8192))
want = {f"check|{g}|{fr}|{s}" for g in F.GRIDS for _, fr, s in F.GRID_DEF[g]}
ok.append(("readings for every grid, framing and set, and the timing set",
           want <= set(r["readings"]) and "timing|yn|alarm|A" in r["readings"]))
ok.append(("single-token words and all three exact-reading checks recorded",
           r["single_token_words"] == {"Yes/No": True, "True/False": True, "A/B": True}
           and all(f"exact_check_{p}" in r for p in ("Yes/No", "True/False", "A/B"))))
ok.append(("generated answers timed and parsed", r["generated_free"]["n"] == 12 and "parsed_share" in r["generated_free"]))
ok.append(("a verdict for every grid", set(r["grids_ok"]) == set(F.GRIDS)))
ok.append(("rendered placebo and probability tails recorded", "rendered_tail_CP" in r and "rendered_tail_CQ2" in r))
ph = F.projected_hours(r, 1100, 600)
ok.append(("projected hours computed", ph["scoring_h"] >= 0 and ph["generation_h"] >= 0))
d = F.decisions_from_check({m: r for m in F.MODELS_V5}, 1100, 600)
ok.append(("decisions computed for all models", set(d) >= {"not_run", "grids_not_run", "gen_per_class"}))
for name, cond in ok:
    print(("PASS " if cond else "FAIL ") + name)
print(f"{sum(c for _, c in ok)} of {len(ok)} checks passed")
print("grids_ok:", r["grids_ok"], "| problems:", r["problems"])
sys.exit(0 if all(c for _, c in ok) else 1)
