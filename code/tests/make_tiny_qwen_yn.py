"""Tiny random Qwen2-architecture model whose tokenizer has single tokens for Yes / No (with and without a
leading space) and a Qwen-style chat template that injects a default system prompt. Offline tests only."""
import sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

out = sys.argv[1]
corpus = ["Task: predict in-hospital mortality. Question: Will this patient die? Answer: Yes Answer: No",
          "Answer with Yes or No. Answer: Yes Answer: No Yes No Yes No",
          "Evidence from a machine-learning model (XGBoost) increases the risk of death (impact +0.42)",
          "- Heart rate, highest (beats/min): 120 - Glasgow Coma Scale, lowest (3-15): 6"] * 60
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=700, special_tokens=specials,
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tpl = ("{% if messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}"
       "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
       "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                              additional_special_tokens=["<|im_start|>"])
tok.chat_template = tpl
v = tok.get_vocab()
for w in ("Yes", "ĠYes", "No", "ĠNo"):
    assert w in v, f"{w} missing from tiny vocab"
tok.save_pretrained(out)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, bos_token_id=None,
                  tie_word_embeddings=True)
model = Qwen2ForCausalLM(cfg)
model.generation_config.eos_token_id = tok.eos_token_id
model.save_pretrained(out)
print("tiny model:", out, "vocab", len(tok))
