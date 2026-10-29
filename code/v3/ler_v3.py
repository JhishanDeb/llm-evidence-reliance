"""Third study (PREREGISTRATION_V3.md, release prereg-v3): new model families, and which part of the evidence card
drives the two asymmetries.

Everything that Studies 1 and 2 registered (data, splits, XGBoost, cards, solved examples, prompts, the reading of
the answer-word log-probabilities, the statistics a and a') comes from ler_main.py, version main-2.0, which is not
changed (its SHA-256 is checked below). This module adds only what the third study needs:

  1. the card formats: the registered card ('full'), the card without its numbers ('words'), the card without its
     direction words ('numbers'), a one-sentence direction cue ('cue'), and, in the reversed framing, the registered
     card worded in terms of the outcome that the question names ('matched');
  2. the prompt grid, the models (with the engineering-check candidates) and the check against the protocol text;
  3. the model window (max_model_len) per model, a smoke test and the engineering check that use it;
  4. a fake model for the CPU tests;
  5. the registered analysis: the outcome part O and the answer-word part W of the asymmetry for every card format,
     their changes against the registered card, and the decision rules of R4 to R7.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

import ler_main as L

V3_VERSION = "v3-1.1"
MAIN_SHA256 = "0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e"     # ler_main.py, main-2.0
PROTOCOL_V3_TAG = "prereg-v3"
PROTOCOL_V3_URL = f"{L._RAW}/{PROTOCOL_V3_TAG}/PREREGISTRATION_V3.md"
PROTOCOL_V1_AT_V3_URL = f"{L._RAW}/{PROTOCOL_V3_TAG}/PREREGISTRATION.md"
PROTOCOL_V2_AT_V3_URL = f"{L._RAW}/{PROTOCOL_V3_TAG}/PREREGISTRATION_V2.md"
V3_MARKERS = ["## 4. Models", "## 5. Card formats", "## 6. Prompts", "### R4.", "### R5.", "### R6.", "### R7.",
              "### Q8.", "## 8. Analysis"]
N_BOOT = 10_000

# =============================================================================================================
# 1. Card formats (PREREGISTRATION_V3.md, Section 5)
# =============================================================================================================
FORMATS = ("full", "words", "numbers", "cue", "matched")
REDUCED = ("words", "numbers", "cue")              # cards that keep less than the registered card
CHANGED = ("words", "numbers", "cue", "matched")   # every format other than the registered card
FMT_SET = {"full": "A", "words": "Awords", "numbers": "Anumbers", "cue": "Acue", "matched": "Amatched"}
SET_FMT = {v: k for k, v in FMT_SET.items()}
# one line per factor; {feature}, {value} and the order of the five factors are those of the registered card
FULL_LINE = "{k}. {feature} = {value}: {word} the risk of {outcome} (impact {impact})"       # = ler_main.card_text
WORDS_LINE = "{k}. {feature} = {value}: {word} the risk of {outcome}"
NUMBERS_LINE = "{k}. {feature} = {value}: effect on the risk of {outcome} (impact {impact})"
MATCHED_LINE = "{k}. {feature} = {value}: {word} the chance of {safe} (impact {impact})"
CUE_TEXT = ("Evidence from a machine-learning model (XGBoost) trained on other patients: taken together, the {n} "
            "factors that most influenced its risk estimate for this patient {verb} the risk of {outcome}.")
# the safe outcome as the reversed question names it (used only by the 'matched' card)
SAFE_OUTCOME = {"physionet2012": "survival", "physionet2019": "staying free of sepsis",
                "diabetes130": "avoiding readmission within 30 days"}


def _impact(x: float) -> str:
    """The registered card's number: two decimals, and the printed sign always matches the direction word."""
    return ("+" if x > 0 else "-") + f"{abs(x):.2f}"


def _word(x: float) -> str:
    return "increases" if x > 0 else "decreases"


def card_text_v3(b, card, fmt: str) -> str:
    """The evidence block of a prompt in one card format. 'full' is the registered card (ler_main.card_text)."""
    if fmt == "full":
        return b.card_text(card)
    if fmt == "cue":
        verb = "increase" if sum(it["shap"] for it in card) > 0 else "decrease"   # = the card direction of ler_main
        return CUE_TEXT.format(n=len(card), verb=verb, outcome=b.spec["outcome"])
    lines = []
    for k, it in enumerate(card, 1):
        feat = L.display_name(b.name, it["col"])
        val = L.display_value(b.name, it["col"], it["value"], b.id_maps)
        x = float(it["shap"])
        if fmt == "words":
            lines.append(WORDS_LINE.format(k=k, feature=feat, value=val, word=_word(x), outcome=b.spec["outcome"]))
        elif fmt == "numbers":
            lines.append(NUMBERS_LINE.format(k=k, feature=feat, value=val, outcome=b.spec["outcome"], impact=_impact(x)))
        elif fmt == "matched":     # the effect on the log-odds of the safe outcome is -SHAP
            lines.append(MATCHED_LINE.format(k=k, feature=feat, value=val, word=_word(-x), safe=SAFE_OUTCOME[b.name],
                                             impact=_impact(-x)))
        else:
            raise ValueError(fmt)
    return L.EVIDENCE_HEADER + "\n" + "\n".join(lines)


def prompt_v3(b, rid, cond: str, framing: str, fmt: str = "full"):
    """-> (text, card_dir). The registered prompt of ler_main (example set A) with its evidence block replaced by the
    block of `fmt`; nothing else changes. card_dir is the registered one (1 = the shown card points to the alarming
    outcome). C0 has no card, so every format shares the registered C0 prompt."""
    text, cdir = b.prompt(rid, cond, framing, "A")
    if text is None or cond == "C0" or fmt == "full":
        return text, cdir
    if L.is_choice(framing):
        raise ValueError("the card formats are used with the Yes/No question only")
    if fmt == "matched" and framing != "reversed":
        raise ValueError("the matched card is used in the reversed framing only")
    card = b.shown_card(rid, cond)
    old = b.card_text(card)
    if text.count(old) != 1:
        raise RuntimeError("the registered evidence block is not found exactly once")
    return text.replace(old, card_text_v3(b, card, fmt)), cdir


def check_format_prompt(b, rid, cond, framing, fmt, text, cdir):
    """Assertions for one prompt of a changed format (used by the run notebook and the tests): the prompt equals the
    registered prompt outside the evidence block, and the block is the registered card with exactly the registered
    change (numbers removed, direction words removed, the one-sentence cue, or the card worded for the safe outcome)."""
    ref, rdir = b.prompt(rid, cond, framing, "A")
    assert cdir == rdir
    card = b.shown_card(rid, cond)
    full = b.card_text(card)
    new = card_text_v3(b, card, fmt)
    assert text == ref.replace(full, new) and text.count(new) == 1 and full not in text
    fl = full.split("\n")[1:]
    if fmt == "words":
        nl = new.split("\n")[1:]
        assert [re.sub(r" \(impact [+-]\d+\.\d\d\)$", "", x) for x in fl] == nl
    elif fmt == "numbers":
        nl = new.split("\n")[1:]
        assert [re.sub(r": (increases|decreases) the risk of ", ": effect on the risk of ", x) for x in fl] == nl
    elif fmt == "matched":
        nl = new.split("\n")[1:]
        for a, c, it in zip(fl, nl, card):
            head = a.split(": ")[0]
            assert c.startswith(head + ": ") and f"the chance of {SAFE_OUTCOME[b.name]} (impact {_impact(-it['shap'])})" in c
    elif fmt == "cue":
        assert new.count("\n") == 0 and (" increase the risk of " in new) == (cdir == 1)
    return True


# =============================================================================================================
# 2. Prompt grid, models, protocol check
# =============================================================================================================
SPLIT_CONDS = ("C0", "C2", "C4")


def grid_v3(grids) -> list:
    """(condition, framing, example-set label) of a third-study run. The label is 'A' for the registered card and
    'A' + format for a changed card ('Awords', ...); the solved examples are always those of set A.
      'split'   = C0, C2, C4 x alarm and reversed framing, registered card (6 per patient);
      'formats' = C2, C4 x alarm and reversed framing x words, numbers, cue (12);
      'matched' = C2, C4 x reversed framing, matched card (2);
      'choice'  = C0, C2, C4 x both option orders (6; the answer-option grid of the second study)."""
    out = []
    if "split" in grids:
        out += [(c, f, "A") for f in L.FRAMINGS for c in SPLIT_CONDS]
    if "formats" in grids:
        out += [(c, f, FMT_SET[m]) for m in REDUCED for f in L.FRAMINGS for c in ("C2", "C4")]
    if "matched" in grids:
        out += [(c, "reversed", FMT_SET["matched"]) for c in ("C2", "C4")]
    if "choice" in grids:
        out += [(c, f, "A") for c in L.CHOICE_GRID["conditions"] for f in L.CHOICE_FRAMINGS]
    return out


EARLIER_GRIDS = ("split", "formats", "matched")              # 20 prompts per patient
NEW_GRIDS = ("split", "formats", "matched", "choice")        # 26 prompts per patient

# The three models of the earlier studies whose two parts are best established (Section 4)
EXISTING_V3 = {
    "Qwen/Qwen2.5-7B-Instruct": dict(role="earlier", kind="instruct", family="qwen", tag="qwen-instruct",
                                     grids=EARLIER_GRIDS, revision="a09a35458c702b33eeacc393d103063234e8bc28",
                                     earlier_run="main_qwen-instruct", developer="Alibaba (Qwen team)"),
    "meta-llama/Llama-3.1-8B-Instruct": dict(role="earlier", kind="instruct", family="llama", tag="llama-instruct",
                                             grids=EARLIER_GRIDS, revision="0e9e39f249a16976918f6564b8830bc894c89659",
                                             earlier_run="main_llama-instruct", developer="Meta"),
    "allenai/Olmo-3-7B-Instruct": dict(role="earlier", kind="instruct", family="olmo3", tag="olmo3-instruct",
                                       grids=EARLIER_GRIDS, revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
                                       earlier_run="v2_olmo3-instruct", developer="Allen Institute for AI"),
}
# Candidates for the three new families, in the order of the engineering check (set before the check)
CANDIDATES_V3 = {
    "zai-org/GLM-4-9B-0414": dict(role="new", kind="instruct", family="glm4", tag="glm4-9b", developer="Zhipu AI"),
    "tiiuae/Falcon3-7B-Instruct": dict(role="new", kind="instruct", family="falcon3", tag="falcon3-7b",
                                       developer="Technology Innovation Institute"),
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": dict(role="new", kind="instruct", family="exaone", tag="exaone35-8b",
                                                 developer="LG AI Research"),
    "CohereLabs/aya-expanse-8b": dict(role="new", kind="instruct", family="aya", tag="aya-expanse-8b",
                                      developer="Cohere Labs"),
    "01-ai/Yi-1.5-9B-Chat-16K": dict(role="new", kind="instruct", family="yi", tag="yi15-9b", developer="01.AI"),
    "microsoft/Phi-4-mini-instruct": dict(role="new", kind="instruct", family="phi4", tag="phi4-mini",
                                          developer="Microsoft"),
}
EARLIER_FAMILIES = ("qwen", "qwen3", "llama", "olmo3", "granite3", "mistral")
# The first three eligible candidates of the engineering check (2026-10-29, results/v3/engcheck_v3.zip), in the
# registered order, with the revision the check loaded. Not eligible: GLM-4-9B-0414 (vLLM 0.26.0 refuses float16 for
# this model type), EXAONE-3.5-7.8B-Instruct (its configuration cannot be read without the model's own code) and
# aya-expanse-8b (median A/B mass below 0.5). No eligible candidate remains as a replacement.
PINNED_NEW: dict = {
    "tiiuae/Falcon3-7B-Instruct": "1e57a0ecd176c7c139f289c60a74e57f887c3dfb",
    "01-ai/Yi-1.5-9B-Chat-16K": "2f8753bed95bd4fede3fd64f8305428620b4b212",
    "microsoft/Phi-4-mini-instruct": "cfbefacb99257ffa30c83adab238a50856ac3083",
}
ENGCHECK_ZIP_SHA256 = "6984a4d96861563db7d35f66d0880bc63cfc2c3366a77e7d35738497088c6e00"   # results/v3/engcheck_v3.zip
ENGCHECK_CODE_SHA256 = "148e1990bcf15905c07657c2aad54099afdb4bc8d64b6830dbf8d4665781e61d"  # the ler_v3.py it ran
REGISTERED_MODELS_V3 = {**EXISTING_V3,
                        **{m: dict(CANDIDATES_V3[m], grids=NEW_GRIDS, revision=r) for m, r in PINNED_NEW.items()}}

# Cells in which a part was established with the registered card in Studies 1 and 2 (exploratory split of the
# published paper: valid cell, a and a' of the same sign, intersection-union p < 0.05 without correction), with the
# sign of that part. They are the pre-listed eligible cells of R6 (word part W) and R7 (outcome part O).
PRELISTED = {
    "W": {"qwen-instruct|physionet2012": 1, "qwen-instruct|physionet2019": -1,
          "llama-instruct|physionet2012": -1, "llama-instruct|diabetes130": -1,
          "olmo3-instruct|physionet2012": -1, "olmo3-instruct|physionet2019": -1},
    "O": {"qwen-instruct|physionet2012": -1, "llama-instruct|physionet2012": -1,
          "llama-instruct|physionet2019": -1, "olmo3-instruct|physionet2012": 1},
}


def models_v3(role=None) -> list:
    return [m["tag"] for m in REGISTERED_MODELS_V3.values() if role is None or m["role"] == role]


def protocol_strings_v3() -> list:
    """Every fixed string the third study adds, the ler_main.py hash, and every registered model id and revision,
    for the check against PREREGISTRATION_V3.md."""
    out = [WORDS_LINE, NUMBERS_LINE, MATCHED_LINE, CUE_TEXT, MAIN_SHA256, ENGCHECK_ZIP_SHA256, ENGCHECK_CODE_SHA256]
    out += [f"| {SAFE_OUTCOME[ds]} |" for ds in L.DATASETS]
    for mid, m in REGISTERED_MODELS_V3.items():
        out += [mid, m["revision"]]
    for mid in CANDIDATES_V3:
        out.append(mid)
    return out


def check_against_protocol_v3(text: str) -> list:
    """Strings of the third study missing from PREREGISTRATION_V3.md, and missing section markers (should be empty)."""
    flat = re.sub(r"\s+", " ", text)
    miss = [s for s in protocol_strings_v3() if re.sub(r"\s+", " ", s) not in flat]
    return miss + [m for m in V3_MARKERS if m not in text]


# =============================================================================================================
# 3. Model window, smoke test, engineering check
# =============================================================================================================
def model_window(model_path, cap: int = L.MAX_MODEL_LEN) -> int:
    """max_model_len for vLLM: 16,384 as in both studies, or the model's own window if it is shorter. For models
    with 'longrope' scaling (Phi-4-mini) the window is the original one (4,096 for Phi-4-mini), so that vLLM uses the
    rotary factors that the model uses for prompts of this length. It changes no prompt and no reading."""
    cfg = json.load(open(os.path.join(model_path, "config.json")))
    tc = cfg.get("text_config") or {}
    mpe = cfg.get("max_position_embeddings") or tc.get("max_position_embeddings") or cfg.get("seq_length")
    win = min(int(cap), int(mpe)) if mpe else int(cap)
    rs = cfg.get("rope_scaling") or tc.get("rope_scaling") or {}
    orig = cfg.get("original_max_position_embeddings") or rs.get("original_max_position_embeddings")
    if (rs.get("type") or rs.get("rope_type")) in ("longrope", "su") and orig:
        win = min(win, int(orig))
    return int(win)


def smoke_v3(model_path, kind="instruct", prompts_path=None):
    """ler_main.smoke_test with the model window: run in a subprocess before vLLM is loaded in the notebook kernel.
    Also runs the exact-reading check of Deviation D2 (recorded; the runs read the top 1,000, Deviation D5)."""
    import torch
    tp = 2 if torch.cuda.device_count() >= 2 else 1
    t0 = time.time()
    win = model_window(model_path)
    be = L.VLLMBackend(model_path, tensor_parallel_size=tp, max_model_len=win, read="token_ids")
    prompts = [L.render("Is the sky usually blue on a clear day?\n" + L.ANSWER_INSTRUCTION, kind, be.tokenizer)]
    choice = []
    if prompts_path:
        data = json.load(open(prompts_path))
        prompts += data.get("yesno", [])
        choice = data.get("choice", [])
    check = be.check_exact(prompts)
    check_choice = be.check_exact(choice, ("A", "B")) if choice else None
    read = "token_ids" if check.get("ok") and (check_choice is None or check_choice.get("ok")) else "topk"
    be.read = read
    r = be.yes_no(prompts[:1])
    res = {"ok": True, "load_seconds": time.time() - t0, "tensor_parallel_size": tp, "max_model_len": win, "read": read,
           "exact_check": check, "sky_prompt": r[0], "info": be.info()}
    if check_choice is not None:
        res["exact_check_choice"] = check_choice
    print("SMOKE_V3_RESULT " + json.dumps(res, default=str))
    return res


FMT_LABELS = ("yesno_full", "yesno_words", "yesno_numbers", "yesno_cue", "yesno_matched", "choice")
REQUIRED_NEW = FMT_LABELS
REQUIRED_EARLIER = FMT_LABELS[:5]


def fmt_label(framing: str, shot_set: str) -> str:
    return "choice" if L.is_choice(framing) else "yesno_" + SET_FMT[shot_set]


def engineering_check_v3(model_id, kind, prompts_path, cache_dir, revision=None, required=None):
    """BEFORE registration (PREREGISTRATION_V3.md, Section 10): can this model be run as registered? Engineering
    only: download, the vLLM float16 load on the Kaggle GPUs with the model window, the chat template and its default
    system text, the single-token answer words, the exact-reading check (D2), finiteness, the answer-word mass of every
    card format and of the answer options, the share of imputed readings under the top-1,000 reading (D5), prompt
    lengths, the time per prompt, and the four questions with an obvious answer. No accuracy, no answer direction and
    no asymmetry is computed; the prompts are dev-sample prompts. Prints one ENGCHECK_V3_RESULT line."""
    required = tuple(required or (REQUIRED_NEW if kind == "instruct" else ("yesno_full",)))
    res = {"model_id": model_id, "kind": kind, "formats_required": list(required), "ok": False, "problems": [],
           "code": {"ler_v3": V3_VERSION, "ler_main": L.CODE_VERSION}}
    t0 = time.time()
    try:
        path, sha = L.download_model(model_id, cache_dir, revision, os.environ.get("HF_TOKEN"))
        res.update(revision=sha, download_seconds=round(time.time() - t0, 1))
        cfg = json.load(open(os.path.join(path, "config.json")))
        tcfg = cfg.get("text_config") or {}
        res["config"] = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                         "dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tcfg.get("torch_dtype"),
                         "quantization": (cfg.get("quantization_config") or {}).get("quant_method"),
                         "max_position_embeddings": cfg.get("max_position_embeddings") or tcfg.get("max_position_embeddings"),
                         "rope_scaling": cfg.get("rope_scaling") or tcfg.get("rope_scaling")}
        win = model_window(path)
        res["max_model_len"] = win
        import torch
        tp = 2 if torch.cuda.device_count() >= 2 else 1
        t1 = time.time()
        be = L.VLLMBackend(path, tensor_parallel_size=tp, max_model_len=win, read="token_ids")
        res.update(load_seconds=round(time.time() - t1, 1), tensor_parallel_size=tp)
        items = json.load(open(prompts_path))
        tok = be.tokenizer
        res["template_system_text"] = L.template_system_text(tok) if kind == "instruct" else None
        for pair in (("Yes", "No"), ("A", "B")):
            try:
                be.ensure_words(pair)
            except Exception as e:
                res["problems"].append(f"no single-token variant for {pair}: {e!r}"[:300])
        res["variants"] = {w: [[i, be.strings[i]] for i in ids] for w, ids in be.variants.items()}
        groups = {}
        for it in items:
            if it["fmt"] == "choice" and kind != "instruct":
                continue
            groups.setdefault((it["set"], it["fmt"]), []).append(it)
        rendered = {k: [L.render(it["text"], kind, tok) for it in v] for k, v in groups.items()}
        ntok = [len(tok.encode(r, add_special_tokens=False)) for v in rendered.values() for r in v]
        res["prompt_tokens"] = {"median": int(np.median(ntok)), "max": int(max(ntok))}
        if max(ntok) >= win - 1:
            res["problems"].append(f"a prompt is longer than the model window ({max(ntok)} >= {win - 1})")
        for (sset, fmt), rs in rendered.items():
            if sset == "check":
                res[f"rendered_tail_{fmt}"] = rs[-1][-700:]
        for fmt, pair in (("yesno_full", ("Yes", "No")), ("choice", ("A", "B"))):
            if ("check", fmt) in rendered and pair[0] in be.variants:
                res[f"exact_check_{fmt}"] = be.check_exact(rendered[("check", fmt)], pair)
        be.read = "topk"
        for (sset, fmt), rs in sorted(rendered.items()):
            pair = ("A", "B") if fmt == "choice" else ("Yes", "No")
            if pair[0] not in be.variants:
                continue
            t1 = time.time()
            rows = be.answer_logprobs(rs, pair)
            el = time.time() - t1
            res[f"reading_{sset}_{fmt}"] = dict(L._mass(rows), seconds=round(el, 1),
                                                 seconds_per_prompt=round(el / max(1, len(rs)), 4))
        san = {}
        for name, (text, pair, first) in L.SANITY_PROMPTS.items():
            if pair[0] not in be.variants:
                continue
            r = be.answer_logprobs([L.render(text, kind, tok)], pair)[0]
            san[name] = bool((r["lp_pos"] > r["lp_neg"]) == first)
        res["sanity_right_word_more_likely"] = san
        for k, v in list(res.items()):
            if k.startswith("reading_"):
                fmt = k.split("_", 2)[2]
                if fmt in required:
                    if v["finite_share"] < 1:
                        res["problems"].append(f"{k}: non-finite readings ({v['finite_share']:.3f} finite)")
                    if not v["mass_median"] >= L.VALIDITY_MIN_MASS:
                        res["problems"].append(f"{k}: median answer-word mass {v['mass_median']:.3f} < "
                                               f"{L.VALIDITY_MIN_MASS}")
        for fmt in required:
            if not any(k == f"reading_check_{fmt}" for k in res):
                res["problems"].append(f"no readings for the required format {fmt}")
        need_yn, need_ab = any(f.startswith("yesno") for f in required), "choice" in required
        failed = [k for k, v in san.items() if not v and ((k in ("sky_yes", "fire_no") and need_yn) or
                                                          (k in ("color_A", "color_B") and need_ab))]
        if failed:
            res["problems"].append(f"sanity prompts failed: {failed}")
        if res["config"].get("quantization"):
            res["problems"].append(f"quantized weights ({res['config']['quantization']}); the study runs 16-bit weights")
        res["ok"] = not res["problems"]
    except Exception as e:
        import traceback
        res["problems"].append(f"error: {e!r}"[:500])
        res["traceback"] = traceback.format_exc()[-3000:]
    res["total_seconds"] = round(time.time() - t0, 1)
    print("ENGCHECK_V3_RESULT " + json.dumps(res, default=str))
    return res


def parse_result(output: str, key: str) -> dict | None:
    """The JSON printed after `key` (the last occurrence), even when other output ends up on the same line before it.
    None if there is none or it cannot be read."""
    key = key.rstrip() + " "
    i = output.rfind(key)
    if i < 0:
        return None
    line = output[i + len(key):].split("\n", 1)[0]
    try:
        return json.loads(line)
    except ValueError:
        return None


# =============================================================================================================
# 4. Fake model for the CPU tests (never used on Kaggle)
# =============================================================================================================
class MockV3(L.MockBackend):
    """ler_main.MockBackend that reads every card format. A line of the words card counts as an impact of
    +-`words_impact`, the cue as +-`cue_impact`; the matched card's impacts are turned back into the alarming
    direction. `fmt_gain` scales the evidence effect per format, and `no_pull` (one value or one per format) moves the
    Yes/No answer toward 'No' whenever a card is shown: a pure answer-word part W = -2 * no_pull. Unequal
    `alarm_gain` and `safe_gain` give an outcome part."""
    name = "mock"
    _CUE = re.compile(r"taken together, the \d+ factors that most influenced its risk estimate for this patient "
                      r"(increase|decrease) the risk of")
    _MATCHED = re.compile(r": (increases|decreases) the chance of [^\n]* \(impact ([+-]\d+\.\d+)\)")
    _NUMBERS = re.compile(r": effect on the risk of [^\n]* \(impact ([+-]\d+\.\d+)\)")
    _FULL = re.compile(r": (increases|decreases) the risk of [^\n]* \(impact ([+-]\d+\.\d+)\)")
    _WORDS = re.compile(r"^\d+\. [^\n]*: (increases|decreases) the risk of [^\n(]*$", re.M)

    def __init__(self, reversed_questions, no_pull=0.0, fmt_gain=None, words_impact=0.35, cue_impact=1.0, **kw):
        super().__init__(reversed_questions, **kw)
        self.no_pull = ({f: float(no_pull.get(f, 0.0)) for f in FORMATS} if isinstance(no_pull, dict)
                        else {f: float(no_pull) for f in FORMATS})
        self.fmt_gain = {f: 1.0 for f in FORMATS}
        self.fmt_gain.update(fmt_gain or {})
        self.words_impact, self.cue_impact = float(words_impact), float(cue_impact)

    def evidence(self, p: str):
        part = p.split(L.PATIENT_HEADER)[-1]
        m = self._CUE.search(part)
        if m:
            return "cue", (1.0 if m.group(1) == "increase" else -1.0) * self.cue_impact
        x = self._MATCHED.findall(part)
        if x:
            return "matched", -sum(float(v) for _, v in x)
        x = self._NUMBERS.findall(part)
        if x:
            return "numbers", sum(float(v) for v in x)
        x = self._FULL.findall(part)
        if x:
            return "full", sum(float(v) for _, v in x)
        x = self._WORDS.findall(part)
        if x:
            return "words", sum((1.0 if w == "increases" else -1.0) * self.words_impact for w in x)
        return None, 0.0

    def word_logprobs(self, prompts, words=("Yes", "No")):
        wanted = self.wanted_ids(words)
        res = []
        for p in prompts:
            key = p.split(L.PATIENT_HEADER)[-1].split("\n\n")[0]
            base = 3.0 * (L._u(key, self.seed) - 0.5) + self.lean
            if key in self.truth:
                base += self.signal * (2 * self.truth[key] - 1)
            fmt, s = self.evidence(p)
            choice = L.CHOICE_INSTRUCTION in p
            ga, gs = (self.c_alarm_gain, self.c_safe_gain) if choice else (self.alarm_gain, self.safe_gain)
            g = self.fmt_gain.get(fmt, 1.0) if fmt else 1.0
            z = self.scale * (base + g * (ga * s if s > 0 else gs * s))
            if choice:
                opt_a = re.findall(r"\nA: (.*)\nB: ", p)[-1]
                zp = (z if opt_a in self.alarm_options else -z) + self.letter_bias
                ids = (6, 5, 8, 7)
            else:
                q = re.findall(r"Question: (.*)\n" + re.escape(L.ANSWER_INSTRUCTION), p)
                rev = bool(q) and q[-1] in self.reversed_q
                zp = (-z if rev else z) - (self.no_pull[fmt] if fmt else 0.0)
                ids = (2, 1, 4, 3)
            mass = 0.80 + 0.19 * L._u(key, "m")
            lmy = math.log(mass) - math.log1p(math.exp(-zp)) if zp > -30 else math.log(mass) + zp
            lmn = math.log(mass) - math.log1p(math.exp(zp)) if zp < 30 else math.log(mass) - zp
            full = {ids[0]: lmy + math.log(0.9), ids[1]: lmy + math.log(0.1), ids[2]: lmn + math.log(0.9),
                    ids[3]: lmn + math.log(0.1)}
            other = math.log(1 - mass) - math.log(4) if mass < 1 else -50.0
            res.append(({t: full.get(t, other) for t in wanted}, 0))
        return res


# =============================================================================================================
# 5. Analysis (PREREGISTRATION_V3.md, Sections 7 and 8)
# =============================================================================================================
def slice_v3(df: pd.DataFrame, framing: str, fmt: str = "full") -> pd.DataFrame:
    """Records of one framing and card format in the first study's shape: C0 (no card, shared by every format) and
    C2, C4 with the card of `fmt`, all labelled example set 'A'. In the alarm framing the registered card already names
    the outcome of the question, so the 'matched' slice of the alarm framing is the registered one."""
    a = df[(df["framing"] == framing) & (df["shot_set"] == "A") & df["cond"].isin(SPLIT_CONDS)]
    if fmt == "full" or (fmt == "matched" and framing == "alarm"):
        return a
    v = df[(df["framing"] == framing) & (df["shot_set"] == FMT_SET[fmt]) & df["cond"].isin(("C2", "C4"))]
    return pd.concat([a[a["cond"] == "C0"], v.assign(shot_set="A")], ignore_index=True)


def _median_mass(r: pd.DataFrame) -> float:
    return float(r["yn_mass"].astype(float).fillna(0.0).median()) if len(r) else float("nan")


def validity_v3(df: pd.DataFrame) -> dict:
    """Section 8. Split cell (model x dataset): median Yes+No mass >= 0.5 over the registered-card records of the
    alarm framing (C0, C2, C4) and of the reversed framing, and C0 accuracy >= 0.55 (alarm framing). Format cell: the
    split cell is valid and the median mass over the records of that format (C2, C4) is >= 0.5. A record without a
    usable reading counts as mass 0."""
    al, rv = slice_v3(df, "alarm"), slice_v3(df, "reversed")
    c0 = al[(al["cond"] == "C0") & (al["valid"] == 1)]
    acc = float(c0["correct"].mean()) if len(c0) else float("nan")
    m_al, m_rv = _median_mass(al), _median_mass(rv)
    ok = bool(m_al >= L.VALIDITY_MIN_MASS and m_rv >= L.VALIDITY_MIN_MASS and acc >= L.VALIDITY_MIN_C0_ACC)
    fm = {}
    for f in CHANGED:
        r = df[df["shot_set"] == FMT_SET[f]]
        m = _median_mass(r)
        fm[f] = {"median_mass": m, "n_records": int(len(r)), "passes": bool(ok and m >= L.VALIDITY_MIN_MASS)}
    return {"alarm_median_mass": m_al, "reversed_median_mass": m_rv, "c0_accuracy": acc,
            "n_c0_valid": int(len(c0)), "passes": ok, "formats": fm}


# ---------------------------------------------------------------- the statistics a and a', vectorised bootstrap
def _solve(X, W, d):
    """Batched weighted least squares; X: B x n x k (or n x k), W: B x n, d: n -> B x k."""
    if X.ndim == 2:
        A = np.einsum("bn,nk,nl->bkl", W, X, X)
        r = np.einsum("bn,nk,n->bk", W, X, d)
    else:
        A = np.einsum("bn,bnk,bnl->bkl", W, X, X)
        r = np.einsum("bn,bnk,n->bk", W, X, d)
    try:
        return np.linalg.solve(A, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = []
        for i in range(len(A)):
            try:
                out.append(np.linalg.solve(A[i], r[i]))
            except np.linalg.LinAlgError:
                out.append(np.full(A.shape[-1], np.nan))
        return np.stack(out)


def _wstd(X, W):
    n = W.sum(1, keepdims=True)
    m = (W * X).sum(1, keepdims=True) / n
    return np.sqrt((W * (X - m) ** 2).sum(1, keepdims=True) / n)


def stat_a(x: dict, W) -> np.ndarray:
    """a of Deviation D1 (ler_main.adjusted_fit) for B weight vectors at once. Weights are resample counts, so a
    weighted fit equals the ordinary fit on the resampled rows (z standardized and s = SD of e with the same counts)."""
    l0, la, ls, t, z, y = (x[k] for k in ("l0", "la", "ls", "t", "z", "y"))
    n = W.sum(1, keepdims=True)
    zs = (z[None, :] - (W * z).sum(1, keepdims=True) / n) / _wstd(np.broadcast_to(z, W.shape), W)
    Z = np.stack([np.ones_like(zs), zs, zs ** 2, zs ** 3, np.broadcast_to(y, zs.shape)], -1)
    e = np.einsum("bnk,bk->bn", Z, _solve(Z, W, l0))
    s = _wstd(e, W)
    X = np.stack([np.ones_like(e), e, (e / s) ** 3, np.broadcast_to(t, e.shape)], -1)
    return _solve(X, W, (la - l0) + (ls - l0))[:, 0]


def stat_ap(x: dict, W) -> np.ndarray:
    """a' of Deviation D1 (ler_main.reading_adjusted_asymmetry) for B weight vectors at once."""
    l0, la, ls, t = (x[k] for k in ("l0", "la", "ls", "t"))
    Lb = np.broadcast_to(l0, W.shape)
    s = _wstd(Lb, W)
    X = np.stack([np.ones_like(Lb), Lb, (Lb / s) ** 3, np.broadcast_to(t, Lb.shape)], -1)
    return _solve(X, W, (la - l0) + (ls - l0))[:, 0]


def stat_move(x: dict, W) -> np.ndarray:
    """Mean movement between the alarm-pointing and the safe-pointing card, mean(la - ls) (log-odds of the alarming
    outcome): how strongly the card moves the model at all."""
    return (W @ (x["la"] - x["ls"])) / W.sum(1)


def bootstrap(n: int, fns: dict, n_boot: int = N_BOOT, seed: int = L.BOOT_SEED, chunk: int = 1000):
    """Paired bootstrap over patients: multinomial resample counts (seed 2026, `n_boot` resamples in chunks of 1,000),
    the same resamples for every statistic of one comparison. -> (estimates, draws)."""
    rng = np.random.default_rng(seed)
    one = np.ones((1, n))
    est = {k: float(f(one)[0]) for k, f in fns.items()}
    draws = {k: [] for k in fns}
    for start in range(0, n_boot, chunk):
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=min(chunk, n_boot - start)).astype(float)
        for k, f in fns.items():
            draws[k].append(f(W))
    return est, {k: np.concatenate(v) for k, v in draws.items()}


def _ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])] if len(b) else [float("nan")] * 2


def _p2(b):
    return L.boot_p(b)


def _joint(e1, b1, e2, b2) -> dict:
    """Intersection-union reading of the two statistics: p = max of the two bootstrap p-values; a direction only
    when both estimates have the same sign."""
    p = max(_p2(b1), _p2(b2))
    ok = bool(np.isfinite(e1) and np.isfinite(e2))
    sign = int(np.sign(e1)) if ok and np.sign(e1) == np.sign(e2) else 0
    return {"p": float(p) if ok else float("nan"), "sign": sign, "conclusive": bool(sign != 0 and p < 0.05)}


def _summ(e, b) -> dict:
    return {"est": float(e), "ci": _ci(b), "p": _p2(b)}


def _two(est, dr, ka, kap) -> dict:
    """One quantity in both versions (from a and from a'), with the joint reading."""
    out = {"a": _summ(est[ka], dr[ka]), "a_prime": _summ(est[kap], dr[kap])}
    out.update(_joint(est[ka], dr[ka], est[kap], dr[kap]))
    return out


def _wide_slice(df, framing, fmt):
    return L._wide(slice_v3(df, framing, fmt))


def _parts(est, dr, fmt):
    """Adds O, W and the alarm-framing statistic of one format from the per-framing statistics."""
    for st in ("a", "ap"):
        for part, sg in (("O", 1), ("W", -1)):
            k = f"{part}_{st}|{fmt}"
            est[k] = (est[f"{st}|alarm|{fmt}"] + sg * est[f"{st}|reversed|{fmt}"]) / 2
            dr[k] = (dr[f"{st}|alarm|{fmt}"] + sg * dr[f"{st}|reversed|{fmt}"]) / 2
    est[f"m|{fmt}"] = (est[f"m|alarm|{fmt}"] + est[f"m|reversed|{fmt}"]) / 2
    dr[f"m|{fmt}"] = (dr[f"m|alarm|{fmt}"] + dr[f"m|reversed|{fmt}"]) / 2


def compare_formats(df: pd.DataFrame, formats, n_boot: int = N_BOOT) -> dict | None:
    """O, W and the alarm-framing statistic for each format in `formats` (and their differences from the first
    one) on the patients with valid C0, C2 and C4 readings in both framings of every format, with one paired
    bootstrap. O = (s_alarm + s_reversed) / 2 and W = (s_alarm - s_reversed) / 2 for s = a and s = a'; W < 0 means
    toward the answer 'No'."""
    wides = {}
    for f in formats:
        for fr in L.FRAMINGS:
            w = _wide_slice(df, fr, f)
            if w is None:
                return None
            wides[(fr, f)] = w
    ids = None
    for w in wides.values():
        ids = w.index if ids is None else ids.intersection(w.index)
    ids = sorted(ids)
    if len(ids) < 10:
        return None
    xs = {k: L.asym_arrays(w.loc[ids]) for k, w in wides.items()}
    fns = {}
    for (fr, f), x in xs.items():
        fns[f"a|{fr}|{f}"] = lambda W, x=x: stat_a(x, W)
        fns[f"ap|{fr}|{f}"] = lambda W, x=x: stat_ap(x, W)
        fns[f"m|{fr}|{f}"] = lambda W, x=x: stat_move(x, W)
    est, dr = bootstrap(len(ids), fns, n_boot)
    out = {"n": len(ids), "formats": {}}
    for f in formats:
        _parts(est, dr, f)
        r = {"O": _two(est, dr, f"O_a|{f}", f"O_ap|{f}"), "W": _two(est, dr, f"W_a|{f}", f"W_ap|{f}"),
             "alarm": _two(est, dr, f"a|alarm|{f}", f"ap|alarm|{f}"),
             "reversed": _two(est, dr, f"a|reversed|{f}", f"ap|reversed|{f}"),
             "movement": _summ(est[f"m|{f}"], dr[f"m|{f}"])}
        for part in ("O", "W"):     # relative part: the part divided by the mean movement (scale-free)
            for st in ("a", "ap"):
                est[f"r{part}_{st}|{f}"] = est[f"{part}_{st}|{f}"] / est[f"m|{f}"]
                dr[f"r{part}_{st}|{f}"] = dr[f"{part}_{st}|{f}"] / dr[f"m|{f}"]
            r[f"relative_{part}"] = _two(est, dr, f"r{part}_a|{f}", f"r{part}_ap|{f}")
        out["formats"][f] = r
    ref = formats[0]
    for f in formats[1:]:          # change against the first format (the registered card)
        d = {}
        for part in ("O", "W"):
            for st in ("a", "ap"):
                k = f"d{part}_{st}|{f}"
                est[k] = est[f"{part}_{st}|{f}"] - est[f"{part}_{st}|{ref}"]
                dr[k] = dr[f"{part}_{st}|{f}"] - dr[f"{part}_{st}|{ref}"]
                k2 = f"dr{part}_{st}|{f}"
                est[k2] = est[f"r{part}_{st}|{f}"] - est[f"r{part}_{st}|{ref}"]
                dr[k2] = dr[f"r{part}_{st}|{f}"] - dr[f"r{part}_{st}|{ref}"]
            d[part] = _two(est, dr, f"d{part}_a|{f}", f"d{part}_ap|{f}")
            d[f"relative_{part}"] = _two(est, dr, f"dr{part}_a|{f}", f"dr{part}_ap|{f}")
        est[f"dm|{f}"], dr[f"dm|{f}"] = est[f"m|{f}"] - est[f"m|{ref}"], dr[f"m|{f}"] - dr[f"m|{ref}"]
        d["movement"] = _summ(est[f"dm|{f}"], dr[f"dm|{f}"])
        out["formats"][f]["change_from_" + ref] = d
    return out


def format_descriptives(df: pd.DataFrame) -> dict:
    """Accuracy by condition and the share of C2 / C4 answers that follow the card, per framing and format."""
    out = {}
    for f in FORMATS:
        for fr in L.FRAMINGS:
            if f == "matched" and fr == "alarm":
                continue
            s = slice_v3(df, fr, f)
            v = s[s["valid"] == 1]
            if not len(v[v["cond"] != "C0"]):
                continue
            ev = v[v["cond"].isin(("C2", "C4"))]
            out[f"{fr}|{f}"] = {"by_condition": L._accuracy_table(v, SPLIT_CONDS),
                                "follow_rate": float((ev["pred_alarm"] == ev["card_dir"]).mean()),
                                "median_mass": _median_mass(s[s["cond"] != "C0"])}
    return out


def analyse_choice_v3(df: pd.DataFrame) -> dict | None:
    """The answer-option asymmetry of the second study (R2 statistic), its validity and accuracy, for R5."""
    c = L.choice_records(df)
    if not len(c):
        return None
    pooled = L.choice_pooled(df)
    out = {"validity": L.validity_choice(df), "n_records": int(len(c)), "n_invalid": int((c["valid"] == 0).sum())}
    v = c[c["valid"] == 1]
    out["letter_A_logodds_mean"] = float((v["lp_yes"] - v["lp_no"]).mean()) if len(v) else float("nan")
    out["by_condition"] = L._accuracy_table(pooled, SPLIT_CONDS) if pooled is not None else {}
    out["asymmetry"] = L.asymmetry(pooled) if pooled is not None else None
    a = L._get(out["asymmetry"], "adjusted", "a")
    ap = L._get(out["asymmetry"], "reading_adjusted", "a")
    p = L._get(out["asymmetry"], "h3_test", "p")
    ok = L._finite(a) and L._finite(ap) and L._finite(p)
    sign = int(np.sign(a)) if ok and np.sign(a) == np.sign(ap) else 0
    out["joint"] = {"p": p if ok else float("nan"), "sign": sign, "conclusive": bool(sign != 0 and ok and p < 0.05)}
    return out


def analyse_cell_v3(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Everything for one model x dataset: validity, the split with the registered card, every changed format
    against the registered card, descriptives, and the answer options when the model has them."""
    cell = {"n_records": int(len(df)), "n_invalid": int((df["valid"] == 0).sum()),
            "n_records_imputed": int((df["n_imputed"] > 0).sum()) if "n_imputed" in df else 0,
            "validity": validity_v3(df)}
    cell["split"] = compare_formats(df, ("full",), n_boot)
    cell["formats"] = {}
    for f in CHANGED:
        if (df["shot_set"] == FMT_SET[f]).any():
            cell["formats"][f] = compare_formats(df, ("full", f), n_boot)
    cell["descriptives"] = format_descriptives(df)
    cell["choice"] = analyse_choice_v3(df)
    return cell


# ---------------------------------------------------------------- decision rules
def _part(cell, part, fmt="full"):
    """The joint reading of a part from the split (fmt 'full') or from a format comparison."""
    if fmt == "full":
        return L._get(cell, "split", "formats", "full", part)
    return L._get(cell, "formats", fmt, "formats", fmt, part)


def _change(cell, part, fmt):
    return L._get(cell, "formats", fmt, "formats", fmt, "change_from_full", part)


def _both(r, sign) -> bool:
    """Both versions (a and a') of a part have the given sign."""
    a, ap = L._get(r, "a", "est"), L._get(r, "a_prime", "est")
    return bool(L._finite(a) and L._finite(ap) and np.sign(a) == sign and np.sign(ap) == sign)


def _pval(r):
    p = L._get(r, "p")
    return p if L._finite(p) else None


def _below(p, alpha=0.05) -> bool:
    return p is not None and p < alpha


def r4(cells: dict, new_tags) -> dict:
    """R4: the word part in the new families. Prediction W < 0 (both versions). Holm over the new models x 3 datasets;
    a cell that fails the split validity rule, or is missing, enters with p = 1. A model shows the pull toward 'No' if
    W is supported in more than half of its valid cells (at least one) and no valid cell has a conclusive W in the
    other direction (both versions > 0, p < 0.05 without correction). Supported if at least two new models, and more
    than half of them, show the pull; not testable if fewer than two new models have a valid cell."""
    ent = {}
    for tag in new_tags:
        for ds in L.DATASETS:
            c = cells.get(f"{tag}|{ds}")
            w = _part(c, "W") if c else None
            ent[f"{tag}|{ds}"] = {"p": _pval(w), "direction_ok": _both(w, -1),
                                  "included": bool(c and c["validity"]["passes"] and _pval(w) is not None),
                                  "W_a": L._get(w, "a", "est"), "W_a_prime": L._get(w, "a_prime", "est"),
                                  "opposite": bool(c and c["validity"]["passes"] and _both(w, 1) and _below(_pval(w)))}
    fam = L.holm_family(ent)
    per = {}
    for tag in new_tags:
        ks = [f"{tag}|{ds}" for ds in L.DATASETS]
        nv = sum(fam[k]["included"] for k in ks)
        ns = sum(fam[k]["supported"] for k in ks)
        no = sum(fam[k]["opposite"] for k in ks)
        per[tag] = {"valid_cells": int(nv), "supported_cells": int(ns), "opposite_cells": int(no),
                    "shows_pull": bool(nv >= 1 and ns > nv / 2 and no == 0)}
    tested = [t for t in new_tags if per[t]["valid_cells"] >= 1]
    k = sum(per[t]["shows_pull"] for t in new_tags)
    testable = len(tested) >= 2
    return {"cells": fam, "per_model": per, "models_tested": len(tested), "models_showing_pull": int(k),
            "testable": testable, "supported": bool(testable and k >= 2 and k > len(new_tags) / 2)}


def r5(cells: dict, new_tags) -> dict:
    """R5: the outcome part in the new families. No direction is predicted: O is reported with two-sided tests and
    Holm over the new models x 3 datasets. Registered check of the split: in every valid cell where O and the
    answer-option asymmetry are both conclusive, they have the same sign. Supported if they agree in every such cell,
    with at least two such cells; not testable with fewer."""
    ent = {}
    for tag in new_tags:
        for ds in L.DATASETS:
            c = cells.get(f"{tag}|{ds}")
            o = _part(c, "O") if c else None
            sign = L._get(o, "sign") or 0
            ent[f"{tag}|{ds}"] = {"p": _pval(o), "direction_ok": bool(sign != 0),
                                  "included": bool(c and c["validity"]["passes"] and _pval(o) is not None),
                                  "sign": int(sign), "O_a": L._get(o, "a", "est"), "O_a_prime": L._get(o, "a_prime", "est")}
    fam = L.holm_family(ent)
    agree = {}
    for key, e in fam.items():
        c = cells.get(key)
        ch = (c or {}).get("choice")
        o = _part(c, "O") if c else None
        if not (e["included"] and o and o["conclusive"] and ch and ch["validity"]["passes"] and ch["joint"]["conclusive"]):
            continue
        agree[key] = {"O_sign": int(o["sign"]), "answer_option_sign": int(ch["joint"]["sign"]),
                      "agree": bool(o["sign"] == ch["joint"]["sign"])}
    n = len(agree)
    ok = sum(v["agree"] for v in agree.values())
    return {"cells": fam, "agreement": agree, "n_compared": n, "n_agree": int(ok), "testable": n >= 2,
            "supported": bool(n >= 2 and ok == n)}


def eligible_cells(cells: dict, part: str, new_fam: dict) -> dict:
    """Eligible cells of R6 (part 'W') and R7 (part 'O'): the pre-listed cells of the earlier models, and every
    new-family cell whose part is established in this study (valid, both versions of the same sign, Holm-adjusted
    p < 0.05 in the R4 or R5 family), each with the sign of its part."""
    out = {k: {"sign": s, "source": "pre-listed (Studies 1 and 2)"} for k, s in PRELISTED[part].items()}
    for k, e in new_fam.items():
        c = cells.get(k)
        r = _part(c, part) if c else None
        if e.get("included") and e.get("p_holm", 1) < 0.05 and r and r["sign"] != 0:
            out[k] = {"sign": int(r["sign"]), "source": "new family, established in this study"}
    return out


def _kept_family(cells, elig, part, fmt):
    """Holm over the eligible cells for one format: the part with this card has the sign it has with the
    registered card (both versions, intersection-union p)."""
    ent = {}
    for k, e in elig.items():
        c = cells.get(k)
        r = _part(c, part, fmt) if c else None
        valid = bool(c and L._get(c, "validity", "formats", fmt, "passes"))
        ch = _change(c, part, fmt) if c else None
        ent[k] = {"p": _pval(r), "direction_ok": _both(r, e["sign"]), "included": bool(valid and _pval(r) is not None),
                  "sign_full": e["sign"], "a": L._get(r, "a", "est"), "a_prime": L._get(r, "a_prime", "est"),
                  "change_a": L._get(ch, "a", "est"), "change_a_prime": L._get(ch, "a_prime", "est"),
                  # smaller: the change points toward 0 (opposite to the part's sign) in both versions, p < 0.05
                  "reduced": bool(valid and ch and _both(ch, -e["sign"]) and _below(_pval(ch)))}
    fam = L.holm_family(ent)
    n = len(fam)
    ns = sum(v["supported"] for v in fam.values())
    nr = sum(v["reduced"] for v in fam.values())
    return {"cells": fam, "n_eligible": n, "n_kept": int(ns), "n_reduced": int(nr), "testable": n >= 1,
            "kept": bool(n >= 1 and ns > n / 2), "reduced_in_majority": bool(n >= 1 and nr > n / 2)}


def r6(cells: dict, r4_res: dict, prelisted_only: bool = False) -> dict:
    """R6: the word part with each changed card (words, numbers, cue, matched). Prediction: in every eligible cell, W
    keeps the sign it has with the registered card. Per format, Holm over the eligible cells; the part is kept with a
    format if it is supported in more than half of the eligible cells. Supported if W is kept with all four formats."""
    elig = eligible_cells(cells, "W", {} if prelisted_only else r4_res["cells"])
    per = {f: _kept_family(cells, elig, "W", f) for f in CHANGED}
    testable = all(v["testable"] for v in per.values())
    return {"eligible": elig, "formats": per, "testable": testable,
            "supported": bool(testable and all(v["kept"] for v in per.values()))}


def r7(cells: dict, r5_res: dict, prelisted_only: bool = False) -> dict:
    """R7: the outcome part with each reduced card (words, numbers, cue); the matched card is reported with it.
    No overall direction is predicted; in each eligible cell the prediction is the sign of O with the registered card.
    Per format, Holm over the eligible cells, kept as in R6. Interpretation (Section 7): which card keeps O."""
    elig = eligible_cells(cells, "O", {} if prelisted_only else r5_res["cells"])
    per = {f: _kept_family(cells, elig, "O", f) for f in CHANGED}
    kw, kn = per["words"]["kept"], per["numbers"]["kept"]
    if not (per["words"]["testable"] and per["numbers"]["testable"]):
        interp = "not testable"
    elif kw and kn:
        interp = "kept with either the direction words or the numbers alone"
    elif kn:
        interp = "carried by the numbers: kept without the direction words, not shown without the numbers"
    elif kw:
        interp = "carried by the direction words: kept without the numbers, not shown without the direction words"
    else:
        interp = "not shown with either the direction words or the numbers alone"
    cue = ("not testable" if not per["cue"]["testable"] else
           "a one-sentence direction cue is enough" if per["cue"]["kept"] else "not shown with the one-sentence cue")
    return {"eligible": elig, "formats": per, "interpretation": interp, "cue": cue}


def confirmatory_v3(cells: dict, new_tags) -> dict:
    out = {"R4": r4(cells, new_tags), "R5": r5(cells, new_tags)}
    out["R6"] = r6(cells, out["R4"])
    out["R7"] = r7(cells, out["R5"])
    out["sensitivity_prelisted_only"] = {"R6": r6(cells, out["R4"], True), "R7": r7(cells, out["R5"], True)}
    return out


def replay_check(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """Q8a: the registered-card prompts scored again in this study against the same prompts in Study 1 or 2
    (C0, C2, C4, alarm and reversed framing, example set A): differences in the log-odds of the alarming outcome."""
    keep = lambda d: d[(d["shot_set"] == "A") & d["framing"].isin(L.FRAMINGS) & d["cond"].isin(SPLIT_CONDS)]
    m = keep(new).merge(keep(old), on="key", suffixes=("", "_old"))
    out = {"n_shared_prompts": int(len(m)), "n_text_differs": int((m["text_sha"] != m["text_sha_old"]).sum())}
    v = m[(m["valid"] == 1) & (m["valid_old"] == 1) & (m["text_sha"] == m["text_sha_old"])]
    d = (v["lo_alarm"].astype(float) - v["lo_alarm_old"].astype(float)).abs()
    out.update(n_compared=int(len(v)), max_abs_diff=float(d.max()) if len(d) else float("nan"),
               median_abs_diff=float(d.median()) if len(d) else float("nan"),
               share_within_0_01=float((d <= 0.01).mean()) if len(d) else float("nan"),
               share_same_answer=float((v["pred_alarm"] == v["pred_alarm_old"]).mean()) if len(v) else float("nan"))
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "smoke":
        smoke_v3(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "instruct", sys.argv[4] if len(sys.argv) > 4 else None)
    elif len(sys.argv) >= 6 and sys.argv[1] == "engcheck":
        engineering_check_v3(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                             sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "None" else None,
                             tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else None)
