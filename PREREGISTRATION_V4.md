# Pre-registration, fourth study: what the answer-polarity part follows, and whether post-training or the chat template produces it

- **Author:** Jhishan Deb
- **Registered:** November 2026, committed and tagged `prereg-v4` before any test prompt of this study was scored (the GitHub commit and the `prereg-v4` release date are the authoritative timestamps)
- **Builds on:** the first study (`PREREGISTRATION.md`, releases `prereg-v1` to `prereg-v1.3`, results in `results/main/`), the second study (`PREREGISTRATION_V2.md`, release `prereg-v2`, results in `results/v2/`), the third study (`PREREGISTRATION_V3.md`, release `prereg-v3`, results in `results/v3/`) and the two exploratory scale checks
- **Status:** written after the results of all three studies, and of exploratory analyses of them, were known (Section 1). They motivate every prediction below. The prompts with the registered card in the registered framings are the prompts of the earlier studies, so their readings for the models of this study are known. No prompt in a new format (Section 5) has been scored on a test patient. New formats were scored only on dev patients, in the engineering check of Section 10, and only the technical quantities listed there were computed from them.
- **Changes:** any change after the tag is listed in the "Deviations" section at the end, with date and reason. This file is never edited silently.

In this file, **reassuring evidence** points to the safe outcome (survival, no sepsis, no readmission) and **alarming evidence** to the alarming outcome. The asymmetry statistics a and a′ are those of Deviation D1 of the first study. a < 0 means that reassuring evidence moved the model more than alarming evidence.

## 1. What is known

- **The split.** Each patient is asked about the alarming outcome ("Will this patient die during the current hospital stay?", the alarm framing) and about the safe outcome ("Will this patient survive the current hospital stay?", the reversed framing). For a statistic s (a or a′) computed in each framing, the **outcome part** is O = (s_alarm + s_reversed) / 2 and the **answer-polarity part** is W = (s_alarm − s_reversed) / 2. W < 0 means that evidence moved the model more readily toward the answer "No", whatever outcome "No" stood for. In the third study W kept its sign with every changed card (R6 supported). Across the 37 valid cells of post-trained checkpoints in the three studies and the scale checks, W was conclusive toward "No" in 20 cells and toward "Yes" in 6 (Falcon3-7B-Instruct, Yi-1.5-9B-Chat-16K, and Qwen2.5-7B-Instruct on PhysioNet 2012). O was conclusive toward the safe outcome in 12 cells and toward the alarming outcome in 9.
- **What W cannot separate.** The reversed question changes three things at once: (i) which answer word stands for the alarming outcome, (ii) the outcome that the question names (death or survival), and (iii) the wording ("survive" is an antonym; "stay free of sepsis" and "avoid readmission" are implicit negations). A preference for the token "No", a tendency to deny rather than affirm a stated outcome, and an effect of the named outcome all enter W. In the paper W was therefore renamed from "answer-word part" to "answer-polarity part".
- **Exploratory analyses after the third study (in the paper, not registered).** Without a card, the models leaned toward "No" in 35 of the 37 cells, and this static lean was correlated with W across cells (Spearman 0.74). A simulation of both framings with a static answer lean and symmetric updating showed that the joint test of W kept its false-positive rate at or below 5% in all 120 null settings, while the test of O did not (up to 53% toward the safe outcome).
- **Post-training or prompt format.** Base models had |W| ≤ 0.33, but they were prompted as plain text followed by "Answer:", while every post-trained checkpoint was prompted through its chat template. The plain-text control of the second study (Olmo-3-7B-Instruct) gave a median Yes+No mass below 0.02 and could not be interpreted. The paper's statement that W belongs to post-trained checkpoints therefore cannot say whether post-training or the chat template produces it.
- **Example order.** The order control of the second study (solved examples ending with an alarming case) ran in the alarm framing only, so it says nothing about O.

## 2. Research questions

1. **R8.** Does W keep its sign when the answer words are "True" and "False" instead of "Yes" and "No"?
2. **R9.** Does W keep its sign when the safe answer is "Yes" because the question is negated ("Will this patient not die …?") instead of reversed?
3. **R10.** With the prompt format held fixed, does post-training change W? With the checkpoint held fixed, does the chat template change it?
4. **R11.** Do W and O keep their signs when the solved examples end with an alarming case, in both framings?
5. **R12.** Do W and O keep their signs with a paraphrased prompt?
6. **R13.** Does a one-sentence card that states the combined direction in words move the models more than one that states the same combined effect as a number?
7. **Q14.** Descriptive questions without a decision rule (Section 7).

R8 and R9 together separate three readings of W (Section 7, "Joint reading").

## 3. What is unchanged from the earlier studies

Everything below is as registered in `PREREGISTRATION.md` (with Deviations D1 to D5), `PREREGISTRATION_V2.md` and `PREREGISTRATION_V3.md`, and is implemented by the same code: `ler_main.py`, version main-2.0 (SHA-256 0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e), and `ler_v3.py`, version v3-1.1 (SHA-256 89d4c0c0ed995e82aa472eeac3f22ea0728583c7c8c84a01c87befe18f5e92ec), both byte-identical to the third study's files (every run checks this). Everything new is in `ler_v4.py`.

- The three datasets, the splits (seed 2027), the eligibility rule, the XGBoost model, the 16 features, the cards (top 5 SHAP values), the conditions C0 (no card), C2 (the patient's true card) and C4 (the same card with every sign flipped), the solved examples of set A and their order "Asf" (the example-order control of the second study), the Yes/No prompts in the alarm and the reversed framing, the one-sentence cue of the third study, and the plain-text format of the base models (text followed by a new line and "Answer:").
- The **same test patients**: 300 per class per dataset, 1,800 in total.
- The reading: the next-token log-probabilities after "Answer:", from the top 1,000 log-probabilities with the imputation rule of Deviation D5. The exact-reading check of Deviation D2 still runs and is recorded. Every run uses vLLM 0.26.0 in float16 on two T4 GPUs; there is no fallback to another backend.
- The statistics a and a′, the intersection–union test, the two-sided bootstrap p-value of the first study, the validity thresholds (median answer-word mass ≥ 0.5, C0 accuracy ≥ 0.55) and the treatment of invalid readings (Deviation D3, item 5).
- No system message is added. A chat template's own default system text is kept and logged.

Not used in this study: condition C6, example sets B and C, the answer options of the second study, the words, numbers and matched cards of the third study, and the memorization check (every model of this study was checked in an earlier study).

## 4. Models

| Model | Kind | Role | Prompts per patient | Revision |
|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | instruct | R8, R9, R11 to R13; R10 (first family) | 31, and 6 in plain text | 0e9e39f249a16976918f6564b8830bc894c89659 |
| Qwen/Qwen2.5-7B-Instruct | instruct | R8, R9, R11 to R13; R10 (second family) | 31, and 6 in plain text | a09a35458c702b33eeacc393d103063234e8bc28 |
| allenai/Olmo-3-7B-Instruct | instruct | R8, R9, R11 to R13 | 31 | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc |
| tiiuae/Falcon3-7B-Instruct | instruct | R8, R9, R11 to R13 | 31 | 1e57a0ecd176c7c139f289c60a74e57f887c3dfb |
| 01-ai/Yi-1.5-9B-Chat-16K | instruct | R8, R9, R11 to R13 | 31 | 2f8753bed95bd4fede3fd64f8305428620b4b212 |
| meta-llama/Llama-3.1-8B | base | R10 (first family) | 12 | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b |
| Qwen/Qwen2.5-7B | base | R10 (second family) | 12 | d149729398750b98c0af14eb82c78cfe92750796 |
| allenai/Olmo-3-1025-7B | base | backup family of R10; not run (not eligible, Section 10) | – | a81bae42db3975be1671e27b9c9a56da1a9f980f |

- **The instruction-tuned models** are the five models of the third study whose cells hold the pre-listed W and O cells of Section 7. Two pull toward "No" (Llama 3.1, Olmo 3), two toward "Yes" (Falcon3, Yi-1.5), and Qwen2.5 has one conclusive cell in each direction, so every test below has eligible cells of both signs. Phi-4-mini-instruct, the sixth model of the third study, is not included: 52% of its readings had an imputed answer word under the top-1,000 reading, and under the imputation bound three of its conclusive readings were no longer conclusive (Deviation V3-D1 of the third study).
- **The base models** are the base checkpoints of the first study (Llama 3.1, Qwen2.5) and of the second study (Olmo 3), with the revisions used there. They are run only for R10. R10 uses the first two families, in the order Llama 3.1, Qwen2.5, Olmo 3, whose base model passes the engineering check in the chat format (Section 10). These are Llama 3.1 and Qwen2.5; Olmo-3-1025-7B did not pass, so it is not run.
- **Model window.** As in the third study: 16,384 tokens, or the model's own maximum length if that is shorter.
- **If a model cannot be run after registration.** As in the third study: a model "cannot be run" only if vLLM 0.26.0 fails to load it, or if more than 20% of its first 256 or more readings are invalid. Its tests then count it as missing (p = 1), and this is recorded as a deviation. No model is replaced after registration; if a base model cannot be run, R10 is reported with the families that remain.

## 5. Prompt formats

Every new format changes only the lines named below; every other line of the prompt (task line, solved examples with their patient blocks, the patient block, the evidence block) is the registered text of the same patient, condition and example set. The run notebook checks this for every prompt. `{statement}`, `{q}` and the other fields are filled per dataset from the tables below; `Answer:` is always the last line of each solved example's answer and is prefilled after the prompt, as in every earlier study.

### 5.1 True/False statements (R8)

The question line of the registered prompt is replaced by a statement, and the answer words are "True" and "False". The statements are the answer-option sentences of the second study, word for word.

- Each solved example ends with `Statement: {statement}` and `Answer: True` or `Answer: False`.
- The prompt ends with `Statement: {statement}` and the line `Is this statement true or false? Answer with True or False.`
- **tf_alarm:** the statement names the alarming outcome, so "True" stands for it. **tf_reversed:** the statement names the safe outcome, so "True" stands for that.

| Dataset | Statement, tf_alarm | Statement, tf_reversed |
|---|---|---|
| PhysioNet 2012 | The patient dies during the current hospital stay. | The patient survives the current hospital stay. |
| PhysioNet 2019 | The patient develops sepsis later in this ICU stay. | The patient stays free of sepsis for the rest of this ICU stay. |
| Diabetes 130 | The patient is readmitted within 30 days of discharge. | The patient avoids readmission within 30 days of discharge. |

### 5.2 The negated question (R9)

The registered alarm question with "not" after "Will this patient", in the solved examples and at the end; the answer instruction is the registered "Answer with Yes or No." Here "Yes" stands for the safe outcome, as in the reversed framing, but the question names the alarming outcome, as in the alarm framing.

| Dataset | Negated question |
|---|---|
| PhysioNet 2012 | Will this patient not die during the current hospital stay? |
| PhysioNet 2019 | Will this patient not develop sepsis later in this ICU stay? |
| Diabetes 130 | Will this patient not be readmitted within 30 days of discharge? |

### 5.3 Alarming-last example order in both framings (R11)

The solved examples of set A in the order safe, alarming, safe, alarming ("Asf", the order control of the second study), with the registered question of the alarm framing and, new in this study, of the reversed framing. With set A, the last solved example is a safe case; with Asf it is an alarming case. In both framings, Asf therefore turns a pull toward repeating the last example's outcome from the safe to the alarming outcome.

### 5.4 Paraphrased prompt (R12)

The task line, the header of the solved examples, the label of each solved example, the header of the patient, both questions and the answer instruction are paraphrased; the patient blocks, the evidence block and `Answer:` are unchanged.

| Line | Registered | Paraphrased |
|---|---|---|
| example header | Here are 4 example patients with known outcomes: | Four earlier cases and what actually happened to them: |
| example label | Example {k} | Case {k} |
| patient header | Now the patient to predict: | The case to assess: |
| answer instruction | Answer with Yes or No. | Reply with Yes or No. |

| Dataset | Paraphrased task (after `Task: `) |
|---|---|
| PhysioNet 2012 | estimate whether a patient in the intensive care unit (ICU) will die before leaving the hospital, using the measurements recorded in the first 48 hours of the ICU stay. |
| PhysioNet 2019 | estimate whether a patient in the intensive care unit (ICU) will develop sepsis after the first 24 hours of the ICU stay, using the measurements recorded in those 24 hours. |
| Diabetes 130 | estimate whether a patient with diabetes who was treated in hospital will be admitted to hospital again within 30 days of going home, using the information recorded during the stay. |

| Dataset | para_alarm | para_reversed |
|---|---|---|
| PhysioNet 2012 | Is this patient going to die before leaving the hospital? | Is this patient going to leave the hospital alive? |
| PhysioNet 2019 | Is this patient going to get sepsis later in this ICU stay? | Is this patient going to remain sepsis-free for the rest of this ICU stay? |
| Diabetes 130 | Is this patient going to be admitted again within 30 days of going home? | Is this patient going to stay out of hospital for the 30 days after going home? |

### 5.5 The net-number card (R13)

One sentence in place of the whole evidence block, parallel to the cue of the third study but with the combined effect stated as a number and without a direction word:

```
Evidence from a machine-learning model (XGBoost) trained on other patients: taken together, the {n} factors that most influenced its risk estimate for this patient have a combined impact of {impact} on the risk of {outcome}.
```

`{n}` is 5, `{outcome}` as on the registered card, and `{impact}` is the sum of the five shown SHAP values with two decimals and a sign ("+0.71", "-0.85"). The cue of the third study is scored again in the same run:

```
Evidence from a machine-learning model (XGBoost) trained on other patients: taken together, the {n} factors that most influenced its risk estimate for this patient {verb} the risk of {outcome}.
```

Both sentences point the same way as the card they replace (the card direction of every study: the sign of the sum). If a sum rounds to "+0.00" or "-0.00", the sign printed is still the sign of the sum; the number of such patients is reported.

### 5.6 Prompt format of the checkpoint (R10)

- **plain:** the text, a new line and `Answer:`, preceded by the tokenizer's beginning-of-sequence token if it has one. This is the format of the base models in the first and second studies (`ler_main.render` with kind "base").
- **chat:** the text as the only user message of the instruction-tuned model's chat template, with the generation prompt, followed by `Answer:`. This is the format of every post-trained checkpoint in the earlier studies (`ler_main.render` with kind "instruct"). For a base model, the template and tokenizer of the instruction-tuned model of the same family are used; the engineering check verifies that the base model's tokenizer turns the rendered text into exactly the same token ids.

The four cells of a family are B·plain (the base model as in the earlier studies), B·chat (new), I·chat (the instruction-tuned model as in the earlier studies) and I·plain (the plain-text control of the second study, run only if it passes the engineering check, Section 10).

### 5.7 Generated answers (Q14d)

For a check of the first-token reading, each instruction-tuned model also generates text with greedy decoding (temperature 0) for a subsample: 50 test patients per class per dataset, drawn without replacement from the sorted patient ids of each class (the alarming class first, then the safe class, with one generator per dataset, seed 2031), × C0, C2 and C4 × the alarm and the reversed framing with the registered card (1,800 prompts per model), in two modes:

- **prefill:** the registered rendered prompt, ending with `Answer:`; at most 8 new tokens.
- **free:** the same prompt through the chat template without `Answer:`; at most 512 new tokens (64 in the first and 256 in the second engineering check, Section 10).

The answer is read from the text by a fixed rule: after removing the characters `*`, `_`, `` ` ``, `#` and `>`, the text's first word if it is "yes" or "no" (any case); otherwise the first "yes" or "no" that follows the word "answer" (with "is", ":" or both in between, or nothing); otherwise the answer is "unparsed". No generated text is used in any test.

## 6. Prompts

Per patient, for each instruction-tuned model:

- **registered:** C0, C2 and C4 × the alarm and the reversed framing, registered card (6 prompts; the prompts of the earlier studies);
- **True/False:** C0, C2 and C4 × tf_alarm and tf_reversed (6);
- **negated:** C0, C2 and C4 × the negated question (3);
- **order:** C0, C2 and C4 × both framings × Asf (6);
- **paraphrase:** C0, C2 and C4 × para_alarm and para_reversed (6);
- **aggregate cards:** C2 and C4 × the alarm framing × the cue and the net-number card (4).

That is 31 prompts per patient (55,800 per model), plus 6 per patient in plain text for Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, whose I·plain cells are run (10,800 each), plus 3,600 generated answers. Each base model gets C0, C2 and C4 × both framings × plain and chat (12 prompts per patient, 21,600 per model). With five instruction-tuned and two base models, the study scores 343,800 prompts.

C0 has no card, so each framing has its own C0 prompts. The cue and the net-number card use the C0 reading of the registered alarm framing.

## 7. Hypotheses and tests

**Common to all tests.** A cell is one model and one dataset. For each framing and format, a and a′ are computed exactly as in Deviation D1 of the first study, from the C0 reading and the C2 and C4 readings of that framing and format. A **pair** is two framings of one format in which the first answer word ("Yes" or "True") stands for the alarming outcome in the first framing and for the safe outcome in the second. For a pair, O = (s_first + s_second) / 2 and W = (s_first − s_second) / 2, for s = a and s = a′; W < 0 means that evidence moved the model more readily toward the second answer word ("No" or "False"). The pairs:

| Pair | First framing | Second framing |
|---|---|---|
| registered | alarm, set A | reversed, set A |
| True/False | tf_alarm | tf_reversed |
| negated | alarm, set A | negated |
| order | alarm, Asf | reversed, Asf |
| paraphrase | para_alarm | para_reversed |
| plain, chat (R10) | alarm in that format | reversed in that format |

Each comparison uses the patients with valid C0, C2 and C4 readings in every framing it involves, and one paired bootstrap over those patients (10,000 resamples, seed 2026), with the same resamples for every statistic of the comparison. Comparisons across two runs (R10) pair the readings by patient. The p-value of O, W or a change in them is the intersection–union p: the larger of the two-sided bootstrap p-values of its two versions, and a direction is stated only when both versions have the same sign.

**Eligible cells.** The tests of R8, R9, R11 and R12 use the cells in which a part was established with the registered card before this study: the eligible cells of R6 (for W) and R7 (for O) of the third study, without those of Phi-4-mini, each with its sign.

- **W, 11 cells:** Qwen2.5-7B-Instruct on PhysioNet 2012 (W > 0) and on PhysioNet 2019 (W < 0); Llama-3.1-8B-Instruct on PhysioNet 2012 and on Diabetes 130 (W < 0); Olmo-3-7B-Instruct on PhysioNet 2012 and on PhysioNet 2019 (W < 0); Falcon3-7B-Instruct on all three datasets (W > 0); Yi-1.5-9B-Chat-16K on PhysioNet 2012 and on PhysioNet 2019 (W > 0).
- **O, 7 cells:** Qwen2.5-7B-Instruct on PhysioNet 2012 (O < 0); Llama-3.1-8B-Instruct on PhysioNet 2012 and on PhysioNet 2019 (O < 0); Olmo-3-7B-Instruct on PhysioNet 2012 (O > 0); Falcon3-7B-Instruct on PhysioNet 2019 and on Diabetes 130 (O > 0); Yi-1.5-9B-Chat-16K on PhysioNet 2019 (O < 0).

**Kept and removed.** For a format and a part with its eligible cells:

- In a cell, the part is **supported** with the format if the cell and the format are valid (Section 8), both versions of the part have the eligible sign, and the Holm-adjusted p (over the eligible cells) is below 0.05. An invalid or missing cell enters with p = 1.
- In a cell, the part is **reduced** with the format if the change Δ = part(format) − part(registered) has the opposite of the eligible sign in both versions and its Holm-adjusted p (over the eligible cells) is below 0.05.
- The part is **kept** with the format if it is supported in more than half of the eligible cells (at least 6 of 11 for W, at least 4 of 7 for O). It is **removed** if it is not kept and is reduced in more than half of the eligible cells. Otherwise the result is **not settled**.

### R8. The answer-polarity part with True/False answers

- **Prediction.** W is kept with the True/False pair.
- **Also reported:** whether O is kept with the True/False pair (O, 7 cells), with the same rules.

### R9. The answer-polarity part with the negated question

- **Prediction.** W is kept with the negated pair (the alarm question against the negated question).
- **Also reported:** O from the negated pair, and the **wording contrast** D = (s_reversed − s_negated) / 2. Both framings of D have "Yes" for the safe outcome, so D compares the reversed wording with the negated wording at a fixed answer mapping.

### Joint reading of R8 and R9

The three explanations of W in Section 1 make different predictions. A preference tied to the words "Yes" and "No" predicts that W is removed with True/False and kept with the negated question. A preference tied to the outcome the question names predicts that W is kept with True/False (the statement names the outcome as the question did) and removed with the negated question (both questions of that pair name the alarming outcome). A preference for denying rather than affirming the stated outcome, whatever the words, predicts that W is kept with both. The registered reading:

| True/False | Negated | Reading |
|---|---|---|
| kept | kept | **answer polarity**: W follows whether the answer affirms or denies the stated outcome, not the words "Yes" and "No" and not the outcome named |
| removed | kept | **the answer words**: W is tied to "Yes" and "No" |
| kept | removed | **the named outcome**: W follows which outcome the question names |
| removed | removed | **the Yes/No question as worded**: neither the answer words nor the named outcome alone carries it |
| any other combination | | **not settled by this study** |

### R10. Post-training and the chat template

- **Families and eligible cells.** Each family run for R10 contributes its pre-listed W cells: Llama 3.1 on PhysioNet 2012 and on Diabetes 130 (W < 0); Qwen2.5 on PhysioNet 2012 (W > 0) and on PhysioNet 2019 (W < 0); for Olmo 3, if it is used, PhysioNet 2012 and PhysioNet 2019 (W < 0).
- **Contrasts.** In each eligible cell, with W from the registered pair in each format: **P** = W(I·chat) − W(B·chat), the effect of post-training with the chat format held fixed, and **F** = W(B·chat) − W(B·plain), the effect of the chat format with the base checkpoint held fixed. P + F is the difference between the instruction-tuned and the base model that the earlier studies compared. If the I·plain cell is run: P′ = W(I·plain) − W(B·plain) and F′ = W(I·chat) − W(I·plain).
- **Prediction.** P has the eligible sign. Test: intersection–union test of P in the cell's direction, Holm over the eligible cells (one family); a cell enters the test of P only if I·chat and B·chat are valid in it, and the test of F only if B·chat and B·plain are valid. **R10 is supported** if P is supported in more than half of the eligible cells.
- **Also tested, with no direction predicted:** F, two-sided, Holm over the same cells; F is "in the direction of W" in a cell if both versions have the eligible sign and its Holm-adjusted p is below 0.05.
- **Known before registration.** In the first study, Llama-3.1-8B in the plain format failed the validity rule on Diabetes 130 (C0 accuracy 0.545 against the threshold of 0.55). If B·plain fails it again, F cannot be tested in that cell and enters with p = 1, so the reading "the chat template produces the part" then needs F in the direction of W in all three other cells.
- **Interpretation rule.** R10 supported and F in the direction of W in no eligible cell: "post-training produces the part, not the chat template". R10 not supported and F in the direction of W in more than half of the eligible cells: "the chat template produces the part, not post-training". R10 supported and F in the direction of W in more than half: "both contribute". Otherwise: "not separated by this study". P′ and F′ are reported without a decision rule.

### R11. Example order

- **Prediction.** W (11 cells) and O (7 cells) are both kept with the order pair.
- **Interpretation.** If O is removed with the order pair, O is described as depending on the order of the solved examples. If O is kept, it is described as not produced by the outcome of the last solved example.

### R12. Paraphrased prompt

- **Prediction.** W (11 cells) and O (7 cells) are both kept with the paraphrase pair.

### R13. Direction words and numbers at the level of the combined effect

- **Cells.** Every cell of the five instruction-tuned models (15 cells).
- **Statistic.** In the alarm framing, the mean movement m = mean(la − ls) between the alarm-pointing and the safe-pointing version of a card (log-odds of the alarming outcome), with the cue and with the net-number card, on the patients with valid C2 and C4 readings with the registered card, the cue and the net-number card (the movement with the registered card is reported alongside); Δm = m(cue) − m(net-number).
- **Prediction.** Δm > 0. Test: two-sided bootstrap p of Δm, Holm over the 15 cells; a cell that is invalid for either card enters with p = 1. **Supported** if Δm > 0 with Holm-adjusted p < 0.05 in more than half of the valid cells.
- This compares words and numbers when both state only the combined effect in one sentence. The third study compared them at the level of the five factors (the words card and the numbers card); together they cover both levels.

### Selection of eligible cells

The eligible cells were selected on the registered-card readings of the earlier studies, and this study scores the same registered prompts again. Selected cells tend to be those whose part was estimated as large, so a change from the registered card can look like a reduction partly by regression to the mean. The patients and the registered readings are the same in both measurements, which limits this, but it is why "kept" rests on the support test, which asks only whether the part keeps its sign.

### Q14. Descriptive questions (no decision rules)

- **a. Replay.** The registered prompts are compared with the same prompts in the earlier studies: the third study for the five instruction-tuned models, the first and second studies for the base models in the plain format, and the second study for the alarm framing with Asf. Reported: the largest and median absolute difference in the log-odds of the alarming outcome, and the share of identical answers.
- **b.** O and W from every pair in every cell, with intervals, and the wording contrast D (R9).
- **c. Static answer lean.** For every pair and cell, the median over patients of the C0 log-odds of the first answer word averaged over the two framings. Reported: whether the sign of W matches the sign of the lean in the cells where W is conclusive, and the Spearman correlation of the lean and W over the valid cells, for each pair. The exploratory expectation from Section 1 is that W follows the lean.
- **d. Generated answers.** For each model and mode: the share of parsed answers, the agreement of the parsed answer with the first-token reading (Yes if the log-odds of "Yes" against "No" is above 0), and the decision-level difference E_W = (E_alarm − E_reversed) / 2, where E is the added false alarms minus the added missed alarms in one framing at the model's own threshold, from the generated answers and from the first-token reading on the same prompts (patients with all their answers parsed).
- **e.** For every format: accuracy by condition, the share of C2 and C4 answers that follow the card, the movement, the median answer-word mass and the share of imputed readings.
- **f. Imputation bound.** For every cell with imputed readings, the results of R8 to R13 recomputed with the bound of Deviation D5 (in each record with an imputed variant, the probability of the less likely answer word is halved; 2,000 resamples). This time it is part of the registered analysis notebook.
- **g.** W, O and the movement in all four cells of each R10 family, and P′ and F′.

## 8. Analysis

- **Validity of a cell.** As in the third study, with the registered card: median Yes+No mass ≥ 0.5 over the alarm-framing records (C0, C2, C4), median mass ≥ 0.5 over the reversed-framing records (C0, C2, C4), and C0 accuracy ≥ 0.55 in the alarm framing. A record without a usable reading counts as mass 0.
- **Validity of a new framing in a valid cell.** (i) The median answer-word mass over its records is ≥ 0.5; (ii) the mean movement mean(la − ls) on the log-odds of the alarming outcome is above 0, that is, the cards move the answer the way they point; (iii) the C0 log-odds of the alarming outcome have a positive Spearman correlation with those of the registered alarm question over the cell's patients. Criteria (ii) and (iii) show that the model reads the question or statement the right way round, which matters most for the negated question. A pair is valid if both its framings are. For the cue and the net-number card, criterion (i) applies to the C2 and C4 records of each card.
- **Validity of an R10 cell.** The cell rule above, applied to the records of that checkpoint and format.
- **Estimators.** a and a′ as in `ler_main.py`, computed for many resamples at once with the weighted least-squares functions of `ler_v3.py` (`stat_a`, `stat_ap`), exactly as in the third study. Each comparison starts its own random generator with seed 2026.
- **Code.** `ler_main.py` (main-2.0) and `ler_v3.py` (v3-1.1), both unchanged, `ler_v4.py` (v4-1.2) and the notebooks `llm_evidence_reliance_engcheck_v4.ipynb`, `llm_evidence_reliance_main_run_v4.ipynb` (one run per model) and `llm_evidence_reliance_analysis_v4.ipynb` are committed with this file, at the tag `prereg-v4`, before any run. Each run checks `ler_main.py` and `ler_v3.py` against the hashes above, checks the four protocol files at the tag (every prompt string, model id and pinned revision in the code must appear in them) and stops otherwise. Each run downloads exactly the pinned revision and stops if it differs.
- **Families.** R8, R9, R10, R11, R12 and R13 are separate confirmatory questions, each with its own Holm correction (one family per part and pair), decision rule and α = 0.05.
- **One analysis.** The analysis notebook is run once, after all runs, on the run outputs, and (for Q14a only) on the earlier runs from `results/`. Before it analyses anything, it checks that all runs used the same code, protocol text, patients, cards and solved examples, that every prompt shared by two runs has the same text, that each run is complete, and that each model's revision is the registered one. A missing run enters its tests with p = 1. The output and the run outputs are committed to `results/v4/`.
- **Changes after the tag.** Only engineering settings that do not touch prompts, readings, models or analysis may change without a deviation. Any other change is recorded as a dated deviation, and the tests it affects are marked in the report.

## 9. Order of work

1. Engineering check (Section 10), before this file is tagged. Done in two checks.
2. Record its result in Section 10 (which formats each model runs, which R10 families are used), commit it with the engineering-check outputs, and create the release `prereg-v4`.
3. One run per model, one Kaggle session each, in this order: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Olmo-3-7B-Instruct, Falcon3-7B-Instruct, Yi-1.5-9B-Chat-16K, then Llama-3.1-8B and Qwen2.5-7B, the base models of R10. In the third study's runs, scoring took 0.16 seconds per prompt for the 7 to 8 billion parameter models, 0.18 for Falcon3 and 0.31 for Yi-1.5. With installation, data, loading and the generated answers, a run is expected to take about 3 to 4 hours for each of the first four models, about 5.5 hours for Yi-1.5 and about 1.5 hours for each base model: about 23 GPU hours in total.
4. One run of the analysis notebook.

## 10. What was known before registration, and the engineering check

- **Known:** all results of the three studies and of the scale checks, and the exploratory analyses of Section 1.
- **Engineering check.** Each model of Section 4 is downloaded and run on dev-sample prompts only (never test patients): per dataset, 6 dev patients × every prompt of its grid (Section 6), and for the instruction-tuned models also the 6 plain-text prompts; a timing set of 100 prompts per dataset; for the instruction-tuned models, 12 generated answers per dataset in each mode; and two True/False questions with an obvious answer ("Statement: The sky is usually blue on a clear day." and "Statement: Fire is usually cold to the touch.", each followed by the True/False instruction), in addition to the four obvious questions of the second study. The check records only technical quantities: whether the model loads in float16 with vLLM 0.26.0 on two T4 GPUs, the model window, the chat template and its default system text, whether "Yes", "No", "True" and "False" have single-token variants, the exact-reading check of Deviation D2 for both pairs of answer words, whether every reading is finite, the median answer-word mass in every framing and format, the share of imputed readings, prompt lengths, the time per prompt, and, for the generated answers, the time and the share of parsed answers. For a base model in the chat format, it also checks that its tokenizer gives exactly the token ids of the instruction-tuned tokenizer for every rendered prompt. It computes no accuracy, no answer direction, no agreement of generated answers with the reading, and no asymmetry, and it saves no per-prompt reading.
- **Rules applied to the check, set before it is run.** (i) A format of an instruction-tuned model whose median mass is below 0.5 in either of its framings, or whose answer words have no single-token variant, is not run for that model, and its cells enter the tests with p = 1. (ii) The I·plain cell of an R10 family is run if both of its framings reach a median mass of 0.5. (iii) A base model is used for R10 if it loads, every reading is finite, the median mass is at least 0.5 in both framings in both formats, and the token ids in the chat format are identical to those of the instruction-tuned tokenizer; the first two families in the order of Section 4 that meet this are used. (iv) An instruction-tuned model that fails the obvious questions in a format is reported; nothing is changed for it, because the obvious questions are not part of the validity rules.
- **First check** (2026-11-02, Kaggle, two T4 GPUs, 68 minutes; 1,110 dev prompts; `ler_v4.py` v4-1.0, SHA-256 5ef560a98d3045139bf6d4d3d98ac47615d784b5ff4e9ec7341367890015f41c). Median answer-word mass on the check prompts of each framing and format (alarm / reversed where a format has two framings), and the share of readings with an imputed variant on the timing prompts:

| Model | Registered | True/False | Negated | Asf | Paraphrase | Cue / net | Plain or chat | Imputed | Note |
|---|---|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | – | – | – | – | – | – | – | – | not downloaded: Hugging Face refused access to the gated repository (403); second check |
| Qwen/Qwen2.5-7B-Instruct | 0.995 / 1.000 | 0.999 / 1.000 | 1.000 | 0.997 / 0.998 | 1.000 / 1.000 | 0.999 / 0.990 | plain 0.968 / 0.989 | 15% | generated answers, free mode: 0 of 36 parsed (cut at 64 tokens before an answer); second check |
| allenai/Olmo-3-7B-Instruct | 0.997 / 0.996 | 0.999 / 0.999 | 0.998 | 0.997 / 0.996 | 0.997 / 0.997 | 0.999 / 0.999 | plain 0.016 / 0.018 | 0% | |
| tiiuae/Falcon3-7B-Instruct | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.998 | plain 0.313 / 0.490 | 0% | |
| 01-ai/Yi-1.5-9B-Chat-16K | 0.999 / 0.999 | 1.000 / 0.999 | 0.999 | 0.999 / 0.999 | 0.998 / 0.998 | 1.000 / 0.999 | plain 0.992 / 0.990 | 0% | |
| meta-llama/Llama-3.1-8B | – | | | | | | – | – | not downloaded (403); second check |
| Qwen/Qwen2.5-7B | plain 0.879 / 0.896 | | | | | | chat 0.724 / 0.748 | 0% | chat token ids identical in 108 of 108 prompts: eligible for R10 |
| allenai/Olmo-3-1025-7B | plain 0.761 / 0.720 | | | | | | chat 0.781 / 0.788 | 0% | chat token ids identical in 0 of 108 prompts: not eligible for R10 |

Every reading of the models that were downloaded was finite, every model has single-token variants of "Yes", "No", "True" and "False", and the four instruction-tuned models answered the six obvious questions the right way round. The exact-reading check of Deviation D2 gave largest differences of 0.047 and 0.031 (Qwen2.5, Yes/No and True/False), 0.015 and 0.016 (Olmo 3), 0.010 and 0.014 (Falcon3), 0.016 and 0.030 (Yi-1.5), 0.015 (Qwen2.5 base) and 0.004 (Olmo 3 base). The longest prompts have between 2,097 and 2,297 tokens. In free mode, Olmo 3, Falcon3 and Yi-1.5 began every answer with "Yes" or "No" (36 of 36 parsed); Qwen2.5 began every answer with a sentence such as "Based on the provided data and the patterns observed in the examples," and reached no answer within 64 tokens. In prefill mode every answer was parsed. The time per prompt on the timing prompts was 0.38 to 0.45 seconds for the 7 to 8 billion parameter models and 0.67 for Yi-1.5, about twice the time per prompt of the third study's runs, as in that study's check.

- **Changes after the first check, before the second.** (i) Free mode generates at most 256 new tokens instead of 64, so that a model that reasons before answering can reach its answer; the parsing rule is unchanged. (ii) The plain format is run only for an instruction-tuned model of an R10 family, as rule (ii) says; the first version of `decisions_from_check` also listed models outside the R10 families, which the run notebook never used. (iii) The notebooks stop at the start if Hugging Face refuses the download of a gated model. These are `ler_v4.py` v4-1.1. No prompt, reading, model, test or decision rule changed.
- **Second check** (2026-11-02, Kaggle, two T4 GPUs, 36 minutes; the same 1,110 dev prompts; `ler_v4.py` v4-1.1, SHA-256 c2ed9d95bde5906fc5e0fc3bd94e205dbb9014581394c3b2020694a665268cdd). The two Llama models, after access to their gated repositories was restored, and Qwen2.5-7B-Instruct, for free mode with 256 tokens:

| Model | Registered | True/False | Negated | Asf | Paraphrase | Cue / net | Plain or chat | Imputed | Note |
|---|---|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | 0.989 / 0.992 | 0.987 / 0.988 | 0.992 | 0.991 / 0.990 | 0.991 / 0.994 | 0.990 / 0.988 | plain 0.823 / 0.856 | 0% | generated answers, free mode: 10 of 36 parsed |
| Qwen/Qwen2.5-7B-Instruct | 0.995 / 1.000 | 0.999 / 1.000 | 1.000 | 0.997 / 0.998 | 1.000 / 1.000 | 0.999 / 0.990 | plain 0.968 / 0.989 | 15% | generated answers, free mode: 16 of 36 parsed |
| meta-llama/Llama-3.1-8B | plain 0.738 / 0.761 | | | | | | chat 0.654 / 0.670 | 0% | chat token ids identical in 108 of 108 prompts: eligible for R10 |

Every median mass of Qwen2.5-7B-Instruct equals that of the first check to three decimals. Every reading was finite; Llama-3.1-8B-Instruct has single-token variants of all four answer words, answered the six obvious questions the right way round, and gave largest exact-reading differences of 0.016 (Yes/No) and 0.014 (True/False); Llama-3.1-8B gave 0.011. Its chat template adds the text "Cutting Knowledge Date: December 2023 Today Date: 26 Jul 2024", as in the first study. In free mode, 26 of the 36 answers of Llama 3.1 and 20 of the 36 answers of Qwen2.5 could not be parsed after 256 tokens; the three answers of each model that the output keeps begin with a step-by-step analysis ("To predict in-hospital mortality for the given patient, we need to analyze …", "Based on the provided data and the patterns observed in the examples, …").

- **Change after the second check.** Free mode generates at most 512 new tokens, so that more of the answers of models that reason before answering are complete; the parsing rule is unchanged, and this setting was not checked again. Generated answers enter no test (Q14d reports them, with the share parsed). The registered `ler_v4.py` (v4-1.2) differs from v4-1.1 only by this limit, the decisions below, the hashes of the two outputs and of the two earlier versions (which the protocol check now requires in this file), a function that merges the results of the two checks, and its version string.
- **Decisions** (rules (i) to (iii), on the results of both checks, with the second check replacing the first for the models it repeated). (i) No format is dropped: every instruction-tuned model runs all 31 prompts per patient. (ii) The I·plain cells of Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct are run. (iii) The R10 families are Llama 3.1 and Qwen2.5; Olmo-3-1025-7B is not eligible and is not run. (iv) No model failed an obvious question in a format it is run with. These decisions are `ENGCHECK_DECISIONS` in `ler_v4.py`, which the run notebook uses; the tests check that the rules applied to the two outputs give exactly these decisions.
- **Outputs.** Both outputs are committed as `results/v4/engcheck_v4.zip` (first check, SHA-256 805ebbac4f896f50f2e00ddccf4c14585373da7d63e3dcd3cb7a04150f610e7f) and `results/v4/engcheck_v4_run2.zip` (second check, SHA-256 c1ed543c5c39ca5b4cb1e5d3e61014c21a108f39392883864a6d680c099efbc1). Each contains the `ler_v4.py` that produced it.

## References

- First study: `PREREGISTRATION.md`, releases `prereg-v1` to `prereg-v1.3`; results in `results/main/`.
- Second study: `PREREGISTRATION_V2.md`, release `prereg-v2`; results in `results/v2/`.
- Third study: `PREREGISTRATION_V3.md`, release `prereg-v3`; results in `results/v3/`.

## Deviations

None so far.
