# Pre-registration, third study: new model families, and which part of the evidence card drives the two asymmetries

- **Author:** Jhishan Deb
- **Registered:** October 2026, committed and tagged `prereg-v3` before any test prompt of this study was scored (the GitHub commit and the `prereg-v3` release date are the authoritative timestamps)
- **Builds on:** the first study (`PREREGISTRATION.md`, releases `prereg-v1` to `prereg-v1.3`, results in `results/main/`), the second study (`PREREGISTRATION_V2.md`, release `prereg-v2`, results in `results/v2/`) and the two exploratory scale checks (`SCALE_CHECK.md`, `SCALE_CHECK_2.md`, results in `results/scale/` and `results/scale2/`)
- **Status:** written after the results of both studies, and of an exploratory analysis of them, were known (Section 1). They motivate every prediction below. The prompts with the registered card are the prompts of the earlier studies, so their readings for the three earlier models of this study are known. No prompt with a changed card (Section 5) has been scored on a test patient. Changed cards were scored only on dev patients, in the engineering check of Section 10, and only the technical quantities listed there were computed from them.
- **Changes:** any change after the tag is listed in the "Deviations" section at the end, with date and reason. This file is never edited silently.

In this file, **reassuring evidence** points to the safe outcome (survival, no sepsis, no readmission) and **alarming evidence** to the alarming outcome. The asymmetry statistics a and a′ are those of Deviation D1 of the first study. a < 0 means that reassuring evidence moved the model more than alarming evidence.

## 1. What is known

- **The two studies.** With the Yes/No question about the alarming outcome ("Will this patient die during the current hospital stay?", the alarm framing), reassuring evidence moved Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct more than alarming evidence (first study). Under the registered rule, this did not replicate in Olmo-3-7B-Instruct, Qwen3-8B and Granite-3.3-8B-Instruct (second study, R1).
- **The split (exploratory, after the second study, not registered).** Each patient was also asked the reversed question ("Will this patient survive the current hospital stay?"). For a statistic s (a or a′) computed in each framing, the **outcome part** is O = (s_alarm + s_reversed) / 2 and the **answer-word part** is W = (s_alarm − s_reversed) / 2. A model that follows evidence more readily toward the answer "No", whatever outcome "No" stands for, has W < 0. A model that follows evidence more readily toward one outcome, whatever answer word names it, has O ≠ 0. In the 28 valid cells (model × dataset) of post-trained checkpoints in both studies and the scale checks, W was below 0 in both versions with p < 0.05 in 18 cells and above 0 in 1. In base models, |W| was at most 0.33. O pointed to the safe outcome in 10 cells (Qwen and Llama models) and to the alarming outcome in 5 (Olmo 3, Granite 3.3, and Qwen3-14B on PhysioNet 2019). In all 8 cells where both were conclusive, O had the same sign as the answer-option asymmetry of the second study.
- **What the earlier studies cannot answer.** The registered card states the direction of each factor twice: in words ("increases the risk of death") and as a signed number ("(impact +0.42)"). It also lists five features with their values, and it always speaks of the alarming outcome, even when the question names the safe one. The earlier studies cannot say which of these the two parts depend on. They also come from four developers only (Alibaba, Meta, the Allen Institute for AI and IBM).

## 2. Research questions

1. **R4.** Does the answer-word part appear in instruction-tuned models from three further developers?
2. **R5.** In those models, does the outcome part agree with the answer-option format, as it did in the second study?
3. **R6.** Does the answer-word part remain when the card loses its numbers, loses its direction words, is reduced to one sentence that states only the direction, or names the outcome of the reversed question?
4. **R7.** Which part of the card carries the outcome part: the numbers, the direction words, or either one? Is a one-sentence cue enough?
5. **Q8.** Descriptive questions without a decision rule (Section 7).

## 3. What is unchanged from the earlier studies

Everything below is as registered in `PREREGISTRATION.md` (with Deviations D1 to D5) and `PREREGISTRATION_V2.md`, and is implemented by the same code: `ler_main.py`, version main-2.0, byte-identical to the second study's file (SHA-256 0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e; every run checks this). Everything new is in `ler_v3.py`.

- The three datasets, the splits (seed 2027), the eligibility rule, the XGBoost model, the 16 features, the cards (top 5 SHAP values), the conditions C0 (no card), C2 (the patient's true card) and C4 (the same card with every sign flipped), the solved examples of set A, the Yes/No prompts in the alarm and the reversed framing, and the answer-option prompts of the second study.
- The **same test patients**: 300 per class per dataset, 1,800 in total.
- The reading: the next-token log-probabilities after "Answer:", from the top 1,000 log-probabilities with the imputation rule of Deviation D5. The exact-reading check of Deviation D2 still runs and is recorded. Every run uses vLLM 0.26.0 in float16 on two T4 GPUs; there is no fallback to another backend.
- The statistics a and a′, the intersection–union test, the two-sided bootstrap p-value of the first study (its Section 7), the thresholds of the validity rule (median answer-word mass ≥ 0.5, C0 accuracy ≥ 0.55), the treatment of invalid readings (Deviation D3, item 5) and the memorization check (Deviation D3, item 7), which runs for every model new to the studies.
- No system message is added. A chat template's own default system text is kept and logged.

Not used in this study: condition C6, example sets B and C, and the two controls of the second study.

## 4. Models

| Model | Developer | Role | Prompts per patient | Revision |
|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | Alibaba (Qwen team) | earlier model (R6, R7) | 20 | a09a35458c702b33eeacc393d103063234e8bc28 |
| meta-llama/Llama-3.1-8B-Instruct | Meta | earlier model (R6, R7) | 20 | 0e9e39f249a16976918f6564b8830bc894c89659 |
| allenai/Olmo-3-7B-Instruct | Allen Institute for AI | earlier model (R6, R7) | 20 | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc |
| tiiuae/Falcon3-7B-Instruct | Technology Innovation Institute | new model (R4 to R7) | 26 | 1e57a0ecd176c7c139f289c60a74e57f887c3dfb |
| 01-ai/Yi-1.5-9B-Chat-16K | 01.AI | new model (R4 to R7) | 26 | 2f8753bed95bd4fede3fd64f8305428620b4b212 |
| microsoft/Phi-4-mini-instruct | Microsoft | new model (R4 to R7) | 26 | cfbefacb99257ffa30c83adab238a50856ac3083 |

- **The earlier models** are the two instruction-tuned models of the first study and the final Olmo 3 model of the second study, all three run in float16 in the earlier studies. Together they hold 6 of the 19 cells with a conclusive W and 4 of the 15 cells with a conclusive O (the pre-listed cells of Section 7), and O points both ways among them (toward the safe outcome in Qwen2.5 and Llama 3.1, toward the alarming outcome in Olmo 3). They are scored again with the registered card in this study, so that every comparison within a model uses readings from one run. They are not checked for memorization again.
- **The new models** come from developers not in the earlier studies, have open weights in 16-bit precision and are instruction-tuned. The candidates, in the order of the engineering check (set before the check): zai-org/GLM-4-9B-0414 (Zhipu AI), tiiuae/Falcon3-7B-Instruct (Technology Innovation Institute), LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct (LG AI Research), CohereLabs/aya-expanse-8b (Cohere Labs), 01-ai/Yi-1.5-9B-Chat-16K (01.AI) and microsoft/Phi-4-mini-instruct (Microsoft). The first five have between 7 and 10 billion parameters, like the earlier models; Phi-4-mini has 3.8 billion and is last for that reason. The first three eligible candidates are registered: Falcon3-7B-Instruct, Yi-1.5-9B-Chat-16K and Phi-4-mini-instruct (Section 10). Phi-4-mini is registered because three of the five larger candidates were not eligible. Gemma models are not considered, because they are known to fail in float16, the only format the T4 GPUs support. Mistral models are not considered, because in the second study's engineering check Ministral 3 could not be loaded and Mistral-7B-Instruct-v0.3 answered an obvious question the wrong way round.
- **Model window.** vLLM's `max_model_len` is 16,384 tokens, as in both studies, or the model's own maximum length if that is shorter. For a model with "longrope" rotary scaling (Phi-4-mini), it is the model's original window (4,096 tokens), so that vLLM uses the rotary factors the model uses for prompts of this length. The window changes no prompt and no reading. It is logged in every run.
- **If a model cannot be run after registration.** A model "cannot be run" only if vLLM 0.26.0 fails to load it, or if more than 20% of its first 256 or more readings are invalid (the run notebook then stops). No eligible candidate remains after the engineering check (Section 10), so if any model cannot be run, the tests that need it are reported as not testable, and this is recorded as a deviation. A model that runs but fails a validity rule is never replaced: its cells enter the tests with p = 1.

## 5. Card formats

The evidence block of a C2 or C4 prompt appears in five formats. Each line of a card is written from a template in `ler_v3.py`; `{feature}` and `{value}` are the registered display name and value, `{outcome}` is the dataset's alarming outcome as in both studies ("death", "sepsis", "readmission within 30 days"), `{word}` is "increases" when the value described is above 0 and "decreases" otherwise, and `{impact}` is that value with two decimals and a sign that always matches the word, exactly as on the registered card. The five factors, their order and the header line are those of the registered card. C4 flips the sign of every factor, as in both studies.

| Format | What it removes or changes | Template (exact) |
|---|---|---|
| **full** | nothing: the registered card | `{k}. {feature} = {value}: {word} the risk of {outcome} (impact {impact})` |
| **words** | the numbers | `{k}. {feature} = {value}: {word} the risk of {outcome}` |
| **numbers** | the direction words | `{k}. {feature} = {value}: effect on the risk of {outcome} (impact {impact})` |
| **cue** | everything but the direction: one sentence, no header, no features, no values, no numbers | see below |
| **matched** (reversed framing only) | the card speaks of the outcome that the reversed question names | `{k}. {feature} = {value}: {word} the chance of {safe} (impact {impact})` |

The cue, as one sentence in place of the whole evidence block:

```
Evidence from a machine-learning model (XGBoost) trained on other patients: taken together, the {n} factors that most influenced its risk estimate for this patient {verb} the risk of {outcome}.
```

Here `{n}` is 5, and `{verb}` is "increase" if the five shown SHAP values sum to more than 0 and "decrease" otherwise. This is the card direction of both studies, so the cue points the same way as the card it replaces.

On the matched card, `{word}` and `{impact}` describe the negative of each SHAP value, which is the factor's effect on the log-odds of the safe outcome, and `{safe}` names the safe outcome as the reversed question does:

| Dataset | `{outcome}` | `{safe}` |
|---|---|---|
| PhysioNet 2012 | death | survival |
| PhysioNet 2019 | sepsis | staying free of sepsis |
| Diabetes 130 | readmission within 30 days | avoiding readmission within 30 days |

One line in each format, for a factor with SHAP value +0.42 (an illustration, not a study patient):

```
full:     1. Glasgow Coma Scale, lowest (3-15) = 7: increases the risk of death (impact +0.42)
words:    1. Glasgow Coma Scale, lowest (3-15) = 7: increases the risk of death
numbers:  1. Glasgow Coma Scale, lowest (3-15) = 7: effect on the risk of death (impact +0.42)
matched:  1. Glasgow Coma Scale, lowest (3-15) = 7: decreases the chance of survival (impact -0.42)
```

Everything else in the prompt (task line, solved examples, patient, question, answer instruction) is identical to the registered prompt of the same patient, condition and framing; the run notebook checks this for every prompt. The comparisons the formats allow:

- **full against words:** whether a part needs the numbers;
- **full against numbers:** whether it needs the direction words;
- **words against cue:** whether it needs the factor-level detail (features, values, five lines);
- **full against matched** (reversed framing): whether it depends on the card naming the alarming outcome when the question names the safe one. With the matched card, "increases" points to "Yes" in both framings; with the registered card, it points to "Yes" in the alarm framing and to "No" in the reversed framing.

## 6. Prompts

Per patient:

- **registered card:** C0, C2 and C4 × the alarm and the reversed framing (6 prompts);
- **words, numbers and cue:** C2 and C4 × both framings (12 prompts);
- **matched card:** C2 and C4 × the reversed framing (2 prompts);
- **new models only, answer options:** C0, C2 and C4 × both option orders, as in the second study (6 prompts).

That is 20 prompts per patient for an earlier model (36,000 per run) and 26 for a new model (46,800 per run); 248,400 in total. C0 has no card, so a format's statistics use the C0 reading of the same framing. In the alarm framing the registered card already names the outcome of the question, so the alarm-framing statistic of the matched card is that of the registered card.

## 7. Hypotheses and tests

**Common to all tests.** A cell is one model and one dataset. For each framing and card format, a and a′ are computed exactly as in Deviation D1 of the first study, from the C0 reading and the C2 and C4 readings with that card. O and W are then formed as in Section 1. Each comparison uses the patients with valid C0, C2 and C4 readings in every framing and format it involves, and one paired bootstrap over those patients (10,000 resamples, seed 2026), with the same resamples for every statistic in the comparison. The p-value of O, W or a change in them is the intersection–union p: the larger of the two-sided bootstrap p-values of its two versions (from a and from a′), and a direction is stated only when both versions have the same sign. A test in a cell is **supported** if the cell is valid (Section 8), both versions point in the predicted direction and the Holm-adjusted p is below 0.05. A cell that fails its validity rule, or is missing, enters its Holm family with p = 1.

### R4. The answer-word part in new families

- **Prediction.** W < 0 (toward the answer "No"), in both versions, in each cell of the three new models.
- **Test.** The intersection–union test of W with the registered card; Holm correction over the 9 cells (3 new models × 3 datasets).
- **Decision rule.** A new model **shows the pull toward "No"** if W is supported in more than half of its valid cells (at least one), and no valid cell has W above 0 in both versions with p < 0.05 (without correction). R4 is **supported** if at least 2 of the 3 new models show the pull. It is not testable if fewer than 2 new models have a valid cell.
- **If not supported,** the answer-word part is described as not shown to generalize to these developers' models.

### R5. The outcome part in new families, checked against the answer options

- **No direction is predicted** for O, because it differs between families (Section 1). O is reported for every cell of the new models, with Holm correction over the 9 cells; a direction is stated when the Holm-adjusted p is below 0.05 and both versions have the same sign.
- **Registered check of the split.** A cell is compared if it is valid, O is conclusive (both versions of the same sign, p < 0.05 without correction), the answer-option cell is valid under the rule of the second study, and the answer-option asymmetry (the R2 statistic of the second study, the mean of the two option orders) is conclusive in the same sense. The prediction is that O and the answer-option asymmetry have the same sign.
- **Decision rule.** Supported if they have the same sign in every compared cell, with at least 2 compared cells. Not testable with fewer than 2.

### R6. The answer-word part with each changed card

- **Eligible cells.** (i) The pre-listed cells of the earlier models, in which W was conclusive with the registered card in the exploratory split of Section 1: Qwen2.5-7B-Instruct on PhysioNet 2012 (W > 0) and on PhysioNet 2019 (W < 0); Llama-3.1-8B-Instruct on PhysioNet 2012 and on Diabetes 130 (W < 0); Olmo-3-7B-Instruct on PhysioNet 2012 and on PhysioNet 2019 (W < 0). (ii) The cells of the new models in which W is established in this study: valid, both versions of the same sign, and Holm-adjusted p < 0.05 in the R4 family, in either direction.
- **Prediction.** With each changed card (words, numbers, cue, matched), W keeps the sign it has with the registered card in that cell.
- **Test.** For each card, the intersection–union test of W in the cell's direction, with Holm correction over the eligible cells (one family per card). A cell in which the card fails its validity rule enters with p = 1.
- **Decision rule.** W is **kept** with a card if it is supported in more than half of the eligible cells. R6 is **supported** if W is kept with all four changed cards.
- **Also reported, without a decision rule:** for each cell and card, the change ΔW = W(card) − W(registered card) with its interval. The part is called **smaller** with a card if the change points toward 0 in both versions with p < 0.05 (without correction).

### R7. The outcome part with each reduced card

- **Eligible cells.** (i) The pre-listed cells of the earlier models in which O was conclusive with the registered card: Qwen2.5-7B-Instruct on PhysioNet 2012 (O < 0); Llama-3.1-8B-Instruct on PhysioNet 2012 and on PhysioNet 2019 (O < 0); Olmo-3-7B-Instruct on PhysioNet 2012 (O > 0). (ii) The cells of the new models in which O is established in this study (as for W, in the R5 family).
- **Test.** As in R6, for O: with each card, whether O keeps the sign it has with the registered card in that cell (Holm over the eligible cells, one family per card), and the change ΔO with its interval.
- **Interpretation rule.** O kept with both the words card and the numbers card: "kept with either the direction words or the numbers alone". Kept with the numbers card only: "carried by the numbers". Kept with the words card only: "carried by the direction words". Kept with neither: "not shown with either the direction words or the numbers alone". Separately: "a one-sentence direction cue is enough" if O is kept with the cue. The matched card is reported in the same way, without an interpretation rule.
- **Limit.** A card that moves the model less gives less power, so "not shown" is not evidence that the part is absent. The movement with each card is reported (Q8).

### Selection of eligible cells, and a sensitivity analysis

The eligible cells of the new models are selected on this study's own readings with the registered card. Selected cells tend to be those whose part was estimated as large, so a change from the registered card can look like a reduction by regression to the mean. The pre-listed cells are free of this, because they were selected on the earlier studies. R6 and R7 are therefore also reported on the pre-listed cells alone, as a registered sensitivity analysis without a decision rule.

### Q8. Descriptive questions (no decision rules)

- **a. Replay.** The registered-card prompts of the three earlier models are scored again in this study. Their readings are compared with the readings of the same prompts in the earlier studies: the largest and the median absolute difference in the log-odds of the alarming outcome, and the share of identical answers.
- **b.** The first study's statistic (a and a′ in the alarm framing) with each card.
- **c.** For each card: accuracy in C2 and C4, the share of answers that follow the card, the mean movement between the alarm-pointing and the safe-pointing card (mean of la − ls on the log-odds scale, the mean of the two framings), and O and W divided by that movement, with intervals.
- **d.** O and W with the matched card.
- **e.** For the new models: accuracy by condition, the answer-option results, the mean preference for the letter A, and the memorization check.

## 8. Analysis

- **Validity of a cell.** With the registered card: median Yes+No mass ≥ 0.5 over the alarm-framing records (C0, C2, C4), median mass ≥ 0.5 over the reversed-framing records (C0, C2, C4), and C0 accuracy ≥ 0.55 in the alarm framing. A record without a usable reading counts as mass 0. This is the rule of both studies with two changes: C6 is not part of this study, so the median is over C0, C2 and C4; and the reversed framing must pass the mass criterion too, because both parts use it.
- **Validity of a card format in a cell.** The cell is valid, and the median mass over that format's records (C2 and C4, both framings, or the reversed framing for the matched card) is ≥ 0.5.
- **Validity of an answer-option cell.** The rule of the second study.
- **Estimators.** a and a′ as in `ler_main.py` (`adjusted_fit`, `reading_adjusted_asymmetry`). For speed, `ler_v3.py` computes them for many resamples at once as weighted least-squares fits in which the weights are the resample counts; this gives exactly the estimates of refitting on the resampled rows (the tests compare it with `ler_main.py`). Each comparison starts its own random generator with seed 2026.
- **Code.** `ler_main.py` (main-2.0, unchanged), `ler_v3.py` (v3-1.1) and the notebooks `llm_evidence_reliance_engcheck_v3.ipynb`, `llm_evidence_reliance_main_run_v3.ipynb` (one run per model) and `llm_evidence_reliance_analysis_v3.ipynb` are committed with this file, at the tag `prereg-v3`, before any run. Each run checks `ler_main.py` against the hash above, checks `PREREGISTRATION.md`, `PREREGISTRATION_V2.md` and this file at the tag (every prompt string, model id and pinned revision in the code must appear in them) and stops otherwise. Each run downloads exactly the pinned revision and stops if it differs.
- **Families.** R4, R5, R6 and R7 are separate confirmatory questions, each with its own Holm correction (one family per card in R6 and R7), decision rule and α = 0.05.
- **One analysis.** The analysis notebook is run once, after all six runs, on the six run outputs, and (for Q8a only) on the earlier runs of the three earlier models from `results/main/` and `results/v2/`. Before it analyses anything, it checks that all runs used the same code, protocol text, patients, cards and solved examples, that every prompt shared by two runs has the same text, that each run is complete, and that each model's revision is the registered one. A missing run enters its tests with p = 1, and the tests that need it are reported as not testable. The output and the six run outputs are committed to `results/v3/`.
- **Changes after the tag.** Only engineering settings that do not touch prompts, readings, models or analysis (for example a download retry) may change without a deviation. Any other change is recorded as a dated deviation, and the tests it affects are marked in the report.

## 9. Order of work

1. Engineering check (Section 10), before this file is tagged.
2. Pin the three new models in this file and in `ler_v3.py`, commit them with the engineering-check output, and create the release `prereg-v3`.
3. Six runs, one model per Kaggle session: the three earlier models first (Olmo-3-7B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct), then the three new models in the order of Section 10. Each run first checks the protocol at the tag. In the second study's runs, scoring took 0.21 to 0.32 seconds per prompt. With data, download, loading and the memorization check, a run is expected to take about 3 hours for each earlier model and for Phi-4-mini-instruct, about 4 hours for Falcon3-7B-Instruct, and about 6 to 7 hours for Yi-1.5-9B-Chat-16K, the slowest model in the engineering check.
4. One run of the analysis notebook.

## 10. What was known before registration, and how the new models were chosen

- **Known:** all results of both studies and of the scale checks, and the exploratory split of Section 1.
- **Engineering check.** Each candidate model, and each of the three earlier models, is downloaded and run on dev-sample prompts only (never test patients): per dataset, 6 dev patients × C0, C2 and C4 × both framings with the registered card, the same patients × C2 and C4 × both framings with each reduced card and × C2 and C4 in the reversed framing with the matched card, 3 dev patients × C0, C2 and C4 × both option orders, and a timing set of 100 Yes/No and 100 answer-option prompts. The check records only technical quantities: whether the model loads in float16 with vLLM 0.26.0 on two T4 GPUs, the model window, the weights' format, the chat template and its default system text, whether "Yes", "No", "A" and "B" are single tokens, the exact-reading check of Deviation D2, whether every reading is finite, the median answer-word mass with each card format and with the answer options, the share of imputed readings, prompt lengths, the time per prompt, and the four questions with an obvious answer of the second study. It computes no accuracy, no answer direction and no asymmetry, and it saves no per-prompt reading.
- **Eligibility.** A new candidate is eligible if it loads, its weights are not quantized, every reading is finite, the median answer-word mass is at least 0.5 with every card format and with the answer options, and the four obvious questions are answered the right way round. This is the `ok` field computed by `engineering_check_v3()` with all six formats required. For the three earlier models the check decides nothing; it is reported.
- **Result of the check** (2026-10-29, Kaggle, two T4 GPUs, 101 minutes; 1,014 dev prompts per model). Median answer-word mass on the check prompts of each format, and the share of readings with an imputed variant on the timing prompts:

| Model | Eligible | Window | Mass, full | words | numbers | cue | matched | A/B | Imputed, Yes/No | Imputed, A/B | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zai-org/GLM-4-9B-0414 | no | – | – | – | – | – | – | – | – | – | vLLM 0.26.0 refuses float16 for this model type ("numerical instability") |
| tiiuae/Falcon3-7B-Instruct | yes | 16,384 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 0% | 19% | first new model |
| LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct | no | – | – | – | – | – | – | – | – | – | its tokenizer could not be loaded: transformers cannot read the model's configuration (model type "exaone") without the model's own code, which the registered code does not run |
| CohereLabs/aya-expanse-8b | no | 8,192 | 0.947 | 0.920 | 0.956 | 0.907 | 0.992 | 0.358 | 39% | 59% | median A/B mass below 0.5 |
| 01-ai/Yi-1.5-9B-Chat-16K | yes | 16,384 | 0.999 | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 | 0% | 0% | second new model |
| microsoft/Phi-4-mini-instruct | yes | 4,096 | 0.997 | 0.998 | 0.996 | 0.999 | 0.997 | 1.000 | 46% | 35% | third new model |
| Qwen/Qwen2.5-7B-Instruct | earlier model | 16,384 | 0.998 | 0.999 | 0.999 | 1.000 | 1.000 | (1.000) | 14% | (75%) | |
| meta-llama/Llama-3.1-8B-Instruct | earlier model | 16,384 | 0.990 | 0.993 | 0.989 | 0.992 | 0.992 | (0.998) | 0% | (7%) | |
| allenai/Olmo-3-7B-Instruct | earlier model | 16,384 | 0.997 | 0.997 | 0.996 | 0.999 | 0.996 | (0.999) | 0% | (0%) | |

Values in parentheses are for a format the model is not run with. The three new models answered all four obvious questions the right way round, have 16-bit weights, gave finite readings only, have single-token variants of "Yes", "No", "A" and "B", and their chat templates add no default system text. Their longest prompts have 2,426 (Falcon3), 2,442 (Yi) and 2,153 (Phi-4-mini) tokens, within each window. The exact-reading check of Deviation D2 gave largest differences of 0.013 and 0.016 (Falcon3, Yes/No and A/B), 0.031 and 0.027 (Yi), and 0.030 and 0.061 (Phi-4-mini); as in both studies, every run reads the top 1,000 log-probabilities, and the imputation bound of Deviation D5 is reported for every cell with imputed readings. The check's time per prompt was about twice the time per prompt of the second study's runs for the same models (0.44 to 0.45 seconds against 0.23 to 0.24 seconds), because the check reads small batches; the expected run times of Section 9 take this into account.

For EXAONE-3.5-7.8B-Instruct, the notebook did not recover the result line of the check process and recorded "the check process failed"; the end of the process output, kept in the output file, shows the error above. After the check, the function that reads these result lines (`parse_result` in `ler_v3.py`) was changed so that it finds the result anywhere in the process output. This changes no prompt, reading, model or analysis.

The full output is committed as `results/v3/engcheck_v3.zip` (SHA-256 6984a4d96861563db7d35f66d0880bc63cfc2c3366a77e7d35738497088c6e00). The code that produced it is the `ler_main.py` (main-2.0) and the `ler_v3.py` inside that file; the latter is version v3-1.0 with SHA-256 148e1990bcf15905c07657c2aad54099afdb4bc8d64b6830dbf8d4665781e61d. The registered `ler_v3.py` (v3-1.1) differs from it only by the three pinned models, these two hashes and the change to `parse_result`. The pinned revision of each new model is the revision the check loaded. No card format or prompt string was changed after the check.

## References

- First study: `PREREGISTRATION.md` in this repository, releases `prereg-v1` to `prereg-v1.3`; results in `results/main/`.
- Second study: `PREREGISTRATION_V2.md`, release `prereg-v2`; results in `results/v2/`.
- Olmo 3: Team Olmo, Allen Institute for AI (2025). Olmo 3. arXiv 2512.13961.

## Deviations

### 2026-10-30: after the registered analysis

#### V3-D1. The imputation bound of Deviation D5 is computed by a separate script

- Section 10 says that the imputation bound of Deviation D5 is reported for every cell with imputed readings. The registered analysis notebook does not compute it. I noticed this only after the registered analysis had been run and its results were known.
- The bound is computed by `results/v3/supplement_d5.py`, with the method of the second study's analysis notebook: in every record with an imputed variant, the probability of the less likely answer word is halved, and the statistics are recomputed (2,000 resamples, as in the second study). The script uses the registered code at the tag (`ler_main.py` main-2.0, `ler_v3.py` v3-1.1). Its output is `results/v3/supplement_d5.json`. It is a labelled sensitivity analysis and decides nothing.
- Records with an imputed variant: Falcon3-7B-Instruct 2%, Olmo-3-7B-Instruct 0.2%, Qwen2.5-7B-Instruct 11%, Phi-4-mini-instruct 52%; none for Llama-3.1-8B-Instruct and Yi-1.5-9B-Chat-16K. Under the bound, O and W change by at most 0.03 for Falcon3 and Olmo 3 and by at most 0.20 for Qwen2.5, with every card. For Phi-4-mini they change by up to 0.86 on PhysioNet 2012 and 2019 and up to 2.15 on Diabetes 130; the largest changes are in the statistic a. Under the bound, three conclusive readings of Phi-4-mini are no longer conclusive: its W with the registered card on PhysioNet 2019 (bounded p = 0.149), its O with the cue on PhysioNet 2012, and its O with the matched card on PhysioNet 2019. No registered decision would change if the bounded readings were used: R4 and R5 stay not supported; in R6, every card still keeps W in more than half of the eligible cells when the Phi-4-mini cell on PhysioNet 2019 is left out; and no count of R7 changes.
- I wrote this entry on 2026-10-30, after the registered analysis had been run and its results were known.
