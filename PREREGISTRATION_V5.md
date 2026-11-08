# Pre-registration, fifth study: does a card change the answer prior, and what is left of the polarity part against a card without content, on patients no earlier study used?

- **Author:** the contributors listed on the OSF registration (names are left out of this file so that it can be shared through an anonymized link during review)
- **Registered:** November 2026, committed and released as `prereg-v5` and registered on OSF, before any prompt of this study was scored on a patient of Section 3. The `prereg-v5` release date and the OSF registration date are the authoritative timestamps; the link to the OSF registration is given in the repository's README, so that this file stays as it is at the tag.
- **Builds on:** the first study (`PREREGISTRATION.md`, releases `prereg-v1` to `prereg-v1.3`), the second study (`PREREGISTRATION_V2.md`, `prereg-v2`), the third study (`PREREGISTRATION_V3.md`, `prereg-v3`), the fourth study (`PREREGISTRATION_V4.md`, `prereg-v4`), the two exploratory scale checks, and the post-hoc analyses reported in the paper.
- **Status:** written after the results of all four studies, and of the post-hoc analyses of them, were known (Section 1). They motivate every prediction below. No prompt of this study has been scored on a patient of Section 3. The new prompts (the card without content, the numbers card at three scales and the risk probability, Section 5) were scored only on dev patients, in the engineering check of Section 10, and only the technical quantities listed there were computed from them.
- **Changes:** any change after registration is listed in the "Deviations" section at the end, with date and reason. This file is never edited silently.

In this file, **reassuring evidence** points to the safe outcome (survival, no sepsis) and **alarming evidence** to the alarming outcome. The asymmetry statistics a and a′ are those of Deviation D1 of the first study; a < 0 means that reassuring evidence moved the model more than alarming evidence.

## 1. What is known

- **The split.** Each patient is asked about the alarming outcome ("Will this patient die during the current hospital stay?", the alarm framing) and about the safe outcome ("Will this patient survive the current hospital stay?", the reversed framing). For a statistic s (a or a′) computed in each framing, the **outcome part** is O = (s_alarm + s_reversed) / 2 and the **polarity part** is W = (s_alarm − s_reversed) / 2. W < 0 means that evidence moved the model more readily toward the answer "No", whatever outcome "No" stood for.
- **Two reading points.** The statistics are read at a reading point of the no-card baseline: at the model's own threshold (log-odds 0, the registered reading of every earlier study) or at its **neutral point**, where the baseline is shifted by the static lean λ, the median over patients of half the difference between the no-card log-odds of the first answer word in the two framings. Most post-trained models lean toward "No" without a card (λ < 0), and in models with a strong lean the two reading points give different answers. Across the 37 valid cells of post-trained checkpoints in the earlier studies, W was conclusive toward "No" in 20 cells and toward "Yes" in 6 at the threshold, and toward "No" in 9 and toward "Yes" in 17 at the neutral point.
- **What W cannot separate.** If the presence of a card changes the model's answer prior by a share δ (the lean becomes (1 − δ) λ), this adds −2δλ to W at either reading point, with no asymmetry in how the model updates. The earlier conditions (no card, the true card, the flipped card) cannot separate this from an asymmetry. Every conclusive neutral-point reading of the Qwen3 and Olmo 3 checkpoints would follow from a change of the prior by at most 32%. Qwen3-8B leans most strongly: λ = −7.12 on PhysioNet 2012 and −4.73 on PhysioNet 2019, and its neutral-point W points toward "Yes" in both (a′ = +1.17 and +1.55), which a card that weakens the preference for "No" by 8% and 16% would produce.
- **Readable cells.** A conclusive reading is called readable here if at least 30 patients lie within 1 log-odds of the reading point in both framings and the card would have to change the prior by at least half (δ* = |W| / (2|λ|) ≥ 0.5, with the weaker of the two versions of W) to produce it. On PhysioNet 2012 and PhysioNet 2019, the readable neutral-point readings are Falcon3-7B-Instruct on both datasets (W toward "Yes", a′ = +0.67 and +1.20), Yi-1.5-9B-Chat-16K on PhysioNet 2019 (toward "Yes", a′ = +0.78), Granite-3.3-8B-Instruct on PhysioNet 2012 (toward "No", a′ = −3.35) and Phi-4-mini-instruct on PhysioNet 2012 (toward "No"). The paper's claim is that where the effect can be read, its direction depends on the model family.
- **Other limits of the earlier evidence.** All four studies used the same 1,800 test patients, and the cells above were selected on them. With True/False answers in the fourth study, W pointed toward "False" in all 13 valid cells, so W follows the answer format. The decision-level polarity effect from freely generated answers had the sign of the first-token effect in only 2 of the 6 cells where the first-token effect was conclusive. Over the three Yes/No templates of the fourth study (registered, alarm-last example order, paraphrase), the median standard deviation of W was 0.60 (a) and 0.36 (a′) log-odds at the threshold and 0.45 and 0.38 at the neutral point. The bootstrap over patients does not include this source of variation.

## 2. Research questions

1. **R15.** Does the presence of a card change the answer prior? Measured directly with a card that has the registered card's header, factors and values but no direction and no size (the placebo card, CP), in Qwen3-8B.
2. **R16.** Against the placebo card, at its neutral point and on patients that no earlier study used, does W keep the family-specific direction of the readable cells?
3. **R17.** Does the polarity effect appear in freely generated decisions in the cells of R16?
4. **Q18 to Q22.** Registered questions without a decision rule: what W follows (paraphrase, True/False, letters), whether the size of the impacts matters, a probability instead of a card, the diagnostics of the reading, and the test with the template variance added (Section 7).

## 3. Patients

No patient of this study was used in an earlier study. The engineering check (Section 10) builds both samples and records their fingerprints and sizes; it scores no prompt on them.

- **PhysioNet 2012: set C.** The 4,000 ICU stays of set C of the PhysioNet/Computing in Cardiology Challenge 2012 (`set-c.tar.gz` and `Outcomes-c.txt`, PhysioNet, version 1.0.0). The earlier studies used only sets A and B. Patient records are read with the registered reader of `ler_main.py`, and the outcome is in-hospital death.
- **PhysioNet 2019.** The patients of the registered test and dev pools (seed 2027) that are in neither the registered test sample nor the registered dev sample of the earlier studies. Before they are excluded, the registered test sample is rebuilt and its fingerprints are checked against the earlier run logs (test ids `cda9d34771cde56d`, test cards `b0334b1289dee4da`; for PhysioNet 2012, test ids `eb7ca19be05d8b46`, test cards `e9abe0533092f138`). The run stops if any fingerprint differs.
- **Eligibility.** At least 10 of the 16 registered features measured, as in every earlier study.
- **Sample.** 300 patients per class per dataset, drawn without replacement from the sorted eligible ids of each class with `numpy.random.RandomState(2032)` (the alarming class first, then the safe class), then shuffled with the same generator. **Fallback rule:** if a dataset has fewer than 300 eligible patients in a class, k = the size of the smaller class, and k patients are drawn from each class. A dataset with fewer than 10 eligible patients in a class is not used. PhysioNet 2019 needs the fallback rule, because the earlier test and dev samples used many of its positive patients (Section 10).
- **Model and cards.** The registered XGBoost model of each dataset (trained on the training split of the earlier studies, unchanged), its probability for each patient, and the registered card: the five features with the largest absolute SHAP values, in that order, with the card direction given by the sign of their sum.
- **The fresh samples** (from the engineering check, Section 10). PhysioNet 2012: 3,911 eligible stays of set C (573 deaths), 300 per class, 600 patients, fingerprints `c9555e028ee16b9e` (ids) and `fa60197aec5ea6fa` (cards). PhysioNet 2019: 10,660 eligible patients (271 with sepsis), so the fallback rule applies: 271 per class, 542 patients, fingerprints `9940fba7c625991c` (ids) and `dd0a04f4080e72a3` (cards). Neither sample shares a patient with an earlier sample. Every run rebuilds the samples and stops if a fingerprint differs.
- **Generation subsample.** For generated answers (Section 5.6), 150 patients per class per dataset (100 if rule (iii) of Section 10 applies, or all patients of the class if fewer), drawn from the sorted fresh ids of each class with `numpy.random.RandomState(2033)`, the alarming class first. Rule (iii) applies (Section 10), so the subsample has 100 patients per class: 200 per dataset, 400 in total.
- **Diabetes 130 is not used,** to keep the GPU time of the study down. The cells of R15 and R16 were chosen on PhysioNet 2012 and PhysioNet 2019 only.

## 4. Models

| Model | Role | Revision |
|---|---|---|
| Qwen/Qwen3-8B | R15; Q18 to Q22 | b968826d9c46dd6066d109eabc6255188de91218 |
| tiiuae/Falcon3-7B-Instruct | R16 and R17 (two cells); Q18 to Q22 | 1e57a0ecd176c7c139f289c60a74e57f887c3dfb |
| 01-ai/Yi-1.5-9B-Chat-16K | R16 and R17 (one cell); Q18 to Q22 | 2f8753bed95bd4fede3fd64f8305428620b4b212 |
| ibm-granite/granite-3.3-8b-instruct | R16 and R17 (one cell); Q18 to Q22 | 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b |
| meta-llama/Llama-3.1-8B-Instruct | Q18 to Q22 | 0e9e39f249a16976918f6564b8830bc894c89659 |
| Qwen/Qwen2.5-7B-Instruct | Q18 to Q22 | a09a35458c702b33eeacc393d103063234e8bc28 |
| allenai/Olmo-3-7B-Instruct | Q18 to Q22 | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc |

- Each model was run in an earlier study with the same revision. Qwen3-8B is run with thinking turned off in its chat template, as before.
- **Phi-4-mini-instruct is not included,** as in the fourth study: 52% of its third-study readings needed an imputed answer word under the top-1,000 reading. Its readable cell (Section 1) is therefore not tested.
- **Model window.** 16,384 tokens, or the model's own maximum length if that is shorter.
- **If a model cannot be run after registration.** As in the earlier studies: a model "cannot be run" only if vLLM 0.26.0 fails to load it, or if more than 20% of its first 256 or more readings are invalid. Its tests then count it as missing (p = 1), and this is recorded as a deviation. No model is replaced after registration.

## 5. Prompts

Every new prompt is a registered prompt of an earlier study in which only the evidence block is replaced. The run notebook checks this for every prompt (`check_prompt_v5` in `ler_v5.py`). The registered evidence block of the true card (C2) is

```
Evidence from a machine-learning model (XGBoost) trained on other patients. These are the 5 factors that most influenced its risk estimate for this patient:
1. {feature} = {value}: increases the risk of {outcome} (impact +0.71)
...
```

with "decreases" and a negative impact where a SHAP value is negative, and the absolute SHAP value with two decimals as the impact; C4 is the same card with every sign flipped.

### 5.1 The placebo card (CP)

The registered header, the same five factors in the same order with the same values, and in place of each direction the line

```
{k}. {feature} = {value}: does not change the risk of {outcome} (impact 0.00)
```

The placebo card is built from the patient's true card (C2) and is the same in every framing of that patient. It has no direction. The header still says that the five factors most influenced the risk estimate, which does not fit "does not change the risk" well. I keep the header so that the placebo differs from the true card only in the lines that carry direction; any effect of the header itself is then shared by CP, C2 and C4.

### 5.2 Answer formats

Each pair has two framings in which the first answer word stands for the alarming outcome in the first framing and for the safe outcome in the second.

| Pair | Framings | Answer words | Source |
|---|---|---|---|
| yn | alarm, reversed | Yes / No | the registered prompts of the first study |
| pa | para_alarm, para_reversed | Yes / No | the paraphrased prompts of the fourth study (Section 5.4 of `PREREGISTRATION_V4.md`), with the instruction "Reply with Yes or No." |
| tf | tf_alarm, tf_reversed | True / False | the True/False statements of the fourth study, ending with "Is this statement true or false? Answer with True or False." |
| ab | choice_alarm_A, choice_safe_A | A / B | the answer-option framing of the second study: "Which of these two outcomes is more likely for this patient?", two options "A: …" and "B: …" with the outcome statements, and "Answer with A or B."; option A is the alarming outcome in choice_alarm_A and the safe outcome in choice_safe_A |

In the ab pair both outcomes are named in both framings, so a W in this pair cannot come from the outcome that the question names; W > 0 there means that evidence moved the model more readily toward the letter "A".

### 5.3 The numbers card at three scales (dose)

The numbers card of the third study, in the alarm framing with C2 and C4, with every impact multiplied by 0.5, 1 or 2 before it is printed (two decimals and a sign):

```
{k}. {feature} = {value}: effect on the risk of {outcome} (impact {impact})
```

The card at scale 1 is exactly the numbers card of the third study. The direction of the card is unchanged by the scale.

### 5.4 The risk probability

In place of the evidence block, one sentence with the registered model's probability p for the patient, rounded to whole percent:

```
Evidence from a machine-learning model (XGBoost) trained on other patients: its estimated risk of {outcome} for this patient is {p}%.
```

**CQ2** shows p and **CQ4** shows 100 − p. The evidence points to the alarming outcome if the number shown is above 50 and to the safe outcome if it is below 50. Patients with p = 50 have no direction and are left out of the analyses of the probability; their number is reported. The probability is used in the alarm and the reversed framing.

### 5.5 Prompts per patient

| Grid | Conditions | Framings | Prompts |
|---|---|---|---|
| yn | C0, CP, C2, C4 | alarm, reversed | 8 |
| pa | C0, CP, C2, C4 | para_alarm, para_reversed | 8 |
| tf | C0, CP, C2, C4 | tf_alarm, tf_reversed | 8 |
| ab | C0, CP, C2, C4 | choice_alarm_A, choice_safe_A | 8 |
| dose | C2, C4 | alarm, at scales 0.5, 1 and 2 | 6 |
| prob | CQ2, CQ4 | alarm, reversed | 4 |

That is 42 scored prompts per patient. With the 600 PhysioNet 2012 and 542 PhysioNet 2019 patients of Section 3, a model receives 47,964 scored prompts (38,828 for Qwen3-8B, which does not run the True/False grid; Section 10). Every prompt uses the solved examples of set A, as registered.

### 5.6 Generated answers

For the generation subsample (Section 3), each model generates text with greedy decoding (temperature 0) for C0, C2 and C4 in the alarm and the reversed framing (6 prompts per patient; 2,400 answers per model with 100 patients per class, Section 3), through its chat template without the prefilled "Answer:", with at most 512 new tokens. The answer is read by the fixed rule of the fourth study: after removing the characters `*`, `_`, `` ` ``, `#` and `>`, the first word if it is "yes" or "no" (any case); otherwise the first "yes" or "no" that follows the word "answer" (with "is", ":" or both in between, or nothing); otherwise the answer is "unparsed".

## 6. Readings and statistics

Everything below is as registered in `PREREGISTRATION.md` (with Deviations D1 to D5) and the later protocols, and is implemented by the same code: `ler_main.py`, version main-2.0 (SHA-256 0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e), `ler_v3.py`, version v3-1.1 (SHA-256 89d4c0c0ed995e82aa472eeac3f22ea0728583c7c8c84a01c87befe18f5e92ec), and `ler_v4.py`, version v4-1.2 (SHA-256 8a23f2ece854317185a720606d1bf3b427ea8e5589bf9ec9577f9ea4941af0a7), all byte-identical to the fourth study's files (every run checks this). Everything new is in `ler_v5.py`.

- **Reading.** The next-token log-probabilities of the two answer words after the prefilled "Answer:", from the top 1,000 log-probabilities with the imputation rule of Deviation D5; vLLM 0.26.0 in float16 on two T4 GPUs, with no fallback to another backend. No system message is added; a chat template's own default system text is kept and logged. For each record, lo_alarm is the log-odds of the answer word that stands for the alarming outcome in that framing.
- **Baseline.** Every statistic is computed against a baseline condition: **C0** (no card), as in the earlier studies, or **CP** (the placebo card). With CP as the baseline, l0 is the CP reading of the same patient and framing.
- **a and a′** as in Deviation D1 of the first study, from the baseline reading l0 and the readings with the alarm-pointing and the safe-pointing evidence (C2 and C4, or CQ2 and CQ4, ordered by their direction for each patient). O and W of a pair as in Section 1.
- **Static lean** of a pair and baseline: λ = the lower median over patients of (l0_first − l0_second) / 2, where l0 is the baseline log-odds of the alarming outcome in the first and the second framing. For each patient, (l0_first − l0_second) / 2 is the log-odds of the first answer word averaged over the two framings, so λ < 0 means a preference for the second answer word.
- **Threshold reading:** a and a′ as registered. **Neutral-point reading:** the same statistics with the baseline shifted by λ in the first framing and by −λ in the second before the reading point is fitted, so that each framing is read where the model is undecided once its lean is removed. λ is recomputed in every bootstrap resample.
- **The joint test** of a part (O or W) or of a change: the larger of the two-sided bootstrap p-values of its a and a′ versions, p = 2 × min(share of resamples ≤ 0, share ≥ 0); a direction is stated only when both versions have the same sign, and otherwise p = 1. A part is **conclusive** if this p is below 0.05. The bootstrap resamples patients (10,000 resamples), and each analysis starts its own random generator with seed 2026.

## 7. Hypotheses and tests

A cell is one model and one dataset. Every test uses the patients with valid readings in every condition and framing it involves.

### R15. The placebo card and the answer prior

- **Statistic.** In the yn pair, Δλ = λ(CP) − λ(C0), the change of the static lean when the placebo card is shown, on the patients with valid C0 and CP readings in both framings; two-sided bootstrap p.
- **Cells.** Qwen3-8B on PhysioNet 2012 and on PhysioNet 2019.
- **Prediction.** Δλ > 0: the placebo card weakens the preference for "No" (Section 1).
- **Decision rule.** In a cell, R15 is supported if the cell is valid (Section 8), Δλ > 0 and its Holm-adjusted p (over the two cells) is below 0.05. **R15 is supported** if it is supported in both cells. An invalid cell enters with p = 1.
- **Interpretation.** If R15 is supported, the presence of a card changes the answer prior of Qwen3-8B, so its W against C0 cannot be read as an asymmetry in updating, and W against CP is the reading to use. If Δλ is conclusive and negative in both cells, the card strengthens the preference for "No", and the paper says so. Otherwise, a change of the prior by the presence of a card is not shown in this model.

### R16. W against the placebo card

- **Statistic.** W of the yn pair with CP as the baseline, at the neutral point of CP, joint test over patients (Section 6).
- **Cells and predicted signs.** The readable neutral-point cells of Section 1 on these datasets, without Phi-4-mini: Falcon3-7B-Instruct on PhysioNet 2012 (W > 0) and on PhysioNet 2019 (W > 0), Yi-1.5-9B-Chat-16K on PhysioNet 2019 (W > 0), and Granite-3.3-8B-Instruct on PhysioNet 2012 (W < 0).
- **Decision rule.** In a cell, W is supported if the cell is valid, both versions have the predicted sign and the Holm-adjusted p (over the four cells) is below 0.05; it is **opposite** if both versions have the other sign with a Holm-adjusted p below 0.05. An invalid cell enters with p = 1. **R16 is supported** if W is supported in at least 3 of the 4 cells and opposite in none.
- **Interpretation.** If R16 is supported, the family-specific direction of W is not produced by the presence of a card and is not specific to the earlier patients. If it is not supported, the paper reports that the family-specific readings did not replicate against a card without content on new patients.
- **Selection.** The four cells were selected because their readings were conclusive and readable on the earlier patients, so a smaller W on new patients is expected from regression to the mean alone. R16 therefore asks only whether W keeps its sign.

### R17. Generated decisions

- **Statistic.** In each cell of R16, the decision-level polarity effect of the generated answers, E_W = (E_alarm − E_reversed) / 2, where E in one framing is the share of added false alarms with the alarm-pointing card minus the share of added missed alarms with the safe-pointing card, both relative to C0 (the definition of the fourth study), on the patients whose six answers were all parsed; 95% percentile interval from a paired bootstrap over patients.
- **Decision rule.** In a cell, R17 is supported if at least 100 patients have all six answers parsed, the cell is valid and the interval lies entirely on the side of the predicted sign of R16. **R17 is supported** if it is supported in at least 3 of the 4 cells.
- **Interpretation.** If R17 is supported, the polarity effect reaches the decisions a model writes out, not only its first-token reading. If it is not, the paper keeps its claims to the first-token reading.

### Q18. What W follows

For every valid cell and every pair that passes its validity rule (Section 8): O and W of the pa, tf and ab pairs against C0 and against CP, at both reading points, with intervals. Reported: whether W keeps the sign of the yn pair. The paraphrase changes the wording with the answer words fixed; True/False changes the answer words; the letters name both outcomes in both framings.

### Q19. The size of the impacts

In the alarm framing of every valid cell, the mean movement m = mean(la − ls) between the alarm-pointing and the safe-pointing version of the numbers card at scales 0.5, 1 and 2, and with the registered card, on the patients with valid C2 and C4 readings with all of them; and Δm = m(2) − m(0.5), with interval and two-sided bootstrap p. Δm > 0 means that the models read the size of the impacts, not only their sign.

### Q20. A probability instead of a card

For every valid cell: O and W of the yn pair with CQ2 and CQ4 as the evidence, against C0 and against CP, at both reading points, and the mean movement; and whether W has the sign it has with the card.

### Q21. Diagnostics of the reading

- **a.** Δλ (R15) in every valid cell, and the placebo card's own shift of the log-odds of the alarming outcome, mean(CP − C0) averaged over the two framings (an outcome effect of the placebo, which cancels in W and λ).
- **b.** O and W of the yn pair against C0 at both reading points in every valid cell, and whether they keep the sign of the same cell on the earlier patients.
- **c.** For every pair and baseline: the number of patients within 1 log-odds of each reading point in both framings, δ*, and whether the reading is readable (Section 1).
- **d.** For the generated answers of every cell: the share of parsed answers, the agreement of the parsed answer with the first-token reading of the same prompt, and E_W.
- **e.** For every format: accuracy by condition, the share of C2 and C4 answers that follow the card, the median answer-word mass and the share of imputed readings.
- **f. Imputation bound.** For every run with imputed readings, R15 and R16 computed again with the bound of Deviation D5 (in each record with an imputed variant, the probability of the less likely answer word is halved; 2,000 resamples), as in the fourth study. It decides nothing.

### Q22. Intervals that include the template variance

The median standard deviation of W over the three Yes/No templates of the fourth study (Section 1: a 0.597 and a′ 0.362 at the threshold, a 0.453 and a′ 0.382 at the neutral point) is added to each cell's bootstrap variance, and both versions are tested with a normal approximation. Reported: which conclusive readings of every pair stay conclusive, and R16 recomputed with this test (no decision rule). This test is stricter than the registered one because it asks whether W would hold under a different wording of the prompt, which the patient-level test of R16 does not ask.

## 8. Analysis

- **Validity of a cell.** The median Yes+No mass is at least 0.5 over the alarm-framing records (C0, CP, C2, C4) and over the reversed-framing records, and the C0 accuracy in the alarm framing is at least 0.55. A record without a usable reading counts as mass 0.
- **Validity of another pair in a valid cell.** In each of its framings: (i) the median answer-word mass over its records is at least 0.5; (ii) the mean movement mean(la − ls) on the log-odds of the alarming outcome is above 0; (iii) the C0 log-odds of the alarming outcome have a positive Spearman correlation with those of the alarm question over the cell's patients. For the dose and the probability grids, criterion (i) applies to their records.
- **Estimators.** a and a′ are computed for many resamples at once with weighted least squares, exactly as in the third and fourth studies; the neutral-point versions shift the baseline as in Section 6.
- **Families.** R15, R16 and R17 are separate confirmatory questions, each with its own correction (Holm over its cells for R15 and R16), decision rule and α = 0.05. Q18 to Q22 have no decision rule and no correction.
- **Code.** `ler_main.py`, `ler_v3.py` and `ler_v4.py`, unchanged; `ler_v5.py` (v5-1.0); the notebooks of the two engineering checks, exactly as they were run; and the notebooks `llm_evidence_reliance_main_run_v5.ipynb` (one run per model) and `llm_evidence_reliance_analysis_v5.ipynb` are committed at the tag `prereg-v5` before any run, and the SHA-256 of each file is listed in Section 10 of this file. The code is not attached to the OSF registration, because the registered `ler_main.py` names the repository and would break the anonymized link; the hashes pin it instead. Each run checks the code against the hashes above, checks this file (every prompt string, model id, pinned revision, hash and fingerprint in the code must appear in it) and stops otherwise. Each run downloads exactly the pinned revision and stops if it differs.
- **One analysis.** The analysis notebook is run once, after all runs. Before it analyses anything, it checks that all runs used the same code, protocol text, patients, cards and solved examples, that every prompt shared by two runs has the same text, that each run is complete, and that each model's revision is the registered one. A missing run enters its tests with p = 1.
- **Reporting.** The results of R15, R16 and R17 are reported in the paper whatever they are, with the registered decision rules, and the answers to Q18 to Q22 in its appendix. Analyses that are not registered here are labelled as post hoc.
- **Changes after registration.** Only engineering settings that do not touch prompts, readings, patients, models or analysis may change without a deviation. Any other change is recorded as a dated deviation, and the tests it affects are marked in the report.

## 9. Order of work

1. Engineering check (Section 10) on dev patients, before registration.
2. Record its result in Section 10 and in `ler_v5.py` (the decisions, the fingerprints of the fresh samples and the hash of the check's output), freeze the code and the notebooks, register this file on OSF, and create the release `prereg-v5`.
3. One run per model, one Kaggle session each, in this order: Qwen3-8B, Falcon3-7B-Instruct, Yi-1.5-9B-Chat-16K, Granite-3.3-8B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Olmo-3-7B-Instruct. The models of R15 to R17 come first, so that the confirmatory tests do not depend on the last runs. The engineering check measured 0.48 to 0.74 seconds per scored prompt (Section 10). In the third and fourth studies, the runs took about half the time per prompt that their checks measured, because they score much larger batches. With 47,964 scored prompts per model (38,828 for Qwen3-8B, which does not run the True/False grid) and the generated answers, a run is expected to take about 4 to 7 hours, and the study about 30 to 40 GPU hours. Granite 3.3's chat template adds the date of the run to its system text, so a Granite run that is resumed on another day starts its scoring again: one run never mixes two dates.
4. One run of the analysis notebook.

## 10. What was known before registration, and the engineering check

- **Known:** all results of the four studies and of the scale checks, and the post-hoc analyses of Section 1.
- **Engineering check.** Each model of Section 4 is downloaded and run on dev-sample prompts only (never on a patient of Section 3): per dataset, 6 dev patients × every prompt of the grid (42), a timing set of 100 prompts (every dev patient, C2, alarm framing), 12 generated answers (2 dev patients × C0, C2 and C4 × both framings), and the six obvious questions of the second and fourth studies (two with Yes/No, two with True/False and two with A/B). The check records only technical quantities: whether the model loads in float16 with vLLM 0.26.0 on two T4 GPUs, the model window, the chat template and its default system text, whether "Yes", "No", "True", "False", "A" and "B" have single-token variants, the exact-reading check of Deviation D2 for each pair of answer words, whether every reading is finite, the median answer-word mass in every grid and framing, the share of imputed readings, prompt lengths, the time per scored prompt, and, for the generated answers, the time and the share of parsed answers. It computes no accuracy, no answer direction, no agreement of generated answers with the reading, and no asymmetry, and it saves no per-prompt reading. It also builds the fresh samples of Section 3 and records their sizes and fingerprints, without scoring them.
- **Rules applied to the check, set before it is run.** (i) A model is not run if its yn grid has a median mass below 0.5 or a reading that is not finite, or if "Yes" and "No" have no single-token variant. (ii) An optional grid (pa, tf, ab, dose, prob) is not run for a model if, in any of its framings, the median mass is below 0.5 or a reading is not finite, or if its answer words have no single-token variant; its cells then enter the tests with p = 1 and its questions are reported as not run. (iii) If the projected time of the generated answers over all models that are run exceeds 8 hours at 150 patients per class, the generation subsample has 100 patients per class. (iv) A model that answers an obvious question the wrong way round is reported; nothing is changed for it, because the obvious questions are not part of the validity rules.
- **First check** (2026-11-04, Kaggle, two T4 GPUs, 79 minutes; 728 dev prompts; `ler_v5.py` v5-0.1, SHA-256 b9aeb9b2dca6cdddff343ac35b6647969920e6aa68cb746388d0b4e03d2a2e64). The check confirmed the code hashes and rebuilt the earlier dev and test samples exactly (all fingerprints matched). Median answer-word mass on the check prompts of each grid (first framing / second framing; for the dose grid the scales 0.5 / 1 / 2), the share of readings with an imputed variant and the time per prompt on the timing prompts, and the share of parsed generated answers:

| Model | yn | pa | tf | ab | dose | prob | Imputed | Seconds per prompt | Parsed |
|---|---|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | 0.988 / 0.990 | 0.990 / 0.991 | 0.987 / 0.989 | 0.998 / 0.998 | 0.990 / 0.990 / 0.989 | 0.986 / 0.989 | 0% | 0.50 | 14 of 24 |
| Qwen/Qwen2.5-7B-Instruct | 0.998 / 1.000 | 1.000 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 0.998 / 0.998 / 0.997 | 0.969 / 0.998 | 17% | 0.48 | 24 of 24 |
| Qwen/Qwen3-8B | 0.777 / 0.769 | 0.655 / 0.512 | 0.000 / 0.000 | 0.732 / 0.536 | 0.846 / 0.848 / 0.854 | 0.873 / 0.952 | 22% | 0.61 | 24 of 24 |
| allenai/Olmo-3-7B-Instruct | 0.997 / 0.996 | 0.997 / 0.997 | 0.999 / 0.999 | 1.000 / 0.999 | 0.997 / 0.996 / 0.996 | 0.999 / 0.999 | 0% | 0.53 | 24 of 24 |
| ibm-granite/granite-3.3-8b-instruct | 0.981 / 0.971 | 0.893 / 0.929 | 0.986 / 0.985 | 0.998 / 0.998 | 0.957 / 0.966 / 0.973 | 0.788 / 0.691 | 4.5% | 0.74 | 23 of 24 |
| tiiuae/Falcon3-7B-Instruct | 0.999 / 0.999 | 0.999 / 0.998 | 0.998 / 0.999 | 1.000 / 1.000 | 0.999 / 0.999 / 0.999 | 1.000 / 0.999 | 0% | 0.52 | 24 of 24 |
| 01-ai/Yi-1.5-9B-Chat-16K | – | – | – | – | – | – | – | – | – |

Every reading of the six models that were read was finite, each of them has single-token variants of all six answer words, and each answered the six obvious questions the right way round. The exact-reading check of Deviation D2 gave largest differences between 0.008 and 0.078 (Qwen3-8B, Yes/No). The longest prompts have between 2,234 and 2,543 tokens. Qwen3-8B puts almost no probability on "True" and "False" (median mass 0.0003 and 0.0001), so its True/False grid fails rule (ii); its paraphrased and A/B grids pass, with median masses of 0.51 to 0.73. In free generation, Olmo 3 and Falcon3 began every answer with "Yes" or "No", Qwen3-8B wrote "Answer: Yes" or "Answer: **Yes**", and Llama 3.1, Qwen2.5 and Granite 3.3 often began with an analysis ("To predict in-hospital mortality for the given patient, we need to analyze …"). Granite 3.3's chat template adds the current date to its default system text ("Today's Date: November 04, 2026"). The fresh samples are those of Section 3; 1 PhysioNet 2012 patient and 7 PhysioNet 2019 patients have a probability of exactly 50%.

  Yi-1.5-9B-Chat-16K was not read: 23 seconds into its download, writing to the model cache failed with `OSError(30, 'Read-only file system')` in `/tmp/hf_cache`. This is a fault of the Kaggle session, not of the model, and it is not covered by rule (i), which concerns the model's readings.
- **Change after the first check, before the second.** The engineering-check notebook tests that the model cache can be written before each download, moves it to another folder if it cannot, and records the free disk space. The run notebook does the same. `ler_v5.py` is not changed for the second check. The rule added here: a model whose check stops for a reason of the session (download, disk or network) rather than its readings is checked again in a fresh session before registration, and the second check replaces the first for that model.
- **Second check** (2026-11-04, Kaggle, two T4 GPUs, 21 minutes; Yi-1.5-9B-Chat-16K only; the same 728 dev prompts, `ler_v5.py` v5-0.1 with the same SHA-256 as in the first check). The model cache could be written (1,034 GB free), the code hashes and all earlier fingerprints matched, and the check built the same fresh samples as the first check. Yi-1.5-9B-Chat-16K loaded with a window of 16,384 tokens, has single-token variants of all six answer words, answered the six obvious questions the right way round, and every reading was finite. Median answer-word mass: yn 0.999 / 0.998, pa 0.998 / 0.998, tf 1.000 / 0.999, ab 0.999 / 0.999, dose 0.999 / 0.999 / 0.999, prob 0.999 / 0.999; no imputed reading on the timing prompts; 0.75 seconds per prompt; exact-reading differences 0.024 (Yes/No), 0.030 (True/False) and 0.027 (A/B); the longest prompt has 2,441 tokens. Its chat template adds no system text. All 24 generated answers were parsed, but they took 2.2 seconds each: in two of the three answers that the output keeps, the model writes its answer word, then the text "<|endoftext|>", and continues with a new task until the limit of 512 tokens. The parsing rule reads the first word, so this does not change the parsed answer.
- **Decisions** (rules (i) to (iii), on the results of both checks, with the second check replacing the first for the model it repeated). (i) All seven models are run. (ii) Qwen3-8B does not run the True/False grid (34 scored prompts per patient); every other model runs all six grids (42). (iii) The projected time of the generated answers at 150 patients per class is 9.14 hours over the seven models (from 0.29 seconds per answer for Falcon3 to 2.33 for Llama 3.1), above the limit of 8 hours, so the generation subsample has 100 patients per class. (iv) No model answered an obvious question the wrong way round. These decisions are `ENGCHECK_DECISIONS` in `ler_v5.py`, which the run notebook uses; the tests check that the rules applied to the output of the check give exactly these decisions.
- **Outputs.** The first check's output is `results/v5/engcheck_v5.zip` (SHA-256 a2adc60251ffac8d078c5e7fe6d148a54f913a004bfa6e67835cc56bae91b0b7). The second check's output is `results/v5/engcheck_v5_run2.zip` (SHA-256 6f44b4bd695bfc763490041323fdc9f49489d631d0ea126cfc59bb0da2fc4b01). Each contains the `ler_v5.py` that produced it.
- **Registered files** (all in `code/v5/` at the tag `prereg-v5`; SHA-256):

| File | SHA-256 |
|---|---|
| `ler_v5.py` (v5-1.0) | a976d88ac60e534dfb4fab9a9ab80d152621f60c30014f9c6c1e4281223434f8 |
| `notebooks/llm_evidence_reliance_main_run_v5.ipynb` | 2c3ae148586ae900ff898a2bc837a21a51912afe1c8a7e6bd7aff3cfc5d08726 |
| `notebooks/llm_evidence_reliance_analysis_v5.ipynb` | 0fb87be689479fcd406b08b2cff5d8d869b79e2250cd1ce5bd89e3f88461b9ed |
| `notebooks/llm_evidence_reliance_engcheck_v5.ipynb` (first check, as run) | 2bb27400a9ce3eb8c79a14ea8f62a79c7c8c35da2d621c44482f1b97b8374480 |
| `notebooks/llm_evidence_reliance_engcheck_v5_run2.ipynb` (second check, as run) | 92cbdbb323cc0c7be13c830798cb403efe8eb1cce726077b067c7f5137d839f8 |

  `ler_v5.py` v5-1.0 differs from v5-0.1, which ran both checks, only by the run and analysis functions (the merge of the two checks, the diagnostics of Q21 and Q22, the imputation bound, the protocol addresses), the decisions, fingerprints and hashes recorded above, its version string and its description. The run notebook stops if the SHA-256 of its `ler_v5.py` is not in this file. Before registration, the code was tested on the CPU with a fake model whose effects are known: unit tests of every prompt, sample, estimator and decision rule; the engineering check with a stand-in for vLLM; and the whole study (seven runs, one of them stopped and resumed, and the analysis), which recovered every effect built into the fake model.

## References

- First study: `PREREGISTRATION.md`, releases `prereg-v1` to `prereg-v1.3`.
- Second study: `PREREGISTRATION_V2.md`, release `prereg-v2`.
- Third study: `PREREGISTRATION_V3.md`, release `prereg-v3`.
- Fourth study: `PREREGISTRATION_V4.md`, release `prereg-v4`.
- PhysioNet/Computing in Cardiology Challenge 2012, version 1.0.0 (sets A, B and C).

## Deviations

### V5-D1 (2026-11-08): order of the runs

Section 9 registers this order of the runs: Qwen3-8B, Falcon3-7B-Instruct, Yi-1.5-9B-Chat-16K, Granite-3.3-8B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Olmo-3-7B-Instruct. The runs were started in this order instead (start times in UTC, from the timestamps in the run outputs): Qwen3-8B (5 November, 06:00), Falcon3-7B-Instruct (5 November, 10:00), Llama-3.1-8B-Instruct (5 November, 10:26), Olmo-3-7B-Instruct (5 November, 17:34), Yi-1.5-9B-Chat-16K (5 November, 17:35), Qwen2.5-7B-Instruct (6 November, 03:16) and Granite-3.3-8B-Instruct (7 November, 04:44). Up to two runs were in progress at the same time.

The order was registered so that the confirmatory tests would not depend on the last runs if the GPU time ran out. All seven runs finished, each in one session and without a restart, with the registered code (`ler_v5.py` v5-1.0), the protocol at `prereg-v5`, the registered model revisions, the fresh samples of Section 3 and the decisions of Section 10. The order therefore changes no prompt, reading, patient or analysis. The runs took 28.96 GPU hours in total, from 3.32 hours (Qwen3-8B) to 5.97 hours (Yi-1.5-9B-Chat-16K).

This deviation was written on 8 November 2026, after all runs had finished and their technical records (revisions, hashes, fingerprints, completeness, invalid and imputed readings, and the registered validity rules) had been checked, and before the analysis notebook was run. No result of a registered test was known when it was written.
