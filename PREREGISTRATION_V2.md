# Pre-registration, second study: does the pull toward reassuring evidence replicate, and what drives it?

- **Author:** Jhishan Deb
- **Registered:** October 2026, committed and tagged `prereg-v2` before any test prompt of this study was scored (the GitHub commit and the `prereg-v2` release date are the authoritative timestamps)
- **Builds on:** the first study, registered in `PREREGISTRATION.md` (releases `prereg-v1` to `prereg-v1.3`), with its results committed in `results/main/` (commit c6bf787)
- **Status:** written after the first study's results were known (they motivate every prediction below). No test patient has been scored by any model of this study. Answer-option prompts were scored only on dev patients, in the engineering check of Section 9, and only the technical quantities listed there were computed from them.
- **Changes:** any change after the tag is listed in the "Deviations" section at the end, with date and reason. This file is never edited silently.

In this file, **reassuring evidence** means evidence pointing to the safe outcome (survival, no sepsis, no readmission), and **alarming evidence** means evidence pointing to the alarming outcome.

## 1. What the first study found

The first study predicted that alarming evidence would move an LLM more than reassuring evidence (H3). The data showed the opposite. In all five valid instruct cells (Qwen2.5-7B-Instruct on PhysioNet 2012 and 2019; Llama-3.1-8B-Instruct on all three datasets), both asymmetry statistics of Deviation D1 were below 0, with Holm p < 0.001: reassuring evidence moved the models more. The first study's originally registered measure, Δ = A − S, pointed the same way in four of the five cells.

Two further results shape this study:

- **Outcome or answer word (Q3).** With the reversed question ("Will this patient survive?"), the asymmetry followed the safe outcome in one cell (Qwen, PhysioNet 2012), followed the answer word "No" in another (Qwen, PhysioNet 2019), and was unresolved in the three Llama cells.
- **Instruction tuning (H4).** For Qwen, the asymmetry toward the safe outcome was larger in the instruct model than in the base model (PhysioNet 2012 and 2019, Holm p ≤ 0.001).

## 2. Research questions

1. **R1.** Does the pull toward reassuring evidence replicate in instruction-tuned models from other developers and a newer generation?
2. **R2.** Does it follow the outcome, or the Yes/No question format (in particular the word "No")? A format in which the two outcomes are offered as options A and B separates the two.
3. **R3.** Does post-training increase it, and at which stage of post-training does it appear?

## 3. What is unchanged from the first study

Everything below is exactly as registered in `PREREGISTRATION.md` with Deviations D1 to D5, and is implemented by the same code. `ler_main.py` version main-2.0 keeps every function of main-1.0; the test `tests/test_v2_equivalence.py` confirms that every first-study prompt is byte-identical, and `tests/test_v1_analysis_unchanged.py` confirms that the first-study analysis gives identical numbers on the first-study records.

- The three datasets, the splits (seed 2027), the eligibility rule, the XGBoost model, the 16 features, the evidence cards (top 5 SHAP values), the conditions C0, C2, C4 and C6, the solved examples (sets A, B and C), the Yes/No prompts, and the alarm and reversed framings.
- The **same test patients**: 300 per class per dataset, 1,800 in total. The second study generalizes across models, not across patients.
- The reading of the next-token log-probabilities after "Answer:", from the top 1,000 log-probabilities with the imputation rule of Deviation D5. The exact-reading check of Deviation D2 still runs and is reported. Every run uses vLLM 0.26.0; there is no fallback to another backend.
- The asymmetry statistics a and a′ (Deviation D1), the intersection–union test p = max(p_a, p_a′), the paired bootstrap over patients (10,000 resamples, seed 2026), the validity rule (median answer-word mass ≥ 0.5 and C0 accuracy ≥ 0.55, except where Section 6 says otherwise for R3), the treatment of invalid readings (Deviation D3, item 5), and the memorization check (Deviation D3, item 7). The memorization check runs for every model new to this study. It does not run again for the two first-study models, which were checked in the first study.
- No system message is added. A chat template's own default system text is kept and logged (Deviation D3, item 6).

## 4. Models

| Model | Role | Kind | Prompts per patient | Revision |
|---|---|---|---|---|
| allenai/Olmo-3-7B-Instruct | new (R1, R2); final Olmo 3 stage (R3) | instruct | Yes/No grid (14), answer-option grid (6), example-order control (3), base-format control (3) | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc |
| Qwen/Qwen3-8B | new (R1, R2) | instruct, thinking mode off | Yes/No grid (14), answer-option grid (6), example-order control (3) | b968826d9c46dd6066d109eabc6255188de91218 |
| ibm-granite/granite-3.3-8b-instruct | new (R1, R2) | instruct | Yes/No grid (14), answer-option grid (6), example-order control (3) | 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b |
| allenai/Olmo-3-1025-7B | Olmo 3 base (R3) | base | Yes/No grid (14) | a81bae42db3975be1671e27b9c9a56da1a9f980f |
| allenai/Olmo-3-7B-Instruct-SFT | Olmo 3 after supervised fine-tuning (stage trace) | instruct | Yes/No grid (14) | e1452fc572d51966ff4aaeb25118b891eb93e549 |
| allenai/Olmo-3-7B-Instruct-DPO | Olmo 3 after preference tuning (stage trace) | instruct | Yes/No grid (14) | b33130b7de49f0c2553b5c2b3bc8409ff3e627d1 |
| Qwen/Qwen2.5-7B-Instruct | first study (R2) | instruct | answer-option grid (6), example-order control (3) | a09a35458c702b33eeacc393d103063234e8bc28 |
| meta-llama/Llama-3.1-8B-Instruct | first study (R2) | instruct | answer-option grid (6), example-order control (3) | 0e9e39f249a16976918f6564b8830bc894c89659 |

- The **Yes/No grid** is the first study's grid: C0, C2, C4 and C6 × alarm and reversed framing × example set A, plus C0, C2 and C4 × alarm framing × example sets B and C (25,200 prompts per model). The **answer-option grid** is C0, C2 and C4 × both option orders × example set A (10,800 prompts per model). The two **controls** are described in Section 7 (5,400 prompts each).
- Olmo 3 is chosen because its developer publishes the base model and each post-training checkpoint (supervised fine-tuning, then preference tuning, then reinforcement learning with verifiable rewards, which gives the final instruct model), so one model can be followed through post-training.
- Qwen3-8B is run in its documented non-thinking mode (`enable_thinking=False` in the chat template), so that it answers directly, as the other models do.
- The third new model was planned to be Ministral-3-8B-Instruct-2512 (Mistral AI). It could not be loaded by vLLM 0.26.0 in the engineering check, in either repository, so the first eligible backup, Granite 3.3 8B Instruct (IBM), takes its place (Section 9). Granite's chat template inserts the current date into its default system text; the date of each run is therefore logged, and a run that is resumed on a later day scores all its prompts again, so that every prompt of a run carries the same date.
- All models run in float16 with vLLM 0.26.0 on two T4 GPUs, as in the first study.

**If a model cannot be run after registration.** A model "cannot be run" only if vLLM 0.26.0 fails to load it, or if more than 20% of its first 256 or more readings are invalid (the run notebook then stops). A model that runs but fails a validity rule is never replaced: its cells enter the tests with p = 1. No eligible backup model remains after the engineering check (Section 9), so if any model cannot be run, the tests that need it are reported as not testable, and this is recorded as a deviation.

## 5. Answer-option framing

The question names both outcomes as options A and B, so that neither outcome is tied to the word "Yes" or "No". Each patient is asked twice, once in each order of the options. Averaging the two orders cancels, to first order, a preference for a letter or for a position.

The question, word for word (PhysioNet 2012, alarming outcome as option A):

```
Question: Which of these two outcomes is more likely for this patient?
A: The patient dies during the current hospital stay.
B: The patient survives the current hospital stay.
Answer with A or B.
```

The other order swaps the two option lines. The solved examples carry the same question and end with `Answer: {A or B}`, the letter of the example's true outcome. Everything else in the prompt (task line, examples, patient, evidence card) is identical to the alarm-framing prompt of the same patient and condition. The options for all datasets:

| Dataset | Alarming outcome | Safe outcome |
|---|---|---|
| PhysioNet 2012 | The patient dies during the current hospital stay. | The patient survives the current hospital stay. |
| PhysioNet 2019 | The patient develops sepsis later in this ICU stay. | The patient stays free of sepsis for the rest of this ICU stay. |
| Diabetes 130 | The patient is readmitted within 30 days of discharge. | The patient avoids readmission within 30 days of discharge. |

Two features stay tied to the outcome in every framing and are stated as limits of R2: the evidence card always speaks of "the risk of" the alarming outcome, and two of the three safe options use negative wording ("free of", "avoids"). PhysioNet 2012, where the safe option is affirmative ("survives"), is reported separately in any case.

**Reading.** The log-probabilities of the single-token variants of "A" and "B" (with and without a leading space) are read exactly as "Yes" and "No" are in the first study (top 1,000 with the imputation rule of Deviation D5). In each order, the log-odds of the alarming outcome is log P(A) − log P(B) when option A is the alarming outcome, and log P(B) − log P(A) otherwise. For each patient and condition, the analysed log-odds is the mean of the two orders. A patient whose reading is invalid in either order is left out for that condition (Deviation D3, item 5).

**Validity of an answer-option cell.** The same rule as in the first study, on the answer-option records: the median A+B mass over all answer-option records of the cell is at least 0.5, and the accuracy of the averaged C0 reading is at least 0.55.

## 6. Hypotheses and tests

### R1. Replication: the pull toward reassuring evidence

- **Prediction.** For each new instruct model (Olmo-3-7B-Instruct, Qwen3-8B, Granite-3.3-8B-Instruct) and each dataset: a < 0 and a′ < 0, in the alarm framing with example set A.
- **Test.** The first study's intersection–union test: p = max(p_a, p_a′), with both estimates required to be below 0. Holm correction over the 9 cells (3 models × 3 datasets). A cell that fails the validity rule, or a missing run, enters with p = 1.
- **Decision rule.** Replicated if, for at least 2 of the 3 new models, the predicted direction holds with Holm p < 0.05 in at least 2 of the 3 datasets (the **R1 per-model criterion**). The result for each model is reported either way.
- **If not replicated,** the first study's asymmetry is described as not shown to generalize beyond the first study's models. A model whose cells fail validity is reported as untested, not as failing to replicate.
- Reported with the same prominence, as in the first study: Δ = A − S next to a and a′, with the D1 sign rule (a cell where a, a′ and Δ disagree in sign is described as not robust; this changes no decision).

### R2. Outcome or Yes/No question format?

- **Prediction.** The pull follows the outcome: in the answer-option framing, a < 0 and a′ < 0.
- **Eligible models.** R2 asks what drives an asymmetry that exists, so it is evaluated on the models whose Yes/No asymmetry is established: the two first-study instruct models, and each new model that meets the R1 per-model criterion. With k eligible models (2 to 5), there are 3k cells.
- **Test.** The intersection–union test on the averaged answer-option reading. Holm correction over the 3k cells. A cell that fails the answer-option validity rule enters with p = 1.
- **Decision rule.** Supported if, for a majority of the eligible models (more than k/2), the predicted direction holds with Holm p < 0.05 in at least 2 of the 3 datasets.
- **Secondary comparison.** For each cell, the change between framings δ = a(answer option) − a(alarm Yes/No), and the same for a′, with a paired bootstrap over the patients present in both (10,000 resamples, seed 2026) and the intersection–union p. For the two first-study models, the Yes/No readings are those of the first study (same patients, same prompts, same reading). A cell **counts** for the interpretation only if it is valid in both framings and a and a′ are both below 0 in the alarm Yes/No framing. A counted cell is **weaker with answer options** if δ > 0 for both statistics with unadjusted intersection–union p < 0.05. A model shows this pattern if more than half of its counted cells (at least one) are weaker with answer options.
- **Interpretation.** R2 supported, and a majority of the eligible models show the pattern: "follows the outcome, and is larger with the Yes/No question". R2 supported otherwise: "follows the outcome". R2 not supported, and a majority of the eligible models show the pattern: "depends on the Yes/No question format". Otherwise: "unresolved". The answer-option format differs from the Yes/No question in more than the answer word (it names both outcomes and asks a comparative question), so the interpretation speaks of the question format; the first study's reversed-framing table (Q3) remains the test within the Yes/No format and is reported for every new model.

### R3. Post-training in Olmo 3

- **Prediction.** Post-training increases the pull toward reassuring evidence: a(Olmo-3-7B-Instruct) − a(Olmo-3-1025-7B) < 0 and the same for a′, per dataset (alarm framing, example set A).
- **Validity for R3 and the stage trace.** Median Yes+No mass ≥ 0.5 and C0 AUROC ≥ 0.55 (alarm framing, example set A). AUROC replaces accuracy here because a strong lean of a base model lowers its accuracy without making its readings uninformative (in the first study, Llama base on PhysioNet 2019 had C0 accuracy 0.500 but AUROC 0.631), and the statistic a is built to absorb a lean. The first-study rule is reported next to it.
- **Test.** The first study's H4 procedure with the direction reversed: an intersection–union test on the two differences, with a paired bootstrap over the patients scored by both models. Holm correction over the 3 datasets. A dataset where either model fails the R3 validity rule enters with p = 1.
- **Decision rule.** Supported if the predicted direction holds with Holm p < 0.05 in at least 2 of the 3 datasets. If either Olmo model fails the R3 validity rule in two or more datasets, R3 cannot be supported, and this is reported as such.
- **Prompt format.** The base model is prompted as plain text followed by "Answer:", the other checkpoints through their chat template. R3 therefore compares post-training together with prompt format; the base-format control (Section 7) separates the two. The run logs record each template's default system text, and the report states whether the SFT, DPO and final templates are identical.
- **Stage trace (described, no decision rule).** For each dataset: a and a′ with 95% intervals for the base model, the SFT checkpoint, the DPO checkpoint and the final instruct model, and the change at each step (SFT − base, DPO − SFT, final − DPO) with paired bootstrap intervals. Stages that fail the R3 validity rule are shown and marked. The step with the largest change in a is described, not tested.

## 7. Analysis

- **Code.** `ler_main.py` main-2.0, the notebook `llm_evidence_reliance_main_run_v2.ipynb` (one run per model) and the notebook `llm_evidence_reliance_analysis_v2.ipynb` are committed with this file, at the tag `prereg-v2`, before any run. Each run checks this file at the tag (every prompt string, model id and pinned revision must appear in it) and stops otherwise. Each run downloads exactly the pinned revision and stops if it differs.
- **Families.** R1, R2 and R3 are three separate confirmatory questions, each with its own Holm correction, decision rule and α = 0.05.
- **Example-order control (labelled secondary).** In the registered order of the solved examples (alarming, safe, alarming, safe), the last example before the patient always has the safe outcome. Each of the five instruct models is also run with the same four examples in the order safe, alarming, safe, alarming (C0, C2 and C4, alarm framing; 5,400 prompts). Reported: a and a′ in this order, and the change from the registered order with paired bootstrap intervals.
- **Base-format control (labelled secondary).** Olmo-3-7B-Instruct is also prompted in the base model's plain-text format (C0, C2 and C4, alarm framing, example set A; 5,400 prompts). Reported: a and a′ in that format, its difference from the base model (post-training with the format held fixed), and the difference between the chat and the plain format of the final model.
- **Other secondary analyses, reported for each new model as in the first study and labelled:** H1 and H2 (accuracy with true and flipped evidence, exact McNemar, no multiplicity correction), the Q3 table with the reversed framing, Δ = A − S, the unadjusted log-odds asymmetry, follow rates, a with example sets B and C, a with d winsorized at 1% and 99%, the share of patients with an expected baseline within 1 SD of 0, the first-stage R², the imputation bound of Deviation D5 (2,000 resamples), a and a′ in each option order on its own (2,000 resamples), and the mean preference for the letter A.
- **Not repeated:** H5 and H6 of the first study (trust signals) are not part of this study.
- **One analysis.** The analysis notebook is run once, after all eight runs, on the eight run outputs and the first study's outputs of the two first-study instruct models; it stops if either of those is missing. Before it analyses anything, it checks that all runs used the same patients, cards and partners, that every prompt shared by two runs has the same text, that all runs of this study used the same code and protocol text, and that each model's revision is the registered one. A missing second-study run enters its tests with p = 1, and the tests that need it are reported as not testable. The output and the eight run outputs are committed to `results/v2/`.
- **Changes after the tag.** Only engineering settings that do not touch prompts, readings, models or analysis (for example a download retry) may change without a deviation. Any other change is recorded as a dated deviation, and the tests it affects are marked in the report.

## 8. Order of work

1. Engineering check (Section 9), done before this file was tagged.
2. Commit this file, the code and the engineering-check output, and create the release `prereg-v2`.
3. Eight runs, one model per Kaggle session: the three new instruct models, the three earlier Olmo 3 stages, and the answer-option grid and example-order control for the two first-study models. Each run first checks this file at the tag.
4. One run of the analysis notebook.

## 9. What was known before registration, and how the models were chosen

- **Known:** all first-study results (Section 1).
- **Engineering check (2026-10-22, Kaggle, two T4 GPUs, 113 minutes).** Each candidate model was downloaded and run on dev-sample prompts only (never test patients): 108 prompts for the checks and 600 for timing. The check recorded only technical quantities: whether the model loads in float16 with vLLM 0.26.0 on two T4 GPUs, the weights' format, the chat template and its default system text, whether "Yes", "No", "A" and "B" are single tokens, the exact-reading check of Deviation D2, whether every reading is finite, the median answer-word mass, the share of imputed readings, prompt lengths, the time per prompt, and questions with an obvious answer (four for an instruct model, for example "Is the sky usually blue on a clear day?"; the two Yes/No questions for the base model). It computed no accuracy, no answer direction and no asymmetry, and it saved no per-prompt reading. The full output is committed as `results/v2/engcheck_v2.zip` (SHA-256 068c0351ae00111c03d7d8e95a3a1a73c716cfe0c50f713c0d0b6e4c1fb82a4d); the code that produced it is the `ler_main.py` inside that file (SHA-256 a6a1a3d6c12390c03f98eadfffe64c7fda09e0cf2e5352294f63d83ccbee6e74). The pinned revision of each model is the revision the check loaded. The answer-option wording was not changed after the check.
- **Eligibility.** A model is eligible if, in each answer format it will be run with, it loads, every reading is finite, the median answer-word mass is at least 0.5, and the obvious questions are answered the right way round, and if its weights are not quantized. This is the `ok` field computed by `engineering_check()`, with the formats listed in its `formats_required` field, with one exception stated below.
- **Candidates and backup order (set before the check).** Ministral 3 in its BF16 repository and in its main repository; backups, in order: ibm-granite/granite-3.3-8b-instruct, then mistralai/Mistral-7B-Instruct-v0.3. Gemma models were not considered, because they are known to fail in float16, the only format the T4 GPUs support.

**Result of the check** (median answer-word mass on the 300 timing prompts; share of readings with an imputed variant):

| Model | Eligible | Yes/No mass | A/B mass | Imputed, Yes/No | Imputed, A/B | Note |
|---|---|---|---|---|---|---|
| allenai/Olmo-3-7B-Instruct | yes | 0.997 | 0.999 | 0% | 0.3% | |
| allenai/Olmo-3-7B-Instruct-DPO | yes | 0.979 | (0.995) | 0% | (0.3%) | run with Yes/No only |
| allenai/Olmo-3-7B-Instruct-SFT | yes | 0.994 | (0.999) | 0% | (0%) | run with Yes/No only |
| allenai/Olmo-3-1025-7B | yes, by the exception below | 0.796 | – | 0% | – | one obvious question answered the wrong way round |
| Qwen/Qwen3-8B | yes | 0.856 | 0.649 | 16% | 33% | exact-reading check not passed (0.064, 0.110) |
| mistralai/Ministral-3-8B-Instruct-2512-BF16 | no | – | – | – | – | vLLM could not load it (image processor error) |
| mistralai/Ministral-3-8B-Instruct-2512 | no | – | – | – | – | FP8 weights; vLLM could not load it (same error) |
| ibm-granite/granite-3.3-8b-instruct | yes | 0.987 | 0.997 | 5% | 4% | takes the third place |
| mistralai/Mistral-7B-Instruct-v0.3 | no | 0.906 | 0.999 | 29% | 23% | one obvious A/B question answered the wrong way round |
| Qwen/Qwen2.5-7B-Instruct | yes | (0.998) | 1.000 | (15%) | 75% | first-study model, run with A/B only |
| meta-llama/Llama-3.1-8B-Instruct | yes | (0.990) | 0.998 | (0%) | 7% | first-study model, run with A/B only |

Values in parentheses are for a format the model is not run with. The exact-reading check of Deviation D2 failed for some models; as in the first study, every run reads the top 1,000 log-probabilities (Deviation D5), and the imputation bound of D5 is reported for every cell with imputed readings.

**Exception, decided after the check and disclosed here.** The base model Olmo-3-1025-7B answered the Yes/No question "Is the sky usually blue on a clear day?" with "No" (its other obvious question was answered correctly). These questions are asked without solved examples, which a base model is not trained to answer, while every registered prompt contains four solved examples. The questions exist to detect a broken or reversed reading of the answer words. Olmo-3-1025-7B uses the same tokenizer and the same "Yes" and "No" token ids (9642, 7566, 2822, 2360) as the three post-trained Olmo 3 checkpoints, which answered all four obvious questions correctly, so its reading is not broken. The base model is therefore treated as eligible. Without this exception, R3 and the stage trace could not be run at all.

## References

- Olmo 3: Team Olmo, Allen Institute for AI (2025). Olmo 3. arXiv 2512.13961.
- First study: `PREREGISTRATION.md` in this repository, releases `prereg-v1` to `prereg-v1.3`; results in `results/main/`.

## Deviations

None yet.
