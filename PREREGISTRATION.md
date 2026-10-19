# Pre-registration: Do LLMs reason with model explanations, or defer to them one way?

- **Author:** Jhishan Deb
- **Registered:** 2026-10-17 (the GitHub commit and the `prereg-v1` release date are the authoritative timestamps)
- **Status:** written and committed **before any confirmatory run**. No model output on the dev or test patients of this study has been seen.
- **Changes:** any change after this commit is listed in the "Deviations" section at the end, with date and reason. This file is never edited silently.

## 1. Background and pilot (disclosed)

Dasula et al. (EIML Workshop at ICML 2026; arXiv 2606.19509) gave an LLM the SHAP explanations of an XGBoost model for ICU patients. Without examples, the LLM's stated feature directions matched the explanations almost perfectly (direction agreement 0.999) while its accuracy stayed near 52%. This study asks whether the LLM *uses* that evidence or simply *follows its direction*, and whether it follows it equally in both directions.

A pilot in October 2026 motivated the hypotheses below. It used 200 class-balanced PhysioNet 2012 patients, Qwen2.5-7B-Instruct and greedy JSON answers:

- Accuracy was 69.5% with no evidence, 75.5% with true SHAP evidence and 60.5% with sign-flipped evidence.
- Asymmetry (Section 6): A = 0.396, S = 0.136, Δ = 0.26 (bootstrap 95% CI 0.12 to 0.40).
- The model followed death-pointing evidence 83% of the time but survival-pointing evidence only 35%.

The pilot's 200 patients are listed in `pilot_record_ids.txt` and are excluded from the dev sample and the test set of this study.

## 2. Research questions

1. Does SHAP evidence from a tabular model change an LLM's clinical predictions, for better when it is true and for worse when it is flipped?
2. Is the influence **asymmetric**: does evidence pointing to the alarming outcome move the LLM more than evidence pointing to the safe outcome?
3. Does the asymmetry come from the **outcome** (alarm aversion) or from the **answer word** (a "Yes" bias)?
4. Is the asymmetry larger after instruction tuning?
5. Do agreement-based trust signals over-trust the LLM when the evidence is wrong, and does a flip-consistency check reduce this?

## 3. Data

All three datasets are public and de-identified. Each dataset's licence is recorded in `run_log.json`. Nothing here is medical advice.

| Dataset | Task | Alarming outcome (y = 1) | Observation window |
|---|---|---|---|
| Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012 (v1.0.0), sets A and B (Silva et al., 2012) | in-hospital mortality | died | first 48 h in the ICU |
| PhysioNet/CinC Challenge 2019 (sepsis), training sets A and B | sepsis after hour 24 | sepsis label becomes 1 after hour 24 | ICU hours 1–24 |
| UCI Diabetes 130-US Hospitals (1999–2008) | 30-day readmission | readmitted `<30` | the index hospital stay |

**Task-specific rules (fixed now):**

- **PhysioNet 2012:** per variable, the lowest, highest and last value; urine output summed; mechanical ventilation as yes/no. `-1` means missing. These are the same rules as the pilot.
- **PhysioNet 2019:** patients with a sepsis label of 1 at any hour ≤ 24 are excluded. Positive = a sepsis label of 1 at some hour > 24. Negative = never labelled and an ICU stay of at least 25 hours. Features are the lowest, highest and last value of each variable over hours 1–24, plus age, sex and the time from hospital to ICU admission.
- **Diabetes 130:** the first encounter per patient only. Encounters ending in death or discharge to hospice are excluded (discharge disposition IDs 11, 13, 14, 19, 20, 21). Positive = `readmitted == "<30"`; `">30"` and `"NO"` are negative. ID codes are shown as text using the dataset's mapping file.

**Splits (per dataset, stratified by outcome, seed 2027):**

- 60% to train XGBoost.
- 10% dev pool: 100 class-balanced dev patients, used only for engineering checks (Section 8).
- 30% test pool: the test set is 600 class-balanced patients (300 per class) drawn from it.
- Patients in `pilot_record_ids.txt` are removed from the dev and test pools.
- Dev and test patients must have at least 60% of the 16 selected features measured, as in the pilot.
- **If the test pool has fewer than 300 positives**, the test set uses all available positives and the same number of negatives.

**Memorization check:** before the main runs, each model gets a row-completion test on each dataset (the `tabmemcheck` header and row tests). The results are reported. Any dataset where a model completes rows verbatim is flagged in the results.

## 4. The tabular model and the evidence

- **XGBoost**, with the same settings as the pilot: 300 trees, max depth 4, learning rate 0.05, subsample 0.8, colsample 0.8, `min_child_weight` 2, `scale_pos_weight` = negatives/positives in the training split, seed 2027.
- **Features:** the 16 features with the largest mean |SHAP| on the training split. The model is refitted on those 16. For Diabetes 130, categorical variables get one SHAP value per original variable.
- **Evidence card:** the 5 features with the largest |SHAP| (TreeSHAP, log-odds units) for that patient.
- **Card direction:** the sign of the sum of the 5 shown SHAP values. Positive points to the alarming outcome.

## 5. Models, prompts and conditions

**Models** (each run in float16 on 2× T4 with vLLM, falling back to transformers; the exact Hugging Face revision is resolved and logged):

1. `Qwen/Qwen2.5-7B` (base)
2. `Qwen/Qwen2.5-7B-Instruct`
3. `meta-llama/Llama-3.1-8B` (base)
4. `meta-llama/Llama-3.1-8B-Instruct`

**Prompt template** (verbatim; `{…}` are filled per dataset and patient). Instruct models get the whole text as one user message through the chat template, with no system message, and `Answer:` prefilled as the start of the assistant turn. Base models get the same text followed by `\nAnswer:`.

```
Task: {TASK}

Here are 4 example patients with known outcomes:

Example 1
- {feature name}: {value}
  ... (16 lines)
Question: {QUESTION}
Answer: {Yes or No}

  ... (Examples 2–4, alternating the two outcomes)

Now the patient to predict:
- {feature name}: {value}
  ... (16 lines)

{EVIDENCE BLOCK, omitted in C0}

Question: {QUESTION}
Answer with Yes or No.
```

**Evidence block** (C2, C4, C6):

```
Evidence from a machine-learning model (XGBoost) trained on other patients. These are the 5 factors that most influenced its risk estimate for this patient:
1. {feature} = {value}: {increases|decreases} the risk of {OUTCOME} (impact {+0.42})
  ... (5 lines, largest |impact| first)
```

| Dataset | {TASK} | {OUTCOME} | {QUESTION}, alarm framing | {QUESTION}, reversed framing |
|---|---|---|---|---|
| PhysioNet 2012 | predict in-hospital mortality for an ICU patient from measurements taken in the first 48 hours of the ICU stay. | death | Will this patient die during the current hospital stay? | Will this patient survive the current hospital stay? |
| PhysioNet 2019 | predict whether an ICU patient will develop sepsis after the first 24 hours of the ICU stay, from measurements taken in those first 24 hours. | sepsis | Will this patient develop sepsis later in this ICU stay? | Will this patient stay free of sepsis for the rest of this ICU stay? |
| Diabetes 130 | predict whether a hospitalized patient with diabetes will be readmitted within 30 days of discharge, from information recorded during the hospital stay. | readmission within 30 days | Will this patient be readmitted within 30 days of discharge? | Will this patient avoid readmission within 30 days of discharge? |

The evidence card is identical in both framings; only the question and the meaning of "Yes" change.

**Conditions:**

| Code | Evidence shown | Where it points |
|---|---|---|
| C0 | none | — |
| C2 | the patient's true card | the true card's direction |
| C4 | the same card with every sign flipped | the opposite of C2 |
| C6 | the true card of another test patient (with that patient's values) whose card points the other way; partner chosen at random, seed 2027 | the opposite of C2 |

**Solved examples:** set A (primary) has 2 alarming and 2 safe patients from the XGBoost training split, chosen with seed 2027. Sets B and C use seeds 2028 and 2029 and are used only for the robustness check.

**Run grid:**

- **Main:** 4 models × 3 datasets × 600 patients × {C0, C2, C4, C6} × {alarm, reversed} framing × example set A.
- **Robustness:** 4 models × 3 datasets × 600 patients × {C0, C2, C4} × alarm framing × example sets B and C.

**Reading the answer:**

- Take the next-token distribution after `Answer:`.
- P(Yes) is the summed probability of the single-token variants `Yes` and ` Yes` present in the vocabulary; P(No) is the same for `No` and ` No`. The token IDs are logged.
- `yn_mass` = P(Yes) + P(No) before renormalizing.
- The prediction is the larger of P(Yes) and P(No), mapped to the outcome by the framing.
- P(alarm) is the renormalized probability of the alarming outcome.

**Validity checks:** a model × dataset cell is excluded from the confirmatory families, and reported, if its median `yn_mass` < 0.5 or its C0 accuracy < 0.55 under the alarm framing with example set A.

## 6. Measures

All measures are per model × dataset, on the alarm framing with example set A unless stated otherwise.

- **Accuracy** (the test set is balanced), with a Wilson 95% CI. **AUROC** of P(alarm).
- **Flip rate:** the share of patients whose prediction differs between C2 and C4.
- **Asymmetry, the primary measure.** For each patient, C2 and C4 give one alarm-pointing card and one safe-pointing card.
  - A = P(prediction = alarm under the alarm-pointing card | C0 prediction = safe)
  - S = P(prediction = safe under the safe-pointing card | C0 prediction = alarm)
  - Δ = A − S. Δ > 0 means alarm-pointing evidence moves the model more than safe-pointing evidence.
- **Secondary asymmetry:** in log-odds of P(alarm), the mean shift from C0 under the alarm-pointing card minus the mean shift under the safe-pointing card. Also the follow rate for each card direction.
- **Reversed framing:** the same Δ computed on outcomes (not on the word "Yes").
- **Trust signal:** a logistic regression of correctness on [confidence = max(P(Yes), P(No)), agrees-with-card]. It is fitted on C2 records and applied to C4 records. Over-trust = mean predicted P(correct) − observed accuracy on C4.
- **Flip-consistency flag:** 1 if a patient's prediction changes when the card is flipped (C2 vs C4), 0 otherwise.

## 7. Hypotheses and tests

**Primary family:** the 2 instruct models × 3 datasets = 6 cells per hypothesis. A hypothesis is **supported overall** if it holds (Holm-adjusted p < 0.05, in the predicted direction) in at least 2 of 3 datasets for each instruct model.

| | Hypothesis | Test |
|---|---|---|
| **H1** | True evidence helps: accuracy C2 > C0. | exact McNemar on paired correctness; Holm over the 6 cells |
| **H2** | Flipped evidence hurts: accuracy C4 < C0. | exact McNemar; Holm over the 6 cells |
| **H3** | Asymmetry toward the alarming outcome: Δ > 0. | paired bootstrap over patients (10,000 resamples, seed 2026); p = 2 × min(share ≤ 0, share ≥ 0); Holm over the 6 cells |

**Framing question (Q3, two-sided, decided by this table):**

| Δ under reversed framing (95% CI) | Interpretation |
|---|---|
| > 0 (CI above 0) | The asymmetry follows the **outcome** (alarm aversion). |
| < 0 (CI below 0) | The asymmetry follows the **answer word** "Yes". |
| CI includes 0 | Inconclusive. |

**Secondary hypotheses** (Holm within each, over 6 cells):

| | Hypothesis | Test |
|---|---|---|
| **H4** | Instruction tuning increases the asymmetry: Δ(instruct) − Δ(base) > 0 for the same model family. | paired bootstrap over the same patients |
| **H5** | Agreement-based trust over-trusts the model on flipped evidence: over-trust > 0. | bootstrap 95% CI |
| **H6** | Adding the flip-consistency flag to the trust regression improves its AUROC for correctness in a mixed setting, where each patient gets the true or the flipped card at random (p = 0.5, seed 2027). | 5-fold cross-fitted predictions; paired DeLong test |

**Exploratory analyses** (labelled as such and not used for conclusions): C6 versus C4; robustness across example sets A, B and C; base models on H1–H3; memorization effects; per-feature patterns.

## 8. Order of work and decision rules

1. **Commit this file and tag the `prereg-v1` release** before running anything on dev or test patients.
2. **Dev phase.** Use the 100 dev patients per dataset with Qwen2.5-7B-Instruct to check the code, parsing, `yn_mass` and runtime. Dev results may change only engineering settings, such as batch size or backend. Any change to prompts, measures or hypotheses is logged as a deviation before the main run.
3. **Compute fallback.** If the projected GPU time for the full grid exceeds the budget, N per dataset is reduced from 600 to 400 for all datasets. This is decided and logged before the first test run.
4. **Main run,** then the analysis exactly as specified above.
5. **No result-dependent changes.** All hypotheses are reported whatever their outcome. If H3 is not supported, the paper reports that, and no alternative asymmetry measure is promoted to primary afterwards.

## 9. What will be shared

- The code, prompts, per-patient predictions and probabilities (no raw text from the source datasets beyond what their licences allow), `run_log.json` with versions, revisions, seeds and timings, and this file with its deviations log.

## References

- Dasula, A., Desikan, P., & Srivastava, J. (2026). LLM Doesn't Know What It Doesn't Know: Detecting Epistemic Blind Spots via Cross-Model Attribution Divergence on Clinical Tabular Data. 2nd Workshop on Epistemic Intelligence in Machine Learning (EIML@ICML 2026). arXiv:2606.19509.
- Silva, I., Moody, G., Scott, D. J., Celi, L. A., & Mark, R. G. (2012). Predicting In-Hospital Mortality of Patients in ICU: The PhysioNet/Computing in Cardiology Challenge 2012. Computing in Cardiology 2012.
- Dataset: Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012, v1.0.0. PhysioNet. Open Data Commons Attribution License v1.0.
- PhysioNet standard citation: Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

## Deviations

- 2026-10-17 (clerical): filled in the registration date, which was left blank in prereg-v1. No change to the design, hypotheses or analysis.

### 2026-10-17: changes made after the dev phase and before the main run (tag `prereg-v1.1`)

I ran the dev phase (Section 8, step 2) on 2026-10-17: 100 dev patients per dataset, Qwen2.5-7B-Instruct, the full main and robustness grid. The code, the answer parsing, `yn_mass` and the runtime checks worked. No test patient has been scored, and no test-patient output exists. While checking the dev output, I found a problem with the primary measure. All changes below are made before the main run, as Section 8, step 2 requires. The dev output is shared with the other materials.

#### D1. Primary asymmetry statistic (affects H3, Q3 and H4)

- **Problem.** Δ = A − S depends on how often the model predicts the alarming outcome without evidence (its baseline lean), even when the model treats both card directions in the same way. A is computed only on patients the model calls safe at C0 and S only on patients it calls alarming, so the two groups sit at different distances from the decision boundary whenever the lean is not 50%. The registered secondary log-odds measure has a related problem when the model moves part of the way toward the card's side instead of adding a fixed amount. There is a third issue: the true card points to the alarming outcome for only 23% to 40% of the dev patients, so most alarm-pointing cards are flipped cards. If the model follows true cards more than flipped cards, this alone makes alarm-pointing cards look weaker.
- **What I saw before this change.** On the 100 dev patients per dataset, Δ was −0.48 [−0.68, −0.29] for PhysioNet 2012, +0.10 [−0.10, +0.29] for PhysioNet 2019 and −0.19 [−0.37, 0.00] for Diabetes 130, so it pointed against H3 in two of three datasets (the pilot gave +0.26). These values led me to check how Δ behaves. The change rests on the simulations below, not on those values. I did not compute the new statistics on the dev data before committing this entry.
- **Evidence.** Two scripts, committed with this entry, with their outputs:
  - `analysis/simulate_asymmetry_measures.py` (bias of each measure). The simulation crosses four ways in which a model could respond to a card, a lean toward either outcome, a preference for true cards, and an idiosyncratic error in the no-evidence reading. Across the 32 settings with no real asymmetry, Δ ranges from −0.25 to +0.22 and the log-odds measure from −3.57 to +2.45. A regression on the no-evidence reading itself (a′ below) ranges from −0.45 to +1.15, because an error in that reading enters both sides of the regression (regression to the mean). The statistic a below stays between −0.02 and +0.07. Across the 32 settings with a real asymmetry toward alarm, Δ is at or below 0 in 7 and the log-odds measure in 12; a is between +1.20 and +1.88 in all 32.
  - `analysis/simulate_asymmetry_tests.py` (how often each bootstrap test declares an asymmetry toward alarm). It adds skewed and bimodal distributions of the model's own reading of the patient and a first stage that is not cubic, because a can be biased when that reading is skewed and the response is not linear. It counts how often each bootstrap test declares an asymmetry toward alarm (estimate above 0 and p < 0.05; the nominal rate is 2.5%). Across the 120 settings with no real asymmetry, Δ does so in up to 96% of simulated studies and the log-odds measure in up to 100%. a alone reaches 50% and a′ alone 100%, each in settings where the other stays close to its nominal rate. The H3 test of this entry, which requires both statistics (below), stays at or below 6% in 116 of the 120 settings. It reaches 12% to 21% in four settings that combine a saturating response, a right-skewed or bimodal distribution of the model's own reading, an idiosyncratic no-evidence error and a lean toward alarm. Across the 120 settings with a real asymmetry toward alarm, its power is at least 0.66 (median 1.00).
- **New primary statistics.** For each patient in a model × dataset cell (alarm framing, example set A):
  - l0, la and ls are the log-odds of P(alarm) at C0, under the alarm-pointing card and under the safe-pointing card (C2 or C4, whichever points that way).
  - d = (la − l0) + (ls − l0). d > 0 means that the alarm-pointing card moved the model up more than the safe-pointing card moved it down.
  - t = +1 if the patient's true card (C2) points to the alarming outcome, and −1 otherwise.
  - e is the expected baseline: the fitted value of an ordinary least squares regression of l0 on z, z², z³ and y across the patients of the cell, where z is the patient's XGBoost log-odds (standardized within the cell) and y is the true outcome.
  - **a** is the intercept of an ordinary least squares fit of d = a + g1·e + g3·(e/s)³ + c·t, where s is the standard deviation of e in the cell (s only rescales g3; it does not change a).
  - **a′** is the intercept of the same fit with l0 in place of e: d = a′ + h1·l0 + h3·(l0/s0)³ + c′·t, where s0 is the standard deviation of l0.
  - Both are the asymmetry in log-odds for a patient whose baseline is undecided, with true and flipped cards weighted equally (t = 0). a > 0 or a′ > 0 means that alarm-pointing evidence moves the model more.
  - A patient is used only if the C0, C2 and C4 records are all valid (D3, item 5).
- **When each statistic is unbiased.** If alarm-pointing and safe-pointing cards act as mirror images of each other, d has an expected value that is an odd function of the patient's baseline. a then has expected value 0 if the response is linear in the baseline, or if the model's own reading of the patient (its baseline minus e) is symmetrically distributed, and if the terms e and e³ describe the odd function. a′ has expected value 0 under the same symmetry if the no-evidence reading has no idiosyncratic error, which is never certain. The two statistics fail under different conditions, so H3 requires both.
- **H3** is supported in a cell only if a > 0 and a′ > 0. The test is an intersection–union test: each statistic gets the registered paired bootstrap over patients (10,000 resamples, seed 2026; every regression, including the first stage, refitted in each resample) and the registered p = 2 × min(share of resampled values ≤ 0, share ≥ 0); the cell's p-value is the larger of the two. Holm runs over the 6 primary cells on these p-values, and the supported-overall rule is unchanged. If either statistic cannot be estimated in a cell (for example, if all true cards point the same way), H3 is not supported in that cell, which enters Holm with p = 1. Resamples in which a regression cannot be fitted are dropped and counted.
- **Q3** applies the following table to a and to a′ separately, each with its bootstrap 95% percentile CI under the alarm and the reversed framing (on outcomes, as before). The answer is stated only when a and a′ give the same answer; otherwise the cell is inconclusive. The registered table covers only the case in which the asymmetry under the alarm framing is positive; this table covers every sign combination and gives the same answers in that case.

  | Statistic, alarm framing (95% CI) | Statistic, reversed framing (95% CI) | Interpretation |
  |---|---|---|
  | above 0 | above 0 | follows the outcome, toward alarm (alarm aversion) |
  | below 0 | below 0 | follows the outcome, toward safe |
  | above 0 | below 0 | follows the answer word "Yes" |
  | below 0 | above 0 | follows the answer word "No" |
  | any other combination | | inconclusive |

- **H4** becomes a(instruct) − a(base) > 0 and a′(instruct) − a′(base) > 0 for the same model family and dataset, with the same intersection–union rule: a paired bootstrap over the patients valid in both models (all regressions of both models refitted in each resample), 10,000 resamples, seed 2026, the same p rule, and Holm over the 6 comparisons (2 families × 3 datasets).
- **Reported with the same prominence.** The results table for H3 shows the registered Δ with its CI next to a and a′ for every cell. If a, a′ and Δ do not agree in sign in a cell, I describe that cell as not robust. The registered log-odds measure and the follow rates stay in the report as secondary measures. None of these is used to decide H3, Q3 or H4.
- **Labelled sensitivity analyses** (reported, not used for decisions): a with d winsorized at its 1st and 99th percentiles within the cell, because the exact log-odds of D2 make d heavy-tailed; the share of patients whose expected baseline e lies within one standard deviation of 0, which shows how far a extrapolates; and the R² of the first stage.
- **Note on e.** e depends on the patient's own reading l0 only through the in-sample first-stage fit, in which each patient has a leverage of about 5/n (more for extreme z).
- Section 8, step 5 forbids promoting another asymmetry measure to primary after the results. This change is made before any test-patient output exists.

#### D2. Reading P(Yes) and P(No) exactly

- The dev code read the variant probabilities from the top 20 log-probabilities that vLLM returns, and a variant outside the top 20 counted as 0. Section 5 asks for the probability of each variant in the vocabulary, and D1 needs exact log-odds for very confident answers. On the dev data for PhysioNet 2012, the median C0 P(alarm) was 0.000 among patients predicted safe and 0.999 among patients predicted alarming.
- From now on, the log-probability of each variant token is read directly from the full next-token distribution: with vLLM's `logprob_token_ids` option, or with the full softmax in the transformers fallback. The log-odds of P(alarm) is log(sum of the alarm-word variant probabilities) − log(sum of the other-word variant probabilities), computed from log-probabilities with no clipping. `yn_mass`, the prediction and P(alarm) are defined as in Section 5.
- Check and fallback: before scoring, the notebook compares `logprob_token_ids` with the ordinary top-20 log-probabilities on the smoke-test prompts. If the option fails, or if any log-probability differs by more than 0.05, the run reads the top 1,000 log-probabilities instead. A variant absent from them gets the log-probability of the 1,000th token (an upper bound), and the number of such records is reported.
- The maximum context length is 16,384 tokens instead of 4,096, so that the memorization prompts fit (D3, item 7). This does not change the scored prompts; the longest dev prompt had 2,238 tokens.
- vLLM is pinned to version 0.26.0, the dev-phase version, for all four runs. If a run cannot use it, the run falls back to transformers (Section 5), and the backend and reading mode of every run are reported. If the runs of one model family differ in reading mode, H4 for that family is marked as affected.

#### D3. Clarifications

These describe how the code implements points that Sections 3 to 7 leave open. They do not change the intent of the design.

1. **Eligibility.** "At least 60% of the 16 selected features measured" means at least 10 of 16, because 60% of 16 is 9.6. (The pilot code used int(0.6 × 16) = 9.)
2. **Diabetes 130.** The first encounter of each patient is taken first. Patients whose first encounter ended in death or hospice are then excluded; they are not replaced by a later encounter.
3. **Validity checks.** The median `yn_mass` is taken over all alarm-framing, set-A records of the cell (C0, C2, C4 and C6). C0 accuracy is computed on the alarm-framing, set-A records. An excluded cell enters Holm with p = 1 and counts as not supporting the hypothesis. H4 is tested for a family × dataset only if both the instruct cell and the base cell pass the checks.
4. **Solved examples.** They are drawn from all training-split patients of each outcome, with seeds 2027, 2028 and 2029 for sets A, B and C, and shown in the order alarming, safe, alarming, safe.
5. **Invalid records.** A record with zero Yes/No mass or a non-finite log-odds is invalid. It is excluded, not read as "No". A paired analysis drops a patient if any record that it needs is invalid (C0, C2 and C4 for the asymmetry statistics and H1 to H4; C2 and C4 for H5 and H6). The counts are reported.
6. **System message.** When no system message is given, the Qwen2.5 chat template inserts its default system message ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."), and the Llama-3.1 template inserts a system header with its knowledge-cutoff and date lines. I use both templates as they are; "no system message" in Section 5 means that I add none. The rendered text is logged.
7. **Memorization check (Section 3).** tabmemcheck 0.1.6, `header_test` and `row_completion_test` with their default settings, greedy decoding and random seed 2027. Each response is limited to 300 new tokens, except in the base-model row test, where tabmemcheck sets its own limit (one more than the length of the true row in characters). The files are the public files that contain the outcomes: `Outcomes-a.txt` (PhysioNet 2012), `training_setA/p000001.psv` (PhysioNet 2019) and `diabetic_data.csv` (Diabetes 130). Instruct models are tested in chat mode and base models in completion mode. The header test is run separately at each of its four default split rows (2, 4, 6 and 8), and the best match is kept; a split row whose prompt and response do not fit the context is skipped and logged. If a row-completion prompt and its response do not fit the context, the number of few-shot examples is reduced (7, 5, 3, 1), and the number used is logged. A response is never shortened to make it fit. If a file has fewer than 36 rows, the number of row-completion queries is reduced from 25 to the number of rows minus 11. An error inside tabmemcheck is recorded, and the run continues. A dataset is flagged for a model if at least one row is completed verbatim. A row can also be predictable without memorization (for example, a row of missing values), so the matched rows of every flagged dataset are reported. Each model is checked in its own session, before its test prompts are scored.

#### D4. Sample size (Section 8, step 3)

The dev projection was 1.68 GPU hours per model (6.7 hours for the four models), and each test pool has at least 300 eligible positives (302, 466 and 1,883). The test set is therefore 600 patients per dataset (300 per class) for all three datasets. The compute fallback is not used.
