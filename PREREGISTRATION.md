# Pre-registration: Do LLMs reason with model explanations, or defer to them one way?

- **Author:** Jhishan Deb
- **Registered:** 2026-10-__ (the GitHub commit and the `prereg-v1` release date are the authoritative timestamps)
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

_None yet._
