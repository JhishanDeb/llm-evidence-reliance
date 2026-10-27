# Exploratory scale check, second plan: reading answers written in bold

Written on 27 October 2026, after the runs of the first scale plan (`SCALE_CHECK.md`, tag `scale-plan`) and before
any run under this plan. This file is committed and tagged `scale-plan-2` before the first run. The run notebook
downloads it at that tag and stops if a model id or revision below is missing.

## Status

This is an exploratory analysis, like the first plan. It is not part of either pre-registration. It changes no
registered result and no result of the first scale plan; those results are reported in full as they are.

## What happened under the first plan

- Qwen3-8B-AWQ and Qwen3-4B-AWQ were run. Qwen3-14B-AWQ and Qwen3-32B-AWQ failed the model check: their median
  Yes+No mass on the dev prompts was 0.169 and 0.176 (the rule asks for at least 0.5). They were never scored.
- A diagnostic on the same dev prompts (engineering only: no accuracy and no asymmetry;
  `results/scale/scale_diagnostic.zip`) showed the reason. After "Answer:", these two models put most of their
  probability on the token " **" (the start of markdown bold) and then write the answer in bold ("**Yes**", "**A**").
  Mean probability of " **" after "Answer:": Yes/No question 0.65 (14B) and 0.76 (32B), against 0.14 for the 8B
  model; answer options 0.90 (14B), 0.85 (32B) and 0.25 (8B). Every other token outside the answer words had a mean
  probability below 0.001.
- When this plan was written, the asymmetry of the 8B and 4B models under the first plan had been computed (the 8B
  model showed the pattern in 3 of 3 datasets, the 4B model in 1 of 3). Nothing has been computed for the 14B or 32B
  models.

## The two-step reading

Each prompt is read twice, both times with the top 1,000 log-probabilities, as in both studies (Deviation D5):

1. after "Answer:", the probabilities of the answer-word tokens and of the token " **";
2. after "Answer: **" (the same prompt with the token " **" added), the probabilities of the answer-word tokens.

The probability of an answer word is P1(word) + P1(" **") × P2(word), each term summed over the single-token
variants of the word ("Yes" and " Yes", "A" and " A", and so on). The answer-word mass, the log-odds and every measure
are then computed from these probabilities exactly as before. Other wrappers (" *", " _") are not added, because
their mean probability in the diagnostic was below 0.001. A variant outside the top 1,000 in either step gets the last
returned log-probability (an upper bound) and is counted as imputed, as in Deviation D5.

The prompts are the same 13 per patient as in the first plan, built by the registered code of the second study
(`ler_main.py`, main-2.0, SHA-256 0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e), which is not
changed.

## Models and order

| Model | Revision | Role |
|---|---|---|
| Qwen/Qwen3-8B-AWQ | 4da05a8edb55c6046cce958586c33b61da07bb79 | bridge: the same model under both readings |
| Qwen/Qwen3-14B-AWQ | 31c69efc29464b6bb0aee1398b5a7b50a99340c3 | larger |
| Qwen/Qwen3-32B-AWQ | 0499c3ac83fdef8810b907a23894ba91e95eddd8 | largest |

The runs follow this order: 8B, 14B, 32B. The model check is the one of the first plan with the two-step reading:
the model loads, the four sanity questions get the right word, and the median two-step Yes+No mass on the dev prompts
is at least 0.5. A model that fails is reported as "cannot be run", with the reason. The memorization check runs for
the 14B and 32B models; the 8B model was checked under the first plan (nothing flagged).

The 4B model is not run again. In its first-plan run, every one of its 23,400 readings had an answer-word mass of at
least 0.99, so the bold path can carry at most 0.01 of the probability of any prompt. Its first-plan records are used
and the report says so.

## Measures

The measures of the first plan (validity, a and a′ with the intersection–union p, Q3, the averaged answer-option
reading, accuracy), all with the two-step reading, plus:

- the median probability of " **" after "Answer:", per model, format and dataset;
- paired differences over the same patients, for a and a′, with both question formats: Qwen3-8B-AWQ under the
  two-step reading minus Qwen3-8B-AWQ under the first reading (bridge), and each other model minus Qwen3-8B-AWQ, both
  under the two-step reading (size).

## Reading rules

These rules are fixed now. There is no correction for multiple tests.

- A cell shows the pattern if it is valid, a < 0 and a′ < 0, and the intersection–union p < 0.05. A model shows the
  pattern if at least 2 of the 3 datasets do.
- Bridge: if Qwen3-8B-AWQ shows the pattern under the two-step reading (it did under the first reading, in 3 of 3
  datasets), the two-step reading is taken not to remove it, and the size comparison is read with the two-step
  reading. If it does not, the size comparison is reported but not interpreted.
- Size: the pattern "persists with size" if Qwen3-14B-AWQ and Qwen3-32B-AWQ (if it runs) show it. It "weakens with
  size" if a larger model does not show it while Qwen3-8B-AWQ does. Any other result is described as it is. The 4B
  model is described.
- The answer-option format is read with the same rules. It is described, not tested.
- Every cell is reported, including excluded ones (in an appendix).
- Nothing in this check changes R1, R2, R3, any result of the first study, or the results of the first scale plan.
