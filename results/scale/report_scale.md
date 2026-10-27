# Exploratory scale check: results

Plan `scale-plan` (SCALE_CHECK.md sha256 f2c1bba028998a98); registered code main-2.0 (sha256 0ba0853b4b52b897); ler_scale scale-1.0. Exploratory: not part of either pre-registration; no correction for multiple tests.

## Runs

| model | revision | status | records | imputed | dev mass Yes/No, A/B | s per prompt (dev) | memorization flagged |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (float16, second study) | b968826d9c46 | reference | 23400 | 2173 | –, – | – | none |
| Qwen3-4B-AWQ | 74d4bd2bd4bf | ran | 23400 | 307 | 1.000, 1.000 | 0.046 | none |
| Qwen3-8B-AWQ | 4da05a8edb55 | ran | 23400 | 2751 | 0.953, 0.808 | 0.036 | none |
| Qwen3-14B-AWQ | 31c69efc2946 | cannot be run: median Yes+No mass on the dev prompts 0.169 < 0.5 | 0 | – | 0.169, 0.033 | 0.049 | none |
| Qwen3-32B-AWQ | 0499c3ac83fd | cannot be run: median Yes+No mass on the dev prompts 0.176 < 0.5 | 0 | – | 0.176, 0.127 | 1.067 | none |

Rendered example prompts compared with the reference: qwen3-4b-awq 39/39 identical; qwen3-8b-awq 39/39 identical

## Validity

| model | dataset | Yes/No: mass | Yes/No: C0 accuracy | Yes/No | A/B: mass | A/B: C0 accuracy | A/B |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (float16, second study) | physionet2012 | 0.737 | 0.678 | passes | 0.441 | 0.717 | **excluded** |
| Qwen3-8B (float16, second study) | physionet2019 | 0.547 | 0.652 | passes | 0.762 | 0.642 | passes |
| Qwen3-8B (float16, second study) | diabetes130 | 0.673 | 0.563 | passes | 0.282 | 0.547 | **excluded** |
| Qwen3-4B-AWQ | physionet2012 | 1.000 | 0.642 | passes | 1.000 | 0.670 | passes |
| Qwen3-4B-AWQ | physionet2019 | 1.000 | 0.598 | passes | 1.000 | 0.522 | **excluded** |
| Qwen3-4B-AWQ | diabetes130 | 1.000 | 0.555 | passes | 1.000 | 0.523 | **excluded** |
| Qwen3-8B-AWQ | physionet2012 | 0.950 | 0.650 | passes | 0.844 | 0.688 | passes |
| Qwen3-8B-AWQ | physionet2019 | 0.902 | 0.642 | passes | 0.957 | 0.628 | passes |
| Qwen3-8B-AWQ | diabetes130 | 0.929 | 0.565 | passes | 0.786 | 0.582 | passes |

## Yes/No question: a and a′ (alarm framing, example set A)

| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |
|---|---|---|---|---|---|
| Qwen3-8B (float16, second study) | physionet2012 | -5.64 [-6.57, -4.72] | -6.76 [-7.34, -6.19] | <0.001 | **yes** |
| Qwen3-8B (float16, second study) | physionet2019 | -1.74 [-2.26, -1.28] | -2.21 [-2.63, -1.80] | <0.001 | **yes** |
| Qwen3-8B (float16, second study) | diabetes130 | -5.73 [-7.12, -4.50] | -5.87 [-6.35, -5.40] | <0.001 | **yes** |
| Qwen3-4B-AWQ | physionet2012 | -3.68 [-4.24, -3.11] | -3.92 [-4.41, -3.43] | <0.001 | **yes** |
| Qwen3-4B-AWQ | physionet2019 | +2.88 [+2.53, +3.28] | +2.83 [+2.50, +3.15] | <0.001 | no |
| Qwen3-4B-AWQ | diabetes130 | +7.10 [+5.87, +8.81] | +7.22 [+6.79, +7.64] | <0.001 | no |
| Qwen3-8B-AWQ | physionet2012 | -5.50 [-6.85, -4.11] | -5.73 [-6.39, -5.12] | <0.001 | **yes** |
| Qwen3-8B-AWQ | physionet2019 | -3.73 [-4.57, -3.01] | -3.47 [-3.91, -3.04] | <0.001 | **yes** |
| Qwen3-8B-AWQ | diabetes130 | -6.60 [-10.86, -2.26] | -7.16 [-7.72, -6.61] | 0.01 | **yes** |

| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |
|---|---|---|
| Qwen3-8B (float16, second study) | 3/3 | **yes** |
| Qwen3-4B-AWQ | 1/3 | no |
| Qwen3-8B-AWQ | 3/3 | **yes** |

## Answer options: a and a′ on the averaged reading of both option orders (described)

| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |
|---|---|---|---|---|---|
| Qwen3-8B (float16, second study) | physionet2012 | -2.96 [-3.41, -2.53] | -2.89 [-3.32, -2.47] | <0.001 | no (excluded) |
| Qwen3-8B (float16, second study) | physionet2019 | -2.14 [-2.62, -1.70] | -2.18 [-2.59, -1.78] | <0.001 | **yes** |
| Qwen3-8B (float16, second study) | diabetes130 | -7.50 [-9.00, -5.94] | -7.89 [-8.31, -7.47] | <0.001 | no (excluded) |
| Qwen3-4B-AWQ | physionet2012 | -0.24 [-1.36, +0.80] | -0.30 [-0.87, +0.25] | 0.65 | no |
| Qwen3-4B-AWQ | physionet2019 | -5.41 [-11.40, +0.22] | +4.45 [+3.85, +5.05] | 0.062 | no (excluded) |
| Qwen3-4B-AWQ | diabetes130 | +8.54 [+4.68, +12.64] | +5.57 [+5.19, +5.96] | <0.001 | no (excluded) |
| Qwen3-8B-AWQ | physionet2012 | -0.96 [-1.66, -0.25] | -0.82 [-1.30, -0.33] | 0.011 | **yes** |
| Qwen3-8B-AWQ | physionet2019 | -3.95 [-4.41, -3.52] | -4.22 [-4.61, -3.84] | <0.001 | **yes** |
| Qwen3-8B-AWQ | diabetes130 | -3.56 [-4.06, -3.03] | -3.37 [-3.84, -2.89] | <0.001 | **yes** |

| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |
|---|---|---|
| Qwen3-8B (float16, second study) | 1/3 | no |
| Qwen3-4B-AWQ | 0/3 | no |
| Qwen3-8B-AWQ | 3/3 | **yes** |

## Paired differences over the same patients

x − y; below 0 means the asymmetry toward safe-pointing evidence is larger in x.

| comparison | dataset | format | both cells valid | a difference [95% CI] | a′ difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Qwen3-8B-AWQ − Qwen3-8B (float16, second study) | physionet2012 | Yes/No | **yes** | +0.15 [-0.67, +0.99] | +1.02 [+0.51, +1.53] | 0.7 |
| Qwen3-8B-AWQ − Qwen3-8B (float16, second study) | physionet2012 | A/B | no | +2.00 [+1.44, +2.61] | +2.07 [+1.72, +2.43] | <0.001 |
| Qwen3-8B-AWQ − Qwen3-8B (float16, second study) | physionet2019 | Yes/No | **yes** | -1.99 [-2.57, -1.46] | -1.25 [-1.54, -0.97] | <0.001 |
| Qwen3-8B-AWQ − Qwen3-8B (float16, second study) | physionet2019 | A/B | **yes** | -1.81 [-2.04, -1.57] | -2.04 [-2.28, -1.80] | <0.001 |
| Qwen3-8B-AWQ − Qwen3-8B (float16, second study) | diabetes130 | Yes/No | **yes** | -0.86 [-4.46, +3.10] | -1.29 [-1.66, -0.92] | 0.63 |
| Qwen3-8B-AWQ − Qwen3-8B (float16, second study) | diabetes130 | A/B | no | +3.94 [+2.45, +5.34] | +4.52 [+4.20, +4.85] | <0.001 |
| Qwen3-4B-AWQ − Qwen3-8B-AWQ | physionet2012 | Yes/No | **yes** | +1.82 [+0.31, +3.28] | +1.82 [+1.07, +2.59] | 0.02 |
| Qwen3-4B-AWQ − Qwen3-8B-AWQ | physionet2012 | A/B | **yes** | +0.72 [-0.65, +1.95] | +0.52 [-0.20, +1.20] | 0.29 |
| Qwen3-4B-AWQ − Qwen3-8B-AWQ | physionet2019 | Yes/No | **yes** | +6.61 [+5.83, +7.53] | +6.29 [+5.81, +6.78] | <0.001 |
| Qwen3-4B-AWQ − Qwen3-8B-AWQ | physionet2019 | A/B | no | -1.46 [-7.40, +4.18] | +8.67 [+7.99, +9.37] | 0.77 |
| Qwen3-4B-AWQ − Qwen3-8B-AWQ | diabetes130 | Yes/No | **yes** | +13.70 [+9.10, +18.23] | +14.38 [+13.75, +15.03] | <0.001 |
| Qwen3-4B-AWQ − Qwen3-8B-AWQ | diabetes130 | A/B | no | +12.09 [+8.21, +16.14] | +8.94 [+8.37, +9.53] | <0.001 |

## Q3 with the reversed Yes/No question

| model | dataset | a: alarm / reversed | a′: alarm / reversed | interpretation |
|---|---|---|---|---|
| Qwen3-8B (float16, second study) | physionet2012 | -5.64 / -0.15 | -6.76 / +0.82 | inconclusive (a and a' disagree) |
| Qwen3-8B (float16, second study) | physionet2019 | -1.74 / -2.02 | -2.21 / +1.29 | inconclusive (a and a' disagree) |
| Qwen3-8B (float16, second study) | diabetes130 | -5.73 / -33.58 | -5.87 / +3.01 | inconclusive (a and a' disagree) |
| Qwen3-4B-AWQ | physionet2012 | -3.68 / +2.49 | -3.92 / +5.22 | follows the answer word "No" |
| Qwen3-4B-AWQ | physionet2019 | +2.88 / -18.76 | +2.83 / +20.46 | inconclusive (a and a' disagree) |
| Qwen3-4B-AWQ | diabetes130 | +7.10 / +4.03 | +7.22 / +12.18 | inconclusive (a and a' disagree) |
| Qwen3-8B-AWQ | physionet2012 | -5.50 / +4.86 | -5.73 / +6.39 | follows the answer word "No" |
| Qwen3-8B-AWQ | physionet2019 | -3.73 / -3.57 | -3.47 / +6.45 | inconclusive (a and a' disagree) |
| Qwen3-8B-AWQ | diabetes130 | -6.60 / -219.20 | -7.16 / +5.04 | inconclusive (a and a' disagree) |

## Accuracy (Yes/No, alarm framing; exact McNemar, uncorrected)

| model | dataset | C0 | C2 − C0 (p) | C4 − C0 (p) |
|---|---|---|---|---|
| Qwen3-8B (float16, second study) | physionet2012 | 0.678 | +0.01 (0.41) | -0.14 (<0.001) |
| Qwen3-8B (float16, second study) | physionet2019 | 0.652 | +0.01 (0.52) | -0.07 (<0.001) |
| Qwen3-8B (float16, second study) | diabetes130 | 0.563 | -0.01 (0.69) | -0.12 (<0.001) |
| Qwen3-4B-AWQ | physionet2012 | 0.642 | +0.06 (<0.001) | -0.18 (<0.001) |
| Qwen3-4B-AWQ | physionet2019 | 0.598 | +0.05 (0.0084) | -0.10 (<0.001) |
| Qwen3-4B-AWQ | diabetes130 | 0.555 | +0.03 (0.12) | -0.05 (0.063) |
| Qwen3-8B-AWQ | physionet2012 | 0.650 | +0.00 (1) | -0.13 (<0.001) |
| Qwen3-8B-AWQ | physionet2019 | 0.642 | +0.01 (0.75) | -0.08 (<0.001) |
| Qwen3-8B-AWQ | diabetes130 | 0.565 | -0.02 (0.35) | -0.10 (0.0013) |

## Reading rules of the plan

- Quantization, Yes/No question: 4-bit weights are taken not to remove the pattern (Qwen3-8B-AWQ shows it) (Qwen3-8B-AWQ 3/3, float16 3/3).
- Size, Yes/No question: **no larger model was run**.
- Quantization, answer options (described): 4-bit weights are taken not to remove the pattern (Qwen3-8B-AWQ shows it) (Qwen3-8B-AWQ 3/3, float16 1/3).
- Size, answer options (described): no larger model was run.
- Nothing here changes R1, R2, R3 or any result of the first study.
