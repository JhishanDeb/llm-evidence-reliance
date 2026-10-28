# Exploratory scale check, second plan: results (two-step reading)

Plan `scale-plan-2` (SCALE_CHECK_2.md sha256 de98890bed5cf859); registered code main-2.0 (sha256 0ba0853b4b52b897); ler_scale scale-1.0; ler_scale2 scale-2.0. Exploratory: not part of either pre-registration; no correction for multiple tests. The 4B model's records come from the first plan (every reading of the first-plan run has answer-word mass >= 0.99).

## Runs

| run | revision | status | records | imputed | model check: two-step mass Yes/No (first token alone), A/B | s per prompt (dev) | memorization flagged |
|---|---|---|---|---|---|---|---|
| qwen3-4b-awq | 74d4bd2bd4bf | first plan | 23400 | 307 | 1.000, 1.000 | 0.046 | none |
| qwen3-8b-awq | 4da05a8edb55 | first plan | 23400 | 2751 | 0.953, 0.808 | 0.036 | none |
| qwen3-8b-awq-2step | 4da05a8edb55 | second plan | 23400 | 2758 | 1.000 (0.962), 1.000 | 0.333 | not run under this plan |
| qwen3-14b-awq-2step | 31c69efc2946 | second plan | 23400 | 8019 | 1.000 (0.166), 1.000 | 0.559 | none |
| qwen3-32b-awq-2step | 0499c3ac83fd | second plan | 23400 | 6238 | 1.000 (0.182), 1.000 | 1.239 | none |
| qwen3-instruct | b968826d9c46 | second study | 23400 | 2173 | –, – | – | none |

Rendered example prompts: qwen3-14b-awq-2step 39/39 identical to qwen3-instruct; qwen3-32b-awq-2step 39/39 identical to qwen3-instruct; qwen3-8b-awq-2step 39/39 identical to qwen3-8b-awq

## Probability of " **" after "Answer:" (median over test prompts)

| model | dataset | Yes/No | A/B | Yes/No mass, first token alone | Yes/No mass, two-step |
|---|---|---|---|---|---|
| Qwen3-14B-AWQ | physionet2012 | 0.407 | 0.950 | 0.679 | 1.000 |
| Qwen3-14B-AWQ | physionet2019 | 0.971 | 0.988 | 0.035 | 1.000 |
| Qwen3-14B-AWQ | diabetes130 | 0.874 | 0.957 | 0.066 | 1.000 |
| Qwen3-32B-AWQ | physionet2012 | 0.782 | 0.889 | 0.356 | 1.000 |
| Qwen3-32B-AWQ | physionet2019 | 0.922 | 0.899 | 0.104 | 1.000 |
| Qwen3-32B-AWQ | diabetes130 | 0.845 | 0.866 | 0.246 | 1.000 |
| Qwen3-8B-AWQ | physionet2012 | 0.030 | 0.160 | 0.947 | 1.000 |
| Qwen3-8B-AWQ | physionet2019 | 0.068 | 0.044 | 0.900 | 1.000 |
| Qwen3-8B-AWQ | diabetes130 | 0.038 | 0.215 | 0.929 | 1.000 |

## Validity

| model | dataset | Yes/No: mass | Yes/No: C0 accuracy | Yes/No | A/B: mass | A/B: C0 accuracy | A/B |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | physionet2012 | 1.000 | 0.642 | passes | 1.000 | 0.670 | passes |
| Qwen3-4B-AWQ (first reading, reused) | physionet2019 | 1.000 | 0.598 | passes | 1.000 | 0.522 | **excluded** |
| Qwen3-4B-AWQ (first reading, reused) | diabetes130 | 1.000 | 0.555 | passes | 1.000 | 0.523 | **excluded** |
| Qwen3-8B-AWQ | physionet2012 | 1.000 | 0.662 | passes | 1.000 | 0.690 | passes |
| Qwen3-8B-AWQ | physionet2019 | 1.000 | 0.652 | passes | 1.000 | 0.628 | passes |
| Qwen3-8B-AWQ | diabetes130 | 1.000 | 0.555 | passes | 1.000 | 0.580 | passes |
| Qwen3-14B-AWQ | physionet2012 | 1.000 | 0.672 | passes | 1.000 | 0.663 | passes |
| Qwen3-14B-AWQ | physionet2019 | 1.000 | 0.605 | passes | 1.000 | 0.615 | passes |
| Qwen3-14B-AWQ | diabetes130 | 1.000 | 0.580 | passes | 1.000 | 0.592 | passes |
| Qwen3-32B-AWQ | physionet2012 | 1.000 | 0.660 | passes | 1.000 | 0.682 | passes |
| Qwen3-32B-AWQ | physionet2019 | 1.000 | 0.593 | passes | 1.000 | 0.597 | passes |
| Qwen3-32B-AWQ | diabetes130 | 1.000 | 0.582 | passes | 1.000 | 0.585 | passes |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | 0.950 | 0.650 | passes | 0.844 | 0.688 | passes |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | 0.902 | 0.642 | passes | 0.957 | 0.628 | passes |
| Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | 0.929 | 0.565 | passes | 0.786 | 0.582 | passes |
| Qwen3-8B float16 (second study, first reading) | physionet2012 | 0.737 | 0.678 | passes | 0.441 | 0.717 | **excluded** |
| Qwen3-8B float16 (second study, first reading) | physionet2019 | 0.547 | 0.652 | passes | 0.762 | 0.642 | passes |
| Qwen3-8B float16 (second study, first reading) | diabetes130 | 0.673 | 0.563 | passes | 0.282 | 0.547 | **excluded** |

## Yes/No question: a and a′ (alarm framing, example set A)

| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |
|---|---|---|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | physionet2012 | -3.68 [-4.24, -3.11] | -3.92 [-4.41, -3.43] | <0.001 | **yes** |
| Qwen3-4B-AWQ (first reading, reused) | physionet2019 | +2.88 [+2.53, +3.28] | +2.83 [+2.50, +3.15] | <0.001 | no |
| Qwen3-4B-AWQ (first reading, reused) | diabetes130 | +7.10 [+5.87, +8.81] | +7.22 [+6.79, +7.64] | <0.001 | no |
| Qwen3-8B-AWQ | physionet2012 | -5.27 [-6.43, -4.07] | -5.30 [-5.87, -4.76] | <0.001 | **yes** |
| Qwen3-8B-AWQ | physionet2019 | -3.24 [-3.94, -2.65] | -3.00 [-3.40, -2.61] | <0.001 | **yes** |
| Qwen3-8B-AWQ | diabetes130 | -7.12 [-10.00, -4.17] | -7.27 [-7.80, -6.75] | <0.001 | **yes** |
| Qwen3-14B-AWQ | physionet2012 | -7.22 [-8.67, -5.80] | -6.49 [-7.46, -5.52] | <0.001 | **yes** |
| Qwen3-14B-AWQ | physionet2019 | -3.46 [-4.45, -2.71] | -3.57 [-4.18, -2.98] | <0.001 | **yes** |
| Qwen3-14B-AWQ | diabetes130 | -3.11 [-4.87, -1.17] | -4.71 [-5.31, -4.11] | 0.004 | **yes** |
| Qwen3-32B-AWQ | physionet2012 | -4.82 [-5.41, -4.22] | -5.01 [-5.27, -4.74] | <0.001 | **yes** |
| Qwen3-32B-AWQ | physionet2019 | -4.29 [-4.54, -4.04] | -4.41 [-4.61, -4.21] | <0.001 | **yes** |
| Qwen3-32B-AWQ | diabetes130 | -2.28 [-2.96, -1.50] | -2.42 [-2.72, -2.12] | <0.001 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | -5.50 [-6.85, -4.11] | -5.73 [-6.39, -5.12] | <0.001 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | -3.73 [-4.57, -3.01] | -3.47 [-3.91, -3.04] | <0.001 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | -6.60 [-10.86, -2.26] | -7.16 [-7.72, -6.61] | 0.01 | **yes** |
| Qwen3-8B float16 (second study, first reading) | physionet2012 | -5.64 [-6.57, -4.72] | -6.76 [-7.34, -6.19] | <0.001 | **yes** |
| Qwen3-8B float16 (second study, first reading) | physionet2019 | -1.74 [-2.26, -1.28] | -2.21 [-2.63, -1.80] | <0.001 | **yes** |
| Qwen3-8B float16 (second study, first reading) | diabetes130 | -5.73 [-7.12, -4.50] | -5.87 [-6.35, -5.40] | <0.001 | **yes** |

| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |
|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | 1/3 | no |
| Qwen3-8B-AWQ | 3/3 | **yes** |
| Qwen3-14B-AWQ | 3/3 | **yes** |
| Qwen3-32B-AWQ | 3/3 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | 3/3 | **yes** |
| Qwen3-8B float16 (second study, first reading) | 3/3 | **yes** |

## Answer options: a and a′ on the averaged reading of both option orders (described)

| model | dataset | a [95% CI] | a′ [95% CI] | intersection–union p | pattern |
|---|---|---|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | physionet2012 | -0.24 [-1.36, +0.80] | -0.30 [-0.87, +0.25] | 0.65 | no |
| Qwen3-4B-AWQ (first reading, reused) | physionet2019 | -5.41 [-11.40, +0.22] | +4.45 [+3.85, +5.05] | 0.062 | no (excluded) |
| Qwen3-4B-AWQ (first reading, reused) | diabetes130 | +8.54 [+4.68, +12.64] | +5.57 [+5.19, +5.96] | <0.001 | no (excluded) |
| Qwen3-8B-AWQ | physionet2012 | -0.70 [-1.32, -0.07] | -0.54 [-0.99, -0.08] | 0.035 | **yes** |
| Qwen3-8B-AWQ | physionet2019 | -3.57 [-3.99, -3.16] | -3.83 [-4.21, -3.47] | <0.001 | **yes** |
| Qwen3-8B-AWQ | diabetes130 | -2.88 [-3.33, -2.42] | -2.68 [-3.11, -2.25] | <0.001 | **yes** |
| Qwen3-14B-AWQ | physionet2012 | -4.61 [-5.69, -3.56] | -4.10 [-4.69, -3.51] | <0.001 | **yes** |
| Qwen3-14B-AWQ | physionet2019 | +0.45 [-0.09, +1.00] | +0.25 [-0.15, +0.64] | 0.21 | no |
| Qwen3-14B-AWQ | diabetes130 | +1.27 [+0.47, +2.20] | +3.09 [+2.72, +3.47] | 0.0038 | no |
| Qwen3-32B-AWQ | physionet2012 | -1.79 [-2.05, -1.53] | -1.98 [-2.21, -1.73] | <0.001 | **yes** |
| Qwen3-32B-AWQ | physionet2019 | +0.35 [-0.64, +1.30] | +0.39 [+0.12, +0.66] | 0.49 | no |
| Qwen3-32B-AWQ | diabetes130 | -5.10 [-6.20, -3.89] | -2.95 [-3.20, -2.69] | <0.001 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | -0.96 [-1.66, -0.25] | -0.82 [-1.30, -0.33] | 0.011 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | -3.95 [-4.41, -3.52] | -4.22 [-4.61, -3.84] | <0.001 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | -3.56 [-4.06, -3.03] | -3.37 [-3.84, -2.89] | <0.001 | **yes** |
| Qwen3-8B float16 (second study, first reading) | physionet2012 | -2.96 [-3.41, -2.53] | -2.89 [-3.32, -2.47] | <0.001 | no (excluded) |
| Qwen3-8B float16 (second study, first reading) | physionet2019 | -2.14 [-2.62, -1.70] | -2.18 [-2.59, -1.78] | <0.001 | **yes** |
| Qwen3-8B float16 (second study, first reading) | diabetes130 | -7.50 [-9.00, -5.94] | -7.89 [-8.31, -7.47] | <0.001 | no (excluded) |

| model | datasets with the pattern | model shows the pattern (≥ 2 of 3) |
|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | 0/3 | no |
| Qwen3-8B-AWQ | 3/3 | **yes** |
| Qwen3-14B-AWQ | 1/3 | no |
| Qwen3-32B-AWQ | 2/3 | **yes** |
| Qwen3-8B-AWQ (first reading: bridge) | 3/3 | **yes** |
| Qwen3-8B float16 (second study, first reading) | 1/3 | no |

## Paired differences over the same patients

x − y; below 0 means the asymmetry toward safe-pointing evidence is larger in x.

| comparison | dataset | format | both cells valid | a difference [95% CI] | a′ difference [95% CI] | p |
|---|---|---|---|---|---|---|
| bridge: Qwen3-8B-AWQ − Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | Yes/No | **yes** | +0.23 [-0.00, +0.47] | +0.43 [+0.32, +0.55] | 0.051 |
| bridge: Qwen3-8B-AWQ − Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | A/B | **yes** | +0.26 [+0.15, +0.36] | +0.28 [+0.23, +0.33] | <0.001 |
| bridge: Qwen3-8B-AWQ − Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | Yes/No | **yes** | +0.49 [+0.30, +0.68] | +0.46 [+0.38, +0.54] | <0.001 |
| bridge: Qwen3-8B-AWQ − Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | A/B | **yes** | +0.39 [+0.33, +0.45] | +0.39 [+0.35, +0.43] | <0.001 |
| bridge: Qwen3-8B-AWQ − Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | Yes/No | **yes** | -0.52 [-2.04, +0.96] | -0.11 [-0.18, -0.04] | 0.41 |
| bridge: Qwen3-8B-AWQ − Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | A/B | **yes** | +0.67 [+0.44, +0.87] | +0.69 [+0.61, +0.76] | <0.001 |
| size: Qwen3-4B-AWQ (first reading, reused) − Qwen3-8B-AWQ | physionet2012 | Yes/No | **yes** | +1.59 [+0.26, +2.88] | +1.38 [+0.69, +2.10] | 0.022 |
| size: Qwen3-4B-AWQ (first reading, reused) − Qwen3-8B-AWQ | physionet2012 | A/B | **yes** | +0.46 [-0.84, +1.65] | +0.24 [-0.45, +0.91] | 0.51 |
| size: Qwen3-4B-AWQ (first reading, reused) − Qwen3-8B-AWQ | physionet2019 | Yes/No | **yes** | +6.12 [+5.45, +6.89] | +5.83 [+5.38, +6.29] | <0.001 |
| size: Qwen3-4B-AWQ (first reading, reused) − Qwen3-8B-AWQ | physionet2019 | A/B | no | -1.84 [-7.80, +3.79] | +8.28 [+7.61, +8.97] | 0.66 |
| size: Qwen3-4B-AWQ (first reading, reused) − Qwen3-8B-AWQ | diabetes130 | Yes/No | **yes** | +14.22 [+10.88, +17.47] | +14.49 [+13.88, +15.12] | <0.001 |
| size: Qwen3-4B-AWQ (first reading, reused) − Qwen3-8B-AWQ | diabetes130 | A/B | no | +11.42 [+7.59, +15.45] | +8.25 [+7.73, +8.80] | <0.001 |
| size: Qwen3-14B-AWQ − Qwen3-8B-AWQ | physionet2012 | Yes/No | **yes** | -1.95 [-3.41, -0.58] | -1.19 [-2.15, -0.25] | 0.015 |
| size: Qwen3-14B-AWQ − Qwen3-8B-AWQ | physionet2012 | A/B | **yes** | -3.91 [-4.94, -2.91] | -3.56 [-4.24, -2.90] | <0.001 |
| size: Qwen3-14B-AWQ − Qwen3-8B-AWQ | physionet2019 | Yes/No | **yes** | -0.21 [-1.04, +0.46] | -0.57 [-1.11, -0.04] | 0.46 |
| size: Qwen3-14B-AWQ − Qwen3-8B-AWQ | physionet2019 | A/B | **yes** | +4.02 [+3.59, +4.48] | +4.09 [+3.68, +4.50] | <0.001 |
| size: Qwen3-14B-AWQ − Qwen3-8B-AWQ | diabetes130 | Yes/No | **yes** | +4.01 [+0.93, +7.36] | +2.56 [+1.79, +3.35] | 0.013 |
| size: Qwen3-14B-AWQ − Qwen3-8B-AWQ | diabetes130 | A/B | **yes** | +4.16 [+3.32, +5.12] | +5.77 [+5.25, +6.28] | <0.001 |
| size: Qwen3-32B-AWQ − Qwen3-8B-AWQ | physionet2012 | Yes/No | **yes** | +0.44 [-0.72, +1.58] | +0.29 [-0.24, +0.85] | 0.46 |
| size: Qwen3-32B-AWQ − Qwen3-8B-AWQ | physionet2012 | A/B | **yes** | -1.09 [-1.73, -0.48] | -1.44 [-1.88, -0.99] | <0.001 |
| size: Qwen3-32B-AWQ − Qwen3-8B-AWQ | physionet2019 | Yes/No | **yes** | -1.05 [-1.66, -0.37] | -1.41 [-1.77, -1.04] | 0.0046 |
| size: Qwen3-32B-AWQ − Qwen3-8B-AWQ | physionet2019 | A/B | **yes** | +3.92 [+2.83, +4.92] | +4.22 [+3.81, +4.65] | <0.001 |
| size: Qwen3-32B-AWQ − Qwen3-8B-AWQ | diabetes130 | Yes/No | **yes** | +4.83 [+1.76, +7.81] | +4.85 [+4.24, +5.45] | 0.0062 |
| size: Qwen3-32B-AWQ − Qwen3-8B-AWQ | diabetes130 | A/B | **yes** | -2.22 [-3.29, -1.03] | -0.27 [-0.74, +0.22] | 0.28 |

## Q3 with the reversed Yes/No question

| model | dataset | a: alarm / reversed | a′: alarm / reversed | interpretation |
|---|---|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | physionet2012 | -3.68 / +2.49 | -3.92 / +5.22 | follows the answer word "No" |
| Qwen3-4B-AWQ (first reading, reused) | physionet2019 | +2.88 / -18.76 | +2.83 / +20.46 | inconclusive (a and a' disagree) |
| Qwen3-4B-AWQ (first reading, reused) | diabetes130 | +7.10 / +4.03 | +7.22 / +12.18 | inconclusive (a and a' disagree) |
| Qwen3-8B-AWQ | physionet2012 | -5.27 / +4.03 | -5.30 / +5.35 | follows the answer word "No" |
| Qwen3-8B-AWQ | physionet2019 | -3.24 / -2.72 | -3.00 / +5.81 | inconclusive (a and a' disagree) |
| Qwen3-8B-AWQ | diabetes130 | -7.12 / -158.91 | -7.27 / +3.86 | inconclusive (a and a' disagree) |
| Qwen3-14B-AWQ | physionet2012 | -7.22 / -0.28 | -6.49 / -0.77 | inconclusive (a and a' disagree) |
| Qwen3-14B-AWQ | physionet2019 | -3.46 / +7.06 | -3.57 / +5.58 | follows the answer word "No" |
| Qwen3-14B-AWQ | diabetes130 | -3.11 / +3.96 | -4.71 / +11.49 | inconclusive (a and a' disagree) |
| Qwen3-32B-AWQ | physionet2012 | -4.82 / -0.86 | -5.01 / -0.88 | follows the outcome, toward safe |
| Qwen3-32B-AWQ | physionet2019 | -4.29 / +1.14 | -4.41 / +0.66 | inconclusive (a and a' disagree) |
| Qwen3-32B-AWQ | diabetes130 | -2.28 / +1.73 | -2.42 / +2.63 | follows the answer word "No" |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | -5.50 / +4.86 | -5.73 / +6.39 | follows the answer word "No" |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | -3.73 / -3.57 | -3.47 / +6.45 | inconclusive (a and a' disagree) |
| Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | -6.60 / -219.20 | -7.16 / +5.04 | inconclusive (a and a' disagree) |
| Qwen3-8B float16 (second study, first reading) | physionet2012 | -5.64 / -0.15 | -6.76 / +0.82 | inconclusive (a and a' disagree) |
| Qwen3-8B float16 (second study, first reading) | physionet2019 | -1.74 / -2.02 | -2.21 / +1.29 | inconclusive (a and a' disagree) |
| Qwen3-8B float16 (second study, first reading) | diabetes130 | -5.73 / -33.58 | -5.87 / +3.01 | inconclusive (a and a' disagree) |

## Accuracy (Yes/No, alarm framing; exact McNemar, uncorrected)

| model | dataset | C0 | C2 − C0 (p) | C4 − C0 (p) |
|---|---|---|---|---|
| Qwen3-4B-AWQ (first reading, reused) | physionet2012 | 0.642 | +0.06 (<0.001) | -0.18 (<0.001) |
| Qwen3-4B-AWQ (first reading, reused) | physionet2019 | 0.598 | +0.05 (0.0084) | -0.10 (<0.001) |
| Qwen3-4B-AWQ (first reading, reused) | diabetes130 | 0.555 | +0.03 (0.12) | -0.05 (0.063) |
| Qwen3-8B-AWQ | physionet2012 | 0.662 | +0.00 (1) | -0.14 (<0.001) |
| Qwen3-8B-AWQ | physionet2019 | 0.652 | +0.01 (0.59) | -0.09 (<0.001) |
| Qwen3-8B-AWQ | diabetes130 | 0.555 | -0.01 (0.58) | -0.09 (0.0037) |
| Qwen3-14B-AWQ | physionet2012 | 0.672 | +0.02 (0.16) | -0.10 (<0.001) |
| Qwen3-14B-AWQ | physionet2019 | 0.605 | +0.04 (0.02) | -0.09 (<0.001) |
| Qwen3-14B-AWQ | diabetes130 | 0.580 | -0.02 (0.37) | -0.13 (<0.001) |
| Qwen3-32B-AWQ | physionet2012 | 0.660 | -0.00 (0.89) | -0.16 (<0.001) |
| Qwen3-32B-AWQ | physionet2019 | 0.593 | +0.03 (0.15) | -0.15 (<0.001) |
| Qwen3-32B-AWQ | diabetes130 | 0.582 | -0.01 (0.57) | -0.14 (<0.001) |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2012 | 0.650 | +0.00 (1) | -0.13 (<0.001) |
| Qwen3-8B-AWQ (first reading: bridge) | physionet2019 | 0.642 | +0.01 (0.75) | -0.08 (<0.001) |
| Qwen3-8B-AWQ (first reading: bridge) | diabetes130 | 0.565 | -0.02 (0.35) | -0.10 (0.0013) |
| Qwen3-8B float16 (second study, first reading) | physionet2012 | 0.678 | +0.01 (0.41) | -0.14 (<0.001) |
| Qwen3-8B float16 (second study, first reading) | physionet2019 | 0.652 | +0.01 (0.52) | -0.07 (<0.001) |
| Qwen3-8B float16 (second study, first reading) | diabetes130 | 0.563 | -0.01 (0.69) | -0.12 (<0.001) |

## Reading rules of the second plan

- Bridge, Yes/No question: the two-step reading is taken not to remove the pattern (Qwen3-8B-AWQ shows it) (Qwen3-8B-AWQ 3/3, Qwen3-8B-AWQ (first reading: bridge) 3/3).
- Size, Yes/No question: **persists with size (Qwen3-4B-AWQ (first reading, reused) 1/3, Qwen3-8B-AWQ 3/3, Qwen3-14B-AWQ 3/3, Qwen3-32B-AWQ 3/3)**.
- Bridge, answer options (described): the two-step reading is taken not to remove the pattern (Qwen3-8B-AWQ shows it) (Qwen3-8B-AWQ 3/3, Qwen3-8B-AWQ (first reading: bridge) 3/3).
- Size, answer options (described): weakens with size: Qwen3-14B-AWQ does not show it (Qwen3-4B-AWQ (first reading, reused) 0/3, Qwen3-8B-AWQ 3/3, Qwen3-14B-AWQ 1/3, Qwen3-32B-AWQ 2/3).
- Nothing here changes R1, R2, R3, any result of the first study, or the results of the first scale plan.
