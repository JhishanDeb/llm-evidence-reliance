# Registered analysis: results

Protocol `prereg-v1.1`; analysis code main-1.0 (sha256 10557671c645bb2e).

## Runs

| model | revision | backend | read | records | invalid | imputed |
|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a1 | vllm | topk | 25200 | 0 | 0 |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c70 | vllm | topk | 25200 | 0 | 2539 |
| meta-llama/Llama-3.1-8B | d04e592bb4f6 | vllm | topk | 25200 | 0 | 0 |
| Qwen/Qwen2.5-7B | d14972939875 | vllm | topk | 25200 | 0 | 0 |

## Validity (Section 5, D3.3)

| model | dataset | median Yes+No mass | C0 accuracy | status |
|---|---|---|---|---|
| llama-instruct | physionet2012 | 0.985 | 0.607 | passes |
| llama-instruct | physionet2019 | 0.994 | 0.618 | passes |
| llama-instruct | diabetes130 | 0.990 | 0.558 | passes |
| qwen-instruct | physionet2012 | 0.998 | 0.692 | passes |
| qwen-instruct | physionet2019 | 0.995 | 0.608 | passes |
| qwen-instruct | diabetes130 | 0.991 | 0.528 | **excluded** |
| llama-base | physionet2012 | 0.779 | 0.673 | passes |
| llama-base | physionet2019 | 0.680 | 0.500 | **excluded** |
| llama-base | diabetes130 | 0.771 | 0.545 | **excluded** |
| qwen-base | physionet2012 | 0.879 | 0.670 | passes |
| qwen-base | physionet2019 | 0.886 | 0.570 | passes |
| qwen-base | diabetes130 | 0.856 | 0.577 | passes |

## Memorization check (Section 3, D3.7)

| model | dataset | file | header: chars matched | rows completed verbatim | flagged |
|---|---|---|---|---|---|
| llama-instruct | physionet2012 | Outcomes-a.txt | 7 | 0 | no |
| llama-instruct | physionet2019 | p000001.psv | 51 | 0 | no |
| llama-instruct | diabetes130 | diabetic_data.csv | 1 | 0 | no |
| qwen-instruct | physionet2012 | Outcomes-a.txt | 4 | 0 | no |
| qwen-instruct | physionet2019 | p000001.psv | 51 | 0 | no |
| qwen-instruct | diabetes130 | diabetic_data.csv | 19 | 0 | no |
| llama-base | physionet2012 | Outcomes-a.txt | 1 | 0 | no |
| llama-base | physionet2019 | p000001.psv | 86 | 0 | no |
| llama-base | diabetes130 | diabetic_data.csv | 19 | 0 | no |
| qwen-base | physionet2012 | Outcomes-a.txt | 7 | 0 | no |
| qwen-base | physionet2019 | p000001.psv | 52 | 0 | no |
| qwen-base | diabetes130 | diabetic_data.csv | 19 | 0 | no |

## Primary hypotheses (instruct models; Holm over the 6 cells)

### H1. True evidence helps: accuracy C2 > C0 (exact McNemar)

| model | dataset | accuracy difference | p | Holm p | supported |
|---|---|---|---|---|---|
| qwen-instruct | physionet2012 | +0.02 | 0.28 | 0.84 | no |
| qwen-instruct | physionet2019 | +0.03 | 0.072 | 0.29 | no |
| qwen-instruct | diabetes130 | +0.02 | 0.26 | 1 | no (not included) |
| llama-instruct | physionet2012 | +0.04 | 0.0011 | 0.0064 | **yes** |
| llama-instruct | physionet2019 | +0.04 | 0.055 | 0.28 | no |
| llama-instruct | diabetes130 | -0.03 | 0.31 | 0.84 | no |

**H1 supported overall: NO** (qwen-instruct: 0/3 datasets; llama-instruct: 1/3 datasets; rule: at least 2 of 3 datasets for each instruct model)

### H2. Flipped evidence hurts: accuracy C4 < C0 (exact McNemar)

| model | dataset | accuracy difference | p | Holm p | supported |
|---|---|---|---|---|---|
| qwen-instruct | physionet2012 | -0.09 | <0.001 | <0.001 | **yes** |
| qwen-instruct | physionet2019 | -0.12 | <0.001 | <0.001 | **yes** |
| qwen-instruct | diabetes130 | -0.07 | 0.018 | 1 | no (not included) |
| llama-instruct | physionet2012 | -0.05 | <0.001 | 0.0011 | **yes** |
| llama-instruct | physionet2019 | -0.14 | <0.001 | <0.001 | **yes** |
| llama-instruct | diabetes130 | -0.07 | 0.0098 | 0.02 | **yes** |

**H2 supported overall: YES** (qwen-instruct: 2/3 datasets; llama-instruct: 3/3 datasets; rule: at least 2 of 3 datasets for each instruct model)

### H3. Asymmetry toward alarm: a > 0 and a' > 0 (D1; intersection-union test, p = max(p_a, p_a'))

| model | dataset | a [95% CI] (p) | a' [95% CI] (p) | test p | Holm p | supported | registered Δ = A − S [95% CI] |
|---|---|---|---|---|---|---|---|
| qwen-instruct | physionet2012 | -1.71 [-2.56, -0.89] (<0.001) | -2.58 [-3.15, -2.01] (<0.001) | <0.001 | <0.001 | no | -0.34 [-0.42, -0.26] |
| qwen-instruct | physionet2019 | -2.27 [-2.87, -1.75] (<0.001) | -2.70 [-3.08, -2.32] (<0.001) | <0.001 | <0.001 | no | +0.01 [-0.07, +0.09] |
| qwen-instruct | diabetes130 | -2.23 [-4.84, +0.24] (0.068) | -4.01 [-4.38, -3.66] (<0.001) | 0.068 | 1 | no (not included) | -0.15 [-0.23, -0.07] |
| llama-instruct | physionet2012 | -0.59 [-0.88, -0.27] (<0.001) | -0.61 [-0.76, -0.47] (<0.001) | <0.001 | <0.001 | no | -0.43 [-0.53, -0.33] |
| llama-instruct | physionet2019 | -0.55 [-0.68, -0.46] (<0.001) | -0.63 [-0.70, -0.55] (<0.001) | <0.001 | <0.001 | no | -0.11 [-0.19, -0.03] |
| llama-instruct | diabetes130 | -2.09 [-2.28, -1.88] (<0.001) | -1.93 [-2.03, -1.82] (<0.001) | <0.001 | <0.001 | no | -0.46 [-0.53, -0.39] |

**H3 supported overall: NO** (qwen-instruct: 0/3 datasets; llama-instruct: 0/3 datasets; rule: at least 2 of 3 datasets for each instruct model)

Where a, a' and the registered Δ do not agree in sign, the cell is described as not robust in the paper (Deviation D1).

## Q3. Does the asymmetry follow the outcome or the answer word? (D1 four-way table)

| model | dataset | a: alarm / reversed framing [95% CI] | a': alarm / reversed framing [95% CI] | interpretation |
|---|---|---|---|---|
| qwen-instruct | physionet2012 | -1.71 [-2.56, -0.89] / -5.17 [-5.76, -4.59] | -2.58 [-3.15, -2.01] / -6.14 [-6.78, -5.47] | follows the outcome, toward safe |
| qwen-instruct | physionet2019 | -2.27 [-2.87, -1.75] / +3.48 [+1.30, +5.91] | -2.70 [-3.08, -2.32] / +3.13 [+2.74, +3.51] | follows the answer word "No" |
| qwen-instruct | diabetes130 | -2.23 [-4.84, +0.24] / -7.72 [-13.33, -2.61] | -4.01 [-4.38, -3.66] / -1.11 [-1.60, -0.63] | inconclusive (a and a' disagree) (cell excluded) |
| llama-instruct | physionet2012 | -0.59 [-0.88, -0.27] / +0.05 [-0.05, +0.16] | -0.61 [-0.76, -0.47] / +0.14 [+0.04, +0.23] | inconclusive (a and a' disagree) |
| llama-instruct | physionet2019 | -0.55 [-0.68, -0.46] / -1.00 [-2.01, +0.02] | -0.63 [-0.70, -0.55] / -0.50 [-0.65, -0.35] | inconclusive (a and a' disagree) |
| llama-instruct | diabetes130 | -2.09 [-2.28, -1.88] / +2.80 [+0.61, +5.81] | -1.93 [-2.03, -1.82] / -0.12 [-0.50, +0.25] | inconclusive (a and a' disagree) |

## Secondary hypotheses (Holm within each over 6)

### H4. Instruction tuning increases the asymmetry: a(instruct) − a(base) > 0 and the same for a' (intersection-union test)

| family / dataset | a difference [95% CI] | a' difference [95% CI] | test p | Holm p | supported |
|---|---|---|---|---|---|
| qwen / physionet2012 | -1.57 [-2.41, -0.77] | -2.41 [-2.98, -1.85] | <0.001 | 0.001 | no |
| qwen / physionet2019 | -2.35 [-2.94, -1.85] | -2.76 [-3.14, -2.40] | <0.001 | <0.001 | no |
| qwen / diabetes130 | – – | – – | – | 1 | no (not tested) |
| llama / physionet2012 | -0.35 [-0.63, -0.04] | -0.32 [-0.46, -0.19] | 0.026 | 0.1 | no |
| llama / physionet2019 | – – | – – | – | 1 | no (not tested) |
| llama / diabetes130 | – – | – – | – | 1 | no (not tested) |

### H5. Agreement-based trust over-trusts on flipped evidence: over-trust > 0

| cell | estimate | 95% CI | p | Holm p | supported |
|---|---|---|---|---|---|
| qwen-instruct / physionet2012 | -0.12 | [-0.21, -0.04] | 0.0022 | 0.0066 | no |
| qwen-instruct / physionet2019 | +0.12 | [+0.06, +0.17] | <0.001 | <0.001 | **yes** |
| qwen-instruct / diabetes130 | +0.07 | [+0.00, +0.14] | 0.035 | 1 | no (not tested) |
| llama-instruct / physionet2012 | -0.15 | [-0.20, -0.09] | <0.001 | <0.001 | no |
| llama-instruct / physionet2019 | +0.15 | [+0.09, +0.21] | <0.001 | <0.001 | **yes** |
| llama-instruct / diabetes130 | -0.03 | [-0.11, +0.04] | 0.36 | 0.72 | no |

### H6. The flip-consistency flag improves the trust AUROC (mixed setting; DeLong)

| cell | estimate | 95% CI | p | Holm p | supported |
|---|---|---|---|---|---|
| qwen-instruct / physionet2012 | -0.00 | – | 0.79 | 1 | no |
| qwen-instruct / physionet2019 | +0.01 | – | 0.65 | 1 | no |
| qwen-instruct / diabetes130 | -0.01 | – | 0.47 | 1 | no (not tested) |
| llama-instruct / physionet2012 | -0.00 | – | 0.64 | 1 | no |
| llama-instruct / physionet2019 | +0.02 | – | 0.24 | 1 | no |
| llama-instruct / diabetes130 | -0.00 | – | 0.55 | 1 | no |

## Sensitivity analyses for the asymmetry (labelled; D1)

| model | dataset | a, d winsorized at 1%/99% [95% CI] | share of patients with expected baseline within 1 SD of 0 | first-stage R² |
|---|---|---|---|---|
| llama-instruct | physionet2012 | -0.57 [-0.84, -0.27] | 0.343 | 0.639 |
| llama-instruct | physionet2019 | -0.55 [-0.67, -0.46] | 0.547 | 0.133 |
| llama-instruct | diabetes130 | -2.10 [-2.29, -1.89] | 0.650 | 0.079 |
| qwen-instruct | physionet2012 | -1.63 [-2.50, -0.83] | 0.438 | 0.621 |
| qwen-instruct | physionet2019 | -2.28 [-2.87, -1.75] | 0.588 | 0.146 |
| qwen-instruct | diabetes130 | -2.25 [-4.85, +0.22] | 0.048 | 0.067 |
| llama-base | physionet2012 | -0.24 [-0.27, -0.21] | 0.570 | 0.659 |
| llama-base | physionet2019 | -0.61 [-0.92, -0.28] | 0.000 | 0.126 |
| llama-base | diabetes130 | -0.11 [-0.14, -0.07] | 0.315 | 0.077 |
| qwen-base | physionet2012 | -0.13 [-0.20, -0.07] | 0.437 | 0.620 |
| qwen-base | physionet2019 | +0.07 [-0.01, +0.18] | 0.578 | 0.131 |
| qwen-base | diabetes130 | -0.57 [-0.65, -0.47] | 0.293 | 0.090 |

## Secondary asymmetry measures (Section 6; not used for H3, Q3 or H4)

| model | dataset | n | A | S | Δ = A − S [95% CI] | log-odds, unadjusted [95% CI] | follow alarm card | follow safe card | true card points to alarm | C0 share alarm |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-instruct | physionet2012 | 600 | 0.068 | 0.500 | -0.43 [-0.53, -0.33] | +0.96 [+0.84, +1.08] | 0.207 | 0.913 | 0.390 | 0.163 |
| llama-instruct | physionet2019 | 600 | 0.540 | 0.649 | -0.11 [-0.19, -0.03] | -0.65 [-0.75, -0.56] | 0.708 | 0.820 | 0.385 | 0.432 |
| llama-instruct | diabetes130 | 600 | 0.425 | 0.885 | -0.46 [-0.53, -0.39] | -2.22 [-2.38, -2.05] | 0.582 | 0.927 | 0.225 | 0.565 |
| qwen-instruct | physionet2012 | 600 | 0.109 | 0.449 | -0.34 [-0.42, -0.26] | +1.05 [+0.49, +1.62] | 0.328 | 0.842 | 0.390 | 0.278 |
| qwen-instruct | physionet2019 | 600 | 0.474 | 0.463 | +0.01 [-0.07, +0.09] | -3.20 [-4.01, -2.38] | 0.722 | 0.717 | 0.385 | 0.515 |
| qwen-instruct | diabetes130 | 600 | 0.638 | 0.791 | -0.15 [-0.23, -0.07] | -0.24 [-0.80, +0.30] | 0.682 | 0.942 | 0.225 | 0.232 |
| llama-base | physionet2012 | 600 | 0.071 | 0.544 | -0.47 [-0.55, -0.40] | -0.34 [-0.36, -0.31] | 0.308 | 0.862 | 0.390 | 0.300 |
| llama-base | physionet2019 | 600 | 0.100 | 0.536 | -0.44 [-0.57, -0.20] | -0.46 [-0.47, -0.45] | 0.888 | 0.543 | 0.385 | 0.983 |
| llama-base | diabetes130 | 600 | 0.597 | 0.738 | -0.14 [-0.22, -0.07] | -0.14 [-0.16, -0.12] | 0.702 | 0.883 | 0.225 | 0.388 |
| qwen-base | physionet2012 | 600 | 0.113 | 0.350 | -0.24 [-0.32, -0.15] | -0.05 [-0.09, -0.00] | 0.305 | 0.827 | 0.390 | 0.233 |
| qwen-base | physionet2019 | 600 | 0.453 | 0.273 | +0.18 [+0.10, +0.26] | -0.07 [-0.11, -0.02] | 0.785 | 0.512 | 0.385 | 0.647 |
| qwen-base | diabetes130 | 600 | 0.224 | 0.886 | -0.66 [-0.72, -0.60] | -0.47 [-0.52, -0.43] | 0.372 | 0.957 | 0.225 | 0.367 |
## Sensitivity to the top-1,000 reading (labelled; D5)

In records where one variant of a word was outside the top 1,000, the less likely word's probability is halved, which bounds the effect of the imputation. 2,000 bootstrap resamples.

| model | dataset | records with an imputed variant | a [95% CI] (main) | a, bounded | a' [95% CI] (main) | a', bounded |
|---|---|---|---|---|---|---|
| qwen-instruct | physionet2012 | 15.5% | -1.71 [-2.56, -0.89] | -1.70 [-2.56, -0.88] | -2.58 [-3.15, -2.01] | -2.64 [-3.22, -2.09] |
| qwen-instruct | physionet2019 | 10.1% | -2.27 [-2.87, -1.75] | -2.33 [-2.92, -1.78] | -2.70 [-3.08, -2.32] | -2.76 [-3.14, -2.40] |
| qwen-instruct | diabetes130 | 4.6% | -2.23 [-4.84, +0.24] | -2.25 [-4.97, +0.17] | -4.01 [-4.38, -3.66] | -4.04 [-4.41, -3.68] |


## Exploratory (labelled; not used for conclusions)

### Base models on H1–H3 (no multiplicity correction)

| model | dataset | H1 diff (p) | H2 diff (p) | a [95% CI] (p) |
|---|---|---|---|---|
| qwen-base | physionet2012 | +0.01 (0.3) | -0.06 (<0.001) | -0.14 [-0.21, -0.07] (<0.001) |
| qwen-base | physionet2019 | +0.04 (0.037) | -0.06 (0.0016) | +0.08 [-0.01, +0.18] (0.075) |
| qwen-base | diabetes130 | -0.04 (0.12) | -0.06 (0.012) | -0.57 [-0.65, -0.47] (<0.001) |
| llama-base | physionet2012 | +0.01 (0.6) | -0.08 (<0.001) | -0.24 [-0.27, -0.21] (<0.001) |
| llama-base | physionet2019 | +0.15 (<0.001) | -0.02 (0.24) | -0.61 [-0.93, -0.27] (<0.001) |
| llama-base | diabetes130 | -0.00 (1) | -0.09 (0.0013) | -0.11 [-0.14, -0.07] (<0.001) |

### C6 (another patient's card) versus C4 (flipped card), and robustness across example sets

| model | dataset | accuracy C4 / C6 | follow rate C2 / C4 / C6 | a with set A / B / C |
|---|---|---|---|---|
| llama-instruct | physionet2012 | 0.562 / 0.533 | 0.795 / 0.325 / 0.353 | -0.59 / -0.91 / -1.12 |
| llama-instruct | physionet2019 | 0.482 / 0.515 | 0.832 / 0.697 / 0.443 | -0.55 / -1.57 / -1.79 |
| llama-instruct | diabetes130 | 0.490 / 0.500 | 0.850 / 0.658 / 0.225 | -2.09 / -1.30 / -2.34 |
| qwen-instruct | physionet2012 | 0.597 / 0.605 | 0.877 / 0.293 / 0.325 | -1.71 / -4.23 / -1.53 |
| qwen-instruct | physionet2019 | 0.487 / 0.552 | 0.773 / 0.665 / 0.477 | -2.27 / -3.76 / -2.58 |
| qwen-instruct | diabetes130 | 0.458 / 0.505 | 0.877 / 0.747 / 0.273 | -2.23 / -2.23 / -3.54 |
| llama-base | physionet2012 | 0.592 / 0.517 | 0.845 / 0.325 / 0.473 | -0.24 / -0.14 / -0.22 |
| llama-base | physionet2019 | 0.477 / 0.452 | 0.763 / 0.668 / 0.713 | -0.61 / -0.47 / -0.50 |
| llama-base | diabetes130 | 0.458 / 0.500 | 0.852 / 0.733 / 0.495 | -0.11 / -0.09 / +0.02 |
| qwen-base | physionet2012 | 0.615 / 0.550 | 0.860 / 0.272 / 0.357 | -0.14 / +0.06 / -0.20 |
| qwen-base | physionet2019 | 0.510 / 0.532 | 0.695 / 0.602 / 0.547 | +0.08 / +0.16 / +0.19 |
| qwen-base | diabetes130 | 0.517 / 0.510 | 0.837 / 0.492 / 0.245 | -0.57 / -0.60 / -0.44 |

### Accuracy by condition (alarm framing, example set A)

| model | dataset | C0 | C2 | C4 | C6 | AUROC C0 |
|---|---|---|---|---|---|---|
| llama-instruct | physionet2012 | 0.607 | 0.642 | 0.562 | 0.533 | 0.771 |
| llama-instruct | physionet2019 | 0.618 | 0.660 | 0.482 | 0.515 | 0.651 |
| llama-instruct | diabetes130 | 0.558 | 0.528 | 0.490 | 0.500 | 0.600 |
| qwen-instruct | physionet2012 | 0.692 | 0.707 | 0.597 | 0.605 | 0.783 |
| qwen-instruct | physionet2019 | 0.608 | 0.642 | 0.487 | 0.552 | 0.654 |
| qwen-instruct | diabetes130 | 0.528 | 0.552 | 0.458 | 0.505 | 0.575 |
| llama-base | physionet2012 | 0.673 | 0.682 | 0.592 | 0.517 | 0.762 |
| llama-base | physionet2019 | 0.500 | 0.645 | 0.477 | 0.452 | 0.631 |
| llama-base | diabetes130 | 0.545 | 0.543 | 0.458 | 0.500 | 0.591 |
| qwen-base | physionet2012 | 0.670 | 0.683 | 0.615 | 0.550 | 0.798 |
| qwen-base | physionet2019 | 0.570 | 0.607 | 0.510 | 0.532 | 0.613 |
| qwen-base | diabetes130 | 0.577 | 0.542 | 0.517 | 0.510 | 0.606 |
