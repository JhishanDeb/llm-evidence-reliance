# Registered analysis, second study: results

Protocol `prereg-v2`; analysis code main-2.0 (sha256 0ba0853b4b52b897).

## Runs

| study | model | revision | grids | read | records | invalid | imputed |
|---|---|---|---|---|---|---|---|
| v2 | ibm-granite/granite-3.3-8b-instruct | 51dd4bc2ade4 | yesno, choice, order | topk | 41400 | 0 | 2165 |
| v2 | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a1 | choice, order | topk | 16200 | 0 | 506 |
| v2 | allenai/Olmo-3-1025-7B | a81bae42db39 | yesno | topk | 25200 | 0 | 0 |
| v2 | allenai/Olmo-3-7B-Instruct-DPO | b33130b7de49 | yesno | topk | 25200 | 0 | 160 |
| v2 | allenai/Olmo-3-7B-Instruct | 6e5971d9eba4 | yesno, choice, order, baseformat | topk | 46800 | 0 | 5700 |
| v2 | allenai/Olmo-3-7B-Instruct-SFT | e1452fc572d5 | yesno | topk | 25200 | 0 | 0 |
| v2 | Qwen/Qwen2.5-7B-Instruct | a09a35458c70 | choice, order | topk | 16200 | 0 | 7269 |
| v2 | Qwen/Qwen3-8B | b968826d9c46 | yesno, choice, order | topk | 41400 | 0 | 3257 |
| v1 | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a1 | yesno (first study) | topk | 25200 | 0 | 0 |
| v1 | Qwen/Qwen2.5-7B-Instruct | a09a35458c70 | yesno (first study) | topk | 25200 | 0 | 2539 |

## Validity

| model | dataset | Yes/No: median mass | Yes/No: C0 accuracy | Yes/No status | A/B: median mass | A/B: C0 accuracy | A/B status |
|---|---|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | 0.991 | 0.693 | passes | 0.998 | 0.642 | passes |
| granite3-instruct | physionet2019 | 0.977 | 0.543 | **excluded** | 0.999 | 0.513 | **excluded** |
| granite3-instruct | diabetes130 | 0.991 | 0.510 | **excluded** | 0.998 | 0.562 | passes |
| llama-instruct | physionet2012 | – | – | – | 0.998 | 0.650 | passes |
| llama-instruct | physionet2019 | – | – | – | 0.998 | 0.593 | passes |
| llama-instruct | diabetes130 | – | – | – | 0.998 | 0.538 | **excluded** |
| olmo3-base | physionet2012 | 0.775 | 0.565 | passes | – | – | – |
| olmo3-base | physionet2019 | 0.818 | 0.563 | passes | – | – | – |
| olmo3-base | diabetes130 | 0.606 | 0.500 | **excluded** | – | – | – |
| olmo3-dpo | physionet2012 | 0.977 | 0.618 | passes | – | – | – |
| olmo3-dpo | physionet2019 | 0.982 | 0.597 | passes | – | – | – |
| olmo3-dpo | diabetes130 | 0.983 | 0.538 | **excluded** | – | – | – |
| olmo3-instruct | physionet2012 | 0.997 | 0.612 | passes | 0.999 | 0.670 | passes |
| olmo3-instruct | physionet2019 | 0.996 | 0.597 | passes | 1.000 | 0.627 | passes |
| olmo3-instruct | diabetes130 | 0.997 | 0.540 | **excluded** | 1.000 | 0.502 | **excluded** |
| olmo3-sft | physionet2012 | 0.994 | 0.615 | passes | – | – | – |
| olmo3-sft | physionet2019 | 0.994 | 0.583 | passes | – | – | – |
| olmo3-sft | diabetes130 | 0.993 | 0.565 | passes | – | – | – |
| qwen-instruct | physionet2012 | – | – | – | 1.000 | 0.625 | passes |
| qwen-instruct | physionet2019 | – | – | – | 1.000 | 0.588 | passes |
| qwen-instruct | diabetes130 | – | – | – | 1.000 | 0.533 | **excluded** |
| qwen3-instruct | physionet2012 | 0.737 | 0.678 | passes | 0.441 | 0.717 | **excluded** |
| qwen3-instruct | physionet2019 | 0.547 | 0.652 | passes | 0.762 | 0.642 | passes |
| qwen3-instruct | diabetes130 | 0.673 | 0.563 | passes | 0.282 | 0.547 | **excluded** |

## Memorization check

| model | dataset | file | header: chars matched | rows completed verbatim | flagged |
|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | Outcomes-a.txt | 2 | 0 | no |
| granite3-instruct | physionet2019 | p000001.psv | 60 | 0 | no |
| granite3-instruct | diabetes130 | diabetic_data.csv | 19 | 0 | no |
| olmo3-base | physionet2012 | Outcomes-a.txt | 3 | 0 | no |
| olmo3-base | physionet2019 | p000001.psv | 48 | 0 | no |
| olmo3-base | diabetes130 | diabetic_data.csv | 19 | 0 | no |
| olmo3-dpo | physionet2012 | Outcomes-a.txt | 2 | 0 | no |
| olmo3-dpo | physionet2019 | p000001.psv | 0 | 0 | no |
| olmo3-dpo | diabetes130 | diabetic_data.csv | 0 | 0 | no |
| olmo3-instruct | physionet2012 | Outcomes-a.txt | 3 | 0 | no |
| olmo3-instruct | physionet2019 | p000001.psv | 0 | 0 | no |
| olmo3-instruct | diabetes130 | diabetic_data.csv | 0 | 0 | no |
| olmo3-sft | physionet2012 | Outcomes-a.txt | 1 | 0 | no |
| olmo3-sft | physionet2019 | p000001.psv | 0 | 0 | no |
| olmo3-sft | diabetes130 | diabetic_data.csv | 1 | 0 | no |
| qwen3-instruct | physionet2012 | Outcomes-a.txt | 1 | 0 | no |
| qwen3-instruct | physionet2019 | p000001.psv | 1 | 0 | no |
| qwen3-instruct | diabetes130 | diabetic_data.csv | 19 | 0 | no |

## R1. Replication: a < 0 and a' < 0 in the new instruct models (alarm framing, example set A)

Intersection-union test, Holm over the 9 cells.

| model | dataset | a [95% CI] (p) | a' [95% CI] (p) | test p | Holm p | supported | registered Δ = A − S [95% CI] |
|---|---|---|---|---|---|---|---|
| olmo3-instruct | physionet2012 | +1.51 [+0.79, +2.12] (<0.001) | -0.10 [-0.26, +0.05] (0.2) | 0.2 | 0.79 | no | -0.08 [-0.16, -0.00] |
| olmo3-instruct | physionet2019 | -1.87 [-2.28, -1.51] (<0.001) | -1.33 [-1.44, -1.22] (<0.001) | <0.001 | <0.001 | **yes** | -0.51 [-0.59, -0.44] |
| olmo3-instruct | diabetes130 | -1.08 [-1.54, -0.71] (<0.001) | -0.60 [-0.73, -0.47] (<0.001) | <0.001 | 1 | no (not included) | -0.15 [-0.22, -0.06] |
| qwen3-instruct | physionet2012 | -5.64 [-6.57, -4.72] (<0.001) | -6.76 [-7.34, -6.19] (<0.001) | <0.001 | <0.001 | **yes** | -0.56 [-0.64, -0.49] |
| qwen3-instruct | physionet2019 | -1.74 [-2.26, -1.28] (<0.001) | -2.21 [-2.63, -1.80] (<0.001) | <0.001 | <0.001 | **yes** | -0.12 [-0.19, -0.05] |
| qwen3-instruct | diabetes130 | -5.73 [-7.12, -4.50] (<0.001) | -5.87 [-6.35, -5.40] (<0.001) | <0.001 | <0.001 | **yes** | -0.16 [-0.22, -0.10] |
| granite3-instruct | physionet2012 | -0.60 [-0.97, -0.21] (0.0026) | -0.66 [-0.99, -0.33] (<0.001) | 0.0026 | 0.013 | **yes** | +0.03 [-0.05, +0.11] |
| granite3-instruct | physionet2019 | -4.01 [-7.27, -0.87] (0.014) | -1.46 [-1.74, -1.18] (<0.001) | 0.014 | 1 | no (not included) | +0.13 [+0.08, +0.18] |
| granite3-instruct | diabetes130 | -14.29 [-29.99, -2.24] (0.017) | +0.24 [-0.76, +1.22] (0.65) | 0.65 | 1 | no (not included) | +0.36 [+0.32, +0.39] |

**R1 replicated: NO** (olmo3-instruct: 1/3 datasets; qwen3-instruct: 3/3 datasets; granite3-instruct: 1/3 datasets; rule: at least 2 of 3 datasets for at least 2 of the 3 new models)

Where a, a' and the registered Δ do not agree in sign, the cell is described as not robust (Deviation D1 of the first study).

## R2. Outcome or Yes/No question format: a < 0 and a' < 0 in the answer-option framing

Mean of the two option orders. Eligible models: the two first-study models and each new model that meets the R1 per-model criterion (qwen-instruct, llama-instruct, qwen3-instruct). Intersection-union test, Holm over their cells.

| model | dataset | a [95% CI] (p) | a' [95% CI] (p) | test p | Holm p | supported |
|---|---|---|---|---|---|---|
| qwen-instruct | physionet2012 | -4.15 [-5.69, -2.56] (<0.001) | -4.59 [-5.35, -3.82] (<0.001) | <0.001 | <0.001 | **yes** |
| qwen-instruct | physionet2019 | -2.01 [-3.20, -0.98] (<0.001) | -0.49 [-0.87, -0.12] (0.0094) | 0.0094 | 0.056 | no |
| qwen-instruct | diabetes130 | -0.05 [-4.36, +3.87] (0.73) | -5.58 [-6.00, -5.15] (<0.001) | 0.73 | 1 | no (not included) |
| llama-instruct | physionet2012 | -0.15 [-0.28, -0.01] (0.038) | -0.26 [-0.32, -0.20] (<0.001) | 0.038 | 0.19 | no |
| llama-instruct | physionet2019 | -1.12 [-1.36, -0.93] (<0.001) | -0.96 [-1.04, -0.90] (<0.001) | <0.001 | <0.001 | **yes** |
| llama-instruct | diabetes130 | -0.23 [-1.32, +1.04] (0.59) | -0.82 [-1.00, -0.65] (<0.001) | 0.59 | 1 | no (not included) |
| qwen3-instruct | physionet2012 | -2.96 [-3.41, -2.53] (<0.001) | -2.89 [-3.32, -2.47] (<0.001) | <0.001 | 1 | no (not included) |
| qwen3-instruct | physionet2019 | -2.14 [-2.62, -1.70] (<0.001) | -2.18 [-2.59, -1.78] (<0.001) | <0.001 | <0.001 | **yes** |
| qwen3-instruct | diabetes130 | -7.50 [-9.00, -5.94] (<0.001) | -7.89 [-8.31, -7.47] (<0.001) | <0.001 | 1 | no (not included) |

**R2 supported: NO** (qwen-instruct: 1/3 datasets; llama-instruct: 1/3 datasets; qwen3-instruct: 1/3 datasets; rule: at least 2 of 3 datasets for at least 2 of the 3 eligible models)

### Secondary comparison: δ = a(answer option) − a(alarm Yes/No), same patients

When a < 0 with Yes/No, δ > 0 means the asymmetry is weaker with answer options. A cell counts for the interpretation only if it is valid in both framings and a and a' are both below 0 with Yes/No.

| model | dataset | a, a' with Yes/No | δ for a [95% CI] (p) | δ for a' [95% CI] (p) | test p | counts |
|---|---|---|---|---|---|---|
| olmo3-instruct | physionet2012 | +1.51, -0.10 | +0.23 [-0.40, +0.91] (0.42) | +1.61 [+1.47, +1.76] (<0.001) | 0.42 | no |
| olmo3-instruct | physionet2019 | -1.87, -1.33 | +2.18 [+1.72, +2.65] (<0.001) | +1.47 [+1.36, +1.57] (<0.001) | <0.001 | no |
| olmo3-instruct | diabetes130 | -1.08, -0.60 | +4.13 [-7.94, +16.85] (0.33) | +1.37 [+1.19, +1.54] (<0.001) | 0.33 | no |
| qwen3-instruct | physionet2012 | -5.64, -6.76 | +2.68 [+1.88, +3.48] (<0.001) | +3.87 [+3.44, +4.31] (<0.001) | <0.001 | no |
| qwen3-instruct | physionet2019 | -1.74, -2.21 | -0.40 [-0.80, +0.02] (0.064) | +0.03 [-0.40, +0.46] (0.88) | 0.88 | yes |
| qwen3-instruct | diabetes130 | -5.73, -5.87 | -1.76 [-3.11, -0.24] (0.033) | -2.02 [-2.49, -1.56] (<0.001) | 0.033 | no |
| granite3-instruct | physionet2012 | -0.60, -0.66 | +6.10 [+5.05, +7.15] (<0.001) | +6.12 [+5.84, +6.40] (<0.001) | <0.001 | no |
| granite3-instruct | physionet2019 | -4.01, -1.46 | -22.95 [-40.98, -3.77] (0.014) | +8.68 [+7.83, +9.50] (<0.001) | 0.014 | no |
| granite3-instruct | diabetes130 | -14.29, +0.24 | +17.92 [+6.25, +33.41] (0.0032) | +4.74 [+3.85, +5.69] (<0.001) | 0.0032 | no |
| qwen-instruct | physionet2012 | -1.71, -2.58 | -2.45 [-3.81, -1.02] (0.0014) | -2.01 [-2.71, -1.32] (<0.001) | 0.0014 | yes |
| qwen-instruct | physionet2019 | -2.27, -2.70 | +0.27 [-0.99, +1.57] (0.58) | +2.21 [+1.82, +2.58] (<0.001) | 0.58 | yes |
| qwen-instruct | diabetes130 | -2.23, -4.01 | +2.17 [-0.98, +5.20] (0.22) | -1.56 [-2.08, -1.04] (<0.001) | 0.22 | no |
| llama-instruct | physionet2012 | -0.59, -0.61 | +0.44 [+0.21, +0.66] (<0.001) | +0.35 [+0.23, +0.48] (<0.001) | <0.001 | yes |
| llama-instruct | physionet2019 | -0.55, -0.63 | -0.57 [-0.79, -0.36] (<0.001) | -0.34 [-0.40, -0.27] (<0.001) | <0.001 | yes |
| llama-instruct | diabetes130 | -2.09, -1.93 | +1.86 [+0.75, +3.12] (0.0026) | +1.10 [+0.92, +1.28] (<0.001) | 0.0026 | no |

**R2 interpretation (Section 6 rule): unresolved.**

## R3. Instruction tuning in Olmo 3: a(final) − a(base) < 0 and the same for a'

Validity for R3 and the stage trace: median Yes+No mass ≥ 0.5 and C0 AUROC ≥ 0.55.

| dataset | a difference [95% CI] | a' difference [95% CI] | test p | Holm p | supported |
|---|---|---|---|---|---|
| physionet2012 | +0.93 [+0.23, +1.52] | -0.50 [-0.67, -0.35] | 0.009 | 0.018 | no |
| physionet2019 | -2.32 [-2.78, -1.92] | -1.68 [-1.80, -1.57] | <0.001 | <0.001 | **yes** |
| diabetes130 | -3.72 [-8.10, +0.90] | -1.21 [-1.44, -0.96] | 0.083 | 0.083 | no |

**R3 supported: NO** (1/3 datasets; rule: at least 2 of 3)

## Olmo 3 stage trace (described, no decision rule)

Stages that fail the R3 validity rule are shown and marked. The base model is prompted as plain text; the SFT, DPO and final models through their chat template. Default system text of the templates: olmo3-base: None; olmo3-sft: 'You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>'; olmo3-dpo: 'You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>'; olmo3-instruct: 'You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>'.

| dataset | stage | valid (mass; C0 AUROC) | a [95% CI] | a' [95% CI] |
|---|---|---|---|---|
| physionet2012 | olmo3-base | yes (0.775; 0.752) | +0.58 [+0.53, +0.64] | +0.40 [+0.37, +0.43] |
| physionet2012 | olmo3-sft | yes (0.994; 0.706) | +0.33 [+0.07, +0.56] | -0.22 [-0.29, -0.16] |
| physionet2012 | olmo3-dpo | yes (0.977; 0.696) | +1.25 [+0.70, +1.71] | +0.04 [-0.10, +0.16] |
| physionet2012 | olmo3-instruct | yes (0.997; 0.693) | +1.51 [+0.79, +2.12] | -0.10 [-0.26, +0.05] |
| physionet2019 | olmo3-base | yes (0.818; 0.652) | +0.45 [+0.28, +0.65] | +0.35 [+0.31, +0.40] |
| physionet2019 | olmo3-sft | yes (0.994; 0.670) | -1.33 [-1.61, -1.08] | -0.94 [-1.00, -0.88] |
| physionet2019 | olmo3-dpo | yes (0.982; 0.680) | -1.85 [-2.19, -1.56] | -1.39 [-1.49, -1.30] |
| physionet2019 | olmo3-instruct | yes (0.996; 0.681) | -1.87 [-2.28, -1.51] | -1.33 [-1.44, -1.22] |
| diabetes130 | olmo3-base | yes (0.606; 0.564) | +2.64 [-1.90, +6.94] | +0.61 [+0.40, +0.80] |
| diabetes130 | olmo3-sft | yes (0.993; 0.584) | -0.09 [-0.18, +0.03] | -0.14 [-0.20, -0.08] |
| diabetes130 | olmo3-dpo | yes (0.983; 0.560) | -1.50 [-2.16, -1.01] | -0.77 [-0.90, -0.63] |
| diabetes130 | olmo3-instruct | yes (0.997; 0.564) | -1.08 [-1.54, -0.71] | -0.60 [-0.73, -0.47] |

| dataset | step | a change [95% CI] | a' change [95% CI] |
|---|---|---|---|
| physionet2012 | olmo3-sft - olmo3-base | -0.25 [-0.50, -0.04] | -0.63 [-0.70, -0.56] |
| physionet2012 | olmo3-dpo - olmo3-sft | +0.91 [+0.61, +1.17] | +0.26 [+0.18, +0.33] |
| physionet2012 | olmo3-instruct - olmo3-dpo | +0.26 [+0.05, +0.45] | -0.14 [-0.18, -0.10] |
| physionet2019 | olmo3-sft - olmo3-base | -1.78 [-2.12, -1.48] | -1.29 [-1.36, -1.23] |
| physionet2019 | olmo3-dpo - olmo3-sft | -0.52 [-0.66, -0.41] | -0.45 [-0.51, -0.40] |
| physionet2019 | olmo3-instruct - olmo3-dpo | -0.02 [-0.11, +0.06] | +0.06 [+0.04, +0.09] |
| diabetes130 | olmo3-sft - olmo3-base | -2.73 [-7.01, +1.83] | -0.75 [-0.95, -0.54] |
| diabetes130 | olmo3-dpo - olmo3-sft | -1.41 [-2.09, -0.90] | -0.63 [-0.72, -0.53] |
| diabetes130 | olmo3-instruct - olmo3-dpo | +0.43 [+0.22, +0.71] | +0.17 [+0.14, +0.20] |

### Base-format control (labelled): Olmo-3-7B-Instruct prompted as plain text, like the base model

| dataset | a, final in plain text [95% CI] | final (plain) − base: a [95% CI] | final (chat) − final (plain): a [95% CI] |
|---|---|---|---|
| physionet2012 | – [-2681.10, +5653.25] | – [-2681.65, +5652.68] | – [-5651.69, +2682.77] |
| physionet2019 | +2167.17 [-351.67, +4260.45] | +2166.72 [-352.10, +4259.88] | -2169.04 [-4262.49, +350.10] |
| diabetes130 | – [-4268.80, +2344.82] | – [-4271.24, +2344.17] | – [-2345.86, +4267.80] |

## Example-order control (labelled): the four solved examples in the order safe first

In the registered order the last solved example always has the safe outcome. Here it has the alarming one.

| model | dataset | a, safe first [95% CI] | a', safe first [95% CI] | change in a [95% CI] | change in a' [95% CI] |
|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | -0.46 [-0.86, -0.05] | +0.01 [-0.30, +0.34] | +0.13 [-0.20, +0.49] | +0.68 [+0.46, +0.90] |
| granite3-instruct | physionet2019 | -1.99 [-2.44, -1.50] | -2.43 [-2.64, -2.22] | +2.02 [-1.00, +5.14] | -0.98 [-1.20, -0.75] |
| granite3-instruct | diabetes130 | -25.50 [-77.93, -0.43] | +0.99 [+0.06, +1.96] | -11.21 [-54.79, +8.79] | +0.74 [-0.02, +1.57] |
| llama-instruct | physionet2012 | -1.32 [-1.64, -0.99] | -1.34 [-1.50, -1.18] | -0.74 [-0.88, -0.61] | -0.72 [-0.80, -0.65] |
| llama-instruct | physionet2019 | -0.66 [-0.84, -0.53] | -0.65 [-0.73, -0.57] | -0.11 [-0.26, +0.07] | -0.02 [-0.07, +0.03] |
| llama-instruct | diabetes130 | -2.03 [-2.36, -1.60] | -2.21 [-2.34, -2.09] | +0.06 [-0.37, +0.56] | -0.29 [-0.39, -0.19] |
| olmo3-instruct | physionet2012 | +1.18 [+0.42, +1.83] | -0.10 [-0.30, +0.08] | -0.33 [-0.69, +0.03] | +0.00 [-0.11, +0.11] |
| olmo3-instruct | physionet2019 | -2.22 [-3.84, -0.73] | -0.00 [-0.13, +0.13] | -0.34 [-1.67, +0.88] | +1.33 [+1.25, +1.41] |
| olmo3-instruct | diabetes130 | -1.27 [-1.89, -0.83] | -0.66 [-0.80, -0.52] | -0.19 [-0.68, +0.22] | -0.06 [-0.13, +0.01] |
| qwen-instruct | physionet2012 | -7.01 [-8.11, -5.97] | -7.63 [-8.36, -6.93] | -5.30 [-5.92, -4.68] | -5.05 [-5.49, -4.60] |
| qwen-instruct | physionet2019 | -3.21 [-4.06, -2.44] | -3.87 [-4.26, -3.49] | -0.94 [-1.53, -0.34] | -1.17 [-1.47, -0.88] |
| qwen-instruct | diabetes130 | -3.93 [-8.39, -0.06] | -4.72 [-5.15, -4.31] | -1.70 [-4.78, +0.90] | -0.71 [-1.02, -0.39] |
| qwen3-instruct | physionet2012 | -8.17 [-9.49, -6.93] | -8.70 [-9.36, -8.08] | -2.53 [-3.28, -1.85] | -1.95 [-2.31, -1.59] |
| qwen3-instruct | physionet2019 | -5.17 [-6.24, -4.27] | -4.73 [-5.18, -4.29] | -3.42 [-4.31, -2.63] | -2.52 [-2.83, -2.20] |
| qwen3-instruct | diabetes130 | -5.31 [-6.60, -4.00] | -5.62 [-6.06, -5.19] | +0.42 [-0.45, +1.44] | +0.25 [-0.09, +0.60] |

## Secondary (labelled; as in the first study)

### H1 and H2: accuracy with true (C2) and flipped (C4) evidence against none (C0), exact McNemar, no correction

| model | dataset | C0 | C2 − C0 (p) | C4 − C0 (p) |
|---|---|---|---|---|
| granite3-instruct | physionet2012 | 0.693 | +0.03 (0.08) | -0.22 (<0.001) |
| granite3-instruct | physionet2019 | 0.543 | +0.09 (0.0013) | -0.17 (<0.001) |
| granite3-instruct | diabetes130 | 0.510 | +0.06 (0.081) | -0.04 (<0.001) |
| olmo3-base | physionet2012 | 0.565 | +0.13 (<0.001) | -0.04 (0.13) |
| olmo3-base | physionet2019 | 0.563 | +0.07 (<0.001) | -0.10 (<0.001) |
| olmo3-base | diabetes130 | 0.500 | +0.04 (0.021) | -0.02 (0.43) |
| olmo3-dpo | physionet2012 | 0.618 | +0.04 (<0.001) | -0.03 (0.029) |
| olmo3-dpo | physionet2019 | 0.597 | +0.02 (0.4) | -0.09 (<0.001) |
| olmo3-dpo | diabetes130 | 0.538 | +0.02 (0.45) | -0.06 (0.0033) |
| olmo3-instruct | physionet2012 | 0.612 | +0.03 (0.011) | -0.03 (0.012) |
| olmo3-instruct | physionet2019 | 0.597 | +0.04 (0.03) | -0.10 (<0.001) |
| olmo3-instruct | diabetes130 | 0.540 | +0.02 (0.48) | -0.06 (0.0063) |
| olmo3-sft | physionet2012 | 0.615 | +0.03 (0.0051) | -0.05 (<0.001) |
| olmo3-sft | physionet2019 | 0.583 | +0.02 (0.23) | -0.12 (<0.001) |
| olmo3-sft | diabetes130 | 0.565 | -0.01 (0.59) | -0.09 (0.0028) |
| qwen3-instruct | physionet2012 | 0.678 | +0.01 (0.41) | -0.14 (<0.001) |
| qwen3-instruct | physionet2019 | 0.652 | +0.01 (0.52) | -0.07 (<0.001) |
| qwen3-instruct | diabetes130 | 0.563 | -0.01 (0.69) | -0.12 (<0.001) |

### Q3 with the reversed Yes/No question (the first study's four-way table)

| model | dataset | a: alarm / reversed [95% CI] | a': alarm / reversed [95% CI] | interpretation |
|---|---|---|---|---|
| granite3-instruct | physionet2012 | -0.60 [-0.97, -0.21] / +6.40 [+6.01, +6.80] | -0.66 [-0.99, -0.33] / +6.52 [+6.21, +6.84] | follows the answer word "No" |
| granite3-instruct | physionet2019 | -4.01 [-7.27, -0.87] / -9.13 [-16.50, -1.50] | -1.46 [-1.74, -1.18] / +8.46 [+7.72, +9.19] | inconclusive (a and a' disagree) (cell excluded) |
| granite3-instruct | diabetes130 | -14.29 [-29.99, -2.24] / +6.52 [+3.39, +9.05] | +0.24 [-0.76, +1.22] / +9.35 [+8.88, +9.80] | inconclusive (a and a' disagree) (cell excluded) |
| olmo3-base | physionet2012 | +0.58 [+0.53, +0.64] / +0.42 [+0.40, +0.44] | +0.40 [+0.37, +0.43] / +0.38 [+0.36, +0.40] | follows the outcome, toward alarm (alarm aversion) |
| olmo3-base | physionet2019 | +0.45 [+0.28, +0.65] / +0.39 [-0.04, +0.83] | +0.35 [+0.31, +0.40] / +0.43 [+0.40, +0.47] | inconclusive (a and a' disagree) |
| olmo3-base | diabetes130 | +2.64 [-1.90, +6.94] / +5.47 [+1.86, +10.82] | +0.61 [+0.40, +0.80] / +0.72 [+0.48, +0.92] | inconclusive (a and a' disagree) (cell excluded) |
| olmo3-dpo | physionet2012 | +1.25 [+0.70, +1.71] / +3.34 [+3.19, +3.50] | +0.04 [-0.10, +0.16] / +3.37 [+3.24, +3.51] | inconclusive (a and a' disagree) |
| olmo3-dpo | physionet2019 | -1.85 [-2.19, -1.56] / +0.62 [-0.77, +2.26] | -1.39 [-1.49, -1.30] / +0.68 [+0.45, +0.91] | inconclusive (a and a' disagree) |
| olmo3-dpo | diabetes130 | -1.50 [-2.16, -1.01] / -10.27 [-41.33, +22.04] | -0.77 [-0.90, -0.63] / +3.48 [+2.89, +4.07] | inconclusive (a and a' disagree) (cell excluded) |
| olmo3-instruct | physionet2012 | +1.51 [+0.79, +2.12] / +3.16 [+2.99, +3.33] | -0.10 [-0.26, +0.05] / +3.03 [+2.89, +3.17] | inconclusive (a and a' disagree) |
| olmo3-instruct | physionet2019 | -1.87 [-2.28, -1.51] / +1.39 [+0.43, +2.47] | -1.33 [-1.44, -1.22] / +1.35 [+1.15, +1.55] | follows the answer word "No" |
| olmo3-instruct | diabetes130 | -1.08 [-1.54, -0.71] / -2.69 [-30.58, +22.45] | -0.60 [-0.73, -0.47] / +3.79 [+3.32, +4.29] | inconclusive (a and a' disagree) (cell excluded) |
| olmo3-sft | physionet2012 | +0.33 [+0.07, +0.56] / +1.58 [+1.51, +1.65] | -0.22 [-0.29, -0.16] / +1.54 [+1.47, +1.61] | inconclusive (a and a' disagree) |
| olmo3-sft | physionet2019 | -1.33 [-1.61, -1.08] / +0.51 [+0.31, +0.75] | -0.94 [-1.00, -0.88] / +0.34 [+0.27, +0.42] | follows the answer word "No" |
| olmo3-sft | diabetes130 | -0.09 [-0.18, +0.03] / -6.76 [-16.24, +4.77] | -0.14 [-0.20, -0.08] / +2.65 [+2.47, +2.83] | inconclusive (a and a' disagree) |
| qwen3-instruct | physionet2012 | -5.64 [-6.57, -4.72] / -0.15 [-1.18, +0.83] | -6.76 [-7.34, -6.19] / +0.82 [+0.35, +1.28] | inconclusive (a and a' disagree) |
| qwen3-instruct | physionet2019 | -1.74 [-2.26, -1.28] / -2.02 [-4.30, +0.11] | -2.21 [-2.63, -1.80] / +1.29 [+0.93, +1.65] | inconclusive (a and a' disagree) |
| qwen3-instruct | diabetes130 | -5.73 [-7.12, -4.50] / -33.58 [-72.67, -1.64] | -5.87 [-6.35, -5.40] / +3.01 [+1.59, +4.44] | inconclusive (a and a' disagree) |

### Robustness and sensitivity of a (Yes/No grid)

| model | dataset | a with set A / B / C | a, d winsorized [95% CI] | share of expected baselines within 1 SD of 0 | first-stage R² | log-odds, unadjusted [95% CI] | follow alarm / safe card |
|---|---|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | -0.60 / -1.07 / -2.35 | -0.59 [-0.97, -0.20] | 0.617 | 0.615 | +0.54 [+0.03, +1.06] | 0.727 / 0.745 |
| granite3-instruct | physionet2019 | -4.01 / -4.70 / -13.97 | -3.88 [-7.08, -0.83] | 0.000 | 0.107 | -5.25 [-5.64, -4.87] | 0.958 / 0.833 |
| granite3-instruct | diabetes130 | -14.29 / -42.82 / -26.05 | -13.97 [-29.27, -2.10] | 0.000 | 0.082 | -5.91 [-6.21, -5.61] | 0.960 / 0.647 |
| olmo3-base | physionet2012 | +0.58 / +0.57 / +0.40 | +0.58 [+0.52, +0.63] | 0.228 | 0.552 | +0.65 [+0.63, +0.67] | 0.510 / 0.787 |
| olmo3-base | physionet2019 | +0.45 / +0.63 / +0.60 | +0.45 [+0.28, +0.65] | 0.000 | 0.217 | +0.37 [+0.35, +0.38] | 0.543 / 0.870 |
| olmo3-base | diabetes130 | +2.64 / +0.60 / +1.68 | +2.61 [-1.82, +6.88] | 0.000 | 0.045 | +0.96 [+0.94, +0.98] | 0.543 / 0.943 |
| olmo3-dpo | physionet2012 | +1.25 / +1.22 / +0.39 | +1.24 [+0.70, +1.71] | 0.273 | 0.376 | +3.62 [+3.29, +3.94] | 0.297 / 0.817 |
| olmo3-dpo | physionet2019 | -1.85 / -2.57 / -1.69 | -1.85 [-2.18, -1.56] | 0.378 | 0.215 | -0.46 [-0.61, -0.31] | 0.377 / 0.927 |
| olmo3-dpo | diabetes130 | -1.50 / -7.11 / -2.47 | -1.50 [-2.14, -1.02] | 0.162 | 0.036 | -1.13 [-1.25, -1.01] | 0.803 / 0.780 |
| olmo3-instruct | physionet2012 | +1.51 / +1.54 / +0.52 | +1.50 [+0.79, +2.12] | 0.218 | 0.369 | +4.72 [+4.35, +5.10] | 0.263 / 0.842 |
| olmo3-instruct | physionet2019 | -1.87 / -2.44 / -1.63 | -1.87 [-2.28, -1.51] | 0.360 | 0.213 | -0.16 [-0.34, +0.02] | 0.437 / 0.930 |
| olmo3-instruct | diabetes130 | -1.08 / -6.24 / -2.02 | -1.08 [-1.52, -0.72] | 0.347 | 0.036 | -0.93 [-1.07, -0.79] | 0.795 / 0.795 |
| olmo3-sft | physionet2012 | +0.33 / +0.34 / -0.31 | +0.34 [+0.08, +0.56] | 0.282 | 0.398 | +1.63 [+1.46, +1.79] | 0.283 / 0.847 |
| olmo3-sft | physionet2019 | -1.33 / -1.58 / -1.01 | -1.33 [-1.61, -1.07] | 0.285 | 0.186 | -0.30 [-0.39, -0.22] | 0.380 / 0.965 |
| olmo3-sft | diabetes130 | -0.09 / -1.19 / -0.32 | -0.09 [-0.18, +0.03] | 0.450 | 0.038 | -0.05 [-0.12, +0.02] | 0.793 / 0.837 |
| qwen3-instruct | physionet2012 | -5.64 / -6.67 / -7.62 | -5.62 [-6.54, -4.71] | 0.430 | 0.543 | -1.05 [-1.73, -0.34] | 0.303 / 0.900 |
| qwen3-instruct | physionet2019 | -1.74 / -4.26 / -4.14 | -1.74 [-2.25, -1.27] | 0.535 | 0.265 | +0.68 [-0.11, +1.47] | 0.555 / 0.692 |
| qwen3-instruct | diabetes130 | -5.73 / -7.87 / -6.96 | -5.72 [-7.09, -4.49] | 0.400 | 0.072 | -3.10 [-3.85, -2.34] | 0.770 / 0.940 |

### Answer-option framing: accuracy and preference for the letter A

| model | dataset | C0 | C2 | C4 | mean log P(A) − log P(B) |
|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | 0.642 | 0.723 | 0.390 | +0.91 |
| granite3-instruct | physionet2019 | 0.513 | 0.602 | 0.458 | +0.65 |
| granite3-instruct | diabetes130 | 0.562 | 0.578 | 0.492 | +0.82 |
| llama-instruct | physionet2012 | 0.650 | 0.665 | 0.608 | +0.11 |
| llama-instruct | physionet2019 | 0.593 | 0.640 | 0.525 | +1.12 |
| llama-instruct | diabetes130 | 0.538 | 0.580 | 0.485 | -0.13 |
| olmo3-instruct | physionet2012 | 0.670 | 0.710 | 0.542 | +0.46 |
| olmo3-instruct | physionet2019 | 0.627 | 0.658 | 0.455 | +0.39 |
| olmo3-instruct | diabetes130 | 0.502 | 0.562 | 0.475 | +0.89 |
| qwen-instruct | physionet2012 | 0.625 | 0.662 | 0.532 | -2.52 |
| qwen-instruct | physionet2019 | 0.588 | 0.638 | 0.473 | -1.58 |
| qwen-instruct | diabetes130 | 0.533 | 0.552 | 0.443 | -0.86 |
| qwen3-instruct | physionet2012 | 0.717 | 0.725 | 0.575 | -1.76 |
| qwen3-instruct | physionet2019 | 0.642 | 0.658 | 0.572 | -2.68 |
| qwen3-instruct | diabetes130 | 0.547 | 0.538 | 0.457 | -1.22 |

### Answer-option framing, each option order on its own (labelled; 2,000 resamples)

| model | dataset | a, alarming outcome as A | a', alarming as A | a, safe outcome as A | a', safe as A |
|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | +8.74 | +8.35 | +2.38 | +2.52 |
| granite3-instruct | physionet2019 | -51.63 | +5.93 | -10.09 | +7.34 |
| granite3-instruct | diabetes130 | +6.70 | +7.60 | +0.55 | +2.43 |
| llama-instruct | physionet2012 | +0.27 | +0.14 | -0.58 | -0.68 |
| llama-instruct | physionet2019 | -2.72 | +0.28 | -1.55 | -1.61 |
| llama-instruct | diabetes130 | -1.47 | -1.03 | +1.98 | -0.56 |
| olmo3-instruct | physionet2012 | +1.48 | +1.53 | +1.85 | +1.47 |
| olmo3-instruct | physionet2019 | +1.02 | +0.79 | -0.37 | -0.58 |
| olmo3-instruct | diabetes130 | -7.01 | +0.55 | +1.34 | +1.26 |
| qwen-instruct | physionet2012 | -5.45 | -6.49 | -2.57 | -2.38 |
| qwen-instruct | physionet2019 | -4.14 | -1.95 | +0.02 | +0.47 |
| qwen-instruct | diabetes130 | +2.04 | -6.59 | -1.77 | -4.76 |
| qwen3-instruct | physionet2012 | -2.28 | -2.59 | -3.89 | -4.26 |
| qwen3-instruct | physionet2019 | -1.78 | -1.09 | -3.65 | -4.27 |
| qwen3-instruct | diabetes130 | -8.18 | -9.50 | -6.35 | -6.18 |

### Sensitivity to the top-1,000 reading (D5 bound; 2,000 resamples)

| model | dataset | records with an imputed variant | a, bounded | a', bounded | a (A/B), bounded | a' (A/B), bounded |
|---|---|---|---|---|---|---|
| granite3-instruct | physionet2012 | 4.0% | -0.58 | -0.65 | +5.56 | +5.43 |
| granite3-instruct | physionet2019 | 4.2% | -3.75 | -1.41 | -26.85 | +7.24 |
| granite3-instruct | diabetes130 | 7.5% | -11.47 | +0.21 | +3.63 | +4.99 |
| llama-instruct | physionet2012 | 9.0% | – | – | -0.14 | -0.27 |
| llama-instruct | physionet2019 | 0.2% | – | – | -1.13 | -0.97 |
| llama-instruct | diabetes130 | 0.2% | – | – | -0.20 | -0.82 |
| olmo3-dpo | physionet2012 | 1.9% | +1.27 | +0.03 | – | – |
| olmo3-dpo | physionet2019 | 0.0% | -1.85 | -1.39 | – | – |
| olmo3-dpo | diabetes130 | 0.0% | -1.50 | -0.77 | – | – |
| olmo3-instruct | physionet2012 | 13.3% | +1.56 | -0.11 | +1.73 | +1.51 |
| olmo3-instruct | physionet2019 | 11.6% | -1.87 | -1.33 | +0.31 | +0.13 |
| olmo3-instruct | diabetes130 | 11.7% | -1.08 | -0.60 | +3.21 | +0.77 |
| qwen-instruct | physionet2012 | 58.8% | – | – | -4.46 | -4.84 |
| qwen-instruct | physionet2019 | 35.5% | – | – | -2.09 | -0.53 |
| qwen-instruct | diabetes130 | 40.3% | – | – | -0.04 | -5.84 |
| qwen3-instruct | physionet2012 | 13.6% | -5.62 | -6.82 | -3.03 | -2.97 |
| qwen3-instruct | physionet2019 | 6.9% | -1.75 | -2.23 | -2.14 | -2.18 |
| qwen3-instruct | diabetes130 | 3.2% | -5.74 | -5.87 | -7.56 | -7.98 |
