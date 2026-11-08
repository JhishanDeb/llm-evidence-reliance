# Fifth study: results (`prereg-v5`)

Runs: falcon3-7b, granite3-instruct, llama-instruct, olmo3-instruct, qwen-instruct, qwen3-instruct, yi15-9b. Missing registered runs: none. Bootstrap resamples: 10000.

## Decisions

- **R15** (the placebo card weakens Qwen3-8B's preference for “No”): not supported.
- **R16** (W against the placebo card keeps the family-specific direction): not supported (3 of 4 cells).
- **R17** (the effect appears in generated decisions): not supported (1 of 4 cells).

## Validity

| cell | mass alarm | mass reversed | C0 acc. | valid | pa | tf | ab | dose | prob |
|---|---|---|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | 0.999 | 0.999 | 0.653 | yes | yes | yes | yes | yes | yes |
| falcon3-7b\|physionet2019 | 0.999 | 0.999 | 0.627 | yes | no | yes | yes | yes | yes |
| granite3-instruct\|physionet2012 | 0.990 | 0.988 | 0.715 | yes | yes | yes | yes | yes | yes |
| granite3-instruct\|physionet2019 | 0.969 | 0.959 | 0.555 | yes | yes | yes | yes | yes | yes |
| llama-instruct\|physionet2012 | 0.985 | 0.988 | 0.598 | yes | yes | yes | yes | yes | yes |
| llama-instruct\|physionet2019 | 0.993 | 0.995 | 0.601 | yes | yes | yes | yes | yes | yes |
| olmo3-instruct\|physionet2012 | 0.998 | 0.994 | 0.640 | yes | yes | yes | yes | yes | yes |
| olmo3-instruct\|physionet2019 | 0.997 | 0.997 | 0.611 | yes | yes | yes | yes | yes | yes |
| qwen-instruct\|physionet2012 | 0.999 | 1.000 | 0.688 | yes | yes | yes | yes | yes | yes |
| qwen-instruct\|physionet2019 | 0.996 | 1.000 | 0.605 | yes | yes | yes | yes | yes | yes |
| qwen3-instruct\|physionet2012 | 0.822 | 0.907 | 0.673 | yes | yes | no | no | yes | yes |
| qwen3-instruct\|physionet2019 | 0.665 | 0.571 | 0.633 | yes | no | no | yes | yes | yes |
| yi15-9b\|physionet2012 | 0.999 | 0.998 | 0.630 | yes | yes | yes | yes | yes | yes |
| yi15-9b\|physionet2019 | 0.998 | 0.998 | 0.598 | yes | yes | yes | yes | yes | yes |

## R15. The placebo card and the answer prior (Qwen3-8B)

| cell | valid | λ(C0) | λ(CP) | Δλ | 95% interval | p | Holm p | supported |
|---|---|---|---|---|---|---|---|---|
| qwen3-instruct\|physionet2012 | yes | -7.16 | -7.30 | -0.14 | [-0.32, -0.02] | 0.032 | 0.032 | no |
| qwen3-instruct\|physionet2019 | yes | -5.31 | -6.98 | -1.67 | [-1.98, -1.28] | <0.001 | <0.001 | no |

**R15: not supported** (prediction: Δλ > 0 in both cells).

## R16. W against the placebo card, at its neutral point

| cell | predicted | valid | W (a) | W (a′) | p | Holm p | supported | opposite |
|---|---|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | + | yes | +2.24 [+2.05, +2.45] | +2.16 [+2.04, +2.29] | <0.001 | <0.001 | yes | no |
| falcon3-7b\|physionet2019 | + | yes | +7.75 [+3.81, +11.30] | +4.24 [+4.08, +4.39] | 0.001 | 0.001 | yes | no |
| yi15-9b\|physionet2019 | + | yes | +1.61 [+1.01, +2.25] | +1.13 [+1.04, +1.22] | <0.001 | <0.001 | yes | no |
| granite3-instruct\|physionet2012 | − | yes | +0.36 [+0.25, +0.48] | +0.39 [+0.28, +0.50] | <0.001 | <0.001 | no | yes |

**R16: not supported** (3 of 4 cells supported; needs at least 3 and none opposite).

## R17. Generated decisions

| cell | predicted | testable | patients | E_W | 95% interval | supported |
|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | + | yes | 200 | +0.115 | [+0.049, +0.180] | yes |
| falcon3-7b\|physionet2019 | + | yes | 200 | -0.035 | [-0.113, +0.043] | no |
| yi15-9b\|physionet2019 | + | yes | 200 | +0.015 | [-0.055, +0.085] | no |
| granite3-instruct\|physionet2012 | − | yes | 156 | +0.176 | [+0.076, +0.279] | no |

**R17: not supported** (1 of 4 cells; needs at least 3).

## Q18 and Q21c. O and W of every pair, against no card (C0) and against the placebo card (CP)

W < 0 is a pull toward the second answer word (“No”, “False”, “B”). p is the intersection–union p over patients (no correction). near: patients within 1 log-odds of the reading point in both framings; δ*: the share by which a card would have to change the prior to produce W; readable: conclusive, at least 30 near, δ* ≥ 0.5.

| cell | pair | baseline | reading | n | λ | W (a) | W (a′) | p | W | O (a) | O (a′) | p | near | δ* | readable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | yn | C0 | threshold | 600 | -0.08 | +0.73 | +0.64 | <0.001 | toward “Yes” | +0.61 | +0.20 | 0.016 | 91 | +4.17 | yes |
| falcon3-7b\|physionet2012 | yn | C0 | neutral | 600 | -0.08 | +0.76 | +0.68 | <0.001 | toward “Yes” | +0.59 | +0.19 | 0.028 | 91 | +4.48 | yes |
| falcon3-7b\|physionet2012 | yn | CP | threshold | 600 | -0.89 | +1.91 | +1.65 | <0.001 | toward “Yes” | +2.36 | +1.75 | <0.001 | 83 | +0.92 | yes |
| falcon3-7b\|physionet2012 | yn | CP | neutral | 600 | -0.89 | +2.24 | +2.16 | <0.001 | toward “Yes” | +2.28 | +1.71 | <0.001 | 73 | +1.21 | yes |
| falcon3-7b\|physionet2012 | pa | C0 | threshold | 600 | -0.34 | +0.67 | +0.54 | <0.001 | toward “Yes” | +0.67 | +0.24 | 0.001 | 82 | +0.79 | yes |
| falcon3-7b\|physionet2012 | pa | C0 | neutral | 600 | -0.34 | +0.82 | +0.79 | <0.001 | toward “Yes” | +0.63 | +0.24 | 0.002 | 72 | +1.15 | yes |
| falcon3-7b\|physionet2012 | pa | CP | threshold | 600 | -0.81 | +1.57 | +1.23 | <0.001 | toward “Yes” | +2.03 | +1.43 | <0.001 | 83 | +0.76 | yes |
| falcon3-7b\|physionet2012 | pa | CP | neutral | 600 | -0.81 | +1.84 | +1.71 | <0.001 | toward “Yes” | +1.88 | +1.37 | <0.001 | 80 | +1.06 | yes |
| falcon3-7b\|physionet2012 | tf | C0 | threshold | 600 | -0.68 | -1.48 | -1.55 | <0.001 | toward “No” | +0.35 | +0.24 | 0.001 | 79 | +1.09 | yes |
| falcon3-7b\|physionet2012 | tf | C0 | neutral | 600 | -0.68 | -1.14 | -1.12 | <0.001 | toward “No” | +0.24 | +0.10 | 0.167 | 85 | +0.83 | yes |
| falcon3-7b\|physionet2012 | tf | CP | threshold | 600 | -1.28 | +0.43 | +0.10 | 0.107 | not conclusive | +2.30 | +1.95 | <0.001 | 65 | +0.04 | no |
| falcon3-7b\|physionet2012 | tf | CP | neutral | 600 | -1.28 | +0.65 | +0.48 | <0.001 | toward “Yes” | +2.29 | +1.98 | <0.001 | 78 | +0.19 | no |
| falcon3-7b\|physionet2012 | ab | C0 | threshold | 600 | -0.68 | -0.95 | -1.06 | <0.001 | toward “No” | -0.32 | -0.51 | 0.001 | 47 | +0.70 | yes |
| falcon3-7b\|physionet2012 | ab | C0 | neutral | 600 | -0.68 | -0.70 | -0.75 | <0.001 | toward “No” | -0.40 | -0.64 | <0.001 | 56 | +0.52 | yes |
| falcon3-7b\|physionet2012 | ab | CP | threshold | 600 | -1.38 | +0.37 | +0.53 | <0.001 | toward “Yes” | +1.77 | +1.39 | <0.001 | 28 | +0.14 | no |
| falcon3-7b\|physionet2012 | ab | CP | neutral | 600 | -1.38 | +0.59 | +0.41 | <0.001 | toward “Yes” | +1.74 | +1.49 | <0.001 | 26 | +0.15 | no |
| falcon3-7b\|physionet2019 | yn | C0 | threshold | 542 | -0.04 | +1.41 | +1.29 | <0.001 | toward “Yes” | +1.77 | +1.44 | <0.001 | 132 | +16.59 | yes |
| falcon3-7b\|physionet2019 | yn | C0 | neutral | 542 | -0.04 | +1.48 | +1.35 | <0.001 | toward “Yes” | +1.77 | +1.44 | <0.001 | 132 | +17.31 | yes |
| falcon3-7b\|physionet2019 | yn | CP | threshold | 542 | -1.57 | +0.39 | +2.60 | 0.348 | not conclusive | +0.45 | +2.73 | 0.335 | 77 | +0.12 | no |
| falcon3-7b\|physionet2019 | yn | CP | neutral | 542 | -1.57 | +7.75 | +4.24 | 0.001 | toward “Yes” | -0.94 | +2.79 | 1.000 | 125 | +1.35 | yes |
| falcon3-7b\|physionet2019 | tf | C0 | threshold | 542 | -0.74 | -1.39 | -1.59 | <0.001 | toward “No” | +1.18 | +1.17 | <0.001 | 136 | +0.93 | yes |
| falcon3-7b\|physionet2019 | tf | C0 | neutral | 542 | -0.74 | -0.52 | -0.60 | <0.001 | toward “No” | +1.54 | +1.19 | <0.001 | 156 | +0.35 | no |
| falcon3-7b\|physionet2019 | tf | CP | threshold | 542 | -1.27 | -8.75 | -0.52 | <0.001 | toward “No” | -4.96 | +3.33 | 1.000 | 41 | +0.21 | no |
| falcon3-7b\|physionet2019 | tf | CP | neutral | 542 | -1.27 | -0.34 | +0.81 | 1.000 | not conclusive | -0.74 | +3.27 | 1.000 | 76 | – | no |
| falcon3-7b\|physionet2019 | ab | C0 | threshold | 542 | -1.33 | -1.07 | -0.78 | <0.001 | toward “No” | +0.79 | +0.97 | 0.002 | 75 | +0.29 | no |
| falcon3-7b\|physionet2019 | ab | C0 | neutral | 542 | -1.33 | +1.08 | +0.95 | <0.001 | toward “Yes” | +1.32 | +1.00 | <0.001 | 83 | +0.36 | no |
| falcon3-7b\|physionet2019 | ab | CP | threshold | 542 | -1.82 | -6.05 | +0.00 | 1.000 | not conclusive | -2.95 | +2.94 | 1.000 | 25 | – | no |
| falcon3-7b\|physionet2019 | ab | CP | neutral | 542 | -1.82 | +2.37 | +1.79 | 0.004 | toward “Yes” | -0.99 | +3.12 | 1.000 | 31 | +0.49 | no |
| granite3-instruct\|physionet2012 | yn | C0 | threshold | 600 | -0.33 | -3.60 | -3.65 | <0.001 | toward “No” | +3.10 | +2.97 | <0.001 | 66 | +5.49 | yes |
| granite3-instruct\|physionet2012 | yn | C0 | neutral | 600 | -0.33 | -3.33 | -3.28 | <0.001 | toward “No” | +3.11 | +2.95 | <0.001 | 69 | +4.99 | yes |
| granite3-instruct\|physionet2012 | yn | CP | threshold | 600 | -2.06 | -0.90 | -1.37 | <0.001 | toward “No” | +3.33 | +3.38 | <0.001 | 70 | +0.22 | no |
| granite3-instruct\|physionet2012 | yn | CP | neutral | 600 | -2.06 | +0.36 | +0.39 | <0.001 | toward “Yes” | +3.41 | +3.15 | <0.001 | 79 | +0.09 | no |
| granite3-instruct\|physionet2012 | pa | C0 | threshold | 600 | -0.58 | -3.56 | -3.46 | <0.001 | toward “No” | +2.71 | +2.70 | <0.001 | 74 | +2.99 | yes |
| granite3-instruct\|physionet2012 | pa | C0 | neutral | 600 | -0.58 | -3.02 | -2.90 | <0.001 | toward “No” | +2.61 | +2.75 | <0.001 | 73 | +2.51 | yes |
| granite3-instruct\|physionet2012 | pa | CP | threshold | 600 | -1.55 | -1.97 | -2.32 | <0.001 | toward “No” | +3.07 | +2.97 | <0.001 | 90 | +0.64 | yes |
| granite3-instruct\|physionet2012 | pa | CP | neutral | 600 | -1.55 | -0.97 | -0.86 | <0.001 | toward “No” | +2.86 | +2.99 | <0.001 | 74 | +0.28 | no |
| granite3-instruct\|physionet2012 | tf | C0 | threshold | 600 | -3.98 | -6.04 | -7.96 | <0.001 | toward “No” | +1.31 | +2.69 | <0.001 | 73 | +0.76 | yes |
| granite3-instruct\|physionet2012 | tf | C0 | neutral | 600 | -3.98 | -2.45 | -2.77 | <0.001 | toward “No” | +2.42 | +2.62 | <0.001 | 106 | +0.31 | no |
| granite3-instruct\|physionet2012 | tf | CP | threshold | 600 | -6.84 | +18.34 | +0.88 | 0.128 | not conclusive | -7.24 | +1.74 | 1.000 | 0 | +0.06 | no |
| granite3-instruct\|physionet2012 | tf | CP | neutral | 600 | -6.84 | +2.99 | +2.98 | <0.001 | toward “Yes” | +2.95 | +3.01 | <0.001 | 276 | +0.22 | no |
| granite3-instruct\|physionet2012 | ab | C0 | threshold | 600 | -0.00 | +3.00 | +2.90 | <0.001 | toward “Yes” | +5.20 | +5.21 | <0.001 | 30 | +2317141.20 | yes |
| granite3-instruct\|physionet2012 | ab | C0 | neutral | 600 | -0.00 | +3.00 | +2.90 | <0.001 | toward “Yes” | +5.20 | +5.21 | <0.001 | 31 | +2317142.02 | yes |
| granite3-instruct\|physionet2012 | ab | CP | threshold | 600 | -0.13 | +3.16 | +2.94 | <0.001 | toward “Yes” | +5.31 | +4.43 | <0.001 | 66 | +11.08 | yes |
| granite3-instruct\|physionet2012 | ab | CP | neutral | 600 | -0.13 | +3.25 | +3.11 | <0.001 | toward “Yes” | +5.31 | +4.41 | <0.001 | 69 | +11.72 | yes |
| granite3-instruct\|physionet2019 | yn | C0 | threshold | 542 | -0.74 | +1.22 | -5.15 | 1.000 | not conclusive | -6.74 | +3.74 | 1.000 | 46 | – | no |
| granite3-instruct\|physionet2019 | yn | C0 | neutral | 542 | -0.74 | -2.30 | -4.10 | 0.038 | toward “No” | -5.27 | +3.93 | 1.000 | 104 | +1.55 | yes |
| granite3-instruct\|physionet2019 | yn | CP | threshold | 542 | -2.79 | -12.18 | -5.24 | <0.001 | toward “No” | +0.48 | +7.31 | 0.785 | 25 | +0.94 | no |
| granite3-instruct\|physionet2019 | yn | CP | neutral | 542 | -2.79 | -0.55 | -1.21 | 0.245 | not conclusive | +2.87 | +7.74 | 0.100 | 24 | +0.10 | no |
| granite3-instruct\|physionet2019 | pa | C0 | threshold | 542 | -1.73 | +25.62 | -5.43 | 1.000 | not conclusive | -29.37 | +2.50 | 1.000 | 0 | – | no |
| granite3-instruct\|physionet2019 | pa | C0 | neutral | 542 | -1.73 | +13.73 | -2.52 | 1.000 | not conclusive | -16.25 | +2.58 | 1.000 | 5 | – | no |
| granite3-instruct\|physionet2019 | pa | CP | threshold | 542 | -2.55 | -7.78 | -6.36 | <0.001 | toward “No” | +4.42 | +5.83 | <0.001 | 31 | +1.25 | yes |
| granite3-instruct\|physionet2019 | pa | CP | neutral | 542 | -2.55 | -2.69 | -2.53 | <0.001 | toward “No” | +5.77 | +6.25 | <0.001 | 61 | +0.50 | no |
| granite3-instruct\|physionet2019 | tf | C0 | threshold | 542 | -2.14 | +24.83 | -7.28 | 1.000 | not conclusive | -30.37 | +1.66 | 1.000 | 1 | – | no |
| granite3-instruct\|physionet2019 | tf | C0 | neutral | 542 | -2.14 | +5.72 | -3.99 | 1.000 | not conclusive | -3.95 | +2.30 | 1.000 | 157 | – | no |
| granite3-instruct\|physionet2019 | tf | CP | threshold | 542 | -4.92 | +68.87 | -6.28 | 1.000 | not conclusive | +65.76 | +3.56 | <0.001 | 1 | – | no |
| granite3-instruct\|physionet2019 | tf | CP | neutral | 542 | -4.92 | +4.99 | +0.78 | <0.001 | toward “Yes” | +6.95 | +4.22 | <0.001 | 98 | +0.08 | no |
| granite3-instruct\|physionet2019 | ab | C0 | threshold | 542 | +1.02 | -20.59 | -0.14 | 0.745 | not conclusive | -40.40 | +7.29 | 1.000 | 2 | +0.07 | no |
| granite3-instruct\|physionet2019 | ab | C0 | neutral | 542 | +1.02 | -4.94 | -2.04 | 0.160 | not conclusive | -38.56 | +7.26 | 1.000 | 5 | +1.01 | no |
| granite3-instruct\|physionet2019 | ab | CP | threshold | 542 | -1.56 | +4.66 | +1.77 | <0.001 | toward “Yes” | +4.72 | +9.24 | <0.001 | 83 | +0.57 | yes |
| granite3-instruct\|physionet2019 | ab | CP | neutral | 542 | -1.56 | +3.94 | +3.85 | <0.001 | toward “Yes” | +9.05 | +9.19 | <0.001 | 176 | +1.23 | yes |
| llama-instruct\|physionet2012 | yn | C0 | threshold | 600 | -1.81 | -0.20 | -0.40 | 0.019 | toward “No” | -0.19 | -0.32 | 0.018 | 79 | +0.06 | no |
| llama-instruct\|physionet2012 | yn | C0 | neutral | 600 | -1.81 | +0.00 | +0.20 | 0.943 | not conclusive | +0.51 | +0.14 | 0.022 | 103 | +0.00 | no |
| llama-instruct\|physionet2012 | yn | CP | threshold | 600 | -1.91 | +1.93 | +1.23 | <0.001 | toward “Yes” | -0.97 | -0.90 | <0.001 | 99 | +0.32 | no |
| llama-instruct\|physionet2012 | yn | CP | neutral | 600 | -1.91 | +0.86 | +0.97 | <0.001 | toward “Yes” | -0.83 | -1.14 | <0.001 | 183 | +0.23 | no |
| llama-instruct\|physionet2012 | pa | C0 | threshold | 600 | -1.62 | -0.87 | -0.88 | <0.001 | toward “No” | +1.39 | +1.27 | <0.001 | 64 | +0.27 | no |
| llama-instruct\|physionet2012 | pa | C0 | neutral | 600 | -1.62 | +0.27 | +0.33 | <0.001 | toward “Yes” | +1.52 | +1.37 | <0.001 | 60 | +0.08 | no |
| llama-instruct\|physionet2012 | pa | CP | threshold | 600 | -1.91 | +2.12 | +1.22 | <0.001 | toward “Yes” | +1.31 | +0.94 | <0.001 | 99 | +0.32 | no |
| llama-instruct\|physionet2012 | pa | CP | neutral | 600 | -1.91 | +0.81 | +0.97 | <0.001 | toward “Yes” | +1.55 | +0.98 | <0.001 | 151 | +0.21 | no |
| llama-instruct\|physionet2012 | tf | C0 | threshold | 600 | -1.79 | -1.60 | -1.80 | <0.001 | toward “No” | +1.23 | +1.21 | <0.001 | 81 | +0.45 | no |
| llama-instruct\|physionet2012 | tf | C0 | neutral | 600 | -1.79 | -0.27 | -0.30 | <0.001 | toward “No” | +1.46 | +1.30 | <0.001 | 113 | +0.08 | no |
| llama-instruct\|physionet2012 | tf | CP | threshold | 600 | -1.77 | +4.07 | +0.63 | <0.001 | toward “Yes” | -2.36 | +0.62 | 1.000 | 61 | +0.18 | no |
| llama-instruct\|physionet2012 | tf | CP | neutral | 600 | -1.77 | -0.08 | -0.16 | 0.014 | toward “No” | -0.81 | -0.54 | <0.001 | 427 | +0.02 | no |
| llama-instruct\|physionet2012 | ab | C0 | threshold | 600 | -0.13 | +0.39 | +0.38 | <0.001 | toward “Yes” | -0.23 | -0.30 | 0.004 | 141 | +1.52 | yes |
| llama-instruct\|physionet2012 | ab | C0 | neutral | 600 | -0.13 | +0.47 | +0.47 | <0.001 | toward “Yes” | -0.22 | -0.29 | 0.005 | 141 | +1.88 | yes |
| llama-instruct\|physionet2012 | ab | CP | threshold | 600 | -0.09 | +0.34 | +0.39 | <0.001 | toward “Yes” | +0.15 | -0.24 | 1.000 | 238 | +1.96 | yes |
| llama-instruct\|physionet2012 | ab | CP | neutral | 600 | -0.09 | +0.28 | +0.36 | <0.001 | toward “Yes” | +0.17 | -0.22 | 1.000 | 246 | +1.60 | yes |
| llama-instruct\|physionet2019 | yn | C0 | threshold | 542 | -0.81 | +0.09 | -0.08 | 1.000 | not conclusive | -0.73 | -0.58 | 0.006 | 203 | – | no |
| llama-instruct\|physionet2019 | yn | C0 | neutral | 542 | -0.81 | -0.33 | +0.27 | 1.000 | not conclusive | -0.78 | -0.29 | <0.001 | 340 | – | no |
| llama-instruct\|physionet2019 | yn | CP | threshold | 542 | -1.22 | -3.05 | +0.05 | 1.000 | not conclusive | -3.24 | +0.31 | 1.000 | 225 | – | no |
| llama-instruct\|physionet2019 | yn | CP | neutral | 542 | -1.22 | +0.72 | +0.84 | 0.006 | toward “Yes” | +0.72 | +0.81 | 0.006 | 313 | +0.30 | no |
| llama-instruct\|physionet2019 | pa | C0 | threshold | 542 | -1.03 | +0.35 | -0.73 | 1.000 | not conclusive | -0.20 | +0.90 | 1.000 | 179 | – | no |
| llama-instruct\|physionet2019 | pa | C0 | neutral | 542 | -1.03 | -0.24 | -0.07 | 0.149 | not conclusive | +1.07 | +1.37 | 0.006 | 352 | +0.04 | no |
| llama-instruct\|physionet2019 | pa | CP | threshold | 542 | -1.37 | -0.48 | -0.60 | 0.842 | not conclusive | +0.78 | +1.41 | 0.525 | 156 | +0.18 | no |
| llama-instruct\|physionet2019 | pa | CP | neutral | 542 | -1.37 | -1.05 | +0.46 | 1.000 | not conclusive | +0.26 | +1.51 | 0.409 | 305 | – | no |
| llama-instruct\|physionet2019 | tf | C0 | threshold | 542 | -1.74 | -0.16 | -1.68 | 0.751 | not conclusive | -1.61 | +0.16 | 1.000 | 4 | +0.05 | no |
| llama-instruct\|physionet2019 | tf | C0 | neutral | 542 | -1.74 | +0.11 | +0.18 | 0.459 | not conclusive | +0.16 | +0.48 | 0.441 | 270 | +0.03 | no |
| llama-instruct\|physionet2019 | tf | CP | threshold | 542 | -2.05 | +4.65 | -0.99 | 1.000 | not conclusive | +5.19 | +0.30 | 0.618 | 4 | – | no |
| llama-instruct\|physionet2019 | tf | CP | neutral | 542 | -2.05 | +0.40 | +0.59 | 0.579 | not conclusive | +0.68 | +0.88 | 0.385 | 399 | +0.10 | no |
| llama-instruct\|physionet2019 | ab | C0 | threshold | 542 | +1.00 | -0.77 | +1.00 | 1.000 | not conclusive | -3.48 | -0.62 | <0.001 | 243 | – | no |
| llama-instruct\|physionet2019 | ab | C0 | neutral | 542 | +1.00 | +0.40 | +0.46 | <0.001 | toward “Yes” | -1.06 | -0.92 | <0.001 | 402 | +0.20 | no |
| llama-instruct\|physionet2019 | ab | CP | threshold | 542 | +0.62 | +6.28 | +1.06 | 0.005 | toward “Yes” | -3.59 | +1.36 | 1.000 | 67 | +0.85 | yes |
| llama-instruct\|physionet2019 | ab | CP | neutral | 542 | +0.62 | +2.58 | +0.59 | 0.047 | toward “Yes” | -1.34 | +1.13 | 1.000 | 145 | +0.47 | no |
| olmo3-instruct\|physionet2012 | yn | C0 | threshold | 600 | -1.84 | -0.70 | -1.56 | <0.001 | toward “No” | +2.40 | +1.47 | <0.001 | 58 | +0.19 | no |
| olmo3-instruct\|physionet2012 | yn | C0 | neutral | 600 | -1.84 | +0.12 | -0.14 | 1.000 | not conclusive | +2.50 | +1.96 | <0.001 | 58 | – | no |
| olmo3-instruct\|physionet2012 | yn | CP | threshold | 600 | -3.28 | +1.04 | +1.19 | 0.045 | toward “Yes” | +2.82 | +3.23 | <0.001 | 17 | +0.16 | no |
| olmo3-instruct\|physionet2012 | yn | CP | neutral | 600 | -3.28 | +2.35 | +1.90 | <0.001 | toward “Yes” | +6.36 | +5.06 | <0.001 | 36 | +0.29 | no |
| olmo3-instruct\|physionet2012 | pa | C0 | threshold | 600 | -2.12 | -0.54 | -1.20 | <0.001 | toward “No” | +3.15 | +2.75 | <0.001 | 62 | +0.13 | no |
| olmo3-instruct\|physionet2012 | pa | C0 | neutral | 600 | -2.12 | +0.33 | +0.23 | <0.001 | toward “Yes” | +3.30 | +3.03 | <0.001 | 66 | +0.05 | no |
| olmo3-instruct\|physionet2012 | pa | CP | threshold | 600 | -2.77 | +1.23 | +0.39 | <0.001 | toward “Yes” | +4.04 | +3.95 | <0.001 | 52 | +0.07 | no |
| olmo3-instruct\|physionet2012 | pa | CP | neutral | 600 | -2.77 | +1.44 | +1.36 | <0.001 | toward “Yes” | +5.00 | +4.45 | <0.001 | 67 | +0.25 | no |
| olmo3-instruct\|physionet2012 | tf | C0 | threshold | 600 | +0.29 | -3.11 | -3.15 | <0.001 | toward “No” | +3.00 | +2.65 | <0.001 | 71 | +5.38 | yes |
| olmo3-instruct\|physionet2012 | tf | C0 | neutral | 600 | +0.29 | -3.30 | -3.43 | <0.001 | toward “No” | +3.01 | +2.62 | <0.001 | 76 | +5.71 | yes |
| olmo3-instruct\|physionet2012 | tf | CP | threshold | 600 | -2.23 | -2.51 | -0.19 | 0.350 | not conclusive | +1.41 | +3.73 | 0.061 | 2 | +0.04 | no |
| olmo3-instruct\|physionet2012 | tf | CP | neutral | 600 | -2.23 | +1.08 | +1.07 | 0.001 | toward “Yes” | +5.45 | +4.63 | <0.001 | 86 | +0.24 | no |
| olmo3-instruct\|physionet2012 | ab | C0 | threshold | 600 | +0.56 | -0.18 | +0.08 | 1.000 | not conclusive | +1.65 | +1.51 | <0.001 | 116 | – | no |
| olmo3-instruct\|physionet2012 | ab | C0 | neutral | 600 | +0.56 | -0.49 | -0.47 | <0.001 | toward “No” | +1.69 | +1.55 | <0.001 | 123 | +0.42 | no |
| olmo3-instruct\|physionet2012 | ab | CP | threshold | 600 | -0.70 | +2.10 | +1.78 | <0.001 | toward “Yes” | +6.57 | +4.85 | <0.001 | 55 | +1.28 | yes |
| olmo3-instruct\|physionet2012 | ab | CP | neutral | 600 | -0.70 | +2.33 | +2.29 | <0.001 | toward “Yes” | +6.50 | +4.79 | <0.001 | 62 | +1.65 | yes |
| olmo3-instruct\|physionet2019 | yn | C0 | threshold | 542 | -1.95 | -1.35 | -1.50 | 0.001 | toward “No” | -1.10 | +0.13 | 1.000 | 137 | +0.35 | no |
| olmo3-instruct\|physionet2019 | yn | C0 | neutral | 542 | -1.95 | -0.06 | -0.10 | 0.120 | not conclusive | +0.63 | +0.43 | <0.001 | 149 | +0.01 | no |
| olmo3-instruct\|physionet2019 | yn | CP | threshold | 542 | -3.41 | -30.41 | -1.07 | 0.048 | toward “No” | -23.64 | +5.70 | 1.000 | 0 | +0.16 | no |
| olmo3-instruct\|physionet2019 | yn | CP | neutral | 542 | -3.41 | -4.82 | +0.78 | 1.000 | not conclusive | -2.05 | +7.49 | 1.000 | 7 | – | no |
| olmo3-instruct\|physionet2019 | pa | C0 | threshold | 542 | -2.05 | -0.21 | -1.06 | 0.364 | not conclusive | -0.93 | +0.36 | 1.000 | 140 | +0.05 | no |
| olmo3-instruct\|physionet2019 | pa | C0 | neutral | 542 | -2.05 | +0.23 | +0.22 | <0.001 | toward “Yes” | +0.63 | +0.48 | <0.001 | 159 | +0.05 | no |
| olmo3-instruct\|physionet2019 | pa | CP | threshold | 542 | -3.37 | -15.04 | -1.02 | 0.016 | toward “No” | -8.35 | +5.88 | 1.000 | 4 | +0.15 | no |
| olmo3-instruct\|physionet2019 | pa | CP | neutral | 542 | -3.37 | +2.20 | +1.53 | 0.156 | not conclusive | -0.34 | +7.19 | 1.000 | 25 | +0.23 | no |
| olmo3-instruct\|physionet2019 | tf | C0 | threshold | 542 | -1.69 | +2.90 | -1.12 | 1.000 | not conclusive | -4.55 | -0.61 | <0.001 | 11 | – | no |
| olmo3-instruct\|physionet2019 | tf | C0 | neutral | 542 | -1.69 | +0.81 | +0.10 | 0.185 | not conclusive | -1.75 | -0.02 | 0.809 | 117 | +0.03 | no |
| olmo3-instruct\|physionet2019 | tf | CP | threshold | 542 | -1.78 | -5.12 | -2.33 | 0.708 | not conclusive | +2.89 | +5.51 | 0.958 | 0 | +0.65 | no |
| olmo3-instruct\|physionet2019 | tf | CP | neutral | 542 | -1.78 | +0.70 | -1.32 | 1.000 | not conclusive | +4.01 | +6.14 | 0.625 | 15 | – | no |
| olmo3-instruct\|physionet2019 | ab | C0 | threshold | 542 | +0.24 | +0.58 | +0.72 | <0.001 | toward “Yes” | +0.12 | +0.18 | 0.326 | 251 | +1.20 | yes |
| olmo3-instruct\|physionet2019 | ab | C0 | neutral | 542 | +0.24 | +0.44 | +0.50 | <0.001 | toward “Yes” | +0.24 | +0.20 | 0.064 | 258 | +0.91 | yes |
| olmo3-instruct\|physionet2019 | ab | CP | threshold | 542 | +1.02 | +21.89 | +0.09 | 0.342 | not conclusive | -23.69 | +3.13 | 1.000 | 26 | +0.04 | no |
| olmo3-instruct\|physionet2019 | ab | CP | neutral | 542 | +1.02 | +2.48 | -0.75 | 1.000 | not conclusive | -16.45 | +3.36 | 1.000 | 35 | – | no |
| qwen-instruct\|physionet2012 | yn | C0 | threshold | 600 | -1.93 | +1.95 | +1.95 | <0.001 | toward “Yes” | -2.89 | -4.20 | <0.001 | 26 | +0.50 | no |
| qwen-instruct\|physionet2012 | yn | C0 | neutral | 600 | -1.93 | +2.68 | +3.19 | <0.001 | toward “Yes” | -3.06 | -4.38 | <0.001 | 15 | +0.70 | no |
| qwen-instruct\|physionet2012 | yn | CP | threshold | 600 | -2.50 | +1.18 | +0.93 | 0.016 | toward “Yes” | +3.94 | +2.68 | <0.001 | 11 | +0.19 | no |
| qwen-instruct\|physionet2012 | yn | CP | neutral | 600 | -2.50 | +2.46 | +2.46 | <0.001 | toward “Yes” | +3.75 | +2.58 | <0.001 | 14 | +0.49 | no |
| qwen-instruct\|physionet2012 | pa | C0 | threshold | 600 | -2.12 | -0.33 | -0.47 | 0.062 | not conclusive | -3.48 | -4.08 | <0.001 | 16 | +0.08 | no |
| qwen-instruct\|physionet2012 | pa | C0 | neutral | 600 | -2.12 | +0.90 | +1.19 | <0.001 | toward “Yes” | -3.63 | -3.99 | <0.001 | 17 | +0.21 | no |
| qwen-instruct\|physionet2012 | pa | CP | threshold | 600 | -3.64 | +1.96 | +1.13 | <0.001 | toward “Yes” | +3.94 | +1.95 | <0.001 | 21 | +0.15 | no |
| qwen-instruct\|physionet2012 | pa | CP | neutral | 600 | -3.64 | +3.97 | +3.41 | <0.001 | toward “Yes” | +3.98 | +2.52 | <0.001 | 14 | +0.47 | no |
| qwen-instruct\|physionet2012 | tf | C0 | threshold | 600 | -0.34 | -1.39 | -1.38 | <0.001 | toward “No” | -5.27 | -6.34 | <0.001 | 23 | +2.06 | no |
| qwen-instruct\|physionet2012 | tf | C0 | neutral | 600 | -0.34 | -1.15 | -1.13 | <0.001 | toward “No” | -5.31 | -6.33 | <0.001 | 23 | +1.69 | no |
| qwen-instruct\|physionet2012 | tf | CP | threshold | 600 | -1.10 | -0.28 | -0.86 | 0.730 | not conclusive | +4.88 | +2.99 | <0.001 | 8 | +0.13 | no |
| qwen-instruct\|physionet2012 | tf | CP | neutral | 600 | -1.10 | +0.52 | -0.18 | 1.000 | not conclusive | +4.98 | +3.10 | <0.001 | 6 | – | no |
| qwen-instruct\|physionet2012 | ab | C0 | threshold | 600 | -2.16 | -1.27 | -1.99 | <0.001 | toward “No” | -2.93 | -4.18 | <0.001 | 12 | +0.29 | no |
| qwen-instruct\|physionet2012 | ab | C0 | neutral | 600 | -2.16 | +0.56 | -0.01 | 1.000 | not conclusive | -3.05 | -4.31 | <0.001 | 15 | – | no |
| qwen-instruct\|physionet2012 | ab | CP | threshold | 600 | -1.83 | -0.34 | -1.69 | 0.761 | not conclusive | +2.46 | +1.80 | 0.207 | 5 | +0.09 | no |
| qwen-instruct\|physionet2012 | ab | CP | neutral | 600 | -1.83 | +2.39 | -0.14 | 1.000 | not conclusive | +2.26 | +1.58 | 0.241 | 5 | – | no |
| qwen-instruct\|physionet2019 | yn | C0 | threshold | 542 | -2.85 | -2.53 | -3.14 | 0.001 | toward “No” | +0.02 | +0.62 | 0.797 | 23 | +0.44 | no |
| qwen-instruct\|physionet2019 | yn | C0 | neutral | 542 | -2.85 | -0.47 | -0.70 | 0.044 | toward “No” | +0.81 | +0.39 | 0.173 | 25 | +0.08 | no |
| qwen-instruct\|physionet2019 | yn | CP | threshold | 542 | -5.00 | -7.82 | -0.92 | 0.033 | toward “No” | +3.43 | +10.11 | 0.388 | 22 | +0.09 | no |
| qwen-instruct\|physionet2019 | yn | CP | neutral | 542 | -5.00 | +2.50 | +3.32 | 0.134 | not conclusive | +8.46 | +10.70 | 0.011 | 14 | +0.25 | no |
| qwen-instruct\|physionet2019 | pa | C0 | threshold | 542 | -3.87 | -3.41 | -3.94 | <0.001 | toward “No” | +1.25 | +1.74 | 0.033 | 29 | +0.44 | no |
| qwen-instruct\|physionet2019 | pa | C0 | neutral | 542 | -3.87 | -0.08 | -0.18 | 0.730 | not conclusive | +2.34 | +1.80 | <0.001 | 36 | +0.01 | no |
| qwen-instruct\|physionet2019 | pa | CP | threshold | 542 | -6.66 | -5.96 | -0.81 | 0.017 | toward “No” | +3.00 | +8.16 | 0.262 | 23 | +0.06 | no |
| qwen-instruct\|physionet2019 | pa | CP | neutral | 542 | -6.66 | +4.97 | +5.30 | <0.001 | toward “Yes” | +7.95 | +8.47 | <0.001 | 25 | +0.37 | no |
| qwen-instruct\|physionet2019 | tf | C0 | threshold | 542 | -2.13 | -6.92 | -6.12 | <0.001 | toward “No” | -1.99 | -1.40 | <0.001 | 35 | +1.44 | yes |
| qwen-instruct\|physionet2019 | tf | C0 | neutral | 542 | -2.13 | -4.36 | -4.23 | <0.001 | toward “No” | -1.16 | -1.28 | <0.001 | 37 | +1.00 | yes |
| qwen-instruct\|physionet2019 | tf | CP | threshold | 542 | -1.54 | -7.68 | -5.49 | 0.429 | not conclusive | +2.48 | +6.08 | 0.543 | 8 | +1.78 | no |
| qwen-instruct\|physionet2019 | tf | CP | neutral | 542 | -1.54 | -4.41 | -4.42 | 0.461 | not conclusive | +2.84 | +6.03 | 0.392 | 14 | +1.43 | no |
| qwen-instruct\|physionet2019 | ab | C0 | threshold | 542 | -1.58 | -2.66 | -1.19 | 0.001 | toward “No” | -1.94 | -0.20 | 0.349 | 22 | +0.38 | no |
| qwen-instruct\|physionet2019 | ab | C0 | neutral | 542 | -1.58 | +0.14 | -0.01 | 1.000 | not conclusive | -1.86 | +0.00 | 1.000 | 22 | – | no |
| qwen-instruct\|physionet2019 | ab | CP | threshold | 542 | -0.93 | -4.32 | -1.76 | 0.012 | toward “No” | -9.88 | +9.00 | 1.000 | 6 | +0.95 | no |
| qwen-instruct\|physionet2019 | ab | CP | neutral | 542 | -0.93 | -0.37 | -1.05 | 0.622 | not conclusive | -9.96 | +8.99 | 1.000 | 7 | +0.20 | no |
| qwen3-instruct\|physionet2012 | yn | C0 | threshold | 600 | -7.16 | -1.39 | -3.46 | 0.001 | toward “No” | -3.02 | -3.04 | <0.001 | 36 | +0.10 | no |
| qwen3-instruct\|physionet2012 | yn | C0 | neutral | 600 | -7.16 | +1.43 | +1.41 | <0.001 | toward “Yes” | -2.77 | -2.51 | <0.001 | 23 | +0.10 | no |
| qwen3-instruct\|physionet2012 | yn | CP | threshold | 600 | -7.30 | -1.38 | -1.40 | 0.003 | toward “No” | +10.50 | +10.47 | <0.001 | 15 | +0.09 | no |
| qwen3-instruct\|physionet2012 | yn | CP | neutral | 600 | -7.30 | +2.35 | +2.22 | <0.001 | toward “Yes” | +11.76 | +11.90 | <0.001 | 19 | +0.15 | no |
| qwen3-instruct\|physionet2012 | pa | C0 | threshold | 600 | -5.70 | -0.32 | -1.28 | 0.306 | not conclusive | -3.08 | -3.22 | <0.001 | 35 | +0.03 | no |
| qwen3-instruct\|physionet2012 | pa | C0 | neutral | 600 | -5.70 | +2.12 | +2.44 | <0.001 | toward “Yes” | -3.80 | -2.64 | <0.001 | 25 | +0.19 | no |
| qwen3-instruct\|physionet2012 | pa | CP | threshold | 600 | -4.67 | -0.79 | -1.01 | 0.009 | toward “No” | +9.26 | +8.98 | <0.001 | 25 | +0.08 | no |
| qwen3-instruct\|physionet2012 | pa | CP | neutral | 600 | -4.67 | +1.62 | +1.70 | <0.001 | toward “Yes” | +9.09 | +9.09 | <0.001 | 23 | +0.17 | no |
| qwen3-instruct\|physionet2019 | yn | C0 | threshold | 542 | -5.31 | -0.34 | -1.50 | 0.339 | not conclusive | -1.15 | -0.28 | 0.081 | 27 | +0.03 | no |
| qwen3-instruct\|physionet2019 | yn | C0 | neutral | 542 | -5.31 | +2.54 | +2.04 | <0.001 | toward “Yes” | +0.89 | +0.47 | 0.005 | 27 | +0.19 | no |
| qwen3-instruct\|physionet2019 | yn | CP | threshold | 542 | -6.98 | -8.46 | +0.66 | 1.000 | not conclusive | +3.40 | +12.25 | 0.127 | 21 | – | no |
| qwen3-instruct\|physionet2019 | yn | CP | neutral | 542 | -6.98 | +7.16 | +4.80 | <0.001 | toward “Yes” | +6.70 | +13.09 | 0.004 | 11 | +0.34 | no |
| qwen3-instruct\|physionet2019 | ab | C0 | threshold | 542 | -4.16 | +0.56 | +1.71 | 0.170 | not conclusive | -3.04 | -2.88 | <0.001 | 15 | +0.07 | no |
| qwen3-instruct\|physionet2019 | ab | C0 | neutral | 542 | -4.16 | +3.61 | +3.40 | <0.001 | toward “Yes” | -2.39 | -2.24 | <0.001 | 27 | +0.41 | no |
| qwen3-instruct\|physionet2019 | ab | CP | threshold | 542 | -3.55 | -4.44 | +0.89 | 1.000 | not conclusive | +9.64 | +16.33 | 0.009 | 9 | – | no |
| qwen3-instruct\|physionet2019 | ab | CP | neutral | 542 | -3.55 | -0.69 | +1.52 | 1.000 | not conclusive | +11.87 | +17.02 | <0.001 | 8 | – | no |
| yi15-9b\|physionet2012 | yn | C0 | threshold | 600 | -1.19 | +0.96 | +0.59 | <0.001 | toward “Yes” | -0.02 | -0.34 | 0.748 | 145 | +0.25 | no |
| yi15-9b\|physionet2012 | yn | C0 | neutral | 600 | -1.19 | +0.93 | +0.86 | <0.001 | toward “Yes” | -0.34 | -0.49 | <0.001 | 158 | +0.36 | no |
| yi15-9b\|physionet2012 | yn | CP | threshold | 600 | -1.17 | +1.05 | +0.68 | <0.001 | toward “Yes” | +0.50 | +0.16 | 0.012 | 135 | +0.29 | no |
| yi15-9b\|physionet2012 | yn | CP | neutral | 600 | -1.17 | +0.80 | +0.74 | <0.001 | toward “Yes” | +0.43 | +0.21 | <0.001 | 145 | +0.32 | no |
| yi15-9b\|physionet2012 | pa | C0 | threshold | 600 | -1.21 | -0.17 | -0.51 | 0.064 | not conclusive | +0.04 | -0.32 | 1.000 | 117 | +0.07 | no |
| yi15-9b\|physionet2012 | pa | C0 | neutral | 600 | -1.21 | +0.17 | +0.05 | 0.089 | not conclusive | -0.08 | -0.27 | 0.332 | 144 | +0.02 | no |
| yi15-9b\|physionet2012 | pa | CP | threshold | 600 | -1.20 | +0.29 | +0.01 | 0.768 | not conclusive | +0.44 | +0.17 | 0.001 | 108 | +0.01 | no |
| yi15-9b\|physionet2012 | pa | CP | neutral | 600 | -1.20 | +0.19 | +0.10 | <0.001 | toward “Yes” | +0.25 | +0.11 | 0.019 | 154 | +0.04 | no |
| yi15-9b\|physionet2012 | tf | C0 | threshold | 600 | -1.64 | -0.77 | -1.06 | <0.001 | toward “No” | -0.91 | -1.04 | <0.001 | 119 | +0.24 | no |
| yi15-9b\|physionet2012 | tf | C0 | neutral | 600 | -1.64 | +0.52 | +0.50 | <0.001 | toward “Yes” | -1.29 | -1.35 | <0.001 | 263 | +0.15 | no |
| yi15-9b\|physionet2012 | tf | CP | threshold | 600 | -1.73 | +1.80 | +0.67 | <0.001 | toward “Yes” | +1.36 | +0.47 | <0.001 | 21 | +0.19 | no |
| yi15-9b\|physionet2012 | tf | CP | neutral | 600 | -1.73 | +0.72 | +0.66 | <0.001 | toward “Yes” | +1.28 | +0.60 | <0.001 | 206 | +0.19 | no |
| yi15-9b\|physionet2012 | ab | C0 | threshold | 600 | -0.84 | +0.34 | +0.35 | <0.001 | toward “Yes” | +0.64 | +0.55 | <0.001 | 127 | +0.20 | no |
| yi15-9b\|physionet2012 | ab | C0 | neutral | 600 | -0.84 | +0.93 | +0.93 | <0.001 | toward “Yes” | +0.72 | +0.70 | <0.001 | 124 | +0.55 | yes |
| yi15-9b\|physionet2012 | ab | CP | threshold | 600 | -0.05 | -0.80 | -0.91 | <0.001 | toward “No” | +3.52 | +3.05 | <0.001 | 79 | +7.34 | yes |
| yi15-9b\|physionet2012 | ab | CP | neutral | 600 | -0.05 | -0.79 | -0.90 | <0.001 | toward “No” | +3.53 | +3.07 | <0.001 | 78 | +7.25 | yes |
| yi15-9b\|physionet2019 | yn | C0 | threshold | 542 | -0.38 | +0.56 | +0.58 | <0.001 | toward “Yes” | -1.34 | -1.34 | <0.001 | 174 | +0.73 | yes |
| yi15-9b\|physionet2019 | yn | C0 | neutral | 542 | -0.38 | +0.65 | +0.77 | <0.001 | toward “Yes” | -1.33 | -1.29 | <0.001 | 220 | +0.85 | yes |
| yi15-9b\|physionet2019 | yn | CP | threshold | 542 | -0.58 | +1.35 | +0.95 | 0.062 | not conclusive | +2.03 | +1.73 | 0.013 | 120 | +0.83 | no |
| yi15-9b\|physionet2019 | yn | CP | neutral | 542 | -0.58 | +1.61 | +1.13 | <0.001 | toward “Yes” | +1.84 | +1.77 | <0.001 | 217 | +0.98 | yes |
| yi15-9b\|physionet2019 | pa | C0 | threshold | 542 | -0.68 | +0.10 | +0.18 | 0.148 | not conclusive | -0.02 | -0.10 | 0.756 | 144 | +0.08 | no |
| yi15-9b\|physionet2019 | pa | C0 | neutral | 542 | -0.68 | +0.57 | +0.58 | <0.001 | toward “Yes” | +0.03 | +0.00 | 0.918 | 178 | +0.42 | no |
| yi15-9b\|physionet2019 | pa | CP | threshold | 542 | -0.70 | +0.13 | +0.57 | 0.587 | not conclusive | +1.82 | +2.26 | <0.001 | 115 | +0.09 | no |
| yi15-9b\|physionet2019 | pa | CP | neutral | 542 | -0.70 | +0.63 | +0.70 | <0.001 | toward “Yes” | +2.09 | +2.28 | <0.001 | 211 | +0.45 | no |
| yi15-9b\|physionet2019 | tf | C0 | threshold | 542 | -1.28 | +1.72 | -0.66 | 1.000 | not conclusive | -2.41 | -0.15 | 0.014 | 66 | – | no |
| yi15-9b\|physionet2019 | tf | C0 | neutral | 542 | -1.28 | +0.62 | +0.30 | <0.001 | toward “Yes” | -0.40 | -0.00 | 0.942 | 321 | +0.12 | no |
| yi15-9b\|physionet2019 | tf | CP | threshold | 542 | -1.51 | +0.48 | +0.10 | 0.737 | not conclusive | +3.25 | +2.86 | 0.077 | 5 | +0.03 | no |
| yi15-9b\|physionet2019 | tf | CP | neutral | 542 | -1.51 | +1.46 | +0.92 | <0.001 | toward “Yes” | +2.14 | +2.28 | <0.001 | 232 | +0.31 | no |
| yi15-9b\|physionet2019 | ab | C0 | threshold | 542 | -0.72 | +0.79 | +0.56 | <0.001 | toward “Yes” | -0.09 | +0.24 | 1.000 | 75 | +0.39 | no |
| yi15-9b\|physionet2019 | ab | C0 | neutral | 542 | -0.72 | +1.09 | +1.13 | <0.001 | toward “Yes” | +0.01 | +0.34 | 0.863 | 125 | +0.76 | yes |
| yi15-9b\|physionet2019 | ab | CP | threshold | 542 | +0.62 | -0.45 | -1.42 | 0.186 | not conclusive | +3.34 | +4.51 | <0.001 | 70 | +0.36 | no |
| yi15-9b\|physionet2019 | ab | CP | neutral | 542 | +0.62 | -1.75 | -1.98 | <0.001 | toward “No” | +3.63 | +4.49 | <0.001 | 116 | +1.41 | yes |

## Q19. The size of the impacts

Mean movement m = mean(la − ls) in the alarm framing; Δm = m(×2) − m(×0.5).

| cell | n | registered card | numbers ×0.5 | ×1 | ×2 | Δm | p |
|---|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | 600 | +3.22 | +1.09 | +0.94 | +0.90 | -0.19 [-0.23, -0.15] | <0.001 |
| falcon3-7b\|physionet2019 | 542 | +4.62 | +2.27 | +2.38 | +2.19 | -0.08 [-0.12, -0.03] | 0.003 |
| granite3-instruct\|physionet2012 | 600 | +6.87 | +0.18 | -0.11 | +0.02 | -0.17 [-0.27, -0.06] | 0.002 |
| granite3-instruct\|physionet2019 | 542 | +9.79 | +1.39 | +1.17 | +0.74 | -0.66 [-0.77, -0.56] | <0.001 |
| llama-instruct\|physionet2012 | 600 | +1.41 | +0.42 | +0.36 | +0.28 | -0.14 [-0.16, -0.12] | <0.001 |
| llama-instruct\|physionet2019 | 542 | +1.58 | +0.56 | +0.60 | +0.58 | +0.02 [+0.00, +0.03] | 0.028 |
| olmo3-instruct\|physionet2012 | 600 | +1.71 | +1.10 | +1.08 | +1.09 | -0.01 [-0.04, +0.01] | 0.269 |
| olmo3-instruct\|physionet2019 | 542 | +2.51 | +0.93 | +0.94 | +0.86 | -0.08 [-0.10, -0.06] | <0.001 |
| qwen-instruct\|physionet2012 | 600 | +5.11 | +2.34 | +2.23 | +2.36 | +0.02 [-0.16, +0.20] | 0.777 |
| qwen-instruct\|physionet2019 | 542 | +9.12 | +3.62 | +3.85 | +3.96 | +0.35 [+0.17, +0.53] | <0.001 |
| qwen3-instruct\|physionet2012 | 600 | +7.93 | +2.97 | +3.27 | +3.96 | +1.00 [+0.87, +1.12] | <0.001 |
| qwen3-instruct\|physionet2019 | 542 | +8.11 | +2.68 | +2.86 | +2.89 | +0.21 [+0.11, +0.31] | <0.001 |
| yi15-9b\|physionet2012 | 600 | +1.47 | +0.12 | +0.15 | +0.20 | +0.08 [+0.04, +0.12] | <0.001 |
| yi15-9b\|physionet2019 | 542 | +1.81 | +0.24 | +0.27 | +0.28 | +0.03 [+0.02, +0.04] | <0.001 |

## Q20. A probability instead of a card

| cell | baseline | reading | n | W (a) | W (a′) | W | O (a) | O (a′) | O |
|---|---|---|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | C0 | threshold | 599 | -0.70 | -0.72 | toward “No” | +3.08 | +3.09 | toward alarm |
| falcon3-7b\|physionet2012 | C0 | neutral | 599 | -0.60 | -0.60 | toward “No” | +3.07 | +3.09 | toward alarm |
| falcon3-7b\|physionet2012 | CP | threshold | 599 | -0.42 | -0.51 | toward “No” | +3.81 | +3.52 | toward alarm |
| falcon3-7b\|physionet2012 | CP | neutral | 599 | +1.13 | +1.04 | toward “Yes” | +3.69 | +3.48 | toward alarm |
| falcon3-7b\|physionet2019 | C0 | threshold | 535 | -0.97 | -1.06 | toward “No” | +4.04 | +3.81 | toward alarm |
| falcon3-7b\|physionet2019 | C0 | neutral | 535 | -0.89 | -1.00 | toward “No” | +4.04 | +3.81 | toward alarm |
| falcon3-7b\|physionet2019 | CP | threshold | 535 | -1.97 | -0.31 | toward “No” | +2.80 | +4.57 | toward alarm |
| falcon3-7b\|physionet2019 | CP | neutral | 535 | +4.66 | +1.92 | toward “Yes” | +1.70 | +4.64 | not conclusive |
| granite3-instruct\|physionet2012 | C0 | threshold | 599 | -0.28 | -0.22 | toward “No” | +3.44 | +3.48 | toward alarm |
| granite3-instruct\|physionet2012 | C0 | neutral | 599 | +0.24 | +0.26 | toward “Yes” | +3.44 | +3.45 | toward alarm |
| granite3-instruct\|physionet2012 | CP | threshold | 599 | +1.37 | +1.34 | toward “Yes” | +3.72 | +3.75 | toward alarm |
| granite3-instruct\|physionet2012 | CP | neutral | 599 | +3.79 | +3.85 | toward “Yes” | +3.59 | +3.56 | toward alarm |
| granite3-instruct\|physionet2019 | C0 | threshold | 535 | -0.18 | -1.56 | not conclusive | +7.31 | +9.41 | toward alarm |
| granite3-instruct\|physionet2019 | C0 | neutral | 535 | -0.03 | -0.46 | not conclusive | +7.61 | +9.45 | toward alarm |
| granite3-instruct\|physionet2019 | CP | threshold | 535 | -2.09 | -0.25 | toward “No” | +11.62 | +13.45 | toward alarm |
| granite3-instruct\|physionet2019 | CP | neutral | 535 | +3.68 | +3.56 | toward “Yes” | +12.18 | +13.40 | toward alarm |
| llama-instruct\|physionet2012 | C0 | threshold | 599 | +0.99 | +0.77 | toward “Yes” | +1.88 | +1.65 | toward alarm |
| llama-instruct\|physionet2012 | C0 | neutral | 599 | +2.03 | +2.07 | toward “Yes” | +2.15 | +2.06 | toward alarm |
| llama-instruct\|physionet2012 | CP | threshold | 599 | +2.58 | +1.86 | toward “Yes” | +1.36 | +1.17 | toward alarm |
| llama-instruct\|physionet2012 | CP | neutral | 599 | +2.63 | +2.75 | toward “Yes” | +1.17 | +1.00 | toward alarm |
| llama-instruct\|physionet2019 | C0 | threshold | 535 | -2.38 | +0.63 | not conclusive | +5.29 | +2.18 | toward alarm |
| llama-instruct\|physionet2019 | C0 | neutral | 535 | +1.87 | +1.44 | toward “Yes” | +4.65 | +2.13 | toward alarm |
| llama-instruct\|physionet2019 | CP | threshold | 535 | +3.68 | +0.90 | not conclusive | +10.09 | +2.95 | not conclusive |
| llama-instruct\|physionet2019 | CP | neutral | 535 | +2.11 | +2.28 | toward “Yes” | +3.15 | +2.90 | toward alarm |
| olmo3-instruct\|physionet2012 | C0 | threshold | 599 | -1.18 | -1.71 | toward “No” | +3.79 | +3.16 | toward alarm |
| olmo3-instruct\|physionet2012 | C0 | neutral | 599 | +0.17 | -0.09 | not conclusive | +3.98 | +3.71 | toward alarm |
| olmo3-instruct\|physionet2012 | CP | threshold | 599 | +1.20 | +0.89 | not conclusive | +4.93 | +4.88 | toward alarm |
| olmo3-instruct\|physionet2012 | CP | neutral | 599 | +2.42 | +1.92 | toward “Yes” | +8.09 | +6.35 | toward alarm |
| olmo3-instruct\|physionet2019 | C0 | threshold | 535 | -4.43 | -4.15 | toward “No” | +6.20 | +6.66 | toward alarm |
| olmo3-instruct\|physionet2019 | C0 | neutral | 535 | -1.52 | -1.48 | toward “No” | +7.01 | +6.61 | toward alarm |
| olmo3-instruct\|physionet2019 | CP | threshold | 535 | +8.88 | -3.24 | not conclusive | +21.25 | +9.70 | not conclusive |
| olmo3-instruct\|physionet2019 | CP | neutral | 535 | +10.19 | +1.06 | toward “Yes” | +7.25 | +9.88 | not conclusive |
| qwen-instruct\|physionet2012 | C0 | threshold | 599 | +5.34 | +5.23 | toward “Yes” | +0.18 | -0.92 | not conclusive |
| qwen-instruct\|physionet2012 | C0 | neutral | 599 | +6.55 | +6.61 | toward “Yes” | +0.21 | -0.67 | not conclusive |
| qwen-instruct\|physionet2012 | CP | threshold | 599 | +4.19 | +3.52 | toward “Yes” | +6.04 | +5.01 | toward alarm |
| qwen-instruct\|physionet2012 | CP | neutral | 599 | +5.92 | +5.43 | toward “Yes” | +6.03 | +5.36 | toward alarm |
| qwen-instruct\|physionet2019 | C0 | threshold | 535 | +0.04 | +2.22 | not conclusive | +5.43 | +2.82 | toward alarm |
| qwen-instruct\|physionet2019 | C0 | neutral | 535 | +4.80 | +4.90 | toward “Yes” | +4.62 | +2.52 | toward alarm |
| qwen-instruct\|physionet2019 | CP | threshold | 535 | +3.50 | +3.86 | not conclusive | +10.99 | +11.10 | toward alarm |
| qwen-instruct\|physionet2019 | CP | neutral | 535 | +8.79 | +8.79 | toward “Yes” | +12.05 | +11.79 | toward alarm |
| qwen3-instruct\|physionet2012 | C0 | threshold | 599 | -8.43 | -10.72 | toward “No” | -1.60 | +1.31 | not conclusive |
| qwen3-instruct\|physionet2012 | C0 | neutral | 599 | +0.26 | -0.31 | not conclusive | +1.43 | +1.58 | toward alarm |
| qwen3-instruct\|physionet2012 | CP | threshold | 599 | -18.36 | -12.65 | toward “No” | -1.48 | +4.02 | not conclusive |
| qwen3-instruct\|physionet2012 | CP | neutral | 599 | -1.31 | -0.28 | not conclusive | +4.23 | +6.28 | toward alarm |
| qwen3-instruct\|physionet2019 | C0 | threshold | 535 | -2.68 | -3.62 | toward “No” | +7.17 | +6.65 | toward alarm |
| qwen3-instruct\|physionet2019 | C0 | neutral | 535 | +2.59 | +2.60 | toward “Yes” | +6.05 | +5.98 | toward alarm |
| qwen3-instruct\|physionet2019 | CP | threshold | 535 | -16.46 | -0.19 | not conclusive | -1.22 | +14.81 | not conclusive |
| qwen3-instruct\|physionet2019 | CP | neutral | 535 | +8.71 | +7.32 | toward “Yes” | +7.20 | +13.97 | toward alarm |
| yi15-9b\|physionet2012 | C0 | threshold | 599 | -2.30 | -2.46 | toward “No” | +0.69 | +0.65 | toward alarm |
| yi15-9b\|physionet2012 | C0 | neutral | 599 | -1.25 | -1.26 | toward “No” | +0.70 | +0.60 | toward alarm |
| yi15-9b\|physionet2012 | CP | threshold | 599 | -2.26 | -2.53 | toward “No” | +1.02 | +0.84 | toward alarm |
| yi15-9b\|physionet2012 | CP | neutral | 599 | -1.35 | -1.36 | toward “No” | +1.11 | +0.90 | toward alarm |
| yi15-9b\|physionet2019 | C0 | threshold | 535 | -1.42 | -1.03 | toward “No” | +1.25 | +0.87 | toward alarm |
| yi15-9b\|physionet2019 | C0 | neutral | 535 | -0.84 | -0.62 | toward “No” | +1.11 | +0.86 | toward alarm |
| yi15-9b\|physionet2019 | CP | threshold | 535 | -0.59 | -0.52 | not conclusive | +2.76 | +2.93 | toward alarm |
| yi15-9b\|physionet2019 | CP | neutral | 535 | +0.11 | -0.03 | not conclusive | +2.74 | +2.86 | toward alarm |

## Q21. Diagnostics

**a. The placebo card's effect on the static lean and on the outcome:**

| cell | λ(C0) | λ(CP) | Δλ | p | CP − C0 (alarm log-odds) | 95% interval |
|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | -0.08 | -0.89 | -0.82 | <0.001 | -0.51 | [-0.58, -0.44] |
| falcon3-7b\|physionet2019 | -0.04 | -1.57 | -1.53 | <0.001 | -1.45 | [-1.52, -1.37] |
| granite3-instruct\|physionet2012 | -0.33 | -2.06 | -1.73 | <0.001 | +0.09 | [-0.08, +0.25] |
| granite3-instruct\|physionet2019 | -0.74 | -2.79 | -2.04 | <0.001 | -6.37 | [-6.49, -6.26] |
| llama-instruct\|physionet2012 | -1.81 | -1.91 | -0.09 | <0.001 | +0.94 | [+0.88, +1.01] |
| llama-instruct\|physionet2019 | -0.81 | -1.22 | -0.41 | <0.001 | -0.68 | [-0.72, -0.64] |
| olmo3-instruct\|physionet2012 | -1.84 | -3.28 | -1.45 | <0.001 | -0.63 | [-0.79, -0.46] |
| olmo3-instruct\|physionet2019 | -1.95 | -3.41 | -1.46 | <0.001 | -4.19 | [-4.30, -4.08] |
| qwen-instruct\|physionet2012 | -1.93 | -2.50 | -0.57 | <0.001 | -4.81 | [-5.17, -4.44] |
| qwen-instruct\|physionet2019 | -2.85 | -5.00 | -2.15 | <0.001 | -8.80 | [-9.19, -8.41] |
| qwen3-instruct\|physionet2012 | -7.16 | -7.30 | -0.14 | 0.032 | -8.18 | [-8.54, -7.83] |
| qwen3-instruct\|physionet2019 | -5.31 | -6.98 | -1.67 | <0.001 | -8.13 | [-8.42, -7.84] |
| yi15-9b\|physionet2012 | -1.19 | -1.17 | +0.02 | 0.304 | -0.31 | [-0.36, -0.25] |
| yi15-9b\|physionet2019 | -0.38 | -0.58 | -0.20 | <0.001 | -1.84 | [-1.88, -1.81] |

**b. The Yes/No pair against C0: earlier patients → fresh patients** (+1 / −1 conclusive sign, 0 not conclusive):

| cell | W threshold | O threshold | W neutral | O neutral |
|---|---|---|---|---|
| llama-instruct\|physionet2012 | -1 → -1 | -1 → -1 | +0 → +0 | +1 → +1 |
| llama-instruct\|physionet2019 | +0 → +0 | -1 → -1 | +0 → +0 | -1 → -1 |
| qwen-instruct\|physionet2012 | +1 → +1 | -1 → -1 | +1 → +1 | -1 → -1 |
| qwen-instruct\|physionet2019 | -1 → -1 | +0 → +0 | +0 → -1 | +0 → +0 |
| qwen3-instruct\|physionet2012 | -1 → -1 | -1 → -1 | +1 → +1 | -1 → -1 |
| qwen3-instruct\|physionet2019 | +0 → +0 | -1 → +0 | +1 → +1 | +0 → +1 |
| olmo3-instruct\|physionet2012 | -1 → -1 | +1 → +1 | +0 → +0 | +1 → +1 |
| olmo3-instruct\|physionet2019 | -1 → -1 | +0 → +0 | -1 → +0 | +1 → +1 |
| granite3-instruct\|physionet2012 | -1 → -1 | +1 → +1 | -1 → -1 | +1 → +1 |
| falcon3-7b\|physionet2012 | +1 → +1 | +1 → +1 | +1 → +1 | +0 → +1 |
| falcon3-7b\|physionet2019 | +1 → +1 | +1 → +1 | +1 → +1 | +1 → +1 |
| yi15-9b\|physionet2012 | +1 → +1 | -1 → +0 | +1 → +1 | -1 → -1 |
| yi15-9b\|physionet2019 | +1 → +1 | -1 → -1 | +1 → +1 | -1 → -1 |

| reading | earlier conclusive | same sign | opposite | not conclusive now |
|---|---|---|---|---|
| threshold\|W | 11 | 11 | 0 | 0 |
| threshold\|O | 11 | 9 | 0 | 2 |
| neutral\|O | 10 | 10 | 0 | 0 |
| neutral\|W | 9 | 8 | 0 | 1 |

**d. Generated answers** (agreement with the first-token reading among parsed answers; E_W from the generated answers and from the first-token reading of the same patients):

| cell | answers | parsed | agreement | patients with 6 parsed | E_W generated | E_W first token |
|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | 1200 | 1.000 | 0.913 | 200 | +0.115 | +0.060 |
| falcon3-7b\|physionet2019 | 1200 | 1.000 | 0.893 | 200 | -0.035 | +0.090 |
| granite3-instruct\|physionet2012 | 1200 | 0.959 | 0.858 | 156 | +0.176 | -0.260 |
| granite3-instruct\|physionet2019 | 1200 | 0.978 | 0.885 | 175 | +0.317 | -0.118 |
| llama-instruct\|physionet2012 | 1200 | 0.592 | 0.697 | 8 | – | – |
| llama-instruct\|physionet2019 | 1200 | 0.573 | 0.698 | 9 | – | – |
| olmo3-instruct\|physionet2012 | 1200 | 1.000 | 0.936 | 200 | -0.380 | -0.305 |
| olmo3-instruct\|physionet2019 | 1200 | 1.000 | 0.973 | 200 | +0.050 | -0.020 |
| qwen-instruct\|physionet2012 | 1200 | 0.936 | 0.654 | 140 | -0.250 | +0.025 |
| qwen-instruct\|physionet2019 | 1200 | 0.912 | 0.748 | 113 | -0.076 | -0.135 |
| qwen3-instruct\|physionet2012 | 1200 | 1.000 | 0.980 | 200 | -0.195 | -0.140 |
| qwen3-instruct\|physionet2019 | 1200 | 1.000 | 0.960 | 200 | -0.105 | -0.090 |
| yi15-9b\|physionet2012 | 1200 | 1.000 | 0.835 | 200 | -0.035 | +0.115 |
| yi15-9b\|physionet2019 | 1200 | 1.000 | 0.628 | 200 | +0.015 | +0.180 |

**e. Per framing and set:** invalid records, median answer-word mass, imputed share, accuracy and the share of answers that follow the evidence, by condition:

| cell | framing | set | n | invalid | mass | imputed | accuracy | follows the evidence |
|---|---|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | alarm\|A | 3600 | 0 | 0.999 | 0.000 | C0 0.65, C2 0.76, C4 0.65, CP 0.64, CQ2 0.78, CQ4 0.29 | C2 0.90, C4 0.33, CQ2 0.94, CQ4 0.84 |
| falcon3-7b\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.999 | 0.000 | C2 0.73, C4 0.69 | C2 0.89, C4 0.19 |
| falcon3-7b\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.999 | 0.000 | C2 0.74, C4 0.70 | C2 0.89, C4 0.18 |
| falcon3-7b\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.999 | 0.000 | C2 0.73, C4 0.69 | C2 0.88, C4 0.18 |
| falcon3-7b\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 1.000 | 0.256 | C0 0.69, C2 0.71, C4 0.62, CP 0.65 | C2 0.89, C4 0.38 |
| falcon3-7b\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 1.000 | 0.062 | C0 0.70, C2 0.76, C4 0.66, CP 0.69 | C2 0.92, C4 0.39 |
| falcon3-7b\|physionet2012 | para_alarm\|A | 2400 | 0 | 0.998 | 0.000 | C0 0.68, C2 0.76, C4 0.60, CP 0.65 | C2 0.91, C4 0.44 |
| falcon3-7b\|physionet2012 | para_reversed\|A | 2400 | 0 | 0.998 | 0.000 | C0 0.65, C2 0.75, C4 0.66, CP 0.69 | C2 0.89, C4 0.29 |
| falcon3-7b\|physionet2012 | reversed\|A | 3600 | 0 | 0.999 | 0.000 | C0 0.62, C2 0.68, C4 0.61, CP 0.65, CQ2 0.76, CQ4 0.32 | C2 0.79, C4 0.29, CQ2 0.88, CQ4 0.80 |
| falcon3-7b\|physionet2012 | tf_alarm\|A | 2400 | 0 | 0.999 | 0.084 | C0 0.67, C2 0.70, C4 0.56, CP 0.56 | C2 0.86, C4 0.36 |
| falcon3-7b\|physionet2012 | tf_reversed\|A | 2400 | 0 | 0.999 | 0.005 | C0 0.69, C2 0.77, C4 0.69, CP 0.71 | C2 0.89, C4 0.34 |
| falcon3-7b\|physionet2019 | alarm\|A | 3252 | 0 | 0.999 | 0.000 | C0 0.63, C2 0.64, C4 0.42, CP 0.58, CQ2 0.68, CQ4 0.38 | C2 0.76, C4 0.80, CQ2 0.88, CQ4 0.91 |
| falcon3-7b\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.999 | 0.000 | C2 0.65, C4 0.54 | C2 0.74, C4 0.65 |
| falcon3-7b\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.999 | 0.000 | C2 0.66, C4 0.51 | C2 0.72, C4 0.68 |
| falcon3-7b\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.999 | 0.000 | C2 0.65, C4 0.51 | C2 0.68, C4 0.68 |
| falcon3-7b\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 1.000 | 0.102 | C0 0.59, C2 0.65, C4 0.40, CP 0.56 | C2 0.88, C4 0.86 |
| falcon3-7b\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 1.000 | 0.021 | C0 0.62, C2 0.64, C4 0.43, CP 0.58 | C2 0.79, C4 0.81 |
| falcon3-7b\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.999 | 0.000 | C0 0.64, C2 0.65, C4 0.43, CP 0.57 | C2 0.80, C4 0.82 |
| falcon3-7b\|physionet2019 | para_reversed\|A | 2168 | 0 | 0.998 | 0.000 | C0 0.58, C2 0.54, C4 0.57, CP 0.51 | C2 0.39, C4 0.37 |
| falcon3-7b\|physionet2019 | reversed\|A | 3252 | 0 | 0.999 | 0.000 | C0 0.60, C2 0.59, C4 0.52, CP 0.60, CQ2 0.66, CQ4 0.46 | C2 0.51, C4 0.44, CQ2 0.61, CQ4 0.76 |
| falcon3-7b\|physionet2019 | tf_alarm\|A | 2168 | 0 | 0.998 | 0.000 | C0 0.61, C2 0.65, C4 0.41, CP 0.52 | C2 0.89, C4 0.84 |
| falcon3-7b\|physionet2019 | tf_reversed\|A | 2168 | 0 | 0.999 | 0.000 | C0 0.63, C2 0.65, C4 0.46, CP 0.59 | C2 0.68, C4 0.73 |
| granite3-instruct\|physionet2012 | alarm\|A | 3600 | 0 | 0.981 | 0.105 | C0 0.71, C2 0.77, C4 0.47, CP 0.67, CQ2 0.77, CQ4 0.32 | C2 0.90, C4 0.58, CQ2 0.96, CQ4 0.84 |
| granite3-instruct\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.993 | 0.041 | C2 0.70, C4 0.70 | C2 0.82, C4 0.16 |
| granite3-instruct\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.993 | 0.024 | C2 0.71, C4 0.69 | C2 0.82, C4 0.16 |
| granite3-instruct\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.992 | 0.025 | C2 0.70, C4 0.69 | C2 0.81, C4 0.17 |
| granite3-instruct\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 0.996 | 0.015 | C0 0.62, C2 0.73, C4 0.40, CP 0.65 | C2 0.81, C4 0.71 |
| granite3-instruct\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 0.997 | 0.066 | C0 0.64, C2 0.71, C4 0.38, CP 0.72 | C2 0.83, C4 0.75 |
| granite3-instruct\|physionet2012 | para_alarm\|A | 2400 | 0 | 0.884 | 0.080 | C0 0.74, C2 0.77, C4 0.41, CP 0.72 | C2 0.90, C4 0.69 |
| granite3-instruct\|physionet2012 | para_reversed\|A | 2400 | 0 | 0.938 | 0.076 | C0 0.75, C2 0.76, C4 0.43, CP 0.76 | C2 0.88, C4 0.68 |
| granite3-instruct\|physionet2012 | reversed\|A | 3600 | 0 | 0.971 | 0.086 | C0 0.71, C2 0.73, C4 0.44, CP 0.77, CQ2 0.78, CQ4 0.33 | C2 0.82, C4 0.67, CQ2 0.96, CQ4 0.85 |
| granite3-instruct\|physionet2012 | tf_alarm\|A | 2400 | 0 | 0.977 | 0.000 | C0 0.61, C2 0.64, C4 0.43, CP 0.50 | C2 0.76, C4 0.53 |
| granite3-instruct\|physionet2012 | tf_reversed\|A | 2400 | 0 | 0.984 | 0.000 | C0 0.56, C2 0.63, C4 0.50, CP 0.50 | C2 0.57, C4 0.58 |
| granite3-instruct\|physionet2019 | alarm\|A | 3252 | 0 | 0.951 | 0.093 | C0 0.56, C2 0.65, C4 0.40, CP 0.54, CQ2 0.58, CQ4 0.50 | C2 0.88, C4 0.91, CQ2 0.41, CQ4 0.69 |
| granite3-instruct\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.921 | 0.000 | C2 0.61, C4 0.49 | C2 0.52, C4 0.68 |
| granite3-instruct\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.935 | 0.000 | C2 0.54, C4 0.48 | C2 0.44, C4 0.68 |
| granite3-instruct\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.941 | 0.000 | C2 0.51, C4 0.50 | C2 0.39, C4 0.65 |
| granite3-instruct\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 0.998 | 0.007 | C0 0.50, C2 0.63, C4 0.46, CP 0.59 | C2 0.66, C4 0.74 |
| granite3-instruct\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 0.998 | 0.000 | C0 0.54, C2 0.64, C4 0.46, CP 0.54 | C2 0.64, C4 0.73 |
| granite3-instruct\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.918 | 0.087 | C0 0.53, C2 0.65, C4 0.38, CP 0.58 | C2 0.91, C4 0.93 |
| granite3-instruct\|physionet2019 | para_reversed\|A | 2168 | 0 | 0.947 | 0.075 | C0 0.50, C2 0.63, C4 0.48, CP 0.63 | C2 0.57, C4 0.70 |
| granite3-instruct\|physionet2019 | reversed\|A | 3252 | 0 | 0.930 | 0.055 | C0 0.50, C2 0.62, C4 0.48, CP 0.61, CQ2 0.54, CQ4 0.50 | C2 0.54, C4 0.69, CQ2 0.37, CQ4 0.69 |
| granite3-instruct\|physionet2019 | tf_alarm\|A | 2168 | 0 | 0.990 | 0.000 | C0 0.59, C2 0.58, C4 0.36, CP 0.50 | C2 0.83, C4 0.74 |
| granite3-instruct\|physionet2019 | tf_reversed\|A | 2168 | 0 | 0.992 | 0.000 | C0 0.50, C2 0.59, C4 0.49, CP 0.51 | C2 0.49, C4 0.67 |
| llama-instruct\|physionet2012 | alarm\|A | 3600 | 0 | 0.984 | 0.000 | C0 0.60, C2 0.63, C4 0.56, CP 0.57, CQ2 0.72, CQ4 0.41 | C2 0.76, C4 0.36, CQ2 0.91, CQ4 0.68 |
| llama-instruct\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.984 | 0.000 | C2 0.63, C4 0.61 | C2 0.78, C4 0.26 |
| llama-instruct\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.984 | 0.000 | C2 0.63, C4 0.62 | C2 0.78, C4 0.26 |
| llama-instruct\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.983 | 0.000 | C2 0.63, C4 0.62 | C2 0.78, C4 0.25 |
| llama-instruct\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 0.998 | 0.170 | C0 0.61, C2 0.66, C4 0.59, CP 0.61 | C2 0.81, C4 0.30 |
| llama-instruct\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 0.998 | 0.163 | C0 0.65, C2 0.65, C4 0.59, CP 0.64 | C2 0.78, C4 0.30 |
| llama-instruct\|physionet2012 | para_alarm\|A | 2400 | 0 | 0.988 | 0.000 | C0 0.63, C2 0.70, C4 0.62, CP 0.59 | C2 0.82, C4 0.30 |
| llama-instruct\|physionet2012 | para_reversed\|A | 2400 | 0 | 0.989 | 0.000 | C0 0.68, C2 0.73, C4 0.67, CP 0.71 | C2 0.83, C4 0.33 |
| llama-instruct\|physionet2012 | reversed\|A | 3600 | 0 | 0.987 | 0.000 | C0 0.71, C2 0.74, C4 0.72, CP 0.69, CQ2 0.77, CQ4 0.43 | C2 0.88, C4 0.24, CQ2 0.92, CQ4 0.67 |
| llama-instruct\|physionet2012 | tf_alarm\|A | 2400 | 0 | 0.987 | 0.000 | C0 0.61, C2 0.66, C4 0.56, CP 0.53 | C2 0.81, C4 0.35 |
| llama-instruct\|physionet2012 | tf_reversed\|A | 2400 | 0 | 0.983 | 0.000 | C0 0.72, C2 0.70, C4 0.50, CP 0.51 | C2 0.69, C4 0.59 |
| llama-instruct\|physionet2019 | alarm\|A | 3252 | 0 | 0.992 | 0.000 | C0 0.60, C2 0.63, C4 0.44, CP 0.49, CQ2 0.59, CQ4 0.48 | C2 0.85, C4 0.70, CQ2 0.56, CQ4 0.76 |
| llama-instruct\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.994 | 0.000 | C2 0.61, C4 0.51 | C2 0.69, C4 0.58 |
| llama-instruct\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.994 | 0.000 | C2 0.61, C4 0.52 | C2 0.68, C4 0.60 |
| llama-instruct\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.994 | 0.000 | C2 0.62, C4 0.52 | C2 0.68, C4 0.60 |
| llama-instruct\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 0.998 | 0.026 | C0 0.51, C2 0.64, C4 0.45, CP 0.56 | C2 0.67, C4 0.74 |
| llama-instruct\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 0.998 | 0.037 | C0 0.56, C2 0.56, C4 0.51, CP 0.51 | C2 0.71, C4 0.40 |
| llama-instruct\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.992 | 0.000 | C0 0.59, C2 0.64, C4 0.43, CP 0.53 | C2 0.80, C4 0.78 |
| llama-instruct\|physionet2019 | para_reversed\|A | 2168 | 0 | 0.993 | 0.000 | C0 0.51, C2 0.54, C4 0.49, CP 0.51 | C2 0.41, C4 0.67 |
| llama-instruct\|physionet2019 | reversed\|A | 3252 | 0 | 0.992 | 0.000 | C0 0.52, C2 0.60, C4 0.50, CP 0.54, CQ2 0.54, CQ4 0.50 | C2 0.57, C4 0.65, CQ2 0.37, CQ4 0.69 |
| llama-instruct\|physionet2019 | tf_alarm\|A | 2168 | 0 | 0.988 | 0.000 | C0 0.58, C2 0.57, C4 0.46, CP 0.50 | C2 0.77, C4 0.50 |
| llama-instruct\|physionet2019 | tf_reversed\|A | 2168 | 0 | 0.993 | 0.000 | C0 0.50, C2 0.50, C4 0.50, CP 0.50 | C2 0.35, C4 0.65 |
| olmo3-instruct\|physionet2012 | alarm\|A | 3600 | 0 | 0.998 | 0.028 | C0 0.64, C2 0.67, C4 0.61, CP 0.50, CQ2 0.69, CQ4 0.47 | C2 0.80, C4 0.32, CQ2 0.85, CQ4 0.56 |
| olmo3-instruct\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.997 | 0.000 | C2 0.65, C4 0.62 | C2 0.77, C4 0.29 |
| olmo3-instruct\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.996 | 0.000 | C2 0.65, C4 0.62 | C2 0.78, C4 0.29 |
| olmo3-instruct\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.996 | 0.000 | C2 0.66, C4 0.62 | C2 0.77, C4 0.29 |
| olmo3-instruct\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 0.999 | 0.003 | C0 0.70, C2 0.74, C4 0.51, CP 0.58 | C2 0.78, C4 0.57 |
| olmo3-instruct\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 0.999 | 0.000 | C0 0.69, C2 0.71, C4 0.58, CP 0.62 | C2 0.78, C4 0.47 |
| olmo3-instruct\|physionet2012 | para_alarm\|A | 2400 | 0 | 0.997 | 0.001 | C0 0.65, C2 0.69, C4 0.62, CP 0.57 | C2 0.82, C4 0.34 |
| olmo3-instruct\|physionet2012 | para_reversed\|A | 2400 | 0 | 0.995 | 0.006 | C0 0.70, C2 0.72, C4 0.54, CP 0.67 | C2 0.72, C4 0.53 |
| olmo3-instruct\|physionet2012 | reversed\|A | 3600 | 0 | 0.996 | 0.001 | C0 0.68, C2 0.71, C4 0.59, CP 0.66, CQ2 0.75, CQ4 0.48 | C2 0.77, C4 0.42, CQ2 0.82, CQ4 0.62 |
| olmo3-instruct\|physionet2012 | tf_alarm\|A | 2400 | 0 | 0.999 | 0.000 | C0 0.66, C2 0.68, C4 0.48, CP 0.50 | C2 0.84, C4 0.52 |
| olmo3-instruct\|physionet2012 | tf_reversed\|A | 2400 | 0 | 0.998 | 0.000 | C0 0.66, C2 0.72, C4 0.49, CP 0.69 | C2 0.74, C4 0.60 |
| olmo3-instruct\|physionet2019 | alarm\|A | 3252 | 0 | 0.998 | 0.027 | C0 0.61, C2 0.60, C4 0.49, CP 0.50, CQ2 0.65, CQ4 0.46 | C2 0.78, C4 0.53, CQ2 0.77, CQ4 0.76 |
| olmo3-instruct\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.997 | 0.000 | C2 0.58, C4 0.55 | C2 0.73, C4 0.40 |
| olmo3-instruct\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.996 | 0.000 | C2 0.58, C4 0.55 | C2 0.74, C4 0.41 |
| olmo3-instruct\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.996 | 0.000 | C2 0.58, C4 0.55 | C2 0.73, C4 0.42 |
| olmo3-instruct\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 1.000 | 0.000 | C0 0.61, C2 0.65, C4 0.46, CP 0.54 | C2 0.65, C4 0.74 |
| olmo3-instruct\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 1.000 | 0.002 | C0 0.62, C2 0.66, C4 0.44, CP 0.51 | C2 0.80, C4 0.77 |
| olmo3-instruct\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.998 | 0.015 | C0 0.62, C2 0.62, C4 0.51, CP 0.50 | C2 0.77, C4 0.54 |
| olmo3-instruct\|physionet2019 | para_reversed\|A | 2168 | 0 | 0.997 | 0.000 | C0 0.53, C2 0.65, C4 0.48, CP 0.59 | C2 0.63, C4 0.70 |
| olmo3-instruct\|physionet2019 | reversed\|A | 3252 | 0 | 0.998 | 0.000 | C0 0.52, C2 0.64, C4 0.47, CP 0.58, CQ2 0.50, CQ4 0.50 | C2 0.64, C4 0.72, CQ2 0.31, CQ4 0.69 |
| olmo3-instruct\|physionet2019 | tf_alarm\|A | 2168 | 0 | 0.999 | 0.000 | C0 0.61, C2 0.63, C4 0.44, CP 0.50 | C2 0.82, C4 0.67 |
| olmo3-instruct\|physionet2019 | tf_reversed\|A | 2168 | 0 | 0.999 | 0.000 | C0 0.50, C2 0.60, C4 0.49, CP 0.60 | C2 0.54, C4 0.68 |
| qwen-instruct\|physionet2012 | alarm\|A | 3600 | 0 | 0.998 | 0.102 | C0 0.69, C2 0.72, C4 0.60, CP 0.60, CQ2 0.75, CQ4 0.59 | C2 0.83, C4 0.31, CQ2 0.89, CQ4 0.42 |
| qwen-instruct\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.998 | 0.055 | C2 0.71, C4 0.66 | C2 0.83, C4 0.26 |
| qwen-instruct\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.998 | 0.058 | C2 0.71, C4 0.66 | C2 0.82, C4 0.26 |
| qwen-instruct\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.999 | 0.068 | C2 0.69, C4 0.65 | C2 0.82, C4 0.26 |
| qwen-instruct\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 1.000 | 0.884 | C0 0.58, C2 0.62, C4 0.52, CP 0.54 | C2 0.72, C4 0.40 |
| qwen-instruct\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 1.000 | 0.720 | C0 0.63, C2 0.69, C4 0.56, CP 0.57 | C2 0.83, C4 0.40 |
| qwen-instruct\|physionet2012 | para_alarm\|A | 2400 | 0 | 1.000 | 0.117 | C0 0.72, C2 0.74, C4 0.61, CP 0.61 | C2 0.87, C4 0.35 |
| qwen-instruct\|physionet2012 | para_reversed\|A | 2400 | 0 | 1.000 | 0.250 | C0 0.73, C2 0.76, C4 0.66, CP 0.70 | C2 0.87, C4 0.29 |
| qwen-instruct\|physionet2012 | reversed\|A | 3600 | 0 | 1.000 | 0.311 | C0 0.68, C2 0.70, C4 0.66, CP 0.64, CQ2 0.71, CQ4 0.51 | C2 0.80, C4 0.28, CQ2 0.85, CQ4 0.48 |
| qwen-instruct\|physionet2012 | tf_alarm\|A | 2400 | 0 | 1.000 | 0.030 | C0 0.67, C2 0.65, C4 0.51, CP 0.53 | C2 0.77, C4 0.45 |
| qwen-instruct\|physionet2012 | tf_reversed\|A | 2400 | 0 | 1.000 | 0.007 | C0 0.66, C2 0.69, C4 0.58, CP 0.58 | C2 0.80, C4 0.37 |
| qwen-instruct\|physionet2019 | alarm\|A | 3252 | 0 | 0.989 | 0.151 | C0 0.61, C2 0.63, C4 0.46, CP 0.58, CQ2 0.63, CQ4 0.48 | C2 0.75, C4 0.68, CQ2 0.69, CQ4 0.72 |
| qwen-instruct\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.990 | 0.072 | C2 0.61, C4 0.57 | C2 0.68, C4 0.47 |
| qwen-instruct\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.990 | 0.077 | C2 0.62, C4 0.56 | C2 0.67, C4 0.48 |
| qwen-instruct\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.991 | 0.079 | C2 0.61, C4 0.55 | C2 0.68, C4 0.50 |
| qwen-instruct\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 1.000 | 0.613 | C0 0.57, C2 0.61, C4 0.44, CP 0.57 | C2 0.80, C4 0.66 |
| qwen-instruct\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 1.000 | 0.438 | C0 0.59, C2 0.63, C4 0.46, CP 0.57 | C2 0.77, C4 0.68 |
| qwen-instruct\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.999 | 0.103 | C0 0.61, C2 0.64, C4 0.47, CP 0.56 | C2 0.79, C4 0.69 |
| qwen-instruct\|physionet2019 | para_reversed\|A | 2168 | 0 | 1.000 | 0.025 | C0 0.60, C2 0.63, C4 0.49, CP 0.59 | C2 0.61, C4 0.70 |
| qwen-instruct\|physionet2019 | reversed\|A | 3252 | 0 | 0.999 | 0.013 | C0 0.60, C2 0.63, C4 0.49, CP 0.58, CQ2 0.63, CQ4 0.47 | C2 0.62, C4 0.71, CQ2 0.64, CQ4 0.73 |
| qwen-instruct\|physionet2019 | tf_alarm\|A | 2168 | 0 | 0.998 | 0.000 | C0 0.60, C2 0.57, C4 0.45, CP 0.51 | C2 0.78, C4 0.54 |
| qwen-instruct\|physionet2019 | tf_reversed\|A | 2168 | 0 | 0.998 | 0.002 | C0 0.60, C2 0.63, C4 0.46, CP 0.56 | C2 0.72, C4 0.75 |
| qwen3-instruct\|physionet2012 | alarm\|A | 3600 | 0 | 0.901 | 0.214 | C0 0.67, C2 0.69, C4 0.56, CP 0.53, CQ2 0.76, CQ4 0.32 | C2 0.81, C4 0.36, CQ2 0.99, CQ4 0.90 |
| qwen3-instruct\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.896 | 0.188 | C2 0.66, C4 0.61 | C2 0.77, C4 0.30 |
| qwen3-instruct\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.893 | 0.176 | C2 0.66, C4 0.60 | C2 0.77, C4 0.30 |
| qwen3-instruct\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.890 | 0.176 | C2 0.67, C4 0.59 | C2 0.78, C4 0.30 |
| qwen3-instruct\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 0.637 | 0.369 | C0 0.71, C2 0.74, C4 0.54, CP 0.57 | C2 0.89, C4 0.42 |
| qwen3-instruct\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 0.392 | 0.236 | C0 0.73, C2 0.76, C4 0.59, CP 0.62 | C2 0.89, C4 0.38 |
| qwen3-instruct\|physionet2012 | para_alarm\|A | 2400 | 0 | 0.630 | 0.122 | C0 0.73, C2 0.73, C4 0.61, CP 0.61 | C2 0.88, C4 0.35 |
| qwen3-instruct\|physionet2012 | para_reversed\|A | 2400 | 0 | 0.783 | 0.022 | C0 0.69, C2 0.76, C4 0.58, CP 0.73 | C2 0.86, C4 0.47 |
| qwen3-instruct\|physionet2012 | reversed\|A | 3600 | 0 | 0.952 | 0.002 | C0 0.66, C2 0.74, C4 0.56, CP 0.72, CQ2 0.78, CQ4 0.33 | C2 0.83, C4 0.48, CQ2 0.91, CQ4 0.80 |
| qwen3-instruct\|physionet2019 | alarm\|A | 3252 | 0 | 0.686 | 0.119 | C0 0.63, C2 0.64, C4 0.57, CP 0.57, CQ2 0.68, CQ4 0.43 | C2 0.74, C4 0.53, CQ2 0.85, CQ4 0.87 |
| qwen3-instruct\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.690 | 0.020 | C2 0.63, C4 0.63 | C2 0.69, C4 0.39 |
| qwen3-instruct\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.679 | 0.014 | C2 0.63, C4 0.62 | C2 0.70, C4 0.40 |
| qwen3-instruct\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.658 | 0.010 | C2 0.63, C4 0.62 | C2 0.69, C4 0.40 |
| qwen3-instruct\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 0.864 | 0.387 | C0 0.62, C2 0.63, C4 0.56, CP 0.56 | C2 0.75, C4 0.49 |
| qwen3-instruct\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 0.768 | 0.168 | C0 0.63, C2 0.63, C4 0.56, CP 0.59 | C2 0.74, C4 0.52 |
| qwen3-instruct\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.562 | 0.110 | C0 0.62, C2 0.65, C4 0.55, CP 0.58 | C2 0.77, C4 0.61 |
| qwen3-instruct\|physionet2019 | para_reversed\|A | 2168 | 0 | 0.322 | 0.000 | C0 0.54, C2 0.63, C4 0.50, CP 0.63 | C2 0.56, C4 0.67 |
| qwen3-instruct\|physionet2019 | reversed\|A | 3252 | 0 | 0.547 | 0.000 | C0 0.59, C2 0.66, C4 0.51, CP 0.63, CQ2 0.65, CQ4 0.50 | C2 0.65, C4 0.68, CQ2 0.59, CQ4 0.70 |
| yi15-9b\|physionet2012 | alarm\|A | 3600 | 0 | 0.999 | 0.000 | C0 0.63, C2 0.68, C4 0.58, CP 0.61, CQ2 0.65, CQ4 0.47 | C2 0.81, C4 0.30, CQ2 0.76, CQ4 0.49 |
| yi15-9b\|physionet2012 | alarm\|Anum05 | 1200 | 0 | 0.999 | 0.000 | C2 0.67, C4 0.67 | C2 0.81, C4 0.20 |
| yi15-9b\|physionet2012 | alarm\|Anum1 | 1200 | 0 | 0.999 | 0.000 | C2 0.67, C4 0.68 | C2 0.81, C4 0.20 |
| yi15-9b\|physionet2012 | alarm\|Anum2 | 1200 | 0 | 0.999 | 0.000 | C2 0.68, C4 0.68 | C2 0.81, C4 0.21 |
| yi15-9b\|physionet2012 | choice_alarm_A\|A | 2400 | 0 | 0.999 | 0.000 | C0 0.66, C2 0.74, C4 0.55, CP 0.64 | C2 0.87, C4 0.49 |
| yi15-9b\|physionet2012 | choice_safe_A\|A | 2400 | 0 | 0.999 | 0.000 | C0 0.69, C2 0.72, C4 0.55, CP 0.62 | C2 0.84, C4 0.51 |
| yi15-9b\|physionet2012 | para_alarm\|A | 2400 | 0 | 0.999 | 0.000 | C0 0.61, C2 0.66, C4 0.56, CP 0.58 | C2 0.77, C4 0.33 |
| yi15-9b\|physionet2012 | para_reversed\|A | 2400 | 0 | 0.997 | 0.000 | C0 0.69, C2 0.75, C4 0.58, CP 0.71 | C2 0.86, C4 0.42 |
| yi15-9b\|physionet2012 | reversed\|A | 3600 | 0 | 0.998 | 0.000 | C0 0.70, C2 0.74, C4 0.61, CP 0.71, CQ2 0.76, CQ4 0.44 | C2 0.85, C4 0.36, CQ2 0.87, CQ4 0.66 |
| yi15-9b\|physionet2012 | tf_alarm\|A | 2400 | 0 | 1.000 | 0.000 | C0 0.59, C2 0.58, C4 0.50, CP 0.51 | C2 0.67, C4 0.41 |
| yi15-9b\|physionet2012 | tf_reversed\|A | 2400 | 0 | 0.999 | 0.000 | C0 0.66, C2 0.73, C4 0.52, CP 0.65 | C2 0.81, C4 0.57 |
| yi15-9b\|physionet2019 | alarm\|A | 3252 | 0 | 0.998 | 0.000 | C0 0.60, C2 0.64, C4 0.43, CP 0.51, CQ2 0.66, CQ4 0.46 | C2 0.82, C4 0.77, CQ2 0.76, CQ4 0.78 |
| yi15-9b\|physionet2019 | alarm\|Anum05 | 1084 | 0 | 0.999 | 0.000 | C2 0.61, C4 0.57 | C2 0.57, C4 0.53 |
| yi15-9b\|physionet2019 | alarm\|Anum1 | 1084 | 0 | 0.999 | 0.000 | C2 0.61, C4 0.56 | C2 0.57, C4 0.54 |
| yi15-9b\|physionet2019 | alarm\|Anum2 | 1084 | 0 | 0.999 | 0.000 | C2 0.61, C4 0.56 | C2 0.57, C4 0.53 |
| yi15-9b\|physionet2019 | choice_alarm_A\|A | 2168 | 0 | 1.000 | 0.000 | C0 0.59, C2 0.64, C4 0.44, CP 0.56 | C2 0.73, C4 0.74 |
| yi15-9b\|physionet2019 | choice_safe_A\|A | 2168 | 0 | 1.000 | 0.000 | C0 0.55, C2 0.66, C4 0.47, CP 0.53 | C2 0.69, C4 0.72 |
| yi15-9b\|physionet2019 | para_alarm\|A | 2168 | 0 | 0.997 | 0.000 | C0 0.61, C2 0.63, C4 0.49, CP 0.52 | C2 0.77, C4 0.64 |
| yi15-9b\|physionet2019 | para_reversed\|A | 2168 | 0 | 0.998 | 0.000 | C0 0.57, C2 0.64, C4 0.46, CP 0.58 | C2 0.65, C4 0.67 |
| yi15-9b\|physionet2019 | reversed\|A | 3252 | 0 | 0.998 | 0.000 | C0 0.56, C2 0.61, C4 0.48, CP 0.56, CQ2 0.65, CQ4 0.49 | C2 0.68, C4 0.63, CQ2 0.59, CQ4 0.72 |
| yi15-9b\|physionet2019 | tf_alarm\|A | 2168 | 0 | 1.000 | 0.000 | C0 0.58, C2 0.59, C4 0.51, CP 0.50 | C2 0.78, C4 0.56 |
| yi15-9b\|physionet2019 | tf_reversed\|A | 2168 | 0 | 1.000 | 0.000 | C0 0.50, C2 0.57, C4 0.49, CP 0.61 | C2 0.47, C4 0.67 |

**f. Imputation bound.** Share of records with an imputed variant: falcon3-7b 0.026, granite3-instruct 0.046, llama-instruct 0.020, olmo3-instruct 0.005, qwen-instruct 0.206, qwen3-instruct 0.131, yi15-9b 0.000.
Under the bound: R15 not supported, R16 not supported (3 of 4). Cells whose decision changes: none.

## Q22. With the template variance of the fourth study

| R16 cell | predicted | p (template) | Holm p | supported | opposite |
|---|---|---|---|---|---|
| falcon3-7b\|physionet2012 | + | <0.001 | <0.001 | yes | no |
| falcon3-7b\|physionet2019 | + | <0.001 | <0.001 | yes | no |
| yi15-9b\|physionet2019 | + | 0.003 | 0.007 | yes | no |
| granite3-instruct\|physionet2012 | − | 0.426 | 0.426 | no | no |

R16 with this test: 3 of 4 cells supported; it would be supported (no decision rule).

| pair | baseline | reading | part | conclusive | stay conclusive |
|---|---|---|
| yn\|C0\|threshold\|W | 11 | 4 |
| yn\|C0\|threshold\|O | 9 | 6 |
| yn\|C0\|neutral\|W | 10 | 7 |
| yn\|C0\|neutral\|O | 12 | 6 |
| yn\|CP\|threshold\|W | 10 | 3 |
| yn\|CP\|threshold\|O | 8 | 6 |
| yn\|CP\|neutral\|W | 11 | 7 |
| yn\|CP\|neutral\|O | 11 | 8 |
| pa\|C0\|threshold\|W | 5 | 2 |
| pa\|C0\|threshold\|O | 7 | 5 |
| pa\|C0\|neutral\|W | 8 | 2 |
| pa\|C0\|neutral\|O | 9 | 7 |
| pa\|CP\|threshold\|W | 9 | 6 |
| pa\|CP\|threshold\|O | 9 | 8 |
| pa\|CP\|neutral\|W | 10 | 7 |
| pa\|CP\|neutral\|O | 10 | 9 |
| tf\|C0\|threshold\|W | 8 | 7 |
| tf\|C0\|threshold\|O | 10 | 5 |
| tf\|C0\|neutral\|W | 9 | 5 |
| tf\|C0\|neutral\|O | 7 | 7 |
| tf\|CP\|threshold\|W | 3 | 0 |
| tf\|CP\|threshold\|O | 4 | 3 |
| tf\|CP\|neutral\|W | 7 | 3 |
| tf\|CP\|neutral\|O | 8 | 6 |
| ab\|C0\|threshold\|W | 9 | 2 |
| ab\|C0\|threshold\|O | 9 | 4 |
| ab\|C0\|neutral\|W | 10 | 5 |
| ab\|C0\|neutral\|O | 9 | 6 |
| ab\|CP\|threshold\|W | 8 | 4 |
| ab\|CP\|threshold\|O | 7 | 7 |
| ab\|CP\|neutral\|W | 9 | 5 |
| ab\|CP\|neutral\|O | 7 | 7 |

