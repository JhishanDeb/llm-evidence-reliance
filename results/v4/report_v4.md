# Fourth study: results (`prereg-v4`)

Runs: falcon3-7b, llama-base, llama-instruct, olmo3-instruct, qwen-base, qwen-instruct, yi15-9b. Missing registered runs: none. R10 families: llama, qwen. Bootstrap resamples: 10000.

## Decisions

- **R8** (W with True/False): removed (prediction: kept); O kept.
- **R9** (W with the negated question): removed (prediction: kept); O kept.
- **Joint reading:** the Yes/No question as worded.
- **R10** (post-training with the format fixed): not supported; not separated by this study.
- **R11** (example order): W kept, O kept; the outcome part is not produced by the outcome of the last solved example.
- **R12** (paraphrase): W not settled, O not settled.
- **R13** (words over numbers in one sentence): supported.

## Validity

| cell | mass alarm | mass reversed | C0 acc. | valid | tf valid | neg valid | order valid | para valid | aggregate valid | plain valid |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-instruct|physionet2012 | 0.986 | 0.988 | 0.607 | yes | yes | yes | yes | yes | yes | yes |
| llama-instruct|physionet2019 | 0.994 | 0.995 | 0.618 | yes | yes | yes | yes | yes | yes | yes |
| llama-instruct|diabetes130 | 0.990 | 0.994 | 0.557 | yes | yes | yes | yes | yes | yes | yes |
| qwen-instruct|physionet2012 | 0.999 | 1.000 | 0.692 | yes | yes | yes | yes | yes | yes | yes |
| qwen-instruct|physionet2019 | 0.994 | 0.999 | 0.608 | yes | yes | yes | yes | yes | yes | yes |
| qwen-instruct|diabetes130 | 0.989 | 0.998 | 0.528 | no | no | no | no | no | no | no |
| olmo3-instruct|physionet2012 | 0.997 | 0.994 | 0.612 | yes | yes | yes | yes | yes | yes | – |
| olmo3-instruct|physionet2019 | 0.997 | 0.997 | 0.597 | yes | yes | yes | yes | yes | yes | – |
| olmo3-instruct|diabetes130 | 0.998 | 0.997 | 0.540 | no | no | no | no | no | no | – |
| falcon3-7b|physionet2012 | 0.999 | 0.999 | 0.670 | yes | yes | yes | yes | yes | yes | – |
| falcon3-7b|physionet2019 | 0.999 | 0.999 | 0.620 | yes | yes | yes | yes | no | yes | – |
| falcon3-7b|diabetes130 | 0.998 | 0.999 | 0.602 | yes | yes | yes | yes | yes | yes | – |
| yi15-9b|physionet2012 | 0.999 | 0.998 | 0.607 | yes | yes | yes | yes | yes | yes | – |
| yi15-9b|physionet2019 | 0.999 | 0.998 | 0.587 | yes | yes | yes | yes | yes | yes | – |
| yi15-9b|diabetes130 | 0.999 | 0.999 | 0.582 | yes | yes | yes | yes | yes | yes | – |

**Validity of each new framing** (mass, movement, C0 Spearman correlation with the registered alarm question):

| cell | framing / set | mass | movement | C0 Spearman | valid |
|---|---|---|---|---|---|
| llama-instruct|physionet2012 | alarm/Asf | 0.989 | +1.30 | +0.96 | yes |
| llama-instruct|physionet2012 | alarm/Acue | 0.986 | – | – | yes |
| llama-instruct|physionet2012 | alarm/Anet | 0.986 | – | – | yes |
| llama-instruct|physionet2012 | alarm/Abase | 0.892 | +0.79 | +0.98 | yes |
| llama-instruct|physionet2012 | reversed/Asf | 0.985 | +1.07 | +0.93 | yes |
| llama-instruct|physionet2012 | reversed/Abase | 0.878 | +0.72 | +0.96 | yes |
| llama-instruct|physionet2012 | tf_alarm/A | 0.989 | +1.04 | +0.98 | yes |
| llama-instruct|physionet2012 | tf_reversed/A | 0.984 | +1.43 | +0.95 | yes |
| llama-instruct|physionet2012 | negated/A | 0.989 | +1.24 | +0.97 | yes |
| llama-instruct|physionet2012 | para_alarm/A | 0.988 | +1.76 | +0.97 | yes |
| llama-instruct|physionet2012 | para_reversed/A | 0.990 | +1.77 | +0.95 | yes |
| llama-instruct|physionet2012 | split/Abase | 0.892 | – | – | yes |
| llama-instruct|physionet2019 | alarm/Asf | 0.995 | +1.42 | +0.86 | yes |
| llama-instruct|physionet2019 | alarm/Acue | 0.994 | – | – | yes |
| llama-instruct|physionet2019 | alarm/Anet | 0.993 | – | – | yes |
| llama-instruct|physionet2019 | alarm/Abase | 0.820 | +0.89 | +0.93 | yes |
| llama-instruct|physionet2019 | reversed/Asf | 0.994 | +0.82 | +0.79 | yes |
| llama-instruct|physionet2019 | reversed/Abase | 0.821 | +0.90 | +0.84 | yes |
| llama-instruct|physionet2019 | tf_alarm/A | 0.989 | +1.41 | +0.95 | yes |
| llama-instruct|physionet2019 | tf_reversed/A | 0.994 | +1.13 | +0.88 | yes |
| llama-instruct|physionet2019 | negated/A | 0.995 | +1.37 | +0.85 | yes |
| llama-instruct|physionet2019 | para_alarm/A | 0.993 | +1.66 | +0.92 | yes |
| llama-instruct|physionet2019 | para_reversed/A | 0.994 | +1.23 | +0.83 | yes |
| llama-instruct|physionet2019 | split/Abase | 0.820 | – | – | no |
| llama-instruct|diabetes130 | alarm/Asf | 0.991 | +3.03 | +0.88 | yes |
| llama-instruct|diabetes130 | alarm/Acue | 0.992 | – | – | yes |
| llama-instruct|diabetes130 | alarm/Anet | 0.986 | – | – | yes |
| llama-instruct|diabetes130 | alarm/Abase | 0.780 | +0.85 | +0.93 | yes |
| llama-instruct|diabetes130 | reversed/Asf | 0.992 | +3.23 | +0.77 | yes |
| llama-instruct|diabetes130 | reversed/Abase | 0.845 | +1.97 | +0.82 | yes |
| llama-instruct|diabetes130 | tf_alarm/A | 0.986 | +2.67 | +0.95 | yes |
| llama-instruct|diabetes130 | tf_reversed/A | 0.984 | +2.62 | +0.72 | yes |
| llama-instruct|diabetes130 | negated/A | 0.993 | +2.06 | +0.83 | yes |
| llama-instruct|diabetes130 | para_alarm/A | 0.992 | +3.37 | +0.78 | yes |
| llama-instruct|diabetes130 | para_reversed/A | 0.996 | +3.25 | +0.86 | yes |
| llama-instruct|diabetes130 | split/Abase | 0.780 | – | – | no |
| qwen-instruct|physionet2012 | alarm/Asf | 1.000 | +5.27 | +0.94 | yes |
| qwen-instruct|physionet2012 | alarm/Acue | 0.997 | – | – | yes |
| qwen-instruct|physionet2012 | alarm/Anet | 0.998 | – | – | yes |
| qwen-instruct|physionet2012 | alarm/Abase | 0.985 | +2.77 | +0.98 | yes |
| qwen-instruct|physionet2012 | reversed/Asf | 1.000 | +4.40 | +0.92 | yes |
| qwen-instruct|physionet2012 | reversed/Abase | 0.994 | +2.05 | +0.94 | yes |
| qwen-instruct|physionet2012 | tf_alarm/A | 0.999 | +6.39 | +0.94 | yes |
| qwen-instruct|physionet2012 | tf_reversed/A | 1.000 | +5.22 | +0.90 | yes |
| qwen-instruct|physionet2012 | negated/A | 1.000 | +4.60 | +0.95 | yes |
| qwen-instruct|physionet2012 | para_alarm/A | 1.000 | +6.90 | +0.94 | yes |
| qwen-instruct|physionet2012 | para_reversed/A | 1.000 | +5.84 | +0.92 | yes |
| qwen-instruct|physionet2012 | split/Abase | 0.985 | – | – | yes |
| qwen-instruct|physionet2019 | alarm/Asf | 0.996 | +9.05 | +0.92 | yes |
| qwen-instruct|physionet2019 | alarm/Acue | 0.999 | – | – | yes |
| qwen-instruct|physionet2019 | alarm/Anet | 0.971 | – | – | yes |
| qwen-instruct|physionet2019 | alarm/Abase | 0.971 | +4.38 | +0.98 | yes |
| qwen-instruct|physionet2019 | reversed/Asf | 0.997 | +4.69 | +0.77 | yes |
| qwen-instruct|physionet2019 | reversed/Abase | 0.993 | +2.33 | +0.90 | yes |
| qwen-instruct|physionet2019 | tf_alarm/A | 0.996 | +7.20 | +0.94 | yes |
| qwen-instruct|physionet2019 | tf_reversed/A | 0.997 | +7.22 | +0.88 | yes |
| qwen-instruct|physionet2019 | negated/A | 1.000 | +3.81 | +0.91 | yes |
| qwen-instruct|physionet2019 | para_alarm/A | 0.998 | +9.18 | +0.96 | yes |
| qwen-instruct|physionet2019 | para_reversed/A | 1.000 | +5.47 | +0.85 | yes |
| qwen-instruct|physionet2019 | split/Abase | 0.971 | – | – | yes |
| qwen-instruct|diabetes130 | alarm/Asf | 0.988 | +9.05 | +0.85 | no |
| qwen-instruct|diabetes130 | alarm/Acue | 0.999 | – | – | no |
| qwen-instruct|diabetes130 | alarm/Anet | 0.992 | – | – | no |
| qwen-instruct|diabetes130 | alarm/Abase | 0.919 | +5.10 | +0.98 | no |
| qwen-instruct|diabetes130 | reversed/Asf | 0.994 | +6.19 | +0.41 | no |
| qwen-instruct|diabetes130 | reversed/Abase | 0.957 | +3.47 | +0.76 | no |
| qwen-instruct|diabetes130 | tf_alarm/A | 0.999 | +7.80 | +0.91 | no |
| qwen-instruct|diabetes130 | tf_reversed/A | 1.000 | +7.44 | +0.49 | no |
| qwen-instruct|diabetes130 | negated/A | 1.000 | +4.48 | +0.78 | no |
| qwen-instruct|diabetes130 | para_alarm/A | 1.000 | +10.36 | +0.79 | no |
| qwen-instruct|diabetes130 | para_reversed/A | 1.000 | +5.98 | +0.61 | no |
| qwen-instruct|diabetes130 | split/Abase | 0.919 | – | – | no |
| olmo3-instruct|physionet2012 | alarm/Asf | 0.997 | +1.51 | +0.97 | yes |
| olmo3-instruct|physionet2012 | alarm/Acue | 0.998 | – | – | yes |
| olmo3-instruct|physionet2012 | alarm/Anet | 0.998 | – | – | yes |
| olmo3-instruct|physionet2012 | reversed/Asf | 0.995 | +1.27 | +0.90 | yes |
| olmo3-instruct|physionet2012 | tf_alarm/A | 0.999 | +1.78 | +0.96 | yes |
| olmo3-instruct|physionet2012 | tf_reversed/A | 0.998 | +1.56 | +0.93 | yes |
| olmo3-instruct|physionet2012 | negated/A | 0.998 | +0.77 | +0.91 | yes |
| olmo3-instruct|physionet2012 | para_alarm/A | 0.997 | +1.51 | +0.93 | yes |
| olmo3-instruct|physionet2012 | para_reversed/A | 0.995 | +1.45 | +0.88 | yes |
| olmo3-instruct|physionet2019 | alarm/Asf | 0.997 | +2.39 | +0.87 | yes |
| olmo3-instruct|physionet2019 | alarm/Acue | 0.999 | – | – | yes |
| olmo3-instruct|physionet2019 | alarm/Anet | 0.999 | – | – | yes |
| olmo3-instruct|physionet2019 | reversed/Asf | 0.996 | +2.66 | +0.72 | yes |
| olmo3-instruct|physionet2019 | tf_alarm/A | 0.999 | +1.72 | +0.92 | yes |
| olmo3-instruct|physionet2019 | tf_reversed/A | 0.999 | +2.51 | +0.96 | yes |
| olmo3-instruct|physionet2019 | negated/A | 0.999 | +1.73 | +0.73 | yes |
| olmo3-instruct|physionet2019 | para_alarm/A | 0.997 | +2.37 | +0.97 | yes |
| olmo3-instruct|physionet2019 | para_reversed/A | 0.997 | +2.36 | +0.93 | yes |
| olmo3-instruct|diabetes130 | alarm/Asf | 0.998 | +3.42 | +0.73 | no |
| olmo3-instruct|diabetes130 | alarm/Acue | 0.999 | – | – | no |
| olmo3-instruct|diabetes130 | alarm/Anet | 0.999 | – | – | no |
| olmo3-instruct|diabetes130 | reversed/Asf | 0.997 | +3.80 | +0.46 | no |
| olmo3-instruct|diabetes130 | tf_alarm/A | 0.998 | +2.44 | +0.87 | no |
| olmo3-instruct|diabetes130 | tf_reversed/A | 0.999 | +3.53 | +0.88 | no |
| olmo3-instruct|diabetes130 | negated/A | 0.998 | +3.02 | +0.85 | no |
| olmo3-instruct|diabetes130 | para_alarm/A | 0.998 | +2.84 | +0.78 | no |
| olmo3-instruct|diabetes130 | para_reversed/A | 0.999 | +2.47 | +0.62 | no |
| falcon3-7b|physionet2012 | alarm/Asf | 0.999 | +3.01 | +0.97 | yes |
| falcon3-7b|physionet2012 | alarm/Acue | 0.999 | – | – | yes |
| falcon3-7b|physionet2012 | alarm/Anet | 0.999 | – | – | yes |
| falcon3-7b|physionet2012 | reversed/Asf | 0.999 | +1.68 | +0.96 | yes |
| falcon3-7b|physionet2012 | tf_alarm/A | 0.999 | +2.45 | +0.98 | yes |
| falcon3-7b|physionet2012 | tf_reversed/A | 0.999 | +1.89 | +0.97 | yes |
| falcon3-7b|physionet2012 | negated/A | 0.999 | +1.37 | +0.98 | yes |
| falcon3-7b|physionet2012 | para_alarm/A | 0.998 | +3.84 | +0.99 | yes |
| falcon3-7b|physionet2012 | para_reversed/A | 0.998 | +1.49 | +0.97 | yes |
| falcon3-7b|physionet2019 | alarm/Asf | 0.999 | +4.62 | +0.98 | yes |
| falcon3-7b|physionet2019 | alarm/Acue | 0.999 | – | – | yes |
| falcon3-7b|physionet2019 | alarm/Anet | 0.999 | – | – | yes |
| falcon3-7b|physionet2019 | reversed/Asf | 0.999 | +2.14 | +0.91 | yes |
| falcon3-7b|physionet2019 | tf_alarm/A | 0.999 | +4.17 | +0.98 | yes |
| falcon3-7b|physionet2019 | tf_reversed/A | 0.999 | +2.80 | +0.93 | yes |
| falcon3-7b|physionet2019 | negated/A | 0.999 | +1.09 | +0.95 | yes |
| falcon3-7b|physionet2019 | para_alarm/A | 0.999 | +4.67 | +0.98 | yes |
| falcon3-7b|physionet2019 | para_reversed/A | 0.998 | -0.61 | +0.81 | no |
| falcon3-7b|diabetes130 | alarm/Asf | 0.998 | +4.09 | +0.91 | yes |
| falcon3-7b|diabetes130 | alarm/Acue | 0.999 | – | – | yes |
| falcon3-7b|diabetes130 | alarm/Anet | 0.998 | – | – | yes |
| falcon3-7b|diabetes130 | reversed/Asf | 0.999 | +3.82 | +0.87 | yes |
| falcon3-7b|diabetes130 | tf_alarm/A | 0.999 | +4.63 | +0.95 | yes |
| falcon3-7b|diabetes130 | tf_reversed/A | 0.999 | +4.09 | +0.92 | yes |
| falcon3-7b|diabetes130 | negated/A | 0.998 | +2.84 | +0.93 | yes |
| falcon3-7b|diabetes130 | para_alarm/A | 0.999 | +3.95 | +0.54 | yes |
| falcon3-7b|diabetes130 | para_reversed/A | 0.999 | +0.82 | +0.87 | yes |
| yi15-9b|physionet2012 | alarm/Asf | 0.999 | +1.48 | +0.97 | yes |
| yi15-9b|physionet2012 | alarm/Acue | 1.000 | – | – | yes |
| yi15-9b|physionet2012 | alarm/Anet | 1.000 | – | – | yes |
| yi15-9b|physionet2012 | reversed/Asf | 0.999 | +1.79 | +0.93 | yes |
| yi15-9b|physionet2012 | tf_alarm/A | 1.000 | +1.02 | +0.96 | yes |
| yi15-9b|physionet2012 | tf_reversed/A | 0.999 | +1.00 | +0.94 | yes |
| yi15-9b|physionet2012 | negated/A | 0.999 | +1.48 | +0.96 | yes |
| yi15-9b|physionet2012 | para_alarm/A | 0.999 | +1.53 | +0.97 | yes |
| yi15-9b|physionet2012 | para_reversed/A | 0.998 | +1.87 | +0.96 | yes |
| yi15-9b|physionet2019 | alarm/Asf | 0.999 | +2.12 | +0.77 | yes |
| yi15-9b|physionet2019 | alarm/Acue | 1.000 | – | – | yes |
| yi15-9b|physionet2019 | alarm/Anet | 0.999 | – | – | yes |
| yi15-9b|physionet2019 | reversed/Asf | 0.999 | +0.89 | +0.66 | yes |
| yi15-9b|physionet2019 | tf_alarm/A | 1.000 | +1.20 | +0.94 | yes |
| yi15-9b|physionet2019 | tf_reversed/A | 1.000 | +0.86 | +0.66 | yes |
| yi15-9b|physionet2019 | negated/A | 0.999 | +0.84 | +0.73 | yes |
| yi15-9b|physionet2019 | para_alarm/A | 0.998 | +1.20 | +0.90 | yes |
| yi15-9b|physionet2019 | para_reversed/A | 0.998 | +1.44 | +0.68 | yes |
| yi15-9b|diabetes130 | alarm/Asf | 0.999 | +2.39 | +0.87 | yes |
| yi15-9b|diabetes130 | alarm/Acue | 1.000 | – | – | yes |
| yi15-9b|diabetes130 | alarm/Anet | 0.999 | – | – | yes |
| yi15-9b|diabetes130 | reversed/Asf | 0.999 | +1.70 | +0.74 | yes |
| yi15-9b|diabetes130 | tf_alarm/A | 0.999 | +1.64 | +0.79 | yes |
| yi15-9b|diabetes130 | tf_reversed/A | 0.999 | +1.38 | +0.23 | yes |
| yi15-9b|diabetes130 | negated/A | 0.999 | +1.23 | +0.55 | yes |
| yi15-9b|diabetes130 | para_alarm/A | 0.998 | +2.60 | +0.73 | yes |
| yi15-9b|diabetes130 | para_reversed/A | 0.999 | +2.87 | +0.58 | yes |

## O and W with the registered pair

O = (s_first + s_second) / 2 and W = (s_first − s_second) / 2, for s = a and s = a′ (95% intervals); W < 0 is a pull toward the second answer word. p is the intersection–union p (no correction).

| cell | valid | O (a) | O (a′) | p | W (a) | W (a′) | p | movement |
|---|---|---|---|---|---|---|---|---|
| llama-instruct|physionet2012 | yes | -0.27 [-0.43, -0.11] | -0.24 [-0.34, -0.14] | 0.001 | -0.32 [-0.47, -0.15] | -0.37 [-0.44, -0.31] | <0.001 | +1.31 |
| llama-instruct|physionet2019 | yes | -0.77 [-1.29, -0.26] | -0.56 [-0.65, -0.47] | 0.003 | +0.22 [-0.30, +0.72] | -0.06 [-0.14, +0.01] | 0.459 | +1.25 |
| llama-instruct|diabetes130 | yes | +0.34 [-0.72, +1.86] | -1.02 [-1.23, -0.82] | 0.600 | -2.43 [-3.91, -1.36] | -0.91 [-1.09, -0.72] | <0.001 | +2.83 |
| qwen-instruct|physionet2012 | yes | -3.43 [-4.04, -2.85] | -4.35 [-4.88, -3.81] | <0.001 | +1.73 [+1.32, +2.16] | +1.79 [+1.49, +2.08] | <0.001 | +4.76 |
| qwen-instruct|physionet2019 | yes | +0.59 [-0.65, +1.87] | +0.20 [-0.12, +0.51] | 0.329 | -2.87 [-4.01, -1.87] | -2.90 [-3.13, -2.69] | <0.001 | +7.30 |
| qwen-instruct|diabetes130 | no | -4.96 [-8.30, -2.03] | -2.55 [-2.87, -2.22] | <0.001 | +2.75 [+0.03, +5.85] | -1.45 [-1.74, -1.16] | 0.047 | +8.43 |
| olmo3-instruct|physionet2012 | yes | +2.33 [+1.93, +2.68] | +1.46 [+1.32, +1.59] | <0.001 | -0.82 [-1.18, -0.52] | -1.56 [-1.63, -1.50] | <0.001 | +1.41 |
| olmo3-instruct|physionet2019 | yes | -0.24 [-0.78, +0.33] | +0.01 [-0.11, +0.13] | 0.854 | -1.63 [-2.22, -1.14] | -1.34 [-1.45, -1.24] | <0.001 | +2.53 |
| olmo3-instruct|diabetes130 | no | -1.90 [-15.00, +10.78] | +1.60 [+1.34, +1.87] | 0.669 | +0.82 [-12.03, +13.75] | -2.20 [-2.43, -1.97] | 0.804 | +3.24 |
| falcon3-7b|physionet2012 | yes | +0.47 [+0.23, +0.71] | +0.16 [+0.00, +0.32] | 0.049 | +0.66 [+0.56, +0.77] | +0.63 [+0.55, +0.71] | <0.001 | +2.16 |
| falcon3-7b|physionet2019 | yes | +1.67 [+1.39, +1.86] | +1.42 [+1.32, +1.52] | <0.001 | +1.41 [+1.29, +1.53] | +1.30 [+1.23, +1.36] | <0.001 | +2.34 |
| falcon3-7b|diabetes130 | yes | +1.79 [+1.59, +1.98] | +2.08 [+1.98, +2.18] | <0.001 | +1.07 [+0.95, +1.19] | +1.08 [+1.01, +1.15] | <0.001 | +3.32 |
| yi15-9b|physionet2012 | yes | -0.16 [-0.29, -0.03] | -0.32 [-0.43, -0.22] | 0.013 | +0.86 [+0.72, +1.00] | +0.63 [+0.54, +0.72] | <0.001 | +1.71 |
| yi15-9b|physionet2019 | yes | -1.49 [-1.85, -1.13] | -1.39 [-1.47, -1.32] | <0.001 | +0.70 [+0.39, +0.99] | +0.62 [+0.56, +0.69] | <0.001 | +1.40 |
| yi15-9b|diabetes130 | yes | -0.08 [-0.65, +0.47] | +0.08 [-0.02, +0.18] | 0.756 | -0.13 [-0.65, +0.44] | -0.21 [-0.30, -0.12] | 0.645 | +2.27 |

## R8. True/False answers

**True/False answers, W.** Supported in 4 of 11 eligible cells, reduced in 6, opposite in 5: **removed**.

| cell | sign | valid | W (a) | W (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | + | yes | -1.27 | -1.41 | <0.001 | <0.001 | no | -3.00 | -3.19 | <0.001 | yes | yes |
| qwen-instruct|physionet2019 | − | yes | -6.64 | -6.05 | <0.001 | <0.001 | yes | -3.77 | -3.15 | <0.001 | no | no |
| llama-instruct|physionet2012 | − | yes | -1.69 | -1.74 | <0.001 | <0.001 | yes | -1.37 | -1.37 | <0.001 | no | no |
| llama-instruct|diabetes130 | − | yes | -3.13 | -1.83 | 0.001 | 0.004 | yes | -0.71 | -0.92 | 0.294 | no | no |
| olmo3-instruct|physionet2012 | − | yes | -3.09 | -3.09 | <0.001 | <0.001 | yes | -2.27 | -1.53 | <0.001 | no | no |
| olmo3-instruct|physionet2019 | − | yes | +1.55 | -1.01 | 0.123 | 0.123 | no | +3.18 | +0.33 | 0.002 | yes | no |
| falcon3-7b|physionet2012 | + | yes | -1.47 | -1.50 | <0.001 | <0.001 | no | -2.13 | -2.13 | <0.001 | yes | yes |
| falcon3-7b|physionet2019 | + | yes | -1.45 | -1.51 | <0.001 | <0.001 | no | -2.87 | -2.80 | <0.001 | yes | yes |
| falcon3-7b|diabetes130 | + | yes | -1.85 | -2.32 | <0.001 | <0.001 | no | -2.92 | -3.40 | <0.001 | yes | yes |
| yi15-9b|physionet2012 | + | yes | -0.82 | -1.10 | <0.001 | <0.001 | no | -1.68 | -1.73 | <0.001 | yes | yes |
| yi15-9b|physionet2019 | + | yes | +3.32 | -0.62 | 0.012 | 0.024 | no | +2.62 | -1.24 | 0.087 | no | no |

**True/False answers, O.** Supported in 5 of 7 eligible cells, reduced in 2, opposite in 1: **kept**.

| cell | sign | valid | O (a) | O (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | − | yes | -5.35 | -6.19 | <0.001 | <0.001 | yes | -1.92 | -1.85 | <0.001 | no | no |
| llama-instruct|physionet2012 | − | yes | +1.10 | +1.23 | <0.001 | <0.001 | no | +1.37 | +1.47 | <0.001 | yes | yes |
| llama-instruct|physionet2019 | − | yes | -0.55 | +0.00 | 0.973 | 0.973 | no | +0.21 | +0.57 | 0.748 | no | no |
| olmo3-instruct|physionet2012 | + | yes | +3.08 | +2.59 | <0.001 | <0.001 | yes | +0.75 | +1.12 | <0.001 | no | no |
| falcon3-7b|physionet2019 | + | yes | +1.15 | +1.12 | <0.001 | <0.001 | yes | -0.52 | -0.31 | 0.020 | yes | no |
| falcon3-7b|diabetes130 | + | yes | +2.04 | +2.16 | <0.001 | <0.001 | yes | +0.25 | +0.08 | 0.092 | no | no |
| yi15-9b|physionet2019 | − | yes | -4.10 | -0.23 | 0.002 | 0.003 | yes | -2.61 | +1.16 | 0.092 | no | no |

## R9. The negated question

**The negated question, W.** Supported in 5 of 11 eligible cells, reduced in 6, opposite in 0: **removed**.

| cell | sign | valid | W (a) | W (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | + | yes | +0.96 | +0.28 | 0.063 | 0.254 | no | -0.77 | -1.50 | <0.001 | yes | no |
| qwen-instruct|physionet2019 | − | yes | -2.05 | -2.32 | <0.001 | <0.001 | yes | +0.82 | +0.59 | 0.081 | no | no |
| llama-instruct|physionet2012 | − | yes | -0.74 | -0.77 | <0.001 | <0.001 | yes | -0.42 | -0.40 | <0.001 | no | no |
| llama-instruct|diabetes130 | − | yes | -1.66 | -1.35 | <0.001 | 0.002 | yes | +0.76 | -0.44 | 0.163 | no | no |
| olmo3-instruct|physionet2012 | − | yes | +0.36 | -1.22 | 0.219 | 0.406 | no | +1.18 | +0.35 | <0.001 | yes | no |
| olmo3-instruct|physionet2019 | − | yes | +8.52 | -0.90 | 0.027 | 0.136 | no | +10.15 | +0.44 | 0.201 | no | no |
| falcon3-7b|physionet2012 | + | yes | +0.20 | +0.10 | 0.015 | 0.091 | no | -0.46 | -0.53 | <0.001 | yes | no |
| falcon3-7b|physionet2019 | + | yes | +0.71 | +0.47 | 0.001 | 0.007 | yes | -0.71 | -0.82 | 0.008 | yes | no |
| falcon3-7b|diabetes130 | + | yes | +0.52 | +0.40 | <0.001 | <0.001 | yes | -0.56 | -0.68 | <0.001 | yes | no |
| yi15-9b|physionet2012 | + | yes | +0.11 | -0.16 | 0.126 | 0.379 | no | -0.75 | -0.79 | <0.001 | yes | no |
| yi15-9b|physionet2019 | + | yes | +0.87 | +0.04 | 0.203 | 0.406 | no | +0.17 | -0.58 | 0.597 | no | no |

**The negated question, O.** Supported in 5 of 7 eligible cells, reduced in 4, opposite in 2: **kept**.

| cell | sign | valid | O (a) | O (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | − | yes | -2.66 | -2.84 | <0.001 | <0.001 | yes | +0.77 | +1.50 | <0.001 | yes | no |
| llama-instruct|physionet2012 | − | yes | +0.15 | +0.16 | 0.034 | 0.043 | no | +0.42 | +0.40 | <0.001 | yes | yes |
| llama-instruct|physionet2019 | − | yes | +0.44 | +0.23 | 0.022 | 0.043 | no | +1.21 | +0.80 | <0.001 | yes | yes |
| olmo3-instruct|physionet2012 | + | yes | +1.15 | +1.12 | <0.001 | <0.001 | yes | -1.18 | -0.35 | <0.001 | yes | no |
| falcon3-7b|physionet2019 | + | yes | +2.38 | +2.25 | <0.001 | <0.001 | yes | +0.71 | +0.82 | 0.003 | no | no |
| falcon3-7b|diabetes130 | + | yes | +2.34 | +2.76 | <0.001 | <0.001 | yes | +0.56 | +0.68 | <0.001 | no | no |
| yi15-9b|physionet2019 | − | yes | -1.66 | -0.81 | <0.001 | <0.001 | yes | -0.17 | +0.58 | 0.597 | no | no |

## R11. Alarming-last example order

**Alarming-last example order, W.** Supported in 7 of 11 eligible cells, reduced in 6, opposite in 1: **kept**.

| cell | sign | valid | W (a) | W (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | + | yes | -0.93 | -1.02 | <0.001 | 0.001 | no | -2.66 | -2.80 | <0.001 | yes | yes |
| qwen-instruct|physionet2019 | − | yes | -1.74 | -1.99 | <0.001 | 0.002 | yes | +1.13 | +0.91 | 0.046 | yes | no |
| llama-instruct|physionet2012 | − | yes | -0.19 | -0.22 | 0.037 | 0.074 | no | +0.13 | +0.15 | 0.005 | yes | no |
| llama-instruct|diabetes130 | − | yes | -0.25 | -0.54 | 0.145 | 0.145 | no | +2.17 | +0.37 | <0.001 | yes | no |
| olmo3-instruct|physionet2012 | − | yes | -1.01 | -1.64 | <0.001 | <0.001 | yes | -0.19 | -0.07 | 0.115 | no | no |
| olmo3-instruct|physionet2019 | − | yes | -2.81 | -1.64 | <0.001 | <0.001 | yes | -1.18 | -0.30 | 0.046 | no | no |
| falcon3-7b|physionet2012 | + | yes | +0.18 | +0.19 | <0.001 | <0.001 | yes | -0.49 | -0.44 | <0.001 | yes | no |
| falcon3-7b|physionet2019 | + | yes | +0.92 | +0.86 | <0.001 | <0.001 | yes | -0.50 | -0.44 | <0.001 | yes | no |
| falcon3-7b|diabetes130 | + | yes | +1.48 | +1.48 | <0.001 | <0.001 | yes | +0.41 | +0.40 | <0.001 | no | no |
| yi15-9b|physionet2012 | + | yes | +0.97 | +0.67 | <0.001 | <0.001 | yes | +0.11 | +0.04 | 0.511 | no | no |
| yi15-9b|physionet2019 | + | yes | +0.56 | +0.64 | 0.018 | 0.055 | no | -0.14 | +0.02 | 0.576 | no | no |

**Alarming-last example order, O.** Supported in 7 of 7 eligible cells, reduced in 2, opposite in 0: **kept**.

| cell | sign | valid | O (a) | O (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | − | yes | -6.07 | -6.60 | <0.001 | <0.001 | yes | -2.64 | -2.26 | <0.001 | no | no |
| llama-instruct|physionet2012 | − | yes | -1.14 | -1.11 | <0.001 | <0.001 | yes | -0.87 | -0.87 | <0.001 | no | no |
| llama-instruct|physionet2019 | − | yes | -0.52 | -0.45 | <0.001 | <0.001 | yes | +0.25 | +0.11 | 0.462 | no | no |
| olmo3-instruct|physionet2012 | + | yes | +2.19 | +1.53 | <0.001 | <0.001 | yes | -0.14 | +0.07 | 0.462 | no | no |
| falcon3-7b|physionet2019 | + | yes | +1.47 | +1.22 | <0.001 | <0.001 | yes | -0.20 | -0.21 | 0.045 | yes | no |
| falcon3-7b|diabetes130 | + | yes | +0.82 | +1.22 | <0.001 | <0.001 | yes | -0.97 | -0.85 | <0.001 | yes | no |
| yi15-9b|physionet2019 | − | yes | -1.02 | -1.21 | 0.001 | 0.001 | yes | +0.47 | +0.18 | 0.113 | no | no |

## R12. Paraphrased prompt

**Paraphrased prompt, W.** Supported in 5 of 11 eligible cells, reduced in 4, opposite in 1: **not settled**.

| cell | sign | valid | W (a) | W (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | + | yes | -0.10 | -0.21 | 0.541 | 1.000 | no | -1.84 | -1.99 | <0.001 | yes | no |
| qwen-instruct|physionet2019 | − | yes | -3.76 | -3.92 | <0.001 | <0.001 | yes | -0.89 | -1.02 | 0.236 | no | no |
| llama-instruct|physionet2012 | − | yes | -0.97 | -0.82 | <0.001 | <0.001 | yes | -0.66 | -0.45 | <0.001 | no | no |
| llama-instruct|diabetes130 | − | yes | -1.54 | -2.16 | 0.421 | 1.000 | no | +0.89 | -1.25 | 1.000 | no | no |
| olmo3-instruct|physionet2012 | − | yes | -0.66 | -1.20 | <0.001 | <0.001 | yes | +0.16 | +0.36 | 0.329 | no | no |
| olmo3-instruct|physionet2019 | − | yes | -0.76 | -0.86 | 0.021 | 0.104 | no | +0.87 | +0.49 | <0.001 | yes | no |
| falcon3-7b|physionet2012 | + | yes | +0.58 | +0.53 | <0.001 | <0.001 | yes | -0.08 | -0.10 | 0.329 | no | no |
| falcon3-7b|physionet2019 | + | no | +1.26 | +0.99 | 1.000 | 1.000 | no | -0.16 | -0.30 | 1.000 | no | no |
| falcon3-7b|diabetes130 | + | yes | +2.04 | +1.61 | 0.004 | 0.023 | yes | +0.97 | +0.54 | 0.240 | no | no |
| yi15-9b|physionet2012 | + | yes | -0.30 | -0.45 | 0.001 | 0.010 | no | -1.17 | -1.08 | <0.001 | yes | yes |
| yi15-9b|physionet2019 | + | yes | +0.20 | +0.20 | 0.023 | 0.104 | no | -0.50 | -0.43 | <0.001 | yes | no |

**Paraphrased prompt, O.** Supported in 2 of 7 eligible cells, reduced in 3, opposite in 1: **not settled**.

| cell | sign | valid | O (a) | O (a′) | p | Holm p | supported | change (a) | change (a′) | Holm p | reduced | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | − | yes | -3.85 | -4.33 | <0.001 | <0.001 | yes | -0.42 | +0.02 | 1.000 | no | no |
| llama-instruct|physionet2012 | − | yes | +1.28 | +1.37 | <0.001 | <0.001 | no | +1.55 | +1.61 | <0.001 | yes | yes |
| llama-instruct|physionet2019 | − | yes | +0.76 | +0.94 | 0.224 | 0.448 | no | +1.53 | +1.51 | 0.080 | no | no |
| olmo3-instruct|physionet2012 | + | yes | +3.01 | +2.80 | <0.001 | <0.001 | yes | +0.67 | +1.34 | <0.001 | no | no |
| falcon3-7b|physionet2019 | + | no | +1.52 | +1.49 | 1.000 | 1.000 | no | -0.15 | +0.07 | 1.000 | no | no |
| falcon3-7b|diabetes130 | + | yes | +0.70 | +1.85 | 0.112 | 0.335 | no | -1.09 | -0.23 | 0.042 | yes | no |
| yi15-9b|physionet2019 | − | yes | -0.17 | -0.18 | 0.041 | 0.162 | no | +1.32 | +1.21 | <0.001 | yes | no |

## Joint reading of R8 and R9

W with True/False: removed; W with the negated question: removed. **Reading: the Yes/No question as worded.**

**Wording contrast** D = (s_reversed − s_negated) / 2 (both framings have “Yes” for the safe outcome):

| cell | D (a) | D (a′) | p |
|---|---|---|---|
| llama-instruct|physionet2012 | -0.42 [-0.46, -0.38] | -0.40 [-0.43, -0.37] | <0.001 |
| llama-instruct|physionet2019 | -1.21 [-1.73, -0.76] | -0.80 [-0.86, -0.73] | <0.001 |
| llama-instruct|diabetes130 | +0.76 [-0.02, +1.89] | -0.44 [-0.60, -0.28] | 0.054 |
| qwen-instruct|physionet2012 | -0.77 [-1.09, -0.47] | -1.50 [-1.74, -1.27] | <0.001 |
| qwen-instruct|physionet2019 | +0.82 [+0.15, +1.57] | +0.59 [+0.46, +0.73] | 0.020 |
| qwen-instruct|diabetes130 | -2.48 [-4.58, -0.87] | -0.50 [-0.68, -0.32] | 0.004 |
| olmo3-instruct|physionet2012 | +1.18 [+0.93, +1.40] | +0.35 [+0.25, +0.45] | <0.001 |
| olmo3-instruct|physionet2019 | +10.15 [+2.60, +17.16] | +0.44 [-0.10, +0.89] | 0.100 |
| olmo3-instruct|diabetes130 | -2.60 [-12.99, +7.80] | +0.33 [+0.15, +0.50] | 0.506 |
| falcon3-7b|physionet2012 | -0.46 [-0.56, -0.37] | -0.53 [-0.58, -0.48] | <0.001 |
| falcon3-7b|physionet2019 | -0.71 [-1.10, -0.37] | -0.82 [-0.86, -0.79] | 0.002 |
| falcon3-7b|diabetes130 | -0.56 [-0.68, -0.43] | -0.68 [-0.72, -0.64] | <0.001 |
| yi15-9b|physionet2012 | -0.75 [-0.80, -0.71] | -0.79 [-0.83, -0.76] | <0.001 |
| yi15-9b|physionet2019 | +0.17 [-0.26, +0.67] | -0.58 [-0.63, -0.53] | 0.597 |
| yi15-9b|diabetes130 | +0.46 [-0.17, +1.08] | -0.25 [-0.34, -0.17] | 0.166 |

## R10. Post-training and the chat template

| cell | W B·plain (a′) | W B·chat (a′) | W I·chat (a′) | P (a / a′) | Holm p | P supported | F (a / a′) | Holm p | F in direction of W |
|---|---|---|---|---|---|---|---|---|---|
| llama|physionet2012 | -0.17 [-0.18, -0.16] | -0.17 [-0.19, -0.15] | -0.37 [-0.44, -0.31] | -0.01 / -0.21 | 1.000 | no | -0.17 / +0.00 | 1.000 | no |
| llama|diabetes130 | -0.26 [-0.31, -0.20] | -0.28 [-0.31, -0.24] | -0.91 [-1.09, -0.72] | -3.35 / -0.63 | 1.000 | no | +1.74 / -0.02 | 1.000 | no |
| qwen|physionet2012 | +0.33 [+0.29, +0.37] | +0.47 [+0.44, +0.50] | +1.79 [+1.49, +2.08] | +1.32 / +1.32 | <0.001 | yes | +0.16 / +0.14 | <0.001 | yes |
| qwen|physionet2019 | +0.06 [+0.04, +0.08] | +0.05 [+0.03, +0.08] | -2.90 [-3.13, -2.69] | -3.01 / -2.96 | 1.000 | no | -0.05 / -0.01 | 1.000 | no |

P supported in 1 of 4 eligible cells; F in the direction of W in 1. **R10: not supported; not separated by this study.**

**Q14g.** With I·plain: W(I·plain), P′ = W(I·plain) − W(B·plain) and F′ = W(I·chat) − W(I·plain):

| cell | I·plain valid | W I·plain (a′) | P′ (a) | P′ (a′) | p | F′ (a) | F′ (a′) | p |
|---|---|---|---|---|---|---|---|---|
| llama|physionet2012 | yes | +0.78 [+0.73, +0.83] | +1.02 [+0.94, +1.10] | +0.95 [+0.90, +1.00] | <0.001 | -1.20 [-1.33, -1.07] | -1.15 [-1.21, -1.10] | <0.001 |
| llama|diabetes130 | no | -0.44 [-0.55, -0.32] | +0.30 [-0.54, +1.04] | -0.18 [-0.29, -0.06] | 0.565 | -1.90 [-3.45, -0.69] | -0.47 [-0.62, -0.33] | 0.003 |
| qwen|physionet2012 | yes | +0.83 [+0.69, +0.97] | +0.74 [+0.49, +0.99] | +0.50 [+0.37, +0.64] | <0.001 | +0.74 [+0.47, +1.01] | +0.96 [+0.75, +1.17] | <0.001 |
| qwen|physionet2019 | yes | -2.08 [-2.20, -1.97] | -2.70 [-3.41, -2.06] | -2.14 [-2.26, -2.03] | <0.001 | -0.36 [-0.99, +0.21] | -0.82 [-0.98, -0.67] | 0.162 |

## R13. Direction words and numbers in one sentence

| cell | valid | movement, card | cue | net number | Δm | p | Holm p | supported |
|---|---|---|---|---|---|---|---|---|
| llama-instruct|physionet2012 | yes | +1.44 | +3.43 | +0.61 | +2.81 | <0.001 | <0.001 | yes |
| llama-instruct|physionet2019 | yes | +1.62 | +5.16 | +0.64 | +4.52 | <0.001 | <0.001 | yes |
| llama-instruct|diabetes130 | yes | +2.45 | +6.07 | +0.56 | +5.52 | <0.001 | <0.001 | yes |
| qwen-instruct|physionet2012 | yes | +5.34 | +12.62 | +6.06 | +6.56 | <0.001 | <0.001 | yes |
| qwen-instruct|physionet2019 | yes | +9.06 | +21.66 | +8.41 | +13.25 | <0.001 | <0.001 | yes |
| qwen-instruct|diabetes130 | no | +10.40 | +17.11 | +8.06 | +9.06 | 1.000 | 1.000 | no |
| olmo3-instruct|physionet2012 | yes | +1.69 | +3.03 | +2.13 | +0.90 | <0.001 | <0.001 | yes |
| olmo3-instruct|physionet2019 | yes | +2.57 | +4.87 | +2.27 | +2.60 | <0.001 | <0.001 | yes |
| olmo3-instruct|diabetes130 | no | +2.99 | +6.71 | +3.22 | +3.48 | 1.000 | 1.000 | no |
| falcon3-7b|physionet2012 | yes | +3.27 | +5.40 | +1.63 | +3.77 | <0.001 | <0.001 | yes |
| falcon3-7b|physionet2019 | yes | +4.69 | +8.18 | +4.80 | +3.38 | <0.001 | <0.001 | yes |
| falcon3-7b|diabetes130 | yes | +4.31 | +8.92 | +4.88 | +4.04 | <0.001 | <0.001 | yes |
| yi15-9b|physionet2012 | yes | +1.54 | +3.15 | +0.71 | +2.44 | <0.001 | <0.001 | yes |
| yi15-9b|physionet2019 | yes | +1.71 | +3.57 | +0.82 | +2.75 | <0.001 | <0.001 | yes |
| yi15-9b|diabetes130 | yes | +2.42 | +4.93 | +1.64 | +3.29 | <0.001 | <0.001 | yes |

**R13: supported** (13 of 13 valid cells).

## Q14. Descriptive

**Q14c. Static answer lean and W:**

| pair | valid cells | conclusive W | sign of W = sign of lean | Spearman (lean, W a′) |
|---|---|---|---|---|
| registered | 13 | 11 | 7 | +0.57 |
| tf | 13 | 9 | 8 | -0.03 |
| neg | 13 | 7 | 4 | +0.47 |
| order | 13 | 11 | 7 | +0.54 |
| para | 12 | 9 | 7 | +0.73 |

**Q14d. Generated answers** (agreement with the first-token reading among parsed answers; E_W from the generated answers / from the first-token reading, same prompts):

| model | mode | answers | parsed | agreement | E_W physionet2012 | E_W physionet2019 | E_W diabetes130 |
|---|---|---|---|---|---|---|---|
| llama-instruct | free | 1800 | 0.608 | 0.717 | -0.21 / -0.13 | -0.43 / -0.11 | -0.09 / -0.04 |
| llama-instruct | prefill | 1800 | 1.000 | 0.997 | -0.07 / -0.06 | -0.05 / -0.03 | +0.02 / +0.02 |
| qwen-instruct | free | 1800 | 0.896 | 0.707 | -0.23 / +0.11 | -0.05 / -0.20 | +0.14 / +0.21 |
| qwen-instruct | prefill | 1800 | 1.000 | 0.998 | +0.05 / +0.06 | -0.21 / -0.21 | +0.21 / +0.21 |
| olmo3-instruct | free | 1800 | 1.000 | 0.930 | -0.34 / -0.23 | +0.03 / -0.11 | +0.06 / -0.02 |
| olmo3-instruct | prefill | 1800 | 1.000 | 0.997 | -0.26 / -0.23 | -0.12 / -0.11 | -0.02 / -0.02 |
| falcon3-7b | free | 1800 | 1.000 | 0.916 | +0.04 / +0.05 | +0.03 / +0.19 | +0.05 / -0.08 |
| falcon3-7b | prefill | 1800 | 1.000 | 0.999 | +0.04 / +0.05 | +0.19 / +0.19 | -0.08 / -0.08 |
| yi15-9b | free | 1800 | 1.000 | 0.741 | +0.01 / +0.18 | -0.03 / +0.08 | +0.41 / +0.33 |
| yi15-9b | prefill | 1800 | 1.000 | 0.996 | +0.16 / +0.18 | +0.08 / +0.08 | +0.36 / +0.33 |

**Q14a. Replay** of prompts shared with earlier runs:

| comparison | prompts | text differs | max |Δ| | median |Δ| | same answer |
|---|---|---|---|---|---|
| falcon3-7b vs v3, set A | 10800 | 0 | 0.0391 | 0.0000 | 1.000 |
| llama-base vs v1, set A | 10800 | 0 | 0.0312 | 0.0000 | 0.997 |
| llama-instruct vs v3, set A | 10800 | 0 | 0.0781 | 0.0000 | 1.000 |
| llama-instruct vs v2, Asf alarm | 5400 | 0 | 0.0625 | 0.0156 | 0.998 |
| llama-instruct vs v1, set A | 10800 | 0 | 0.1094 | 0.0156 | 0.999 |
| olmo3-instruct vs v3, set A | 10800 | 0 | 0.0469 | 0.0000 | 0.999 |
| olmo3-instruct vs v2, set A | 10800 | 0 | 0.0469 | 0.0000 | 1.000 |
| olmo3-instruct vs v2, Asf alarm | 5400 | 0 | 0.0625 | 0.0000 | 1.000 |
| qwen-base vs v1, set A | 10800 | 0 | 0.0625 | 0.0000 | 0.997 |
| qwen-instruct vs v3, set A | 10800 | 0 | 0.7812 | 0.0312 | 0.999 |
| qwen-instruct vs v2, Asf alarm | 5400 | 0 | 0.3594 | 0.0000 | 0.999 |
| qwen-instruct vs v1, set A | 10800 | 0 | 0.6562 | 0.0312 | 0.999 |
| yi15-9b vs v3, set A | 10800 | 0 | 0.0313 | 0.0000 | 1.000 |

**Q14f. Imputation bound.** Share of records with an imputed variant: falcon3-7b 0.002, llama-base 0.000, llama-instruct 0.000, olmo3-instruct 0.002, qwen-base 0.000, qwen-instruct 0.078, yi15-9b 0.000.
Under the bound: R8 W removed, R9 W removed (joint reading: the Yes/No question as worded); R10 not separated by this study; R11 W kept, O kept; R12 W not settled, O not settled; R13 supported. Cells whose support or reduction changes: none.

