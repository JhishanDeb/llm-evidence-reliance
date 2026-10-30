# Third study: results (`prereg-v3`)

Runs: falcon3-7b, llama-instruct, olmo3-instruct, phi4-mini, qwen-instruct, yi15-9b. Missing registered runs: none. Bootstrap resamples: 10000.

## Decisions

- **R4** (answer-word part in the new families): not supported.
- **R5** (outcome part agrees with the answer options): not supported.
- **R6** (answer-word part kept with every changed card): supported; kept with: words, numbers, cue, matched.
- **R7** (outcome part): kept with either the direction words or the numbers alone; cue: a one-sentence direction cue is enough.

## Validity

| cell | mass alarm | mass reversed | C0 acc. | valid | mass words | mass numbers | mass cue | mass matched | answer options valid |
|---|---|---|---|---|---|---|---|---|---|
| falcon3-7b|physionet2012 | 0.999 | 0.999 | 0.672 | yes | 0.999 | 0.999 | 0.999 | 0.999 | yes |
| falcon3-7b|physionet2019 | 0.999 | 0.999 | 0.620 | yes | 0.999 | 0.999 | 0.999 | 0.999 | yes |
| falcon3-7b|diabetes130 | 0.998 | 0.999 | 0.602 | yes | 0.999 | 0.999 | 0.999 | 0.999 | yes |
| llama-instruct|physionet2012 | 0.986 | 0.988 | 0.607 | yes | 0.987 | 0.986 | 0.989 | 0.989 | – |
| llama-instruct|physionet2019 | 0.994 | 0.995 | 0.618 | yes | 0.994 | 0.995 | 0.994 | 0.996 | – |
| llama-instruct|diabetes130 | 0.990 | 0.994 | 0.557 | yes | 0.995 | 0.988 | 0.994 | 0.992 | – |
| olmo3-instruct|physionet2012 | 0.997 | 0.994 | 0.612 | yes | 0.997 | 0.995 | 0.998 | 0.992 | – |
| olmo3-instruct|physionet2019 | 0.997 | 0.997 | 0.597 | yes | 0.998 | 0.997 | 0.999 | 0.997 | – |
| olmo3-instruct|diabetes130 | 0.998 | 0.997 | 0.540 | no | 0.997 | 0.997 | 0.999 | 0.997 | – |
| phi4-mini|physionet2012 | 0.997 | 0.996 | 0.663 | yes | 0.997 | 0.995 | 0.999 | 0.994 | yes |
| phi4-mini|physionet2019 | 0.995 | 0.997 | 0.610 | yes | 0.996 | 0.995 | 1.000 | 0.997 | no |
| phi4-mini|diabetes130 | 0.992 | 0.998 | 0.558 | yes | 0.999 | 0.995 | 0.999 | 0.996 | no |
| qwen-instruct|physionet2012 | 0.999 | 1.000 | 0.692 | yes | 0.999 | 1.000 | 0.999 | 1.000 | – |
| qwen-instruct|physionet2019 | 0.994 | 0.999 | 0.608 | yes | 0.999 | 0.997 | 1.000 | 1.000 | – |
| qwen-instruct|diabetes130 | 0.989 | 0.998 | 0.528 | no | 1.000 | 0.995 | 1.000 | 0.999 | – |
| yi15-9b|physionet2012 | 0.999 | 0.998 | 0.607 | yes | 0.999 | 0.999 | 1.000 | 0.999 | yes |
| yi15-9b|physionet2019 | 0.999 | 0.998 | 0.587 | yes | 0.999 | 0.999 | 0.999 | 0.999 | yes |
| yi15-9b|diabetes130 | 0.999 | 0.999 | 0.580 | yes | 0.999 | 0.999 | 1.000 | 0.999 | yes |

## The two parts with the registered card

O = (s_alarm + s_reversed) / 2 and W = (s_alarm − s_reversed) / 2, for s = a and s = a′ (95% intervals); W < 0 is a pull toward the answer “No”. p is the intersection–union p (no correction).

| cell | valid | O (a) | O (a′) | p | W (a) | W (a′) | p | movement |
|---|---|---|---|---|---|---|---|---|
| falcon3-7b|physionet2012 | yes | +0.47 [+0.22, +0.71] | +0.16 [-0.00, +0.32] | 0.053 | +0.66 [+0.56, +0.76] | +0.62 [+0.55, +0.70] | <0.001 | +2.17 |
| falcon3-7b|physionet2019 | yes | +1.67 [+1.39, +1.86] | +1.42 [+1.32, +1.52] | <0.001 | +1.41 [+1.29, +1.53] | +1.30 [+1.23, +1.36] | <0.001 | +2.34 |
| falcon3-7b|diabetes130 | yes | +1.79 [+1.59, +1.98] | +2.08 [+1.98, +2.18] | <0.001 | +1.07 [+0.95, +1.19] | +1.08 [+1.01, +1.15] | <0.001 | +3.32 |
| llama-instruct|physionet2012 | yes | -0.27 [-0.43, -0.11] | -0.24 [-0.34, -0.14] | <0.001 | -0.32 [-0.47, -0.16] | -0.38 [-0.45, -0.31] | <0.001 | +1.31 |
| llama-instruct|physionet2019 | yes | -0.77 [-1.29, -0.26] | -0.56 [-0.66, -0.47] | 0.003 | +0.22 [-0.30, +0.72] | -0.06 [-0.14, +0.02] | 0.461 | +1.25 |
| llama-instruct|diabetes130 | yes | +0.34 [-0.72, +1.86] | -1.02 [-1.23, -0.81] | 0.601 | -2.43 [-3.91, -1.36] | -0.91 [-1.09, -0.72] | <0.001 | +2.83 |
| olmo3-instruct|physionet2012 | yes | +2.34 [+1.93, +2.68] | +1.46 [+1.32, +1.60] | <0.001 | -0.83 [-1.19, -0.53] | -1.56 [-1.63, -1.50] | <0.001 | +1.41 |
| olmo3-instruct|physionet2019 | yes | -0.25 [-0.78, +0.33] | +0.02 [-0.10, +0.13] | 0.800 | -1.63 [-2.22, -1.13] | -1.34 [-1.45, -1.24] | <0.001 | +2.53 |
| olmo3-instruct|diabetes130 | no | -1.91 [-15.07, +10.96] | +1.59 [+1.33, +1.87] | 0.670 | +0.83 [-12.09, +13.78] | -2.20 [-2.44, -1.97] | 0.808 | +3.24 |
| phi4-mini|physionet2012 | yes | +1.74 [+1.55, +1.92] | +1.87 [+1.74, +1.99] | <0.001 | -2.02 [-2.10, -1.93] | -2.08 [-2.13, -2.03] | <0.001 | +1.89 |
| phi4-mini|physionet2019 | yes | +0.91 [-0.31, +2.00] | +1.44 [+1.34, +1.54] | 0.119 | -1.54 [-2.57, -0.46] | -2.64 [-2.72, -2.56] | 0.013 | +2.43 |
| phi4-mini|diabetes130 | yes | +3.93 [+0.66, +7.33] | +1.87 [+1.67, +2.08] | 0.025 | -2.22 [-4.80, +0.79] | -2.74 [-2.96, -2.52] | 0.127 | +2.96 |
| qwen-instruct|physionet2012 | yes | -3.45 [-4.06, -2.87] | -4.37 [-4.91, -3.84] | <0.001 | +1.74 [+1.32, +2.16] | +1.79 [+1.50, +2.08] | <0.001 | +4.75 |
| qwen-instruct|physionet2019 | yes | +0.60 [-0.63, +1.90] | +0.21 [-0.10, +0.52] | 0.320 | -2.87 [-4.02, -1.87] | -2.90 [-3.12, -2.68] | <0.001 | +7.28 |
| qwen-instruct|diabetes130 | no | -4.98 [-8.33, -2.04] | -2.55 [-2.88, -2.23] | <0.001 | +2.76 [+0.04, +5.90] | -1.45 [-1.74, -1.15] | 0.047 | +8.42 |
| yi15-9b|physionet2012 | yes | -0.16 [-0.29, -0.03] | -0.32 [-0.43, -0.22] | 0.013 | +0.86 [+0.72, +1.00] | +0.63 [+0.54, +0.72] | <0.001 | +1.71 |
| yi15-9b|physionet2019 | yes | -1.49 [-1.85, -1.13] | -1.39 [-1.47, -1.32] | <0.001 | +0.70 [+0.39, +1.00] | +0.62 [+0.56, +0.69] | <0.001 | +1.40 |
| yi15-9b|diabetes130 | yes | -0.07 [-0.65, +0.47] | +0.08 [-0.02, +0.18] | 0.758 | -0.13 [-0.65, +0.44] | -0.21 [-0.30, -0.12] | 0.641 | +2.27 |

## R4. The answer-word part in the new families

| cell | valid | W (a) | W (a′) | p | Holm p | supported |
|---|---|---|---|---|---|---|
| falcon3-7b|physionet2012 | yes | +0.66 | +0.62 | <0.001 | <0.001 | no |
| falcon3-7b|physionet2019 | yes | +1.41 | +1.30 | <0.001 | <0.001 | no |
| falcon3-7b|diabetes130 | yes | +1.07 | +1.08 | <0.001 | <0.001 | no |
| yi15-9b|physionet2012 | yes | +0.86 | +0.63 | <0.001 | <0.001 | no |
| yi15-9b|physionet2019 | yes | +0.70 | +0.62 | <0.001 | <0.001 | no |
| yi15-9b|diabetes130 | yes | -0.13 | -0.21 | 0.641 | 0.641 | no |
| phi4-mini|physionet2012 | yes | -2.02 | -2.08 | <0.001 | <0.001 | yes |
| phi4-mini|physionet2019 | yes | -1.54 | -2.64 | 0.013 | 0.039 | yes |
| phi4-mini|diabetes130 | yes | -2.22 | -2.74 | 0.127 | 0.254 | no |

Per model: falcon3-7b: 0 of 3 valid cells, 3 in the other direction, shows the pull: no; yi15-9b: 0 of 3 valid cells, 2 in the other direction, shows the pull: no; phi4-mini: 2 of 3 valid cells, 0 in the other direction, shows the pull: yes.

**R4: not supported** (1 of 3 new models show the pull toward “No”).

## R5. The outcome part in the new families, and the answer options

| cell | valid | O (a) | O (a′) | p | Holm p | direction |
|---|---|---|---|---|---|---|
| falcon3-7b|physionet2012 | yes | +0.47 | +0.16 | 0.053 | 0.159 | not shown |
| falcon3-7b|physionet2019 | yes | +1.67 | +1.42 | <0.001 | <0.001 | toward alarm |
| falcon3-7b|diabetes130 | yes | +1.79 | +2.08 | <0.001 | <0.001 | toward alarm |
| yi15-9b|physionet2012 | yes | -0.16 | -0.32 | 0.013 | 0.066 | not shown |
| yi15-9b|physionet2019 | yes | -1.49 | -1.39 | <0.001 | <0.001 | toward safe |
| yi15-9b|diabetes130 | yes | -0.07 | +0.08 | 0.758 | 0.758 | not shown |
| phi4-mini|physionet2012 | yes | +1.74 | +1.87 | <0.001 | <0.001 | toward alarm |
| phi4-mini|physionet2019 | yes | +0.91 | +1.44 | 0.119 | 0.239 | not shown |
| phi4-mini|diabetes130 | yes | +3.93 | +1.87 | 0.025 | 0.099 | not shown |

| cell | sign of O | sign with answer options | agree |
|---|---|---|---|
| falcon3-7b|physionet2019 | 1 | 1 | yes |
| yi15-9b|physionet2012 | -1 | 1 | no |
| phi4-mini|physionet2012 | 1 | 1 | yes |

**R5: not supported** (2 of 3 cells agree).

## R6. The answer-word part W with each changed card

Eligible cells: qwen-instruct|physionet2012 (+; pre-listed (Studies 1 and 2)), qwen-instruct|physionet2019 (−; pre-listed (Studies 1 and 2)), llama-instruct|physionet2012 (−; pre-listed (Studies 1 and 2)), llama-instruct|diabetes130 (−; pre-listed (Studies 1 and 2)), olmo3-instruct|physionet2012 (−; pre-listed (Studies 1 and 2)), olmo3-instruct|physionet2019 (−; pre-listed (Studies 1 and 2)), falcon3-7b|physionet2012 (+; new family, established in this study), falcon3-7b|physionet2019 (+; new family, established in this study), falcon3-7b|diabetes130 (+; new family, established in this study), yi15-9b|physionet2012 (+; new family, established in this study), yi15-9b|physionet2019 (+; new family, established in this study), phi4-mini|physionet2012 (−; new family, established in this study), phi4-mini|physionet2019 (−; new family, established in this study).

**Card: words.** Kept in 11 of 13 eligible cells (kept with this card: yes); smaller than with the registered card in 5.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | +0.29 | +0.43 | 0.222 | 0.222 | no | -1.45 | -1.36 | yes |
| qwen-instruct|physionet2019 | yes | -3.41 | -3.07 | <0.001 | <0.001 | yes | -0.53 | -0.17 | no |
| llama-instruct|physionet2012 | yes | -0.69 | -0.66 | <0.001 | <0.001 | yes | -0.37 | -0.29 | no |
| llama-instruct|diabetes130 | yes | -1.84 | -1.10 | 0.007 | 0.022 | yes | +0.59 | -0.19 | no |
| olmo3-instruct|physionet2012 | yes | -0.86 | -1.49 | <0.001 | <0.001 | yes | -0.03 | +0.08 | no |
| olmo3-instruct|physionet2019 | yes | -0.98 | -0.98 | 0.006 | 0.022 | yes | +0.65 | +0.36 | yes |
| falcon3-7b|physionet2012 | yes | +0.48 | +0.44 | <0.001 | <0.001 | yes | -0.17 | -0.19 | yes |
| falcon3-7b|physionet2019 | yes | +1.27 | +1.17 | <0.001 | <0.001 | yes | -0.15 | -0.13 | yes |
| falcon3-7b|diabetes130 | yes | +1.04 | +1.15 | <0.001 | <0.001 | yes | -0.03 | +0.07 | no |
| yi15-9b|physionet2012 | yes | +0.46 | +0.34 | <0.001 | <0.001 | yes | -0.40 | -0.29 | yes |
| yi15-9b|physionet2019 | yes | +0.58 | +0.36 | <0.001 | <0.001 | yes | -0.12 | -0.27 | no |
| phi4-mini|physionet2012 | yes | -2.40 | -2.47 | <0.001 | <0.001 | yes | -0.38 | -0.39 | no |
| phi4-mini|physionet2019 | yes | -1.47 | -2.83 | 0.068 | 0.136 | no | +0.08 | -0.19 | no |

**Card: numbers.** Kept in 11 of 13 eligible cells (kept with this card: yes); smaller than with the registered card in 6.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | +3.28 | +2.86 | <0.001 | <0.001 | yes | +1.54 | +1.08 | no |
| qwen-instruct|physionet2019 | yes | -2.54 | -1.96 | <0.001 | <0.001 | yes | +0.33 | +0.94 | no |
| llama-instruct|physionet2012 | yes | +1.03 | +0.77 | <0.001 | <0.001 | no | +1.35 | +1.14 | yes |
| llama-instruct|diabetes130 | yes | -3.21 | -0.02 | 0.879 | 0.879 | no | -0.78 | +0.89 | no |
| olmo3-instruct|physionet2012 | yes | -0.39 | -1.15 | 0.007 | 0.014 | yes | +0.43 | +0.42 | yes |
| olmo3-instruct|physionet2019 | yes | -1.60 | -1.19 | <0.001 | <0.001 | yes | +0.03 | +0.15 | no |
| falcon3-7b|physionet2012 | yes | +0.48 | +0.46 | <0.001 | <0.001 | yes | -0.18 | -0.17 | yes |
| falcon3-7b|physionet2019 | yes | +0.51 | +0.47 | <0.001 | <0.001 | yes | -0.90 | -0.83 | yes |
| falcon3-7b|diabetes130 | yes | +0.41 | +0.37 | <0.001 | <0.001 | yes | -0.67 | -0.71 | yes |
| yi15-9b|physionet2012 | yes | +1.24 | +0.92 | <0.001 | <0.001 | yes | +0.38 | +0.29 | no |
| yi15-9b|physionet2019 | yes | +0.88 | +0.94 | <0.001 | <0.001 | yes | +0.18 | +0.32 | no |
| phi4-mini|physionet2012 | yes | -1.38 | -1.45 | <0.001 | <0.001 | yes | +0.64 | +0.63 | yes |
| phi4-mini|physionet2019 | yes | -1.90 | -1.84 | <0.001 | 0.002 | yes | -0.36 | +0.80 | no |

**Card: cue.** Kept in 10 of 13 eligible cells (kept with this card: yes); smaller than with the registered card in 6.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | +1.94 | +1.91 | <0.001 | <0.001 | yes | +0.21 | +0.13 | no |
| qwen-instruct|physionet2019 | yes | -2.20 | -1.45 | 0.001 | 0.002 | yes | +0.68 | +1.45 | no |
| llama-instruct|physionet2012 | yes | -1.54 | -1.66 | <0.001 | <0.001 | yes | -1.22 | -1.28 | no |
| llama-instruct|diabetes130 | yes | -5.49 | -4.90 | <0.001 | <0.001 | yes | -3.06 | -3.99 | no |
| olmo3-instruct|physionet2012 | yes | -0.54 | -0.97 | <0.001 | <0.001 | yes | +0.29 | +0.59 | yes |
| olmo3-instruct|physionet2019 | yes | -2.01 | -1.63 | <0.001 | <0.001 | yes | -0.38 | -0.29 | no |
| falcon3-7b|physionet2012 | yes | -0.63 | -0.59 | <0.001 | <0.001 | no | -1.29 | -1.21 | yes |
| falcon3-7b|physionet2019 | yes | -1.84 | -1.93 | <0.001 | <0.001 | no | -3.25 | -3.22 | yes |
| falcon3-7b|diabetes130 | yes | -2.06 | -2.14 | <0.001 | <0.001 | no | -3.13 | -3.21 | yes |
| yi15-9b|physionet2012 | yes | +0.33 | +0.15 | <0.001 | <0.001 | yes | -0.53 | -0.48 | yes |
| yi15-9b|physionet2019 | yes | +0.50 | +0.15 | <0.001 | 0.002 | yes | -0.20 | -0.48 | no |
| phi4-mini|physionet2012 | yes | -1.88 | -1.83 | <0.001 | <0.001 | yes | +0.14 | +0.25 | yes |
| phi4-mini|physionet2019 | yes | -2.06 | -1.88 | <0.001 | <0.001 | yes | -0.52 | +0.77 | no |

**Card: matched.** Kept in 12 of 13 eligible cells (kept with this card: yes); smaller than with the registered card in 7.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | +1.65 | +1.64 | <0.001 | <0.001 | yes | -0.09 | -0.14 | no |
| qwen-instruct|physionet2019 | yes | -3.22 | -3.25 | <0.001 | <0.001 | yes | -0.34 | -0.35 | no |
| llama-instruct|physionet2012 | yes | +0.36 | +0.31 | <0.001 | <0.001 | no | +0.68 | +0.68 | yes |
| llama-instruct|diabetes130 | yes | -1.99 | -0.46 | 0.002 | 0.007 | yes | +0.43 | +0.46 | no |
| olmo3-instruct|physionet2012 | yes | -0.34 | -1.10 | 0.025 | 0.025 | yes | +0.48 | +0.47 | yes |
| olmo3-instruct|physionet2019 | yes | -1.49 | -1.00 | <0.001 | <0.001 | yes | +0.14 | +0.34 | no |
| falcon3-7b|physionet2012 | yes | +0.30 | +0.35 | <0.001 | <0.001 | yes | -0.36 | -0.28 | yes |
| falcon3-7b|physionet2019 | yes | +0.51 | +0.38 | <0.001 | <0.001 | yes | -0.90 | -0.92 | yes |
| falcon3-7b|diabetes130 | yes | +0.67 | +0.70 | <0.001 | <0.001 | yes | -0.40 | -0.38 | yes |
| yi15-9b|physionet2012 | yes | +0.53 | +0.34 | <0.001 | <0.001 | yes | -0.33 | -0.29 | yes |
| yi15-9b|physionet2019 | yes | +0.52 | +0.74 | 0.005 | 0.011 | yes | -0.18 | +0.12 | no |
| phi4-mini|physionet2012 | yes | -1.57 | -1.53 | <0.001 | <0.001 | yes | +0.45 | +0.56 | yes |
| phi4-mini|physionet2019 | yes | -1.70 | -1.77 | 0.001 | 0.005 | yes | -0.16 | +0.87 | no |

## R7. The outcome part O with each reduced card

Eligible cells: qwen-instruct|physionet2012 (−; pre-listed (Studies 1 and 2)), llama-instruct|physionet2012 (−; pre-listed (Studies 1 and 2)), llama-instruct|physionet2019 (−; pre-listed (Studies 1 and 2)), olmo3-instruct|physionet2012 (+; pre-listed (Studies 1 and 2)), falcon3-7b|physionet2019 (+; new family, established in this study), falcon3-7b|diabetes130 (+; new family, established in this study), yi15-9b|physionet2019 (−; new family, established in this study), phi4-mini|physionet2012 (+; new family, established in this study).

**Card: words.** Kept in 6 of 8 eligible cells (kept with this card: yes); smaller than with the registered card in 3.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | -2.71 | -3.33 | <0.001 | <0.001 | yes | +0.74 | +1.04 | yes |
| llama-instruct|physionet2012 | yes | -0.14 | -0.07 | 0.126 | 0.126 | no | +0.13 | +0.17 | yes |
| llama-instruct|physionet2019 | yes | -0.57 | -0.70 | 0.031 | 0.063 | no | +0.19 | -0.13 | no |
| olmo3-instruct|physionet2012 | yes | +2.43 | +1.60 | <0.001 | <0.001 | yes | +0.10 | +0.13 | no |
| falcon3-7b|physionet2019 | yes | +1.40 | +1.16 | <0.001 | <0.001 | yes | -0.27 | -0.27 | yes |
| falcon3-7b|diabetes130 | yes | +2.49 | +2.91 | <0.001 | <0.001 | yes | +0.70 | +0.83 | no |
| yi15-9b|physionet2019 | yes | -1.72 | -1.58 | <0.001 | <0.001 | yes | -0.22 | -0.19 | no |
| phi4-mini|physionet2012 | yes | +1.75 | +1.95 | <0.001 | <0.001 | yes | +0.02 | +0.08 | no |

**Card: numbers.** Kept in 6 of 8 eligible cells (kept with this card: yes); smaller than with the registered card in 4.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | -2.81 | -4.58 | <0.001 | <0.001 | yes | +0.65 | -0.21 | no |
| llama-instruct|physionet2012 | yes | +0.26 | +0.10 | 0.112 | 0.223 | no | +0.53 | +0.34 | yes |
| llama-instruct|physionet2019 | yes | -0.33 | -0.17 | 0.214 | 0.223 | no | +0.44 | +0.39 | yes |
| olmo3-instruct|physionet2012 | yes | +1.97 | +1.09 | <0.001 | <0.001 | yes | -0.36 | -0.37 | yes |
| falcon3-7b|physionet2019 | yes | +1.75 | +1.61 | <0.001 | <0.001 | yes | +0.08 | +0.19 | no |
| falcon3-7b|diabetes130 | yes | +1.90 | +1.96 | <0.001 | <0.001 | yes | +0.11 | -0.12 | no |
| yi15-9b|physionet2019 | yes | -0.77 | -0.90 | <0.001 | <0.001 | yes | +0.73 | +0.50 | yes |
| phi4-mini|physionet2012 | yes | +1.74 | +1.88 | <0.001 | <0.001 | yes | +0.00 | +0.01 | no |

**Card: cue.** Kept in 6 of 8 eligible cells (kept with this card: yes); smaller than with the registered card in 5.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | +1.10 | +0.17 | 0.279 | 0.279 | no | +4.55 | +4.54 | yes |
| llama-instruct|physionet2012 | yes | -1.81 | -1.85 | <0.001 | <0.001 | yes | -1.53 | -1.61 | no |
| llama-instruct|physionet2019 | yes | -2.03 | -1.65 | <0.001 | <0.001 | yes | -1.26 | -1.09 | no |
| olmo3-instruct|physionet2012 | yes | +1.25 | +0.71 | <0.001 | <0.001 | yes | -1.08 | -0.75 | yes |
| falcon3-7b|physionet2019 | yes | +0.46 | +0.32 | <0.001 | <0.001 | yes | -1.21 | -1.10 | yes |
| falcon3-7b|diabetes130 | yes | +0.36 | +0.58 | <0.001 | <0.001 | yes | -1.43 | -1.49 | yes |
| yi15-9b|physionet2019 | yes | -3.33 | -2.91 | <0.001 | <0.001 | yes | -1.83 | -1.51 | no |
| phi4-mini|physionet2012 | yes | -0.51 | -0.63 | <0.001 | <0.001 | no | -2.25 | -2.50 | yes |

**Card: matched.** Kept in 7 of 8 eligible cells (kept with this card: yes); smaller than with the registered card in 2.

| cell | valid | a | a′ | p | Holm p | kept | change (a) | change (a′) | smaller |
|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct|physionet2012 | yes | -3.37 | -4.23 | <0.001 | <0.001 | yes | +0.09 | +0.14 | no |
| llama-instruct|physionet2012 | yes | -0.95 | -0.92 | <0.001 | <0.001 | yes | -0.68 | -0.68 | no |
| llama-instruct|physionet2019 | yes | +0.11 | -0.65 | 0.765 | 0.765 | no | +0.87 | -0.09 | no |
| olmo3-instruct|physionet2012 | yes | +1.85 | +1.00 | <0.001 | <0.001 | yes | -0.48 | -0.47 | yes |
| falcon3-7b|physionet2019 | yes | +2.57 | +2.34 | <0.001 | <0.001 | yes | +0.90 | +0.92 | no |
| falcon3-7b|diabetes130 | yes | +2.19 | +2.46 | <0.001 | <0.001 | yes | +0.40 | +0.38 | no |
| yi15-9b|physionet2019 | yes | -1.32 | -1.51 | <0.001 | <0.001 | yes | +0.18 | -0.12 | no |
| phi4-mini|physionet2012 | yes | +1.29 | +1.31 | <0.001 | <0.001 | yes | -0.45 | -0.56 | yes |

**R6: supported** (the answer-word part is kept with all four changed cards).

**R7:** kept with either the direction words or the numbers alone; cue: a one-sentence direction cue is enough.

Sensitivity (pre-listed cells only): R6 supported; R7 not shown with either the direction words or the numbers alone; cue: a one-sentence direction cue is enough.

## Q8. Descriptive

**Q8b–c.** The alarm-framing statistic of the first study, the movement mean(la − ls), and the parts relative to the movement, per card:

| cell | card | alarm a | alarm a′ | movement | O / movement (a) | W / movement (a) |
|---|---|---|---|---|---|---|
| falcon3-7b|physionet2012 | words | +0.64 [+0.33, +0.94] | +0.31 [+0.11, +0.51] | +2.25 | +0.07 | +0.21 |
| falcon3-7b|physionet2012 | numbers | +1.27 [+0.99, +1.57] | +0.81 [+0.58, +1.05] | +0.54 | +1.47 | +0.88 |
| falcon3-7b|physionet2012 | cue | -1.67 [-1.84, -1.49] | -1.85 [-1.93, -1.78] | +4.29 | -0.24 | -0.15 |
| falcon3-7b|physionet2012 | matched | +1.12 [+0.83, +1.41] | +0.78 [+0.57, +1.00] | +2.73 | +0.30 | +0.11 |
| falcon3-7b|physionet2019 | words | +2.67 [+2.29, +2.94] | +2.33 [+2.16, +2.49] | +2.18 | +0.64 | +0.58 |
| falcon3-7b|physionet2019 | numbers | +2.26 [+2.00, +2.48] | +2.08 [+1.91, +2.25] | +1.02 | +1.72 | +0.50 |
| falcon3-7b|physionet2019 | cue | -1.38 [-1.60, -1.14] | -1.60 [-1.64, -1.56] | +6.77 | +0.07 | -0.27 |
| falcon3-7b|physionet2019 | matched | +3.08 [+2.71, +3.35] | +2.72 [+2.57, +2.87] | +3.80 | +0.68 | +0.13 |
| falcon3-7b|diabetes130 | words | +3.53 [+3.15, +3.90] | +4.06 [+3.88, +4.25] | +3.61 | +0.69 | +0.29 |
| falcon3-7b|diabetes130 | numbers | +2.31 [+2.07, +2.54] | +2.33 [+2.18, +2.48] | +1.90 | +1.00 | +0.21 |
| falcon3-7b|diabetes130 | cue | -1.70 [-1.85, -1.54] | -1.55 [-1.60, -1.50] | +8.92 | +0.04 | -0.23 |
| falcon3-7b|diabetes130 | matched | +2.86 [+2.60, +3.10] | +3.16 [+3.02, +3.30] | +4.64 | +0.47 | +0.14 |
| llama-instruct|physionet2012 | words | -0.83 [-1.13, -0.52] | -0.74 [-0.88, -0.59] | +1.52 | -0.09 | -0.46 |
| llama-instruct|physionet2012 | numbers | +1.29 [+0.96, +1.63] | +0.86 [+0.70, +1.03] | +0.33 | +0.78 | +3.09 |
| llama-instruct|physionet2012 | cue | -3.35 [-3.54, -3.16] | -3.51 [-3.60, -3.43] | +3.28 | -0.55 | -0.47 |
| llama-instruct|physionet2012 | matched | -0.59 [-0.89, -0.28] | -0.62 [-0.76, -0.47] | +1.32 | -0.72 | +0.27 |
| llama-instruct|physionet2019 | words | -0.85 [-0.97, -0.76] | -0.93 [-1.01, -0.85] | +1.29 | -0.44 | -0.22 |
| llama-instruct|physionet2019 | numbers | +0.60 [+0.50, +0.68] | +0.56 [+0.48, +0.64] | +0.47 | -0.70 | +2.00 |
| llama-instruct|physionet2019 | cue | -3.86 [-3.92, -3.81] | -3.87 [-3.92, -3.83] | +4.33 | -0.47 | -0.42 |
| llama-instruct|physionet2019 | matched | -0.54 [-0.68, -0.45] | -0.63 [-0.70, -0.55] | +1.39 | +0.08 | -0.47 |
| llama-instruct|diabetes130 | words | -1.64 [-1.83, -1.42] | -1.48 [-1.58, -1.38] | +2.92 | +0.07 | -0.63 |
| llama-instruct|diabetes130 | numbers | -0.47 [-0.64, -0.25] | -0.47 [-0.56, -0.37] | +1.24 | +2.21 | -2.60 |
| llama-instruct|diabetes130 | cue | -4.12 [-4.27, -3.97] | -3.98 [-4.04, -3.92] | +7.37 | +0.19 | -0.74 |
| llama-instruct|diabetes130 | matched | -2.09 [-2.28, -1.88] | -1.93 [-2.04, -1.82] | +2.67 | -0.04 | -0.75 |
| olmo3-instruct|physionet2012 | words | +1.57 [+0.81, +2.22] | +0.11 [-0.07, +0.28] | +2.05 | +1.19 | -0.42 |
| olmo3-instruct|physionet2012 | numbers | +1.58 [+0.91, +2.17] | -0.05 [-0.20, +0.09] | +0.90 | +2.19 | -0.44 |
| olmo3-instruct|physionet2012 | cue | +0.72 [+0.21, +1.16] | -0.26 [-0.38, -0.15] | +3.33 | +0.38 | -0.16 |
| olmo3-instruct|physionet2012 | matched | +1.51 [+0.76, +2.12] | -0.10 [-0.26, +0.05] | +1.96 | +0.95 | -0.18 |
| olmo3-instruct|physionet2019 | words | -1.71 [-2.22, -1.28] | -1.20 [-1.33, -1.07] | +2.79 | -0.26 | -0.35 |
| olmo3-instruct|physionet2019 | numbers | -0.46 [-0.79, -0.15] | -0.06 [-0.17, +0.04] | +1.06 | +1.08 | -1.52 |
| olmo3-instruct|physionet2019 | cue | +0.07 [-0.27, +0.37] | +0.30 [+0.23, +0.36] | +5.91 | +0.35 | -0.34 |
| olmo3-instruct|physionet2019 | matched | -1.87 [-2.29, -1.51] | -1.33 [-1.44, -1.22] | +1.90 | -0.20 | -0.78 |
| olmo3-instruct|diabetes130 | words | -0.68 [-1.49, -0.18] | +0.07 [-0.07, +0.22] | +4.03 | +0.08 | -0.25 |
| olmo3-instruct|diabetes130 | numbers | -0.28 [-0.48, +0.10] | -0.26 [-0.37, -0.15] | +2.05 | +2.87 | -3.00 |
| olmo3-instruct|diabetes130 | cue | -0.39 [-0.55, -0.20] | -0.49 [-0.56, -0.42] | +8.52 | +0.36 | -0.41 |
| olmo3-instruct|diabetes130 | matched | -1.08 [-1.54, -0.72] | -0.60 [-0.73, -0.47] | +2.21 | +4.11 | -4.60 |
| phi4-mini|physionet2012 | words | -0.65 [-0.82, -0.47] | -0.52 [-0.64, -0.39] | +3.03 | +0.58 | -0.79 |
| phi4-mini|physionet2012 | numbers | +0.36 [+0.19, +0.52] | +0.43 [+0.29, +0.57] | +0.35 | +4.90 | -3.90 |
| phi4-mini|physionet2012 | cue | -2.40 [-2.50, -2.29] | -2.46 [-2.49, -2.43] | +5.71 | -0.09 | -0.33 |
| phi4-mini|physionet2012 | matched | -0.28 [-0.43, -0.13] | -0.21 [-0.33, -0.09] | +2.23 | +0.58 | -0.70 |
| phi4-mini|physionet2019 | words | -0.20 [-0.60, +0.18] | -0.74 [-0.86, -0.62] | +3.06 | +0.41 | -0.48 |
| phi4-mini|physionet2019 | numbers | +0.10 [-0.23, +0.45] | -0.29 [-0.44, -0.14] | +0.99 | +2.02 | -1.91 |
| phi4-mini|physionet2019 | cue | -2.15 [-2.54, -1.70] | -2.72 [-2.76, -2.68] | +7.52 | -0.01 | -0.27 |
| phi4-mini|physionet2019 | matched | -0.63 [-1.03, -0.26] | -1.20 [-1.33, -1.07] | +2.53 | +0.42 | -0.67 |
| phi4-mini|diabetes130 | words | +2.87 [+1.45, +4.84] | -0.32 [-0.49, -0.15] | +3.58 | +0.99 | -0.19 |
| phi4-mini|diabetes130 | numbers | +1.73 [+0.44, +3.69] | -1.07 [-1.22, -0.94] | +1.70 | +5.10 | -4.08 |
| phi4-mini|diabetes130 | cue | -0.61 [-1.49, +0.52] | -1.39 [-1.44, -1.34] | +7.20 | +0.50 | -0.59 |
| phi4-mini|diabetes130 | matched | +1.72 [+0.43, +3.61] | -0.86 [-0.99, -0.74] | +3.33 | +1.96 | -1.44 |
| qwen-instruct|physionet2012 | words | -2.42 [-3.34, -1.52] | -2.90 [-3.52, -2.31] | +5.88 | -0.46 | +0.05 |
| qwen-instruct|physionet2012 | numbers | +0.47 [-0.37, +1.30] | -1.72 [-2.40, -1.06] | +1.67 | -1.68 | +1.97 |
| qwen-instruct|physionet2012 | cue | +3.04 [+2.29, +3.75] | +2.08 [+1.62, +2.55] | +14.73 | +0.07 | +0.13 |
| qwen-instruct|physionet2012 | matched | -1.72 [-2.59, -0.90] | -2.59 [-3.16, -2.02] | +5.96 | -0.56 | +0.28 |
| qwen-instruct|physionet2019 | words | -1.43 [-2.11, -0.85] | -1.77 [-2.21, -1.35] | +7.92 | +0.25 | -0.43 |
| qwen-instruct|physionet2019 | numbers | -1.51 [-2.02, -1.02] | -1.55 [-1.99, -1.13] | +2.81 | +0.37 | -0.91 |
| qwen-instruct|physionet2019 | cue | -1.47 [-2.06, -0.87] | -1.36 [-1.67, -1.06] | +20.38 | +0.04 | -0.11 |
| qwen-instruct|physionet2019 | matched | -2.27 [-2.86, -1.74] | -2.69 [-3.08, -2.31] | +9.05 | +0.10 | -0.36 |
| qwen-instruct|diabetes130 | words | +2.42 [-0.15, +5.48] | -0.63 [-1.05, -0.23] | +9.12 | -0.16 | +0.42 |
| qwen-instruct|diabetes130 | numbers | +2.12 [-0.22, +5.28] | -1.24 [-1.70, -0.78] | +4.24 | -0.43 | +0.93 |
| qwen-instruct|diabetes130 | cue | -4.45 [-6.41, -2.69] | -5.70 [-6.04, -5.38] | +18.39 | -0.16 | -0.08 |
| qwen-instruct|diabetes130 | matched | -2.21 [-4.77, +0.26] | -4.00 [-4.38, -3.64] | +10.42 | -0.09 | -0.12 |
| yi15-9b|physionet2012 | words | -0.11 [-0.38, +0.15] | -0.33 [-0.50, -0.16] | +2.04 | -0.28 | +0.22 |
| yi15-9b|physionet2012 | numbers | +1.65 [+1.42, +1.88] | +1.14 [+0.98, +1.30] | +0.27 | +1.49 | +4.52 |
| yi15-9b|physionet2012 | cue | -0.38 [-0.61, -0.15] | -0.74 [-0.88, -0.60] | +2.87 | -0.25 | +0.11 |
| yi15-9b|physionet2012 | matched | +0.70 [+0.47, +0.92] | +0.31 [+0.15, +0.46] | +1.88 | +0.09 | +0.28 |
| yi15-9b|physionet2019 | words | -1.14 [-1.38, -0.91] | -1.23 [-1.34, -1.12] | +1.56 | -1.10 | +0.37 |
| yi15-9b|physionet2019 | numbers | +0.11 [-0.10, +0.31] | +0.04 [-0.05, +0.14] | +0.19 | -4.10 | +4.69 |
| yi15-9b|physionet2019 | cue | -2.83 [-3.01, -2.66] | -2.76 [-2.83, -2.69] | +3.07 | -1.08 | +0.16 |
| yi15-9b|physionet2019 | matched | -0.80 [-1.01, -0.58] | -0.77 [-0.87, -0.67] | +1.47 | -0.90 | +0.36 |
| yi15-9b|diabetes130 | words | -1.18 [-1.28, -1.07] | -1.08 [-1.17, -0.99] | +2.76 | -0.17 | -0.26 |
| yi15-9b|diabetes130 | numbers | +0.37 [+0.27, +0.49] | +0.36 [+0.26, +0.45] | +0.59 | +1.73 | -1.09 |
| yi15-9b|diabetes130 | cue | -2.64 [-2.74, -2.53] | -2.60 [-2.68, -2.51] | +4.16 | -0.41 | -0.22 |
| yi15-9b|diabetes130 | matched | -0.21 [-0.31, -0.10] | -0.14 [-0.22, -0.05] | +2.56 | +0.55 | -0.63 |

**Q8a.** Registered-card prompts scored again, against the earlier study:

| model | earlier study | prompts | text differs | max |Δ| | median |Δ| | within 0.01 | same answer |
|---|---|---|---|---|---|---|---|
| llama-instruct | v1 | 10800 | 0 | 0.1094 | 0.0156 | 0.463 | 0.999 |
| olmo3-instruct | v2 | 10800 | 0 | 0.0469 | 0.0000 | 0.568 | 0.999 |
| qwen-instruct | v1 | 10800 | 0 | 0.8594 | 0.0312 | 0.185 | 0.999 |

**Q8e.** Memorization check of the new models (tabmemcheck; flag = at least one row completed verbatim):

| model | dataset | file | header characters matched | rows completed verbatim | flag |
|---|---|---|---|---|---|
| falcon3-7b | physionet2012 | Outcomes-a.txt | 3 | 0 | no |
| falcon3-7b | physionet2019 | p000001.psv | 0 | 0 | no |
| falcon3-7b | diabetes130 | diabetic_data.csv | 0 | 0 | no |
| yi15-9b | physionet2012 | Outcomes-a.txt | 7 | 0 | no |
| yi15-9b | physionet2019 | p000001.psv | 60 | 0 | no |
| yi15-9b | diabetes130 | diabetic_data.csv | 19 | 0 | no |
| phi4-mini | physionet2012 | Outcomes-a.txt | 1 | 0 | no |
| phi4-mini | physionet2019 | p000001.psv | 0 | 0 | no |
| phi4-mini | diabetes130 | diabetic_data.csv | None | 0 | no |

