# Results of the exploratory scale check (second plan, two-step reading)

Plan: `SCALE_CHECK_2.md`, tag `scale-plan-2` (commit 371f14e). Code: `code/scale2/`. This check is exploratory and is
not part of either pre-registration.

## Runs

Kaggle, two T4 GPUs, vLLM 0.26.0, 4-bit AWQ weights with float16 activations, the two-step reading with the top 1,000
log-probabilities in each step.

| File | Model | Outcome | SHA-256 |
|---|---|---|---|
| `scale2_qwen3-8b-awq-2step.zip` | Qwen/Qwen3-8B-AWQ | ran: 23,400 prompts, 0 invalid, all six cells valid | `309781e932de97ddd705229b89ce77e883cb9da55dd0a8f0d52f6bb05473d17c` |
| `scale2_qwen3-14b-awq-2step.zip` | Qwen/Qwen3-14B-AWQ | ran: 23,400 prompts, 0 invalid, all six cells valid | `2e945c00bb8afe620af7c590297010f219b1ee6a03c046f5e4794ad2206677d9` |
| `scale2_qwen3-32b-awq-2step.zip` | Qwen/Qwen3-32B-AWQ | ran: 23,400 prompts, 0 invalid, all six cells valid | `5cdba8f271b0646ad88caa6cb439c2469b5b3755bc7cba496038e06cbf8da624` |

Every run used `ler_main.py` main-2.0 (SHA-256 beginning `0ba0853b4b52b897`), `ler_scale.py` scale-1.0 (beginning
`5b9ce08693132ed5`) and `ler_scale2.py` scale-2.0 (beginning `4edb907560fb36a6`), and checked the plan at the tag
(SCALE_CHECK_2.md SHA-256 beginning `de98890bed5cf859`). As the plan says, the 4B model was not run again; its
first-plan records (`results/scale/scale_qwen3-4b-awq.zip`) are used.

## Analysis

`scale2_analysis_results.zip` (SHA-256 `7f7d88ff0529cc5d3f5265f7b3b11633ced6830702eb64a7d71d49c1a474148d`) and
`report_scale2.md`: the analysis notebook as committed at `scale-plan-2`, run on the three runs above, the first
plan's 8B and 4B runs (`results/scale/`) and `results/v2/v2_qwen3-instruct.zip`.

In short, with the reading rules of the plan:

- Bridge: under the two-step reading, Qwen3-8B-AWQ shows the pattern in 3 of 3 datasets, as under the first reading.
- Size, Yes/No question: the pattern persists with size. Qwen3-14B-AWQ and Qwen3-32B-AWQ show it in 3 of 3 datasets;
  the 4B model shows it in 1 of 3.
- Answer options (described, not tested): the 8B model shows the pattern in 3 of 3 datasets, the 14B model in 1 of 3
  and the 32B model in 2 of 3.

On the T4 GPUs, the 4-bit kernels give slightly different numbers for the same prompt when it is batched differently.
For the 8B model, the alarm log-odds of the two runs correlate at 0.998, and 98.7% of the predictions agree.
