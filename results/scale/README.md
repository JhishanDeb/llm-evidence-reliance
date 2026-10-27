# Results of the exploratory scale check (first plan)

Plan: `SCALE_CHECK.md`, tag `scale-plan` (commit 93ed379). Code: `code/scale/`. This check is exploratory and is not
part of either pre-registration.

## Runs

Kaggle, two T4 GPUs, vLLM 0.26.0, 4-bit AWQ weights with float16 activations, the top 1,000 log-probabilities.

| File | Model | Outcome | SHA-256 |
|---|---|---|---|
| `scale_qwen3-8b-awq.zip` | Qwen/Qwen3-8B-AWQ | ran: 23,400 prompts, 0 invalid, all six cells valid | `b4db5f218e6c43d96e9da51c9578d07151031b26beb1f584f1d3f94ddd71917e` |
| `scale_qwen3-14b-awq.zip` | Qwen/Qwen3-14B-AWQ | cannot be run: median Yes+No mass on the dev prompts 0.169 (rule: at least 0.5) | `269038efeba76d728ad42620f50679f08e97ae1ed3e55347ccd3779f9cddf141` |
| `scale_qwen3-32b-awq.zip` | Qwen/Qwen3-32B-AWQ | cannot be run: median Yes+No mass on the dev prompts 0.176 | `6b13a06d161faf1189fe6cfa5c67c28970b805efe5c5fcfa7e317b6ffddf03bd` |
| `scale_qwen3-4b-awq.zip` | Qwen/Qwen3-4B-AWQ | ran: 23,400 prompts, 0 invalid | `f8598277d42a445481d848ce93bb09a3e5e2469d847da08be4f0e5363721f09f` |

Every run used `ler_main.py` main-2.0 (SHA-256 beginning `0ba0853b4b52b897`) and `ler_scale.py` scale-1.0 (beginning
`5b9ce08693132ed5`), and checked the plan at the tag (SCALE_CHECK.md SHA-256 beginning `f2c1bba028998a98`).

## Diagnostic

`scale_diagnostic.zip` (SHA-256 `0033f9a45d1fdfd09fe67e081878a7d38fd64fe3896affbc0e79f242fc2c7c9d`): what the 8B, 14B
and 32B models put their probability on after "Answer:", on the dev prompts of the model check (dev patients only; no
accuracy and no asymmetry). Code: `code/scale/scale_diag.py` and
`code/scale/notebooks/llm_evidence_reliance_scale_diagnostic.ipynb`. The 14B and 32B models mostly start markdown
bold (" **") and then write the answer in bold. This led to the second plan (`SCALE_CHECK_2.md`).

## Analysis

`scale_analysis_results.zip` (SHA-256 `7ac549a09579eea421ea74947b7a1dfe7b846c1f1eaf4fe05e2ab009301fe9d2`) and
`report_scale.md`: the analysis notebook as committed at `scale-plan`, run on the four runs above and on
`results/v2/v2_qwen3-instruct.zip` (Qwen3-8B in float16, the reference).

In short: Qwen3-8B with 4-bit weights shows the pattern in 3 of 3 datasets, as it does in float16. The 14B and 32B
models could not be run, so the plan's size question has no answer from this plan. The 4B model shows the pattern in
1 of 3 datasets; in the other two, alarming evidence moves it more.
