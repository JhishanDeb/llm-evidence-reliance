# Study 5: run outputs and the registered analysis

The fifth study is registered as the release `prereg-v5` and on OSF. The protocol is `PREREGISTRATION_V5.md` in the root of this repository. Deviation V5-D1, about the order of the runs, was added on 8 November 2026, before the analysis was run.

## Run outputs (Kaggle, two T4 GPUs, 5 to 7 November 2026)

Each zip holds the records, the generated answers, the run log, example prompts and the code of the run. Every run used `ler_v5.py` v5-1.0, the protocol at `prereg-v5`, the registered model revision, and the fresh samples of Section 3. No reading was invalid.

| File | Model | Scored prompts | Generated answers | Hours |
|---|---|---|---|---|
| `v5_qwen3-instruct.zip` | Qwen/Qwen3-8B | 38,828 | 2,400 | 3.32 |
| `v5_falcon3-7b.zip` | tiiuae/Falcon3-7B-Instruct | 47,964 | 2,400 | 3.65 |
| `v5_llama-instruct.zip` | meta-llama/Llama-3.1-8B-Instruct | 47,964 | 2,400 | 3.99 |
| `v5_olmo3-instruct.zip` | allenai/Olmo-3-7B-Instruct | 47,964 | 2,400 | 3.50 |
| `v5_yi15-9b.zip` | 01-ai/Yi-1.5-9B-Chat-16K | 47,964 | 2,400 | 5.97 |
| `v5_qwen-instruct.zip` | Qwen/Qwen2.5-7B-Instruct | 47,964 | 2,400 | 3.58 |
| `v5_granite3-instruct.zip` | ibm-granite/granite-3.3-8b-instruct | 47,964 | 2,400 | 4.95 |

## The registered analysis

`analysis_v5_results.zip` (with `results_v5.json`, `report_v5.md` and the `ler_v5.py` it used) is the output of `code/v5/notebooks/llm_evidence_reliance_analysis_v5.ipynb`, run exactly as it is at the tag on the seven zips above, with 10,000 bootstrap resamples (8 November 2026, 33 minutes on a CPU). `report_v5.md` is the same report, unpacked for reading.

Registered decisions: R15 not supported (the placebo card strengthened the preference for "No" in Qwen3-8B in both datasets, the opposite of the prediction); R16 not supported (3 of 4 cells supported, Granite 3.3 on PhysioNet 2012 opposite); R17 not supported (1 of 4 cells).

## SHA-256

- `analysis_v5_results.zip`: 51b95a640c4d827878280165ea11345b93344a1f49d5d52ed0b87101ee9ba170
- `v5_falcon3-7b.zip`: a85181ede8661c75cd099f55563957a1b95ac737a42061846f7dbed408844757
- `v5_granite3-instruct.zip`: 490a4be5ecc2bfe842e8d6aa7d60386958518f1a36a43558e8e4dbdeefc33c4f
- `v5_llama-instruct.zip`: dd432fd428705666da9bd4ce7a2dcaf2bf3bff3222ddb1c176fae230cb508e63
- `v5_olmo3-instruct.zip`: a1b7b33a75725f8b3a0ce67e74482c1355f3acde26aa0a9788d16a06a40b6322
- `v5_qwen-instruct.zip`: aceef24e56d2935a9c5239d8c2aafd30cf45311bdd48ba3fea9f4fd62594f649
- `v5_qwen3-instruct.zip`: 4d68ac097ca2c927bdc1571cd893bfaa3d0025aeafd0c9ec6f67656c529efe5b
- `v5_yi15-9b.zip`: b30a52854889c86b2eb3018352189ae03d90dc82b273ef6fa9aa79f270c2848b
- `report_v5.md`: 55d9c47a64da317d3dd7fc91ea341d9d0421b026b8ecb5a15424b2cb88a552cb
