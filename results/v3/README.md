# Results of the third study

Registered in `PREREGISTRATION_V3.md` (release `prereg-v3`, commit f69d8c0). The registered report is `report_v3.md`.

## Files

| File | What it is | SHA-256 |
|---|---|---|
| `engcheck_v3.zip` | The engineering check before registration (2026-10-29, Kaggle, two T4 GPUs): dev patients only, technical quantities only. It contains its summary (`engcheck_v3.md`), the full result per model (`engcheck_v3.json`), the log and the code it ran (`ler_main.py` main-2.0, `ler_v3.py` v3-1.0). | 6984a4d96861563db7d35f66d0880bc63cfc2c3366a77e7d35738497088c6e00 |
| `v3_olmo3-instruct.zip` | Run of allenai/Olmo-3-7B-Instruct (earlier model, 36,000 prompts) | f7175d83ce4fc8dfb43662736b7a45803d338916ee7a7e93bd459df018bcd3b6 |
| `v3_llama-instruct.zip` | Run of meta-llama/Llama-3.1-8B-Instruct (earlier model, 36,000 prompts) | f05c4e75779e2d042416c0819ef3fda14b5fdf706b6a120c744267fedabd941e |
| `v3_qwen-instruct.zip` | Run of Qwen/Qwen2.5-7B-Instruct (earlier model, 36,000 prompts) | 0b5cc121ded27ea19fd57d882df4b253b962a0890b98841fcd42f47f60ef7529 |
| `v3_falcon3-7b.zip` | Run of tiiuae/Falcon3-7B-Instruct (new model, 46,800 prompts) | c176666500d23e8f2c6a7f38b4f72fb332322a14ea9f5f4c75dbe29f1be48e69 |
| `v3_yi15-9b.zip` | Run of 01-ai/Yi-1.5-9B-Chat-16K (new model, 46,800 prompts) | 52a9f1f5c97bc58e0ca67aa2715d3e9e9154548f733590ca99b47fbd8d1f3de8 |
| `v3_phi4-mini.zip` | Run of microsoft/Phi-4-mini-instruct (new model, 46,800 prompts) | 84720baa3a78ab66c82149ad7b059b19755240cf7c050d0fad281838946443c1 |
| `analysis_v3_results.zip` | The registered analysis (`llm_evidence_reliance_analysis_v3.ipynb` as tagged), run once on Kaggle (CPU, 2026-10-30, 10,000 resamples) on the six runs and, for the replay comparison (Q8a), on `results/main/main_qwen-instruct.zip`, `results/main/main_llama-instruct.zip` and `results/v2/v2_olmo3-instruct.zip`. It contains `results_v3.json`, `report_v3.md` and the `ler_v3.py` it used. | 681639219b3db0b204d958cba4905db85c82fa02b79d6473697166b11644b093 |
| `report_v3.md` | The registered report, as in `analysis_v3_results.zip` | 88225e0cb6d80ed8a9d87dab56b81099da1fdd932399b1c8723af13098dace49 |
| `supplement_d5.py` | The imputation bound of Deviation D5, computed by a separate script (Deviation V3-D1 in `PREREGISTRATION_V3.md`) | 650deb14080ea3af1956a6ee29fa1050786218edeedbf297995ab5461d316cb1 |
| `supplement_d5.json` | Its output | 6af83365f5328417892c9c3953bdbabe4726277f6475a2c02b597d7c14b09c15 |

## Runs

Every run used `ler_main.py` main-2.0 (SHA-256 begins 0ba0853b4b52b897) and `ler_v3.py` v3-1.1 (89d4c0c0ed995e82), checked the protocol files at `prereg-v3` (`PREREGISTRATION_V3.md` c9a2f0bc2c92baf4), loaded the registered revision, rebuilt the same patients, cards and solved examples (all fingerprints match the dev phase), read the top 1,000 log-probabilities with vLLM 0.26.0 on two T4 GPUs, and scored every registered prompt. No reading is invalid, and the memorization check flagged nothing for the three new models.

| Model | Revision | Prompts | Records with an imputed variant | Session time |
|---|---|---|---|---|
| allenai/Olmo-3-7B-Instruct | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc | 36,000 | 0.2% | 1.9 h |
| meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 | 36,000 | 0% | 1.9 h |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | 36,000 | 11.0% | 1.9 h |
| tiiuae/Falcon3-7B-Instruct | 1e57a0ecd176c7c139f289c60a74e57f887c3dfb | 46,800 | 2.1% | 3.0 h |
| 01-ai/Yi-1.5-9B-Chat-16K | 2f8753bed95bd4fede3fd64f8305428620b4b212 | 46,800 | 0% | 5.1 h |
| microsoft/Phi-4-mini-instruct | cfbefacb99257ffa30c83adab238a50856ac3083 | 46,800 | 52.2% | 1.9 h |

## Reproduction

The same analysis notebook was also run in a separate environment before the official run. Its `report_v3.md` is byte-identical to the official one, and every number in `results_v3.json` agrees to a relative difference of at most 5 × 10⁻¹⁰.
