# Second study: outputs

The analysis rules are fixed in `PREREGISTRATION_V2.md` (tag `prereg-v2`, with Deviation V2-D1 committed on 2026-10-22). The code that produced these files is in `code/v2/` at the same tag.

## Files

| File | What it is | SHA-256 |
|---|---|---|
| `engcheck_v2.zip` | The engineering check before registration (2026-10-22): dev patients only, technical quantities only. | `068c0351ae00111c03d7d8e95a3a1a73c716cfe0c50f713c0d0b6e4c1fb82a4d` |
| `v2_olmo3-instruct.zip` | allenai/Olmo-3-7B-Instruct: Yes/No grid, answer-option grid, both controls | `e225c812a4c90c61c64a4c4917b2cbcb20b24beccb3c7ceee73c72755eb38a4b` |
| `v2_qwen3-instruct.zip` | Qwen/Qwen3-8B (thinking off): Yes/No grid, answer-option grid, example-order control | `3a42b60bfd27806007074d5e85ad0440227a0a5a26d511bc5dc5a54c65904bb3` |
| `v2_granite3-instruct.zip` | ibm-granite/granite-3.3-8b-instruct: Yes/No grid, answer-option grid, example-order control | `680159158c91848bbfb6a281d6ee0ce967fbe4a381ff3d27f9f7b3fe3efd7373` |
| `v2_olmo3-base.zip` | allenai/Olmo-3-1025-7B (base): Yes/No grid | `3044c95e080fa4d7ea58d3108d780cf21e60270059f8a30c0c954627e6623af5` |
| `v2_olmo3-sft.zip` | allenai/Olmo-3-7B-Instruct-SFT: Yes/No grid | `8bec7331415397a2ecff0ed1874b8259a3b8cb84c5f7a7f35d5cdc60e3c4705d` |
| `v2_olmo3-dpo.zip` | allenai/Olmo-3-7B-Instruct-DPO: Yes/No grid | `0a25b9d1e2330f07c51b2e2f9041caf7d4be8ce6631f84c95c7b9651089ee944` |
| `v2_qwen-instruct.zip` | Qwen/Qwen2.5-7B-Instruct (first-study model): answer-option grid, example-order control | `cebe935821494a4d3bd751a3f9b784833bc669ea288ee63be92d1552caf227cf` |
| `v2_llama-instruct.zip` | meta-llama/Llama-3.1-8B-Instruct (first-study model): answer-option grid, example-order control | `d23a4ce6074dbcc7bc713d6ced28f02daa59ce5041aedd4dd54d42dadea0fe11` |
| `analysis_v2_results.zip` | Output of the registered analysis: `report_v2.md`, `results_v2.json`, `cells_summary_v2.csv` | `8b6ae0f263e8482d3a32293d1d19f5ee647735bd7db64f42836aa0bbf103d250` |
| `report_v2.md` | A copy of the report in `analysis_v2_results.zip`, so that it can be read on GitHub | `e68330844e9d639dff45355f64d48791ca99835912493abe9e9742b2895ebcc5` |

Each `v2_<model>.zip` contains `records_<model>.csv.gz` (one row per prompt), `run_log_<model>.json` (settings, pinned and loaded revision, protocol check at the tag, reading checks, validity, timings), `example_prompts_<model>.md`, `memcheck_<model>.json` (models new to this study) and the `ler_main.py` of the run (main-2.0, SHA-256 begins `0ba0853b4b52b897`, the same in all eight runs).

## How the files were made

- **Runs:** `code/v2/notebooks/llm_evidence_reliance_main_run_v2.ipynb` on Kaggle (two T4 GPUs, vLLM 0.26.0, float16), one model per session, 2026-10-22 to 2026-10-26. Every run checked the protocol at the tag `prereg-v2`, loaded the pinned revision, matched the dev-phase fingerprints and scored every registered prompt with no invalid reading. The Yes/No prompts are text-identical to the first study's.
- **Analysis:** `code/v2/notebooks/llm_evidence_reliance_analysis_v2.ipynb` on Kaggle (CPU), 2026-10-26, on the eight run outputs and the first study's `main_qwen-instruct.zip` and `main_llama-instruct.zip` (in `results/main/`). The analysis is deterministic. An earlier run of the same notebook on another machine gave a byte-identical `report_v2.md`; the numbers in `results_v2.json` agree to a relative difference of at most 10^-8 (floating-point rounding).
- **Deviation V2-D1:** the base-format control did not meet the answer-word mass criterion (median mass 0.015 to 0.018) and is not interpretable; its numbers in the report are shown but not used.
