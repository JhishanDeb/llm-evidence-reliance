# Main runs and the registered analysis

This folder holds the raw outputs of the four main runs and the output of the registered analysis. The analysis rules are fixed in `PREREGISTRATION.md` (tag `prereg-v1.1`; Deviation D5 at tag `prereg-v1.3`). The code that produced these files is in `code/` and was committed before the analysis was run.

## Files

| File | What it is | SHA-256 |
|---|---|---|
| `main_qwen-instruct.zip` | Run 1: Qwen/Qwen2.5-7B-Instruct, 2026-10-20 | `6d6b8d9d38ec947f0d10aaebf6845d7d5b10508275f996b67eee7fa10da75a88` |
| `main_qwen-base.zip` | Run 2: Qwen/Qwen2.5-7B, 2026-10-20 | `a9b3157c0408481b117ef0a6b06e519bc572e43cad02ddd0570a013ca522945d` |
| `main_llama-instruct.zip` | Run 3: meta-llama/Llama-3.1-8B-Instruct, 2026-10-21 | `593bc904e281d715c5dc359c6376b53295bbd733c289bab4610c4a7659d14a15` |
| `main_llama-base.zip` | Run 4: meta-llama/Llama-3.1-8B, 2026-10-21 | `2862ff00d415bf13f3b8e6f906e3cd23f3981f4c6e83d3ca78e5317da9f8d256` |
| `analysis_results.zip` | Output of the analysis notebook: `report.md`, `results.json`, `cells_summary.csv` | `0737fa3f17a0a3d48b165e0f00bd0b8338516f6c238fb6390ca9d11d95426b3c` |
| `report.md` | A copy of the report in `analysis_results.zip`, so that it can be read on GitHub | `78ea3a36b0850ed717da12745a86a21684625137dc7413613e7878b794c16f7e` |

Each `main_<model>.zip` contains:

- `records_<model>.csv.gz`: one row per prompt (25,200 rows), with the Yes/No log-probabilities and the log-odds of the alarming outcome;
- `run_log_<model>.json`: settings, model revision, software versions, the protocol and fingerprint checks, the reading check, the validity summary and timings;
- `memcheck_<model>.json`: the memorization check;
- `example_prompts_<model>.md`: example prompts as the model received them;
- `ler_main.py`: the code used for the run (version main-1.0, SHA-256 begins `10557671c645bb2e`, the same in all four runs).

## How the files were made

- **Main runs:** `code/notebooks/llm_evidence_reliance_main_run.ipynb` on Kaggle (2x T4), one model per session. Every run passed the protocol check, matched the dev-phase fingerprints, used the same code and scored the same 25,200 prompt texts. All four runs read the top 1,000 log-probabilities (Deviation D5).
- **Analysis:** `code/notebooks/llm_evidence_reliance_analysis.ipynb` on Google Colab (CPU), 2026-10-21, on the four zip files above. The analysis is deterministic. An earlier run of the same notebook on another machine gave the same numbers (the largest relative difference was about 2 x 10^-12, which is floating-point rounding). The order of the rows in the report tables depends on the order in which the zip files are found, and it does not affect any result.

## Checking a file

```
sha256sum main_*.zip analysis_results.zip report.md
```

The output should match the table above.
