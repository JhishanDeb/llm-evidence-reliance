# Code of the exploratory scale check

The plan is fixed in `SCALE_CHECK.md` (tag `scale-plan`). This check is exploratory: it is not part of either
pre-registration and changes no registered result.

## Files

| File | What it is |
|---|---|
| `ler_scale.py` | Module, version scale-1.0. It uses the registered `code/v2/ler_main.py` (main-2.0) without any change and adds the four Qwen3 4-bit models with their pinned revisions, the grid of 13 prompts per patient, the check against the plan, a vLLM backend whose memory settings can be set per model, and the model check that runs before each run. |
| `notebooks/llm_evidence_reliance_scale_run.ipynb` | One run per model (Kaggle, two T4 GPUs). It checks the code hash and the plan at the tag, downloads the pinned revision, checks the model, runs the memorization check and scores the test prompts. It stops cleanly after 11 hours, and a second session continues from the saved checkpoint. |
| `notebooks/llm_evidence_reliance_scale_analysis.ipynb` | The analysis (CPU) of the scale runs, with the second study's Qwen3-8B run (`results/v2/v2_qwen3-instruct.zip`) as the float16 reference. |
| `src/` | Notebook sources (percent format) and `build_nb.py`, which builds the notebooks from them. |
| `tests/` | `test_scale_logic.py` (the backend reads exactly like the registered one, the model check, the plan check, the grid) and `run_mock_scale.sh` (mock runs of the whole pipeline, including a model that cannot be run and a run that is resumed after a stop). The shell script contains the paths of the machine it was run on. |

## Versions

- Software in the runs: vLLM 0.26.0 on Kaggle (two T4 GPUs, float16 activations, 4-bit AWQ weights), xgboost 3.4.1, as in both studies.
- `ler_main.py` used by every run: SHA-256 `0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e` (main-2.0, the copy in `code/v2/`).
