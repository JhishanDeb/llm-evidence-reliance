# Code of the third study

The analysis rules are fixed in `PREREGISTRATION_V3.md` (release `prereg-v3`); this folder only implements them.

## Files

| File | What it is |
|---|---|
| `ler_main.py` | The second study's shared module, version main-2.0, byte-identical (SHA-256 `0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e`). Data, splits, XGBoost, cards, prompts, the reading and the statistics a and a′. Every notebook checks the hash. |
| `ler_v3.py` | What the third study adds (version v3-1.0): the card formats (no numbers, no direction words, a one-sentence cue, the card worded for the outcome of the reversed question), the prompt grid, the models and the engineering-check candidates, the check against `PREREGISTRATION_V3.md`, the model window, the smoke test and engineering check, a fake model for the CPU tests, and the registered analysis (the outcome part O and the answer-word part W for every card, and the decision rules of R4 to R7). |
| `notebooks/llm_evidence_reliance_engcheck_v3.ipynb` | The engineering check before registration (dev patients only, technical quantities only). Its output decides which three new models are registered. |
| `notebooks/llm_evidence_reliance_main_run_v3.ipynb` | One run per registered model (Kaggle, two T4 GPUs). Checks the code and the protocol at the tag, downloads the pinned revision, runs the memorization check (new models) and scores every registered test prompt. |
| `notebooks/llm_evidence_reliance_analysis_v3.ipynb` | The registered analysis (CPU), run once on the six run outputs. |
| `src/` | Notebook sources (percent format), `build_nb.py` and `build_all.sh`, which build the notebooks from them. |
| `tests/` | Offline tests (below). |

## Tests (CPU, no GPU and no model)

- `unit_tests_v3.py`: `ler_main.py` is main-2.0; the registered prompts are unchanged; every changed-card prompt is the registered prompt with exactly the registered change; the protocol strings are in `PREREGISTRATION_V3.md`; the vectorised a and a′ equal `ler_main.py` on real records of the earlier studies, and a weighted fit with resample counts equals refitting on the resampled rows; the split reproduces the paper's exploratory split exactly; the pre-listed cells are exactly the conclusive cells of the three earlier models; the decision rules of R4 to R7 on hand-made cells; the model-window rule.
- `test_engcheck_v3.py`: the engineering check and the smoke test against a stand-in for vLLM (`fakevllm/`) and torch (`faketorch/`) with a tiny tokenizer.
- `run_mock_v3.sh`: the whole pipeline with a fake model on fake data: six runs (one stopped part-way and resumed), the engineering-check notebook and the analysis notebook. The fake models have known parts (for example a pull toward "No" that one model loses with the cue card), and the analysis recovers them.
- `make_fake_p12.py` and the generators of the first study (`code/tests/make_fake_p19.py`, `code/tests/make_fake_d130.py`) make the fake data; `run_nb.py` executes a notebook offline.

## How to run the study (Kaggle, two T4 GPUs)

1. **Engineering check** (done on 2026-10-29; output in `results/v3/engcheck_v3.zip`). Import `notebooks/llm_evidence_reliance_engcheck_v3.ipynb`, set Accelerator GPU T4 x2 and Internet on, add the `HF_TOKEN` secret (accept the terms of `CohereLabs/aya-expanse-8b` on Hugging Face first, as for Llama), and Save & Run All. About 2 to 2.5 hours. Download `engcheck_v3.zip`.
2. **Registration.** The first three eligible new candidates (Falcon3-7B-Instruct, Yi-1.5-9B-Chat-16K, Phi-4-mini-instruct) and the revisions the check loaded are written into `ler_v3.py` (`PINNED_NEW`) and into Sections 4 and 10 of `PREREGISTRATION_V3.md`, with the SHA-256 of `engcheck_v3.zip`. The notebooks are rebuilt (`src/build_all.sh`). Commit everything, with `engcheck_v3.zip` in `results/v3/`, and create the GitHub release `prereg-v3`.
3. **Six runs.** Import `notebooks/llm_evidence_reliance_main_run_v3.ipynb`, set `MODEL_ID` in the first code cell, Save & Run All, one model per session, in this order: `allenai/Olmo-3-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `tiiuae/Falcon3-7B-Instruct`, `01-ai/Yi-1.5-9B-Chat-16K`, `microsoft/Phi-4-mini-instruct`. About 3 hours for each earlier model and for Phi-4-mini, about 4 hours for Falcon3 and about 6 to 7 hours for Yi. Download each `v3_<model>.zip`. If a run stops part-way, attach its output (Add Input → Your Work) and run again; it resumes.
4. **Analysis.** Upload the six zip files (and, for the replay comparison, `results/main/main_qwen-instruct.zip`, `results/main/main_llama-instruct.zip` and `results/v2/v2_olmo3-instruct.zip`) as one dataset, attach it to `notebooks/llm_evidence_reliance_analysis_v3.ipynb` and Save & Run All (CPU). Commit the outputs to `results/v3/`.

## Versions

- `ler_v3.py` v3-1.1 is the registered version. The engineering check ran v3-1.0 (SHA-256 `148e1990bcf15905c07657c2aad54099afdb4bc8d64b6830dbf8d4665781e61d`, inside `results/v3/engcheck_v3.zip`); v3-1.1 differs from it only by the three pinned new models, the hashes of the check's output and code, and a more robust reading of the result lines of the check and smoke-test processes (`parse_result`).
- Software in the runs: vLLM 0.26.0 on Kaggle (two T4 GPUs, float16), xgboost 3.4.1, as in both earlier studies.
