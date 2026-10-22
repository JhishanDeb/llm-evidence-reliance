# Code of the second study

The analysis rules are fixed in `PREREGISTRATION_V2.md` (release `prereg-v2`); this folder only implements them.

## Files

| File | What it is |
|---|---|
| `ler_main.py` | Shared module, version main-2.0. It keeps every function of main-1.0 (the first study's code in `code/ler_main.py`) and adds the answer-option framing, the A/B reading, the example-order and base-format controls, the models of the second study with their pinned revisions, the protocol check against `PREREGISTRATION_V2.md`, the engineering check, and the second study's analysis. |
| `notebooks/llm_evidence_reliance_engcheck_v2.ipynb` | The engineering check run before registration (dev patients only, technical quantities only). Kept exactly as it was run; its output is `results/v2/engcheck_v2.zip`. |
| `notebooks/llm_evidence_reliance_main_run_v2.ipynb` | One run per registered model (Kaggle, two T4 GPUs). Checks the protocol at the tag, downloads the pinned revision, runs the memorization check (models new to this study) and scores every registered test prompt. |
| `notebooks/llm_evidence_reliance_analysis_v2.ipynb` | The registered analysis of the second study (CPU), run once on the eight run outputs and the first study's outputs of its two instruct models. |
| `src/` | Notebook sources (percent format) and `build_nb.py`, which builds the notebooks from them. |
| `tests/` | Offline tests: first-study prompts byte-identical to main-1.0 (`test_v2_equivalence.py`), the reading logic against a stand-in for vLLM (`test_vllm_logic_v2.py`), the decision rules on simulated data with a known truth (`unit_tests_v2.py`), first-study analysis numbers unchanged on the real records (`test_v1_analysis_unchanged.py`), and a mock run of the whole pipeline (`run_mock_v2.sh`). The shell scripts contain the paths of the machine they were run on. |

## Versions

- Software in the runs: vLLM 0.26.0 on Kaggle (two T4 GPUs, float16), xgboost 3.4.1, as in the first study.
- The engineering check used `ler_main.py` with SHA-256 beginning `a6a1a3d6c12390c0`. The registered version differs from it only by the pinned model revisions, Granite in place of Ministral, and the record of the check's output.
