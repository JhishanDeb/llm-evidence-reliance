# Code

All code used for the study. The analysis rules are fixed in `PREREGISTRATION.md` (tags `prereg-v1` to `prereg-v1.3`); this folder only implements them.

## Files

| File | What it is |
|---|---|
| `ler_dev.py` | Shared module of the dev phase (version dev-1.1). Data loading, splits, XGBoost, evidence cards, prompts. |
| `ler_main.py` | Shared module of the main run (version main-1.0). The dev-phase code for data, splits, cards and prompts, unchanged, plus the test sample, the Yes/No reading (D2), the memorization check (D3.7) and the registered analysis (D1). |
| `notebooks/llm_evidence_reliance_dev_phase.ipynb` | Dev phase: 100 dev patients per dataset, Qwen2.5-7B-Instruct (Kaggle, 2x T4). |
| `notebooks/llm_evidence_reliance_main_run.ipynb` | Main run: one registered model per Kaggle session (2x T4). Writes `ler_main.py` from its own cell, checks the tagged protocol, rebuilds the data, checks the dev fingerprints, runs the memorization check and scores all 25,200 test prompts. |
| `notebooks/llm_evidence_reliance_analysis.ipynb` | The registered analysis on the four `main_<model>.zip` files (CPU). |
| `src/` | The notebook sources (percent format) and `build_nb.py`, which builds the notebooks from them. |
| `tests/` | Offline tests: unit tests of the analysis, equivalence of the main and dev modules, the Yes/No reading logic against a stand-in for vLLM, the memorization check, and generators for fake data and a tiny test model. |

## Versions

- `ler_main.py` (main-1.0): SHA-256 begins `10557671c645bb2e`. Every main run records this hash in its `run_log_<model>.json`, and the analysis notebook stops if the four runs used different code.
- `ler_dev.py` (dev-1.1): SHA-256 begins `d79a8ea1763b69e9`.
- Software in the runs: Python 3.13, torch 2.11.0+cu128, vLLM 0.26.0, xgboost 3.4.1, scikit-learn 1.6.1, pandas 2.3.3, numpy 2.1.3, tabmemcheck 0.1.6.

## Data

The notebooks download the three public datasets themselves: PhysioNet/CinC Challenge 2012 (ODC-By 1.0), PhysioNet/CinC Challenge 2019 (CC BY 4.0) and UCI Diabetes 130-US Hospitals (CC BY 4.0). No data are stored in this repository except derived per-patient predictions in `results/`.
