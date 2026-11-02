# Code of the fourth study (`PREREGISTRATION_V4.md`, release `prereg-v4`)

- `ler_main.py`: the registered code of the earlier studies (main-2.0), byte-identical (SHA-256 checked by every run).
- `ler_v3.py`: the registered code of the third study (v3-1.1), byte-identical (SHA-256 checked by every run).
- `ler_v4.py`: what the fourth study adds: the True/False, negated and paraphrased prompts and the net-number card,
  with checks that each new prompt is the registered prompt with exactly the registered change; the grids and models;
  the plain and chat formats; generated answers and their parsing; the smoke test and the engineering check; a fake
  model for the CPU tests; and the registered analysis (validity, O and W of every pair, the 2 x 2 comparison of R10,
  the movement comparison of R13, the decision rules of R8 to R13 and the descriptive Q14).
- `notebooks/`: the engineering check (run once before the tag), the run notebook (one Kaggle session per model) and
  the analysis notebook (CPU, run once after all runs). They are built from `src/*_src.py` by `src/build_all.sh`.
- `tests/`: `unit_tests_v4.py` (prompts, readings, estimators against the third study's records, decision rules),
  `test_engcheck_v4.py` (the engineering check with a stand-in for vLLM), `run_mock_v4.sh` (seven runs with fake models
  on fake data, with a resume, the engineering-check notebook offline, the analysis notebook, and `check_mock_v4.py`,
  which checks that the analysis recovers the effects built into the fake models). `run_all_v4.sh` runs all of them.

## Order of work

1. Run `llm_evidence_reliance_engcheck_v4.ipynb` on Kaggle (GPU T4 x2, Internet on, secret `HF_TOKEN`). It uses dev
   patients only and does not need the release.
2. Record its decisions in `ler_v4.py` (`ENGCHECK_DECISIONS`, `ENGCHECK_ZIP_SHA256`) and in Section 10 of
   `PREREGISTRATION_V4.md`; commit both with `results/v4/engcheck_v4.zip`; create the release `prereg-v4`.
3. Run `llm_evidence_reliance_main_run_v4.ipynb` once per model, in the order of Section 9 of the protocol.
4. Run `llm_evidence_reliance_analysis_v4.ipynb` once on all `v4_*.zip` files (and, for the replay comparison, the
   earlier runs from `results/`).
