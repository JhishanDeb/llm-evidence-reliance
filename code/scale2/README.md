# Code of the exploratory scale check, second plan (two-step reading)

The plan is fixed in `SCALE_CHECK_2.md` (tag `scale-plan-2`). This check is exploratory: it is not part of either
pre-registration and changes no registered result or result of the first scale plan.

## Files

| File | What it is |
|---|---|
| `ler_scale2.py` | Module, version scale-2.0. It uses `code/v2/ler_main.py` (main-2.0) and `code/scale/ler_scale.py` (scale-1.0) without any change and adds the two-step reading: P(word) = P1(word) + P1(" **") × P2(word), where the second step reads the same prompt with the token " **" added. Also the models of the second plan, the check against the plan and the model check with the two-step reading. |
| `notebooks/llm_evidence_reliance_scale2_run.ipynb` | One run per model (Kaggle, two T4 GPUs): the 8B, 14B and 32B Qwen3 4-bit models. Its output files carry the suffix `-2step`, so they never clash with the first plan's files. |
| `notebooks/llm_evidence_reliance_scale2_analysis.ipynb` | The analysis (CPU) of the second plan. It needs the second plan's runs, the first plan's 8B run (the bridge between the two readings) and 4B run (reused, as the plan says), all in `results/scale/` once committed; `results/v2/v2_qwen3-instruct.zip` is optional. |
| `src/` | Notebook sources (percent format) and `build_nb.py`, which builds the notebooks from them. |
| `tests/` | `test_scale2_logic.py` (the two-step reading against a hand computation on a stand-in for vLLM, with and without imputation; the requests; the model check; the plan check) and `run_mock_scale2.sh` (mock runs of the whole pipeline, including a model that cannot be run and a run that is resumed after a stop). The shell script contains the paths of the machine it was run on. |
