# LLM Evidence Reliance

Do LLMs reason with model explanations (SHAP), or defer to them one way?

This work is pre-registered in three studies. The first study's protocol is in PREREGISTRATION.md (release prereg-v1), with later deviations dated in the same file (releases prereg-v1.1 to prereg-v1.3). The second study, which tests the first study's results on new models and asks what drives them, is registered in PREREGISTRATION_V2.md (release prereg-v2). The third study, which tests the two parts of the asymmetry (the outcome and the answer word) in models from three further developers and asks which part of the evidence card drives them, is registered in PREREGISTRATION_V3.md (release prereg-v3).

## Repository layout

- `PREREGISTRATION.md`: the first study's registered protocol and the deviations D1 to D5.
- `PREREGISTRATION_V2.md`: the second study's registered protocol.
- `PREREGISTRATION_V3.md`: the third study's registered protocol.
- `analysis/`: the simulations behind Deviation D1, with their outputs.
- `code/`: all code and notebooks of the first study (dev phase, main runs, registered analysis).
- `code/v2/`: the code, notebooks and tests of the second study.
- `code/v3/`: the code, notebooks and tests of the third study.
- `results/dev/`: output of the dev phase, used only for engineering checks.
- `results/main/`: raw outputs of the first study's four main runs and its registered analysis (`report.md`).
- `results/v2/`: outputs of the second study: the engineering check before registration, the eight runs and the registered analysis.
- `results/v3/`: outputs of the third study (the engineering check before registration; the runs and the analysis are added later).
