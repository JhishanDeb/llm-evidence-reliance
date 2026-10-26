# Exploratory scale check: Qwen3 from 4B to 32B

Written on 26 October 2026, before any run of the models below. This file is committed and tagged `scale-plan`
before the first run. The run notebook downloads it at that tag and stops if a model id or revision below is missing.

## Status

This is an exploratory analysis. It is not part of either pre-registration (`prereg-v1.3`, `prereg-v2`) and it
changes no registered result. The paper reports it in its own labelled section, with the date and commit of this
plan.

## Question

The registered models of both studies have 7 to 8 billion parameters. Does the asymmetry persist in larger models
of the one family in which it replicated in all three datasets (Qwen3)? And does 4-bit quantization, which is needed
to fit the larger models on the Kaggle GPUs, change it?

## Models

Official 4-bit AWQ checkpoints from Qwen, run with float16 activations in vLLM 0.26.0 on 2× T4, thinking mode off
(as Qwen3-8B in the second study):

| Model | Revision | Role |
|---|---|---|
| Qwen/Qwen3-8B-AWQ | 4da05a8edb55c6046cce958586c33b61da07bb79 | quantization control: Qwen3-8B with 4-bit weights |
| Qwen/Qwen3-14B-AWQ | 31c69efc29464b6bb0aee1398b5a7b50a99340c3 | larger |
| Qwen/Qwen3-32B-AWQ | 0499c3ac83fdef8810b907a23894ba91e95eddd8 | largest; run only if it loads on 2× T4 |
| Qwen/Qwen3-4B-AWQ | 74d4bd2bd4bff9cafc9345221320bffb08b406a3 | smaller |

The runs follow this order: 8B, 14B, 32B, 4B. Before scoring, each run checks that the model loads, answers four
obvious questions correctly (the sanity questions of the second study's engineering check) and puts a median Yes+No
mass of at least 0.5 on the dev prompts. A model that fails is reported as "cannot be run", with the reason.

Reference: Qwen3-8B in float16 (revision b968826d9c46dd6066d109eabc6255188de91218) from the second study. Its
records are reused, not run again.

## Prompts and reading

The registered code of the second study (`ler_main.py`, main-2.0, SHA-256
0ba0853b4b52b897a9d9d8153d2c437eba2e7f45244b42bf4e082c5c4d05121e) builds every prompt and reads every answer. It is
not changed; the notebook checks its hash. Only the vLLM memory settings may differ, because the 32B model needs more
of the GPU memory. The test patients, cards and solved examples are the same as in both studies (the notebook checks
the dev fingerprints).

Each patient gets 13 prompts, all with example set A:

- Yes/No question, alarm framing: C0, C2, C4, C6
- Yes/No question, reversed framing: C0, C2, C4
- answer options: C0, C2, C4, each in both option orders

The reading is the top 1,000 log-probabilities (Deviation D5), as in both studies. The memorization check runs as in
the second study, for every model.

## Measures

All measures use the functions of main-2.0, for each model and dataset:

1. Validity, by the registered rules. Yes/No: median Yes+No mass at least 0.5 and C0 accuracy at least 0.55.
   Answer options: median A+B mass at least 0.5 and C0 accuracy of the averaged reading at least 0.55.
2. a and a′ in the alarm framing, with 95% intervals and the intersection–union p (10,000 paired bootstrap
   resamples, seed 2026).
3. Q3: the four-way reading with the reversed question.
4. a and a′ on the averaged answer-option reading.
5. Accuracy under C0, and C2 − C0 and C4 − C0 (exact McNemar).
6. Paired differences over the same patients, for a and a′, with the Yes/No question and with answer options:
   Qwen3-8B-AWQ minus Qwen3-8B (quantization), and each other size minus Qwen3-8B-AWQ (size).

## Reading rules

These rules are fixed now. There is no correction for multiple tests.

- A cell shows the pattern if it is valid, a < 0 and a′ < 0, and the intersection–union p < 0.05.
- A model shows the pattern if at least 2 of the 3 datasets do (the per-model rule of R1).
- Quantization: if Qwen3-8B-AWQ shows the pattern, 4-bit weights are taken not to remove it, and the size
  comparison is read on the 4-bit models. If it does not, the size comparison is reported but not interpreted.
- Size: the pattern "persists with size" if Qwen3-14B-AWQ and Qwen3-32B-AWQ (if it runs) show it. It "weakens with
  size" if a larger model does not show it while Qwen3-8B-AWQ does. Any other result is described as it is.
- The answer-option format is read with the same rules. It is described, not tested.
- Every cell is reported, including excluded ones (in an appendix).
- Nothing in this check changes R1, R2, R3 or any result of the first study.
