#!/bin/bash
# Mock scale-check runs on the fake data (CPU, fake model), then the scale analysis.
#   scenario "all":    the four models run; 8B-AWQ equals the float16 reference mock (seed q3), larger models move
#                      more with safe-pointing evidence, 4B hardly at all
#   scenario "fail32": the 32B model check fails (cannot be run)
#   scenario "partial": the 14B run stops at the time budget, then resumes from its output
S=/tmp/claude-0/-home-claude/daecffe7-6520-5bab-a7bd-ff09280e6f7d/scratchpad
SCEN=${1:-all}
export LER_MOCK=1 LER_DATA_DIR=$S/devnb/fakedata LER_P19_MIN_PER_SET=1000 LER_P12_MIN_PER_SET=4000 \
       LER_PILOT_IDS_PATH=$S/ler_repo/repo/pilot_record_ids.txt LER_SCALE_PLAN_PATH=$S/scale/SCALE_CHECK.md
R=$S/scale/runs/mock_$SCEN; rm -rf $R; mkdir -p $R/analysis
declare -A CFG
CFG["Qwen/Qwen3-8B-AWQ"]='{"alarm_gain": 0.7, "safe_gain": 1.6, "lean": 0.3, "scale": 2.5, "seed": "q3", "choice_alarm_gain": 1.1, "choice_safe_gain": 1.1, "letter_bias": -0.3}'
CFG["Qwen/Qwen3-14B-AWQ"]='{"alarm_gain": 0.6, "safe_gain": 1.9, "lean": 0.2, "scale": 2.5, "seed": "q14", "choice_alarm_gain": 0.9, "choice_safe_gain": 1.3, "letter_bias": -0.2}'
CFG["Qwen/Qwen3-32B-AWQ"]='{"alarm_gain": 0.5, "safe_gain": 2.1, "lean": 0.1, "scale": 2.5, "seed": "q32", "choice_alarm_gain": 0.8, "choice_safe_gain": 1.5, "letter_bias": -0.1}'
CFG["Qwen/Qwen3-4B-AWQ"]='{"alarm_gain": 1.0, "safe_gain": 1.05, "lean": 0.4, "scale": 2.0, "seed": "q4", "letter_bias": 0.5}'
NB=$S/scale/llm_evidence_reliance_scale_run.ipynb
for M in "Qwen/Qwen3-8B-AWQ" "Qwen/Qwen3-14B-AWQ" "Qwen/Qwen3-32B-AWQ" "Qwen/Qwen3-4B-AWQ"; do
  T=$(echo $M | tr '/.' '__')
  EXTRA=""
  if [ "$SCEN" = "fail32" ] && [ "$M" = "Qwen/Qwen3-32B-AWQ" ]; then EXTRA="LER_SCALE_FAIL_CHECK=1"; fi
  if [ "$SCEN" = "partial" ] && [ "$M" = "Qwen/Qwen3-14B-AWQ" ]; then
    # first session: a time budget so small that it stops after the first chunks
    env LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" LER_TEST_STOP_AT_CHUNK=40 \
      $S/nbenv/bin/python $S/devnb/tests/run_nb.py $NB $R/${T}_s1 $R/${T}_s1.out.ipynb > $R/${T}_s1.log 2>&1
    echo "$M session 1: $(tail -1 $R/${T}_s1.log) | files: $(ls $R/${T}_s1/scale_outputs_* | tr '\n' ' ')"
    EXTRA="LER_EXTRA_ROOTS=$R/${T}_s1"
  fi
  env $EXTRA LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" \
    $S/nbenv/bin/python $S/devnb/tests/run_nb.py $NB $R/$T $R/$T.out.ipynb > $R/$T.log 2>&1
  echo "$M: $(tail -1 $R/$T.log)"
done
cp $R/*/scale_*.zip $R/analysis/
cp $S/v2/runs/mock_all/analysis/v2_qwen3-instruct.zip $R/analysis/
ls $R/analysis
if [ -f $S/scale/llm_evidence_reliance_scale_analysis.ipynb ]; then
  LER_TEST_BOOT=300 LER_INPUT_DIRS=$R/analysis $S/nbenv/bin/python $S/devnb/tests/run_nb.py $S/scale/llm_evidence_reliance_scale_analysis.ipynb \
    $R/analysis_work $R/analysis.out.ipynb
fi
