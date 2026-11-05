#!/bin/bash
# Mock runs of the fifth study on the fake data (CPU, fake model), then the registered analysis and a check that it
# recovers the effects built into the fake models.
# usage: run_mock_v5.sh SCRATCH_DIR CODE_DIR PROTOCOL_DIR
#   SCRATCH_DIR holds fakedata/ and nbenv/; PROTOCOL_DIR holds PREREGISTRATION.md, _V2, _V3, _V4, _V5 and
#   pilot_record_ids.txt.
#   qwen3-instruct:   strong preference for 'No' that any evidence block weakens (Δλ > 0), no asymmetry in updating
#                     (W against the placebo near 0); True/False not run (as the engineering check decided)
#   falcon3-7b:       updating pull toward 'Yes' (W > 0)      yi15-9b: the same, smaller
#   granite3-instruct: updating pull toward 'No' (W < 0); ignores the size of the impacts (Δm near 0)
#   llama-instruct, qwen-instruct, olmo3-instruct: other combinations
#   Qwen3-8B stops after 5 chunks and is resumed from its own output (resume test)
S=$1; C=$2; PR=$3
export LER_MOCK=1 LER_DATA_DIR=$S/fakedata LER_P19_MIN_PER_SET=1000 LER_P12_MIN_PER_SET=1500 LER_TEST_MAX_PATIENTS=0 LER_TEST_PER_CLASS=15 \
       LER_PROTOCOL_PATH=$PR/PREREGISTRATION.md LER_PROTOCOL_V2_PATH=$PR/PREREGISTRATION_V2.md \
       LER_PROTOCOL_V3_PATH=$PR/PREREGISTRATION_V3.md LER_PROTOCOL_V4_PATH=$PR/PREREGISTRATION_V4.md \
       LER_PROTOCOL_V5_PATH=$PR/PREREGISTRATION_V5.md LER_PILOT_IDS_PATH=$PR/pilot_record_ids.txt \
       LER_MOCK_DECISIONS='{"not_run": [], "grids_not_run": {"qwen3-instruct": ["tf"]}, "gen_per_class": 150}'
R=$S/mock_runs_v5; rm -rf $R; mkdir -p $R/analysis
NB=$C/notebooks/llm_evidence_reliance_main_run_v5.ipynb
declare -A CFG
CFG["Qwen/Qwen3-8B"]='{"prior": 3.0, "card_prior_factor": 0.6, "pull": 0.0, "signal": 5.0, "seed": "q3", "alarm_gain": 1.0, "safe_gain": 1.0, "scale": 1.0}'
CFG["tiiuae/Falcon3-7B-Instruct"]='{"prior": 0.1, "pull": -0.5, "signal": 1.5, "seed": "f3", "alarm_gain": 1.0, "safe_gain": 1.0, "scale": 1.0}'
CFG["01-ai/Yi-1.5-9B-Chat-16K"]='{"prior": 0.3, "pull": -0.4, "signal": 1.5, "seed": "y1", "alarm_gain": 1.0, "safe_gain": 1.0, "scale": 1.0}'
CFG["ibm-granite/granite-3.3-8b-instruct"]='{"prior": 0.2, "pull": 0.5, "dose_power": 0.0, "signal": 1.5, "seed": "g3", "alarm_gain": 1.0, "safe_gain": 1.0, "scale": 1.0}'
CFG["meta-llama/Llama-3.1-8B-Instruct"]='{"prior": 0.6, "pull": 0.3, "gen_flip": 1.0, "signal": 1.5, "seed": "li", "alarm_gain": 0.8, "safe_gain": 1.4, "scale": 1.0}'
CFG["Qwen/Qwen2.5-7B-Instruct"]='{"prior": 1.0, "pull": 0.0, "signal": 1.5, "seed": "qi", "alarm_gain": 1.0, "safe_gain": 1.0, "scale": 1.0}'
CFG["allenai/Olmo-3-7B-Instruct"]='{"prior": 0.5, "pull": 0.4, "letter_prior": 0.5, "signal": 1.5, "seed": "oi", "alarm_gain": 1.0, "safe_gain": 1.0, "scale": 1.0}'
for M in "Qwen/Qwen3-8B" "tiiuae/Falcon3-7B-Instruct" "01-ai/Yi-1.5-9B-Chat-16K" "ibm-granite/granite-3.3-8b-instruct" \
         "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "allenai/Olmo-3-7B-Instruct"; do
  T=$(echo $M | tr '/.' '__')
  EXTRA=""
  if [ "$M" = "Qwen/Qwen3-8B" ]; then
    env LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" LER_TEST_STOP_AT_CHUNK=5 \
      $S/nbenv/bin/python $C/tests/run_nb.py $NB $R/${T}_s1 $R/${T}_s1.out.ipynb > $R/${T}_s1.log 2>&1
    echo "$M session 1: $(grep -o 'OFFLINE TEST: stopped at chunk [0-9][0-9]*' $R/${T}_s1.log | head -1)"
    EXTRA="LER_EXTRA_ROOTS=$R/${T}_s1"
  fi
  env $EXTRA LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" \
    $S/nbenv/bin/python $C/tests/run_nb.py $NB $R/$T $R/$T.out.ipynb > $R/$T.log 2>&1
  echo "$M: $(tail -1 $R/$T.log | cut -c1-300)"
done
cp $R/*/v5_*.zip $R/analysis/
ls $R/analysis
LER_TEST_BOOT=400 LER_INPUT_DIRS=$R/analysis $S/nbenv/bin/python $C/tests/run_nb.py \
  $C/notebooks/llm_evidence_reliance_analysis_v5.ipynb $R/analysis_work $R/analysis.out.ipynb
$S/nbenv/bin/python -I $C/tests/check_mock_v5.py $R/analysis_work/analysis_v5_results/results_v5.json
