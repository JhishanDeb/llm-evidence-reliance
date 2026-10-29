#!/bin/bash
# Mock runs of the third study on the fake data (CPU, fake model), then the registered analysis.
# usage: run_mock_v3.sh SCRATCH_DIR   (needs SCRATCH_DIR/fakedata, SCRATCH_DIR/nbenv and the repository in SCRATCH_DIR/repo)
#   the three earlier models: an outcome part toward safe (Qwen, Llama) or alarm (Olmo) and a pull toward "No";
#     Olmo loses the pull with the one-sentence cue
#   the three registered new models: Falcon3 and Yi with a pull toward "No", Phi-4-mini without one
#   Llama stops after 5 chunks and is resumed from its own output (resume test)
S=$1
C=$S/repo/code/v3
export LER_MOCK=1 LER_DATA_DIR=$S/fakedata LER_P19_MIN_PER_SET=1000 LER_P12_MIN_PER_SET=1500 LER_TEST_MAX_PATIENTS=120 \
       LER_PROTOCOL_PATH=$S/repo/PREREGISTRATION.md LER_PROTOCOL_V2_PATH=$S/repo/PREREGISTRATION_V2.md \
       LER_PROTOCOL_V3_PATH=$S/repo/PREREGISTRATION_V3.md LER_PILOT_IDS_PATH=$S/repo/pilot_record_ids.txt
R=$S/mock_runs; rm -rf $R; mkdir -p $R/analysis
NB=$C/notebooks/llm_evidence_reliance_main_run_v3.ipynb
declare -A CFG
CFG["Qwen/Qwen2.5-7B-Instruct"]='{"alarm_gain": 0.6, "safe_gain": 1.7, "lean": 0.8, "scale": 2.0, "seed": "qi", "no_pull": 0.6}'
CFG["meta-llama/Llama-3.1-8B-Instruct"]='{"alarm_gain": 0.8, "safe_gain": 1.4, "lean": -0.6, "scale": 2.0, "seed": "li", "no_pull": 0.4}'
CFG["allenai/Olmo-3-7B-Instruct"]='{"alarm_gain": 1.7, "safe_gain": 0.7, "lean": 0.0, "scale": 2.0, "seed": "oi", "no_pull": {"full": 0.5, "words": 0.5, "numbers": 0.5, "cue": 0.0, "matched": 0.5}}'
CFG["tiiuae/Falcon3-7B-Instruct"]='{"alarm_gain": 0.7, "safe_gain": 1.5, "lean": 0.2, "scale": 2.0, "seed": "g4", "no_pull": 0.5, "letter_bias": 0.3}'
CFG["01-ai/Yi-1.5-9B-Chat-16K"]='{"alarm_gain": 1.0, "safe_gain": 1.0, "lean": -0.2, "scale": 2.0, "seed": "y1", "no_pull": 0.5, "letter_bias": -0.2}'
CFG["microsoft/Phi-4-mini-instruct"]='{"alarm_gain": 1.6, "safe_gain": 0.7, "lean": 0.1, "scale": 2.0, "seed": "ex", "no_pull": 0.0}'
for M in "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "allenai/Olmo-3-7B-Instruct" \
         "tiiuae/Falcon3-7B-Instruct" "01-ai/Yi-1.5-9B-Chat-16K" "microsoft/Phi-4-mini-instruct"; do
  T=$(echo $M | tr '/.' '__')
  EXTRA=""
  if [ "$M" = "meta-llama/Llama-3.1-8B-Instruct" ]; then
    env LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" LER_TEST_STOP_AT_CHUNK=5 \
      $S/nbenv/bin/python $C/tests/run_nb.py $NB $R/${T}_s1 $R/${T}_s1.out.ipynb > $R/${T}_s1.log 2>&1
    echo "$M session 1: $(grep -o 'OFFLINE TEST: stopped at chunk [0-9][0-9]*' $R/${T}_s1.log | head -1)"
    EXTRA="LER_EXTRA_ROOTS=$R/${T}_s1"
  fi
  env $EXTRA LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" \
    $S/nbenv/bin/python $C/tests/run_nb.py $NB $R/$T $R/$T.out.ipynb > $R/$T.log 2>&1
  echo "$M: $(tail -1 $R/$T.log | cut -c1-300)"
done
env LER_MODEL_ID="Qwen/Qwen2.5-7B-Instruct" $S/nbenv/bin/python $C/tests/run_nb.py \
  $C/notebooks/llm_evidence_reliance_engcheck_v3.ipynb $R/engcheck $R/engcheck.out.ipynb > $R/engcheck.log 2>&1
echo "engineering check notebook (offline): $(tail -1 $R/engcheck.log | cut -c1-300)"
cp $R/*/v3_*.zip $R/analysis/
ls $R/analysis
LER_TEST_BOOT=500 LER_INPUT_DIRS=$R/analysis $S/nbenv/bin/python $C/tests/run_nb.py \
  $C/notebooks/llm_evidence_reliance_analysis_v3.ipynb $R/analysis_work $R/analysis.out.ipynb
