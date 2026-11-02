#!/bin/bash
# Mock runs of the fourth study on the fake data (CPU, fake model), the engineering-check notebook offline, then the
# registered analysis and a check that it recovers the effects built into the fake models.
# usage: run_mock_v4.sh SCRATCH_DIR   (needs SCRATCH_DIR/fakedata, SCRATCH_DIR/nbenv and the repository in SCRATCH_DIR/repo)
#   llama-instruct: pull toward the second answer word in every format (W < 0 everywhere), none in plain text
#   qwen-instruct:  pull only with Yes/No (W with True/False near 0); plain text has almost no answer-word mass
#   olmo3-instruct: pull toward denying the named outcome (W with the negated question near 0)
#   falcon3-7b:     pull toward the first word (W > 0); no outcome asymmetry, but a shift toward the last example's
#                   outcome (O changes sign with the alarming-last order)
#   yi15-9b:        pull toward the first word, none in the paraphrased prompt
#   llama-base:     pull only in the chat format (the template produces it); qwen-base: no pull in either format
#   Llama-3.1-8B-Instruct stops after 5 chunks and is resumed from its own output (resume test)
S=$1
C=$S/repo/code/v4
export LER_MOCK=1 LER_DATA_DIR=$S/fakedata LER_P19_MIN_PER_SET=1000 LER_P12_MIN_PER_SET=1500 LER_TEST_MAX_PATIENTS=120 \
       LER_PROTOCOL_PATH=$S/repo/PREREGISTRATION.md LER_PROTOCOL_V2_PATH=$S/repo/PREREGISTRATION_V2.md \
       LER_PROTOCOL_V3_PATH=$S/repo/PREREGISTRATION_V3.md LER_PROTOCOL_V4_PATH=$S/repo/PREREGISTRATION_V4.md \
       LER_PILOT_IDS_PATH=$S/repo/pilot_record_ids.txt
R=$S/mock_runs_v4; rm -rf $R; mkdir -p $R/analysis
NB=$C/notebooks/llm_evidence_reliance_main_run_v4.ipynb
declare -A CFG
CFG["meta-llama/Llama-3.1-8B-Instruct"]='{"alarm_gain": 0.8, "safe_gain": 1.4, "lean": -0.6, "scale": 2.0, "seed": "li", "pull": {"Yes/No": 0.4, "True/False": 0.4}, "render_factor": {"plain": 0.0}}'
CFG["Qwen/Qwen2.5-7B-Instruct"]='{"alarm_gain": 0.6, "safe_gain": 1.7, "lean": 0.8, "scale": 2.0, "seed": "qi", "pull": {"Yes/No": 0.6, "True/False": 0.0}, "plain_mass": 0.05}'
CFG["allenai/Olmo-3-7B-Instruct"]='{"alarm_gain": 1.7, "safe_gain": 0.7, "lean": 0.0, "scale": 2.0, "seed": "oi", "mode": "named", "pull": {"Yes/No": 0.5, "True/False": 0.5}}'
CFG["tiiuae/Falcon3-7B-Instruct"]='{"alarm_gain": 1.0, "safe_gain": 1.0, "lean": 0.2, "scale": 2.0, "seed": "f3", "pull": {"Yes/No": -0.5, "True/False": -0.5}, "order_pull": 0.4}'
CFG["01-ai/Yi-1.5-9B-Chat-16K"]='{"alarm_gain": 1.2, "safe_gain": 0.9, "lean": -0.2, "scale": 2.0, "seed": "y1", "pull": {"Yes/No": -0.4, "True/False": -0.4}, "para_factor": 0.0, "gen_flip": 0.3}'
CFG["meta-llama/Llama-3.1-8B"]='{"alarm_gain": 1.0, "safe_gain": 1.0, "lean": 0.0, "scale": 1.5, "seed": "lb", "pull": {"Yes/No": 0.4}, "render_factor": {"plain": 0.0, "chat": 1.0}}'
CFG["Qwen/Qwen2.5-7B"]='{"alarm_gain": 1.0, "safe_gain": 1.0, "lean": 0.0, "scale": 1.5, "seed": "qb", "pull": {"Yes/No": 0.0}}'
for M in "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "allenai/Olmo-3-7B-Instruct" \
         "tiiuae/Falcon3-7B-Instruct" "01-ai/Yi-1.5-9B-Chat-16K" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B"; do
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
$S/nbenv/bin/python $C/tests/run_nb.py $C/notebooks/llm_evidence_reliance_engcheck_v4.ipynb $R/engcheck \
  $R/engcheck.out.ipynb > $R/engcheck.log 2>&1
echo "engineering check notebook (offline): $(tail -1 $R/engcheck.log | cut -c1-300)"
cp $R/*/v4_*.zip $R/analysis/
ls $R/analysis
LER_TEST_BOOT=400 LER_INPUT_DIRS=$R/analysis $S/nbenv/bin/python $C/tests/run_nb.py \
  $C/notebooks/llm_evidence_reliance_analysis_v4.ipynb $R/analysis_work $R/analysis.out.ipynb
$S/nbenv/bin/python -I $C/tests/check_mock_v4.py $R/analysis_work/analysis_v4_results/results_v4.json
