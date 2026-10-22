#!/bin/bash
# eight mock second-study runs + two mock first-study runs (main-1.0) on the fake data, then the v2 analysis
S=/tmp/claude-0/-home-claude/daecffe7-6520-5bab-a7bd-ff09280e6f7d/scratchpad
export LER_MOCK=1 LER_DATA_DIR=$S/devnb/fakedata LER_P19_MIN_PER_SET=1000 LER_P12_MIN_PER_SET=4000 \
       LER_PROTOCOL_PATH=$S/devv/PREREGISTRATION_v1.3.md LER_PROTOCOL_V2_PATH=$S/v2/PREREGISTRATION_V2.md \
       LER_PILOT_IDS_PATH=$S/ler_repo/repo/pilot_record_ids.txt
R=$S/v2/runs/mock_all; rm -rf $R; mkdir -p $R/analysis
declare -A CFG
# new models: safe-direction asymmetry in Yes/No; Olmo keeps it in A/B (outcome), Qwen3 loses it in A/B (answer word)
CFG["allenai/Olmo-3-7B-Instruct"]='{"alarm_gain": 0.6, "safe_gain": 1.8, "lean": -0.5, "scale": 2.0, "seed": "oi", "letter_bias": 0.4}'
CFG["Qwen/Qwen3-8B"]='{"alarm_gain": 0.7, "safe_gain": 1.6, "lean": 0.3, "scale": 2.5, "seed": "q3", "choice_alarm_gain": 1.1, "choice_safe_gain": 1.1, "letter_bias": -0.3}'
CFG["ibm-granite/granite-3.3-8b-instruct"]='{"alarm_gain": 0.8, "safe_gain": 1.5, "lean": 0.0, "scale": 1.5, "seed": "m3", "letter_bias": 0.2}'
# Olmo 3 stages: the asymmetry grows through post-training
CFG["allenai/Olmo-3-1025-7B"]='{"alarm_gain": 1.0, "safe_gain": 1.0, "lean": 0.2, "scale": 1.5, "seed": "ob"}'
CFG["allenai/Olmo-3-7B-Instruct-SFT"]='{"alarm_gain": 0.9, "safe_gain": 1.2, "lean": 0.0, "scale": 1.8, "seed": "os"}'
CFG["allenai/Olmo-3-7B-Instruct-DPO"]='{"alarm_gain": 0.7, "safe_gain": 1.6, "lean": -0.3, "scale": 2.0, "seed": "od"}'
# first-study models: answer-option grid only
CFG["Qwen/Qwen2.5-7B-Instruct"]='{"alarm_gain": 0.6, "safe_gain": 1.8, "lean": 1.0, "scale": 3.0, "seed": "qi", "letter_bias": 0.5}'
CFG["meta-llama/Llama-3.1-8B-Instruct"]='{"alarm_gain": 0.8, "safe_gain": 1.4, "lean": -1.0, "scale": 2.0, "seed": "li"}'
for M in "${!CFG[@]}"; do
  T=$(echo $M | tr '/.' '__')
  LER_MODEL_ID="$M" LER_MOCK_CFG="${CFG[$M]}" $S/nbenv/bin/python $S/devnb/tests/run_nb.py $S/v2/llm_evidence_reliance_main_run_v2.ipynb $R/$T $R/$T.out.ipynb > $R/$T.log 2>&1
  echo "$M: $(tail -1 $R/$T.log)"
done
# first-study mock runs of the two instruct models (main-1.0 notebook), for the framing comparison
for M in "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"; do
  T=v1_$(echo $M | tr '/.' '__')
  V1CFG=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); [d.pop(k, None) for k in ('letter_bias','choice_alarm_gain','choice_safe_gain')]; print(json.dumps(d))" "${CFG[$M]}")
  LER_PROTOCOL_PATH=$S/devv/PREREGISTRATION_v1.3.md LER_MODEL_ID="$M" LER_MOCK_CFG="$V1CFG" \
    $S/nbenv/bin/python $S/devnb/tests/run_nb.py $S/mainnb/llm_evidence_reliance_main_run.ipynb $R/$T $R/$T.out.ipynb > $R/$T.log 2>&1
  echo "v1 $M: $(tail -1 $R/$T.log)"
done
cp $R/*/v2_*.zip $R/*/main_*.zip $R/analysis/
ls $R/analysis
LER_INPUT_DIRS=$R/analysis $S/nbenv/bin/python $S/devnb/tests/run_nb.py $S/v2/llm_evidence_reliance_analysis_v2.ipynb $R/analysis_work $R/analysis.out.ipynb
