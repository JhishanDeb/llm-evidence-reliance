#!/bin/bash
# All offline tests of the fourth study. usage: run_all_v4.sh SCRATCH_DIR
#   SCRATCH_DIR holds: nbenv/ (Python with numpy, pandas, scikit-learn, scipy, xgboost 3.4.1, nbformat, nbclient,
#   ipykernel, transformers, tokenizers), repo/ (this repository), fakedata/ (built here if missing) and runs3/ (the
#   unpacked outputs of the third study: v3_qwen-instruct/, v3_olmo3-instruct/ and analysis_work/ with results_v3.json).
set -e
S=$1
C=$S/repo/code/v4
P=$S/nbenv/bin/python
export LER_P12_MIN_PER_SET=1500 LER_P19_MIN_PER_SET=1000
if [ ! -d $S/fakedata/physionet2012 ]; then
  $P $C/tests/make_fake_p12.py $S/fakedata 1500
  $P $S/repo/code/tests/make_fake_p19.py $S/fakedata 1000
  $P $S/repo/code/tests/make_fake_d130.py $S/fakedata 30000
fi
(cd $C && PYTHON=$P ./src/build_all.sh)
$P -I $C/tests/unit_tests_v4.py $C $S/fakedata $S/runs3 $S/repo/PREREGISTRATION_V4.md
PYTHONPATH=$C/tests/fakevllm:$C/tests/faketorch $P $C/tests/test_engcheck_v4.py $C $S/fakedata $S/engtest4
$C/tests/run_mock_v4.sh $S
