#!/bin/bash
# All offline tests of the third study. usage: run_all_v3.sh SCRATCH_DIR
#   SCRATCH_DIR holds: nbenv/ (Python with numpy, pandas, scikit-learn, scipy, xgboost 3.4.1, nbformat, nbclient,
#   ipykernel, transformers, tokenizers, tabmemcheck 0.1.6), repo/ (this repository), runs/ (the unpacked outputs of the
#   earlier runs main_qwen-instruct/ and v2_olmo3-instruct/), review_analyses.json (the paper's exploratory split).
set -e
S=$1
C=$S/repo/code/v3
P=$S/nbenv/bin/python
export LER_P12_MIN_PER_SET=1500 LER_P19_MIN_PER_SET=1000
if [ ! -d $S/fakedata/physionet2012 ]; then
  $P $C/tests/make_fake_p12.py $S/fakedata 1500
  $P $S/repo/code/tests/make_fake_p19.py $S/fakedata 1000
  $P $S/repo/code/tests/make_fake_d130.py $S/fakedata 30000
fi
(cd $C && PYTHON=$P ./src/build_all.sh)
$P -I $C/tests/unit_tests_v3.py $C $S/fakedata $S/runs $S/review_analyses.json $S/repo/PREREGISTRATION_V3.md
PYTHONPATH=$C/tests/fakevllm:$C/tests/faketorch $P $C/tests/test_engcheck_v3.py $C $S/fakedata $S/engtest
$C/tests/run_mock_v3.sh $S
