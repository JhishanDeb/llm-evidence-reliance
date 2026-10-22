#!/bin/bash
S=/tmp/claude-0/-home-claude/daecffe7-6520-5bab-a7bd-ff09280e6f7d/scratchpad
cd $S/v2
echo "== equivalence"; $S/nbenv/bin/python tests/test_v2_equivalence.py ler_main_v1_frozen.py ler_main.py $S/devnb/fakedata 2>&1 | tail -2
echo "== vllm logic"; $S/nbenv/bin/python tests/test_vllm_logic_v2.py $S/v2/tests/fakevllm $S/v2 $S/devnb/tiny_yn $S/v2/ler_main_v1_frozen.py 2>&1 | grep -v "Warning\|ENGCHECK_RESULT" | tail -2
echo "== unit"; $S/nbenv/bin/python tests/unit_tests_v2.py $S/v2 2>&1 | tail -1
echo "== first-study analysis unchanged"; $S/nbenv/bin/python tests/test_v1_analysis_unchanged.py ler_main_v1_frozen.py ler_main.py $S/analysis_real/analysis_inputs 2>&1 | tail -1
echo "== mock pipeline"; bash tests/run_mock_v2.sh 2>&1 | grep -v IPKernelApp | tail -14 | cut -c1-600
