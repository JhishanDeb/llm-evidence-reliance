#!/bin/bash
# Build the three notebooks of the fourth study from their percent-format sources (run from code/v4).
set -e
P=${PYTHON:-python}
M="ler_main.py=ler_main.py ler_v3.py=ler_v3.py ler_v4.py=ler_v4.py"
$P src/build_nb.py src/engcheck_v4_src.py notebooks/llm_evidence_reliance_engcheck_v4.ipynb gpu $M
$P src/build_nb.py src/main_run_v4_src.py notebooks/llm_evidence_reliance_main_run_v4.ipynb gpu $M
$P src/build_nb.py src/analysis_v4_src.py notebooks/llm_evidence_reliance_analysis_v4.ipynb cpu
