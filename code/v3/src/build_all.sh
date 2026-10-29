#!/bin/bash
# Build the three notebooks of the third study from their percent-format sources (run from code/v3).
set -e
P=${PYTHON:-python}
$P src/build_nb.py src/engcheck_v3_src.py notebooks/llm_evidence_reliance_engcheck_v3.ipynb gpu ler_main.py=ler_main.py ler_v3.py=ler_v3.py
$P src/build_nb.py src/main_run_v3_src.py notebooks/llm_evidence_reliance_main_run_v3.ipynb gpu ler_main.py=ler_main.py ler_v3.py=ler_v3.py
$P src/build_nb.py src/analysis_v3_src.py notebooks/llm_evidence_reliance_analysis_v3.ipynb cpu
