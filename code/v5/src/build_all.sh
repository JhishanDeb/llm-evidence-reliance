#!/bin/bash
# Build the notebooks of the fifth study from their percent-format sources (run from code/v5).
set -e
P=${PYTHON:-python}
M="ler_main.py=ler_main.py ler_v3.py=ler_v3.py ler_v4.py=ler_v4.py ler_v5.py=ler_v5.py"
# The two engineering-check notebooks are kept exactly as they were run (they contain ler_v5.py v5-0.1); to rebuild
# them, use the sources src/engcheck_v5_src.py and src/engcheck_v5_run2_src.py with ler_v5.py v5-0.1.
$P src/build_nb.py src/main_run_v5_src.py notebooks/llm_evidence_reliance_main_run_v5.ipynb gpu $M
$P src/build_nb.py src/analysis_v5_src.py notebooks/llm_evidence_reliance_analysis_v5.ipynb cpu
