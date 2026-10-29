#!/usr/bin/env bash
# Code-Verifikation (NICHT die wissenschaftlichen Experimente, die liegen in experiments/).
# Unit-, Golden-, Property/Fuzz-, Differential-/Metamorph-Tests + Mutation-Tests, dann deterministischer Bericht.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY -m pytest -q -p no:cacheprovider --junitxml=verification/junit.xml
rm -rf mutants
$PY -m mutmut run --max-children "${JOBS:-6}" >/dev/null 2>&1 || true
$PY -m mutmut results --all true > verification/mutmut_results.txt 2>/dev/null || $PY -m mutmut results > verification/mutmut_results.txt
$PY verification/report.py
