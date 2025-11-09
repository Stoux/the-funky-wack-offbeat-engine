#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a local Python 3.9 virtual environment and install deps (ML mandatory)
# Usage: scripts/bootstrap_venv.sh [python_executable]
# Examples:
#   scripts/bootstrap_venv.sh
#   scripts/bootstrap_venn.sh python3.9

PY=python3.9

for arg in "$@"; do
  PY="$arg"
  break
done

echo "[bootstrap] Using Python: ${PY}"

# Ensure pip is available
${PY} -m ensurepip --upgrade || true
${PY} -m pip install --upgrade pip wheel setuptools

# Create venv if missing
if [ ! -d .venv ]; then
  echo "[bootstrap] Creating virtualenv at .venv"
  ${PY} -m venv .venv
fi

# Activate and install requirements
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

echo "[bootstrap] Done. Activate with: source .venv/bin/activate" 
