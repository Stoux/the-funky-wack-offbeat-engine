#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1

python -m offbeat.worker
