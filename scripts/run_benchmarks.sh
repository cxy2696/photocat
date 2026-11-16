#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -e ".[dev]"
python scripts/fetch_data.py
python -m photocat bench --dataset all --epochs 24 --ensemble 2
