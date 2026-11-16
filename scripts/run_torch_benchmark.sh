#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -e ".[dev,torch]"
python -m photocat bench --dataset gap --backend torch --ensemble 4 --out runs/bench_gap_torch
