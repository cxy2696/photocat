#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -e ".[dev]"
for mode in balanced oversample; do
  echo "=== mitigation: $mode ==="
  python -m photocat demo --fast --mitigation "$mode" --out "runs/fast_$mode" --no-figures
done
