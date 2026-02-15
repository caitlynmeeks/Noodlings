#!/usr/bin/env bash
# ------------------------------------------------------------------
#   NoodleStudio Smoke Tests - Quick Sanity Check
#
#   Run this before committing to catch infrastructure regressions.
#   Exit code 0 = all good, nonzero = something broke.
#
#   Usage:
#     ./run_smoke_tests.sh
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=".:../.." ../../venv/bin/python -m pytest tests/test_smoke.py -v --tb=short "$@"
