#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs
.venv/bin/python exp/dai14.py --stage units > logs/dai14_units.log 2>&1 || { echo FAIL_UNITS > logs/dai14.fail; exit 1; }
.venv/bin/python exp/dai14.py --stage summary > logs/dai14_summary.log 2>&1 || { echo FAIL_SUMMARY > logs/dai14.fail; exit 1; }
echo OK > logs/dai14.finished
