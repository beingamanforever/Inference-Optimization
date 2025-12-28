#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:."

echo "Generating report and visualizations..."
echo ""

echo "[1/2] Creating plots..."
python3 scripts/plot_results.py --output-dir week1/benchmarks

echo "[2/2] Generating summary report..."
python3 scripts/generate_report.py --benchmark-dir week1/benchmarks

echo ""
echo "Done. Check week1/plots/ for visualizations."