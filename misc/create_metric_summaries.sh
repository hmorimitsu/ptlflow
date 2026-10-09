#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/../summary_metrics.py" --metrics_path "$SCRIPT_DIR/../docs/source/results/metrics_all.csv" --chosen_metrics epe

python "$SCRIPT_DIR/../summary_metrics.py" --metrics_path "$SCRIPT_DIR/../docs/source/results/metrics_all.csv" --chosen_metrics epe flall
