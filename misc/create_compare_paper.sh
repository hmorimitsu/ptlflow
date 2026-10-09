#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/../compare_paper_results.py" --paper_results_path "$SCRIPT_DIR/../docs/source/results/paper_results_things.csv" --validate_results_path "$SCRIPT_DIR/../docs/source/results/metrics_all_things.csv" --add_delta
