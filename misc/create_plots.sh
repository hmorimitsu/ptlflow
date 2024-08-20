#!/bin/bash
python ../plot_results.py \
    --metrics_csv_path ../docs/source/results/metrics_all.csv \
    --benchmark_csv_path ../docs/source/results/model_benchmark-all.csv \
    --checkpoint_names things \
    --plot_axes "memory(gb)-fp32" "sintel-final-occ-val/epe"

python ../plot_results.py \
    --metrics_csv_path ../docs/source/results/metrics_all.csv \
    --benchmark_csv_path ../docs/source/results/model_benchmark-all.csv \
    --checkpoint_names things \
    --plot_axes "memory(gb)-fp32" "sintel-final-occ-val/outlier"

python ../plot_results.py \
    --metrics_csv_path ../docs/source/results/metrics_all.csv \
    --benchmark_csv_path ../docs/source/results/model_benchmark-all.csv \
    --checkpoint_names things \
    --plot_axes "time(ms)-fp32" "sintel-final-occ-val/epe"

python ../plot_results.py \
    --metrics_csv_path ../docs/source/results/metrics_all.csv \
    --benchmark_csv_path ../docs/source/results/model_benchmark-all.csv \
    --checkpoint_names things \
    --plot_axes "time(ms)-fp32" "sintel-final-occ-val/outlier"

python ../plot_results.py \
    --metrics_csv_path ../docs/source/results/metrics_all.csv \
    --benchmark_csv_path ../docs/source/results/model_benchmark-all.csv \
    --checkpoint_names things \
    --plot_axes "params" "flops" \
    --log_x \
    --log_y

python ../plot_results.py \
    --metrics_csv_path ../docs/source/results/metrics_all.csv \
    --benchmark_csv_path ../docs/source/results/model_benchmark-all.csv \
    --checkpoint_names things \
    --plot_axes "sintel-final-occ-val/epe" "kitti-2015-val/outlier"