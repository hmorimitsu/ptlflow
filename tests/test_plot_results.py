# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from argparse import Namespace
from pathlib import Path
import shutil

import pandas as pd

import plot_results


def _write_metrics_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "model": ["raft", "pwcnet"],
            "checkpoint": ["things", "things"],
            "sintel-clean-val/epe": [1.5, 2.5],
            "kitti-2015-val/epe": [3.0, 4.0],
        }
    ).to_csv(path)


def _write_benchmark_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "Model": ["raft", "pwcnet"],
            "Params": [1.0, 2.0],
            "Time(ms)-fp32": [100.0, 200.0],
        }
    ).to_csv(path)


def _get_args(**kwargs) -> Namespace:
    args = Namespace(
        models=None,
        exclude_models=None,
        metrics_csv_path=None,
        benchmark_csv_path=None,
        plot_axes=None,
        checkpoint_names=["things"],
        log_x=False,
        log_y=False,
        output_path=None,
    )
    args.__dict__.update(kwargs)
    return args


def test_load_dataframe_metrics_only(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    _write_metrics_csv(metrics_csv)

    args = _get_args(
        metrics_csv_path=str(metrics_csv),
        plot_axes=["sintel-clean-val/epe", "kitti-2015-val/epe"],
    )
    df = plot_results.load_dataframe(args)
    assert df.shape[0] == 2
    assert list(df.columns) == ["model", "sintel-clean-val/epe", "kitti-2015-val/epe"]

    shutil.rmtree(tmp_path)


def test_load_dataframe_benchmark_only(tmp_path: Path) -> None:
    benchmark_csv = tmp_path / "benchmark.csv"
    _write_benchmark_csv(benchmark_csv)

    args = _get_args(
        benchmark_csv_path=str(benchmark_csv),
        plot_axes=["params", "time(ms)-fp32"],
    )
    df = plot_results.load_dataframe(args)
    assert df.shape[0] == 2
    assert list(df.columns) == ["model", "params", "time(ms)-fp32"]

    shutil.rmtree(tmp_path)


def test_load_dataframe_merged(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    benchmark_csv = tmp_path / "benchmark.csv"
    _write_metrics_csv(metrics_csv)
    _write_benchmark_csv(benchmark_csv)

    args = _get_args(
        metrics_csv_path=str(metrics_csv),
        benchmark_csv_path=str(benchmark_csv),
        plot_axes=["sintel-clean-val/epe", "params"],
    )
    df = plot_results.load_dataframe(args)
    assert df.shape[0] == 2
    assert list(df.columns) == ["model", "sintel-clean-val/epe", "params"]

    shutil.rmtree(tmp_path)


def test_load_dataframe_invalid_axis(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    _write_metrics_csv(metrics_csv)

    args = _get_args(
        metrics_csv_path=str(metrics_csv),
        plot_axes=["nonexistent-axis", "kitti-2015-val/epe"],
    )
    try:
        plot_results.load_dataframe(args)
        assert False, "An assertion error was expected"
    except AssertionError as e:
        assert "not a valid axis name" in str(e)

    shutil.rmtree(tmp_path)


def test_save_plot(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    _write_metrics_csv(metrics_csv)

    args = _get_args(
        metrics_csv_path=str(metrics_csv),
        plot_axes=["sintel-clean-val/epe", "kitti-2015-val/epe"],
    )
    df = plot_results.load_dataframe(args)

    plot_results.save_plot(tmp_path, df, False, False)
    plots = list(tmp_path.glob("plot-*.html"))
    assert len(plots) == 1

    shutil.rmtree(tmp_path)
