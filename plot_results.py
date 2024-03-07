"""Save the number of trainable parameter and inference speed of all available models."""

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

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px

from ptlflow.utils.utils import config_logging

config_logging()


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of model names to be loaded from the results. If not provided, all models will be loaded."
        ),
    )
    parser.add_argument(
        "--exclude_models",
        type=str,
        nargs="+",
        default=None,
        help=("Optional list of model names that will not be loaded from the results."),
    )
    parser.add_argument(
        "--metrics_csv_path",
        type=str,
        default=None,
        help=("Path to a csv file with the metrics results."),
    )
    parser.add_argument(
        "--benchmark_csv_path",
        type=str,
        default=None,
        help=("Path to a csv file with the benchmark results."),
    )
    parser.add_argument(
        "--plot_axes",
        type=str,
        nargs=2,
        default=None,
        required=True,
        help=(
            "Name of two measured parameters to create a scatter plot. It must correspond to a column name of the provide CSV files."
        ),
    )
    parser.add_argument(
        "--checkpoint_names",
        type=str,
        nargs="+",
        default=("things",),
        help=(
            "Name of checkpoints to be included in the final outputs. The names must be substrings of the values in "
            "the file from --metrics_csv_path."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/plots")),
        help=("Path to a directory where the outputs will be saved."),
    )
    parser.add_argument(
        "--log_x",
        action="store_true",
        help="If set, the X-axis of the plot will be in log-scale.",
    )
    parser.add_argument(
        "--log_y",
        action="store_true",
        help="If set, the Y-axis of the plot will be in log-scale.",
    )

    return parser


def save_plot(
    output_dir: Union[str, Path],
    df: pd.DataFrame,
    log_x: bool,
    log_y: bool,
) -> None:
    """Create a plot of the results and save to disk.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Path to the directory where the plot will be saved.
    df : pd.DataFrame
        A DataFrame with the benchmark results.
    log_x : bool
        If set, the X-axis is plot in log scale.
    log_y : bool
        If set, the Y-axis is plot in log scale.
    """
    df = df.dropna()

    xkey, ykey = list(df.columns)[-2:]

    symbol_sequence = [
        "circle",
        "diamond",
        "square",
        "x",
        "cross",
        "pentagon",
        "triangle-up",
    ]
    fig = px.scatter(
        df,
        x=xkey,
        y=ykey,
        color="checkpoint" if "checkpoint" in list(df.columns) else "model",
        symbol="model",
        log_x=log_x,
        log_y=log_y,
        title=f"{xkey} x {ykey}",
        symbol_sequence=symbol_sequence,
    )
    fig.update_traces(
        marker={"size": 20, "line": {"width": 2, "color": "DarkSlateGrey"}},
        selector={"mode": "markers"},
    )
    fig.update_layout(title_font_size=30)
    out_name = f"plot-{xkey}-{ykey}.html".replace("/", "_").replace("\\", "_")
    out_path = Path(output_dir) / out_name
    fig.write_html(out_path)
    logging.info("Saved plot between %s and %s at: %s", xkey, ykey, out_path)


def get_available_axes(benchmark_df, metrics_df):
    axes_names_to_source = {}
    if benchmark_df is not None:
        axes_names_to_source.update({c: "benchmark" for c in benchmark_df.columns})
    if metrics_df is not None:
        axes_names_to_source.update({c: "metrics" for c in metrics_df.columns})
    return axes_names_to_source


def load_dataframe(args):
    assert (args.benchmark_csv_path is not None) or (args.metrics_csv_path is not None)

    benchmark_df = None
    if args.benchmark_csv_path is not None:
        benchmark_df = pd.read_csv(args.benchmark_csv_path)
        benchmark_df.rename(
            columns={c: c.lower() for c in benchmark_df.columns}, inplace=True
        )
    metrics_df = None
    if args.benchmark_csv_path is not None:
        metrics_df = pd.read_csv(args.metrics_csv_path)
        metrics_df.rename(
            columns={c: c.lower() for c in metrics_df.columns}, inplace=True
        )

    axes_names_to_source = get_available_axes(benchmark_df, metrics_df)
    assert (
        args.plot_axes[0] in axes_names_to_source
    ), f"{args.plot_axes[0]} is not a valid axis name. The valid names are: {axes_names_to_source.keys()}"
    assert (
        args.plot_axes[1] in axes_names_to_source
    ), f"{args.plot_axes[1]} is not a valid axis name. The valid names are: {axes_names_to_source.keys()}"

    axes_sources = [axes_names_to_source[a] for a in args.plot_axes]
    unique_sources = list(set(axes_sources))

    base_columns = ["model"]
    if len(args.checkpoint_names) > 1 or args.checkpoint_names[0] == "all":
        assert (
            len(unique_sources) == 1 and unique_sources[0] == "metrics"
        ), "Using all or more than one argument for --checkpoint_names is only supported if both --plot_axes are from the metrics CSV"
        base_columns += ["checkpoint"]

    if len(unique_sources) == 1:
        if unique_sources[0] == "benchmark":
            df = benchmark_df
        if unique_sources[0] == "metrics":
            df = metrics_df
            df = df[df["checkpoint"] == args.checkpoint_names[0]]
    else:
        metrics_df = metrics_df[metrics_df["checkpoint"] == args.checkpoint_names[0]]
        df = pd.merge(metrics_df, benchmark_df, "inner", "model")

    df = df[base_columns + args.plot_axes]

    available_model_names = list(df["model"])

    if args.models is not None:
        invalid_model_names = [n for n in args.models if n not in available_model_names]
        if len(invalid_model_names) > 0:
            logging.warning(
                "The following requested models cannot be found in the csvs and will not included in the plot: %s",
                ", ".join(invalid_model_names),
            )
        model_names = [n for n in args.models if n in available_model_names]
    else:
        if args.exclude_models is None:
            model_names = available_model_names
        else:
            model_names = [
                n for n in available_model_names if n not in args.exclude_models
            ]

    df = df[df["model"].isin(model_names)]
    return df


if __name__ == "__main__":
    parser = _init_parser()
    args = parser.parse_args()

    df = load_dataframe(args)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    save_plot(
        args.output_path,
        df,
        args.log_x,
        args.log_y,
    )
