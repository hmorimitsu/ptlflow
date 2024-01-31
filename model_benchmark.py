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
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from tqdm import tqdm

import ptlflow
from ptlflow import get_model_reference
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.timer import Timer
from ptlflow.utils.utils import (
    config_logging,
    count_parameters,
    get_list_of_available_models_list,
    make_divisible,
)

NUM_COMMON_COLUMNS = 6
TABLE_KEYS_LEGENDS = {
    "model": "Model",
    "params": "Params",
    "flops": "FLOPs",
    "input_h": "InputH",
    "input_w": "InputW",
    "input_px": "InputPx",
    "time": "Time(ms)",
    "memory": "Memory(GB)",
}
TABLE_KEYS = list(TABLE_KEYS_LEGENDS.keys())
TABLE_LEGENDS = [TABLE_KEYS_LEGENDS[x] for x in TABLE_KEYS]

config_logging()

from torch.profiler import profile, record_function, ProfilerActivity

try:
    import pynvml
except ImportError:
    pynvml = None
    logging.warning("pynvml is not installed, GPU memory usage will not be measured.")


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        default="all",
        choices=["all", "select"] + get_list_of_available_models_list(),
        help=("Path to a csv file with the speed results."),
    )
    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with model=select. The select mode can be used to run the validation on multiple models "
            "at once. Put a list of model names here separated by spaces."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with model=all. A list of model names that will not be validated."
        ),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=("Path to a csv file with the speed results."),
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help=("Number of times to repeat the test with the same model."),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help=("Number of forwards in one repetition to estimate average time"),
    )
    parser.add_argument(
        "--sleep_interval",
        type=float,
        default=0.0,
        help=("Number of seconds to sleep between each repetition"),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs="+",
        default=(500, 1000),
        help=(
            "Resolution of the input to forward."
            "Must provide an even number of values."
            "Each pair of values will be interpreted as one input size."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/benchmark")),
        help=("Path to a directory where the outputs will be saved."),
    )
    parser.add_argument(
        "--final_speed_mode",
        type=str,
        choices=("avg", "median", "perc1", "perc5", "perc10"),
        default="median",
        help=(
            "How to obtain the final speed results."
            "percX represents reporting the value at the X-th percentile."
        ),
    )
    parser.add_argument(
        "--final_memory_mode",
        type=str,
        choices=("avg", "median", "perc1", "perc5", "perc10", "first"),
        default="first",
        help=(
            "How to obtain the final memory results."
            "percX represents reporting the value at the X-th percentile."
        ),
    )
    parser.add_argument(
        "--plot_axes",
        type=str,
        nargs=2,
        choices=TABLE_KEYS[1:],
        default=None,
        help=("Name of two measured parameters to create a scatter plot."),
    )
    parser.add_argument(
        "--plot_log_x",
        action="store_true",
        help="If set, the X-axis of the plot will be in log-scale.",
    )
    parser.add_argument(
        "--plot_log_y",
        action="store_true",
        help="If set, the Y-axis of the plot will be in log-scale.",
    )
    parser.add_argument(
        "--datatypes",
        type=str,
        nargs="+",
        choices=("fp16", "fp32"),
        default=("fp32",),
        help="Datatypes to use during benchmark.",
    )

    return parser


def benchmark(args: argparse.Namespace, device_handle) -> pd.DataFrame:
    """Run the benchmark on all models.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for configuring the benchmark.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the benchmark results.
    """
    df_dict = {
        TABLE_LEGENDS[0]: pd.Series([], dtype="str"),
        TABLE_LEGENDS[1]: pd.Series([], dtype="float"),
        TABLE_LEGENDS[2]: pd.Series([], dtype="float"),
        TABLE_LEGENDS[3]: pd.Series([], dtype="int"),
        TABLE_LEGENDS[4]: pd.Series([], dtype="int"),
        TABLE_LEGENDS[5]: pd.Series([], dtype="int"),
    }
    for dtype_str in args.datatypes:
        df_dict[f"{TABLE_LEGENDS[6]}-{dtype_str}"] = pd.Series([], dtype="float")
        df_dict[f"{TABLE_LEGENDS[7]}-{dtype_str}"] = pd.Series([], dtype="float")

    df = pd.DataFrame(df_dict)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_args = args
    if args.model == "all":
        model_names = ptlflow.models_dict.keys()
        model_args = None
    elif args.model == "select":
        if args.selection is None:
            raise ValueError(
                "When select is chosen, model names must be provided to --selection."
            )
        model_names = args.selection
        model_args = None
    else:
        model_names = [args.model]

    assert (len(args.input_size) % 2) == 0

    if pynvml is not None and device_handle is not None:
        device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        device_initial_used = device_info.used

    for isize in range(0, len(args.input_size), 2):
        input_size = args.input_size[isize : isize + 2]

        exclude = args.exclude
        if exclude is None:
            exclude = []

        for mname in tqdm(model_names):
            if mname in exclude:
                continue

            new_df_dict = {}
            for idtype, dtype_str in enumerate(args.datatypes):
                try:
                    all_times = []
                    all_memories = []
                    first_memory_used = 0
                    for irep in range(args.num_trials + 1):
                        torch.cuda.empty_cache()
                        time.sleep(args.sleep_interval)
                        if pynvml is not None and device_handle is not None:
                            device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
                            device_start_rep_used = device_info.used
                        model = ptlflow.get_model(mname, args=model_args)
                        model = model.eval()
                        if torch.cuda.is_available():
                            model = model.cuda()
                            if dtype_str == "fp16":
                                model = model.half()
                        model_params = count_parameters(model)
                        repetition_times = estimate_inference_time(
                            args, model, input_size, dtype_str
                        )
                        if irep > 0:
                            all_times.extend(repetition_times)

                        if device_handle is not None:
                            device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
                            model_memory_used = device_info.used - device_start_rep_used
                            if irep > 0:
                                all_memories.extend(
                                    [model_memory_used] * args.num_samples
                                )
                            else:
                                first_memory_used = (
                                    device_info.used - device_initial_used
                                )
                        model = model.cpu()
                        model = None

                    model = ptlflow.get_model(mname, args=model_args)
                    model = model.eval()

                    inputs = {
                        "images": torch.rand(
                            1,
                            2,
                            3,
                            make_divisible(input_size[0], model.output_stride),
                            make_divisible(input_size[1], model.output_stride),
                        )
                    }

                    if torch.cuda.is_available():
                        model = model.cuda()
                        inputs["images"] = inputs["images"].cuda()
                        if dtype_str == "fp16":
                            model = model.half()
                            inputs["images"] = inputs["images"].half()

                    flops = count_flops(model, inputs)

                    all_times.sort()
                    final_times = {
                        "avg": np.array(all_times).mean(),
                        "median": all_times[len(all_times) // 2],
                        "perc1": all_times[len(all_times) // 100],
                        "perc5": all_times[len(all_times) // 20],
                        "perc10": all_times[len(all_times) // 10],
                    }

                    if len(all_memories) == 0:
                        all_memories = [0]
                    all_memories.sort()
                    final_memories = {
                        "avg": np.array(all_memories).mean(),
                        "median": all_memories[len(all_memories) // 2],
                        "perc1": all_memories[len(all_memories) // 100],
                        "perc5": all_memories[len(all_memories) // 20],
                        "perc10": all_memories[len(all_memories) // 10],
                        "first": first_memory_used,
                    }

                    if len(new_df_dict) == 0:
                        values = [
                            mname,
                            float(model_params) / 1e6,
                            flops / 1e9,
                            input_size[0],
                            input_size[1],
                            input_size[0] * input_size[1],
                        ]
                        new_df_dict.update(
                            {
                                c: [v]
                                for c, v in zip(df.columns[:NUM_COMMON_COLUMNS], values)
                            }
                        )

                    values = [
                        final_times[args.final_speed_mode] * 1000,
                        final_memories[args.final_memory_mode] / 1024**3,
                    ]
                    new_df_dict.update(
                        {
                            c: [v]
                            for c, v in zip(
                                df.columns[
                                    NUM_COMMON_COLUMNS
                                    + 2 * idtype : NUM_COMMON_COLUMNS
                                    + 2 * (idtype + 1)
                                ],
                                values,
                            )
                        }
                    )
                except Exception as e:  # noqa: B902
                    logging.warning(
                        "Skipping model %s with datatype %s due to exception %s",
                        mname,
                        dtype_str,
                        e,
                    )

            if len(new_df_dict) > 0:
                new_df = pd.DataFrame(new_df_dict)
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.round(3)
                df.to_csv(
                    output_path / f"model_benchmark-{args.model}.csv", index=False
                )
                save_plot(
                    output_path,
                    args.model,
                    df,
                    args.plot_axes,
                    args.plot_log_x,
                    args.plot_log_y,
                    args.datatypes[0],
                )
    return df


@torch.no_grad()
def count_flops(model, inputs):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            model(inputs)
    key_averages = prof.key_averages()
    flops = 0
    for k in key_averages:
        flops += k.flops
    return flops


@torch.no_grad()
def estimate_inference_time(
    args: argparse.Namespace,
    model: BaseModel,
    input_size: Tuple[int, int],
    dtype_str: str,
) -> float:
    """Compute the average forward time for one model.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for configuring the benchmark.
    model : BaseModel
        The model to perform the estimation.

    Returns
    -------
    float
        The average time of the runs.
    """
    timer = Timer("inference")
    time_vals = []
    for i in range(args.num_samples + 1):
        inputs = {
            "images": torch.rand(
                1,
                2,
                3,
                make_divisible(input_size[0], model.output_stride),
                make_divisible(input_size[1], model.output_stride),
            )
        }
        if torch.cuda.is_available():
            inputs["images"] = inputs["images"].cuda()
            if dtype_str == "fp16":
                inputs["images"] = inputs["images"].half()
        if i > 0:
            # Skip first time, it is slow due to memory allocation
            timer.reset()
            timer.tic()
        model(inputs)
        if i > 0:
            timer.toc()
            time_vals.append(timer.total())
    return time_vals


def save_plot(
    output_dir: Union[str, Path],
    model_name: str,
    df: pd.DataFrame,
    plot_axes: Optional[Tuple[str, str]],
    log_x: bool,
    log_y: bool,
    datatype: str,
) -> None:
    """Create a plot of the results and save to disk.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Path to the directory where the plot will be saved.
    model_name : str
        Name of the model. Used just to name the resulting file.
    df : pd.DataFrame
        A DataFrame with the benchmark results.
    plot_axes : Optional[Tuple[str, str]]
        Name of two parameters to create the scatter plot.
    log_x : bool
        If set, the X-axis is plot in log scale.
    log_y : bool
        If set, the Y-axis is plot in log scale.
    datatype : str
        Name of the datatype.
    """
    if plot_axes is not None:
        assert len(plot_axes) == 2
        xkey, ykey = plot_axes
        assert xkey in TABLE_KEYS
        assert ykey in TABLE_KEYS

        df_tmp = df.copy()
        df_tmp = df_tmp.dropna()

        xlegend = TABLE_KEYS_LEGENDS[xkey]
        if xkey in ("memory", "time"):
            xlegend += f"-{datatype}"
        ylegend = TABLE_KEYS_LEGENDS[ykey]
        if ykey in ("memory", "time"):
            ylegend += f"-{datatype}"

        if log_x:
            log10_col = f"{xlegend}(Log10)"
            df_tmp[log10_col] = np.log10(df[xlegend])
        if log_y:
            log10_col = f"{ylegend}(Log10)"
            df_tmp[log10_col] = np.log10(df[ylegend])

        fig = px.scatter(
            df_tmp,
            x=xlegend,
            y=ylegend,
            color=TABLE_LEGENDS[0],
            symbol=TABLE_LEGENDS[0],
            log_x=log_x,
            log_y=log_y,
            title=f"{xlegend} x {ylegend}",
        )
        fig.update_traces(
            marker={"size": 20, "line": {"width": 2, "color": "DarkSlateGrey"}},
            selector={"mode": "markers"},
        )
        fig.update_layout(title_font_size=30)
        out_name = f"benchmark_plot-{model_name}-{plot_axes[0]}-{plot_axes[1]}.html"
        out_path = Path(output_dir) / out_name
        fig.write_html(out_path)
        logging.info(
            "Saved plot between %s and %s at: %s", plot_axes[0], plot_axes[1], out_path
        )


if __name__ == "__main__":
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] not in ["-h", "--help", "all", "select"]:
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    args = parser.parse_args()

    device_handle = None
    if pynvml is not None:
        try:
            device_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
        except (KeyError, ValueError):
            device_id = 0

        pynvml.nvmlInit()
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    if args.csv_path is None:
        df = benchmark(args, device_handle)
    else:
        df = pd.read_csv(args.csv_path)
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        save_plot(
            args.output_path,
            args.model,
            df,
            args.plot_axes,
            args.plot_log_x,
            args.plot_log_y,
            args.datatypes[0],
        )
    print(f"Results saved to {str(args.output_path)}.")
    print(df)
