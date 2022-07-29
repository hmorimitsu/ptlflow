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
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from tqdm import tqdm

import ptlflow
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.timer import Timer
from ptlflow.utils.utils import config_logging, count_parameters, get_list_of_available_models_list, make_divisible

TABLE_COLS = ['Model', 'Params', 'Time(ms)']

config_logging()


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='all', choices=['all']+get_list_of_available_models_list(),
        help=('Path to a csv file with the speed results.'))
    parser.add_argument(
        '--csv_path', type=str, default=None,
        help=('Path to a csv file with the speed results.'))
    parser.add_argument(
        '--num_samples', type=int, default=20,
        help=('Number of forwards to estimate average time'))
    parser.add_argument(
        '--input_size', type=int, nargs=2, default=(500, 1000),
        help=('Resolution of the input to forward.'))
    parser.add_argument(
        '--output_path', type=str, default=str(Path('outputs/speed')),
        help=('Path to a directory where the outputs will be saved.'))

    return parser


def benchmark(
    args: argparse.Namespace
) -> pd.DataFrame:
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
    df = pd.DataFrame(
        {TABLE_COLS[0]: pd.Series([], dtype='str'),
         TABLE_COLS[1]: pd.Series([], dtype='int'),
         TABLE_COLS[2]: pd.Series([], dtype='float')})

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.model == 'all':
        model_names = ptlflow.models_dict.keys()
    else:
        model_names = [args.model]
    for mname in tqdm(model_names):
        try:
            model = ptlflow.get_model(mname)
            model = model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            model_params = count_parameters(model)
            infer_timer = estimate_inference_time(args, model)
            values = [mname, model_params, infer_timer*1000]
            new_df = pd.DataFrame({c: [v] for c, v in zip(df.columns, values)})
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.round(3)
            df.to_csv(output_path / f'speed_benchmark-{args.model}.csv', index=False)
            save_plot(output_path, args.model, df)
        except Exception as e:  # noqa: B902
            logging.warning('Skipping model %s due to exception %s', mname, e)
    return df


@torch.no_grad()
def estimate_inference_time(
    args: argparse.Namespace,
    model: BaseModel
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
    timer = Timer('inference')
    for i in range(args.num_samples+1):
        inputs = {
            'images': torch.rand(
                1, 2, 3, make_divisible(args.input_size[0], model.output_stride),
                make_divisible(args.input_size[1], model.output_stride))}
        if torch.cuda.is_available():
            inputs['images'] = inputs['images'].cuda()
        if i > 0:
            # Skip first time, it is slow due to memory allocation
            timer.tic()
        model(inputs)
        if i > 0:
            timer.toc()
    return timer.mean()


def save_plot(
    output_dir: Union[str, Path],
    model_name: str,
    df: pd.DataFrame
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
    """
    df = df.dropna()

    output_dir = Path(output_dir)

    log10_col = TABLE_COLS[1]+'(Log10)'
    df_tmp = df.copy()
    df_tmp[log10_col] = np.log10(df[TABLE_COLS[1]])

    fig = px.scatter(
        df, x=TABLE_COLS[1], y=TABLE_COLS[2], color=TABLE_COLS[0], symbol=TABLE_COLS[0], log_x=True, log_y=True,
        title='Parameters x Forward time')
    fig.update_traces(
        marker={
            'size': 20,
            'line': {'width': 2, 'color': 'DarkSlateGrey'}},
        selector={'mode': 'markers'})
    fig.update_layout(
        title_font_size=30
    )
    fig.write_html(output_dir / f'speed_plot-{model_name}.html')


if __name__ == '__main__':
    parser = _init_parser()
    args = parser.parse_args()

    if args.csv_path is None:
        df = benchmark(args)
    else:
        df = pd.read_csv(args.csv_path)
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        save_plot(args.output_path, args.model, df)
    print(f'Results saved to {str(args.output_path)}.')
