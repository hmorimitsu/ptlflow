"""Create a reduced version of the validation metrics table.

The reduced version is created by selecting a subset of columns.

It can also create a plot between two chosen metrics.

Tha parsing of this script is tightly connected to how the results are output by validate.py.
"""

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
from pathlib import Path

import pandas as pd
import plotly.express as px


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--metrics_path', type=str, default=str(Path('docs/source/results/metrics_all.csv')),
        help=('Path to the csv file containing the validation metrics.'))
    parser.add_argument(
        '--chosen_metrics', type=str, nargs='+', default=('epe', 'outlier'),
        help=('Names of which metrics to keep in the summarized results. The chosen names must be at the end of the column '
              'name of the csv file. If exactly two metrics are chosen, a plot between the two will also be generated.'))
    parser.add_argument(
        '--output_dir', type=str, default=str(Path('outputs/metrics')),
        help=('Path to the directory where the outputs will be saved.'))
    parser.add_argument(
        '--sort_by', type=str, default='model',
        help=('Name of the column to use to sort the outputs table. The name must match exactly a column name from the'
              'metrics csv file.'))
    parser.add_argument(
        '--ignore', type=str, default='model',
        help=('Name of the column to use to sort the outputs table. The name must match exactly a column name from the'
              'metrics csv file.'))
    parser.add_argument(
        '--drop_checkpoints', type=str, nargs='*', default=None,
        help=('Name of checkpoints to not be included in the final outputs. The names must be substrings of the values in '
              'the file from --metrics_path.'))

    return parser


def load_summarized_table(
    args: argparse.Namespace
) -> pd.DataFrame:
    """Load the DataFrame and keep only columns according to the selected metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to control the loading.

    Returns
    -------
    pd.DataFrame
        The summarized DataFrame.
    """
    df = pd.read_csv(args.metrics_path)
    keep_cols = list(df.columns)[:2]
    for col in df.columns[2:]:
        for cmet in args.chosen_metrics:
            if col.endswith(cmet):
                keep_cols.append(col)
    summ_df = df[keep_cols]
    summ_df = summ_df.sort_values(args.sort_by)
    summ_df = summ_df.round(3)
    return summ_df


def save_plots(
    args: argparse.Namespace,
    df: pd.DataFrame
) -> None:
    """Generate and save the plot to disk.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to control the plot.
    df : pd.DataFrame
        A DataFrame with the validation metrics.
    """
    metric_pairs = {}
    for col in df.columns[2:]:
        for cmet in args.chosen_metrics:
            if col.endswith(cmet):
                dataset_name = '_'.join(col.split('-')[:2])
                if metric_pairs.get(dataset_name) is None:
                    metric_pairs[dataset_name] = {}
                metric_pairs[dataset_name][cmet] = col

    for dataset_name, col_pair_dict in metric_pairs.items():
        col1, col2 = col_pair_dict.values()
        fig = px.scatter(
            df, x=col1, y=col2, color=df.columns[0], symbol=df.columns[1],
            title=f'{dataset_name} - {args.chosen_metrics[0]} x {args.chosen_metrics[1]}')
        fig.update_traces(
            marker={
                'size': 20,
                'line': {'width': 2, 'color': 'DarkSlateGrey'}},
            selector={'mode': 'markers'})
        fig.update_layout(
            title_font_size=30
        )
        file_name = f'{dataset_name}_{args.chosen_metrics[0]}_{args.chosen_metrics[1]}'
        if args.drop_checkpoints is not None and len(args.drop_checkpoints) > 0:
            file_name += f'-drop_{"_".join(args.drop_checkpoints)}'
        fig.write_html(args.output_dir / (file_name+'.html'))


def summarize(
    args: argparse.Namespace
) -> None:
    """Summarize the results and save them to the disk.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments required to control the process.
    """
    args.output_dir = Path(args.output_dir)
    df = load_summarized_table(args)

    df = _shorten_columns_names(df, len(args.chosen_metrics) > 1)

    if args.drop_checkpoints is not None and len(args.drop_checkpoints) > 0:
        ignore_idx = [i for i in df.index if any(c in df.loc[i, 'checkpoint'] for c in args.drop_checkpoints)]
        df = df.drop(ignore_idx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'summarized_metrics-{"_".join(args.chosen_metrics)}'
    if args.drop_checkpoints is not None and len(args.drop_checkpoints) > 0:
        file_name += f'-drop_{"_".join(args.drop_checkpoints)}'

    df.to_csv(args.output_dir / (file_name+'.csv'), index=False)
    with open(args.output_dir / (file_name+'.md'), 'w') as f:
        df.to_markdown(f)

    if len(args.chosen_metrics) == 2:
        save_plots(args, df)


def _shorten_columns_names(
    df: pd.DataFrame,
    keep_metric_name: bool
) -> pd.DataFrame:
    change_dict = {}
    for col in df.columns:
        tokens = col.split('-')
        if len(tokens) > 1:
            metric_name = tokens[-1].split('/')[1]
            new_col_name = f'{tokens[0]}-{tokens[1]}'
            if keep_metric_name:
                new_col_name += f'-{metric_name}'
            change_dict[col] = new_col_name

    df = df.rename(columns=change_dict)
    return df


if __name__ == '__main__':
    parser = _init_parser()
    args = parser.parse_args()
    summarize(args)
    print(f'Results saved to {str(args.output_dir)}.')
