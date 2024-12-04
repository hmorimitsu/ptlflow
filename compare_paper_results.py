"""Create a side-by-side table comparing the results of PTLFlow with those reported in the original papers.

This script only evaluates results of models that provide the "things" pretrained models.

Tha parsing of this script is tightly connected to how the results are output by validate.py.
"""

# =============================================================================
# Copyright 2024 Henrique Morimitsu
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
import math
from pathlib import Path

from loguru import logger
import pandas as pd

PAPER_VAL_COLS = {
    "model": ("Model", "model"),
    "sclean": ("S.clean", "sintel-clean-val/epe"),
    "sfinal": ("S.final", "sintel-final-val/epe"),
    "k15epe": ("K15-epe", "kitti-2015-val/epe"),
    "k15fl": ("K15-fl", "kitti-2015-val/flall"),
}


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper_results_path",
        type=str,
        default=str(Path("docs/source/results/paper_results_things.csv")),
        help=("Path to the csv file containing the results from the papers."),
    )
    parser.add_argument(
        "--validate_results_path",
        type=str,
        default=str(Path("docs/source/results/metrics_all_things.csv")),
        help=(
            "Path to the csv file containing the results obtained by the validate script."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("outputs/metrics")),
        help=("Path to the directory where the outputs will be saved."),
    )
    parser.add_argument(
        "--add_delta",
        action="store_true",
        help=(
            "If set, adds one more column showing the difference between paper and validation results."
        ),
    )

    return parser


def save_results(args: argparse.Namespace) -> None:
    paper_df = pd.read_csv(args.paper_results_path)
    val_df = pd.read_csv(args.validate_results_path)
    paper_df["model"] = paper_df[PAPER_VAL_COLS["model"][0]]
    val_df["model"] = val_df[PAPER_VAL_COLS["model"][1]]
    df = pd.merge(val_df, paper_df, "left", "model")

    compare_cols = ["ptlflow", "paper"]
    if args.add_delta:
        compare_cols.append("delta")

    out_dict = {"model": ["", ""]}
    for name in list(PAPER_VAL_COLS.keys())[1:]:
        for ic, col in enumerate(compare_cols):
            out_dict[f"{name}-{col}"] = [name if ic == 0 else "", col]

    for _, row in df.iterrows():
        out_dict["model"].append(row["model"])
        for key in list(PAPER_VAL_COLS.keys())[1:]:
            paper_col_name = PAPER_VAL_COLS[key][0]
            paper_res = float(row[paper_col_name])
            val_col_name = PAPER_VAL_COLS[key][1]
            val_res = float(row[val_col_name])
            res_list = [val_res, paper_res]

            if args.add_delta:
                delta = val_res - paper_res
                res_list.append(delta)

            for name, res in zip(compare_cols, res_list):
                out_dict[f"{key}-{name}"].append(
                    "" if (math.isinf(res) or math.isnan(res)) else f"{res:.3f}"
                )

    out_df = pd.DataFrame(out_dict)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / "paper_ptlflow_metrics.csv"
    out_df.to_csv(output_path, index=False, header=False)
    logger.info("Results saved to: {}", output_path)


if __name__ == "__main__":
    parser = _init_parser()
    args = parser.parse_args()
    save_results(args)
