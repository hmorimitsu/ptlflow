"""

Create a file with a list of samples names from the AutoFlow [1] dataset to be used as validation samples.

[1] Sun, Deqing et al. “AutoFlow: Learning a Better Training Set for Optical Flow.” CVPR. 2021.

"""

# =============================================================================
# Copyright 2022 Henrique Morimitsu
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


from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import random

random.seed(42)

THIS_DIR = Path(os.path.abspath(os.path.dirname(__file__)))


def _init_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--autoflow_root", type=str, required=True)
    parser.add_argument(
        "--output_file", type=str, default=str(THIS_DIR / "AutoFlow_val.txt")
    )
    parser.add_argument("--val_percentage", type=float, default=0.05)
    return parser


def main(args: Namespace) -> None:
    """Run the split process.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for configuring the splitting.
    """
    parts_dirs = [f"static_40k_png_{i+1}_of_4" for i in range(4)]
    sample_dirs = []
    for pdir in parts_dirs:
        sample_dirs.extend(
            sorted(
                [
                    f.stem
                    for f in (Path(args.autoflow_root) / pdir).glob("*")
                    if f.is_dir()
                ]
            )
        )
    sample_dirs.sort()
    assert (
        len(sample_dirs) == 40000
    ), f"ERROR: AutoFlow dataset should have 40k samples, but found {len(sample_dirs)}."
    samples_per_table = {}
    for sdir in sample_dirs:
        table_idx = sdir.split("_")[1]
        if table_idx not in samples_per_table:
            samples_per_table[table_idx] = []
        samples_per_table[table_idx].append(sdir)
    assert (
        len(samples_per_table) == 300
    ), f"ERROR: AutoFlow dataset should have 300 tables, but found {len(samples_per_table)}."

    val_samples = []
    carryover_samples = 0.0
    for dir_list in samples_per_table.values():
        num_samples = len(dir_list)
        num_val_samples_float = args.val_percentage * num_samples + carryover_samples
        num_val_samples = int(num_val_samples_float)

        random.shuffle(dir_list)
        val_samples.extend(dir_list[:num_val_samples])

        carryover_samples = num_val_samples_float - num_val_samples

    val_samples.sort(key=lambda x: 1000 * int(x.split("_")[1]) + int(x.split("_")[-1]))
    with open(args.output_file, "w") as f:
        f.write("\n".join(val_samples))

    print(f"Saved {len(val_samples)} sample names to {args.output_file}")


if __name__ == "__main__":
    parser: ArgumentParser = _init_parser()
    args: Namespace = parser.parse_args()
    main(args)
