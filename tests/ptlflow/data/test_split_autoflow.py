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

# NOTE: split_autoflow seeds the random module at import time (not inside
# main()), so main() is only reproducible across processes. Re-seed the module
# manually to compare two runs in the same process.

import random
from argparse import Namespace
from pathlib import Path

from ptlflow.data.split_autoflow import main


def _create_dummy_autoflow(root_dir: Path, num_tables: int = 300) -> None:
    """Create 40000 sample dirs organized into 300 tables.

    AutoFlow requires exactly 40000 samples and 300 tables, so the tables are
    created with 134 samples each for the first 100 tables and 133 samples each
    for the remaining 200 tables (100*134 + 200*133 = 40000).
    """
    for table in range(num_tables):
        num_samples = 134 if table < 100 else 133
        for sample in range(num_samples):
            part = (table % 4) + 1
            sample_dir = (
                root_dir
                / f"static_40k_png_{part}_of_4"
                / f"table_{table}_batch_{sample}"
            )
            sample_dir.mkdir(parents=True)


def test_split_autoflow(tmp_path: Path) -> None:
    root_dir = tmp_path / "autoflow"
    _create_dummy_autoflow(root_dir)

    output_file = tmp_path / "AutoFlow_val.txt"
    args = Namespace(
        autoflow_root=str(root_dir),
        output_file=str(output_file),
        val_percentage=0.05,
    )
    # NOTE: split_autoflow seeds the random module at import time (not inside
    # main()), so the seed has to be reset manually here to reproduce a run.
    random.seed(42)
    main(args)

    val_names = output_file.read_text().strip().splitlines()

    # 5% of 40000 samples
    assert len(val_names) == 2000
    # no duplicated samples
    assert len(set(val_names)) == len(val_names)
    for name in val_names:
        tokens = name.split("_")
        table_idx = int(tokens[1])
        sample_idx = int(tokens[-1])
        assert 0 <= table_idx < 300
        assert 0 <= sample_idx < 134
        # the name must correspond to an existing dir in the dataset
        part = (table_idx % 4) + 1
        assert (root_dir / f"static_40k_png_{part}_of_4" / name).exists()

    # the same RNG state must generate the same validation split
    random.seed(42)
    main(args)
    assert output_file.read_text().strip().splitlines() == val_names
