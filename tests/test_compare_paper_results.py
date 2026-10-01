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

from pathlib import Path
import shutil

import compare_paper_results


def test_compare_paper_results(tmp_path: Path) -> None:
    parser = compare_paper_results._init_parser()
    args = parser.parse_args([])
    args.output_dir = str(tmp_path)
    args.add_delta = True

    compare_paper_results.save_results(args)

    output_path = tmp_path / "paper_ptlflow_metrics.csv"
    assert output_path.exists()
    content = output_path.read_text()
    # the compare columns must have been created
    assert "ptlflow" in content
    assert "paper" in content
    # delta columns must have been created
    assert "delta" in content

    shutil.rmtree(tmp_path)
