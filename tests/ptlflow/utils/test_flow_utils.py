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

from pathlib import Path
import shutil

import numpy as np

from ptlflow.utils import flow_utils

IMG_SIDE = 29
IMG_MIDDLE = IMG_SIDE // 2 + 1


def test_read_write_pfm(tmp_path: Path) -> None:
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=2).astype(np.float32)
    file_path = tmp_path / 'flow.pfm'
    flow_utils.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path)
    assert np.array_equal(flow, loaded_flow)

    shutil.rmtree(tmp_path)
