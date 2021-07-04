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

import cv2 as cv
import numpy as np

import infer
import ptlflow

TEST_MODEL = 'raft_small'


def test_infer(tmp_path: Path) -> None:
    _create_images(tmp_path)

    parser = infer._init_parser()

    model_ref = ptlflow.get_model_reference(TEST_MODEL)
    parser = model_ref.add_model_specific_args(parser)

    input_path = [str(tmp_path / 'img1.png'), str(tmp_path / 'img2.png')]
    args = parser.parse_args([TEST_MODEL, '--input_path', input_path[0], input_path[1]])

    args.write_outputs = True
    args.output_path = tmp_path

    args.flow_format = 'flo'
    model = ptlflow.get_model(TEST_MODEL, None, args)
    infer.infer(args, model)
    assert (tmp_path / 'flows/img1.flo').exists()
    assert (tmp_path / 'flows_viz/img1.png').exists()

    args.flow_format = 'png'
    model = ptlflow.get_model(TEST_MODEL, None, args)
    infer.infer(args, model)
    assert (tmp_path / 'flows/img1.png').exists()
    assert (tmp_path / 'flows_viz/img1.png').exists()

    shutil.rmtree(tmp_path)


def _create_images(tmp_path: Path) -> None:
    for i in range(2):
        img = np.random.randint(0, 255, (400, 400, 3), np.uint8)
        cv.imwrite(str(tmp_path / f'img{i+1}.png'), img)
