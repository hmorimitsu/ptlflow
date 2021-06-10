"""Create dummy datasets.

The datasets will have the same directory and files structure as the original ones, but only contain a single sample
for each directory. The samples are just randomly generated noise.

The main purpose of this script is to be used with tests. But it can also be useful to visualize the structure of a dataset
without having to download it.
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

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2 as cv
import numpy as np

from ptlflow.utils import flow_utils
from ptlflow.utils.utils import config_logging

config_logging()


def write_flying_chairs(
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (384, 512)
) -> None:
    """Generate a dummy version of the Flying Chairs dataset.

    The original dataset is available at:

    https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.htm

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (384, 512)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size+(3,))
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)

    root_dir = Path(root_dir) / 'FlyingChairs_release'
    data_dir = root_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    cv.imwrite(str(data_dir / '00001_img1.ppm'), img)
    cv.imwrite(str(data_dir / '00001_img2.ppm'), img)
    flow_utils.flow_write(data_dir / '00001_flow.flo', flow)

    logging.info('Created dataset on %s.', str(root_dir))


def write_flying_chairs2(
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (384, 512)
) -> None:
    """Generate a dummy version of the Flying Chairs 2 dataset.

    The original dataset is available at:

    https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (384, 512)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size+(3,), np.uint8)
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)
    mask = np.random.randint(0, 2, img_size, np.uint8) * 255

    root_dir = Path(root_dir) / 'FlyingChairs2'
    for split in ['train', 'val']:
        img_dir = root_dir / split
        img_dir.mkdir(parents=True, exist_ok=True)

        cv.imwrite(str(img_dir / '0000001-img_0.png'), img)
        cv.imwrite(str(img_dir / '0000001-img_1.png'), img)
        cv.imwrite(str(img_dir / '0000001-oids_0.png'), img)
        cv.imwrite(str(img_dir / '0000001-oids_1.png'), img)
        cv.imwrite(str(img_dir / '0000001-occ_01.png'), mask)
        cv.imwrite(str(img_dir / '0000001-occ_10.png'), mask)
        cv.imwrite(str(img_dir / '0000001-mb_01.png'), mask)
        cv.imwrite(str(img_dir / '0000001-mb_10.png'), mask)
        flow_utils.flow_write(img_dir / '0000001-flow_01.flo', flow)
        flow_utils.flow_write(img_dir / '0000001-flow_10.flo', flow)

    logging.info('Created dataset on %s.', str(root_dir))


def write_hd1k(
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (1080, 2560)
) -> None:
    """Generate a dummy version of the HD1K dataset.

    The original dataset is available at:

    http://hci-benchmark.org/

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (1080, 2560)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size)
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)

    root_dir = Path(root_dir) / 'HD1K'
    for dir in ['hd1k_challenge', 'hd1k_input']:
        img_dir = root_dir / dir / 'image_2'
        img_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(img_dir / '000000_0010.png'), img)
        cv.imwrite(str(img_dir / '000000_0011.png'), img)

    flow_path = root_dir / 'hd1k_flow_gt' / 'flow_occ' / '000000_0010.png'
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    flow_utils.flow_write(flow_path, flow)

    unc_path = root_dir / 'hd1k_flow_uncertainty' / 'flow_unc' / '000000_0010.png'
    unc_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(unc_path), img)

    logging.info('Created dataset on %s.', str(root_dir))


def write_kitti(
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (375, 1242)
) -> None:
    """Generate a dummy version of the KITTI 2012 and 2015 datasets.

    The original datasets are available at:

    http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow

    http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (375, 1242)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size)
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)

    root_dir = Path(root_dir) / 'KITTI'
    for year in ['2012', '2015']:
        for split in ['training', 'testing']:
            if year == '2012':
                img_dir = 'colored_0'
            else:
                img_dir = 'image_2'

            img_dir_path = root_dir / year / split / img_dir
            img_dir_path.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(img_dir_path / '000000_10.png'), img)
            cv.imwrite(str(img_dir_path / '000000_11.png'), img)

            if split == 'training':
                flow_path = root_dir / year / split / 'flow_occ' / '000000_10.png'
                flow_path.parent.mkdir(parents=True, exist_ok=True)
                flow_utils.flow_write(flow_path, flow)

    logging.info('Created dataset on %s.', str(root_dir))


def write_sintel(
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (436, 1024)
) -> None:
    """Generate a dummy version of the MPI Sintel dataset.

    The original dataset is available at:

    http://sintel.is.tue.mpg.de/

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (436, 1024)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size)
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)
    mask = np.random.randint(0, 2, img_size, np.uint8) * 255

    root_dir = Path(root_dir) / 'MPI-Sintel'
    for split in ['training', 'test']:
        for pass_name in ['clean', 'final']:
            img_dir_path = root_dir / split / pass_name / 'sequence_1'
            img_dir_path.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(img_dir_path / 'frame_0001.png'), img)
            cv.imwrite(str(img_dir_path / 'frame_0002.png'), img)

        if split == 'training':
            flow_path = root_dir / split / 'flow' / 'sequence_1' / 'frame_0001.flo'
            flow_path.parent.mkdir(parents=True, exist_ok=True)
            flow_utils.flow_write(flow_path, flow)

            occ_path = root_dir / split / 'occlusions' / 'sequence_1' / 'frame_0001.png'
            occ_path.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(occ_path), mask)

    logging.info('Created dataset on %s.', str(root_dir))


def write_things(  # noqa: C901
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (540, 960)
) -> None:
    """Generate a dummy version of the Flying Things 3D dataset.

    The original dataset is available at:

    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (540, 960)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size)
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)
    mask = np.random.randint(0, 2, img_size, np.uint8) * 255

    root_dir = Path(root_dir) / 'FlyingThings3D'
    for cat in ['frames_cleanpass', 'frames_finalpass', 'optical_flow', 'occlusions', 'motion_boundaries']:
        for split in ['TEST', 'TRAIN']:
            for letter in ['A', 'B', 'C']:
                for side_dir, side_name in [('left', 'L'), ('right', 'R')]:
                    if cat == 'optical_flow':
                        for direc_dir, direc_name in [('into_future', 'IntoFuture'), ('into_past', 'IntoPast')]:
                            flow_dir = root_dir / cat / split / letter / '0000' / direc_dir / side_dir
                            flow_dir.mkdir(parents=True, exist_ok=True)
                            for num in range(2):
                                flow_utils.flow_write(
                                    flow_dir / f'OpticalFlow{direc_name}_{num:04d}_{side_name}.pfm', flow)
                    elif cat.startswith('frames'):
                        img_dir_path = root_dir / cat / split / letter / '0000' / side_dir
                        img_dir_path.mkdir(parents=True, exist_ok=True)
                        cv.imwrite(str(img_dir_path / '0000.png'), img)
                        cv.imwrite(str(img_dir_path / '0001.png'), img)
                    else:
                        for direc_dir, direc_name in [('into_future', 'IntoFuture'), ('into_past', 'IntoPast')]:
                            mask_dir = root_dir / cat / split / letter / '0000' / direc_dir / side_dir
                            mask_dir.mkdir(parents=True, exist_ok=True)
                            for num in range(2):
                                cv.imwrite(str(mask_dir / f'{cat}{direc_name}_{num:04d}_{side_name}.png'), mask)

    logging.info('Created dataset on %s.', str(root_dir))


def write_things_subset(
    root_dir: Union[str, Path],
    img_size: Tuple[int, int] = (540, 960)
) -> None:
    """Generate a dummy version of the Flying Things 3D subset dataset.

    The original dataset is available at:

    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the directory where the dummy dataset will be created.
    img_size : Tuple[int, int], default (540, 960)
        The size of the images inside of this dataset.
    """
    img = np.random.randint(0, 256, img_size)
    flow = np.random.rand(img_size[0], img_size[1], 2).astype(np.float32)
    mask = np.random.randint(0, 2, img_size, np.uint8) * 255

    root_dir = Path(root_dir) / 'FlyingThings3D_subset'
    for split in ['train', 'val']:
        for cat in ['image_clean', 'flow', 'flow_occlusions', 'motion_boundaries']:
            for side in ['left', 'right']:
                if cat == 'image_clean':
                    img_dir_path = root_dir / split / cat / side
                    img_dir_path.mkdir(parents=True, exist_ok=True)
                    cv.imwrite(str(img_dir_path / '0000000.png'), img)
                    cv.imwrite(str(img_dir_path / '0000001.png'), img)
                else:
                    for direc in ['into_future', 'into_past']:
                        if direc == 'into_future':
                            num = '0000000'
                        else:
                            num = '0000001'

                        if cat == 'flow':
                            flow_path = root_dir / split / cat / side / direc / f'{num}.flo'
                            flow_path.parent.mkdir(parents=True, exist_ok=True)
                            flow_utils.flow_write(flow_path, flow)
                        else:
                            img_path = root_dir / split / cat / side / direc / f'{num}.png'
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            cv.imwrite(str(img_path), mask)

    logging.info('Created dataset on %s.', str(root_dir))
