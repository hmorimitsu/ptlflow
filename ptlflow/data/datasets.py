"""Handle common datasets used in optical flow estimation."""

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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ptlflow.utils import flow_utils
from ptlflow.utils.utils import config_logging

config_logging()

THIS_DIR = Path(__file__).resolve().parent


class BaseFlowDataset(Dataset):
    """Manage optical flow dataset loading.

    This class can be used as the parent for any concrete dataset. It is structured to be able to read most types of inputs
    used in optical flow estimation.

    Classes inheriting from this one should implement the __init__() method and properly load the input paths from the chosen
    dataset. This should be done by populating the lists defined in the attributes below.

    Attributes
    ----------
    img_paths : list[list[str]]
        Paths of the images. Each element of the main list is a list of paths. Typically, the inner list will have two
        elements, corresponding to the paths of two consecutive images, which will be used to estimate the optical flow.
        More than two paths can also be added in case the model is able to use more images for estimating the flow.
    flow_paths : list[list[str]]
        Similar structure to img_paths. However, the inner list must have exactly one element less than img_paths.
        For example, if an entry of img_paths is composed of two paths, then an entry of flow_list should be a list with a
        single path, corresponding to the optical flow from the first image to the second.
    occ_paths : list[list[str]]
        Paths to the occlusion masks, follows the same structure as flow_paths. It can be left empty if not available.
    mb_paths : list[list[str]]
        Paths to the motion boundary masks, follows the same structure as flow_paths. It can be left empty if not available.
    flow_b_paths : list[list[str]]
        The same as flow_paths, but it corresponds to the backward flow. This list must be in the same order as flow_paths.
        For example, flow_b_paths[i] must be backward flow of flow_paths[i]. It can be left empty if backard flows are not
        available.
    occ_b_paths : list[list[str]]
        Backward occlusion mask paths, read occ_paths and flow_b_paths above.
    mb_b_paths : list[list[str]]
        Backward motion boundary mask paths, read mb_paths and flow_b_paths above.
    metadata : list[Any]
        Some metadata for each input. It can include anything. A good recommendation would be to put a dict with the metadata.
    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str = '',
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True
    ) -> None:
        """Initialize BaseFlowDataset.

        Parameters
        ----------
        dataset_name : str
            A string representing the dataset name. It is just used to be stored as metadata, so it can have any value.
        split_name : str, optional
            A string representing the split of the data. It is just used to be stored as metadata, so it can have any value.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        """
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.transform = transform
        self.max_flow = max_flow
        self.get_valid_mask = get_valid_mask
        self.get_occlusion_mask = get_occlusion_mask
        self.get_motion_boundary_mask = get_motion_boundary_mask
        self.get_backward = get_backward
        self.get_meta = get_meta

        self.img_paths = []
        self.flow_paths = []
        self.occ_paths = []
        self.mb_paths = []
        self.flow_b_paths = []
        self.occ_b_paths = []
        self.mb_b_paths = []
        self.metadata = []

    def __getitem__(  # noqa: C901
        self,
        index: int
    ) -> Dict[str, torch.Tensor]:
        """Retrieve and return one input.

        Parameters
        ----------
        index : int
            The index of the entry on the input lists.

        Returns
        -------
        Dict[str, torch.Tensor]
            The retrieved input. This dict may contain the following keys, depending on the initialization choices:
            ['images', 'flows', 'mbs', 'occs', 'valids', 'flows_b', 'mbs_b', 'occs_b', 'valids_b', 'meta'].
            Except for 'meta', all the values are 4D tensors with shape NCHW. Notice that N does not correspond to the batch
            size, but rather to the number of images of a given key. For example, typically 'images' will have N=2, and
            'flows' will have N=1, and so on. Therefore, a batch of these inputs will be a 5D tensor BNCHW.
        """
        inputs = {}

        inputs['images'] = [cv2.imread(str(path)) for path in self.img_paths[index]]

        if index < len(self.flow_paths):
            inputs['flows'], valids = self._get_flows_and_valids(self.flow_paths, index)
            if self.get_valid_mask:
                inputs['valids'] = valids

        if self.get_occlusion_mask and index < len(self.occ_paths):
            inputs['occs'] = [cv2.imread(str(path), 0)[:, :, None] for path in self.occ_paths[index]]
        if self.get_motion_boundary_mask and index < len(self.mb_paths):
            inputs['mbs'] = [cv2.imread(str(path), 0)[:, :, None] for path in self.mb_paths[index]]

        if self.get_backward:
            if index < len(self.flow_b_paths):
                inputs['flows_b'], valids_b = self._get_flows_and_valids(self.flow_b_paths, index)
                if self.get_valid_mask:
                    inputs['valids_b'] = valids_b
            if self.get_occlusion_mask and index < len(self.occ_b_paths):
                inputs['occs_b'] = [cv2.imread(str(path), 0)[:, :, None] for path in self.occ_b_paths[index]]
            if self.get_motion_boundary_mask and index < len(self.mb_b_paths):
                inputs['mbs_b'] = [cv2.imread(str(path), 0)[:, :, None] for path in self.mb_b_paths[index]]

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.get_meta:
            inputs['meta'] = {'dataset_name': self.dataset_name, 'split_name': self.split_name}
            if index < len(self.metadata):
                inputs['meta'].update(self.metadata[index])

        return inputs

    def __len__(self) -> int:
        return len(self.img_paths)

    def _get_flows_and_valids(
        self,
        flow_paths: Sequence[str],
        index: int
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        flows = []
        valids = []
        for path in flow_paths[index]:
            flow = flow_utils.flow_read(path)

            nan_mask = np.isnan(flow)
            flow[nan_mask] = self.max_flow + 1

            if self.get_valid_mask:
                valid = (np.abs(flow) < self.max_flow).astype(np.uint8)*255
                valid = np.minimum(valid[:, :, 0], valid[:, :, 1])
                valids.append(valid[:, :, None])

            flow[nan_mask] = 0

            flow = np.clip(flow, -self.max_flow, self.max_flow)
            flows.append(flow)
        return flows, valids

    def _log_status(self) -> None:
        if self.__len__() == 0:
            logging.warning(
                'No samples were found for %s dataset. Be sure to update the dataset path in datasets.yml, '
                'or provide the path by the argument --[dataset_name]_root_dir.', self.dataset_name)
        else:
            logging.info('Loading %d samples from %s dataset.', self.__len__(), self.dataset_name)


class FlyingChairsDataset(BaseFlowDataset):
    """Handle the FlyingChairs dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True
    ) -> None:
        """Initialize FlyingChairsDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingChairs dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_meta : bool, default True
            Whether to get metadata.
        """
        super().__init__(
            dataset_name='FlyingChairs',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta)
        self.root_dir = root_dir
        self.split_file = THIS_DIR / 'FlyingChairs_val.txt'

        # Read data from disk
        img1_paths = sorted((Path(self.root_dir) / 'data').glob('*img1.ppm'))
        img2_paths = sorted((Path(self.root_dir) / 'data').glob('*img2.ppm'))
        flow_paths = sorted((Path(self.root_dir) / 'data').glob('*flow.flo'))

        # Sanity check
        assert len(img1_paths) == len(img2_paths), f'{len(img1_paths)} vs {len(img2_paths)}'
        assert len(img1_paths) == len(flow_paths), f'{len(img1_paths)} vs {len(flow_paths)}'

        with open(self.split_file, 'r') as f:
            val_names = f.read().strip().splitlines()

        if split == 'trainval':
            remove_names = []
        elif split == 'train':
            remove_names = val_names
        elif split == 'val':
            remove_names = [p.stem.split('_')[0] for p in img1_paths if p.stem.split('_')[0] not in val_names]

        # Keep only data from the correct split
        self.img_paths = [
            [img1_paths[i], img2_paths[i]]
            for i in range(len(img1_paths)) if img1_paths[i].stem.split('_')[0] not in remove_names]
        self.flow_paths = [
            [flow_paths[i]] for i in range(len(flow_paths)) if flow_paths[i].stem.split('_')[0] not in remove_names]
        assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs {len(self.flow_paths)}'

        self.metadata = [
            {'image_paths': [str(p) for p in paths], 'is_val': paths[0].stem in val_names, 'misc': ''}
            for paths in self.img_paths]

        self._log_status()


class FlyingChairs2Dataset(BaseFlowDataset):
    """Handle the FlyingChairs 2 dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True
    ) -> None:
        """Initialize FlyingChairs2Dataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingChairs2 dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        """
        super().__init__(
            dataset_name='FlyingChairs2',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta)
        self.root_dir = root_dir
        self.add_reverse = add_reverse

        if split == 'train':
            dir_names = ['train']
        elif split == 'val':
            dir_names = ['val']
        else:
            dir_names = ['train', 'val']

        for dname in dir_names:
            # Read data from disk
            img1_paths = sorted((Path(self.root_dir) / dname).glob('*img_0.png'))
            img2_paths = sorted((Path(self.root_dir) / dname).glob('*img_1.png'))
            self.img_paths.extend([[img1_paths[i], img2_paths[i]] for i in range(len(img1_paths))])
            self.flow_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*flow_01.flo'))])
            self.occ_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*occ_01.png'))])
            self.mb_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*mb_01.png'))])
            if self.get_backward:
                self.flow_b_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*flow_10.flo'))])
                self.occ_b_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*occ_10.png'))])
                self.mb_b_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*mb_10.png'))])
            if self.add_reverse:
                self.img_paths.extend([[img2_paths[i], img1_paths[i]] for i in range(len(img1_paths))])
                self.flow_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*flow_10.flo'))])
                self.occ_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*occ_10.png'))])
                self.mb_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*mb_10.png'))])
                if self.get_backward:
                    self.flow_b_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*flow_01.flo'))])
                    self.occ_b_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*occ_01.png'))])
                    self.mb_b_paths.extend([[x] for x in sorted((Path(self.root_dir) / dname).glob('*mb_01.png'))])

        self.metadata = [{'image_paths': [str(p) for p in paths], 'is_val': False, 'misc': ''} for paths in self.img_paths]

        # Sanity check
        assert len(img1_paths) == len(img2_paths), f'{len(img1_paths)} vs {len(img2_paths)}'
        assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs {len(self.flow_paths)}'
        assert len(self.img_paths) == len(self.occ_paths), f'{len(self.img_paths)} vs {len(self.occ_paths)}'
        assert len(self.img_paths) == len(self.mb_paths), f'{len(self.img_paths)} vs {len(self.mb_paths)}'
        if self.get_backward:
            assert len(self.img_paths) == len(self.flow_b_paths), f'{len(self.img_paths)} vs {len(self.flow_b_paths)}'
            assert len(self.img_paths) == len(self.occ_b_paths), f'{len(self.img_paths)} vs {len(self.occ_b_paths)}'
            assert len(self.img_paths) == len(self.mb_b_paths), f'{len(self.img_paths)} vs {len(self.mb_b_paths)}'

        self._log_status()


class FlyingThings3DDataset(BaseFlowDataset):
    """Handle the FlyingThings3D dataset.

    Note that this only works for the complete FlyingThings3D dataset. For the subset version, use FlyingThings3DSubsetDataset.
    """

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = 'train',
        pass_names: Union[str, List[str]] = 'clean',
        side_names: Union[str, List[str]] = 'left',
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2
    ) -> None:
        """Initialize FlyingThings3DDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingThings3D dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        side_names : Union[str, List[str]], default 'left'
             Samples from which side view should be loaded. It can be one of {'left', 'right', ['left', 'right']}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        """
        super().__init__(
            dataset_name='FlyingThings3D',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta)
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        if split == 'val':
            split_dir_names = ['TEST']
        elif split == 'train':
            split_dir_names = ['TRAIN']
        else:
            split_dir_names = ['TRAIN', 'TEST']

        pass_dirs = [f'frames_{p}pass' for p in self.pass_names]

        directions = [('into_future', 'into_past')]
        reverts = [False]
        if self.add_reverse:
            directions.append(('into_past', 'into_future'))
            reverts.append(True)

        # Read paths from disk
        for passd in pass_dirs:
            for split in split_dir_names:
                split_path = Path(self.root_dir) / passd / split
                for letter_path in split_path.glob('*'):
                    for seq_path in letter_path.glob('*'):
                        for direcs, rev in zip(directions, reverts):
                            for side in self.side_names:
                                image_paths = sorted((seq_path / side).glob('*.png'), reverse=rev)
                                flow_paths = sorted(
                                    (Path(str(seq_path).replace(passd, 'optical_flow')) / direcs[0] / side).glob('*.pfm'),
                                    reverse=rev)

                                occ_paths = []
                                if (Path(self.root_dir) / 'occlusions').exists():
                                    occ_paths = sorted(
                                        (Path(str(seq_path).replace(passd, 'occlusions')) / direcs[0] / side).glob('*.png'),
                                        reverse=rev)
                                mb_paths = []
                                if (Path(self.root_dir) / 'motion_boundaries').exists():
                                    mb_paths = sorted(
                                        (Path(str(seq_path).replace(passd, 'motion_boundaries')) / direcs[0] / side).glob(
                                            '*.png'),
                                        reverse=rev)

                                flow_b_paths = []
                                occ_b_paths = []
                                mb_b_paths = []
                                if self.get_backward:
                                    flow_b_paths = sorted(
                                        (Path(str(seq_path).replace(passd, 'optical_flow')) / direcs[1] / side).glob('*.pfm'),
                                        reverse=rev)
                                    if (Path(self.root_dir) / 'occlusions').exists():
                                        occ_b_paths = sorted(
                                            (Path(str(seq_path).replace(passd, 'occlusions')) / direcs[1] / side).glob(
                                                '*.png'),
                                            reverse=rev)
                                    if (Path(self.root_dir) / 'motion_boundaries').exists():
                                        mb_b_paths = sorted(
                                            (Path(str(seq_path).replace(passd, 'motion_boundaries')) / direcs[1] / side).glob(
                                                '*.png'),
                                            reverse=rev)

                                for i in range(len(image_paths)-self.sequence_length+1):
                                    self.img_paths.append(image_paths[i:i+self.sequence_length])
                                    if len(flow_paths) > 0:
                                        self.flow_paths.append(flow_paths[i:i+self.sequence_length-1])
                                    if len(occ_paths) > 0:
                                        self.occ_paths.append(occ_paths[i:i+self.sequence_length-1])
                                    if len(mb_paths) > 0:
                                        self.mb_paths.append(mb_paths[i:i+self.sequence_length-1])
                                    self.metadata.append({
                                        'image_paths': [str(p) for p in image_paths[i:i+self.sequence_length]],
                                        'is_val': False,
                                        'misc': ''
                                    })
                                    if self.get_backward:
                                        if len(flow_b_paths) > 0:
                                            self.flow_b_paths.append(flow_b_paths[i+1:i+self.sequence_length])
                                        if len(occ_b_paths) > 0:
                                            self.occ_b_paths.append(occ_b_paths[i+1:i+self.sequence_length])
                                        if len(mb_b_paths) > 0:
                                            self.mb_b_paths.append(mb_b_paths[i+1:i+self.sequence_length])

        assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs {len(self.flow_paths)}'
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(self.occ_paths), (
            f'{len(self.img_paths)} vs {len(self.occ_paths)}')
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(self.mb_paths), (
            f'{len(self.img_paths)} vs {len(self.mb_paths)}')
        if self.get_backward:
            assert len(self.img_paths) == len(self.flow_b_paths), f'{len(self.img_paths)} vs {len(self.flow_b_paths)}'
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(self.occ_b_paths), (
                f'{len(self.img_paths)} vs {len(self.occ_b_paths)}')
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(self.mb_b_paths), (
                f'{len(self.img_paths)} vs {len(self.mb_b_paths)}')

        self._log_status()


class FlyingThings3DSubsetDataset(BaseFlowDataset):
    """Handle the FlyingThings3D subset dataset.

    Note that this only works for the FlyingThings3D subset dataset. For the complete version, use FlyingThings3DDataset.
    """

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = 'train',
        pass_names: Union[str, List[str]] = 'clean',
        side_names: Union[str, List[str]] = 'left',
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2
    ) -> None:
        """Initialize FlyingThings3DSubsetDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the FlyingThings3D dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        side_names : Union[str, List[str]], default 'left'
             Samples from which side view should be loaded. It can be one of {'left', 'right', ['left', 'right']}.
        add_reverse : bool, default True
            If True, double the number of samples by appending the backward samples as additional samples.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_motion_boundary_mask : bool, default True
            Whether to get motion boundary masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        """
        super().__init__(
            dataset_name='FlyingThings3DSubset',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta)
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        if split == 'train' or split == 'val':
            split_dir_names = [split]
        else:
            split_dir_names = ['train', 'val']

        directions = [('into_future', 'into_past')]
        reverts = [False]
        if self.add_reverse:
            directions.append(('into_past', 'into_future'))
            reverts.append(True)

        # Read paths from disk
        for split in split_dir_names:
            for side in self.side_names:
                for direcs, rev in zip(directions, reverts):
                    flow_dir = Path(self.root_dir) / split / 'flow' / side / direcs[0]
                    flow_paths = sorted(flow_dir.glob('*.flo'), reverse=rev)

                    # Create groups to separate different sequences
                    flow_groups_paths = [[flow_paths[0]]]
                    prev_idx = int(flow_paths[0].stem)
                    for path in flow_paths[1:]:
                        idx = int(path.stem)
                        if (idx - 1) == prev_idx:
                            flow_groups_paths[-1].append(path)
                        else:
                            flow_groups_paths.append([path])
                        prev_idx = idx

                    for flow_group in flow_groups_paths:
                        for i in range(len(flow_group)-self.sequence_length+2):
                            flow_paths = flow_group[i:i+self.sequence_length-1]
                            self.flow_paths.append(flow_paths)

                            img_dir = Path(self.root_dir) / split / 'image_clean' / side
                            img_paths = [img_dir / (fp.stem+'.png') for fp in flow_paths]
                            if rev:
                                idx = int(img_paths[0].stem) - 1
                            else:
                                idx = int(img_paths[-1].stem) + 1
                            img_paths.append(img_dir / f'{idx:07d}.png')
                            self.img_paths.append(img_paths)

                            if (Path(self.root_dir) / split / 'flow_occlusions').exists():
                                occ_paths = [
                                    Path(str(fp).replace('flow', 'flow_occlusions').replace('.flo', '.png'))
                                    for fp in flow_paths]
                                self.occ_paths.append(occ_paths)
                            if (Path(self.root_dir) / split / 'motion_boundaries').exists():
                                mb_paths = [
                                    Path(str(fp).replace('flow', 'motion_boundaries').replace('.flo', '.png'))
                                    for fp in flow_paths]
                                self.mb_paths.append(mb_paths)

                    if self.get_backward:
                        flow_dir = Path(self.root_dir) / split / 'flow' / side / direcs[1]
                        flow_paths = sorted(flow_dir.glob('*.flo'), reverse=rev)

                        # Create groups to separate different sequences
                        flow_groups_paths = [[flow_paths[0]]]
                        prev_idx = int(flow_paths[0].stem)
                        for path in flow_paths[1:]:
                            idx = int(path.stem)
                            if (idx - 1) == prev_idx:
                                flow_groups_paths[-1].append(path)
                            else:
                                flow_groups_paths.append([path])
                            prev_idx = idx

                        for flow_group in flow_groups_paths:
                            for i in range(len(flow_group)-self.sequence_length+2):
                                flow_paths = flow_group[i:i+self.sequence_length-1]
                                self.flow_b_paths.append(flow_paths)

                                if (Path(self.root_dir) / split / 'flow_occlusions').exists():
                                    occ_paths = [
                                        Path(str(fp).replace('flow', 'flow_occlusions').replace('.flo', '.png'))
                                        for fp in flow_paths]
                                    self.occ_b_paths.append(occ_paths)
                                if (Path(self.root_dir) / split / 'motion_boundaries').exists():
                                    mb_paths = [
                                        Path(str(fp).replace('flow', 'motion_boundaries').replace('.flo', '.png'))
                                        for fp in flow_paths]
                                    self.mb_b_paths.append(mb_paths)

        assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs  {len(self.flow_paths)}'
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(self.occ_paths), (
            f'{len(self.img_paths)} vs {len(self.occ_paths)}')
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(self.mb_paths), (
            f'{len(self.img_paths)} vs {len(self.mb_paths)}')
        if self.get_backward:
            assert len(self.img_paths) == len(self.flow_b_paths), f'{len(self.img_paths)} vs {len(self.flow_b_paths)}'
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(self.occ_b_paths), (
                f'{len(self.img_paths)} vs {len(self.occ_b_paths)}')
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(self.mb_b_paths), (
                f'{len(self.img_paths)} vs {len(self.mb_b_paths)}')

        self._log_status()


class Hd1kDataset(BaseFlowDataset):
    """Handle the HD1K dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = 'train',
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 512.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2
    ) -> None:
        """Initialize Hd1kDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the HD1K dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 512.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        """
        super().__init__(
            dataset_name='HD1K',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta)
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length

        if split == 'test':
            split_dir = 'hd1k_challenge'
        else:
            split_dir = 'hd1k_input'

        img_paths = sorted((Path(root_dir) / split_dir / 'image_2').glob('*.png'))
        img_names = [p.stem for p in img_paths]

        # Group paths by sequence
        img_names_grouped = {}
        for n in img_names:
            seq_name = n.split('_')[0]
            if img_names_grouped.get(seq_name) is None:
                img_names_grouped[seq_name] = []
            img_names_grouped[seq_name].append(n)

        val_names = []
        split_file = THIS_DIR / 'Hd1k_val.txt'
        with open(split_file, 'r') as f:
            val_names = f.read().strip().splitlines()

        # Remove names that do not belong to the chosen split
        for seq_name, seq_img_names in img_names_grouped.items():
            if split == 'train':
                img_names_grouped[seq_name] = [n for n in seq_img_names if n not in val_names]
            elif split == 'val':
                img_names_grouped[seq_name] = [n for n in seq_img_names if n in val_names]

        for seq_img_names in img_names_grouped.values():
            for i in range(len(seq_img_names) - self.sequence_length + 1):
                self.img_paths.append(
                    [Path(root_dir) / split_dir / 'image_2' / (n+'.png') for n in seq_img_names[i:i+self.sequence_length]])
                if split != 'test':
                    self.flow_paths.append(
                        [Path(root_dir) / 'hd1k_flow_gt' / 'flow_occ' / (n+'.png')
                         for n in seq_img_names[i:i+self.sequence_length-1]])

                self.metadata.append({
                    'image_paths': [str(p) for p in self.img_paths[-1]],
                    'is_val': (seq_img_names[i] in val_names),
                    'misc': ''
                })

        if split != 'test':
            assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs {len(self.flow_paths)}'

        self._log_status()


class KittiDataset(BaseFlowDataset):
    """Handle the KITTI dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir_2012: Optional[str] = None,
        root_dir_2015: Optional[str] = None,
        split: str = 'train',
        versions: Union[str, List[str]] = '2015',
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 512.0,
        get_valid_mask: bool = True,
        get_meta: bool = True
    ) -> None:
        """Initialize KittiDataset.

        Parameters
        ----------
        root_dir_2012 : str, optional.
            Path to the root directory of the KITTI 2012 dataset, if available.
        root_dir_2015 : str, optional.
            Path to the root directory of the KITTI 2015 dataset, if available.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        versions : Union[str, List[str]], default '2015'
            Which version should be loaded. It can be one of {'2012', '2015', ['2012', '2015']}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 512.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_meta : bool, default True
            Whether to get metadata.
        """
        if isinstance(versions, str):
            versions = [versions]
        super().__init__(
            dataset_name=f'KITTI_{"_".join(versions)}',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta)
        self.root_dir = {'2012': root_dir_2012, '2015': root_dir_2015}
        self.versions = versions
        self.split = split

        if split == 'test':
            split_dir = 'testing'
        else:
            split_dir = 'training'

        for ver in versions:
            if self.root_dir[ver] is None:
                continue

            if ver == '2012':
                image_dir = 'colored_0'
            else:
                image_dir = 'image_2'

            img1_paths = sorted((Path(self.root_dir[ver]) / split_dir / image_dir).glob('*_10.png'))
            img2_paths = sorted((Path(self.root_dir[ver]) / split_dir / image_dir).glob('*_11.png'))
            assert len(img1_paths) == len(img2_paths), f'{len(img1_paths)} vs {len(img2_paths)}'
            flow_paths = []
            if split != 'test':
                flow_paths = sorted((Path(self.root_dir[ver]) / split_dir / 'flow_occ').glob('*_10.png'))
                assert len(img1_paths) == len(flow_paths), f'{len(img1_paths)} vs {len(flow_paths)}'

            split_file = THIS_DIR / f'Kitti{ver}_val.txt'
            with open(split_file, 'r') as f:
                val_names = f.read().strip().splitlines()

            if split == 'trainval' or split == 'test':
                remove_names = []
            elif split == 'train':
                remove_names = val_names
            elif split == 'val':
                remove_names = [p.stem for p in img1_paths if p.stem not in val_names]

            self.img_paths = [
                [img1_paths[i], img2_paths[i]] for i in range(len(img1_paths)) if img1_paths[i].stem not in remove_names]
            if split != 'test':
                self.flow_paths = [
                    [flow_paths[i]] for i in range(len(flow_paths)) if flow_paths[i].stem not in remove_names]
            self.metadata = [
                {
                    'image_paths': [str(img1_paths[i]), str(img2_paths[i])],
                    'is_val': img1_paths[i].stem in val_names,
                    'misc': ver
                } for i in range(len(img1_paths)) if img1_paths[i].stem not in remove_names]

        if split != 'test':
            assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs {len(self.flow_paths)}'

        self._log_status()


class SintelDataset(BaseFlowDataset):
    """Handle the MPI Sintel dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = 'train',
        pass_names: Union[str, List[str]] = 'clean',
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2
    ) -> None:
        """Initialize SintelDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the MPI Sintel dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        pass_names : Union[str, List[str]], default 'clean'
            Which passes should be loaded. It can be one of {'clean', 'final', ['clean', 'final']}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        """
        if isinstance(pass_names, str):
            pass_names = [pass_names]
        super().__init__(
            dataset_name=f'Sintel_{"_".join(pass_names)}',
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta)
        self.root_dir = root_dir
        self.split = split
        self.pass_names = pass_names
        self.sequence_length = sequence_length

        # Get sequence names for the given split
        if split == 'test':
            split_dir = 'test'
        else:
            split_dir = 'training'

        split_file = THIS_DIR / 'Sintel_val.txt'
        with open(split_file, 'r') as f:
            val_seqs = f.read().strip().splitlines()

        sequence_names = sorted([p.stem for p in (Path(root_dir) / split_dir / 'clean').glob('*')])
        if split == 'train' or split == 'val':
            if split == 'train':
                sequence_names = [s for s in sequence_names if s not in val_seqs]
            else:
                sequence_names = val_seqs

        # Read paths from disk
        for passd in pass_names:
            for seq_name in sequence_names:
                image_paths = sorted((Path(self.root_dir) / split_dir / passd / seq_name).glob('*.png'))
                flow_paths = []
                occ_paths = []
                if split != 'test':
                    flow_paths = sorted((Path(self.root_dir) / split_dir / 'flow' / seq_name).glob('*.flo'))
                    assert len(image_paths)-1 == len(flow_paths), (
                        f'{passd}, {seq_name}: {len(image_paths)-1} vs {len(flow_paths)}')
                    if (Path(self.root_dir) / split_dir / 'occlusions').exists():
                        occ_paths = sorted((Path(self.root_dir) / split_dir / 'occlusions' / seq_name).glob('*.png'))
                        assert len(occ_paths) == len(flow_paths)
                for i in range(len(image_paths)-self.sequence_length+1):
                    self.img_paths.append(image_paths[i:i+self.sequence_length])
                    if len(flow_paths) > 0:
                        self.flow_paths.append(flow_paths[i:i+self.sequence_length-1])
                    if len(occ_paths) > 0:
                        self.occ_paths.append(occ_paths[i:i+self.sequence_length-1])
                    self.metadata.append({
                        'image_paths': [str(p) for p in image_paths[i:i+self.sequence_length]],
                        'is_val': seq_name in val_seqs,
                        'misc': seq_name
                    })

        # Sanity check
        if split != 'test':
            assert len(self.img_paths) == len(self.flow_paths), f'{len(self.img_paths)} vs {len(self.flow_paths)}'
        if len(self.occ_paths) > 0:
            assert len(self.img_paths) == len(self.occ_paths), f'{len(self.img_paths)} vs {len(self.occ_paths)}'

        self._log_status()
