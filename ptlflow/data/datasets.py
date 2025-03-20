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

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2 as cv
from einops import rearrange
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from ptlflow.utils import flow_utils

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
        split_name: str = "",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
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

        self.flow_format = None

        self.is_two_file_flow = False

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: C901
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

        inputs["images"] = [cv.imread(str(path)) for path in self.img_paths[index]]

        if index < len(self.flow_paths):
            inputs["flows"], valids = self._get_flows_and_valids(
                self.flow_paths[index],
                flow_format=self.flow_format,
            )
            if self.get_valid_mask:
                inputs["valids"] = valids

        if self.get_occlusion_mask:
            if index < len(self.occ_paths):
                inputs["occs"] = []
                for path in self.occ_paths[index]:
                    if str(path).endswith("npy"):
                        occ = np.load(path)
                    else:
                        occ = cv.imread(str(path), 0)
                    inputs["occs"].append(occ[:, :, None])
            elif self.dataset_name.startswith("KITTI"):
                noc_paths = [
                    str(p).replace("flow_occ", "flow_noc")
                    for p in self.flow_paths[index]
                ]
                _, valids_noc = self._get_flows_and_valids(
                    noc_paths,
                    flow_format=self.flow_format,
                )
                inputs["occs"] = [valids[i] - valids_noc[i] for i in range(len(valids))]
        if self.get_motion_boundary_mask and index < len(self.mb_paths):
            inputs["mbs"] = [
                cv.imread(str(path), 0)[:, :, None] for path in self.mb_paths[index]
            ]

        if self.get_backward:
            if index < len(self.flow_b_paths):
                inputs["flows_b"], valids_b = self._get_flows_and_valids(
                    self.flow_b_paths[index],
                    flow_format=self.flow_format,
                )
                if self.get_valid_mask:
                    inputs["valids_b"] = valids_b
            if self.get_occlusion_mask and index < len(self.occ_b_paths):
                inputs["occs_b"] = []
                for path in self.occ_b_paths[index]:
                    if str(path).endswith("npy"):
                        occ = np.load(path)
                    else:
                        occ = cv.imread(str(path), 0)
                    inputs["occs_b"].append(occ[:, :, None])
            if self.get_motion_boundary_mask and index < len(self.mb_b_paths):
                inputs["mbs_b"] = [
                    cv.imread(str(path), 0)[:, :, None]
                    for path in self.mb_b_paths[index]
                ]

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.get_meta:
            inputs["meta"] = {
                "dataset_name": self.dataset_name,
                "split_name": self.split_name,
            }
            if index < len(self.metadata):
                inputs["meta"].update(self.metadata[index])

        return inputs

    def __len__(self) -> int:
        return len(self.img_paths)

    def _get_flows_and_valids(
        self,
        flow_paths: Sequence[str],
        flow_format: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        flows = []
        valids = []
        for path in flow_paths:
            if self.is_two_file_flow:
                flow_x = -flow_utils.flow_read(path[0], format=flow_format)
                flow_y = -flow_utils.flow_read(path[1], format=flow_format)
                flow = np.stack([flow_x, flow_y], 2)
            else:
                flow = flow_utils.flow_read(path, format=flow_format)

            nan_mask = np.isnan(flow)
            flow[nan_mask] = self.max_flow + 1

            if self.get_valid_mask:
                valid = (np.abs(flow) < self.max_flow).astype(np.uint8) * 255
                valid = np.minimum(valid[:, :, 0], valid[:, :, 1])
                valids.append(valid[:, :, None])

            flow[nan_mask] = 0

            flow = np.clip(flow, -self.max_flow, self.max_flow)
            flows.append(flow)
        return flows, valids

    def _log_status(self) -> None:
        if self.__len__() == 0:
            logger.warning(
                "No samples were found for {} dataset. Be sure to update the dataset path in datasets.yml, "
                "or provide the path by the argument --[dataset_name]_root_dir.",
                self.dataset_name,
            )
        else:
            logger.info(
                "Loading {} samples from {} dataset.", self.__len__(), self.dataset_name
            )

    def _extend_paths_list(
        self,
        paths_list: List[Union[str, Path]],
        sequence_length: int,
        sequence_position: str,
    ):
        if sequence_position == "first":
            begin_pad = 0
            end_pad = sequence_length - 2
        elif sequence_position == "middle":
            begin_pad = sequence_length // 2
            end_pad = int(math.ceil(sequence_length / 2.0)) - 2
        elif sequence_position == "last":
            begin_pad = sequence_length - 2
            end_pad = 0
        elif sequence_position == "all":
            begin_pad = 0
            end_pad = 0
        else:
            raise ValueError(
                f"Invalid sequence_position. Must be one of ('first', 'middle', 'last', 'all'). Received: {sequence_position}"
            )
        for _ in range(begin_pad):
            paths_list.insert(0, paths_list[0])
        for _ in range(end_pad):
            paths_list.append(paths_list[-1])
        return paths_list


class AutoFlowDataset(BaseFlowDataset):
    """Handle the AutoFlow dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize AutoFlowDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the AutoFlow dataset.
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
            dataset_name="AutoFlow",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split_file = THIS_DIR / "AutoFlow_val.txt"

        # Read data from disk
        parts_dirs = [f"static_40k_png_{i+1}_of_4" for i in range(4)]
        sample_paths = []
        for pdir in parts_dirs:
            sample_paths.extend(
                [p for p in (Path(root_dir) / pdir).glob("*") if p.is_dir()]
            )

        with open(self.split_file, "r") as f:
            val_names = f.read().strip().splitlines()

        if split == "trainval":
            remove_names = []
        elif split == "train":
            remove_names = val_names
        elif split == "val":
            remove_names = [p.stem for p in sample_paths if p.stem not in val_names]

        # Keep only data from the correct split
        self.img_paths = [
            [p / "im0.png", p / "im1.png"]
            for p in sample_paths
            if p.stem not in remove_names
        ]
        self.flow_paths = [
            [p / "forward.flo"] for p in sample_paths if p.stem not in remove_names
        ]
        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self.metadata = [
            {
                "image_paths": [str(p) for p in paths],
                "is_val": paths[0].stem in val_names,
                "misc": "",
                "is_seq_start": True,
            }
            for paths in self.img_paths
        ]

        self._log_status()


class FlyingChairsDataset(BaseFlowDataset):
    """Handle the FlyingChairs dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
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
            dataset_name="FlyingChairs",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split_file = THIS_DIR / "FlyingChairs_val.txt"

        # Read data from disk
        img1_paths = sorted((Path(self.root_dir) / "data").glob("*img1.ppm"))
        img2_paths = sorted((Path(self.root_dir) / "data").glob("*img2.ppm"))
        flow_paths = sorted((Path(self.root_dir) / "data").glob("*flow.flo"))

        # Sanity check
        assert len(img1_paths) == len(
            img2_paths
        ), f"{len(img1_paths)} vs {len(img2_paths)}"
        assert len(img1_paths) == len(
            flow_paths
        ), f"{len(img1_paths)} vs {len(flow_paths)}"

        with open(self.split_file, "r") as f:
            val_names = f.read().strip().splitlines()

        if split == "trainval":
            remove_names = []
        elif split == "train":
            remove_names = val_names
        elif split == "val":
            remove_names = [
                p.stem.split("_")[0]
                for p in img1_paths
                if p.stem.split("_")[0] not in val_names
            ]

        # Keep only data from the correct split
        self.img_paths = [
            [img1_paths[i], img2_paths[i]]
            for i in range(len(img1_paths))
            if img1_paths[i].stem.split("_")[0] not in remove_names
        ]
        self.flow_paths = [
            [flow_paths[i]]
            for i in range(len(flow_paths))
            if flow_paths[i].stem.split("_")[0] not in remove_names
        ]
        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self.metadata = [
            {
                "image_paths": [str(p) for p in paths],
                "is_val": paths[0].stem in val_names,
                "misc": "",
                "is_seq_start": True,
            }
            for paths in self.img_paths
        ]

        self._log_status()


class FlyingChairs2Dataset(BaseFlowDataset):
    """Handle the FlyingChairs 2 dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        add_reverse: bool = False,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = False,
        get_motion_boundary_mask: bool = False,
        get_backward: bool = False,
        get_meta: bool = True,
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
            dataset_name="FlyingChairs2",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse

        if split == "train":
            dir_names = ["train"]
        elif split == "val":
            dir_names = ["val"]
        else:
            dir_names = ["train", "val"]

        for dname in dir_names:
            # Read data from disk
            img1_paths = sorted((Path(self.root_dir) / dname).glob("*img_0.png"))
            img2_paths = sorted((Path(self.root_dir) / dname).glob("*img_1.png"))
            self.img_paths.extend(
                [[img1_paths[i], img2_paths[i]] for i in range(len(img1_paths))]
            )
            self.flow_paths.extend(
                [
                    [x]
                    for x in sorted((Path(self.root_dir) / dname).glob("*flow_01.flo"))
                ]
            )
            self.occ_paths.extend(
                [[x] for x in sorted((Path(self.root_dir) / dname).glob("*occ_01.png"))]
            )
            self.mb_paths.extend(
                [[x] for x in sorted((Path(self.root_dir) / dname).glob("*mb_01.png"))]
            )
            if self.get_backward:
                self.flow_b_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*flow_10.flo")
                        )
                    ]
                )
                self.occ_b_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*occ_10.png")
                        )
                    ]
                )
                self.mb_b_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*mb_10.png")
                        )
                    ]
                )
            if self.add_reverse:
                self.img_paths.extend(
                    [[img2_paths[i], img1_paths[i]] for i in range(len(img1_paths))]
                )
                self.flow_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*flow_10.flo")
                        )
                    ]
                )
                self.occ_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*occ_10.png")
                        )
                    ]
                )
                self.mb_paths.extend(
                    [
                        [x]
                        for x in sorted(
                            (Path(self.root_dir) / dname).glob("*mb_10.png")
                        )
                    ]
                )
                if self.get_backward:
                    self.flow_b_paths.extend(
                        [
                            [x]
                            for x in sorted(
                                (Path(self.root_dir) / dname).glob("*flow_01.flo")
                            )
                        ]
                    )
                    self.occ_b_paths.extend(
                        [
                            [x]
                            for x in sorted(
                                (Path(self.root_dir) / dname).glob("*occ_01.png")
                            )
                        ]
                    )
                    self.mb_b_paths.extend(
                        [
                            [x]
                            for x in sorted(
                                (Path(self.root_dir) / dname).glob("*mb_01.png")
                            )
                        ]
                    )

        self.metadata = [
            {
                "image_paths": [str(p) for p in paths],
                "is_val": False,
                "misc": "",
                "is_seq_start": True,
            }
            for paths in self.img_paths
        ]

        # Sanity check
        assert len(img1_paths) == len(
            img2_paths
        ), f"{len(img1_paths)} vs {len(img2_paths)}"
        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        assert len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class FlyingThings3DDataset(BaseFlowDataset):
    """Handle the FlyingThings3D dataset.

    Note that this only works for the complete FlyingThings3D dataset. For the subset version, use FlyingThings3DSubsetDataset.
    """

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        pass_names: Union[str, List[str]] = "clean",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
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
            Whether to get the backward version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        """
        super().__init__(
            dataset_name="FlyingThings3D",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        if split == "val":
            split_dir_names = ["TEST"]
        elif split == "train":
            split_dir_names = ["TRAIN"]
        else:
            split_dir_names = ["TRAIN", "TEST"]

        pass_dirs = [f"frames_{p}pass" for p in self.pass_names]

        directions = [("into_future", "into_past")]
        reverts = [False]
        if self.add_reverse:
            directions.append(("into_past", "into_future"))
            reverts.append(True)

        # Read paths from disk
        for passd in pass_dirs:
            for split in split_dir_names:
                split_path = Path(self.root_dir) / passd / split
                for letter_path in split_path.glob("*"):
                    for seq_path in letter_path.glob("*"):
                        for direcs, rev in zip(directions, reverts):
                            for side in self.side_names:
                                image_paths = sorted(
                                    (seq_path / side).glob("*.png"), reverse=rev
                                )
                                image_paths = self._extend_paths_list(
                                    image_paths, sequence_length, sequence_position
                                )
                                flow_paths = sorted(
                                    (
                                        Path(
                                            str(seq_path).replace(passd, "optical_flow")
                                        )
                                        / direcs[0]
                                        / side
                                    ).glob("*.pfm"),
                                    reverse=rev,
                                )
                                flow_paths = self._extend_paths_list(
                                    flow_paths, sequence_length, sequence_position
                                )

                                occ_paths = []
                                if (Path(self.root_dir) / "occlusions").exists():
                                    occ_paths = sorted(
                                        (
                                            Path(
                                                str(seq_path).replace(
                                                    passd, "occlusions"
                                                )
                                            )
                                            / direcs[0]
                                            / side
                                        ).glob("*.png"),
                                        reverse=rev,
                                    )
                                    occ_paths = self._extend_paths_list(
                                        occ_paths, sequence_length, sequence_position
                                    )
                                mb_paths = []
                                if (Path(self.root_dir) / "motion_boundaries").exists():
                                    mb_paths = sorted(
                                        (
                                            Path(
                                                str(seq_path).replace(
                                                    passd, "motion_boundaries"
                                                )
                                            )
                                            / direcs[0]
                                            / side
                                        ).glob("*.png"),
                                        reverse=rev,
                                    )
                                    mb_paths = self._extend_paths_list(
                                        mb_paths, sequence_length, sequence_position
                                    )

                                flow_b_paths = []
                                occ_b_paths = []
                                mb_b_paths = []
                                if self.get_backward:
                                    flow_b_paths = sorted(
                                        (
                                            Path(
                                                str(seq_path).replace(
                                                    passd, "optical_flow"
                                                )
                                            )
                                            / direcs[1]
                                            / side
                                        ).glob("*.pfm"),
                                        reverse=rev,
                                    )
                                    flow_b_paths = self._extend_paths_list(
                                        flow_b_paths, sequence_length, sequence_position
                                    )
                                    if (Path(self.root_dir) / "occlusions").exists():
                                        occ_b_paths = sorted(
                                            (
                                                Path(
                                                    str(seq_path).replace(
                                                        passd, "occlusions"
                                                    )
                                                )
                                                / direcs[1]
                                                / side
                                            ).glob("*.png"),
                                            reverse=rev,
                                        )
                                        occ_b_paths = self._extend_paths_list(
                                            occ_b_paths,
                                            sequence_length,
                                            sequence_position,
                                        )
                                    if (
                                        Path(self.root_dir) / "motion_boundaries"
                                    ).exists():
                                        mb_b_paths = sorted(
                                            (
                                                Path(
                                                    str(seq_path).replace(
                                                        passd, "motion_boundaries"
                                                    )
                                                )
                                                / direcs[1]
                                                / side
                                            ).glob("*.png"),
                                            reverse=rev,
                                        )
                                        mb_b_paths = self._extend_paths_list(
                                            mb_b_paths,
                                            sequence_length,
                                            sequence_position,
                                        )

                                step_size = (
                                    (self.sequence_length - 1)
                                    if sequence_position == "all"
                                    else 1
                                )
                                for i in range(
                                    0,
                                    len(image_paths) - self.sequence_length + 1,
                                    step_size,
                                ):
                                    self.img_paths.append(
                                        image_paths[i : i + self.sequence_length]
                                    )
                                    if len(flow_paths) > 0:
                                        self.flow_paths.append(
                                            flow_paths[i : i + self.sequence_length - 1]
                                        )
                                    if len(occ_paths) > 0:
                                        self.occ_paths.append(
                                            occ_paths[i : i + self.sequence_length - 1]
                                        )
                                    if len(mb_paths) > 0:
                                        self.mb_paths.append(
                                            mb_paths[i : i + self.sequence_length - 1]
                                        )
                                    self.metadata.append(
                                        {
                                            "image_paths": [
                                                str(p)
                                                for p in image_paths[
                                                    i : i + self.sequence_length
                                                ]
                                            ],
                                            "is_val": False,
                                            "misc": "",
                                            "is_seq_start": i == 0,
                                        }
                                    )
                                    if self.get_backward:
                                        if len(flow_b_paths) > 0:
                                            self.flow_b_paths.append(
                                                flow_b_paths[
                                                    i + 1 : i + self.sequence_length
                                                ]
                                            )
                                        if len(occ_b_paths) > 0:
                                            self.occ_b_paths.append(
                                                occ_b_paths[
                                                    i + 1 : i + self.sequence_length
                                                ]
                                            )
                                        if len(mb_b_paths) > 0:
                                            self.mb_b_paths.append(
                                                mb_b_paths[
                                                    i + 1 : i + self.sequence_length
                                                ]
                                            )

        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class FlyingThings3DSubsetDataset(BaseFlowDataset):
    """Handle the FlyingThings3D subset dataset.

    Note that this only works for the FlyingThings3D subset dataset. For the complete version, use FlyingThings3DDataset.
    """

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        pass_names: Union[str, List[str]] = "clean",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_motion_boundary_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
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
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        """
        super().__init__(
            dataset_name="FlyingThings3DSubset",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        if split == "train" or split == "val":
            split_dir_names = [split]
        else:
            split_dir_names = ["train", "val"]

        directions = [("into_future", "into_past")]
        reverts = [False]
        if self.add_reverse:
            directions.append(("into_past", "into_future"))
            reverts.append(True)

        # Read paths from disk
        for split in split_dir_names:
            for pass_name in self.pass_names:
                for side in self.side_names:
                    for direcs, rev in zip(directions, reverts):
                        flow_dir = (
                            Path(self.root_dir) / split / "flow" / side / direcs[0]
                        )
                        flow_paths = sorted(flow_dir.glob("*.flo"), reverse=rev)

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
                            flow_group = self._extend_paths_list(
                                flow_group, sequence_length, sequence_position
                            )
                            step_size = (
                                (self.sequence_length - 1)
                                if sequence_position == "all"
                                else 1
                            )
                            for i in range(
                                0, len(flow_group) - self.sequence_length + 2, step_size
                            ):
                                flow_paths = flow_group[
                                    i : i + self.sequence_length - 1
                                ]
                                self.flow_paths.append(flow_paths)

                                img_dir = (
                                    Path(self.root_dir)
                                    / split
                                    / f"image_{pass_name}"
                                    / side
                                )
                                img_paths = [
                                    img_dir / (fp.stem + ".png") for fp in flow_paths
                                ]
                                if rev:
                                    idx = int(img_paths[0].stem) - 1
                                else:
                                    idx = int(img_paths[-1].stem) + 1
                                img_paths.append(img_dir / f"{idx:07d}.png")
                                self.img_paths.append(img_paths)

                                if (
                                    Path(self.root_dir) / split / "flow_occlusions"
                                ).exists():
                                    occ_paths = [
                                        Path(
                                            str(fp)
                                            .replace("flow", "flow_occlusions")
                                            .replace(".flo", ".png")
                                        )
                                        for fp in flow_paths
                                    ]
                                    self.occ_paths.append(occ_paths)
                                if (
                                    Path(self.root_dir) / split / "motion_boundaries"
                                ).exists():
                                    mb_paths = [
                                        Path(
                                            str(fp)
                                            .replace("flow", "motion_boundaries")
                                            .replace(".flo", ".png")
                                        )
                                        for fp in flow_paths
                                    ]
                                    self.mb_paths.append(mb_paths)

                                self.metadata.append(
                                    {
                                        "image_paths": [str(p) for p in img_paths],
                                        "is_val": False,
                                        "misc": "",
                                        "is_seq_start": i == 0,
                                    }
                                )

                        if self.get_backward:
                            flow_dir = (
                                Path(self.root_dir) / split / "flow" / side / direcs[1]
                            )
                            flow_paths = sorted(flow_dir.glob("*.flo"), reverse=rev)

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
                                flow_group = self._extend_paths_list(
                                    flow_group, sequence_length, sequence_position
                                )
                                for i in range(
                                    len(flow_group) - self.sequence_length + 2
                                ):
                                    flow_paths = flow_group[
                                        i : i + self.sequence_length - 1
                                    ]
                                    self.flow_b_paths.append(flow_paths)

                                    if (
                                        Path(self.root_dir) / split / "flow_occlusions"
                                    ).exists():
                                        occ_paths = [
                                            Path(
                                                str(fp)
                                                .replace("flow", "flow_occlusions")
                                                .replace(".flo", ".png")
                                            )
                                            for fp in flow_paths
                                        ]
                                        self.occ_b_paths.append(occ_paths)
                                    if (
                                        Path(self.root_dir)
                                        / split
                                        / "motion_boundaries"
                                    ).exists():
                                        mb_paths = [
                                            Path(
                                                str(fp)
                                                .replace("flow", "motion_boundaries")
                                                .replace(".flo", ".png")
                                            )
                                            for fp in flow_paths
                                        ]
                                        self.mb_b_paths.append(mb_paths)

        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs  {len(self.flow_paths)}"
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class Hd1kDataset(BaseFlowDataset):
    """Handle the HD1K dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 512.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
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
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        """
        super().__init__(
            dataset_name="HD1K",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        if split == "test":
            split_dir = "hd1k_challenge"
        else:
            split_dir = "hd1k_input"

        img_paths = sorted((Path(root_dir) / split_dir / "image_2").glob("*.png"))
        img_names = [p.stem for p in img_paths]

        # Group paths by sequence
        img_names_grouped = {}
        for n in img_names:
            seq_name = n.split("_")[0]
            if img_names_grouped.get(seq_name) is None:
                img_names_grouped[seq_name] = []
            img_names_grouped[seq_name].append(n)

        val_names = []
        split_file = THIS_DIR / "Hd1k_val.txt"
        with open(split_file, "r") as f:
            val_names = f.read().strip().splitlines()

        # Remove names that do not belong to the chosen split
        for seq_name, seq_img_names in img_names_grouped.items():
            if split == "train":
                img_names_grouped[seq_name] = [
                    n for n in seq_img_names if n not in val_names
                ]
            elif split == "val":
                img_names_grouped[seq_name] = [
                    n for n in seq_img_names if n in val_names
                ]

        for seq_img_names in img_names_grouped.values():
            seq_img_names = self._extend_paths_list(
                seq_img_names, sequence_length, sequence_position
            )
            step_size = (self.sequence_length - 1) if sequence_position == "all" else 1
            for i in range(0, len(seq_img_names) - self.sequence_length + 1, step_size):
                self.img_paths.append(
                    [
                        Path(root_dir) / split_dir / "image_2" / (n + ".png")
                        for n in seq_img_names[i : i + self.sequence_length]
                    ]
                )
                if split != "test":
                    self.flow_paths.append(
                        [
                            Path(root_dir) / "hd1k_flow_gt" / "flow_occ" / (n + ".png")
                            for n in seq_img_names[i : i + self.sequence_length - 1]
                        ]
                    )

                self.metadata.append(
                    {
                        "image_paths": [str(p) for p in self.img_paths[-1]],
                        "is_val": (seq_img_names[i] in val_names),
                        "misc": "",
                        "is_seq_start": True,
                    }
                )

        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class KittiDataset(BaseFlowDataset):
    """Handle the KITTI dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir_2012: Optional[str] = None,
        root_dir_2015: Optional[str] = None,
        split: str = "train",
        versions: Union[str, List[str]] = "2015",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 512.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = False,
        get_meta: bool = True,
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
        get_occlusion_mask : bool, default True
            Whether to get occlusion masks.
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
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = {"2012": root_dir_2012, "2015": root_dir_2015}
        self.versions = versions
        self.split = split

        if split == "test":
            split_dir = "testing"
        else:
            split_dir = "training"

        for ver in versions:
            if self.root_dir[ver] is None:
                continue

            if ver == "2012":
                image_dir = "colored_0"
            else:
                image_dir = "image_2"

            img1_paths = sorted(
                (Path(self.root_dir[ver]) / split_dir / image_dir).glob("*_10.png")
            )
            img2_paths = sorted(
                (Path(self.root_dir[ver]) / split_dir / image_dir).glob("*_11.png")
            )
            assert len(img1_paths) == len(
                img2_paths
            ), f"{len(img1_paths)} vs {len(img2_paths)}"
            flow_paths = []

            if (
                split != "test"
                or (Path(self.root_dir[ver]) / split_dir / "flow_occ").exists()
            ):
                flow_paths = sorted(
                    (Path(self.root_dir[ver]) / split_dir / "flow_occ").glob("*_10.png")
                )
                assert len(img1_paths) == len(
                    flow_paths
                ), f"{len(img1_paths)} vs {len(flow_paths)}"

            split_file = THIS_DIR / f"Kitti{ver}_val.txt"
            with open(split_file, "r") as f:
                val_names = f.read().strip().splitlines()

            if split == "trainval" or split == "test":
                remove_names = []
            elif split == "train":
                remove_names = val_names
            elif split == "val":
                remove_names = [p.stem for p in img1_paths if p.stem not in val_names]

            self.img_paths.extend(
                [
                    [img1_paths[i], img2_paths[i]]
                    for i in range(len(img1_paths))
                    if img1_paths[i].stem not in remove_names
                ]
            )
            if (
                split != "test"
                or (Path(self.root_dir[ver]) / split_dir / "flow_occ").exists()
            ):
                self.flow_paths.extend(
                    [
                        [flow_paths[i]]
                        for i in range(len(flow_paths))
                        if flow_paths[i].stem not in remove_names
                    ]
                )
            self.metadata.extend(
                [
                    {
                        "image_paths": [str(img1_paths[i]), str(img2_paths[i])],
                        "is_val": img1_paths[i].stem in val_names,
                        "misc": ver,
                        "is_seq_start": True,
                    }
                    for i in range(len(img1_paths))
                    if img1_paths[i].stem not in remove_names
                ]
            )

        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class SintelDataset(BaseFlowDataset):
    """Handle the MPI Sintel dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        pass_names: Union[str, List[str]] = "clean",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
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
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
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
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        # Get sequence names for the given split
        if split == "test":
            split_dir = "test"
        else:
            split_dir = "training"

        split_file = THIS_DIR / "Sintel_val.txt"
        with open(split_file, "r") as f:
            val_seqs = f.read().strip().splitlines()

        sequence_names = sorted(
            [p.stem for p in (Path(root_dir) / split_dir / "clean").glob("*")]
        )
        if split == "train" or split == "val":
            if split == "train":
                sequence_names = [s for s in sequence_names if s not in val_seqs]
            else:
                sequence_names = val_seqs

        # Read paths from disk
        for passd in pass_names:
            for seq_name in sequence_names:
                image_paths = sorted(
                    (Path(self.root_dir) / split_dir / passd / seq_name).glob("*.png")
                )
                image_paths = self._extend_paths_list(
                    image_paths, sequence_length, sequence_position
                )
                flow_paths = []
                occ_paths = []
                if (
                    split != "test"
                    or (Path(self.root_dir) / split_dir / "flow").exists()
                ):
                    flow_paths = sorted(
                        (Path(self.root_dir) / split_dir / "flow" / seq_name).glob(
                            "*.flo"
                        )
                    )
                    flow_paths = self._extend_paths_list(
                        flow_paths, sequence_length, sequence_position
                    )
                    assert len(image_paths) - 1 == len(
                        flow_paths
                    ), f"{passd}, {seq_name}: {len(image_paths)-1} vs {len(flow_paths)}"
                    if (Path(self.root_dir) / split_dir / "occlusions").exists():
                        occ_paths = sorted(
                            (
                                Path(self.root_dir)
                                / split_dir
                                / "occlusions"
                                / seq_name
                            ).glob("*.png")
                        )
                        occ_paths = self._extend_paths_list(
                            occ_paths, sequence_length, sequence_position
                        )
                        assert len(occ_paths) == len(flow_paths)

                step_size = (
                    (self.sequence_length - 1) if sequence_position == "all" else 1
                )
                for i in range(
                    0, len(image_paths) - self.sequence_length + 1, step_size
                ):
                    self.img_paths.append(image_paths[i : i + self.sequence_length])
                    if len(flow_paths) > 0:
                        self.flow_paths.append(
                            flow_paths[i : i + self.sequence_length - 1]
                        )
                    if len(occ_paths) > 0:
                        self.occ_paths.append(
                            occ_paths[i : i + self.sequence_length - 1]
                        )
                    self.metadata.append(
                        {
                            "image_paths": [
                                str(p)
                                for p in image_paths[i : i + self.sequence_length]
                            ],
                            "is_val": seq_name in val_seqs,
                            "misc": seq_name,
                            "is_seq_start": i == 0,
                        }
                    )

        # Sanity check
        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        if len(self.occ_paths) > 0:
            assert len(self.img_paths) == len(
                self.occ_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"

        self._log_status()


class SpringDataset(BaseFlowDataset):
    """Handle the Spring dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_backward: bool = False,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
        reverse_only: bool = False,
        subsample: bool = False,
        is_image_4k: bool = False,
    ) -> None:
        """Initialize SintelDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the MPI Sintel dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
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
        get_backward : bool, default False
            Whether to get the backward version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        reverse_only : bool, default False
            If True, only uses the backward samples, discarding the forward ones.
        subsample : bool, default False
            If True, the groundtruth is subsampled from 4K to 2K by neareast subsampling.
            If False, and is_image_4k is also False, then the groundtruth is reshaped as: einops.rearrange("b c (h nh) (w nw) -> b (nh nw) c h w", nh=2, nw=2),
            which corresponds to stacking the predictions of every 2x2 blocks.
            If False, and is_image_4k is True, then the groundtruth is returned in its original 4D-shaped 4K resolution, but the flow values are doubled.
        is_image_4k : bool, default False
            If True, assumes the input images will be provided in 4K resolution, instead of the original 2K.
        """
        if isinstance(side_names, str):
            side_names = [side_names]
        super().__init__(
            dataset_name="Spring",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.side_names = side_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        self.subsample = subsample
        self.is_image_4k = is_image_4k

        if self.is_image_4k:
            assert not self.subsample

        # Get sequence names for the given split
        if split == "test":
            split_dir = "test"
        else:
            split_dir = "train"

        sequence_names = sorted(
            [p.stem for p in (Path(root_dir) / split_dir).glob("*")]
        )

        if reverse_only:
            directions = [("BW", "FW")]
        else:
            directions = [("FW", "BW")]
            if add_reverse:
                directions.append(("BW", "FW"))

        # Read paths from disk
        for seq_name in sequence_names:
            for side in side_names:
                for direcs in directions:
                    rev = direcs[0] == "BW"
                    image_paths = sorted(
                        (
                            Path(self.root_dir) / split_dir / seq_name / f"frame_{side}"
                        ).glob("*.png"),
                        reverse=rev,
                    )
                    image_paths = self._extend_paths_list(
                        image_paths, sequence_length, sequence_position
                    )
                    flow_paths = []
                    flow_b_paths = []
                    if split != "test":
                        flow_paths = sorted(
                            (
                                Path(self.root_dir)
                                / split_dir
                                / seq_name
                                / f"flow_{direcs[0]}_{side}"
                            ).glob("*.flo5"),
                            reverse=rev,
                        )
                        flow_paths = self._extend_paths_list(
                            flow_paths, sequence_length, sequence_position
                        )
                        assert len(image_paths) - 1 == len(
                            flow_paths
                        ), f"{seq_name}, {side}: {len(image_paths)-1} vs {len(flow_paths)}"
                        if self.get_backward:
                            flow_b_paths = sorted(
                                (
                                    Path(self.root_dir)
                                    / split_dir
                                    / seq_name
                                    / f"flow_{direcs[1]}_{side}"
                                ).glob("*.flo5"),
                                reverse=rev,
                            )
                            flow_b_paths = self._extend_paths_list(
                                flow_b_paths, sequence_length, sequence_position
                            )
                            assert len(image_paths) - 1 == len(
                                flow_paths
                            ), f"{seq_name}, {side}: {len(image_paths)-1} vs {len(flow_paths)}"

                    step_size = (
                        (self.sequence_length - 1) if sequence_position == "all" else 1
                    )
                    for i in range(
                        0, len(image_paths) - self.sequence_length + 1, step_size
                    ):
                        self.img_paths.append(image_paths[i : i + self.sequence_length])
                        if len(flow_paths) > 0:
                            self.flow_paths.append(
                                flow_paths[i : i + self.sequence_length - 1]
                            )
                        if self.get_backward and len(flow_b_paths) > 0:
                            self.flow_b_paths.append(
                                flow_b_paths[i : i + self.sequence_length - 1]
                            )
                        self.metadata.append(
                            {
                                "image_paths": [
                                    str(p)
                                    for p in image_paths[i : i + self.sequence_length]
                                ],
                                "is_val": False,
                                "misc": seq_name,
                                "is_seq_start": i == 0,
                            }
                        )

        # Sanity check
        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: C901
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

        inputs["images"] = [cv.imread(str(path)) for path in self.img_paths[index]]

        if index < len(self.flow_paths):
            inputs["flows"], valids = self._get_flows_and_valids(self.flow_paths[index])
            if self.get_valid_mask:
                inputs["valids"] = valids

        if self.get_backward:
            if index < len(self.flow_b_paths):
                inputs["flows_b"], valids_b = self._get_flows_and_valids(
                    self.flow_b_paths[index]
                )
                if self.get_valid_mask:
                    inputs["valids_b"] = valids_b

        if self.subsample:
            inputs["flows"] = [f[::2, ::2] for f in inputs["flows"]]
            inputs["valids"] = [v[::2, ::2] for v in inputs["valids"]]
            if self.get_backward:
                inputs["flows_b"] = [f[::2, ::2] for f in inputs["flows_b"]]
                inputs["valids_b"] = [v[::2, ::2] for v in inputs["valids_b"]]
            if self.transform is not None:
                inputs = self.transform(inputs)
        elif self.is_image_4k:
            inputs["images"] = [
                cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                for img in inputs["images"]
            ]
            if self.transform is not None:
                inputs = self.transform(inputs)
            if "flows" in inputs:
                inputs["flows"] = 2 * inputs["flows"]
                if self.get_backward:
                    inputs["flows_b"] = 2 * inputs["flows_b"]

                process_keys = [("flows", "valids")]
                if self.get_backward:
                    process_keys.append(("flows_b", "valids_b"))

                for flow_key, valid_key in process_keys:
                    flow = inputs[flow_key]
                    flow_stack = rearrange(
                        flow, "b c (h nh) (w nw) -> b (nh nw) c h w", nh=2, nw=2
                    )
                    flow_stack4 = flow_stack.repeat(1, 4, 1, 1, 1)
                    flow_stack4 = rearrange(
                        flow_stack4, "b (m n) c h w -> b m n c h w", m=4
                    )
                    diff = flow_stack[:, :, None] - flow_stack4
                    diff = rearrange(diff, "b m n c h w -> b (m n) c h w")
                    diff = torch.sqrt(torch.pow(diff, 2).sum(2))
                    max_diff, _ = diff.max(1)
                    max_diff = F.interpolate(
                        max_diff[:, None], scale_factor=2, mode="nearest"
                    )
                    inputs[valid_key] = (max_diff < 1.0).float()
        else:
            if self.transform is not None:
                inputs = self.transform(inputs)

            if "flows" in inputs:
                inputs["flows"] = rearrange(
                    inputs["flows"], "b c (h nh) (w nw) -> b (nh nw) c h w", nh=2, nw=2
                )
                inputs["valids"] = inputs["valids"][:, :, ::2, ::2]
                if self.get_backward:
                    inputs["flows_b"] = rearrange(
                        inputs["flows_b"],
                        "b c (h nh) (w nw) -> b (nh nw) c h w",
                        nh=2,
                        nw=2,
                    )
                    inputs["valids_b"] = inputs["valids_b"][:, :, ::2, ::2]

        if self.get_meta:
            inputs["meta"] = {
                "dataset_name": self.dataset_name,
                "split_name": self.split_name,
            }
            if index < len(self.metadata):
                inputs["meta"].update(self.metadata[index])

        return inputs


class TartanAirDataset(BaseFlowDataset):
    """Handle the TartanAir dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        difficulties: Union[str, List[str]] = "easy",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_occlusion_mask: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize TartanAirDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the MPI Sintel dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval', 'test'}.
        difficulties : Union[str, List[str]], default 'easy'
            Which difficulties should be loaded. It can be one of {'easy', 'hard', ['easy', 'hard']}.
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
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        """
        if isinstance(difficulties, str):
            difficulties = [difficulties]
        difficulties = [d.capitalize() for d in difficulties]
        super().__init__(
            dataset_name=f'TartanAir_{"_".join(difficulties)}',
            split_name="trainval",
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.difficulties = difficulties
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        sequence_paths = sorted([p for p in Path(root_dir).glob("*") if p.is_dir()])

        # Read paths from disk
        for seq_path in sequence_paths:
            for diff in difficulties:
                trajectory_paths = sorted(
                    [p for p in (seq_path / diff).glob("*") if p.is_dir()]
                )
                for traj_path in trajectory_paths:
                    image_paths = sorted((traj_path / "image_left").glob("*.png"))
                    image_paths = self._extend_paths_list(
                        image_paths, sequence_length, sequence_position
                    )

                    flow_paths = sorted((traj_path / "flow").glob("*_flow.npy"))
                    flow_paths = self._extend_paths_list(
                        flow_paths, sequence_length, sequence_position
                    )
                    assert len(image_paths) - 1 == len(
                        flow_paths
                    ), f"{seq_path.name}, {traj_path.name}: {len(image_paths)-1} vs {len(flow_paths)}"

                    occ_paths = []
                    if get_occlusion_mask:
                        occ_paths = sorted((traj_path / "flow").glob("*_mask.npy"))
                        occ_paths = self._extend_paths_list(
                            occ_paths, sequence_length, sequence_position
                        )
                        assert len(occ_paths) == len(flow_paths)

                    step_size = (
                        (self.sequence_length - 1) if sequence_position == "all" else 1
                    )
                    for i in range(
                        0, len(image_paths) - self.sequence_length + 1, step_size
                    ):
                        self.img_paths.append(image_paths[i : i + self.sequence_length])
                        if len(flow_paths) > 0:
                            self.flow_paths.append(
                                flow_paths[i : i + self.sequence_length - 1]
                            )
                        if len(occ_paths) > 0:
                            self.occ_paths.append(
                                occ_paths[i : i + self.sequence_length - 1]
                            )
                        self.metadata.append(
                            {
                                "image_paths": [
                                    str(p)
                                    for p in image_paths[i : i + self.sequence_length]
                                ],
                                "is_val": False,
                                "misc": seq_path.name,
                                "is_seq_start": i == 0,
                            }
                        )

        # Sanity check
        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        if len(self.occ_paths) > 0:
            assert len(self.img_paths) == len(
                self.occ_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"

        self._log_status()


class MiddleburyDataset(BaseFlowDataset):
    """Handle the Middlebury dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize MiddleburyDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Middlebury dataset.
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
        """
        super().__init__(
            dataset_name="Middlebury",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = 2

        # Get sequence names for the given split
        if split == "test":
            split_dir = "eval"
        else:
            split_dir = "other"

        sequence_names = sorted(
            [p.stem for p in (Path(root_dir) / f"{split_dir}-gt-flow").glob("*")]
        )

        # Read paths from disk
        for seq_name in sequence_names:
            image_paths = sorted(
                (Path(self.root_dir) / f"{split_dir}-data" / seq_name).glob("*.png")
            )
            flow_paths = []
            if split != "test":
                flow_paths = sorted(
                    (Path(self.root_dir) / f"{split_dir}-gt-flow" / seq_name).glob(
                        "*.flo"
                    )
                )
                assert len(image_paths) - 1 == len(
                    flow_paths
                ), f"{seq_name}: {len(image_paths)-1} vs {len(flow_paths)}"
            for i in range(len(image_paths) - self.sequence_length + 1):
                self.img_paths.append(image_paths[i : i + self.sequence_length])
                if len(flow_paths) > 0:
                    self.flow_paths.append(flow_paths[i : i + self.sequence_length - 1])
                self.metadata.append(
                    {
                        "image_paths": [
                            str(p) for p in image_paths[i : i + self.sequence_length]
                        ],
                        "is_val": False,
                        "misc": seq_name,
                        "is_seq_start": True,
                    }
                )

        # Sanity check
        if split != "test":
            assert len(self.img_paths) == len(
                self.flow_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"

        self._log_status()


class MiddleburySTDataset(BaseFlowDataset):
    """Handle the Middlebury-ST dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize MiddleburySTDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Middlebury dataset.
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
            dataset_name="MiddleburyST",
            split_name="trainval",
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.sequence_length = 2
        self.is_two_file_flow = True

        sequence_names = sorted(
            [p.stem for p in Path(self.root_dir).glob("*") if p.is_dir()]
        )

        # Read paths from disk
        for seq_name in sequence_names:
            image_paths = [
                Path(self.root_dir) / seq_name / "im0.png",
                Path(self.root_dir) / seq_name / "im1.png",
            ]
            self.img_paths.append(image_paths)
            disp_paths = [
                Path(self.root_dir) / seq_name / "disp0.pfm",
                Path(self.root_dir) / seq_name / "disp0y.pfm",
            ]
            self.flow_paths.append([disp_paths])
            self.metadata.append(
                {
                    "image_paths": [str(p) for p in image_paths],
                    "is_val": False,
                    "misc": seq_name,
                    "is_seq_start": True,
                }
            )

        self._log_status()


class MonkaaDataset(BaseFlowDataset):
    """Handle the Monkaa dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        pass_names: Union[str, List[str]] = "clean",
        side_names: Union[str, List[str]] = "left",
        add_reverse: bool = True,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 1000.0,
        get_valid_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
    ) -> None:
        """Initialize MonkaaDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Monkaa dataset.
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
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        """
        super().__init__(
            dataset_name="Monkaa",
            split_name="trainval",
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=get_backward,
            get_semantic_segmentation_labels=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.add_reverse = add_reverse
        self.pass_names = pass_names
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position
        if isinstance(self.pass_names, str):
            self.pass_names = [self.pass_names]
        self.side_names = side_names
        if isinstance(self.side_names, str):
            self.side_names = [self.side_names]

        pass_dirs = [f"frames_{p}pass" for p in self.pass_names]

        directions = [("into_future", "into_past")]
        reverts = [False]
        if self.add_reverse:
            directions.append(("into_past", "into_future"))
            reverts.append(True)

        # Read paths from disk
        for passd in pass_dirs:
            pass_path = Path(self.root_dir) / passd
            for seq_path in pass_path.glob("*"):
                for direcs, rev in zip(directions, reverts):
                    for side in self.side_names:
                        image_paths = sorted(
                            (seq_path / side).glob("*.png"), reverse=rev
                        )
                        image_paths = self._extend_paths_list(
                            image_paths, sequence_length, sequence_position
                        )
                        flow_paths = sorted(
                            (
                                Path(str(seq_path).replace(passd, "optical_flow"))
                                / direcs[0]
                                / side
                            ).glob("*.pfm"),
                            reverse=rev,
                        )
                        flow_paths = self._extend_paths_list(
                            flow_paths, sequence_length, sequence_position
                        )

                        flow_b_paths = []
                        if self.get_backward:
                            flow_b_paths = sorted(
                                (
                                    Path(str(seq_path).replace(passd, "optical_flow"))
                                    / direcs[1]
                                    / side
                                ).glob("*.pfm"),
                                reverse=rev,
                            )
                            flow_b_paths = self._extend_paths_list(
                                flow_b_paths, sequence_length, sequence_position
                            )

                        step_size = (
                            (self.sequence_length - 1)
                            if sequence_position == "all"
                            else 1
                        )
                        for i in range(
                            0, len(image_paths) - self.sequence_length + 1, step_size
                        ):
                            self.img_paths.append(
                                image_paths[i : i + self.sequence_length]
                            )
                            if len(flow_paths) > 0:
                                self.flow_paths.append(
                                    flow_paths[i : i + self.sequence_length - 1]
                                )
                            self.metadata.append(
                                {
                                    "image_paths": [
                                        str(p)
                                        for p in image_paths[
                                            i : i + self.sequence_length
                                        ]
                                    ],
                                    "is_val": False,
                                    "misc": "",
                                    "is_seq_start": i == 0,
                                }
                            )
                            if self.get_backward:
                                if len(flow_b_paths) > 0:
                                    self.flow_b_paths.append(
                                        flow_b_paths[i + 1 : i + self.sequence_length]
                                    )

        assert len(self.img_paths) == len(
            self.flow_paths
        ), f"{len(self.img_paths)} vs {len(self.flow_paths)}"
        assert len(self.occ_paths) == 0 or len(self.img_paths) == len(
            self.occ_paths
        ), f"{len(self.img_paths)} vs {len(self.occ_paths)}"
        assert len(self.mb_paths) == 0 or len(self.img_paths) == len(
            self.mb_paths
        ), f"{len(self.img_paths)} vs {len(self.mb_paths)}"
        if self.get_backward:
            assert len(self.img_paths) == len(
                self.flow_b_paths
            ), f"{len(self.img_paths)} vs {len(self.flow_b_paths)}"
            assert len(self.occ_b_paths) == 0 or len(self.img_paths) == len(
                self.occ_b_paths
            ), f"{len(self.img_paths)} vs {len(self.occ_b_paths)}"
            assert len(self.mb_b_paths) == 0 or len(self.img_paths) == len(
                self.mb_b_paths
            ), f"{len(self.img_paths)} vs {len(self.mb_b_paths)}"

        self._log_status()


class KubricDataset(BaseFlowDataset):
    """Handle datasets generated by Kubric."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_backward: bool = True,
        get_meta: bool = True,
        sequence_length: int = 2,
        sequence_position: str = "first",
        max_seq: Optional[int] = None,
    ) -> None:
        """Initialize KubricDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Kubric dataset.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_backward : bool, default True
            Whether to get the occluded version of the inputs.
        get_meta : bool, default True
            Whether to get metadata.
        sequence_length : int, default 2
            How many consecutive images are loaded per sample. More than two images can be used for model which exploit more
            temporal information.
        sequence_position : str, default "first"
            Only used when sequence_length > 2.
            Determines the position where the main image frame will be in the sequence. It can one of three values:
            - "first": the main frame will be the first one of the sequence,
            - "middle": the main frame will be in the middle of the sequence (at position sequence_length // 2),
            - "last": the main frame will be the penultimate in the sequence,
            - "all": all the frames are considered the main. The next sequence will start from the last frame in the last sequence plus one.
        """
        super().__init__(
            dataset_name=f"Kubric",
            split_name="trainval",
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_motion_boundary_mask=False,
            get_backward=get_backward,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.sequence_position = sequence_position

        self.flow_format = "kubric_png"

        sequence_dirs = sorted([p for p in (Path(root_dir)).glob("*") if p.is_dir()])
        sequence_dirs = sequence_dirs[:max_seq]

        for seq_dir in sequence_dirs:
            seq_name = seq_dir.name
            image_paths = sorted(seq_dir.glob("rgba_*.png"))
            image_paths = self._extend_paths_list(
                image_paths, sequence_length, sequence_position
            )
            flow_paths = sorted(seq_dir.glob("forward_flow_*.png"))[:-1]
            flow_paths = self._extend_paths_list(
                flow_paths, sequence_length, sequence_position
            )
            flow_paths = [(p, "forward_flow") for p in flow_paths]
            assert len(image_paths) - 1 == len(
                flow_paths
            ), f"{seq_name}: {len(image_paths)-1} vs {len(flow_paths)}"

            if get_backward:
                back_flow_paths = sorted(seq_dir.glob("backward_flow_*.png"))[1:]
                back_flow_paths = self._extend_paths_list(
                    back_flow_paths, sequence_length, sequence_position
                )
                back_flow_paths = [(p, "backward_flow") for p in back_flow_paths]
                assert len(image_paths) - 1 == len(
                    back_flow_paths
                ), f"{seq_name}: {len(image_paths)-1} vs {len(back_flow_paths)}"

            step_size = (self.sequence_length - 1) if sequence_position == "all" else 1
            for i in range(0, len(image_paths) - self.sequence_length + 1, step_size):
                self.img_paths.append(image_paths[i : i + self.sequence_length])
                if len(flow_paths) > 0:
                    self.flow_paths.append(flow_paths[i : i + self.sequence_length - 1])

                if get_backward:
                    self.flow_b_paths.append(
                        back_flow_paths[i : i + self.sequence_length - 1]
                    )

                self.metadata.append(
                    {
                        "image_paths": [
                            str(p) for p in image_paths[i : i + self.sequence_length]
                        ],
                        "is_val": False,
                        "misc": seq_name,
                        "is_seq_start": i == 0,
                    }
                )

        self._log_status()


class ViperDataset(BaseFlowDataset):
    """Handle the Viper dataset."""

    def __init__(  # noqa: C901
        self,
        root_dir: str,
        split: str = "train",
        transform: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = None,
        max_flow: float = 10000.0,
        get_valid_mask: bool = True,
        get_meta: bool = True,
    ) -> None:
        """Initialize ViperDataset.

        Parameters
        ----------
        root_dir : str
            path to the root directory of the Middlebury dataset.
        split : str, default 'train'
            Which split of the dataset should be loaded. It can be one of {'train', 'val', 'trainval'}.
        transform : Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], optional
            Transform to be applied on the inputs.
        max_flow : float, default 10000.0
            Maximum optical flow absolute value. Flow absolute values that go over this limit are clipped, and also marked
            as zero in the valid mask.
        img_extension : str
            Extension of the image file. It can be one of {'jpg', 'png'}.
        get_valid_mask : bool, default True
            Whether to get or generate valid masks.
        get_meta : bool, default True
            Whether to get metadata.
        """
        super().__init__(
            dataset_name="VIPER",
            split_name=split,
            transform=transform,
            max_flow=max_flow,
            get_valid_mask=get_valid_mask,
            get_occlusion_mask=False,
            get_motion_boundary_mask=False,
            get_backward=False,
            get_meta=get_meta,
        )
        self.root_dir = root_dir
        self.sequence_length = 2

        self.flow_format = "viper_npz"

        if split == "trainval":
            split_dirs = ["train", "val"]
        else:
            split_dirs = [split]

        for spdir in split_dirs:
            img_dir_path = Path(self.root_dir) / spdir / "img"
            flow_dir_path = Path(self.root_dir) / spdir / "flow"

            sequence_names = sorted(
                [p.stem for p in img_dir_path.glob("*") if p.is_dir()]
            )

            # Read paths from disk
            for seq_name in sequence_names:
                if flow_dir_path.exists():
                    flow_paths = sorted(list((flow_dir_path / seq_name).glob(f"*.npz")))
                    for fpath in flow_paths:
                        file_idx = int(fpath.stem.split("_")[1])
                        img1_path = (
                            img_dir_path / seq_name / f"{seq_name}_{(file_idx):05d}.png"
                        )
                        img2_path = (
                            img_dir_path
                            / seq_name
                            / f"{seq_name}_{(file_idx + 1):05d}.png"
                        )
                        if img1_path.exists() and img2_path.exists():
                            self.img_paths.append([img1_path, img2_path])
                            self.flow_paths.append([fpath])
                            self.metadata.append(
                                {
                                    "image_paths": [
                                        str(p) for p in [img1_path, img2_path]
                                    ],
                                    "is_val": spdir == "val",
                                    "misc": seq_name,
                                    "is_seq_start": True,
                                }
                            )
                else:
                    raise NotImplementedError()

        self._log_status()
