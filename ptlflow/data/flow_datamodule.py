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

from typing import Optional

import lightning.pytorch as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset
import yaml

from ptlflow.data import flow_transforms as ft
from ptlflow.data.datasets import (
    AutoFlowDataset,
    FlyingChairsDataset,
    FlyingChairs2Dataset,
    Hd1kDataset,
    KittiDataset,
    KubricDataset,
    MiddleburySTDataset,
    SintelDataset,
    FlyingThings3DDataset,
    FlyingThings3DSubsetDataset,
    SpringDataset,
    TartanAirDataset,
    ViperDataset,
)
from ptlflow.utils.utils import make_divisible


class FlowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        predict_dataset: Optional[str] = None,
        test_dataset: Optional[str] = None,
        train_dataset: Optional[str] = None,
        val_dataset: Optional[str] = None,
        train_batch_size: Optional[int] = None,
        train_num_workers: int = 4,
        train_crop_size: tuple[int, int] = None,
        train_transform_cuda: bool = False,
        train_transform_fp16: bool = False,
        autoflow_root_dir: Optional[str] = None,
        flying_chairs_root_dir: Optional[str] = None,
        flying_chairs2_root_dir: Optional[str] = None,
        flying_things3d_root_dir: Optional[str] = None,
        flying_things3d_subset_root_dir: Optional[str] = None,
        mpi_sintel_root_dir: Optional[str] = None,
        kitti_2012_root_dir: Optional[str] = None,
        kitti_2015_root_dir: Optional[str] = None,
        hd1k_root_dir: Optional[str] = None,
        tartanair_root_dir: Optional[str] = None,
        spring_root_dir: Optional[str] = None,
        kubric_root_dir: Optional[str] = None,
        middlebury_st_root_dir: Optional[str] = None,
        viper_root_dir: Optional[str] = None,
        dataset_config_path: str = "./datasets.yaml",
    ):
        super().__init__()
        self.predict_dataset = predict_dataset
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_crop_size = train_crop_size
        self.train_transform_cuda = train_transform_cuda
        self.train_transform_fp16 = train_transform_fp16

        self.autoflow_root_dir = autoflow_root_dir
        self.flying_chairs_root_dir = flying_chairs_root_dir
        self.flying_chairs2_root_dir = flying_chairs2_root_dir
        self.flying_things3d_root_dir = flying_things3d_root_dir
        self.flying_things3d_subset_root_dir = flying_things3d_subset_root_dir
        self.mpi_sintel_root_dir = mpi_sintel_root_dir
        self.kitti_2012_root_dir = kitti_2012_root_dir
        self.kitti_2015_root_dir = kitti_2015_root_dir
        self.hd1k_root_dir = hd1k_root_dir
        self.tartanair_root_dir = tartanair_root_dir
        self.spring_root_dir = spring_root_dir
        self.kubric_root_dir = kubric_root_dir
        self.middlebury_st_root_dir = middlebury_st_root_dir
        self.viper_root_dir = viper_root_dir
        self.dataset_config_path = dataset_config_path

        self.predict_dataset_parsed = None
        self.test_dataset_parsed = None
        self.train_dataset_parsed = None
        self.val_dataset_parsed = None

        self.train_dataloader_length = 0
        self.train_epoch_step = 0

        self.val_dataloader_names = []
        self.val_dataloader_lengths = []

        self.test_dataloader_names = []

    def setup(self, stage):
        self._load_dataset_paths()

        if stage == "fit":
            assert (
                self.train_dataset is not None
            ), "You need to provide a value for --data.train_dataset"
            assert (
                self.val_dataset is not None
            ), "You need to provide a value for --data.val_dataset"

            if self.train_dataset is None:
                self.train_dataset = "chairs-train"
                logger.warning(
                    "--data.train_dataset is not set. It will be set as {}",
                    self.train_dataset,
                )
            if self.train_batch_size is None:
                self.train_batch_size = 8
                logger.warning(
                    "--data.train_batch_size is not set. It will be set to {}",
                    self.train_batch_size,
                )

            self.train_dataset_parsed = self._parse_dataset_selection(
                self.train_dataset
            )
            self.val_dataset_parsed = self._parse_dataset_selection(self.val_dataset)
        elif stage == "predict":
            assert (
                self.predict_dataset is not None
            ), "You need to provide a value for --data.predict_dataset"
            self.parsed_predict_dataset_parsed = self._parse_dataset_selection(
                self.predict_dataset
            )
        elif stage == "test":
            assert (
                self.test_dataset is not None
            ), "You need to provide a value for --data.test_dataset"
            self.test_dataset_parsed = self._parse_dataset_selection(self.test_dataset)
        elif stage == "validate":
            assert (
                self.val_dataset is not None
            ), "You need to provide a value for --data.val_dataset"
            self.val_dataset_parsed = self._parse_dataset_selection(self.val_dataset)

    def predict_dataloader(self):
        return super().predict_dataloader()

    def test_dataloader(self):
        dataset_ids = [self.test_dataset]
        if "sintel" in dataset_ids:
            dataset_ids.remove("sintel")
            dataset_ids.extend(["sintel-clean", "sintel-final"])
        elif "spring" in dataset_ids:
            dataset_ids.append("spring-revonly")

        dataloaders = []
        for dataset_id in dataset_ids:
            dataset_id += "-test"
            dataset_tokens = dataset_id.split("-")
            dataset = getattr(self, f"_get_{dataset_tokens[0]}_dataset")(
                False, *dataset_tokens[1:]
            )
            dataloaders.append(
                DataLoader(
                    dataset,
                    1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False,
                )
            )

            self.test_dataloader_names.append(dataset_id)

        return dataloaders

    def train_dataloader(self):
        if self.train_dataset_parsed is not None:
            train_dataset = None
            for parsed_vals in self.train_dataset_parsed:
                multiplier = parsed_vals[0]
                dataset_name = parsed_vals[1]
                dataset = getattr(self, f"_get_{dataset_name}_dataset")(
                    True, *parsed_vals[2:]
                )
                dataset_mult = dataset
                for _ in range(multiplier - 1):
                    dataset_mult += dataset

                if train_dataset is None:
                    train_dataset = dataset_mult
                else:
                    train_dataset += dataset_mult

            train_pin_memory = False if self.train_transform_cuda else True
            train_dataloader = DataLoader(
                train_dataset,
                self.train_batch_size,
                shuffle=True,
                num_workers=self.train_num_workers,
                pin_memory=train_pin_memory,
                drop_last=False,
                persistent_workers=self.train_transform_cuda,
            )
            self.train_dataloader_length = len(train_dataloader)
            return train_dataloader

    def val_dataloader(self):
        dataloaders = []
        self.val_dataloader_names = []
        self.val_dataloader_lengths = []
        for parsed_vals in self.val_dataset_parsed:
            dataset_name = parsed_vals[1]
            dataset = getattr(self, f"_get_{dataset_name}_dataset")(
                False, *parsed_vals[2:]
            )
            dataloaders.append(
                DataLoader(
                    dataset,
                    1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False,
                    persistent_workers=self.train_transform_cuda,
                )
            )

            self.val_dataloader_names.append("-".join(parsed_vals[1:]))
            self.val_dataloader_lengths.append(len(dataset))

        return dataloaders

    def _load_dataset_paths(self):
        with open(self.dataset_config_path, "r") as f:
            dataset_paths = yaml.safe_load(f)
        for name, path in dataset_paths.items():
            if getattr(self, f"{name}_root_dir") is None:
                setattr(self, f"{name}_root_dir", path)

    def _parse_dataset_selection(
        self,
        dataset_selection: str,
    ) -> list[tuple[str, int]]:
        """Parse the input string into the selected dataset and their multipliers and parameters.

        For example, 'chairs-train+3*sintel-clean-trainval+kitti-2012-train*5' will be parsed into
        [(1, 'chairs', 'train'), (3, 'sintel', 'clean', 'trainval'), (5, 'kitti', '2012', 'train')].

        Parameters
        ----------
        dataset_selection : str
            The string defining the dataset selection. Each dataset is separated by a '+' sign. The multiplier must be either
            in the beginning or the end of one dataset string, connected to a '*' sign. The remaining content must be separated
            by '-' symbols.

        Returns
        -------
        List[Tuple[str, int]]
            The parsed choice of datasets and their number of repetitions.

        Raises
        ------
        ValueError
            If the given string is invalid.
        """
        if dataset_selection is None:
            return []

        dataset_selection = dataset_selection.replace(" ", "")
        datasets = dataset_selection.split("+")
        for i in range(len(datasets)):
            tokens = datasets[i].split("*")
            if len(tokens) == 1:
                datasets[i] = (1,) + tuple(tokens[0].split("-"))
            elif len(tokens) == 2:
                try:
                    mult, params = int(tokens[0]), tokens[1]
                except ValueError:
                    params, mult = tokens[0], int(
                        tokens[1]
                    )  # if the multiplier comes last.
                datasets[i] = (mult,) + tuple(params.split("-"))
            else:
                raise ValueError(
                    "The specified dataset string {:} is invalid. Check the BaseModel.parse_dataset_selection() documentation "
                    "to see how to write a valid selection string."
                )
        return datasets

    def _get_model_output_stride(self):
        if hasattr(self, "trainer") and self.trainer is not None:
            if hasattr(self.trainer.model, "module"):
                return self.trainer.model.module.output_stride
            else:
                return self.trainer.model.output_stride
        else:
            return 1

    ###########################################################################
    # _get_datasets
    ###########################################################################

    def _get_autoflow_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        fbocc_transform = False
        for v in args:
            if v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(368, self._get_model_output_stride()),
                    md(496, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx),
                        (-0.1, 1.0),
                        (-0.2, 0.2),
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        split = "trainval"
        if len(args) > 0 and args[0] in ["train", "val", "trainval"]:
            split = args[0]
        dataset = AutoFlowDataset(
            self.autoflow_root_dir, split=split, transform=transform
        )
        return dataset

    def _get_chairs_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        fbocc_transform = False
        split = "trainval"
        for v in args:
            if v in ["train", "val", "trainval"]:
                split = args[0]
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(368, self._get_model_output_stride()),
                    md(496, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx),
                        (-0.1, 1.0),
                        (-0.2, 0.2),
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = FlyingChairsDataset(
            self.flying_chairs_root_dir, split=split, transform=transform
        )
        return dataset

    def _get_chairs2_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        split = "trainval"
        add_reverse = False
        get_occlusion_mask = False
        get_motion_boundary_mask = False
        get_backward = False
        fbocc_transform = False
        for v in args:
            if v in ["train", "val", "trainval"]:
                split = v
            elif v == "rev":
                add_reverse = True
            elif v == "occ":
                get_occlusion_mask = True
            elif v == "mb":
                get_motion_boundary_mask = True
            elif v == "back":
                get_backward = True
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(368, self._get_model_output_stride()),
                    md(496, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx),
                        (-0.1, 1.0),
                        (-0.2, 0.2),
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = FlyingChairs2Dataset(
            self.flying_chairs2_root_dir,
            split=split,
            transform=transform,
            add_reverse=add_reverse,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
        )
        return dataset

    def _get_hd1k_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        split = "trainval"
        sequence_length = 2
        sequence_position = "first"
        fbocc_transform = False
        for v in args:
            if v in ["train", "val", "trainval", "test"]:
                split = args[0]
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(368, self._get_model_output_stride()),
                    md(768, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx), (-0.5, 0.2), (-0.2, 0.2), sparse=True
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = Hd1kDataset(
            self.hd1k_root_dir,
            split=split,
            transform=transform,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_kitti_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        versions = ["2012", "2015"]
        split = "trainval"
        get_occlusion_mask = False
        fbocc_transform = False
        for v in args:
            if v in ["2012", "2015"]:
                versions = [v]
            elif v in ["train", "val", "trainval", "test"]:
                split = v
            elif v == "occ":
                get_occlusion_mask = True
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(288, self._get_model_output_stride()),
                    md(960, self._get_model_output_stride()),
                )
                # cy, cx = (md(416, self._get_model_output_stride()), md(960, self._get_model_output_stride()))
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx), (-0.2, 0.4), (-0.2, 0.2), sparse=True
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = KittiDataset(
            self.kitti_2012_root_dir,
            self.kitti_2015_root_dir,
            get_occlusion_mask=get_occlusion_mask,
            versions=versions,
            split=split,
            transform=transform,
        )
        return dataset

    def _get_kubric_dataset(self, is_train: bool, *args: str) -> Dataset:
        if is_train:
            raise NotImplementedError()
        else:
            transform = ft.ToTensor()

        get_backward = False
        sequence_length = 2
        sequence_position = "first"
        max_seq = None
        for v in args:
            if v == "back":
                get_backward = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v.startswith("maxseq"):
                max_seq = int(v.split("_")[1])

        dataset = KubricDataset(
            self.kubric_root_dir,
            transform=transform,
            get_backward=get_backward,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
            max_seq=max_seq,
        )
        return dataset

    def _get_sintel_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        pass_names = ["clean", "final"]
        split = "trainval"
        get_occlusion_mask = False
        sequence_length = 2
        sequence_position = "first"
        fbocc_transform = False
        for v in args:
            if v in ["clean", "final"]:
                pass_names = [v]
            elif v in ["train", "val", "trainval", "test"]:
                split = v
            elif v == "occ":
                get_occlusion_mask = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(368, self._get_model_output_stride()),
                    md(768, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (-0.2, 0.6), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = SintelDataset(
            self.mpi_sintel_root_dir,
            split=split,
            pass_names=pass_names,
            transform=transform,
            get_occlusion_mask=get_occlusion_mask,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_sintel_finetune_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        fbocc_transform = False
        searaft_split = False
        for v in args:
            if v == "fbocc":
                fbocc_transform = True
            elif v == "searaft_split":
                searaft_split = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(368, self._get_model_output_stride()),
                    md(768, self._get_model_output_stride()),
                )
                # cy, cx = (md(416, self._get_model_output_stride()), md(960, self._get_model_output_stride()))
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform1 = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (-0.2, 0.6), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
            things_dataset = FlyingThings3DDataset(
                self.flying_things3d_root_dir,
                split="train",
                pass_names=["clean"],
                side_names=["left"],
                transform=transform1,
                get_backward=False,
                get_motion_boundary_mask=False,
                get_occlusion_mask=False,
            )

            sintel_clean_dataset = SintelDataset(
                self.mpi_sintel_root_dir,
                split="trainval",
                pass_names=["clean"],
                transform=transform1,
                get_occlusion_mask=False,
            )
            sintel_clean_mult_dataset = sintel_clean_dataset
            for _ in range(19 if searaft_split else 99):
                sintel_clean_mult_dataset += sintel_clean_dataset

            sintel_final_dataset = SintelDataset(
                self.mpi_sintel_root_dir,
                split="trainval",
                pass_names=["final"],
                transform=transform1,
                get_occlusion_mask=False,
            )
            sintel_final_mult_dataset = sintel_final_dataset
            for _ in range(19 if searaft_split else 99):
                sintel_final_mult_dataset += sintel_final_dataset

            transform2 = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx), (-0.3, 0.5), (-0.2, 0.2), sparse=True
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
            kitti_dataset = KittiDataset(
                root_dir_2015=self.kitti_2015_root_dir,
                split="trainval",
                versions=["2015"],
                transform=transform2,
                get_occlusion_mask=False,
            )
            kitti_mult_dataset = kitti_dataset
            for _ in range(79 if searaft_split else 199):
                kitti_mult_dataset += kitti_dataset

            transform3 = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx), (-0.5, 0.2), (-0.2, 0.2), sparse=True
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
            hd1k_dataset = Hd1kDataset(
                self.hd1k_root_dir, split="trainval", transform=transform3
            )
            hd1k_mult_dataset = hd1k_dataset
            for _ in range(29 if searaft_split else 4):
                hd1k_mult_dataset += hd1k_dataset

            mixed_dataset = (
                things_dataset
                + sintel_clean_mult_dataset
                + sintel_final_mult_dataset
                + kitti_mult_dataset
                + hd1k_mult_dataset
            )

            logger.info("Loaded datasets:")
            logger.info(
                "FlyingThings3D: unique samples {} - multiplied samples {}",
                len(things_dataset),
                len(things_dataset),
            )
            logger.info(
                "Sintel clean: unique samples {} - multiplied samples {}",
                len(sintel_clean_dataset),
                len(sintel_clean_mult_dataset),
            )
            logger.info(
                "Sintel final: unique samples {} - multiplied samples {}",
                len(sintel_final_dataset),
                len(sintel_final_mult_dataset),
            )
            logger.info(
                "KITTI 2015: unique samples {} - multiplied samples {}",
                len(kitti_dataset),
                len(kitti_mult_dataset),
            )
            logger.info(
                "HD1K: unique samples {} - multiplied samples {}",
                len(hd1k_dataset),
                len(hd1k_mult_dataset),
            )
            logger.info("Total dataset size: {}", len(mixed_dataset))
        else:
            raise NotImplementedError()

        return mixed_dataset

    def _get_spring_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        split = "train"
        add_reverse = False
        get_backward = False
        sequence_length = 2
        sequence_position = "first"
        reverse_only = False
        subsample = False
        is_image_4k = False
        side_names = []
        fbocc_transform = False
        for v in args:
            if v in ["train", "val", "trainval", "test"]:
                split = v
            elif v == "rev":
                add_reverse = True
            elif v == "revonly":
                reverse_only = True
            elif v == "back":
                get_backward = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "sub":
                subsample = True
            elif v == "4k":
                is_image_4k = True
            elif v == "left":
                side_names.append("left")
            elif v == "right":
                side_names.append("right")
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(540, self._get_model_output_stride()),
                    md(960, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # Transforms copied from SEA-RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (0.0, 0.2), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        if len(side_names) == 0:
            side_names = ["left", "right"]

        dataset = SpringDataset(
            self.spring_root_dir,
            split=split,
            side_names=side_names,
            add_reverse=add_reverse,
            transform=transform,
            get_backward=get_backward,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
            reverse_only=reverse_only,
            subsample=subsample,
            is_image_4k=is_image_4k,
        )
        return dataset

    def _get_tartanair_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        get_occlusion_mask = False
        sequence_length = 2
        sequence_position = "first"
        difficulties = []
        fbocc_transform = False
        for v in args:
            if v in ["easy", "hard"]:
                difficulties.append(v)
            elif v == "occ":
                get_occlusion_mask = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if len(difficulties) == 0:
            difficulties = ["easy"]

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(360, self._get_model_output_stride()),
                    md(480, self._get_model_output_stride()),
                )
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (-0.4, 0.8), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = TartanAirDataset(
            self.tartanair_root_dir,
            difficulties=difficulties,
            transform=transform,
            get_occlusion_mask=get_occlusion_mask,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_things_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.train_transform_cuda else "cpu"
        md = make_divisible

        pass_names = ["clean", "final"]
        split = "trainval"
        is_subset = False
        add_reverse = False
        get_occlusion_mask = False
        get_motion_boundary_mask = False
        get_backward = False
        sequence_length = 2
        sequence_position = "first"
        sintel_transform = False
        fbocc_transform = False
        for v in args:
            if v in ["clean", "final"]:
                pass_names = [v]
            elif v in ["train", "val", "trainval"]:
                split = v
            elif v == "subset":
                is_subset = True
            elif v == "rev":
                add_reverse = True
            elif v == "occ":
                get_occlusion_mask = True
            elif v == "mb":
                get_motion_boundary_mask = True
            elif v == "back":
                get_backward = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "sinteltransform":
                sintel_transform = True
            elif v == "fbocc":
                fbocc_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.train_crop_size is None:
                cy, cx = (
                    md(400, self._get_model_output_stride()),
                    md(720, self._get_model_output_stride()),
                )
                # cy, cx = (md(416, self._get_model_output_stride()), md(960, self._get_model_output_stride()))
                self.train_crop_size = (cy, cx)
                logger.warning(
                    "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.train_crop_size[0], self._get_model_output_stride()),
                    md(self.train_crop_size[1], self._get_model_output_stride()),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            if sintel_transform:
                major_scale = (-0.2, 0.6)
            else:
                major_scale = (-0.4, 0.8)
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), major_scale, (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                    (
                        ft.GenerateFBCheckFlowOcclusion(threshold=1)
                        if fbocc_transform
                        else None
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        if is_subset:
            dataset = FlyingThings3DSubsetDataset(
                self.flying_things3d_subset_root_dir,
                split=split,
                pass_names=pass_names,
                side_names=["left", "right"],
                add_reverse=add_reverse,
                transform=transform,
                get_occlusion_mask=get_occlusion_mask,
                get_motion_boundary_mask=get_motion_boundary_mask,
                get_backward=get_backward,
                sequence_length=sequence_length,
                sequence_position=sequence_position,
            )
        else:
            dataset = FlyingThings3DDataset(
                self.flying_things3d_root_dir,
                split=split,
                pass_names=pass_names,
                side_names=["left", "right"],
                add_reverse=add_reverse,
                transform=transform,
                get_occlusion_mask=get_occlusion_mask,
                get_motion_boundary_mask=get_motion_boundary_mask,
                get_backward=get_backward,
                sequence_length=sequence_length,
                sequence_position=sequence_position,
            )
        return dataset

    def _get_middlebury_st_dataset(self, is_train: bool, *args: str) -> Dataset:
        assert not is_train
        transform = ft.ToTensor()

        dataset = MiddleburySTDataset(
            self.middlebury_st_root_dir,
            transform=transform,
        )
        return dataset

    def _get_viper_dataset(self, is_train: bool, *args: str) -> Dataset:
        assert not is_train
        transform = ft.ToTensor()

        dataset = ViperDataset(
            self.viper_root_dir,
            split="val",
            transform=transform,
        )
        return dataset

    def _get_overfit_dataset(self, is_train: bool, *args: str) -> Dataset:
        md = make_divisible
        if self.train_crop_size is None:
            cy, cx = (
                md(436, self._get_model_output_stride()),
                md(1024, self._get_model_output_stride()),
            )
            self.train_crop_size = (cy, cx)
            logger.warning(
                "--train_crop_size is not set. It will be set as ({}, {}).", cy, cx
            )
        else:
            cy, cx = (
                md(self.train_crop_size[0], self._get_model_output_stride()),
                md(self.train_crop_size[1], self._get_model_output_stride()),
            )
        transform = ft.Compose([ft.ToTensor(), ft.Resize((cy, cx))])

        dataset_name = "sintel"
        if len(args) > 0 and args[0] in ["chairs2"]:
            dataset_name = args[0]

        if dataset_name == "sintel":
            dataset = SintelDataset(
                self.mpi_sintel_root_dir,
                split="trainval",
                pass_names="clean",
                transform=transform,
                get_occlusion_mask=False,
            )
        elif dataset_name == "chairs2":
            dataset = FlyingChairs2Dataset(
                self.flying_chairs2_root_dir,
                split="trainval",
                transform=transform,
                add_reverse=False,
                get_occlusion_mask=True,
                get_motion_boundary_mask=True,
                get_backward=True,
            )

        dataset.img_paths = dataset.img_paths[:1]
        dataset.flow_paths = dataset.flow_paths[:1]
        dataset.occ_paths = dataset.occ_paths[:1]
        dataset.mb_paths = dataset.mb_paths[:1]
        dataset.flow_b_paths = dataset.flow_b_paths[:1]
        dataset.occ_b_paths = dataset.occ_b_paths[:1]
        dataset.mb_b_paths = dataset.mb_b_paths[:1]
        dataset.metadata = dataset.metadata[:1]

        return dataset
