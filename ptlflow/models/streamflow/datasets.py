import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        if self.seg_list is not None:
            f_in = np.array(frame_utils.read_gen(self.seg_list[index]))
            seg_r = f_in[:, :, 0].astype("int32")
            seg_g = f_in[:, :, 1].astype("int32")
            seg_b = f_in[:, :, 2].astype("int32")
            seg_map = (seg_r * 256 + seg_g) * 256 + seg_b
            seg_map = torch.from_numpy(seg_map)

        if self.seg_inv_list is not None:
            seg_inv = frame_utils.read_gen(self.seg_inv_list[index])
            seg_inv = np.array(seg_inv).astype(np.uint8)
            seg_inv = torch.from_numpy(seg_inv // 255).bool()

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        print(img1.shape, img2.shape)

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.occ_list is not None:
            return img1, img2, flow, valid.float(), occ, self.occ_list[index]
        elif self.seg_list is not None and self.seg_inv_list is not None:
            return img1, img2, flow, valid.float(), seg_map, seg_inv
        else:
            return img1, img2, flow, valid.float()  # , self.extra_info[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="training",
        root="/share/zhouzhengguang/backup/data/sintel",
        dstype="clean",
        occlusion=False,
        segmentation=False,
    ):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)
        occ_root = osp.join(root, split, "occlusions")
        # occ_root = osp.join(root, split, 'occ_plus_out')
        # occ_root = osp.join(root, split, 'in_frame_occ')
        # occ_root = osp.join(root, split, 'out_of_frame')

        seg_root = osp.join(root, split, "segmentation")
        seg_inv_root = osp.join(root, split, "segmentation_invalid")
        self.segmentation = segmentation
        self.occlusion = occlusion
        if self.occlusion:
            self.occ_list = []
        if self.segmentation:
            self.seg_list = []
            self.seg_inv_list = []

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))
                if self.occlusion:
                    self.occ_list += sorted(glob(osp.join(occ_root, scene, "*.png")))
                if self.segmentation:
                    self.seg_list += sorted(glob(osp.join(seg_root, scene, "*.png")))
                    self.seg_inv_list += sorted(
                        glob(osp.join(seg_inv_root, scene, "*.png"))
                    )


class FlyingChairs(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="training",
        root="/share/zhouzhengguang/backup/data/FlyingChairs_release/data",
    ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, "*.ppm")))
        flows = sorted(glob(osp.join(root, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt(
            root[:-4] + "FlyingChairs_train_val.txt", dtype=np.int32
        )
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="/share/zhouzhengguang/backup/data/flyingthings3d",
        split="training",
        dstype="frames_cleanpass",
    ):
        super(FlyingThings3D, self).__init__(aug_params)

        if split == "training":
            for cam in ["left"]:
                for direction in ["into_future", "into_past"]:
                    image_dirs = sorted(glob(osp.join(root, dstype, "TRAIN/*/*")))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, "optical_flow/TRAIN/*/*")))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, "*.png")))
                        flows = sorted(glob(osp.join(fdir, "*.pfm")))
                        for i in range(len(flows) - 1):
                            if direction == "into_future":
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == "into_past":
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]

        elif split == "validation":
            for cam in ["left"]:
                for direction in ["into_future", "into_past"]:
                    image_dirs = sorted(glob(osp.join(root, dstype, "TEST/*/*")))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, "optical_flow/TEST/*/*")))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, "*.png")))
                        flows = sorted(glob(osp.join(fdir, "*.pfm")))
                        for i in range(len(flows) - 1):
                            if direction == "into_future":
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == "into_past":
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]

                valid_list = np.loadtxt("things_val_test_set.txt", dtype=np.int32)
                self.image_list = [
                    self.image_list[ind] for ind, sel in enumerate(valid_list) if sel
                ]
                self.flow_list = [
                    self.flow_list[ind] for ind, sel in enumerate(valid_list) if sel
                ]


class KITTI(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="training",
        root="/share/zhouzhengguang/backup/data/KITTI",
    ):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == "testing":
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="/share/zhouzhengguang/backup/data/HD1k"):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS="C+T+K+S+H"):
    """Create the data loader for the corresponding training set"""

    if args.stage == "chairs":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.1,
            "max_scale": 1.0,
            "do_flip": True,
        }
        train_dataset = FlyingChairs(
            aug_params, split="training", root=args.chairs_root
        )

    elif args.stage == "things":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.4,
            "max_scale": 0.8,
            "do_flip": True,
        }
        clean_dataset = FlyingThings3D(
            aug_params,
            dstype="frames_cleanpass",
            split="training",
            root=args.things_root,
        )
        final_dataset = FlyingThings3D(
            aug_params,
            dstype="frames_finalpass",
            split="training",
            root=args.things_root,
        )
        train_dataset = clean_dataset + final_dataset

    elif args.stage == "sintel":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.6,
            "do_flip": True,
        }
        things = FlyingThings3D(
            aug_params, dstype="frames_cleanpass", root=args.things_root
        )
        sintel_clean = MpiSintel(
            aug_params, split="training", dstype="clean", root=args.sintel_root
        )
        sintel_final = MpiSintel(
            aug_params, split="training", dstype="final", root=args.sintel_root
        )

        if TRAIN_DS == "C+T+K+S+H":
            kitti = KITTI(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.3,
                    "max_scale": 0.5,
                    "do_flip": True,
                },
                root=args.kitti_root,
            )
            hd1k = HD1K(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.5,
                    "max_scale": 0.2,
                    "do_flip": True,
                },
                root=args.hd1k_root,
            )
            train_dataset = (
                100 * sintel_clean
                + 100 * sintel_final
                + 200 * kitti
                + 5 * hd1k
                + things
            )

        elif TRAIN_DS == "C+T+K/S":
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.stage == "kitti":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.4,
            "do_flip": False,
        }
        train_dataset = KITTI(aug_params, split="training", root=args.kitti_root)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    print("Training with %d image pairs" % len(train_dataset))
    return train_loader


if __name__ == "__main__":
    # SintelMultiframeEval(split='training', dstype='clean', nframes=2)
    final_dataset = FlyingThings3D(
        dstype="frames_finalpass",
        split="training",
        root="/share/sunshangkun/flyingthings3d",
    )
    final_dataset[0]
