# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import ImageAugmentor

from loguru import logger as loguru_logger

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = ImageAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.image_list = []
        self.aug_params = aug_params

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
        _index = index
    
        while True:
            index = index % len(self.image_list)
            #print(index, "~~~~~~~~~~~~~")
            try:
                img1 = frame_utils.read_gen(self.image_list[index][0])
                img2 = frame_utils.read_gen(self.image_list[index][1])
            except Exception as e:
                loguru_logger.info("Index is {}, error is {}".format(index, e))
                index += 1
                continue
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            
            H, W, _ = img1.shape
            H2, W2, _ = img2.shape
            if H != H2 or W != W2:
                print(self.image_list[index][0])
                index += 1
                continue
            if H < self.aug_params["crop_size"][0] or W < self.aug_params["crop_size"][1]:
                index += 1
            else:
                break

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2 = self.augmentor(img1, img2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        # load mask
        _index = _index % 100000
        mask_file = "mae_mask/mask_46_62_48_{:06d}.npy".format(_index)
        mask = frame_utils.read_gen(mask_file)
        mask = torch.from_numpy(mask)

        return img1, img2, mask


    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class YoutubeVOS(FlowDataset):
    def __init__(self, aug_params=None, root='/mnt/lustre/share_data/shixiaoyu/youtubevos/'):
        super(YoutubeVOS, self).__init__(aug_params)

        splits = ["test_all_frames/JPEGImages", "train_all_frames/JPEGImages", "valid_all_frames/JPEGImages"]

        for split in splits:
            dir_root = osp.join(root, split)
            
            for dir in os.listdir(dir_root):
                image_list = sorted(glob(osp.join(dir_root, dir, '*.jpg')))

                for i in range(9, len(image_list)-1, 3):
                    self.image_list += [[image_list[i-9], image_list[i]]]
                    self.image_list += [[image_list[i], image_list[i-9]]]
        print(self.image_list[:4])
       

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    
    if args.stage == 'youtube':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_dataset = YoutubeVOS(aug_params)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=24, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
