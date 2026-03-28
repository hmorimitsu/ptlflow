#!/bin/bash
python ../validate.py \
    --data.val_dataset sintel-clean-occ+sintel-final-occ+kitti-2012+kitti-2015 \
    --select ${@}