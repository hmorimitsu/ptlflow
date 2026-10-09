#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/../validate.py" \
    --data.val_dataset sintel-clean-occ+sintel-final-occ+kitti-2012+kitti-2015 \
    --select ${@}
