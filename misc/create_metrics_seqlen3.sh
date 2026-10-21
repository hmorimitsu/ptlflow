#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/../validate.py" \
    --data.val_dataset sintel-clean-occ-seqlen_3+sintel-final-occ-seqlen_3 \
    --select ${@}
