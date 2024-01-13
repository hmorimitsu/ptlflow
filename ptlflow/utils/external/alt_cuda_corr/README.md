# alt_cuda_corr

CUDA implementation of local correlation calculation (cost volume).
Originally implemented in RAFT to avoid computing a global cost volume.
It decreases memory consumption, but increases running time.

## Installation instructions

1. Download and install CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
  - IMPORTANT! Be sure to choose the same CUDA version as your PyTorch
2. Enter this folder and then run the setup:
```bash
cd ptlflow/utils/external/alt_cuda_corr/
python setup.py install
```

## Original source

[https://github.com/princeton-vl/RAFT/tree/master/alt_cuda_corr](https://github.com/princeton-vl/RAFT/tree/master/alt_cuda_corr)

## LICENSE

See [LICENSE](LICENSE)