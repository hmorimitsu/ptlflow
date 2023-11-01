# MatchFlow

## Original code

[https://github.com/DQiaole/MatchFlow](https://github.com/DQiaole/MatchFlow)

## Additional requirements

In order to use MatchFlow you need to have CUDA installed and compile the QuadTreeAttention package:

- Download and install CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
  - IMPORTANT! Be sure to choose the same CUDA version as your PyTorch
- Enter the QuadTreeAttention folder and run the setup:
```bash
cd QuadTreeAttention
python setup.py install
```

## Code license

See [LICENSE](LICENSE).

## Pretrained weights license

Not specified.

## Citation

```
@inproceedings{dong2023rethinking,
  title={Rethinking Optical Flow from Geometric Matching Consistent Perspective},
  author={Dong, Qiaole and Cao, Chenjie and Fu, Yanwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```