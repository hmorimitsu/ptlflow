# SeparableFlow

## Original code

[https://github.com/feihuzhang/SeparableFlow](https://github.com/feihuzhang/SeparableFlow)

## Additional requirements

In order to use SeparableFlow you need to have CUDA installed and compile the GANet package:

- Download and install CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
  - IMPORTANT! Be sure to choose the same CUDA version as your PyTorch
- Compile the package:
```bash
bash compile.sh
```

## Code license

See [LICENSE](LICENSE).

## Pretrained weights license

Not specified.

## Citation

```
@inproceedings{Zhang2021SepFlow,
  title={Separable Flow: Learning Motion Cost Volumes for Optical Flow Estimation},
  author={Zhang, Feihu and Woodford, Oliver J. and Prisacariu, Victor Adrian and Torr, Philip H.S.},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
  pages={10807-10817}
}
```