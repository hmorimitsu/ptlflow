# MS-RAFT+

## Original code

[https://github.com/cv-stuttgart/MS_RAFT_plus](https://github.com/cv-stuttgart/MS_RAFT_plus)

## Additional requirements

In order to use MS-RAFT+ you need to have CUDA installed and compile the alt_cuda_corr package:

- Download and install CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
  - IMPORTANT! Be sure to choose the same CUDA version as your PyTorch
- Enter the alt_cuda_corr folder and run the setup:
```bash
cd alt_cuda_corr
python setup.py install
```

## Code license

See [LICENSE](LICENSE).

## Pretrained weights license

Not specified.

## Citation

```
@techreport{jahediHighResolutionMultiScaleRAFT2022,
  title = {High-Resolution Multi-Scale {{RAFT}}},
  author = {Jahedi, Azin and Luz, Maximilian and Mehl, Lukas and Rivinius, Marc and Bruhn, Andr{\'e}s},
  year = {2022},
  month = oct,
  eprint = {2210.16900},
  eprinttype = {arxiv},
  primaryclass = {cs},
  pages = {3},
  archiveprefix = {arXiv},
  langid = {english},
  keywords = {Computer Science - Computer Vision and Pattern Recognition},
}
```