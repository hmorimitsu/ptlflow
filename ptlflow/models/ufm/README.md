# UFM

## Original code

[https://github.com/UniFlowMatch/UFM](https://github.com/UniFlowMatch/UFM)

The original model is assembled from components of the
[UniCeption](https://github.com/castacks/UniCeption) library (pinned at commit
`ee7fa0b50d8295b3f0485e42c1d133fb534bfce7`) and a DINOv2 encoder. The required parts of both
libraries are in this folder, so no external installation is needed. Contrary to the
original code, the DINOv2 architecture is created locally instead of being loaded with torch.hub,
since the UFM checkpoints already contain the DINOv2 weights.

## Code license

See [LICENSE](LICENSE). The UniCeption code is licensed under BSD-3-Clause, and the DINOv2 code under Apache License 2.0.

## Pretrained weights license

As stated by the original repository, the pretrained checkpoints inherit the licenses of the
training datasets and may not be used for commercial purposes (CC BY-NC-SA 4.0).

## Citation

```
@inproceedings{zhang2025ufm,
  title     = {UFM: A Simple Path towards Unified Dense Correspondence with Flow},
  author    = {Zhang, Yuchen and Keetha, Nikhil and Lyu, Chenwei and Jhamb, Bhuvan and Chen, Yutian and Qiu, Yuheng and Karhade, Jay and Jha, Shreyas and Hu, Yaoyu and Ramanan, Deva and Scherer, Sebastian and Wang, Wenshan},
  booktitle = {NeurIPS},
  year      = {2025}
}
```
