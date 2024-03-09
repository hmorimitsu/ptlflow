# RPKNet

## Installation

Follow the [PTLFlow installation instructions](https://ptlflow.readthedocs.io/en/latest/starting/installation.html).

This model can be called using the name `rpknet`.

The exact versions of the packages we used for our tests are listed in [requirements.txt](requirements.txt).

## Data

Our model uses the following datasets. Download and unpack them according to their respective instructions and then configure the paths in `datasets.yml` (see [PTLFlow installation instructions](https://ptlflow.readthedocs.io/en/latest/starting/installation.html)).

### Training datasets

- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [MPI-Sintel](http://sintel.is.tue.mpg.de)
- [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

### Validation/test datasets

- [MPI-Sintel](http://sintel.is.tue.mpg.de)
- [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
- [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [Spring](https://spring-benchmark.org/)

## Training

Follow the [PTLFlow training instructions](https://ptlflow.readthedocs.io/en/latest/starting/training.html).

We train our model in four stages as follows.

### Stage 1: FlyingChairs

```bash
python train.py rpknet --random_seed 1234 --gradient_clip_val 1.0 --lr 2.5e-4 --wdecay 1e-4 --gamma 0.8 --train_dataset chairs --train_batch_size 8 --max_epochs 35 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs --not_cache_pkconv_weights
```

### Stage 2: FlyingThings3D

```bash
python train.py rpknet --pretrained path_to_stage1_ckpt --random_seed 1234 --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-4 --gamma 0.8 --train_dataset things --train_batch_size 4 --max_epochs 40 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs --not_cache_pkconv_weights
```

### Stage 3: FlyingThings3D+Sintel+KITTI+HD1K
```bash
python train.py rpknet --pretrained path_to_stage2_ckpt --random_seed 1234 --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-5 --gamma 0.85 --train_dataset 200*sintel+400*kitti-2015+10*hd1k+things-train-sinteltransform --train_batch_size 6 --max_epochs 4 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs --not_cache_pkconv_weights
```

### Stage 4: KITTI 2015
```bash
python train.py rpknet --pretrained path_to_stage3_ckpt --random_seed 1234 --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-5 --gamma 0.85 --train_dataset kitti-2015 --train_batch_size 6 --max_epochs 150 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs --not_cache_pkconv_weights
```

## Validation

To validate our model on the training sets of Sintel and KITTI, use the following command at the root folder of PTLFlow:

```bash
python validate.py rpknet --iters 12 --pretrained_ckpt things --val_dataset sintel-clean+sintel-final+kitti-2012+kitti-2015
```

It should generate the following results:

| Dataset      | EPE  | Outlier |
|--------------|------|---------|
| Sintel clean | 1.12 | 3.52    |
| Sintel final | 2.45 | 7.08    |
| KITTI 2012   | 1.68 | 6.83    |
| KITTI 2015   | 3.79 | 13.0    |

## Test

The results submitted to the public benchmarks are generated with the respective commands below.

### MPI-Sintel

```bash
python test.py rpknet --iters 32 --pretrained_ckpt sintel --test_dataset sintel --warm_start
```

### KITTI 2015

```bash
python test.py rpknet --iters 32 --pretrained_ckpt kitti --test_dataset kitti-2015 --input_pad_one_side
```

### Spring

```bash
python test.py rpknet --iters 32 --pretrained_ckpt sintel --test_dataset spring --warm_start  --input_bgr_to_rgb
```
*There is no special reason to convert to RGB here. But this mode was used by accident when submitting our results. 

## Code license

The source code is released under the [Apache 2.0 LICENSE](LICENSE).

## Pretrained weights license

Based on the licenses of the datasets used for training the models, our weights are released strictly for academic and research purposes only.

## Citation

If you use this model, please consider citing the paper:

```
@InProceedings{Morimitsu2024RecurrentPartialKernel,
  author    = {Morimitsu, Henrique and Zhu, Xiaobin and Ji, Xiangyang and Yin, Xu-Cheng},
  booktitle = {AAAI},
  title     = {Recurrent Partial Kernel Network for Efficient Optical Flow Estimation},
  year      = {2024},
}
```