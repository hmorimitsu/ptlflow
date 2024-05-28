# RAPIDFlow

## Installation

Follow the [PTLFlow installation instructions](https://ptlflow.readthedocs.io/en/latest/starting/installation.html).

This model can be called using the following names: `rapidflow`, `rapidflow_it1`, `rapidflow_it2`, `rapidflow_it3`, `rapidflow_it6`, `rapidflow_it12`.

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

## Training

Follow the [PTLFlow training instructions](https://ptlflow.readthedocs.io/en/latest/starting/training.html).

We train our model in four stages as follows.

### Stage 1: FlyingChairs

```bash
python train.py rapidflow --random_seed 1234 --gradient_clip_val 1.0 --lr 2.5e-4 --wdecay 1e-4 --gamma 0.8 --train_dataset chairs --train_batch_size 8 --max_epochs 35 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs
```

### Stage 2: FlyingThings3D

```bash
python train.py rapidflow --pretrained path_to_stage1_ckpt --random_seed 1234 --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-4 --gamma 0.8 --train_dataset things --train_batch_size 4 --max_epochs 10 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs
```

### Stage 3: FlyingThings3D+Sintel+KITTI+HD1K
```bash
python train.py rapidflow --pretrained path_to_stage2_ckpt --random_seed 1234 --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-5 --gamma 0.85 --train_dataset 200*sintel+400*kitti-2015+10*hd1k+things-train-sinteltransform --train_batch_size 6 --max_epochs 4 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs
```

### Stage 4: KITTI 2015
```bash
python train.py rapidflow --pretrained path_to_stage3_ckpt --random_seed 1234 --gradient_clip_val 1.0 --lr 1.25e-4 --wdecay 1e-5 --gamma 0.85 --train_dataset kitti-2015 --train_batch_size 6 --max_epochs 150 --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs
```

## Validation

To validate our model on the training sets of Sintel and KITTI, use the following command at the root folder of PTLFlow:

```bash
python validate.py rapidflow_it12 --pretrained_ckpt things --val_dataset sintel-clean+sintel-final+kitti-2012+kitti-2015 --fp16
```

It should generate the following results:

| Dataset      | EPE  | Outlier |
|--------------|------|---------|
| Sintel clean | 1.58 | 4.73    |
| Sintel final | 2.90 | 8.57    |
| KITTI 2012   | 2.33 | 9.57    |
| KITTI 2015   | 5.88 | 17.7    |

## Test

The results submitted to the public benchmarks are generated with the respective commands below.

### MPI-Sintel

```bash
python test.py rapidflow --iters 12 --pretrained_ckpt sintel --test_dataset sintel --warm_start
```

### KITTI 2015

```bash
python test.py rapidflow --iters 12 --pretrained_ckpt kitti --test_dataset kitti-2015 --input_pad_one_side
```

## Converting model to ONNX

The script [convert_to_onnx.py](convert_to_onnx.py) provides a simple example of how to convert RAPIDFlow models to ONNX format.
For example, to convert the 12 iterations version with the checkpoint trained on the Sintel dataset, you can run:
```bash
python convert_to_onnx.py rapidflow_it12 --checkpoint sintel
```

We also provide the script [onnx_infer.py](onnx_infer.py) to quickly test the converted ONNX model.
To test the model converted above, just run:
```bash
python onnx_infer.py rapidflow_it12.onnx
```

You can also provide your own images to test by providing an additional argument:
```bash
python onnx_infer.py rapidflow_it12.onnx --image_paths /path/to/first/image /path/to/second/image
```

## Compiling model to TensorRT

The script [tensorrt_test.py](tensorrt_test.py) provides a simple example of how to compile RAPIDFlow models to TensorRT.
Run it by typing:
```bash
python tensorrt_test.py rapidflow_it12 --checkpoint things
```

### ONNX and TensorRT example limitations

Directly converting the model to ONNX and TensorRT as shown in this example will work, but it is not optimal.
To obtain the best convertion, it would be necessary to rewrite some parts of the code to remove conditions and operations that may change according to the input size.
Also, these convertions only supports `--corr_mode allpairs`, which is not suitable for large images.

## Code license

The source code is released under the [Apache 2.0 LICENSE](LICENSE).

## Pretrained weights license

Based on the licenses of the datasets used for training the models, our weights are released strictly for academic and research purposes only.

## Citation

If you use this model, please consider citing the paper:

```
@InProceedings{Morimitsu2024RAPIDFlow,
  author    = {Morimitsu, Henrique and Zhu, Xiaobin and Cesar-Jr., Roberto M. and Ji, Xiangyang and Yin, Xu-Cheng},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  title     = {RAPIDFlow: Recurrent Adaptable Pyramids with Iterative Decoding for Efficient Optical Flow Estimation},
  year      = {2024},
}
```