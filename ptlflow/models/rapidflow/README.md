# RAPIDFlow

## Installation

Follow the [PTLFlow installation instructions](https://ptlflow.readthedocs.io/en/latest/starting/installation.html).

IMPORTANT: This model was trained and tested on ptlflow v0.3.2.
It should also probably work on newer versions as well, but it has not been validated.

This model can be called using the following names: `rapidflow`, `rapidflow_it1`, `rapidflow_it2`, `rapidflow_it3`, `rapidflow_it6`.

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
python train.py --config ptlflow/models/rapidflow/configs/rapidflow-train1-chairs.yaml
```

### Stage 2: FlyingThings3D

```bash
python train.py --config ptlflow/models/rapidflow/configs/rapidflow-train2-things.yaml
```

### Stage 3: FlyingThings3D+Sintel+KITTI+HD1K
```bash
python train.py --config ptlflow/models/rapidflow/configs/rapidflow-train3-sintel.yaml
```

### Stage 4: KITTI 2015
```bash
python train.py --config ptlflow/models/rapidflow/configs/rapidflow-train4-kitti.yaml
```

## Validation

To validate our model on the training sets of Sintel and KITTI, use the following command at the root folder of PTLFlow:

```bash
python validate.py --config ptlflow/models/rapidflow/configs/rapidflow-validate-things.yaml
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
python test.py --config ptlflow/models/rapidflow/configs/rapidflow-test-sintel.yaml
```

### KITTI 2015

```bash
python test.py --config ptlflow/models/rapidflow/configs/rapidflow-test-kitti.yaml
```

## Converting model to ONNX

### Installation

You will need to install the following additional packages

```bash
pip install onnx onnxruntime
```

### Usage

The script [convert_to_onnx.py](convert_to_onnx.py) provides a simple example of how to convert RAPIDFlow models to ONNX format.
For example, to convert the 12 iterations version with the checkpoint trained on the Sintel dataset, you can run:
```bash
python convert_to_onnx.py --model rapidflow --ckpt_path sintel
```

We also provide the script [onnx_infer.py](onnx_infer.py) to quickly test the converted ONNX model.
To test the model converted above, just run:
```bash
python onnx_infer.py rapidflow.onnx
```

You can also provide your own images to test by providing an additional argument:
```bash
python onnx_infer.py rapidflow.onnx --image_paths /path/to/first/image /path/to/second/image
```

## Compiling model to TensorRT

### Installation

You will need to install the following additional packages

```bash
pip install torch-tensorrt
```

### Usage

The script [tensorrt_test.py](tensorrt_test.py) provides a simple example of how to compile RAPIDFlow models to TensorRT.
Run it by typing:
```bash
python tensorrt_test.py rapidflow --checkpoint things
```

## ONNX and TensorRT example limitations

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