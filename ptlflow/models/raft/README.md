# RAFT

## Original code

[https://github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)

## Code license

See [LICENSE](LICENSE).

## Pretrained weights license

Not specified.

## Citation

```
@inproceedings{teed2020raft,
  title={Raft: Recurrent all-pairs field transforms for optical flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={European Conference on Computer Vision},
  pages={402--419},
  year={2020},
  organization={Springer}
}
```

## Training

This model can be trained on PTLFlow by following the [PTLFlow training instructions](https://ptlflow.readthedocs.io/en/latest/starting/training.html).

### Stage 1: FlyingChairs

```bash
python train.py --config ptlflow/models/raft/configs/raft-train1-chairs.yaml
```

### Stage 2: FlyingThings3D

```bash
python train.py --config ptlflow/models/raft/configs/raft-train2-things.yaml
```

### Stage 3: FlyingThings3D+Sintel+KITTI+HD1K
```bash
python train.py --config ptlflow/models/raft/configs/raft-train3-sintel.yaml
```

### Stage 4: KITTI 2015
```bash
python train.py --config ptlflow/models/raft/configs/raft-train4-kitti.yaml
```