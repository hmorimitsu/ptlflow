# PyTorch Lightning Optical Flow

![GitHub CI flake8 status](https://github.com/hmorimitsu/ptlflow/actions/workflows/flake8.yml/badge.svg)
![GitHub CI python status](https://github.com/hmorimitsu/ptlflow/actions/workflows/python.yml/badge.svg)
![GitHub CI pytorch status](https://github.com/hmorimitsu/ptlflow/actions/workflows/pytorch.yml/badge.svg)
![GitHub CI pytorch-lightning status](https://github.com/hmorimitsu/ptlflow/actions/workflows/pytorch-lightning.yml/badge.svg)
![GitHub CI pytest pip status](https://github.com/hmorimitsu/ptlflow/actions/workflows/pytest_pip.yml/badge.svg)
[![DOI](https://zenodo.org/badge/375416785.svg)](https://zenodo.org/badge/latestdoi/375416785)

## Introduction

This is a collection of state-of-the-art deep model for estimating optical flow. The main goal is to provide a unified framework where multiple models can be trained and tested more easily.

The work and code from many others are present here. I tried to make sure everything is properly referenced, but please let me know if I missed something.

This is still under development, so some things may not work as intended. I plan to add more models in the future, as well keep improving the platform.

- [What's new](#whats-new)
- [Available models](#available-models)
- [Results](#results)
- [Getting started](#getting-started)
- [Licenses](#licenses)
- [Contributing](#contributing)
- [Citing](#citing)
- [Acknowledgements](#acknowledgements)

## What's new

### July 30, 2022 - v0.2.6

- Added new models:
  - CRAFT [https://arxiv.org/abs/2203.16896](https://arxiv.org/abs/2203.16896)
  - CSFlow [https://arxiv.org/abs/2202.00909](https://arxiv.org/abs/2202.00909)
  - FlowFormer [https://arxiv.org/abs/2203.16194](https://arxiv.org/abs/2203.16194)
  - GMFlow [https://arxiv.org/abs/2111.13680](https://arxiv.org/abs/2111.13680)
  - GMFlowNet [https://arxiv.org/abs/2203.11335](https://arxiv.org/abs/2203.11335)
- Added support for AutoFlow dataset [https://arxiv.org/abs/2104.14544](https://arxiv.org/abs/2104.14544)
- Fix Compatibility with Pytorch Lightning 1.6

## Available models

- CRAFT [https://arxiv.org/abs/2203.16896](https://arxiv.org/abs/2203.16896)
- CSFlow [https://arxiv.org/abs/2202.00909](https://arxiv.org/abs/2202.00909)
- DICL-Flow [https://arxiv.org/abs/2010.14851](https://arxiv.org/abs/2010.14851)
- FastFlowNet [https://arxiv.org/abs/2103.04524](https://arxiv.org/abs/2103.04524)
- FlowFormer [https://arxiv.org/abs/2203.16194](https://arxiv.org/abs/2203.16194)
- FlowNet - [https://arxiv.org/abs/1504.06852](https://arxiv.org/abs/1504.06852)
- FlowNet2 - [https://arxiv.org/abs/1612.01925](https://arxiv.org/abs/1612.01925)
- GMA - [https://arxiv.org/abs/2104.02409](https://arxiv.org/abs/2104.02409)
- GMFlow [https://arxiv.org/abs/2111.13680](https://arxiv.org/abs/2111.13680)
- GMFlowNet [https://arxiv.org/abs/2203.11335](https://arxiv.org/abs/2203.11335)
- HD3 - [https://arxiv.org/abs/1812.06264](https://arxiv.org/abs/1812.06264)
- IRR - [https://arxiv.org/abs/1904.05290](https://arxiv.org/abs/1904.05290)
- LCV - [https://arxiv.org/abs/2007.11431](https://arxiv.org/abs/2007.11431)
- LiteFlowNet [https://arxiv.org/abs/1805.07036](https://arxiv.org/abs/1805.07036)
- LiteFlowNet2 [https://arxiv.org/abs/1903.07414](https://arxiv.org/abs/1903.07414)
- LiteFlowNet3 [https://arxiv.org/abs/2007.09319](https://arxiv.org/abs/2007.09319)
- MaskFlownet [https://arxiv.org/abs/2003.10955](https://arxiv.org/abs/2003.10955)
- PWCNet - [https://arxiv.org/abs/1709.02371](https://arxiv.org/abs/1709.02371)
- RAFT - [https://arxiv.org/abs/2003.12039](https://arxiv.org/abs/2003.12039)
- ScopeFlow - [https://arxiv.org/abs/2002.10770](https://arxiv.org/abs/2002.10770)
- SCV - [https://arxiv.org/abs/2104.02166](https://arxiv.org/abs/2104.02166)
- STaRFlow -  [https://arxiv.org/abs/2007.05481](https://arxiv.org/abs/2007.05481)
- VCN - [https://papers.nips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf](https://papers.nips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf)

Read more details about the models on [https://ptlflow.readthedocs.io/en/latest/models/models_list.html](https://ptlflow.readthedocs.io/en/latest/models/models_list.html).

# Results

You can see a table with main evaluation results of the available models [here](https://ptlflow.readthedocs.io/en/latest/results/accuracy_epe.html). More results are also available in the folder [docs/source/results](docs/source/results).

**Disclaimer**: These results are the ones obtained by evaluating the available models in this framework in my machine. Your results may be different due to differences in hardware and software. I also do not guarantee that the results of each model will be similar to the ones presented in the respective papers or other original sources. If you need to replicate the original results from a paper, you should use the original implementations.

## Getting started

Please take a look at the [documentation](https://ptlflow.readthedocs.io/) to learn how to install and use PTLFlow.

You can also check the notebooks below running on Google Colab for some practical examples:

- [Inference with a pretrained model](https://colab.research.google.com/drive/1YARBRUGplqTRnRuY9sKIs6LY_2kWAWZJ?usp=sharing).
- [Training and using the learned weights for inference](https://colab.research.google.com/drive/1mbuAEF728_jZpFEsQHXDxjIGAcB1-nVs?usp=sharing).

## Licenses

The original code of this repository is licensed under the [Apache 2.0 license](LICENSE).

Each model may be subjected to different licenses. The license of each model is included in their respective folders. It is your responsibility to make sure that your project is in compliance with all the licenses and conditions involved.

The external pretrained weights all have different licenses, which are listed in their respective folders.

The pretrained weights that were trained within this project are available under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/), which I believe that covers the licenses of the datasets used in the training. That being said, I am not a legal expert so if you plan to use them to any purpose other than research, you should check all the involved licenses by yourself. Additionally, the datasets used for the training usually require the user to cite the original papers, so be sure to include their respective references in your work.

## Contributing

Contribution are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) to see how to contribute.

## Citing

### BibTeX

```
@misc{morimitsu2021ptlflow,
  author = {Henrique Morimitsu},
  title = {PyTorch Lightning Optical Flow},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hmorimitsu/ptlflow}}
}
```

## Acknowledgements

- This README file is heavily inspired by the one from the [timm](https://github.com/rwightman/pytorch-image-models) repository.
- Some parts of the code were inspired by or taken from [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch).
- [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) was also another important source.
- The current main training routine is based on [RAFT](https://github.com/princeton-vl/RAFT).