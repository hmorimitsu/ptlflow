.. PTLFlow documentation master file, created by
   sphinx-quickstart on Wed May 26 12:43:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================
PTLFlow - PyTorch Lightning Optical Flow
========================================

Welcome to the PTLFlow documentation for the code at `https://github.com/hmorimitsu/ptlflow <https://github.com/hmorimitsu/ptlflow>`_. 
This is an unified platform built on PyTorch Lightning for
training and testing deep optical flow models. The modular design of systems in PyTorch Lightning
is ideal for putting lots of models together while keeping each of them well contained
(see `System vs Model <https://pytorch-lightning.readthedocs.io/en/stable/starter/new-project.html>`_).

PTLFlow is still in early development, so there are only a few models available at the moment,
but hopefully the list of models will grow soon.

Here you will find some basic steps on how to use PTLFlow for training and testing some optical flow models.
This documentation also contains validation results of each model in some common benchmarks.

This documentation is still under development, but hopefully the most important parts for
using PTLFlow are covered here.

Disclaimers
===========

PTLFlow contains the adapted code of many models developed by other authors. References to the papers and
original codes are included in each respective model (but if you see something missing, please inform me on
`GitHub <https://github.com/hmorimitsu/ptlflow>`_).

Most of the original codes or datasets require that any derivative work (such as this) is used strictly for research purposes.
If you want to use it for a different purpose, then you should check by yourself if and how that can be done.

The results presented by the models in this platform are not guaranteed to match the official results
(in either accuracy or speed). If you need to reproduce the official results for some model, then you should
use the its original code.

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   starting/installation
   starting/inference
   starting/validation
   starting/training
   starting/testing

.. toctree::
   :maxdepth: 1
   :caption: Models:

   models/models_list
   models/checkpoint_list

.. toctree::
   :maxdepth: 1
   :caption: Datasets:

   datasets/datasets_list

.. toctree::
   :maxdepth: 1
   :caption: Results:

   results/accuracy_epe
   results/accuracy_epe_outlier
   results/accuracy_plot
   results/speed
   results/speed_plot

.. toctree::
   :maxdepth: 1
   :caption: Customizing:

   custom/new_model
   custom/new_dataset

.. toctree::
   :maxdepth: 1
   :caption: API:

   api/ptlflow/models/train_val_dataset
   api/scripts
   api/ptlflow.models
   api/ptlflow.data
   api/ptlflow.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`