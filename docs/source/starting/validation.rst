.. _validation:

=========================
Run validation on a model
=========================

You can use the script `validate.py <https://github.com/hmorimitsu/ptlflow/tree/main/validate.py>`__
to validate one of the available models. Read :ref:`initial-scripts` to know how to download it.

Once you have the script, you can run a validation as follows:

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path things

This will use the ``raft_small`` model loaded with the weights trained on the FlyingThings3D dataset.

By default, the validation will be run on the following datasets:

- Sintel Final pass trainval split,

- Sintel Clean pass trainval split,

- KITTI 2015 trainval split.

If you want to validate on different datasets, you can do so by using the argument ``--data.val_dataset``. For example,

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path things --data.val_dataset chairs-val+sintel-clean-val

would use the following two datasets for the validation:

- FlyingChairs val split,

- Sintel Clean val split.

You can check the :ref:`train-val-dataset` page for more details about the ``val_dataset`` string options.
If you want to know about the train/val splits of each dataset, check the validation text files at
`https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/data <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/data>`__.

Using a local checkpoint
========================

If you have a local checkpoint which is not one of the pretrained ones, you can also load it for validation by passing a path to
``--ckpt_path`` argument, as:

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path /path/to/checkpoint

Visualizing the predictions during validation
=============================================

You can use the argument ``--show`` to have the images and predictions displayed on the screen during the validation:

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path things --show

Saving results to disk
======================

The predictions can also be saved to disk. Use ``--write_outputs`` to write the optical flow and other
predictions that the selected model may generate. The structure of the outputs should be similar to the inputs.

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path things --write_outputs

Viewing the validation metrics
==============================

A table with the average metrics computed during the validation will be saved in the directory specified by
``--output_path``. By default, it is saved to ``outputs/validate``.

Speeding up the metric computation
==================================

The metric computations (EPE, Flall, WAUC, ...) consist of many small reductions. On the CPU these are dominated by
thread-pool synchronization, which can slow them down by more than an order of magnitude on many-core CPUs. If a torch
accelerator is available, the metrics are automatically computed on it (a CUDA GPU first, then an Intel XPU), which is
usually much faster. You can also choose the device explicitly with ``--metrics_device``:

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path things --metrics_device cuda

The model itself is not affected by this argument, it keeps running on its own device (a CUDA GPU if available,
otherwise the CPU).

If the model runs on the CPU and validation is unusually slow, the number of CPU threads may also be the culprit. Some
CPUs (e.g. with hybrid performance/efficiency cores) run much faster with fewer threads than the default. You can
control it with ``--num_threads``:

.. code-block:: bash

    python validate.py --model raft_small --ckpt_path things --num_threads 8

Other options
=============

The script offers some more options to control the validation process. You can check them with:

.. code-block:: bash

    python validate.py -h