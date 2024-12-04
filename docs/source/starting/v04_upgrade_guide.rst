=================================
Changes from PTLFlow v0.3 to v0.4
=================================

PTLFlow v0.4 introduced several modifications that affect the definition and usage of the models and provided scripts.
Most of the changes were made to make PTLFlow follow the new structure and guidelines introduced by Lightning 2.0.
If you are transitioning from PTLFlow v0.3 to v0.4, below are the main changes that you need to be aware of.

Version upgrades
================

Some packages need to be upgraded for PTLFlow v0.4. The most important requirements are:

- python version >= 3.10 (tested until <= 3.12)
- torch version >= 2.2 (tested until <= 2.5)
- lightning version >= 2.1 (tested until <= 2.4)

Use the --model argument
========================

Until v0.3, the scripts accepted the model name without any leading arguments.
Since v0.4, you need to include ``--model`` before the model name.
For example, in v0.3 you would run:

.. code-block:: bash

    python validate.py raft

In v0.4, this becomes:

.. code-block:: bash

    python validate.py --model raft

If you want to see the list of arguments accepted by the model, run:

.. code-block:: bash

    python validate.py --model.help raft

To give an argument to the model, you also need to add the ``--model.`` prefix, for example:

.. code-block:: bash

    python validate.py --model raft --model.iters 12

Use the --data argument
========================

Until v0.3, the datasets were defined using the arguments ``--{train,val,test}_dataset``.
Since v0.4, these arguments now include the prefix ``--data.``, becoming ``--data.{train,val,test}_dataset``
For example, in v0.3 you would run:

.. code-block:: bash

    python validate.py raft --val_dataset kitti-2015

In v0.4, this becomes:

.. code-block:: bash

    python validate.py --model raft --data.val_dataset kitti-2015

The --ckpt_path argument
========================

v0.3 used two possible arguments to load checkpoints. ``--pretrained_ckpt`` and ``--resume_from_checkpoint``.
Since v0.4, both of them should be replaced by ``--ckpt_path``:

.. code-block:: bash

    python train.py --model raft --ckpt_path things

For training, PTLFlow will automatically choose between continuing the previous training or restoring only the model depending on whether the given checkpoint contains the training state (optimizers, lr_scheduler, etc.) or not.

Model name changes
==================

The names of some models have changed.
Use the command below to see the list of available models.

.. code-block:: bash

    python -c "import ptlflow; print(ptlflow.get_model_names())"

Creating new models
===================

The way of defining and registering new models has changed significantly in v0.4.
Please see the example in :ref:`new-model` or check the code of some existing models to learn more.

Change of ckpt cache dir
========================

To comply with the standard checkpoint functions from Lightning, the directory where downloaded ckpp files are stored have changed from
``${TORCH_HUB_CACHE_DIR}/ptlflow/checkpoints/`` to ``${TORCH_HUB_CACHE_DIR}/checkpoints/``.
Therefore, if you have downloaded multiple ckpt files using PTLFlow v0.3 or earlier, you should move them to the new folder to avoid duplicates.
In Linux, the default ``${TORCH_HUB_CACHE_DIR}`` is ``/home/${USER}/.cache/torch/hub/``.

Config files
============

All the main PTLFlow's scripts support the use of YAML config files to save and restore previous configurations more easily.
Read :ref:`using-config-files` for more information.
