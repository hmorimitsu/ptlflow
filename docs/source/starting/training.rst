.. _training:

=======================
Train an existing model
=======================

Introduction (important, please read it!)
=========================================

Most models in PTLFlow support training (see :ref:`trainable-models`). The instructions below
will show how to select a model and train it inside PTLFlow.

That being said, at the moment, most models just use the default training routine provided by the ``BaseModel``.
Therefore, there is no guarantee that the models trained in this platform will provide results
close to the original ones. In fact, it is possible that some models will not converge with these
settings.

I would like to, at some point, be able to test and configure the correct training settings
for each model. However, at the moment, I do not have the resources for this. If you have successfully
trained some model inside PTLFlow, please feel free to contribute your results by opening an issue on 
[the GitHub repository](https://github.com/hmorimitsu/ptlflow).

How to train a model
====================

You can use the script `train.py <https://github.com/hmorimitsu/ptlflow/tree/master/train.py>`_
Read :ref:`initial-scripts` to know how to download it.

In order to train a model, you should also keep a copy of
`datasets.yml <https://github.com/hmorimitsu/ptlflow/tree/master/datasets.yml>`_
in the same directory as ``train.py``. Then you should update the paths inside ``datasets.yml``
to point to the dataset root folders in your machine.

Once you have both files, training a model from scratch should be as simple as running:

.. code-block:: bash

    python train.py raft_small --train_dataset chairs-train --gpus 1

This code would train a RAFT Small model on the train split of the FlyingChairs dataset.
The ``--gpus`` option specifies one GPU will be used for training (see the
`Trainer arguments from PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_).
Also check :ref:`train-val-dataset` for more details about the ``train_dataset`` string options.

If you want to have more control over the training hyperparameters, you can add more arguments:

.. code-block:: bash

    python train.py raft_small --train_dataset chairs-train --lr 0.0001 --train_batch_size 4 --max_epochs 5 --gpus 1

You can see all the available options with:

.. code-block:: bash

    python train.py -h

(It will be a long list...).

Logging
=======

By default, `Tensorboard <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#tracking-model-training-with-tensorboard>`_
will be used as the logger. Both the logging results, as well as the intermediate checkpoints
will be saved to the dir specified by ``--log_dir`` (by default it is saved to a folder called ``ptlflow_logs``).
Validation results and checkpoints will be saved at the end of each training epoch.

If you want to see the results on Tensorboard, then open a terminal and type:

.. code-block:: bash

    tensorboard --logdir ptlflow_logs

Then open a web browser and go to ``localhost:6006``. The plots and flow predictions (after at least one validation occurs)
should be displayed in the browser.

Resuming an interrupted training
================================

Assuming at least one epoch has concluded before the interruption, you can resume the training using the saved checkpoints.
To do so, just use the argument ``--resume_from_checkpoint`` giving the path to the ``*_train_*.ckpt`` checkpoint
(see :ref:`saved-checkpoints`).

.. code-block:: bash

    python train.py raft_small --train_dataset chairs-train --gpus 1 --resume_from_checkpoint /path/to/train_checkpoint

.. _finetuning:

Finetuning a previous checkpoint
================================

Optical flow models are often trained at multiple stages, in which the weights from the previous stage is used as an
initialization for the next one. This is different from resuming the training, because in this case we do not want to start
a new training routine, but rather recover the model weights from a previous checkpoint. In this case, you should
still use ``--resume_from_checkpoint`` to point to the checkpoint to be restored. However, you should also include
an additional argument ``--clear_train_state``, which will make sure that only the model weights will be loaded:

.. code-block:: bash

    python train.py raft_small --train_dataset things-train --gpus 1 --resume_from_checkpoint /path/to/train_checkpoint --clear_train_state

Finetuning from pretrained weights
==================================

Many models in PTLFlow offer pretrained weights. These can also be used as the starting point for the finetuning.
For this, just use ``--pretrained_ckpt`` instead of ``--resume_from_checkpoint`` to define the checkpoint to load
(but keep ``--clear_train_state``):

.. code-block:: bash

    python train.py raft_small --train_dataset sintel-train --gpus 1 --pretrained_ckpt things --clear_train_state

.. _saved-checkpoints:

Saved checkpoints
=================

By default, 3 checkpoints will be saved at the end of each epoch:

- A "train" checkpoint, named ``*_train_*.ckpt``, where ``*`` can be any text. This checkpoint is much larger than the others
  because it stores information about all the training environment (model weights, optimizer, learning rates scheduler, etc.).
  This checkpoint can be used to resume a training from exactly where it has stopped.

- A "last" checkpoint, named ``*_last_*.ckpt``. This checkpoint contains only the model weights obtained after the most
  recent epoch concluded. This, or the next checkpoint, is what you should usually make available for others to use your model.
  You can also use this checkpoint for :ref:`finetuning`.

- A "best" checkpoint, named ``*_best_metric-name_value_*.ckpt``. This checkpoint is saved whenever ``metric-name`` is better than
  the previous "best" checkpoint. By default, ``metric-name`` will be the EPE (End-Point-Error) value obtained from the
  first dataset specified in ``--val_dataset`` (if not specified, it will be the trainval split of the Sintel Final dataset by default).
  You can check :ref:`train-val-dataset` for more details about the ``val_dataset`` string options.

.. _trainable-models:

Trainable models
================

You can get a list of the model names that support training, using the function ``ptlflow.get_trainable_model_names()``.
In order to print the list on the terminal, type:

.. code-block:: bash

    python -c "import ptlflow; print(ptlflow.get_trainable_model_names())"

Note, however, that for the moment, the actual training of each model has not been tested. Therefore, although the listed
models will indeed be trained, there is no guarantee that they will learn to generate good predictions. This is because
each model may have particular hyperparameter choices which need to be tuned for them to converge.

I hope that in the future all models can be trained successfully inside PTLFlow. But tuning all their trainings
will require great effort and resources, which unfortunately I do not have at the moment.