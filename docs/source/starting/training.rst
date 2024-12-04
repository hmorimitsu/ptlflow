.. _training:

=======================
Train an existing model
=======================

Introduction (important, please read it!)
=========================================

Most models in PTLFlow support training (see :ref:`trainable-models`). The instructions below
will show how to select a model and train it inside PTLFlow.

That being said, at the moment, only a few models have been trained on PTLFlow and can use the provided training script to reproduce its results (see :ref:`ptlflow-trained-models`).
Many other models can still be trained with our training script as well, but **there is no guarantee** that the models trained in this platform will provide results close to the original ones.
In fact, it is possible that some models will not converge with these settings.

I would like to, at some point, be able to test and configure the correct training settings
for each model. However, at the moment, I do not have the resources for this. If you have successfully
trained some model inside PTLFlow, please feel free to contribute your results by opening an issue on 
[the GitHub repository](https://github.com/hmorimitsu/ptlflow).

How to train a model
====================

You can use the script `train.py <https://github.com/hmorimitsu/ptlflow/tree/main/train.py>`_ to train a model.
Read :ref:`initial-scripts` to know how to download it.

In order to train a model, you should also keep a copy of
`datasets.yml <https://github.com/hmorimitsu/ptlflow/tree/main/datasets.yml>`_
in the same directory as ``train.py``. Then you should update the paths inside ``datasets.yml``
to point to the dataset root folders in your machine.

Once you have both files, training a model from scratch should be as simple as running:

.. code-block:: bash

    python train.py --model raft_small --data.train_dataset chairs-train --data.val_dataset chairs-val

This code would train a RAFT Small model on the train split of the FlyingChairs dataset.
Also check :ref:`train-val-dataset` for more details about the ``data.X_dataset`` string options.

If you want to have more control over the training hyperparameters, you can add more arguments:

.. code-block:: bash

    python train.py --model raft_small --data.train_dataset chairs-train --data.val_dataset chairs-val --model.lr 0.0001 --data.train_batch_size 4 --trainer.max_epochs 5

You can see all the available options with:

.. code-block:: bash

    python train.py -h

However, it may be cumbersome to find the correct arguments to set their values in this way.
A much better option is to configure the training with config files (see :ref:`using-config-files`).

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
To do so, just use the argument ``--ckpt_path`` giving the path to the ``*_train_*.ckpt`` checkpoint
(see :ref:`saved-checkpoints`).

.. code-block:: bash

    python train.py --model raft_small --data.train_dataset chairs-train --data.val_dataset chairs-val --ckpt_path /path/to/train_checkpoint

.. _finetuning:

Finetuning a previous checkpoint
================================

Optical flow models are often trained at multiple stages, in which the weights from the previous stage is used as an
initialization for the next one. This is different from resuming the training, because in this case we do not want to start
a new training routine, but rather recover the model weights from a previous checkpoint. In this case, you should
still use ``--ckpt_path`` to point to the checkpoint to be restored. However, in this case, you should point to a checkpoint
that do not contain the training state. During training, PTLFlow automatically saves multiple versions of the checkpoint (see :ref:`saved-checkpoints`),
one containing the training state and another without it. To finetune a model, use a checkpoint that is not identified by ``*_train_*.ckpt``

.. code-block:: bash

    python train.py --model raft_small --data.train_dataset things-train --data.val_dataset things-val --ckpt_path /path/to/checkpoint_without_train_state

.. _saved-checkpoints:

Saved checkpoints
=================

By default, 2 checkpoints will be saved at the end of each epoch:

- A "train" checkpoint, named ``*_train_*.ckpt``, where ``*`` can be any text. This checkpoint is much larger than the others
  because it stores information about all the training environment (model weights, optimizer, learning rates scheduler, etc.).
  This checkpoint can be used to resume a training from exactly where it has stopped.

- A "last" checkpoint, named ``*_last_*.ckpt``. This checkpoint contains only the model weights obtained after the most
  recent epoch concluded. This, or the next checkpoint, is what you should usually make available for others to use your model.
  You can also use this checkpoint for :ref:`finetuning`.

.. _trainable-models:

Trainable models
================

You can get a list of the model names that support training, using the function ``ptlflow.get_trainable_model_names()``.
In order to print the list on the terminal, type:

.. code-block:: bash

    python -c "import ptlflow; print(ptlflow.get_trainable_model_names())"

Note, however, that for the moment, the actual training of most model has not been tested. Therefore, although the listed
models will indeed be trained, there is no guarantee that they will learn to generate good predictions. This is because
each model may have particular hyperparameter choices which need to be tuned for them to converge.

.. _ptlflow-trained-models:

PTLFlow trained models
======================

There are, however, a few models that have been trained in PTLFlow.
You can get a list of the model names that have been trained using PTLFlow's own training script using the function  ``ptlflow.get_ptlflow_trained_model_names()``.
In order to print the list on the terminal, type:

.. code-block:: bash

    python -c "import ptlflow; print(ptlflow.get_ptlflow_trained_model_names())"

The models in this list should produce similar results when trained using PTLFlow's training script.