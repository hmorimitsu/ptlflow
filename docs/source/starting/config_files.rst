.. _using-config-files:

==================
Using config files
==================

One of the nicest additions in Lightning 2.0 is the use of YAML files to save and load configurations more easily.
You can create a config file by using the argument ``--print_config`` and then redirecting the output to a YAML file.
For example, you can save the config of a training:

.. code-block:: bash

    python train.py --model raft --data.train_dataset chairs2 --data.val_dataset kitti-2015-val --print_config > raft_train_config.yaml

This also provides an easier way to know which arguments are available and to tune them according to your needs.
This config can then be used to run the training by loading it with:

.. code-block:: bash

    python train.py --config raft_train_config.yaml

All the basic scripts (train, test, validate, infer, model_benchmark) should be able to save and load config files in this way.

If you want to learn more about config files, check the `LightningCLI documentation <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`__, especially the content about YAML files.