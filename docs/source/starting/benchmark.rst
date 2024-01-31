==================
Model Benchmarking
==================

You can use the script `model_benchmark.py <https://github.com/hmorimitsu/ptlflow/tree/main/model_benchmark.py>`__
to collect some metrics about the models. Read :ref:`initial-scripts` to know how to download it.

Once you have the script, you can run a benchmark as follows:

.. code-block:: bash

    python model_benchmark.py raft_small

This command will collect some metrics from the ``raft_small`` model.
The results are printed in the terminal and also saved to a CSV file at the folder specified by the argument ``--output_path``.

When benchmarking a single model (as in the example above), it is possible to include model-specific arguments as well.
For example:

.. code-block:: bash

    python model_benchmark.py raft_small --iters 12

``--iters`` is an argument available inside the ``raft_small`` model.

Benchmarking multiple models
============================

You can also run the benchmark on several models at the same time, by providing ``select`` as the first argument and then a list of model names for the ``--selection`` argument.
For example, the command:

.. code-block:: bash

    python model_benchmark.py select --selection raft_small pwcnet

would collect the benchmark results for ``raft_small`` and ``pwcnet`` models.

You can also benchmark all available models with:

.. code-block:: bash

    python model_benchmark.py all

IMPORTANT: when benchmarking multiple models with ``select`` or ``all``, it is not possible to provide model-specific argument directly from the command line!

Reported metrics
================

This script report the following metrics:

- Number of model parameters
- FLOPs
- Running time

FLOPs and running time are relative to the input size and the chosen datatypes.

Useful arguments
================

You can find all the arguments accepted by this script by running:

.. code-block:: bash

    python model_benchmark.py -h

Below we explain some of the most useful arguments you can control:

- ``--num_trials``, ``--num_samples``, ``--sleep_interval``: use these to change the number of tests run to average the metrics. Each trial runs the model ``--num_samples`` times. ``--sleep_interval`` can be used to set a delay between each trial.
- ``--input_size``: the height and width, respectively, of the input to be used for benchmarking.
- ``--final_speed_mode``, ``--final_memory_mode``: how to average the speed and memory metrics.
- ``--datatypes``: a list of datatypes (``fp16`` and/or ``fp32``) to be tested.

The command below shows an example with all the above arguments:

.. code-block:: bash

    python model_benchmark.py raft_small --num_trials 2 --num_samples 5 --sleep_interval 1.0 --input_size 384 1280 --final_speed_mode median --final_memory_mode first --datatypes fp16 fp32


Plotting results
================

You can create 2D scatter plots by choosing two of the available metrics.
You can check the names of valid metrics by checking the accepted values of ``--plot_axes`` after running:

.. code-block:: bash

    python model_benchmark.py -h

For example, the command below creates a scatter plot showing time and flops of three models:

.. code-block:: bash

    python model_benchmark.py select --selection raft_small pwcnet flownets --plot_axes time flops

Known issues
============

Different GPU IDs
-----------------

In machines with more than one GPU, sometimes the GPU ID from ``nvidia-smi`` is different from the ID in PyTorch, causing wrong GPU memory usage reports.
If that happens, you will have to manually change the ``device_id`` variable in ``model_benchmark.py`` to synchronize the two IDs.

Variable running times
----------------------

Calculating the running times of multiple models (using the arguments ``select`` or ``all``) may cause later models to become slower.
If you want to get the lowest running times of each model, it is best to benchmark only one model at a time.