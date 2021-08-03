.. _testing:

========================================
Predicting optical flow on test datasets
========================================

Some datasets, such as MPI-Sintel and KITTI, provide a test split of the dataset, which is to be used
for submitting results for the official rankings. PTLFlow provides a script
`test.py <https://github.com/hmorimitsu/ptlflow/tree/master/test.py>`__ that can be used to generate
predictions for the test split of some benchmarks. Read :ref:`initial-scripts` to know how to download it.

    At the moment, ``test.py`` only supports MPI-Sintel, KITTI 2012, and KITTI 2015 datasets by default.
    More datasets will be added in the future.

Note, however, that ``test.py`` just generate optical flow files following the folder and naming structures
according to each benchmark. Usually, an additional step is necessary before submitting the results to the
respective websites. For example, MPI-Sintel provides a `bundler <http://sintel.is.tue.mpg.de/downloads>`__
to be used to package your results, while KITTI requires a ZIP file.
The outputs of ``test.py`` are already in the correct format to be processed
by these tools, but you still need to run them by yourself.

Generating predictions for the test split
=========================================

Once you have downloaded the script, you can generate the test predictions as follows:

.. code-block:: bash

    python test.py raft_small --pretrained_ckpt things --test_dataset sintel kitti-2012 kitti-2015

This is just an example using the ``raft_small`` model loaded with the weights pretrained on the FlyingThings3D dataset,
but you should use your own model when making a submission. In this example, we are generating predictions
for three datasets: MPI-Sintel (both clean and final splits), KITTI 2012, and KITTI 2015. If you want to use
different datasets, then adapt the ``--test_dataset`` arguments accordingly.

Visualizing the outputs during prediction
=========================================

You can use the argument ``--show`` to have the images and predictions displayed on the screen during the test.
This can be useful as a qualitative sanity check to make sure the predictions are as expected.

.. code-block:: bash

    python test.py raft_small --pretrained_ckpt things --test_dataset sintel kitti-2012 kitti-2015 --show

Outputs
=======

The outputs will be saved in the directory specified by the ``--output_path`` argument.
By default, they are saved to the ``outputs/test`` directory.

Other options
=============

The script offers some more options to control the test process. You can check them with:

.. code-block:: bash

    python test.py -h