============
Installation
============

You can install it from PyPI:

.. code-block:: bash

    pip install ptlflow

The PyPI package should contain the most recent stable version. If you want to install the newest
(possibly unstable) version, you can do it by:

.. code-block:: bash

    pip install git+https://github.com/hmorimitsu/ptlflow.git

.. _initial-scripts:

Getting initial scripts
=======================

PTLFlow offers some scripts and config files to help you start using the optical flow models.

In order to get just these scripts without having to clone the whole repository, you can
open a terminal and type:

.. code-block:: bash

    python -c "import ptlflow; ptlflow.download_scripts()"

By default, this will download and save the scripts to a folder called ``ptlflow_scripts``.
If you are going to be doing training or validation, then be sure to edit the file
``datasets.yml`` and add the paths to the datasets you want to use in your machine.

To know more details about how to use each script, please read the next pages in this documentation.

Conda environment
=================

It is recommended to use a virtual environment, such as ``conda`` or ``virtualenv``. 
Most of the PTLFlow tests are done in ``conda``. To install PTLFlow in
a new conda environment, run:

.. code-block:: bash

    conda create --name ptlflow-env
    conda activate ptlflow-env
    pip install ptlflow

Optional dependencies
=====================

The dependencies installed from pip are the minimum required to run everything. Nonetheless, there are some
other dependencies which can be installed separately to improve the performance of some models.

Many models can use the ``spatial-correlation-sampler`` package, which is not installed by default.
With this package, the speed and memory requirements of some models should improve.
If you want to install it, you can run:

.. code-block:: bash

    pip install spatial-correlation-sampler

Troubleshooting
===============

In some machines, the ``spatial-correlation-sampler`` package from PyPI cannot be installed.
If you also see errors when trying to install it, then you can try to install the version from GitHub:

.. code-block:: bash

    pip install git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git

.. _running-from-source:

Running from the source code
============================

If you want to modify PTLFlow in some way (to add a new model, change parameters, etc.), you will have
to clone and use the source code instead. You can first clone the source code to your local machine as:

.. code-block:: bash

    git clone https://github.com/hmorimitsu/ptlflow

Then you can just enter the ``ptlflow`` and work directly from there. Another option is to install
PTLFlow to your environment after your modifications. For that, you will have to build the wheel
package and then install with ``pip``.

First, install ``build``, if you do not already have it:

.. code-block:: bash

    pip install build

Then, enter the directory you cloned, build the package and install it:

.. code-block:: bash

    cd ptlflow
    python -m build
    pip install dist/ptlflow-*.whl

Then you should be able to use ``ptlflow`` in the same as if you had installed it from ``pip``.