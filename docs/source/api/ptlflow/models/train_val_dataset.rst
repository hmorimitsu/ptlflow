.. _train-val-dataset:

=============================
train_dataset and val_dataset
=============================

The datasets used both in training and validation (see :ref:`training` and :ref:`validation`)
can be selected by providing a special composed string to either ``--train_dataset`` or
``--val_dataset``. Here we will provide some more explanations about this string and
show some examples of valid and invalid options.

    Obs: the explanation below concerns the default behavior of :ref:`base-model`.
    It is possible that some optical flow models override the default behavior, in which
    case the input string may be treated differently.

The input string should specify:

- which datasets should be loaded,

- how many times a dataset should be repeated (used to balance the data in case the dataset sizes are too different),

- which split of the dataset should be used,

- and provide additional params required by the dataset to be loaded.

The string may contain one or more datasets. If more than one dataset is specified, then
each dataset is separated by a ``+`` sign. For example ``chairs+sintel`` would load
both the FlyingChairs and the Sintel datasets.

If some dataset needs to be repeated many times to balance the data, then the number of
repetitions must be either the first or the last argument from each dataset. They
must also be connected by a ``*`` sign. For example ``chairs+sintel*10`` would repeat
the Sintel dataset 10 times, while the FlyingChairs dataset would not be repeated.
``chairs*2+15*sintel`` is also valid, as the numbers are in the end and beginning
of their respective datasets.

A split can be informed as well. Typically, the splits will admit of the values in
{train, val, trainval, test}. Not all datasets offer test splits.
If you want to know about the train/val splits of each dataset, check the validation text files at
`https://github.com/hmorimitsu/ptlflow/tree/master/ptlflow/data <https://github.com/hmorimitsu/ptlflow/tree/master/ptlflow/data>`__.
Splits are separated using the ``-`` (dash) symbol. With splits, our example would become ``chairs-train+10*sintel-trainval``.
Note that ``sintel*10-trainval`` would be invalid, as the multiplier cannot be in the middle of the dataset arguments.

Finally, some datasets may use one or more extra arguments. Each argument is also separated
by ``-`` symbols. For example, the Sintel dataset admit a pass name which can be ``clean`` or ``final``
as an additional argument. KITTI also accepts ``2012`` or ``2015`` as arguments. So we could have
``chairs-train+10*sintel-clean-trainval``.

Missing arguments
=================

So what happens when we do not specify some argument? For example, what is the difference between using
``sintel`` or ``sintel-clean-trainval``? It actually may depend on the dataset and the model
being used, but in general, the following rules are applied:

- If no multiplier is given, then it is assumed ``1``.

- If no split is given, then it defaults to ``trainval``.

- If no additional arguments are given, then it defaults to combining all the choices together.
  For example, if the pass name for the Sintel dataset is not given, then both ``clean`` and ``final``
  are put together. Similarly for the KITTI years.

More than one value for an argument
===================================

The default parser does not offer a way to choose more than one value for one argument at a time. For example,
it is not possible to ask for both Sintel clean and final explicitly. If you want to do this,
you can either omit the argument altogether, in which case the combination of all choices will be
returned. Another option is to treat it as a request to two (or more) separated datasets. For example,
``sintel-clean+sintel-final`` would be a way to specify the two pass names.
