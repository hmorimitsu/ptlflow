.. _new-model:

==================
Adding a new model
==================

Suppose you create a new model called ``MyModel``. The code below is an example of how to define it to work with PTLFlow.

.. code-block:: python

    from typing import Dict, Optional, Sequence

    import torch
    from ptlflow.models.base_model.base_model import BaseModel

    # 1. Optionally, define a loss function, if you want to be able to train the model.
    from .loss import my_model_loss


    # 2. Inherit from Base Model.
    class MyModel(BaseModel):
        # 3. Optionally, add a class variable called "pretrained_checkpoints" with the pretrained weights.
        pretrained_checkpoints = {
            'ckpt_id1': '/path/to/local/checkpoint/or/https://online/file',
            'ckpt_id2': '/path/to/another/local/checkpoint/or/https://online/file',
        }

        def __init__(
            my_arg1: int = 0,  # an example of an int argument
            my_arg2: Sequence[float] = (0.0, 1.0, 2.0),  # an example of a list of floats
            my_arg3: Optional[int] = None,  # an example of an optional int argument
            **kwargs,  # Add this to receive the other arguments from BaseModel
        ) -> None:
            # 5. Call the parent constructor.
            super(MyModel, self).__init__(
                loss_fn=my_model_loss,  # Can be None, if there is no loss function
                output_stride=64,  # Or another value, depends on the stride of your model
                **kwargs,
            )

            # Define your model Here

        # 7. Define the forward function
        def forward(
            self,
            inputs: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            # inputs may have many keys, but the main one is 'images'.
            # inputs['images'] will be a 5D tensor with shape BNCHW, where
            # B=batch size, N=number of inputs, C=channels (usually 3 for images).
            # The normal usage for optical flow is

            # Preprocess the input images. Check BaseModel.preprocess_images() for more information.
            images, image_resizer = self.preprocess_images(
                inputs["images"],
                bgr_add=-0.5,
                bgr_mult=2.0,
                bgr_to_rgb=True,
                resize_mode="pad",
                pad_mode="replicate",
                pad_two_side=True,
            )

            # Get the pair of images.
            image1 = inputs['images'][:, 0]
            image2 = inputs['images'][:, 1]

            # Define your forward here

            # Postprocess the predictions. It mostly removes the additional padding and rescales the flow prediction accordingly.
            my_5D_flow_predictions = self.postprocess_predictions(my_5D_flow_predictions, image_resizer, is_flow=True)

            # Pack the model estimations into a dict and return.
            # It must have at least an entry 'flows', which is a 5D tensor BNCHW, typically N=1.
            preds = {
                'flows': my_5D_flow_predictions
            }
            return preds

        # 8. BaseModel already define optimizers, dataloaders, training steps, etc.
        # However, if you want to use different ones, you should create methods overriding those steps.
        # Check the PyTorch Lightning documentation for more details about which methods are required:
        # https://lightning.ai/docs/pytorch/stable/starter/introduction.html

To use a model inside PTLFlow, you need to first clone the source code (see :ref:`running-from-source`).
Then do the following steps:

1. Create a folder with your model name inside the ``models`` folder. For example, you could create a folder ``ptlflow/models/my_model``.
   All the files related to this model should be inside this folder, including: definition of the model, loss function, and
   anything else required to run the model

2. Put the code file in the folder, for example in ``ptlflow/models/my_model/my_model.py``.

3. Create the file ``ptlflow/models/my_model/__init__.py`` with the following content:

.. code-block:: python

    # In file: ptlflow/models/my_model/__init__.py
    from .my_model import *

4. Edit the file ``ptlflow/models/__init__.py`` to import your new model:

.. code-block:: python

    # In file: ptlflow/models/__init__.py

    # There should already be other models being imported here
    # Include your import here as well
    from .my_model import *

5. Follow the example below to register your model:

.. code-block:: python

    # In file: ptlflow/models/my_model/my_model.py
    from ptlflow.utils.registry import register_model, trainable, ptlflow_trained

    class MyModel(BaseModel):
    # Your model definition, as described above...

    # Create a lower caps name for your model and register it by decorating it with @register_model
    @register_model
    @trainable  # Optional. Only add if your model can be trained (i.e. offer a loss function and differentiable operations)
    @ptlflow_trained  # Optional. Only add if your model was trained using PTLFlow's training script
    class my_model(MyModel):
        pass

 This should be all. Now your model can be used as any other one inside the platform.

Detailed explanation
====================

Here, the numbered topics in the code above will be explained in more details.

.. _new-model-loss-function:

1. Loss function
----------------

If you want to train you model, you need to define a loss function for it. The loss can either
be a simple function or an ``torch.nn.Module`` (in which case you define the loss calculation in
the ``forward`` method). Assuming you use a simple function, it should have the following signature:

.. code-block:: python

    def my_model_loss(
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # predictions is the output of the forward method of the model.
        # targets is the same inputs dict that is received by the forward method of the model.
        # This function must return a tensor with a single scalar, representing the calculated loss value,
        # OR a dict containing a key 'loss' with the tensor with a single scalar.

2. BaseModel
------------

BaseModel implements the most common requirements for training, validating, and logging optical flow models.
Several parts of PTLFlow assume we are handing a model which follows the specification from BaseModel.
Therefore, it is recommended that your model inherits from BaseModel and keep its outputs consistent with it.
That being said, the common configuration from BaseModel may not serve your model well. In this case,
you should just override the required methods from BaseModel with the setting you need. See :ref:`new-model-methods` for more details.

3. pretrained_checkpoints
-------------------------

PTLFlow looks for this class variable in order to know how to load pretrained weights for the model.
It must be a ``dict``, in which the key is any identifier string and the value is either a path
to a local file, or a link to an online resource. If you do not have pretrained weights for your
model, then simply do not define this variable.

4. BaseModel constructor
------------------------

Your model should provide 3 arguments to BaseModel:

1. The loss function, as explained in :ref:`new-model-loss-function`. This can be ``None``, in which
   case your model **cannot be trained**.

2. The output stride of your model. This represents how many times the smallest feature map can be inside
   your model. Typically this is a power of 2. For example, PWCNet has output stride 64, while RAFT has stride 8.

3. The kwargs arg to receive the additional args defined in BaseModel, unless you prefer to explicitly type all the arguments from BaseModel one-by-one.

5. add_model_specific_args
--------------------------

This function is **not used** after Lightning 2.0 and it is also **dropped in PTLFlow 0.4**.
Provide the model arguments directly to the class constructor instead.

6. forward function
-------------------

The ``forward`` function must follow the input and output types of ``BaseModel``.
In other words, both inputs and outputs must be ``dict`` s identified by string names.
The inputs must accept the following structure:

- A key called 'images' containing a 5D tensor whose shape is BNCHW, where
  B=batch size, N=number of inputs (usually 2 images), C=channels (usually 3, RGB), H=height, W=width.

- Depending on the dataset, there may be additional keys with other images. Please check :ref:`datasets`
  to see which keys can be generated by the datasets.

The outputs must have the following entries:

- A key called 'flows' with optical flow predictions of your model. The prediction must be the same size
  as the input image. This should also be a 5D tensor with a similar shape to the input images. Note that
  typically it will have N=1, for a single flow estimation.

- Optionally, other keys with the same names and shapes as those from the input dataset.

- Any other outputs which are specific to your model. These are ignored by other parts
  of PTLFLow, but it may be used, for example, in your loss function. Remember that the output
  of the forward will be the input of the loss function.

.. _new-model-methods:

7. Overriding methods
---------------------

We follow the PyTorch Lightning `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__
design for our models. Therefore, if you want to modify any of the methods, please check their documentation.
You can also see the API documentation of :ref:`base-model`.