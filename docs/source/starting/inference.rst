============================================
Predict optical flow with a pretrained model
============================================

infer.py
========

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/master/infer.py>`__

We provide a script for making it easier to perform inference on different inputs.
Read :ref:`initial-scripts` to know how to download it.
Using this script, you can estimate the optical flow between two images by typing:

.. code-block:: bash

    python infer.py raft_small --pretrained_ckpt things --input_path /path/to/img1.jpg /path/to/img2.jpg --show

Note that you can also give a path to a local checkpoint file to ``--pretrained_ckpt`` as well, like:

.. code-block:: bash

    python infer.py raft_small --pretrained_ckpt /path/to/checkpoint --input_path /path/to/img1.jpg /path/to/img2.jpg --show

You can see all the available options of this script with:

.. code-block:: bash

    python infer.py -h

Writing your own script
=======================

If you prefer to write your own script for the inference, then you should do the following:

1. Get one of the models from PTLFlow.

2. Load the RGB images for estimating the flow (usually two consecutive images).

3. Convert the images to the input format of the model.

4. Forward the input through the model and get the predictions.

The code below shows a way to do this:

.. code-block:: python

    import cv2 as cv
    import ptlflow
    from ptlflow.utils import flow_utils
    from ptlflow.utils.io_adapter import IOAdapter

    # Get an optical flow model. As as example, we will use RAFT Small
    # with the weights pretrained on the FlyingThings3D dataset
    model = ptlflow.get_model('raft_small', pretrained_ckpt='things')

    # Load the images
    images = [
        cv.imread('/path/to/image1.png'),
        cv.imread('/path/to/image2.png')
    ]

    # A helper to manage inputs and outputs of the model
    io_adapter = IOAdapter(model, images[0].shape[:2])

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs(images)

    # Forward the inputs through the model
    predictions = model(inputs)

    # Remove extra padding that may have been added to the inputs
    predictions = io_adapter.unpad(predictions)

    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']

    # flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    print(flows.shape)

    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()
    # OpenCV uses BGR format
    flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)

    # Show on the screen
    cv.imshow('image1', images[0])
    cv.imshow('image2', images[1])
    cv.imshow('flow', flow_bgr_npy)
    cv.waitKey()