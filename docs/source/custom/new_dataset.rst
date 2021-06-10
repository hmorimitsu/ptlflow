====================
Adding a new dataset
====================

The simplest way is to create a class which inherits from ``ptlflow.data.datasets.BaseFlowDataset`` and
then populate its lists according to the structure of your dataset. The code below shows an example:

.. code-block:: python

    from ptlflow.data.datasets import BaseFlowDataset


    class MyDataset(BaseFlowDataset):
        def __init__(
            self,
            my_params_here
        ) -> None
            super().__init__(
                dataset_name='MyDatasetName',
                transform=MyAugmentationTransform,
                get_valid_mask=True,  # To return valid pixels masks, recommended to be True
                get_occlusion_mask=True,  # If the dataset has occlusion masks
                get_motion_boundary_mask=False,  # If the dataset has motion boundary masks
                get_backward=False,  # If the dataset has backward flow, occ, mb masks
                get_meta=True  # To return some metadata, such as paths, etc.
            )

            # Read you dataset paths here (for example, using glob) and populate the following lists
            # self.img_paths
            # self.flow_paths
            #
            # The lists below are optional
            # self.occ_paths
            # self.mb_paths
            # self.flow_b_paths
            # self.occ_b_paths
            # self.mb_b_paths
            # self.metadata

            # For example, suppose one sample has two images, one flow file, and one occlusion mask
            # You should add it to the lists as follows:
            self.img_paths.append(['/path/to/first/image.png', '/path/to/second/image.png'])
            self.flow_paths.append(['/path/to/flow.flo'])
            self.occ_paths.append(['/path/to/occlusion_mask.png'])
            # Notice we always append a list.

            # That is all! BaseFlowDataset handles the actual loading of the data, as long it is correctly defined.

If you want to see more details, check the API definition of the all the dataset at :ref:`datasets`.
This could serve as a guide to implement you own.