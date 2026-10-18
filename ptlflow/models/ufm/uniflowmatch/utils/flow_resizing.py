"""
Flow and image resizing utilities.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class ImagePairsManipulationBase:
    def __init__(self):
        pass

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: torch.Tensor,
        img1_region_source: torch.Tensor,
        img0_region_representation: torch.Tensor,
        img1_region_representation: torch.Tensor,
    ):
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """
        raise NotImplementedError

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.

        Args:
            - H: Height of the first image.
            - W: Width of the first image.

        Returns:
            Tuple of (H1, W1, H2, W2) representing the output shape of the images if the manipulation is applied.
        """

        raise NotImplementedError

    def output_shape_pairs(self, H1: int, W1: int, H2: int, W2: int) -> Tuple[int, int, int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        output1 = self.output_shape(H1, W1)
        output2 = self.output_shape(H2, W2)

        return output1[0], output1[1], output2[0], output2[1]

    def check_input(self, H: int, W: int) -> bool:
        """
        Check whether the input shapes are correct for the current manipulation.

        Args:
            - H: Height of the first image.
            - W: Width of the first image.

        Returns:
            Whether the manipualtion can run on the given input shapes.
        """
        raise NotImplementedError

    def check_input_pairs(self, H1: int, W1: int, H2: int, W2: int) -> bool:
        return self.check_input(H1, W1) and self.check_input(H2, W2)


class ResizeHorizontalAxisManipulation(ImagePairsManipulationBase):
    def __init__(self, horizontal_axis: int):
        self.horizontal_axis = horizontal_axis

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """
        resize_ratio = self.horizontal_axis / W

        return (int(H * resize_ratio), self.horizontal_axis)

    def check_input(self, H: int, W: int) -> bool:
        return True

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: torch.Tensor,
        img1_region_source: torch.Tensor,
        img0_region_representation: torch.Tensor,
        img1_region_representation: torch.Tensor,
    ):
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype float32 or uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype float32 or uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """
        # assert img0.shape == img1.shape, "Image shapes must match"

        _, h0, w0, _ = img0.shape
        _, h1, w1, _ = img1.shape

        target_h0, target_w0, target_h1, target_w1 = self.output_shape_pairs(h0, w0, h1, w1)

        assert img0.dtype == img1.dtype, "Image types must match"
        is_uint8 = img0.dtype == torch.uint8

        img0_resized = F.interpolate(
            img0.permute(0, 3, 1, 2).float(), size=(target_h0, target_w0), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        img1_resized = F.interpolate(
            img1.permute(0, 3, 1, 2).float(), size=(target_h1, target_w1), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

        if is_uint8:
            img0_resized = img0_resized.to(torch.uint8)
            img1_resized = img1_resized.to(torch.uint8)

        h_mult0 = target_h0 / h0
        w_mult0 = target_w0 / w0

        multplier0 = torch.tensor([h_mult0, h_mult0, w_mult0, w_mult0])

        h_mult1 = target_h1 / h1
        w_mult1 = target_w1 / w1

        multplier1 = torch.tensor([h_mult1, h_mult1, w_mult1, w_mult1])

        # source region is unchanged
        # target region is scaled
        img0_region_representation = multplier0 * img0_region_representation
        img1_region_representation = multplier1 * img1_region_representation

        return (
            img0_resized,
            img1_resized,
            img0_region_source,
            img1_region_source,
            img0_region_representation,
            img1_region_representation,
        )


class ResizeVerticalAxisManipulation(ImagePairsManipulationBase):
    def __init__(self, vertical_axis: int):
        self.vertical_axis = vertical_axis

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        resize_ratio = self.vertical_axis / H

        return (self.vertical_axis, int(W * resize_ratio))

    def check_input(self, H: int, W: int) -> bool:
        return True

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: torch.Tensor,
        img1_region_source: torch.Tensor,
        img0_region_representation: torch.Tensor,
        img1_region_representation: torch.Tensor,
    ):
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype float32 or uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype float32 or uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """
        # assert img0.shape == img1.shape, "Image shapes must match"

        _, h0, w0, _ = img0.shape
        _, h1, w1, _ = img1.shape

        target_h0, target_w0, target_h1, target_w1 = self.output_shape_pairs(h0, w0, h1, w1)

        assert img0.dtype == img1.dtype, "Image types must match"
        is_uint8 = img0.dtype == torch.uint8

        img0_resized = F.interpolate(
            img0.permute(0, 3, 1, 2).float(), size=(target_h0, target_w0), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        img1_resized = F.interpolate(
            img1.permute(0, 3, 1, 2).float(), size=(target_h1, target_w1), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

        if is_uint8:
            img0_resized = img0_resized.to(torch.uint8)
            img1_resized = img1_resized.to(torch.uint8)

        h_mult0 = target_h0 / h0
        w_mult0 = target_w0 / w0

        multplier0 = torch.tensor([h_mult0, h_mult0, w_mult0, w_mult0])

        h_mult1 = target_h1 / h1
        w_mult1 = target_w1 / w1

        multplier1 = torch.tensor([h_mult1, h_mult1, w_mult1, w_mult1])

        # source region is unchanged
        # target region is scaled
        img0_region_representation = multplier0 * img0_region_representation
        img1_region_representation = multplier1 * img1_region_representation

        return (
            img0_resized,
            img1_resized,
            img0_region_source,
            img1_region_source,
            img0_region_representation,
            img1_region_representation,
        )


class ResizeToFixedManipulation(ImagePairsManipulationBase):
    def __init__(self, target_shape: Tuple[int, int]):
        self.target_shape = target_shape

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        return self.target_shape

    def check_input(self, H: int, W: int) -> bool:
        return True

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: torch.Tensor,
        img1_region_source: torch.Tensor,
        img0_region_representation: torch.Tensor,
        img1_region_representation: torch.Tensor,
    ):
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype float32 or uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype float32 or uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """
        # assert img0.shape == img1.shape, "Image shapes must match"

        _, h0, w0, _ = img0.shape
        _, h1, w1, _ = img1.shape

        target_h0, target_w0, target_h1, target_w1 = self.output_shape_pairs(h0, w0, h1, w1)

        assert img0.dtype == img1.dtype, "Image types must match"
        is_uint8 = img0.dtype == torch.uint8

        img0_resized = F.interpolate(
            img0.permute(0, 3, 1, 2).float(),
            size=(target_h0, target_w0),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)
        img1_resized = F.interpolate(
            img1.permute(0, 3, 1, 2).float(),
            size=(target_h1, target_w1),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)

        if is_uint8:
            img0_resized = img0_resized.to(torch.uint8)
            img1_resized = img1_resized.to(torch.uint8)

        h_mult0 = target_h0 / h0
        w_mult0 = target_w0 / w0

        multplier0 = torch.tensor([h_mult0, h_mult0, w_mult0, w_mult0])

        h_mult1 = target_h1 / h1
        w_mult1 = target_w1 / w1

        multplier1 = torch.tensor([h_mult1, h_mult1, w_mult1, w_mult1])

        # source region is unchanged
        # target region is scaled
        img0_region_representation = (multplier0 * img0_region_representation).to(torch.int64)
        img1_region_representation = (multplier1 * img1_region_representation).to(torch.int64)

        return (
            img0_resized,
            img1_resized,
            img0_region_source,
            img1_region_source,
            img0_region_representation,
            img1_region_representation,
        )


def scale_axis(
    source_low: float,
    source_high: float,
    reference_low: float,
    reference_high: float,
    reference_low_new: float,
    reference_high_new: float,
):
    reference_length = reference_high - reference_low
    coordinate_relative_low = (reference_low_new - reference_low) / reference_length
    coordinate_relative_high = (reference_high_new - reference_low) / reference_length

    source_length = source_high - source_low
    source_low_new = source_low + coordinate_relative_low * source_length
    source_high_new = source_low + coordinate_relative_high * source_length

    return source_low_new, source_high_new


class CenterCropManipulation(ImagePairsManipulationBase):
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        return self.target_size

    def check_input(self, H: int, W: int) -> bool:
        return H >= self.target_size[0] and W >= self.target_size[1]

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: torch.Tensor,
        img1_region_source: torch.Tensor,
        img0_region_representation: torch.Tensor,
        img1_region_representation: torch.Tensor,
    ):
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """

        B0, H0, W0, C0 = img0.shape
        B1, H1, W1, C1 = img1.shape

        target_h, target_w = self.target_size

        assert H0 >= target_h and W0 >= target_w, "Image shapes must be larger than the target size."
        assert H1 >= target_h and W1 >= target_w, "Image shapes must be larger than the target size."

        crop_top_0 = (H0 - target_h) // 2
        crop_bottom_0 = H0 - target_h - crop_top_0
        crop_left_0 = (W0 - target_w) // 2
        crop_right_0 = W0 - target_w - crop_left_0

        crop_top_1 = (H1 - target_h) // 2
        crop_bottom_1 = H1 - target_h - crop_top_1
        crop_left_1 = (W1 - target_w) // 2
        crop_right_1 = W1 - target_w - crop_left_1

        # apply the crops
        img0_cropped = img0[:, crop_top_0 : H0 - crop_bottom_0, crop_left_0 : W0 - crop_right_0, :]
        img1_cropped = img1[:, crop_top_1 : H1 - crop_bottom_1, crop_left_1 : W1 - crop_right_1, :]

        # update the representation region accurately. This is complex as we may or may not crop out the valid regions.
        remaining_region_0 = torch.tensor(
            [
                max(img0_region_representation[0], crop_top_0),
                min(img0_region_representation[1], H0 - crop_bottom_0),
                max(img0_region_representation[2], crop_left_0),
                min(img0_region_representation[3], W0 - crop_right_0),
            ]
        )

        remaining_region_1 = torch.tensor(
            [
                max(img1_region_representation[0], crop_top_1),
                min(img1_region_representation[1], H1 - crop_bottom_1),
                max(img1_region_representation[2], crop_left_1),
                min(img1_region_representation[3], W1 - crop_right_1),
            ]
        )

        # shift the remaining region as the cropped region is removed
        img0_region_representation_new = remaining_region_0 - torch.tensor(
            [crop_top_0, crop_top_0, crop_left_0, crop_left_0]
        )
        img1_region_representation_new = remaining_region_1 - torch.tensor(
            [crop_top_1, crop_top_1, crop_left_1, crop_left_1]
        )

        img0_region_representation_new = img0_region_representation_new.to(torch.int64)
        img1_region_representation_new = img1_region_representation_new.to(torch.int64)

        # the valid region may or may not be cropped out, so we need to adjust the source region as well
        img0_region_source[0], img0_region_source[1] = scale_axis(
            img0_region_source[0],
            img0_region_source[1],
            img0_region_representation[0],
            img0_region_representation[1],
            remaining_region_0[0],
            remaining_region_0[1],
        )

        img0_region_source[2], img0_region_source[3] = scale_axis(
            img0_region_source[2],
            img0_region_source[3],
            img0_region_representation[2],
            img0_region_representation[3],
            remaining_region_0[2],
            remaining_region_0[3],
        )

        img1_region_source[0], img1_region_source[1] = scale_axis(
            img1_region_source[0],
            img1_region_source[1],
            img1_region_representation[0],
            img1_region_representation[1],
            remaining_region_1[0],
            remaining_region_1[1],
        )

        img1_region_source[2], img1_region_source[3] = scale_axis(
            img1_region_source[2],
            img1_region_source[3],
            img1_region_representation[2],
            img1_region_representation[3],
            remaining_region_1[2],
            remaining_region_1[3],
        )

        return (
            img0_cropped,
            img1_cropped,
            img0_region_source,
            img1_region_source,
            img0_region_representation_new,
            img1_region_representation_new,
        )


class ImagePairsManipulationComposite(ImagePairsManipulationBase):
    def __init__(self, *manipulations: List[ImagePairsManipulationBase]):
        self.manipulations = manipulations

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        output_shape = (H, W)
        for manipulation in self.manipulations:
            output_shape = manipulation.output_shape(*output_shape)

        return output_shape

    def output_shape_pairs(self, H1: int, W1: int, H2: int, W2: int) -> Tuple[int, int, int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        output_shape = (H1, W1, H2, W2)
        for manipulation in self.manipulations:
            output_shape = manipulation.output_shape_pairs(*output_shape)

        return output_shape

    def check_input(self, H: int, W: int) -> bool:
        current_shape = (H, W)
        for manipulation in self.manipulations:
            if not manipulation.check_input(*current_shape):
                return False

            current_shape = manipulation.output_shape(*current_shape)

        return True

    def check_input_pairs(self, H1: int, W1: int, H2: int, W2: int) -> bool:
        current_shape = (H1, W1, H2, W2)
        for manipulation in self.manipulations:
            if not manipulation.check_input_pairs(*current_shape):
                return False

            current_shape = manipulation.output_shape_pairs(*current_shape)

        return True

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: torch.Tensor,
        img1_region_source: torch.Tensor,
        img0_region_representation: torch.Tensor,
        img1_region_representation: torch.Tensor,
    ):  # -> tuple[Tensor | Any, Tensor | Any, Tensor | Any, Tensor | ...:
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """

        for manipulation in self.manipulations:
            (
                img0,
                img1,
                img0_region_source,
                img1_region_source,
                img0_region_representation,
                img1_region_representation,
            ) = manipulation(
                img0,
                img1,
                img0_region_source,
                img1_region_source,
                img0_region_representation,
                img1_region_representation,
            )

        return (
            img0,
            img1,
            img0_region_source,
            img1_region_source,
            img0_region_representation,
            img1_region_representation,
        )


class AutomaticShapeSelection(ImagePairsManipulationBase):
    def __init__(self, *manipulations: ImagePairsManipulationBase, strategy="closest_aspect"):
        self.manipulations = manipulations

        if strategy == "closest_aspect":
            self.strategy = self._closest_aspect_strategy
        else:
            raise ValueError("Unknown strategy")

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        output_shape, augmentor = self.strategy(H, W)

        if output_shape is None:
            raise ValueError("No valid shape found for the given resolution.")

        return output_shape

    def output_shape_pairs(self, H1: int, W1: int, H2: int, W2: int) -> Tuple[int, int, int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        output_shape, augmentor = self.strategy(H1, W1, H2, W2)

        if output_shape is None:
            raise ValueError("No valid shape found for the given resolution.")

        return output_shape

    def check_input(self, H: int, W: int) -> bool:
        output_shape, augmentor = self.strategy(H, W)

        if output_shape is None:
            return False

        return True

    def check_input_pairs(self, H1: int, W1: int, H2: int, W2: int) -> bool:
        output_shape, augmentor = self.strategy(H1, W1, H2, W2)

        if output_shape is None:
            return False

        return True

    def _closest_aspect_strategy(self, H: int, W: int, *shape_img1):
        # for all caididate sizes, first check if they can run at the given resolution
        if shape_img1 is None:
            runnable_sizes = [
                (manipulator.output_shape(H, W, *shape_img1), manipulator)
                for manipulator in self.manipulations
                if manipulator.check_input(H, W, *shape_img1)
            ]
        else:
            runnable_sizes = [
                (manipulator.output_shape_pairs(H, W, *shape_img1), manipulator)
                for manipulator in self.manipulations
                if manipulator.check_input_pairs(H, W, *shape_img1)
            ]

        if len(runnable_sizes) == 0:
            return None, None

        # if there are runnable sizes, then select the one that is closest to the given resolution
        if shape_img1 is None:
            closest_size, closest_augmentor = min(runnable_sizes, key=lambda x: abs(x[0][0] / x[0][1] - H / W))
        else:
            closest_size, closest_augmentor = min(
                runnable_sizes,
                key=lambda x: abs(x[0][0] / x[0][1] - H / W) + abs(x[0][2] / x[0][3] - shape_img1[0] / shape_img1[1]),
            )

        return closest_size, closest_augmentor

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        img0_region_source: Optional[torch.Tensor] = None,
        img1_region_source: Optional[torch.Tensor] = None,
        img0_region_representation: Optional[torch.Tensor] = None,
        img1_region_representation: Optional[torch.Tensor] = None,
    ):
        """
        Apply resizing, cropping, and padding to image pairs while recording correspondence information.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        Returns:
            - img0: Tensor of image0 after manipulation.
            - img1: Tensor of image1 after manipulation.
            - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
            - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
            - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
            - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        """

        H0, W0 = img0.shape[1], img0.shape[2]
        H1, W1 = img1.shape[1], img1.shape[2]

        output_shape, augmentor = self.strategy(H0, W0, H1, W1)

        if output_shape is None:
            raise ValueError("No valid shape found for the given resolution.")

        if img0_region_source is None:
            assert img1_region_source is None
            assert img0_region_representation is None
            assert img1_region_representation is None

            img0_region_source = torch.tensor([0, H0, 0, W0])
            img1_region_source = torch.tensor([0, H1, 0, W1])
            img0_region_representation = torch.tensor([0, H0, 0, W0])
            img1_region_representation = torch.tensor([0, H1, 0, W1])

        return augmentor(
            img0, img1, img0_region_source, img1_region_source, img0_region_representation, img1_region_representation
        )


# unmap the predicted flow to match the input. Flow is unique semantically as its value changes
# depending on the source and target region.
def unmap_predicted_flow(
    flow: torch.Tensor,
    img0_region_representation: torch.Tensor,
    img1_region_representation: torch.Tensor,
    img0_region_source: torch.Tensor,
    img1_region_source: torch.Tensor,
    img0_source_shape: Tuple[int, int],
    img1_source_shape: Tuple[int, int],
):
    """
    Unmap the predicted flow to the original image space.

    Args:
        - flow: Tensor of shape (B, 2, H, W) representing the predicted flow between the two regions.
        - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
        - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
        - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
    Returns:
        - flow: Tensor of shape (B, 2, H, W) representing the predicted flow in the original image space.
    """

    B, C, H, W = flow.shape

    # Step 1: Zero the start of flow representing mapping in model's output space
    # the flow end is the source coordinates + the flow
    flow_roi = flow[
        ...,
        img0_region_representation[0] : img0_region_representation[1],
        img0_region_representation[2] : img0_region_representation[3],
    ]

    source_offset = torch.tensor([img0_region_source[2], img0_region_source[0]]).to(flow.device)

    target_offset = torch.tensor([img1_region_source[2], img1_region_source[0]]).to(flow.device)

    flow_valid2valid = flow_roi  # + (source_offset - target_offset).view(1, 2, 1, 1)

    # Step 2: Represent the flow as pairs of source and target coordinates
    source_coordinates = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, flow_valid2valid.shape[3]) + 0.5,
                torch.arange(0, flow_valid2valid.shape[2]) + 0.5,
                indexing="xy",
            ),
            dim=-1,
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(flow.device)
    )

    # Step 3: Scale the flow to the source space. Notice that here we can actually assume
    # valid representation space have the same shape.
    # So it looks like both source and target coordinates are scaled according to the source representation.

    # now we scale the valid2valid flow from representation space to source space
    source_valid_shape = torch.tensor(
        [img0_region_source[1] - img0_region_source[0], img0_region_source[3] - img0_region_source[2]]
    )

    target_valid_shape = torch.tensor(
        [img1_region_source[1] - img1_region_source[0], img1_region_source[3] - img1_region_source[2]]
    )

    # upscale source and target coordinates to the source space
    source_coordinates_valid = F.interpolate(
        source_coordinates.float(), size=source_valid_shape.tolist(), mode="bilinear", align_corners=False
    )

    # This is equivalently we define "target_coordinates = source_coordinates + flow_valid2valid" and apply the scaling.
    # since we have a flow component, we can only do nearest interpolation, but this will cause ~0.5 pixel error
    # because we are interpoling the source_coordinates also linearly.

    target_coordinates_valid = (
        F.interpolate(flow_valid2valid.float(), size=source_valid_shape.tolist(), mode="nearest")
        + source_coordinates_valid
    )

    # print("Change me to nearest interpolation")

    # apply different scaling to the flow: representation for source maps to source_valid_shape in source space
    source_coordinates_valid *= (
        torch.tensor(
            [
                source_valid_shape[1] / (img0_region_representation[3] - img0_region_representation[2]),
                source_valid_shape[0] / (img0_region_representation[1] - img0_region_representation[0]),
            ]
        )
        .view(1, 2, 1, 1)
        .to(flow.device)
    )

    # target coordinates are scaled to the target source space, which may be different from the source space
    target_coordinates_valid *= (
        torch.tensor(
            [
                target_valid_shape[1] / (img0_region_representation[3] - img0_region_representation[2]),
                target_valid_shape[0] / (img0_region_representation[1] - img0_region_representation[0]),
            ]
        )
        .view(1, 2, 1, 1)
        .to(flow.device)
    )

    # Step 4: Offset the flow from valid source space to the original source space
    source_coordinates_valid += (
        torch.tensor([img0_region_source[2], img0_region_source[0]]).view(1, 2, 1, 1).to(flow.device)
    )

    target_coordinates_valid += (
        torch.tensor([img1_region_source[2], img1_region_source[0]]).view(1, 2, 1, 1).to(flow.device)
    )

    # now we can compute the flow in the source space
    flow_source = target_coordinates_valid - source_coordinates_valid

    # Step5: Embed the flow in its original space
    flow_output = torch.zeros((B, 2, img0_source_shape[0], img0_source_shape[1]), dtype=flow.dtype, device=flow.device)

    flow_output[..., img0_region_source[0] : img0_region_source[1], img0_region_source[2] : img0_region_source[3]] = (
        flow_source
    )

    flow_valid = torch.zeros((B, img0_source_shape[0], img0_source_shape[1]), dtype=torch.bool, device=flow.device)
    flow_valid[..., img0_region_source[0] : img0_region_source[1], img0_region_source[2] : img0_region_source[3]] = True

    return flow_output, flow_valid


# unmap predicted source - target point pairs.
def unmap_predicted_pairs(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    img0_region_representation: torch.Tensor,
    img1_region_representation: torch.Tensor,
    img0_region_source: torch.Tensor,
    img1_region_source: torch.Tensor,
    img0_source_shape: Tuple[int, int],
    img1_source_shape: Tuple[int, int],
):
    """
    Unmap the predicted flow to the original image space.

    Args:
        - source_points: Tensor of shape (B, N, 2) representing the predicted source points.
        - target_points: Tensor of shape (B, N, 2) representing the predicted target points.
        - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
        - img1_region_representation: Tensor of size 4 representing the region of img1 in current representation space corresponding to the source region.
        - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
        - img1_region_source: Tensor of size 4 representing the region of img1 that is the source of the correspondence.
    Returns:
        - flow: Tensor of shape (B, 2, H, W) representing the predicted flow in the original image space.
    """

    # 1. scale source points & target points from representation space to source space
    img0_region_source_shape = torch.tensor(
        [img0_region_source[1] - img0_region_source[0], img0_region_source[3] - img0_region_source[2]]
    )

    img1_region_source_shape = torch.tensor(
        [img1_region_source[1] - img1_region_source[0], img1_region_source[3] - img1_region_source[2]]
    )

    source_points[:, :, 0], _ = scale_axis(
        img0_region_source[2],
        img0_region_source[3],
        img0_region_representation[2],
        img0_region_representation[3],
        source_points[:, :, 0],
        0.0,
    )

    source_points[:, :, 1], _ = scale_axis(
        img0_region_source[0],
        img0_region_source[1],
        img0_region_representation[0],
        img0_region_representation[1],
        source_points[:, :, 1],
        0.0,
    )

    target_points[:, :, 0], _ = scale_axis(
        img1_region_source[2],
        img1_region_source[3],
        img1_region_representation[2],
        img1_region_representation[3],
        target_points[:, :, 0],
        0.0,
    )

    target_points[:, :, 1], _ = scale_axis(
        img1_region_source[0],
        img1_region_source[1],
        img1_region_representation[0],
        img1_region_representation[1],
        target_points[:, :, 1],
        0.0,
    )

    return source_points, target_points


# unmap normal channels like confidence, depth, etc.
# much simpler than the flow case
def unmap_predicted_channels(
    channel: torch.Tensor,
    img0_region_representation: torch.Tensor,
    img0_region_source: torch.Tensor,
    img0_source_shape: Tuple[int, int],
):
    """
    Unmap the predicted flow to the original image space.

    Args:
        - channel: Tensor of shape (B, C, H, W) representing the predicted values in img0 representation space
        - img0_region_representation: Tensor of size 4 representing the region of img0 in current representation space corresponding to the source region.
        - img0_region_source: Tensor of size 4 representing the region of img0 that is the source of the correspondence.
        - img0_source_shape: Tuple of size 2 representing the shape of the original image.
    Returns:
        - channel: Tensor of shape (B, C, H, W) representing the predicted flow in the original image space.
        - channel_valid: Tensor of shape (B, H, W) representing the valid region of the channel in the original image space.
    """

    B, C, H, W = channel.shape

    # Step 1: Zero the start of flow representing mapping in model's output space
    # the flow end is the source coordinates + the flow
    channel_roi = channel[
        ...,
        img0_region_representation[0] : img0_region_representation[1],
        img0_region_representation[2] : img0_region_representation[3],
    ]

    # upscale the channel roi into source space roi
    img0_valid_shape = torch.tensor(
        [img0_region_source[1] - img0_region_source[0], img0_region_source[3] - img0_region_source[2]]
    )

    channel_source_roi = F.interpolate(
        channel_roi,
        size=img0_valid_shape.tolist(),
        mode="nearest",
        # align_corners=False
    )

    channel_output = torch.zeros(
        (B, C, img0_source_shape[0], img0_source_shape[1]), dtype=channel.dtype, device=channel.device
    )
    channel_output[
        ..., img0_region_source[0] : img0_region_source[1], img0_region_source[2] : img0_region_source[3]
    ] = channel_source_roi

    channel_valid = torch.zeros(
        (B, img0_source_shape[0], img0_source_shape[1]), dtype=torch.bool, device=channel.device
    )
    channel_valid[..., img0_region_source[0] : img0_region_source[1], img0_region_source[2] : img0_region_source[3]] = (
        True
    )

    return channel_output, channel_valid


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    # make a example test image that have flow in only one pixel from (25%, 25%) to (50%, 75%) of the image.
    img0 = torch.zeros((1, 145, 256, 3), dtype=torch.uint8)  # one below and one above the aspect (288, 512)
    img1 = torch.zeros((1, 135, 256, 3), dtype=torch.uint8)

    source_pt = img0.shape[1] * 0.25, img0.shape[2] * 0.25
    target_pt = img1.shape[1] * 0.5, img1.shape[2] * 0.75

    img0[0, int(source_pt[0]), int(source_pt[1]), :] = 255
    img1[0, int(target_pt[0]), int(target_pt[1]), :] = 255

    flow_gt = torch.zeros((1, 2, 145, 256))
    flow_gt[0, :, int(source_pt[0]), int(source_pt[1])] = torch.tensor(
        [target_pt[1] - source_pt[1], target_pt[0] - source_pt[0]]
    )

    H0, W0 = img0.shape[1], img0.shape[2]
    H1, W1 = img1.shape[1], img1.shape[2]

    manipulation = AutomaticShapeSelection(
        ImagePairsManipulationComposite(ResizeHorizontalAxisManipulation(512), CenterCropManipulation((288, 512))),
        ImagePairsManipulationComposite(ResizeHorizontalAxisManipulation(512), CenterCropManipulation((200, 512))),
    )

    (
        img0_resized,
        img1_resized,
        img0_region_source,
        img1_region_source,
        img0_region_representation,
        img1_region_representation,
    ) = manipulation(img0, img1)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].imshow(img0[0].numpy())
    axs[0, 1].imshow(img0_resized[0].numpy())

    axs[1, 0].imshow(img1[0].numpy())
    axs[1, 1].imshow(img1_resized[0].numpy())

    print(img0_region_source)
    print(img1_region_source)
    print(img0_region_representation)
    print(img1_region_representation)

    flow_pred = torch.zeros((1, 2, 288, 512))
    flow_pred[0, :, 28, 128] = torch.tensor([256, 72])

    # unmap the flow
    flow_unmapped = unmap_predicted_flow(
        flow_pred,
        img0_region_representation,
        img1_region_representation,
        img0_region_source,
        img1_region_source,
        (H0, W0),
        (H1, W1),
    )

    flow_unmapped, flow_validity = flow_unmapped
    flow_unmapped = flow_unmapped[0]
    flow_validity = flow_validity[0]

    import flow_vis

    flow_rgb = flow_vis.flow_to_color(flow_unmapped.permute(1, 2, 0).numpy())
    axs[0, 2].imshow(flow_validity)

    plt.figure()
    plt.imshow(flow_rgb)
    plt.show()
