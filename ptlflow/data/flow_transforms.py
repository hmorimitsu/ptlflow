"""Operations to perform image augmentations for optical flow.

Some operations are adapted from the following sources:

- FlowNetPytorch: https://github.com/ClementPinard/FlowNetPytorch

- RAFT: https://github.com/princeton-vl/RAFT/

- flow-transforms-pytorch: https://github.com/hmorimitsu/flow-transforms-pytorch
"""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from collections.abc import KeysView
import random
from typing import Dict, Optional, Sequence, Tuple, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tt


class Compose(object):
    """Similar to torchvision Compose. Applies a series of transforms from the input list in sequence."""

    def __init__(self, transforms_list: Sequence[object]) -> None:
        """Initialize Compose.

        Parameters
        ----------
        transforms_list : Sequence[object]
            A sequence of transforms to be applied.
        """
        self.transforms_list = [t for t in transforms_list if t is not None]

    def __call__(
        self, inputs: Dict[str, Union[np.ndarray, Sequence[np.ndarray]]]
    ) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, Union[np.ndarray, Sequence[np.ndarray]]]
            Each element of the dict is either a single 3D HWC image or a list of images.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        for t in self.transforms_list:
            inputs = t(inputs)
        return inputs


class ToTensor(object):
    """Converts a 4D numpy.ndarray or a list of 3D numpy.ndarrays into a 4D torch.Tensor.

    If an input is of type uint8, then it is converted to float and its values are divided by 255.
    """

    def __init__(
        self,
        fp16: bool = False,
        device: Union[str, torch.device] = "cpu",
        use_keys: Optional[Union[KeysView, Sequence[str]]] = None,
        ignore_keys: Optional[Union[KeysView, Sequence[str]]] = None,
    ) -> None:
        """Initialize ToTensor.

        Parameters
        ----------
        fp16 : bool, default False
            If True, the tensors use have-precision floating point.
        device : Union[str, torch.device], default 'cpu'
            Name of the torch device where the tensors will be put in.
        use_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If it is not None, then only elements with these keys will be transformed. Otherwise, all elements are transformed,
            except the keys that are listed in ignore_keys.
        ignore_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If use_keys is None, the these keys are NOT transformed by this operation.
        """
        self.dtype = torch.float16 if fp16 else torch.float32
        self.device = device
        self.use_keys = use_keys
        self.ignore_keys = ignore_keys

    def __call__(
        self, inputs: Dict[str, Union[np.ndarray, Sequence[np.ndarray]]]
    ) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, Union[np.ndarray, Sequence[np.ndarray]]]
            Each element of the dict is either a single 3D HWC image or a list of images.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        valid_keys = _get_valid_keys(inputs.keys(), self.use_keys, self.ignore_keys)
        for k in valid_keys:
            v = inputs[k]
            if isinstance(v, list) or isinstance(v, tuple):
                v = np.stack(v)
                if len(v.shape) == 3:
                    v = v[:, :, :, None]

            if len(v.shape) == 2:
                v = v[None, :, :, None]
            elif len(v.shape) == 3:
                v = v[None]

            if v.dtype == np.uint8:
                v = v.astype(np.float32) / 255.0
            v = v.transpose(0, 3, 1, 2)
            inputs[k] = torch.from_numpy(v).to(device=self.device, dtype=self.dtype)
        return inputs


class GenerateFBCheckFlowOcclusion(object):
    """Generate occlusion masks based on forward/backward flow consistency.

    In other words, a pixel p is considered occluded when \|Ff(p) + Fb(p + F(f))\|_2 > threshold,
    where Ff and Fb get the forward and backward flow vectors of a pixel.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        forward_flow_key: str = "flows",
        backward_flow_key: str = "flows_b",
        forward_occlusion_key: str = "occs",
        backward_occlusion_key: str = "occs_b",
        compute_backward_occlusion: bool = True,
    ) -> None:
        """Initialize ColorJitter.

        Parameters
        ----------
        threshold : float, default 0.0
            A pixel is considered occluded if \|Ff(p) + Fb(p + F(f))\|_2 > threshold.
        forward_flow_key : str, default "flows"
            The name of the input dict entry that stores the forward flow.
        backward_flow_key : str, default "flows_b"
            The name of the input dict entry that stores the backward flow.
        forward_occlusion_key : str, default "occs"
            The name that will be added to the input dict to store the calculated occlusion masks for the forward flow.
        backward_occlusion_key : str, default "occs"
            The name that will be added to the input dict to store the calculated occlusion masks for the backward flow.
        compute_backward_occlusion : bool, default True
            If False, the occlusion mask is calculated only for the forward flow.
        """
        self.threshold = threshold
        self.forward_flow_key = forward_flow_key
        self.backward_flow_key = backward_flow_key
        self.forward_occlusion_key = forward_occlusion_key
        self.backward_occlusion_key = backward_occlusion_key
        self.compute_backward_occlusion = compute_backward_occlusion

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        assert self.forward_flow_key in inputs
        assert self.backward_flow_key in inputs
        flow_f = inputs[self.forward_flow_key]
        flow_b = inputs[self.backward_flow_key]
        b, c, h, w = flow_f.shape

        coords = self._coords_grid(b, h, w, dtype=flow_f.dtype, device=flow_f.device)
        flow_b_warped, oob_mask_f = self._bilinear_sampler(flow_b, coords + flow_f)
        diff_f = torch.norm(flow_f + flow_b_warped, p=2, dim=1, keepdim=True)
        inputs[self.forward_occlusion_key] = (
            (diff_f > self.threshold) | oob_mask_f
        ).float()

        if self.compute_backward_occlusion:
            flow_f_warped, oob_mask_b = self._bilinear_sampler(flow_f, coords + flow_b)
            diff_b = torch.norm(flow_b + flow_f_warped, p=2, dim=1, keepdim=True)
            inputs[self.backward_occlusion_key] = (
                (diff_b > self.threshold) | oob_mask_b
            ).float()

        return inputs

    def _bilinear_sampler(self, img, coords):
        # Code adapted from RAFT: https://github.com/princeton-vl/RAFT
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=1)
        img = F.grid_sample(img, grid.permute(0, 2, 3, 1), align_corners=True)

        mask = (xgrid < -1) | (ygrid < -1) | (xgrid > 1) | (ygrid > 1)
        return img, mask

    def _coords_grid(self, batch, ht, wd, dtype, device):
        # Code adapted from RAFT: https://github.com/princeton-vl/RAFT
        coords = torch.meshgrid(
            torch.arange(ht, dtype=dtype, device=device),
            torch.arange(wd, dtype=dtype, device=device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)


class CenterCrop(object):
    """Applies center crop to the inputs."""

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        occlusion_keys: Union[KeysView, Sequence[str]] = ("occs", "occs_b"),
        valid_key: str = "valids",
        ignore_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize RandomScaleAndCrop.

        Parameters
        ----------
        crop_size : Optional[Tuple[int, int]], optional
            If provided, crop the inputs to this size (h, w).
        occlusion_keys : Union[KeysView, Sequence[str]], default ['occs', 'occs_b']
            Indicate which of the input keys correspond to occlusion mask tensors.
        valid_keys : str, default 'valids'
            The name of the key in inputs that contains the binary mask indicating which pixels are valid.
            Only used when sparse=True.
        """
        self.crop_size = crop_size
        self.occlusion_keys = list(occlusion_keys)
        self.valid_key = valid_key
        self.ignore_keys = ignore_keys

    def __call__(  # noqa: C901
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.

        Raises
        ------
        NotImplementedError
            If trying to use time scale.
        """
        h, w = inputs[self.valid_key].shape[2:4]
        y_crop = (h - self.crop_size[0]) // 2
        x_crop = (w - self.crop_size[1]) // 2

        for k, v in inputs.items():
            if k not in self.ignore_keys:
                v = v[
                    :,
                    :,
                    y_crop : y_crop + self.crop_size[0],
                    x_crop : x_crop + self.crop_size[1],
                ]
            inputs[k] = v

        # Update occlusion masks for out-of-bounds flows
        for k, v in inputs.items():
            if self.ignore_keys is None or k not in self.ignore_keys:
                try:
                    i = self.occlusion_keys.index(k)
                    inputs[k] = _update_oob_flows(v, inputs[self.flow_keys[i]])
                except ValueError:
                    pass
        return inputs


class ColorJitter(tt.ColorJitter):
    """Randomly apply color transformations only to the images.

    If asymmetric_prob == 0, then the same transform is applied on all the images, otherwise, the transform for each image
    is randomly sampled independently.

    This is basically a wrapper for torchvision.transforms.ColorJitter.
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.0,
        contrast: Union[float, Tuple[float, float]] = 0.0,
        saturation: Union[float, Tuple[float, float]] = 0.0,
        hue: Union[float, Tuple[float, float]] = 0.0,
        asymmetric_prob: float = 0.0,
        use_keys: Optional[Union[KeysView, Sequence[str]]] = ("images",),
        ignore_keys: Optional[Union[KeysView, Sequence[str]]] = None,
    ) -> None:
        """Initialize ColorJitter.

        Parameters
        ----------
        brightness : Union[float, Tuple[float, float]], default 0.0
            The range to sample the random brightness value.
        contrast : Union[float, Tuple[float, float]], default 0.0
            The range to sample the random contrast value.
        saturation : Union[float, Tuple[float, float]], default 0.0
            The range to sample the random saturation value.
        hue : Union[float, Tuple[float, float]], default 0.0
            The range to sample the random hue value.
        asymmetric_prob : float, default 0.0
            Chance to apply an asymmetric transform, in which the parameters for transforming each image are sampled
            independently.
        use_keys : Optional[Union[KeysView, Sequence[str]]], default ['images']
            If it is not None, then only elements with these keys will be transformed. Otherwise, all elements are transformed,
            except the keys that are listed in ignore_keys.
        ignore_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If use_keys is None, the these keys are NOT transformed by this operation.
        """
        super().__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.asymmetric_prob = asymmetric_prob
        self.use_keys = use_keys
        self.ignore_keys = ignore_keys

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        valid_keys = _get_valid_keys(inputs.keys(), self.use_keys, self.ignore_keys)
        for k in valid_keys:
            v = inputs[k]
            if random.random() < self.asymmetric_prob:
                for i in range(len(v)):
                    inputs[k][i] = super().__call__(v[i])
            else:
                inputs[k] = super().__call__(v)
        return inputs


class GaussianNoise(object):
    """Applies random gaussian noise on the images."""

    def __init__(
        self,
        stdev: float = 0.0,
        use_keys: Optional[Union[KeysView, Sequence[str]]] = ("images",),
        ignore_keys: Optional[Union[KeysView, Sequence[str]]] = None,
    ) -> None:
        """Initialize GaussianNoise.

        Parameters
        ----------
        stdev : float, default 0.0
            The maximum standard deviation of the gaussian noise.
        use_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If it is not None, then only elements with these keys will be transformed. Otherwise, all elements are transformed,
            except the keys that are listed in ignore_keys.
        ignore_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If use_keys is None, the these keys are NOT transformed by this operation.
        """
        self.stdev = stdev
        self.use_keys = use_keys
        self.ignore_keys = ignore_keys

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        valid_keys = _get_valid_keys(inputs.keys(), self.use_keys, self.ignore_keys)
        for k in valid_keys:
            v = inputs[k]
            std = random.uniform(0.0, self.stdev)
            inputs[k] = (
                v + std * torch.randn(*v.shape, dtype=v.dtype, device=v.device)
            ).clamp(0.0, 1.0)
        return inputs


class RandomPatchEraser(object):
    """Randomly covers a rectangular patch on the second image with noise, to simulate a pseudo-occlusion.

    The noise_type may be the mean or random. This transform erases patches ONLY FROM THE SECOND IMAGE.
    """

    def __init__(
        self,
        erase_prob: float = 0.0,
        num_patches: Union[int, Tuple[int, int]] = 1,
        patch_size: Union[Tuple[int, int], Tuple[int, int, int, int]] = (0, 0),
        noise_type: str = "mean",
        use_keys: Optional[Union[KeysView, Sequence[str]]] = ("images",),
        ignore_keys: Optional[Union[KeysView, Sequence[str]]] = None,
    ) -> None:
        """Initialize RandomPatchEraser.

        Parameters
        ----------
        erase_prob : float, default 0.0
            Probability of applying the transformation.
        num_patches : Union[int, Tuple[int, int]], default 1
            Number of occlusion patches to generate. If it is a tuple, the number will be uniformly sampled from the interval.
        patch_size : Union[Tuple[int, int], Tuple[int, int, int, int]], default (0, 0)
            Range of the size of the occlusion patches. If it is a tuple with 2 elements, then both sides are sampled from the
            same interval. Otherwise, different intervals can be specified for each side as (hmin, hmax, wmin, wmax).
        noise_type : str, default 'mean'
            How to fill the occlusion patch. It can be either with the image 'mean' or with random 'noise'.
        use_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If it is not None, then only elements with these keys will be transformed. Otherwise, all elements are transformed,
            except the keys that are listed in ignore_keys.
        ignore_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If use_keys is None, the these keys are NOT transformed by this operation.
        """
        self.erase_prob = erase_prob
        self.noise_type = noise_type
        self.use_keys = use_keys
        self.ignore_keys = ignore_keys
        self.num_patches = num_patches
        if not (isinstance(num_patches, tuple) or isinstance(num_patches, list)):
            self.num_patches = (num_patches, num_patches)
        self.patch_size = patch_size
        if len(patch_size) == 2:
            self.patch_size = (
                patch_size[0],
                patch_size[1],
                patch_size[0],
                patch_size[1],
            )

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        if random.random() < self.erase_prob:
            valid_keys = _get_valid_keys(inputs.keys(), self.use_keys, self.ignore_keys)
            for k in valid_keys:
                im2 = inputs[k][1]
                c, h, w = im2.shape
                if self.noise_type == "mean":
                    mean_color = im2.reshape(c, -1).mean(dim=1)
                for _ in range(
                    random.randint(self.num_patches[0], self.num_patches[1])
                ):
                    hp = random.randint(self.patch_size[0], self.patch_size[1])
                    wp = random.randint(self.patch_size[2], self.patch_size[3])
                    if hp > 0 and wp > 0:
                        yp = random.randint(0, h - hp)
                        xp = random.randint(0, w - wp)
                        if self.noise_type == "mean":
                            noise = mean_color[:, None, None].repeat(1, hp, wp)
                        else:
                            im_min = im2.min()
                            im_max = im2.max()
                            im_inter = im_max - im_min
                            noise = (
                                im_inter
                                * torch.rand(
                                    c, hp, wp, dtype=im2.dtype, device=im2.device
                                )
                                + im_min
                            )
                        im2[:, yp : yp + hp, xp : xp + wp] = noise
        return inputs


class RandomFlip(object):
    """Randomly horizontally and vertically flips the inputs.

    If asymmetric_prob > 0, then each input of the sequence may be flipped differently.
    """

    def __init__(
        self,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        asymmetric_prob: float = 0.0,
        use_keys: Optional[Union[KeysView, Sequence[str]]] = None,
        ignore_keys: Optional[Union[KeysView, Sequence[str]]] = None,
        image_keys: Union[KeysView, Sequence[str]] = ("images",),
        flow_keys: Union[KeysView, Sequence[str]] = ("flows", "flows_b"),
    ) -> None:
        """Initialize RandomFlip.

        Parameters
        ----------
        hflip_prob : float, default 0.0
            Probability of applying a horizontal flip.
        vflip_prob : float, default 0.0
            Probability of applying a vertical flip.
        asymmetric_prob : float, default 0.0
            Chance to apply an asymmetric transform, in which the parameters for transforming each image are sampled
            independently.
        use_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If it is not None, then only elements with these keys will be transformed. Otherwise, all elements are transformed,
            except the keys that are listed in ignore_keys.
        ignore_keys : Optional[Union[KeysView, Sequence[str]]], optional
            If use_keys is None, the these keys are NOT transformed by this operation.
        image_keys : Union[KeysView, Sequence[str]], ['images']
            Indicate which of the input keys correspond to image tensors.
        flow_keys : Union[KeysView, Sequence[str]], ['flows', 'flows_b']
            Indicate which of the input keys correspond to optical flow tensors.
        """
        self.flip_probs = [hflip_prob, vflip_prob]
        self.asymmetric_prob = asymmetric_prob
        self.use_keys = use_keys
        self.ignore_keys = ignore_keys
        self.image_keys = list(image_keys)
        self.flow_keys = list(flow_keys)

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        valid_keys = _get_valid_keys(inputs.keys(), self.use_keys, self.ignore_keys)
        for iorient in range(2):
            if self.asymmetric_prob < 1e-5:
                if random.random() < self.flip_probs[iorient]:
                    inputs = self._flip_inputs(inputs, iorient == 0, valid_keys)
            else:
                is_flips = [
                    random.random() < self.flip_probs[iorient]
                    for _ in range(inputs[self.image_keys[0]].shape[0])
                ]
                for i in range(inputs[self.flow_keys[0]].shape[0]):
                    if is_flips[i]:
                        inputs = self._flip_inputs(
                            inputs, iorient == 0, valid_keys, ibatch=i
                        )
                    if is_flips[i] != is_flips[i + 1]:
                        for fk in self.flow_keys:
                            inputs[fk][i] = self._mirror_flow(
                                inputs[fk][i], iorient == 0
                            )
                if is_flips[-1]:
                    for ik in self.image_keys:
                        inputs = self._flip_inputs(
                            inputs, iorient == 0, inputs_keys=[ik], ibatch=-1
                        )

        return inputs

    def _flip_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        is_hflip: bool,
        valid_keys: Optional[Sequence[str]] = None,
        ibatch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Flips all inputs horizontally or vertically.

        This function properly adjust the flow values after the flipping.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be flipped. Each element is a 4D tensor NCHW.
        is_hflip : bool
            If True, performs a horizontal flip, otherwise, performs a vertical flip.
        valid_keys : Optional[Sequence[str]], optional
            If it is not None, then only elements with these keys will be transformed. Otherwise, all elements are transformed.
        ibatch : Optional[int], optional
            If ibatch is specified, then only one element of the batch is flipped.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs flipped by this operation.
        """
        if is_hflip:
            iinp = 3
            iflow = 0
        else:
            iinp = 2
            iflow = 1

        if valid_keys is None:
            valid_keys = list(inputs.keys())
        for k in valid_keys:
            if ibatch is None:
                inputs[k] = torch.flip(inputs[k], [iinp])
                if "flows" in k:
                    inputs[k][:, iflow] *= -1
            else:
                inputs[k][ibatch] = torch.flip(inputs[k][ibatch], [iinp - 1])
                if "flows" in k:
                    inputs[k][ibatch, iflow] *= -1
        return inputs

    def _mirror_flow(self, flow: torch.Tensor, is_hflip: bool) -> torch.Tensor:
        """Reflects the flow along the center line of the image.

        This function is used when an asymmetric flip happens (one image flips, but the next does not, or vice-versa).

        Parameters
        ----------
        flow : torch.Tensor
            A 3D tensor CHW.
        is_hflip : bool
            If True, performs a horizontal flip, otherwise, performs a vertical flip.

        Returns
        -------
        torch.Tensor
            The mirrored flow.
        """
        grid = torch.meshgrid(
            torch.arange(flow.shape[1]), torch.arange(flow.shape[2]), indexing="ij"
        )
        grid = torch.stack(grid[::-1]).float()
        if is_hflip:
            mean_coord = (flow.shape[2] - 1) / 2.0
            flow[0] = 2 * (mean_coord - grid[0]) - flow[0]
        else:
            mean_coord = (flow.shape[1] - 1) / 2.0
            flow[1] = 2 * (mean_coord - grid[1]) - flow[1]
        return flow


class RandomScaleAndCrop(object):
    """Applies first random scale and then random crop to the inputs.

    The scale is adjusted so that it is not smaller than the crop size.

    The scale calculation is composed of 2 main stages:

    1. A random major scale is sampled. The major scale defines the global scale applied to
       all images and dimensions. The major scale is calculated as:

       ms = 2 ** random.uniform(major_scale[0], major_scale[1])

    2. A random minor space scale is sampled. The space scale dictates the variation
       in scale applied to the width and height of each image. The space scale is
       calculated as:

       ssh = 2 ** random.uniform(space_scale[0], space_scale[1])

       ssw = 2 ** random.uniform(space_scale[2], space_scale[3]).

       If len(space_scale) == 2, then ssw also uses space_scale[0] and space_scale[1].

    The final scale applied to all inputs is:

        scale_height = ms * ssh

        scale_width = ms * ssw

    If time_scale is provided, then a third scale is sampled before computing the final scale.
    The time_scale is sampled independently for each element of a sequence. This allows,
    for example, for the first image have a different scale then the second one.
    The time scales tsh and tsw are calculated as the space scales ssh and ssw.
    With time scales, the final scales are calculated as:

        scale_height_time_t = ms * ssh * tsh_t

        scale_width_time_t = ms * ssw * tsw_t
    """

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        major_scale: Tuple[float, float] = (0.0, 0.0),
        space_scale: Union[Tuple[float, float], Tuple[float, float, float, float]] = (
            0.0,
            0.0,
        ),
        time_scale: Union[Tuple[float, float], Tuple[float, float, float, float]] = (
            0.0,
            0.0,
        ),
        binary_keys: Union[KeysView, Sequence[str]] = (
            "mbs",
            "occs",
            "valids",
            "mbs_b",
            "occs_b",
            "valids_b",
        ),
        flow_keys: Union[KeysView, Sequence[str]] = ("flows", "flows_b"),
        occlusion_keys: Union[KeysView, Sequence[str]] = ("occs", "occs_b"),
        sparse: bool = False,
        valid_key: str = "valids",
    ) -> None:
        """Initialize RandomScaleAndCrop.

        Parameters
        ----------
        crop_size : Optional[Tuple[int, int]], optional
            If provided, crop the inputs to this size (h, w).
        major_scale : Tuple[float, float], default (0.0, 0.0)
            The range of the major scale. See the class description for more details.
        space_scale : Union[Tuple[float, float], Tuple[float, float, float, float]], default (0.0, 0.0)
            The range of the minor scale. See the class description for more details.
        time_scale : Union[Tuple[float, float], Tuple[float, float, float, float]], default (0.0, 0.0)
            NOTE: Currently not implemented. The range of the time scale. See the class description for more details.
        binary_keys : Union[KeysView, Sequence[str]], default ['mbs', 'occs', 'valids', 'mbs_b', 'occs_b', 'valids_b']
            Indicate which of the input keys correspond to binary tensors.
        flow_keys : Union[KeysView, Sequence[str]], default ['flows', 'flows_b']
            Indicate which of the input keys correspond to optical flow tensors.
        occlusion_keys : Union[KeysView, Sequence[str]], default ['occs', 'occs_b']
            Indicate which of the input keys correspond to occlusion mask tensors.
        sparse : bool, default False
            If True, only values at valid positions (indicated by the mask in inputs[valid_key]) will be kept when
            resizing binary and flow inputs. Requires valid_key to exist as a key in inputs.
        valid_keys : str, default 'valids'
            The name of the key in inputs that contains the binary mask indicating which pixels are valid.
            Only used when sparse=True.
        """
        self.crop_size = crop_size
        self.major_scale = major_scale
        if len(major_scale) == 2:
            self.major_scale = (
                major_scale[0],
                major_scale[1],
                major_scale[0],
                major_scale[1],
            )
        self.space_scale = space_scale
        if len(space_scale) == 2:
            self.space_scale = (
                space_scale[0],
                space_scale[1],
                space_scale[0],
                space_scale[1],
            )
        self.time_scale = time_scale
        if len(time_scale) == 2:
            self.time_scale = (
                time_scale[0],
                time_scale[1],
                time_scale[0],
                time_scale[1],
            )

        self.use_time_scale = (
            abs(self.time_scale[1] - self.time_scale[0]) > 1e-5
            and abs(self.time_scale[3] - self.time_scale[2]) > 1e-5
        )
        self.binary_keys = list(binary_keys)
        self.flow_keys = list(flow_keys)
        self.occlusion_keys = list(occlusion_keys)
        self.sparse = sparse
        self.valid_key = valid_key

    def __call__(  # noqa: C901
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.

        Raises
        ------
        NotImplementedError
            If trying to use time scale.
        """
        h, w = inputs[self.flow_keys[0]].shape[2:4]
        major_scale = 2 ** random.uniform(self.major_scale[0], self.major_scale[1])
        space_scales = (
            2 ** random.uniform(self.space_scale[0], self.space_scale[1]),
            2 ** random.uniform(self.space_scale[2], self.space_scale[3]),
        )
        if self.use_time_scale:
            raise NotImplementedError("time_scale is currently not supported")
        else:
            min_size = self.crop_size
            if min_size is None:
                min_size = (1, 1)
            scaled_size = (
                max(min_size[0], int(h * major_scale * space_scales[0])),
                max(min_size[1], int(w * major_scale * space_scales[1])),
            )
            if self.crop_size is not None:
                y_crop = random.randint(0, scaled_size[0] - self.crop_size[0])
                x_crop = random.randint(0, scaled_size[1] - self.crop_size[1])

            inputs = _resize(
                inputs,
                scaled_size,
                self.binary_keys,
                self.flow_keys,
                self.sparse,
                self.valid_key,
            )
            if self.crop_size is not None:
                for k, v in inputs.items():
                    v = v[
                        :,
                        :,
                        y_crop : y_crop + self.crop_size[0],
                        x_crop : x_crop + self.crop_size[1],
                    ]
                    inputs[k] = v

            # Update occlusion masks for out-of-bounds flows
            for k, v in inputs.items():
                try:
                    i = self.occlusion_keys.index(k)
                    inputs[k] = _update_oob_flows(v, inputs[self.flow_keys[i]])
                except ValueError:
                    pass
        return inputs


class RandomTranslate(object):
    """Creates a translation between images by applying a random alternated crop on the sequence of inputs.

    A translation value t is randomly selected first. Then, the first image is cropped by a box translated by t.
    The second image will be cropped by a reversed translation -t. The third will be cropped by t again, and so on...
    """

    def __init__(
        self,
        translation: Union[int, Tuple[int, int]] = 0,
        flow_keys: Union[KeysView, Sequence[str]] = ("flows", "flows_b"),
        occlusion_keys: Union[KeysView, Sequence[str]] = ("occs", "occs_b"),
    ) -> None:
        """Initialize RandomTranslate.

        Parameters
        ----------
        translation : Union[int, Tuple[int, int]], default 0
            Maximum translation (in pixels) to be applied to the inputs. If a tuple, it corresponds to the maximum in the
            (y, x) axes.
        flow_keys : Union[KeysView, Sequence[str]], default ['flows', 'flows_b']
            Indicate which of the input keys correspond to optical flow tensors.
        occlusion_keys : Union[KeysView, Sequence[str]], default ['occs', 'occs_b']
            Indicate which of the input keys correspond to occlusion mask tensors.
        """
        self.translation = translation
        if not isinstance(translation, tuple) or isinstance(translation, list):
            self.translation = (translation, translation)
        self.flow_keys = flow_keys
        self.occlusion_keys = occlusion_keys

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        _, _, h, w = inputs[self.flow_keys[0]].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs

        trans_inputs = {
            k: torch.empty_like(v[:, :, : h - abs(th), : w - abs(tw)])
            for k, v in inputs.items()
        }

        # Translate: 0: even indexed inputs, 1: odd indexed inputs
        for t in range(2):
            if t == 0:
                ftw = tw
                fth = th
            else:
                ftw = -tw
                fth = -th
            x1, x2 = max(0, ftw), min(w + ftw, w)
            y1, y2 = max(0, fth), min(h + fth, h)
            for k, v in inputs.items():
                trans_inputs[k][t::2] = v[t::2, :, y1:y2, x1:x2]
                if k in self.flow_keys:
                    trans_inputs[k][t::2, 0] += ftw
                    trans_inputs[k][t::2, 1] += fth

        # Update occlusion masks for out-of-bounds flows
        for k, v in trans_inputs.items():
            try:
                i = self.occlusion_keys.index(k)
                trans_inputs[k] = _update_oob_flows(v, trans_inputs[self.flow_keys[i]])
            except ValueError:
                pass

        return trans_inputs


class RandomRotate(object):
    """Applies random rotation to the inputs.

    The inputs are rotated around the center of the image. First all inputs are rotated by the same random major `angle`.
    Then, another random angle a is sampled according to `diff_angle`. The first image will be rotated by a. The second image
    will be rotated by a reversed angle -a. The third will be rotated by a again, and so on...
    """

    def __init__(
        self,
        angle: float = 0.0,
        diff_angle: float = 0.0,
        flow_keys: Union[KeysView, Sequence[str]] = ("flows", "flows_b"),
        occlusion_keys: Union[KeysView, Sequence[str]] = ("occs", "occs_b"),
        valid_keys: Union[KeysView, Sequence[str]] = ("valids", "valids_b"),
        binary_keys: Union[KeysView, Sequence[str]] = (
            "mbs",
            "occs",
            "valids",
            "mbs_b",
            "occs_b",
            "valids_b",
        ),
        sparse: bool = False,
    ) -> None:
        """Initialize RandomRotate.

        Parameters
        ----------
        angle : float, default 0.0
            The maximum absolute value to sample the major angle from.
        diff_angle : float, default 0.0
            The maximum absolute value to sample the angle difference between consecutive images.
        flow_keys : Union[KeysView, Sequence[str]], default ['flows', 'flows_b']
            Indicate which of the input keys correspond to optical flow tensors.
        occlusion_keys : Union[KeysView, Sequence[str]], default ['occs', 'occs_b']
            Indicate which of the input keys correspond to occlusion mask tensors.
        valid_keys : Union[KeysView, Sequence[str]], default ['valids', 'valids_b']
            Indicate which of the input keys correspond to valid mask tensors.
        binary_keys : Union[KeysView, Sequence[str]], default ['mbs', 'occs', 'valids', 'mbs_b', 'occs_b', 'valids_b']
            Indicate which of the input keys correspond to binary tensors.
        sparse : bool, default False
            If True, all binary inputs and flows are rotated with nearest grid_sample, instead of bilinear.
        """
        self.angle = angle
        self.diff_angle = diff_angle
        self.flow_keys = flow_keys
        self.occlusion_keys = occlusion_keys
        self.valid_keys = valid_keys
        self.binary_keys = binary_keys
        self.sparse = sparse

    def __call__(  # noqa: C901
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        major_angle = random.uniform(-self.angle, self.angle)
        inter_angle = random.uniform(-self.diff_angle, self.diff_angle)

        input_tensor = inputs[self.flow_keys[0]]
        b, _, h, w = input_tensor.shape

        def generate_rotation_grid(
            rot_angle: float, batch_size: int, dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
            vy, vx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            vx = vx.type(dtype)
            vy = vy.type(dtype)
            vx = vx.to(device)
            vy = vy.to(device)
            vx -= (w - 1.0) / 2.0
            vy -= (h - 1.0) / 2.0
            angle_rad = rot_angle * 2 * np.pi / 360
            rotx = np.cos(angle_rad) * vx - np.sin(angle_rad) * vy
            roty = np.sin(angle_rad) * vx + np.cos(angle_rad) * vy
            rotx = rotx / ((w - 1) / 2)
            roty = roty / ((h - 1) / 2)
            rot_grid = torch.stack((rotx, roty), dim=2)[None]
            rot_grid = rot_grid.repeat(batch_size, 1, 1, 1)
            return rot_grid

        def generate_rotation_matrix(
            rot_angle: float, batch_size: int, dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
            vx, vy = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            vx = vx.type(dtype)
            vy = vy.type(dtype)
            vx = vx.to(device)
            vy = vy.to(device)
            rotx = (vx - h / 2.0) * (rot_angle * np.pi / 180.0)
            roty = -(vy - w / 2.0) * (rot_angle * np.pi / 180.0)
            rot_mat = torch.stack((rotx, roty), dim=0)[None]
            rot_mat = rot_mat.repeat(batch_size, 1, 1, 1)
            return rot_mat

        def rotate_flow(flow: torch.Tensor, rot_angle: float) -> torch.Tensor:
            angle_rad = rot_angle * 2 * np.pi / 360
            rot_flow = flow.clone()
            rot_flow[:, 0] = (
                np.cos(angle_rad) * flow[:, 0] + np.sin(angle_rad) * flow[:, 1]
            )
            rot_flow[:, 1] = (
                -np.sin(angle_rad) * flow[:, 0] + np.cos(angle_rad) * flow[:, 1]
            )
            return rot_flow

        rot_mat = generate_rotation_matrix(
            inter_angle, b // 2 + 1, input_tensor.dtype, input_tensor.device
        )
        for t in range(2):
            if t == 0:
                inangle = -inter_angle
                rmat = rot_mat
            else:
                inangle = inter_angle
                rmat = -rot_mat
            angle = major_angle + inangle / 2
            num_flows = input_tensor[t::2].shape[0]
            num_images = num_flows + 1
            rot_grid = generate_rotation_grid(
                angle, num_images, input_tensor.dtype, input_tensor.device
            )
            for k, v in inputs.items():
                if k in self.flow_keys:
                    v[t::2] += rmat[:num_flows]

                if k in self.binary_keys:
                    v[t::2] = F.grid_sample(
                        v[t::2], rot_grid[: v[t::2].shape[0]], mode="nearest"
                    )
                else:
                    if k in self.flow_keys:
                        if self.sparse:
                            v[t::2] = F.grid_sample(
                                v[t::2], rot_grid[: v[t::2].shape[0]], mode="nearest"
                            )
                        else:
                            v[t::2] = F.grid_sample(
                                v[t::2],
                                rot_grid[: v[t::2].shape[0]],
                                mode="bilinear",
                                align_corners=True,
                            )
                        v[t::2] = rotate_flow(v[t::2], angle)
                    else:
                        v[t::2] = F.grid_sample(
                            v[t::2],
                            rot_grid[: v[t::2].shape[0]],
                            mode="bilinear",
                            align_corners=True,
                        )

                if k in self.flow_keys:
                    v[t::2] = rotate_flow(v[t::2], angle)

                inputs[k] = v

        # Update occlusion masks for out-of-bounds flows
        for k, v in inputs.items():
            try:
                i = self.occlusion_keys.index(k)
                v = _update_oob_flows(v, inputs[self.flow_keys[i]])
                inputs[k] = v
            except ValueError:
                pass

        return inputs


class Resize(object):
    """Resize the image to a given size or scale.

    Size is checked first, if any of its values is zero, then scale is used.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (0, 0),
        scale: float = 1.0,
        binary_keys: Union[KeysView, Sequence[str]] = (
            "mbs",
            "occs",
            "valids",
            "mbs_b",
            "occs_b",
            "valids_b",
        ),
        flow_keys: Union[KeysView, Sequence[str]] = ("flows", "flows_b"),
        sparse: bool = False,
        valid_key: str = "valids",
        ignore_keys: Optional[Union[KeysView, Sequence[str]]] = None,
    ) -> None:
        """Initialize Resize.

        Parameters
        ----------
        size : Tuple[int, int], default (0, 0)
            The target size to resize the inputs. If it is zeros, then the scale will be used instead.
        scale : float, default 1.0
            The scale factor to resize the images. Only used if size is zeros.
        binary_keys : Union[KeysView, Sequence[str]], default ['mbs', 'occs', 'valids', 'mbs_b', 'occs_b', 'valids_b']
            Indicate which of the input keys correspond to binary tensors.
            [description], by default ['mbs', 'occs', 'valids', 'mbs_b', 'occs_b', 'valids_b']
        flow_keys : Union[KeysView, Sequence[str]], default ['flows', 'flows_b']
            Indicate which of the input keys correspond to optical flow tensors.
        sparse : bool, default False
            If True, only values at valid positions (indicated by the mask in inputs[valid_key]) will be kept when
            resizing binary and flow inputs. Requires valid_key to exist as a key in inputs.
        valid_keys : str, default 'valids'
            The name of the key in inputs that contains the binary mask indicating which pixels are valid.
            Only used when sparse=True.
        ignore_keys : Optional[Union[KeysView, Sequence[str]]]
            If not None, remove these keys from the inputs_keys.
        """
        self.size = size
        self.scale = scale
        self.binary_keys = list(binary_keys)
        self.flow_keys = list(flow_keys)
        self.sparse = sparse
        self.valid_key = valid_key
        self.ignore_keys = ignore_keys

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform the transformation on the inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Elements to be transformed. Each element is a 4D tensor NCHW.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs transformed by this operation.
        """
        h, w = inputs[list(inputs.keys())[0]].shape[2:4]
        if self.size is None or self.size[0] < 1 or self.size[1] < 1:
            self.size = (int(self.scale * h), int(self.scale * w))
        if self.size[0] != h or self.size[1] != w:
            inputs = _resize(
                inputs,
                self.size,
                self.binary_keys,
                self.flow_keys,
                self.sparse,
                self.valid_key,
                ignore_keys=self.ignore_keys,
            )
        return inputs


def _get_valid_keys(
    inputs_keys: Union[KeysView, Sequence[str]],
    use_keys: Optional[Union[KeysView, Sequence[str]]],
    ignore_keys: Optional[Union[KeysView, Sequence[str]]],
) -> Union[KeysView, Sequence[str]]:
    """Get only the valid keys from the input.

    Basically, it return use_keys, if not None. Otherwise, return inputs_keys after removing the keys which are in ignore_keys.

    Parameters
    ----------
    inputs_keys : Union[KeysView, Sequence[str]]
        All the keys available as input.
    use_keys : Optional[Union[KeysView, Sequence[str]]]
        If not None, then just use these keys.
    ignore_keys : Optional[Union[KeysView, Sequence[str]]]
        If not None, remove these keys from the inputs_keys.

    Returns
    -------
    Union[KeysView, Sequence[str]]
        The keys remaining after the validity checks.
    """
    if use_keys is not None:
        return use_keys
    if ignore_keys is None:
        return inputs_keys
    return [k for k in inputs_keys if k not in ignore_keys]


def _resize(
    inputs: Dict[str, torch.Tensor],
    target_size: Tuple[int, int],
    binary_keys: Union[KeysView, Sequence[str]],
    flow_keys: Union[KeysView, Sequence[str]],
    sparse: bool,
    valid_key: str,
    ignore_keys: Optional[Sequence[str]] = None,
):
    """Resize inputs to a target size. Set sparse=True when the valid mask has holes.
    This ensures that the resized valid mask does not interpolate the valid positions.

    Parameters
    ----------
    inputs : Dict[str, torch.Tensor]
        Elements to be transformed. Each element is a 4D tensor NCHW.
    target_size : Tuple[int, int]
        Target (height, width) sizes.
    binary_keys : Union[KeysView, Sequence[str]]
        Indicate which of the input keys correspond to binary tensors.
        [description], by default ['mbs', 'occs', 'valids', 'mbs_b', 'occs_b', 'valids_b']
    flow_keys : Union[KeysView, Sequence[str]]
        Indicate which of the input keys correspond to optical flow tensors.
    sparse : bool
        If True, only values at valid positions (indicated by the mask in inputs[valid_key]) will be kept when
        resizing binary and flow inputs. Requires valid_key to exist as a key in inputs.
    valid_keys : str
        The name of the key in inputs that contains the binary mask indicating which pixels are valid.
        Only used when sparse=True.

    Returns
    -------
    torch.Tensor
        The updated occlusion masks. Flows which went out-of-bounds are marked as occluded.
    """
    if sparse:
        assert (
            valid_key in inputs
        ), f"sparse is True, but valid_key({valid_key}) is not in inputs"
        valids = inputs[valid_key]
        n, k, h, w = valids.shape
        hs, ws = target_size
        scale_factor = torch.tensor(
            [float(ws) / w, float(hs) / h], device=valids.device
        )
        valids_flat = rearrange(valids, "n k h w -> n (k h w)")
        xy_scaled_list = []
        inbounds_list = []
        valids_out = torch.zeros(n, k, hs, ws, dtype=torch.float, device=valids.device)
        for i, vflat in enumerate(valids_flat):
            coords = torch.meshgrid(
                torch.arange(w, device=valids.device),
                torch.arange(h, device=valids.device),
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1)
            coords = rearrange(coords, "h w k -> (w h) k")

            coords_valid = coords[vflat >= 1]

            coords_scaled = coords_valid * scale_factor

            x_scaled = torch.round(coords_scaled[:, 0]).long()
            y_scaled = torch.round(coords_scaled[:, 1]).long()
            inbounds = (
                (x_scaled > 0) & (x_scaled < ws) & (y_scaled > 0) & (y_scaled < hs)
            )
            inbounds_list.append(inbounds)
            x_scaled = x_scaled[inbounds]
            y_scaled = y_scaled[inbounds]
            xy_scaled_list.append((x_scaled, y_scaled))

            valids_out[i, 0, y_scaled, x_scaled] = 1

        inputs[valid_key] = valids_out

        for k, v in inputs.items():
            if k != valid_key and (ignore_keys is None or k not in ignore_keys):
                if k in binary_keys or k in flow_keys:
                    v_out = torch.zeros(
                        v.shape[0], v.shape[1], hs, ws, dtype=v.dtype, device=v.device
                    )
                    for i, v_one in enumerate(v):
                        v_flat = rearrange(v_one, "k h w -> (h w) k")
                        v_valid = v_flat[valids_flat[i] >= 1]
                        if k in flow_keys:
                            v_valid = v_valid * scale_factor
                        v_valid = v_valid[inbounds_list[i]]
                        v_valid = rearrange(v_valid, "n k -> k n")
                        v_out[i, :, xy_scaled_list[i][1], xy_scaled_list[i][0]] = (
                            v_valid
                        )
                    v = v_out
                else:
                    v = F.interpolate(
                        v, size=target_size, mode="bilinear", align_corners=True
                    )
            inputs[k] = v
    else:
        for k, v in inputs.items():
            if ignore_keys is None or k not in ignore_keys:
                h, w = v.shape[-2:]
                if k in binary_keys:
                    v = F.interpolate(v, size=target_size, mode="nearest")
                else:
                    v = F.interpolate(
                        v, size=target_size, mode="bilinear", align_corners=True
                    )

                if k in flow_keys:
                    scale_mult = torch.tensor(
                        [float(target_size[1]) / w, float(target_size[0]) / h],
                        device=v.device,
                    )[None, :, None, None]
                    v = v * scale_mult

            inputs[k] = v

    return inputs


def _update_oob_flows(occs: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
    """Update occlusion maps to include flow which went out-of-bounds.

    Parameters
    ----------
    occs : torch.Tensor
        A 4D tensor NCHW of occlusion masks.
    flows : torch.Tensor
        A 4D tensor NCHW of optical flows.

    Returns
    -------
    torch.Tensor
        The updated occlusion masks. Flows which went out-of-bounds are marked as occluded.
    """
    grid = torch.meshgrid(
        torch.arange(flows.shape[2], dtype=flows.dtype, device=flows.device),
        torch.arange(flows.shape[3], dtype=flows.dtype, device=flows.device),
        indexing="ij",
    )
    grid = torch.stack(grid[::-1]).float()[None].repeat(flows.shape[0], 1, 1, 1)
    coords = flows + grid
    oob_occs = coords < 0
    oob_occs[:, 0] |= coords[:, 0] > flows.shape[3]
    oob_occs[:, 1] |= coords[:, 1] > flows.shape[2]
    oob_occs = oob_occs.max(dim=1, keepdim=True)[0].to(
        dtype=occs.dtype, device=occs.device
    )
    occs = torch.max(torch.stack([occs, oob_occs], dim=0), dim=0)[0]
    return occs
