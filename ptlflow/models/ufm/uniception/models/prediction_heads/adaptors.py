"""
Adaptors for the UniCeption Prediction Heads.
"""

from functools import lru_cache
from math import isfinite
from typing import List, Tuple, Union

import numpy as np
import torch

from .base import (
    AdaptorInput,
    AdaptorOutput,
    Covariance2DAdaptorOutput,
    MaskAdaptorOutput,
    RegressionAdaptorOutput,
    RegressionWithConfidenceAdaptorOutput,
    RegressionWithConfidenceAndMaskAdaptorOutput,
    RegressionWithMaskAdaptorOutput,
    UniCeptionAdaptorBase,
)


class FlowAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        flow_mean: Union[Tuple[float, float], List[float]],
        flow_std: Union[Tuple[float, float], List[float]],
        base_shape: Tuple[int, int],
        scale_strategy: str,
        output_normalized_coordinate: bool = False,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Flow head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            flow_mean (torch.Tensor): (2,) Mean of the flow.
            flow_std (torch.Tensor): (2,) Standard deviation of the flow.
            base_shape (Tuple[int, int]): Base shape of the flow mean and std.
            scale_strategy (str): Strategy for scaling the flow, either
            - none: No scaling, network will be unnormalized with the given mean and std for all input shapes
            - scale_width: scale the output for "none" by actual width divided by base width for both X and Y
            - scale_height: scale the output for "none" by actual height divided by base height for both X and Y
            - scale_both: scale the output for "none" by actual dimension / base dimension individually for X and Y
            output_normalized_coordinate (bool): If True, will subtract the (X, Y) coordinate of the output pixel from input x after it is being scaled to pixel coordinates.
            In other words, the network will predict the pixel position that the source pixel will land on the target image, rather than the flow.
        """
        super().__init__(name, required_channels=2, *args, **kwargs)

        self.name: str = name

        flow_mean = list(flow_mean)
        flow_std = list(flow_std)

        # Handle the case where flow_mean and flow_std are passed as tuples
        if isinstance(flow_mean, tuple) or isinstance(flow_mean, list):
            flow_mean = torch.tensor(flow_mean, dtype=torch.float32)
            assert flow_mean.shape == (2,), f"Flow mean must be a 2D tensor, got {flow_mean.shape}"

        if isinstance(flow_std, tuple) or isinstance(flow_std, list):
            flow_std = torch.tensor(flow_std, dtype=torch.float32)
            assert flow_std.shape == (2,), f"Flow std must be a 2D tensor, got {flow_std.shape}"

        self.register_buffer("flow_mean", flow_mean.view(1, 2, 1, 1))
        self.register_buffer("flow_std", flow_std.view(1, 2, 1, 1))

        self.base_shape = list(base_shape)
        self.scale_strategy = scale_strategy
        self.output_normalized_coordinate = output_normalized_coordinate

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the FlowAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)

        Returns:
            AdaptorOutput: Output of the adaptor.
        """

        x = adaptor_input.adaptor_feature

        # Check the number of channels to avoid passing BHWC features
        _, C, _, _ = x.shape
        assert C == 2, f"FlowAdaptor requires BCHW format with 2 channels, got {C} channels"

        output_shape = adaptor_input.output_shape_hw

        if not self.output_normalized_coordinate:
            x_scale, y_scale = self._get_xy_scale(output_shape)

            # Get the stored mean, std and scaling factors to scale the flow
            flow_mean = self.flow_mean.clone()
            flow_mean[:, 0] *= x_scale
            flow_mean[:, 1] *= y_scale
            flow_std = self.flow_std.clone()
            flow_std[:, 0] *= x_scale
            flow_std[:, 1] *= y_scale

            # Unnormalize the flow
            x = x * flow_std + flow_mean
        else:
            # Optionally subtract the coordinate bias
            wh_normalizer = torch.tensor(
                adaptor_input.output_shape_hw[::-1], dtype=torch.float32, device=x.device
            ).view(1, 2, 1, 1)

            x = 0.5 * (x + 1) * wh_normalizer + 0.5

            coords = self._get_coordinate_bias(output_shape, x.device)
            x = x - coords

        return RegressionAdaptorOutput(value=x)

    def _get_xy_scale(self, output_shape: Tuple[int, int]):
        """
        Get the scaling factor for the X and Y dimensions.

        Args:
            output_shape (Tuple[int, int]): HW Shape of the output.

        Returns:
            Tuple[float, float]: Scaling factors for X and Y dimensions.
        """
        if self.scale_strategy == "none":
            return 1.0, 1.0
        elif self.scale_strategy == "scale_width":
            return output_shape[1] / self.base_shape[1], output_shape[1] / self.base_shape[1]
        elif self.scale_strategy == "scale_height":
            return output_shape[0] / self.base_shape[0], output_shape[0] / self.base_shape[0]
        elif self.scale_strategy == "scale_both":
            return output_shape[1] / self.base_shape[1], output_shape[0] / self.base_shape[0]
        else:
            raise ValueError(f"Invalid scaling strategy: {self.scale_strategy}")

    @lru_cache(maxsize=10)
    def _get_coordinate_bias(self, output_shape: Tuple[int, int], device: str):
        """
        Get the (X, Y) coordinate image for the given output shape.

        Args:
            output_shape (Tuple[int, int]): HW Shape of the output.
            device: device to store the tensor on

        Returns:
            torch.Tensor: (2, H, W) tensor with X and Y coordinates, at device. This coordinate value will
            include 0.5 px offset - i.e. the center of the top-left pixel is (0.5, 0.5).
        """

        H, W = output_shape

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, device=device, dtype=torch.float32) + 0.5,
                torch.arange(0, H, device=device, dtype=torch.float32) + 0.5,
                indexing="xy",
            ),
            dim=0,
        )

        return coords


class ScaleAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = 0, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for scale prediction in UniCeption.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the scale prediction, either "linear", "square" or "exp". Scales the predicted scaling factor accordingly.
            vmin (float): Minimum value of the scale prediction after scaling.
            vmax (float): Maximum value of the scale prediction after scaling.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the ScaleAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x 1 x ...)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        predicted_scale_factor = adaptor_input.adaptor_feature
        output_scale_factor = None

        if self.mode == "linear":
            output_scale_factor = predicted_scale_factor
        elif self.mode == "square":
            output_scale_factor = predicted_scale_factor.square()
        elif self.mode == "exp":
            output_scale_factor = torch.exp(predicted_scale_factor)

        if not self.no_bounds:
            output_scale_factor = output_scale_factor.clip(self.vmin, self.vmax)

        return AdaptorOutput(value=output_scale_factor)


class DepthAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = 0, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the Depth head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the depth, either "linear", "square" or "exp". Scales the depth accordingly.
            vmin (float): Minimum value of the depth after scaling.
            vmax (float): Maximum value of the depth after scaling.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the DepthAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        x = adaptor_input.adaptor_feature
        output_depth = None

        if self.mode == "linear":
            output_depth = x
        elif self.mode == "square":
            output_depth = x**2
        elif self.mode == "exp":
            output_depth = torch.exp(x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self.no_bounds:
            output_depth = output_depth.clip(self.vmin, self.vmax)

        return RegressionAdaptorOutput(value=output_depth)


class SceneFlowAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the Motion head in UniCeption.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the SceneFlowAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        dxdydz = adaptor_input.adaptor_feature

        if self.mode == "linear":
            output_dxdydz = dxdydz
        elif self.mode == "square":
            output_dxdydz = dxdydz**2
        elif self.mode == "exp":
            output_dxdydz = torch.exp(dxdydz)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self.no_bounds:
            output_dxdydz = output_dxdydz.clip(self.vmin, self.vmax)

        return RegressionAdaptorOutput(value=output_dxdydz)


class PointMapAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the PointMap head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the point map, either "linear", "square" or "exp". Scales the distance of the points to the world origin accordingly.
            vmin (float): Minimum value of the point map after scaling.
            vmax (float): Maximum value of the point map after scaling.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the PointMapAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        xyz = adaptor_input.adaptor_feature
        output_xyz = None

        if self.mode != "linear":
            if self.mode == "square":
                # Compute distance to world origin
                d = xyz.norm(dim=1, keepdim=True)
                output_xyz = xyz / d.clip(min=1e-8)
                # Scale the distance to world origin based on mode
                output_xyz = output_xyz * d.square()
            elif self.mode == "exp":
                # Compute distance to world origin
                d = xyz.norm(dim=1, keepdim=True)
                output_xyz = xyz / d.clip(min=1e-8)
                # Scale the distance to world origin based on mode
                output_xyz = output_xyz * torch.expm1(d)
            elif self.mode == "z_exp":
                xy, z = xyz.split([2, 1], dim=1)
                z = torch.exp(z)
                output_xyz = torch.cat([xy * z, z], dim=1)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            output_xyz = xyz

        if not self.no_bounds:
            output_xyz = output_xyz.clip(self.vmin, self.vmax)

        return RegressionAdaptorOutput(value=output_xyz)


class RayOriginsAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the RayOrigins head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the ray origins, either "linear", "square" or "exp". Scales the distance of the ray origins to the world origin accordingly.
            vmin (float): Minimum value of the ray origins after scaling.
            vmax (float): Maximum value of the ray origins after scaling.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayOriginsAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_origins = adaptor_input.adaptor_feature
        output_ray_origins = None

        if self.mode != "linear":
            # Compute distance to world origin
            d = ray_origins.norm(dim=1, keepdim=True)
            output_ray_origins = ray_origins / d.clip(min=1e-8)
            # Scale the distance to world origin based on mode
            if self.mode == "square":
                output_ray_origins = output_ray_origins * d.square()
            elif self.mode == "exp":
                output_ray_origins = output_ray_origins * torch.expm1(d)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            output_ray_origins = ray_origins

        if not self.no_bounds:
            output_ray_origins = output_ray_origins.clip(self.vmin, self.vmax)

        return RegressionAdaptorOutput(value=output_ray_origins)


class RayDirectionsAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        mode: str,
        normalize_to_unit_sphere: bool,
        normalize_to_unit_image_plane: bool,
        vmin: float = -np.inf,
        vmax: float = np.inf,
        clamp_min_of_z_dir: bool = False,
        z_dir_min: float = 1,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the ray directions. Scales the directions accordingly. Currently only supports "linear".
            normalize_to_unit_sphere (bool): If True, will normalize the ray directions to unit vectors.
            normalize_to_unit_image_plane (bool): If True, will normalize the ray directions so that the z component is 1.
            vmin (float): Minimum value of the ray directions after scaling & before any sort of normalization. (default: -inf)
            vmax (float): Maximum value of the ray directions after scaling & before any sort of normalization. (default: inf)
            clamp_min_of_z_dir (bool): If True, will clamp the z component of the ray directions before normalization. (default: False)
            z_dir_min (float): If clamp_min_of_z_dir is True, this minimum value is used for clamping. (default: 1)
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.normalize_to_unit_sphere = normalize_to_unit_sphere
        self.normalize_to_unit_image_plane = normalize_to_unit_image_plane
        self.vmin = vmin
        self.vmax = vmax
        self.clamp_min_of_z_dir = clamp_min_of_z_dir
        self.z_dir_min = z_dir_min

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayDirectionsAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_directions = adaptor_input.adaptor_feature

        if self.mode == "linear":
            output_ray_directions = ray_directions
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self.no_bounds:
            output_ray_directions = output_ray_directions.clip(self.vmin, self.vmax)

        if self.clamp_min_of_z_dir:
            # Clamp the z component of ray directions
            output_ray_directions_xy = output_ray_directions[:, :2]
            clamped_output_ray_directions_z = torch.clamp(output_ray_directions[:, 2:3], min=self.z_dir_min)
            output_ray_directions = torch.cat((output_ray_directions_xy, clamped_output_ray_directions_z), dim=1)

        if self.normalize_to_unit_sphere:
            # Normalize the ray directions to unit vectors
            output_ray_dirs_norm = output_ray_directions.norm(dim=1, keepdim=True).clip(min=1e-8)
            output_ray_directions = output_ray_directions / output_ray_dirs_norm
        elif self.normalize_to_unit_image_plane:
            # Normalize the ray directions so that the z component is 1
            output_ray_directions_z = output_ray_directions[:, 2:3]
            output_ray_directions = output_ray_directions / output_ray_directions_z

        return RegressionAdaptorOutput(value=output_ray_directions)


class RayDirectionsPlusDepthAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth head in UniCeption.
        """
        super().__init__(name, required_channels=4, *args, **kwargs)

        self.ray_directions_adaptor = RayDirectionsAdaptor(
            name,
            ray_directions_mode,
            ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin,
            ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min,
        )
        self.depth_adaptor = DepthAdaptor(name, depth_mode, depth_vmin, depth_vmax)

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayMapPlusDepthAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_directions, ray_depths = torch.split(adaptor_input.adaptor_feature, [3, 1], dim=1)
        ray_directions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_directions, output_shape_hw=adaptor_input.output_shape_hw
        )
        depth_adaptor_input = AdaptorInput(adaptor_feature=ray_depths, output_shape_hw=adaptor_input.output_shape_hw)
        output_ray_directions = self.ray_directions_adaptor(ray_directions_adaptor_input)
        output_depth = self.depth_adaptor(depth_adaptor_input)
        output = torch.cat([output_ray_directions.value, output_depth.value], dim=1)

        return RegressionAdaptorOutput(value=output)


class RayDirectionsPlusDepthPlusSceneFlowAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Scene flow adaptor
        scene_flow_mode: str,
        scene_flow_vmin: float,
        scene_flow_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth + SceneFlow head in UniCeption.
        """
        super().__init__(name, required_channels=7, *args, **kwargs)

        self.ray_directions_adaptor = RayDirectionsAdaptor(
            name,
            ray_directions_mode,
            ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin,
            ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min,
        )
        self.depth_adaptor = DepthAdaptor(name, depth_mode, depth_vmin, depth_vmax)

        self.scene_flow_adaptor = SceneFlowAdaptor(name, scene_flow_mode, scene_flow_vmin, scene_flow_vmax)

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayDirectionsPlusDepthPlusSceneFlowAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_directions, ray_depths, scene_flow = torch.split(adaptor_input.adaptor_feature, [3, 1, 3], dim=1)
        ray_directions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_directions, output_shape_hw=adaptor_input.output_shape_hw
        )
        depth_adaptor_input = AdaptorInput(adaptor_feature=ray_depths, output_shape_hw=adaptor_input.output_shape_hw)
        scene_flow_adaptor_input = AdaptorInput(
            adaptor_feature=scene_flow, output_shape_hw=adaptor_input.output_shape_hw
        )
        output_ray_directions = self.ray_directions_adaptor(ray_directions_adaptor_input)
        output_depth = self.depth_adaptor(depth_adaptor_input)
        output_scene_flow = self.scene_flow_adaptor(scene_flow_adaptor_input)
        output = torch.cat([output_ray_directions.value, output_depth.value, output_scene_flow.value], dim=1)

        return RegressionAdaptorOutput(value=output)


class CamTranslationAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, mode: str, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs):
        """
        Adaptor for the Camera Translation or Pose head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the camera translation, either "linear", "square" or "exp". Scales the distance of the camera to the world origin accordingly.
            vmin (float): Minimum value of the camera translation after scaling.
            vmax (float): Maximum value of the camera translation after scaling.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the CamTranslationAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C ...)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        cam_trans = adaptor_input.adaptor_feature
        output_cam_trans = None

        if self.mode != "linear":
            # Compute distance to world origin
            d = cam_trans.norm(dim=1, keepdim=True)
            output_cam_trans = cam_trans / d.clip(min=1e-8)
            # Scale the distance to world origin based on mode
            if self.mode == "square":
                output_cam_trans = output_cam_trans * d.square()
            elif self.mode == "exp":
                output_cam_trans = output_cam_trans * torch.expm1(d)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            output_cam_trans = cam_trans

        if not self.no_bounds:
            output_cam_trans = output_cam_trans.clip(self.vmin, self.vmax)

        return AdaptorOutput(value=output_cam_trans)


class QuaternionsAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self, name: str, mode: str, normalize: bool, vmin: float = -np.inf, vmax: float = np.inf, *args, **kwargs
    ):
        """
        Adaptor for the Quaternions or Pose head in UniCeption.
        Notation of the quaternions: (x, y, z, w)

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the quaternions. Scales the quaternions accordingly before normalization. Currently only supports "linear".
            normalize (bool): If True, will normalize the quaternions to unit quaternions.
            vmin (float): Minimum value of the quaternions after scaling & before normalization to unit quaternions if required.
            vmax (float): Maximum value of the quaternions after scaling & before normalization to unit quaternions if required.
        """
        super().__init__(name, required_channels=4, *args, **kwargs)

        self.mode = mode
        self.normalize = normalize
        self.vmin = vmin
        self.vmax = vmax

        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the QuaternionsAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C ...)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        quaternions = adaptor_input.adaptor_feature

        if self.mode == "linear":
            output_quaternions = quaternions
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self.no_bounds:
            output_quaternions = output_quaternions.clip(self.vmin, self.vmax)

        if self.normalize:
            # Normalize the quaternions to unit quaternions
            output_quats_norm = output_quaternions.norm(dim=1, keepdim=True).clip(min=1e-8)
            output_quaternions = output_quaternions / output_quats_norm

        return AdaptorOutput(value=output_quaternions)


class CamTranslationPlusQuatsAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Cam translation adaptor
        cam_trans_mode: str,
        cam_trans_vmin: float,
        cam_trans_vmax: float,
        # Quaternions adaptor
        quaternions_mode: str,
        quaternions_normalize: bool,
        quaternions_vmin: float,
        quaternions_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Camera Translation + Quaternions head in UniCeption.
        """
        super().__init__(name, required_channels=7, *args, **kwargs)

        self.cam_trans_adaptor = CamTranslationAdaptor(name, cam_trans_mode, cam_trans_vmin, cam_trans_vmax)
        self.quaternions_adaptor = QuaternionsAdaptor(
            name, quaternions_mode, quaternions_normalize, quaternions_vmin, quaternions_vmax
        )

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the CamTranslationPlusQuatsAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C ...)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        cam_trans, quaternions = torch.split(adaptor_input.adaptor_feature, [3, 4], dim=1)
        cam_trans_adaptor_input = AdaptorInput(adaptor_feature=cam_trans, output_shape_hw=adaptor_input.output_shape_hw)
        quaternions_adaptor_input = AdaptorInput(
            adaptor_feature=quaternions, output_shape_hw=adaptor_input.output_shape_hw
        )
        output_cam_trans = self.cam_trans_adaptor(cam_trans_adaptor_input)
        output_quaternions = self.quaternions_adaptor(quaternions_adaptor_input)
        output = torch.cat([output_cam_trans.value, output_quaternions.value], dim=1)

        return AdaptorOutput(value=output)


class RayMapAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Ray origins adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) head in UniCeption.
        """
        super().__init__(name, required_channels=6, *args, **kwargs)

        self.ray_origins_adaptor = RayOriginsAdaptor(name, ray_origins_mode, ray_origins_vmin, ray_origins_vmax)
        self.ray_directions_adaptor = RayDirectionsAdaptor(
            name,
            ray_directions_mode,
            ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin,
            ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min,
        )

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayMapAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_origins, ray_directions = torch.split(adaptor_input.adaptor_feature, 3, dim=1)
        ray_origins_adaptor_input = AdaptorInput(
            adaptor_feature=ray_origins, output_shape_hw=adaptor_input.output_shape_hw
        )
        ray_directions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_directions, output_shape_hw=adaptor_input.output_shape_hw
        )
        output_ray_origins = self.ray_origins_adaptor(ray_origins_adaptor_input)
        output_ray_directions = self.ray_directions_adaptor(ray_directions_adaptor_input)
        output_rays = torch.cat([output_ray_origins.value, output_ray_directions.value], dim=1)

        return RegressionAdaptorOutput(value=output_rays)


class RayMapPlusDepthAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Ray origins adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth head in UniCeption.
        """
        super().__init__(name, required_channels=7, *args, **kwargs)

        self.ray_origins_adaptor = RayOriginsAdaptor(name, ray_origins_mode, ray_origins_vmin, ray_origins_vmax)
        self.ray_directions_adaptor = RayDirectionsAdaptor(
            name,
            ray_directions_mode,
            ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin,
            ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min,
        )
        self.depth_adaptor = DepthAdaptor(name, depth_mode, depth_vmin, depth_vmax)

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayMapPlusDepthAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_origins, ray_directions, ray_depths = torch.split(adaptor_input.adaptor_feature, [3, 3, 1], dim=1)
        ray_origins_adaptor_input = AdaptorInput(
            adaptor_feature=ray_origins, output_shape_hw=adaptor_input.output_shape_hw
        )
        ray_directions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_directions, output_shape_hw=adaptor_input.output_shape_hw
        )
        depth_adaptor_input = AdaptorInput(adaptor_feature=ray_depths, output_shape_hw=adaptor_input.output_shape_hw)
        output_ray_origins = self.ray_origins_adaptor(ray_origins_adaptor_input)
        output_ray_directions = self.ray_directions_adaptor(ray_directions_adaptor_input)
        output_depth = self.depth_adaptor(depth_adaptor_input)
        output = torch.cat([output_ray_origins.value, output_ray_directions.value, output_depth.value], dim=1)

        return RegressionAdaptorOutput(value=output)


class RayMapPlusDepthPlusQuatsAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Ray origins adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Quaternions adaptor
        quaternions_mode: str,
        quaternions_normalize: bool,
        quaternions_vmin: float,
        quaternions_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth + Quaternions head in UniCeption.
        """
        super().__init__(name, required_channels=11, *args, **kwargs)

        self.ray_origins_adaptor = RayOriginsAdaptor(name, ray_origins_mode, ray_origins_vmin, ray_origins_vmax)
        self.ray_directions_adaptor = RayDirectionsAdaptor(
            name,
            ray_directions_mode,
            ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin,
            ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min,
        )
        self.depth_adaptor = DepthAdaptor(name, depth_mode, depth_vmin, depth_vmax)
        self.quaternions_adaptor = QuaternionsAdaptor(
            name, quaternions_mode, quaternions_normalize, quaternions_vmin, quaternions_vmax
        )

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the RayMapPlusDepthPlusQuatsAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        ray_origins, ray_directions, ray_depths, ray_quaternions = torch.split(
            adaptor_input.adaptor_feature, [3, 3, 1, 4], dim=1
        )
        ray_origins_adaptor_input = AdaptorInput(
            adaptor_feature=ray_origins, output_shape_hw=adaptor_input.output_shape_hw
        )
        ray_directions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_directions, output_shape_hw=adaptor_input.output_shape_hw
        )
        depth_adaptor_input = AdaptorInput(adaptor_feature=ray_depths, output_shape_hw=adaptor_input.output_shape_hw)
        quaternions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_quaternions, output_shape_hw=adaptor_input.output_shape_hw
        )
        output_ray_origins = self.ray_origins_adaptor(ray_origins_adaptor_input)
        output_ray_directions = self.ray_directions_adaptor(ray_directions_adaptor_input)
        output_ray_depths = self.depth_adaptor(depth_adaptor_input)
        output_ray_quaternions = self.quaternions_adaptor(quaternions_adaptor_input)
        output = torch.cat(
            [
                output_ray_origins.value,
                output_ray_directions.value,
                output_ray_depths.value,
                output_ray_quaternions.value,
            ],
            dim=1,
        )

        return RegressionAdaptorOutput(value=output)


class PointMapPlusRayDirectionsPlusDepthAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        # Point map adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap + RayDirections + Depth head in UniCeption.
        """
        super().__init__(name, required_channels=7, *args, **kwargs)

        self.pointmap_adaptor = PointMapAdaptor(name, pointmap_mode, pointmap_vmin, pointmap_vmax)
        self.ray_directions_adaptor = RayDirectionsAdaptor(
            name,
            ray_directions_mode,
            ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin,
            ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min,
        )
        self.depth_adaptor = DepthAdaptor(name, depth_mode, depth_vmin, depth_vmax)

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the PointMapPlusRayDirectionsPlusDepthAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        pointmap, ray_directions, ray_depths = torch.split(adaptor_input.adaptor_feature, [3, 3, 1], dim=1)
        pointmap_adaptor_input = AdaptorInput(adaptor_feature=pointmap, output_shape_hw=adaptor_input.output_shape_hw)
        ray_directions_adaptor_input = AdaptorInput(
            adaptor_feature=ray_directions, output_shape_hw=adaptor_input.output_shape_hw
        )
        depth_adaptor_input = AdaptorInput(adaptor_feature=ray_depths, output_shape_hw=adaptor_input.output_shape_hw)
        output_pointmap = self.pointmap_adaptor(pointmap_adaptor_input)
        output_ray_directions = self.ray_directions_adaptor(ray_directions_adaptor_input)
        output_depth = self.depth_adaptor(depth_adaptor_input)
        output = torch.cat([output_pointmap.value, output_ray_directions.value, output_depth.value], dim=1)

        return RegressionAdaptorOutput(value=output)


class ConfidenceAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        confidence_type: str,
        vmin: float,
        vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Confidence head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            confidence_type (str): Type of the confidence, either
            - exp: Exponential confidence
            - sigmoid: Sigmoid confidence
            vmin (float): Minimum value of the confidence.
            vmax (float): Maximum value of the confidence.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

        self.confidence_type = confidence_type
        self.vmin = vmin
        self.vmax = vmax

        assert vmin < vmax, "vmin must be less than vmax"

        if confidence_type == "sigmoid":
            assert isfinite(vmin) and isfinite(vmax), "vmin and vmax must be finite for sigmoid confidence"
            assert vmin >= 0

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the ConfidenceAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """

        x = adaptor_input.adaptor_feature

        if self.confidence_type == "exp":
            confidence = self.vmin + x.exp().clip(max=self.vmax - self.vmin)

            return RegressionAdaptorOutput(value=confidence)

        elif self.confidence_type == "sigmoid":
            confidence = torch.sigmoid(x)

            confidence = confidence * (self.vmax - self.vmin) + self.vmin

            return RegressionAdaptorOutput(value=confidence)

        elif self.confidence_type == "softmax":
            B, C, H, W = x.shape
            confidence = torch.nn.functional.softmax(x.reshape(B, C, -1), dim=-1).reshape(B, C, H, W) * (H * W)

            return RegressionAdaptorOutput(value=confidence)


class Covariance2DAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        parametrization: str = "exp_tanh",
        low_confidence_init: bool = False,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Covariance2D head in UniCeption.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)
        self.parametrization = parametrization
        self.low_confidence_init = low_confidence_init

    def forward(self, adaptor_input: AdaptorInput):
        x = adaptor_input.adaptor_feature

        if self.parametrization == "exp_tanh":
            c1, c2, s = torch.split(x, 1, dim=1)

            if self.low_confidence_init:
                c1 = c1 + 8
                c2 = c2 + 8

            diag_exponent = (c1 + c2) / 2
            tanh_s = s.tanh()

            cov = torch.cat([c1.exp(), c2.exp(), tanh_s * torch.exp(diag_exponent)], dim=1)

            log_det = c1 + c2 + torch.log(1 - torch.square(tanh_s) + 1e-8)

            inv_coeff = 1 / (1 - torch.square(tanh_s) + 1e-8)
            inv_cov = inv_coeff * torch.cat(
                [torch.exp(-c1), torch.exp(-c2), -tanh_s * torch.exp(-diag_exponent)], dim=1
            )

        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")

        return Covariance2DAdaptorOutput(covariance=cov, log_det=log_det, inv_covariance=inv_cov, log_representation=x)

    @classmethod
    def decode(cls, x: torch.Tensor, representation: str):
        if representation == "exp_tanh":
            c1, c2, s = torch.split(x, 1, dim=1)

            # For smooth initialization
            c1 = c1 + 8
            c2 = c2 + 8

            diag_exponent = (c1 + c2) / 2
            tanh_s = s.tanh()

            cov = torch.cat([c1.exp(), c2.exp(), tanh_s * torch.exp(diag_exponent)], dim=1)

            log_det = c1 + c2 + torch.log(1 - torch.square(tanh_s) + 1e-8)

            inv_coeff = 1 / (1 - torch.square(tanh_s) + 1e-8)
            inv_cov = inv_coeff * torch.cat(
                [torch.exp(-c1), torch.exp(-c2), -tanh_s * torch.exp(-diag_exponent)], dim=1
            )

        else:
            raise ValueError(f"Invalid parametrization: {representation}")

        return Covariance2DAdaptorOutput(covariance=cov, log_det=log_det, inv_covariance=inv_cov, log_representation=x)


class MaskAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Mask head in UniCeption.
        """
        super().__init__(name, required_channels=1, *args, **kwargs)

    def forward(self, adaptor_input: AdaptorInput):
        x = adaptor_input.adaptor_feature

        mask = torch.sigmoid(x)

        return MaskAdaptorOutput(logits=x, mask=mask)


class ValueWithConfidenceAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        value_adaptor: UniCeptionAdaptorBase,
        confidence_adaptor: UniCeptionAdaptorBase,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Value with Confidence head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            value_adaptor (UniCeptionAdaptorBase): Adaptor for the value.
            confidence_adaptor (UniCeptionAdaptorBase): Adaptor for the confidence.
        """

        super().__init__(
            name,
            required_channels=value_adaptor.required_channels + confidence_adaptor.required_channels,
            *args,
            **kwargs,
        )

        self.value_adaptor = value_adaptor
        self.confidence_adaptor = confidence_adaptor

    def forward(self, adaptor_input: AdaptorInput):
        value_input, confidence_input = torch.split(
            adaptor_input.adaptor_feature,
            [self.value_adaptor.required_channels, self.confidence_adaptor.required_channels],
            dim=1,
        )
        value_adaptor_input = AdaptorInput(adaptor_feature=value_input, output_shape_hw=adaptor_input.output_shape_hw)
        confidence_adaptor_input = AdaptorInput(
            adaptor_feature=confidence_input, output_shape_hw=adaptor_input.output_shape_hw
        )
        value_output = self.value_adaptor(value_adaptor_input)
        confidence_output = self.confidence_adaptor(confidence_adaptor_input)

        return RegressionWithConfidenceAdaptorOutput(value=value_output.value, confidence=confidence_output.value)


class FlowWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # Flow adaptor
        flow_mean: torch.Tensor,
        flow_std: torch.Tensor,
        base_shape: Tuple[int, int],
        scale_strategy: str,
        output_normalized_coordinate: bool,
        # Confidence adaptor
        confidence_type: str,
        vmin: float,
        vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Flow with Confidence head in UniCeption.
        """
        flow_adaptor = FlowAdaptor(
            name=f"{name}",
            flow_mean=flow_mean,
            flow_std=flow_std,
            base_shape=base_shape,
            scale_strategy=scale_strategy,
            output_normalized_coordinate=output_normalized_coordinate,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=vmin, vmax=vmax
        )

        super().__init__(name, value_adaptor=flow_adaptor, confidence_adaptor=confidence_adaptor, *args, **kwargs)


class PointMapWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # Pointmap adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap with Confidence head in UniCeption.
        """
        pointmap_adaptor = PointMapAdaptor(name=f"{name}", mode=pointmap_mode, vmin=pointmap_vmin, vmax=pointmap_vmax)

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(name, value_adaptor=pointmap_adaptor, confidence_adaptor=confidence_adaptor, *args, **kwargs)


class PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # Point map adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap + RayDirections + Depth with Confidence head in UniCeption.
        """
        pointmap_plus_ray_directions_plus_depth_adaptor = PointMapPlusRayDirectionsPlusDepthAdaptor(
            name=f"{name}",
            pointmap_mode=pointmap_mode,
            pointmap_vmin=pointmap_vmin,
            pointmap_vmax=pointmap_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(
            name,
            value_adaptor=pointmap_plus_ray_directions_plus_depth_adaptor,
            confidence_adaptor=confidence_adaptor,
            *args,
            **kwargs,
        )


class RayDirectionsPlusDepthWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth with Confidence head in UniCeption.
        """
        ray_directions_plus_depth_adaptor = RayDirectionsPlusDepthAdaptor(
            name=f"{name}",
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(
            name,
            value_adaptor=ray_directions_plus_depth_adaptor,
            confidence_adaptor=confidence_adaptor,
            *args,
            **kwargs,
        )


class RayDirectionsPlusDepthPlusSceneFlowWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Scene flow adaptor
        scene_flow_mode: str,
        scene_flow_vmin: float,
        scene_flow_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth with Confidence head in UniCeption.
        """
        ray_directions_plus_depth_plus_scene_flow_adaptor = RayDirectionsPlusDepthPlusSceneFlowAdaptor(
            name=f"{name}",
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
            scene_flow_mode=scene_flow_mode,
            scene_flow_vmin=scene_flow_vmin,
            scene_flow_vmax=scene_flow_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(
            name,
            value_adaptor=ray_directions_plus_depth_plus_scene_flow_adaptor,
            confidence_adaptor=confidence_adaptor,
            *args,
            **kwargs,
        )


class RayMapPlusDepthWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # RayMap adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth with Confidence head in UniCeption.
        """
        raymap_plus_depth_adaptor = RayMapPlusDepthAdaptor(
            name=f"{name}",
            ray_origins_mode=ray_origins_mode,
            ray_origins_vmin=ray_origins_vmin,
            ray_origins_vmax=ray_origins_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(
            name, value_adaptor=raymap_plus_depth_adaptor, confidence_adaptor=confidence_adaptor, *args, **kwargs
        )


class RayMapPlusDepthPlusQuatsWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        # RayMap adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Quaternions adaptor
        quaternions_mode: str,
        quaternions_normalize: bool,
        quaternions_vmin: float,
        quaternions_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth + Quaternions with Confidence head in UniCeption.
        """
        raymap_plus_depth_plus_quats_adaptor = RayMapPlusDepthPlusQuatsAdaptor(
            name=f"{name}",
            ray_origins_mode=ray_origins_mode,
            ray_origins_vmin=ray_origins_vmin,
            ray_origins_vmax=ray_origins_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
            quaternions_mode=quaternions_mode,
            quaternions_normalize=quaternions_normalize,
            quaternions_vmin=quaternions_vmin,
            quaternions_vmax=quaternions_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(
            name,
            value_adaptor=raymap_plus_depth_plus_quats_adaptor,
            confidence_adaptor=confidence_adaptor,
            *args,
            **kwargs,
        )


class ValueWithMaskAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        value_adaptor: UniCeptionAdaptorBase,
        mask_adaptor: UniCeptionAdaptorBase,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Value with Mask head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            value_adaptor (UniCeptionAdaptorBase): Adaptor for the value.
            mask_adaptor (UniCeptionAdaptorBase): Adaptor for the mask.
        """

        super().__init__(
            name,
            required_channels=value_adaptor.required_channels + mask_adaptor.required_channels,
            *args,
            **kwargs,
        )

        self.value_adaptor = value_adaptor
        self.mask_adaptor = mask_adaptor

    def forward(self, adaptor_input: AdaptorInput):
        value_input, mask_input = torch.split(
            adaptor_input.adaptor_feature,
            [self.value_adaptor.required_channels, self.mask_adaptor.required_channels],
            dim=1,
        )
        value_adaptor_input = AdaptorInput(adaptor_feature=value_input, output_shape_hw=adaptor_input.output_shape_hw)
        mask_adaptor_input = AdaptorInput(adaptor_feature=mask_input, output_shape_hw=adaptor_input.output_shape_hw)
        value_output = self.value_adaptor(value_adaptor_input)
        mask_output = self.mask_adaptor(mask_adaptor_input)

        return RegressionWithMaskAdaptorOutput(
            value=value_output.value, mask=mask_output.mask, logits=mask_output.logits
        )


class PointMapWithMaskAdaptor(ValueWithMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Pointmap adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap with Confidence head in UniCeption.
        """
        pointmap_adaptor = PointMapAdaptor(name=f"{name}", mode=pointmap_mode, vmin=pointmap_vmin, vmax=pointmap_vmax)

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(name, value_adaptor=pointmap_adaptor, mask_adaptor=mask_adaptor, *args, **kwargs)


class PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor(ValueWithMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Point map adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap + RayDirections + Depth with Mask head in UniCeption.
        """
        pointmap_plus_ray_directions_plus_depth_adaptor = PointMapPlusRayDirectionsPlusDepthAdaptor(
            name=f"{name}",
            pointmap_mode=pointmap_mode,
            pointmap_vmin=pointmap_vmin,
            pointmap_vmax=pointmap_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=pointmap_plus_ray_directions_plus_depth_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class RayDirectionsPlusDepthWithMaskAdaptor(ValueWithMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth with Mask head in UniCeption.
        """
        ray_directions_plus_depth_adaptor = RayDirectionsPlusDepthAdaptor(
            name=f"{name}",
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name, value_adaptor=ray_directions_plus_depth_adaptor, mask_adaptor=mask_adaptor, *args, **kwargs
        )


class RayDirectionsPlusDepthPlusSceneFlowWithMaskAdaptor(ValueWithMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Scene flow adaptor
        scene_flow_mode: str,
        scene_flow_vmin: float,
        scene_flow_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth with Mask head in UniCeption.
        """
        ray_directions_plus_depth_plus_scene_flow_adaptor = RayDirectionsPlusDepthPlusSceneFlowAdaptor(
            name=f"{name}",
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
            scene_flow_mode=scene_flow_mode,
            scene_flow_vmin=scene_flow_vmin,
            scene_flow_vmax=scene_flow_vmax,
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=ray_directions_plus_depth_plus_scene_flow_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class RayMapPlusDepthWithMaskAdaptor(ValueWithMaskAdaptor):
    def __init__(
        self,
        name: str,
        # RayMap adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth with Mask head in UniCeption.
        """
        raymap_plus_depth_adaptor = RayMapPlusDepthAdaptor(
            name=f"{name}",
            ray_origins_mode=ray_origins_mode,
            ray_origins_vmin=ray_origins_vmin,
            ray_origins_vmax=ray_origins_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(name, value_adaptor=raymap_plus_depth_adaptor, mask_adaptor=mask_adaptor, *args, **kwargs)


class RayMapPlusDepthPlusQuatsWithMaskAdaptor(ValueWithMaskAdaptor):
    def __init__(
        self,
        name: str,
        # RayMap adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Quaternions adaptor
        quaternions_mode: str,
        quaternions_normalize: bool,
        quaternions_vmin: float,
        quaternions_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth + Quaternions with Mask head in UniCeption.
        """
        raymap_plus_depth_plus_quats_adaptor = RayMapPlusDepthPlusQuatsAdaptor(
            name=f"{name}",
            ray_origins_mode=ray_origins_mode,
            ray_origins_vmin=ray_origins_vmin,
            ray_origins_vmax=ray_origins_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
            quaternions_mode=quaternions_mode,
            quaternions_normalize=quaternions_normalize,
            quaternions_vmin=quaternions_vmin,
            quaternions_vmax=quaternions_vmax,
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name, value_adaptor=raymap_plus_depth_plus_quats_adaptor, mask_adaptor=mask_adaptor, *args, **kwargs
        )


class ValueWithConfidenceAndMaskAdaptor(UniCeptionAdaptorBase):
    def __init__(
        self,
        name: str,
        value_adaptor: UniCeptionAdaptorBase,
        confidence_adaptor: UniCeptionAdaptorBase,
        mask_adaptor: UniCeptionAdaptorBase,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the Value with Confidence & Mask head in UniCeption.

        Args:
            name (str): Name of the adaptor.
            value_adaptor (UniCeptionAdaptorBase): Adaptor for the value.
            mask_adaptor (UniCeptionAdaptorBase): Adaptor for the mask.
        """

        super().__init__(
            name,
            required_channels=value_adaptor.required_channels
            + confidence_adaptor.required_channels
            + mask_adaptor.required_channels,
            *args,
            **kwargs,
        )

        self.value_adaptor = value_adaptor
        self.confidence_adaptor = confidence_adaptor
        self.mask_adaptor = mask_adaptor

    def forward(self, adaptor_input: AdaptorInput):
        value_input, confidence_input, mask_input = torch.split(
            adaptor_input.adaptor_feature,
            [
                self.value_adaptor.required_channels,
                self.confidence_adaptor.required_channels,
                self.mask_adaptor.required_channels,
            ],
            dim=1,
        )
        value_adaptor_input = AdaptorInput(adaptor_feature=value_input, output_shape_hw=adaptor_input.output_shape_hw)
        confidence_adaptor_input = AdaptorInput(
            adaptor_feature=confidence_input, output_shape_hw=adaptor_input.output_shape_hw
        )
        mask_adaptor_input = AdaptorInput(adaptor_feature=mask_input, output_shape_hw=adaptor_input.output_shape_hw)
        value_output = self.value_adaptor(value_adaptor_input)
        confidence_output = self.confidence_adaptor(confidence_adaptor_input)
        mask_output = self.mask_adaptor(mask_adaptor_input)

        return RegressionWithConfidenceAndMaskAdaptorOutput(
            value=value_output.value,
            confidence=confidence_output.value,
            mask=mask_output.mask,
            logits=mask_output.logits,
        )


class PointMapWithConfidenceAndMaskAdaptor(ValueWithConfidenceAndMaskAdaptor):
    def __init__(
        self,
        name: str,
        # PointMap adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap with Confidence & Mask head in UniCeption.
        """
        pointmap_adaptor = PointMapAdaptor(name=f"{name}", mode=pointmap_mode, vmin=pointmap_vmin, vmax=pointmap_vmax)

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=pointmap_adaptor,
            confidence_adaptor=confidence_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(ValueWithConfidenceAndMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Point map adaptor
        pointmap_mode: str,
        pointmap_vmin: float,
        pointmap_vmax: float,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap + RayDirections + Depth with Confidence & Mask head in UniCeption.
        """
        pointmap_plus_ray_directions_plus_depth_adaptor = PointMapPlusRayDirectionsPlusDepthAdaptor(
            name=f"{name}",
            pointmap_mode=pointmap_mode,
            pointmap_vmin=pointmap_vmin,
            pointmap_vmax=pointmap_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=pointmap_plus_ray_directions_plus_depth_adaptor,
            confidence_adaptor=confidence_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(ValueWithConfidenceAndMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth with Confidence & Mask head in UniCeption.
        """
        ray_directions_plus_depth_adaptor = RayDirectionsPlusDepthAdaptor(
            name=f"{name}",
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=ray_directions_plus_depth_adaptor,
            confidence_adaptor=confidence_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class RayDirectionsPlusDepthPlusSceneFlowWithConfidenceAndMaskAdaptor(ValueWithConfidenceAndMaskAdaptor):
    def __init__(
        self,
        name: str,
        # Ray directions adaptor
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Scene flow adaptor
        scene_flow_mode: str,
        scene_flow_vmin: float,
        scene_flow_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayDirections + Depth with Confidence & Mask head in UniCeption.
        """
        ray_directions_plus_depth_plus_scene_flow_adaptor = RayDirectionsPlusDepthPlusSceneFlowAdaptor(
            name=f"{name}",
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
            scene_flow_mode=scene_flow_mode,
            scene_flow_vmin=scene_flow_vmin,
            scene_flow_vmax=scene_flow_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=ray_directions_plus_depth_plus_scene_flow_adaptor,
            confidence_adaptor=confidence_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class RayMapPlusDepthWithConfidenceAndMaskAdaptor(ValueWithConfidenceAndMaskAdaptor):
    def __init__(
        self,
        name: str,
        # RayMap adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth with Confidence & Mask head in UniCeption.
        """
        raymap_plus_depth_adaptor = RayMapPlusDepthAdaptor(
            name=f"{name}",
            ray_origins_mode=ray_origins_mode,
            ray_origins_vmin=ray_origins_vmin,
            ray_origins_vmax=ray_origins_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=raymap_plus_depth_adaptor,
            confidence_adaptor=confidence_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )


class RayMapPlusDepthPlusQuatsWithConfidenceAndMaskAdaptor(ValueWithConfidenceAndMaskAdaptor):
    def __init__(
        self,
        name: str,
        # RayMap adaptor
        ray_origins_mode: str,
        ray_origins_vmin: float,
        ray_origins_vmax: float,
        ray_directions_mode: str,
        ray_directions_normalize_to_unit_sphere: bool,
        ray_directions_normalize_to_unit_image_plane: bool,
        ray_directions_vmin: float,
        ray_directions_vmax: float,
        ray_directions_clamp_min_of_z_dir: bool,
        ray_directions_z_dir_min: float,
        # Depth adaptor
        depth_mode: str,
        depth_vmin: float,
        depth_vmax: float,
        # Quaternions adaptor
        quaternions_mode: str,
        quaternions_normalize: bool,
        quaternions_vmin: float,
        quaternions_vmax: float,
        # Confidence adaptor
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the RayMap (RayOrigins + RayDirections) + Depth + Quaternions with Confidence & Mask head in UniCeption.
        """
        raymap_plus_depth_plus_quats_adaptor = RayMapPlusDepthPlusQuatsAdaptor(
            name=f"{name}",
            ray_origins_mode=ray_origins_mode,
            ray_origins_vmin=ray_origins_vmin,
            ray_origins_vmax=ray_origins_vmax,
            ray_directions_mode=ray_directions_mode,
            ray_directions_normalize_to_unit_sphere=ray_directions_normalize_to_unit_sphere,
            ray_directions_normalize_to_unit_image_plane=ray_directions_normalize_to_unit_image_plane,
            ray_directions_vmin=ray_directions_vmin,
            ray_directions_vmax=ray_directions_vmax,
            ray_directions_clamp_min_of_z_dir=ray_directions_clamp_min_of_z_dir,
            ray_directions_z_dir_min=ray_directions_z_dir_min,
            depth_mode=depth_mode,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
            quaternions_mode=quaternions_mode,
            quaternions_normalize=quaternions_normalize,
            quaternions_vmin=quaternions_vmin,
            quaternions_vmax=quaternions_vmax,
        )

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        mask_adaptor = MaskAdaptor(name=f"{name}_mask")

        super().__init__(
            name,
            value_adaptor=raymap_plus_depth_plus_quats_adaptor,
            confidence_adaptor=confidence_adaptor,
            mask_adaptor=mask_adaptor,
            *args,
            **kwargs,
        )
