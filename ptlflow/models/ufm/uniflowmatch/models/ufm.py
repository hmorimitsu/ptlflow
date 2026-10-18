"""
UFM (Unified Flow & Matching) model for dense correspondence prediction between two views.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ...uniception.models.encoders import ViTEncoderInput, feature_returner_encoder_factory
from ...uniception.models.info_sharing import INFO_SHARING_CLASSES, MultiViewTransformerInput
from ...uniception.models.prediction_heads.adaptors import (
    ConfidenceAdaptor,
    Covariance2DAdaptor,
    FlowAdaptor,
    FlowWithConfidenceAdaptor,
    MaskAdaptor,
)
from ...uniception.models.prediction_heads.base import AdaptorMap, PredictionHeadInput, PredictionHeadLayeredInput
from ...uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from ...uniception.models.prediction_heads.mlp_feature import MLPFeature
from ...uniception.models.prediction_heads.moge_conv import MoGeConvFeature

from .base import (
    UFMClassificationRefinementOutput,
    UFMFlowFieldOutput,
    UFMMaskFieldOutput,
    UFMOutputInterface,
    UniFlowMatchModelsBase,
)
from .unet_encoder import UNet
from .utils import get_meshgrid_torch

CLASSNAME_TO_ADAPTOR_CLASS = {
    "FlowWithConfidenceAdaptor": FlowWithConfidenceAdaptor,
    "FlowAdaptor": FlowAdaptor,
    "MaskAdaptor": MaskAdaptor,
    "Covariance2DAdaptor": Covariance2DAdaptor,
    "ConfidenceAdaptor": ConfidenceAdaptor,
}


def is_symmetrized(gt1, gt2):
    """
    Function to check if input pairs are symmetrized, i.e., (a, b) and (b, a) always exist in the input.
    Reference: DUSt3R

    Args:
        gt1 (Dict[str, Any]): First ground truth data containing 'instance' key
        gt2 (Dict[str, Any]): Second ground truth data containing 'instance' key

    Returns:
        bool: True if the input pairs are symmetrized, False otherwise
    """
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])

    return ok


def interleave(tensor1, tensor2):
    """
    Interleave two tensors along the first dimension (used to avoid redundant encoding for symmetrized pairs).

    Args:
        tensor1 (torch.Tensor): First input tensor
        tensor2 (torch.Tensor): Second input tensor

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two interleaved tensors (tensor1, tensor2) and (tensor2, tensor1)
    """
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def modify_state_dict(original_state_dict, mappings):
    """
    Modify state dict keys according to replacement mappings

    Args:
        original_state_dict: Loaded checkpoint state dict
        mappings: Dictionary of {old_key_substr: new_key_substr_or_None}

    Returns:
        Modified state dictionary with updated keys
    """
    new_state_dict = {}

    for k, v in original_state_dict.items():
        new_key = None
        skip = False

        # Check for all possible replacements
        for replace_key, replace_value in mappings.items():
            if replace_key in k:
                if replace_value is None:
                    skip = True
                    break  # Skip this key entirely
                else:
                    new_key = k.replace(replace_key, replace_value)
                    break  # Only apply first matching replacement

        if skip:
            continue

        new_state_dict[new_key if new_key is not None else k] = v

    return new_state_dict


class UniFlowMatch(UniFlowMatchModelsBase):
    """
    UniFlowMatch model for dense correspondence prediction between two views.

    This is the base model that performs multi-view feature extraction, information sharing,
    and correspondence prediction between image pairs. It combines a feature encoder with
    an information sharing module and prediction heads to output dense flow fields and
    optional covisibility masks.
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any],
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",  # only dual+single is supported
        # Information sharing configurations
        info_sharing_str: str = "global_attention",
        info_sharing_kwargs: Dict[str, Any] = {},
        # Skip-connections between encoder and info-sharing
        encoder_skip_connection: Optional[List[int]] = None,
        info_sharing_skip_connection: Optional[List[int]] = None,
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Load Pretrained Weights
        pretrained_checkpoint_path: Optional[str] = None,
        # Inference Settings
        inference_resolution: Optional[Tuple[int, int]] = (560, 420),  # (W, H)
        *args,
        **kwargs,
    ):
        """
        Initialize the UniFlowMatch Model.

        Args:
            encoder_str (str): Encoder identifier string
            encoder_kwargs (Dict[str, Any]): Encoder configuration parameters
            info_sharing_and_head_structure (str): Info sharing and head structure type.
                Currently only "dual+single" is supported (dual view info sharing and single view prediction head)
            info_sharing_str (str): Info sharing method identifier.
                Supported: "global_attention_transformer", "global_attention"
            info_sharing_kwargs (Dict[str, Any]): Info sharing module configuration parameters
            encoder_skip_connection (Optional[List[int]]): Skip connection indices from encoder
            info_sharing_skip_connection (Optional[List[int]]): Skip connection indices from info sharing
            head_type (str): Prediction head type. Supported: "dpt", "moge_conv"
            feature_head_kwargs (Dict[str, Any]): Feature head configuration parameters
            adaptors_kwargs (Dict[str, Any]): Adaptors configuration parameters
            pretrained_checkpoint_path (Optional[str]): Path to pretrained model checkpoint
            inference_resolution (Optional[Tuple[int, int]]): Target resolution for inference (W, H)
            *args: Additional arguments passed to parent classes
            **kwargs: Additional keyword arguments passed to parent classes
        """

        # minor adjustment from V0.1.0 uniCeption to V0.6.0 UniCeption
        info_sharing_replacement_dict = {
            "max_num_views": "max_num_views_for_pe",
            "pretrained_checkpoint_path": None
        }

        info_sharing_kwargs = {info_sharing_replacement_dict.get(k, k): v for k, v in info_sharing_kwargs.items() if info_sharing_replacement_dict.get(k, k) is not None}


        UniFlowMatchModelsBase.__init__(self, inference_resolution=inference_resolution, *args, **kwargs)

        # assertion on architectures
        assert info_sharing_and_head_structure == "dual+single", "Only dual+single is supported now"

        # initialize the skip-connections
        self.encoder_skip_connection = encoder_skip_connection
        self.info_sharing_skip_connection = info_sharing_skip_connection

        # initialize encoder
        self.encoder: nn.Module = feature_returner_encoder_factory(encoder_str, **encoder_kwargs)

        # initialize info-sharing module
        assert head_type != "linear", "Linear head is not supported, because it have major disadvantage to DPTs"
        self.head_type = head_type

        self.info_sharing: nn.Module = INFO_SHARING_CLASSES[info_sharing_str][1](**info_sharing_kwargs)

        self.head1: nn.Module = self._initialize_prediction_heads(head_type, feature_head_kwargs, adaptors_kwargs)

        # load pretrained weights
        if pretrained_checkpoint_path is not None:
            ckpt = torch.load(pretrained_checkpoint_path, map_location="cpu")

            if "state_dict" in ckpt:
                # we are loading from training checkpoint directly.
                model_state_dict = ckpt["state_dict"]
                model_state_dict = {
                    k[6:]: v for k, v in model_state_dict.items() if k.startswith("model.")
                }  # remove "model." prefix

                model_state_dict = modify_state_dict(
                    model_state_dict, {"feature_matching_proj": None, "encoder.model.mask_token": None}
                )

                self.load_state_dict(model_state_dict, strict=True)
            else:
                model_state_dict = ckpt["model"]

                load_result = self.load_state_dict(model_state_dict, strict=False)
                assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"

    @classmethod
    def from_pretrained_ckpt(cls, pretrained_model_name_or_path, strict=True, **kw):
        """
        Load a model from a pretrained checkpoint file.

        Args:
            pretrained_model_name_or_path (str): Path to the pretrained model checkpoint file
            strict (bool): Whether to strictly enforce that the keys in state_dict match exactly
            **kw: Additional keyword arguments

        Returns:
            UniFlowMatch: Loaded model instance

        Raises:
            ValueError: If the pretrained model file is not found
        """
        if os.path.isfile(pretrained_model_name_or_path):
            ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
            model = cls(**ckpt["model_args"])
            model.load_state_dict(ckpt["model"], strict=strict)
            return model
        else:
            raise ValueError(f"Pretrained model {pretrained_model_name_or_path} not found.")

    def _initialize_prediction_heads(
        self, head_type: str, feature_head_kwargs: Dict[str, Any], adaptors_kwargs: Dict[str, Any]
    ):
        """
        Initialize prediction heads and adaptors.

        Args:
            head_type (str): Head type, supported: "dpt", "moge_conv"
            feature_head_kwargs (Dict[str, Any]): Feature head configuration parameters
            adaptors_kwargs (Dict[str, Any]): Adaptors configuration parameters

        Returns:
            nn.Module: Sequential module containing feature processor and adaptor map

        Raises:
            ValueError: If unsupported head type is specified
        """
        feature_processor: nn.Module
        if head_type == "dpt":
            feature_processor = nn.Sequential(
                DPTFeature(**feature_head_kwargs["dpt_feature"]),
                DPTRegressionProcessor(**feature_head_kwargs["dpt_processor"]),
            )
        elif head_type == "moge_conv":
            feature_processor = MoGeConvFeature(**feature_head_kwargs)
        else:
            raise ValueError(f"Head type {head_type} not supported.")

        adaptors = self._initialize_adaptors(adaptors_kwargs)

        return nn.Sequential(feature_processor, AdaptorMap(*adaptors.values()))

    def _initialize_adaptors(self, adaptors_kwargs: Dict[str, Any]):
        """
        Initialize a dictionary of adaptors.

        Args:
            adaptors_kwargs (Dict[str, Any]): Adaptors configuration parameters.
                Each key is an adaptor name, and each value contains 'class' and 'kwargs' keys.

        Returns:
            Dict[str, nn.Module]: Dictionary mapping adaptor names to their corresponding adaptor modules
        """
        return {
            name: CLASSNAME_TO_ADAPTOR_CLASS[configs["class"]](**configs["kwargs"])
            for name, configs in adaptors_kwargs.items()
        }

    def _encode_image_pairs(self, img1, img2, data_norm_type):
        """
        Encode two different batches of images (each batch can have different image shape).

        Args:
            img1 (torch.Tensor): First batch of images with shape (B, C, H, W)
            img2 (torch.Tensor): Second batch of images with shape (B, C, H, W)
            data_norm_type (str): Data normalization type for the encoder

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Two lists of encoded features,
                one for each image batch

        Raises:
            NotImplementedError: If the images have different sizes
        """
        if img1.shape[-2:] == img2.shape[-2:]:
            encoder_input = ViTEncoderInput(image=torch.cat((img1, img2), dim=0), data_norm_type=data_norm_type)
            encoder_output = self.encoder(encoder_input)
            out_list, out2_list = [], []

            for encoder_output_ in encoder_output:
                out, out2 = encoder_output_.features.chunk(2, dim=0)
                out_list.append(out)
                out2_list.append(out2)
        else:
            raise NotImplementedError("Unequal Image sizes are not supported now")

        return out_list, out2_list

    def _encode_symmetrized(self, view1, view2, symmetrized=False):
        """
        Encode image pairs accounting for symmetrization, i.e., (a, b) and (b, a) always exist in the input.

        Args:
            view1 (Dict[str, Any]): First view containing 'img' and 'data_norm_type' keys
            view2 (Dict[str, Any]): Second view containing 'img' and 'data_norm_type' keys
            symmetrized (bool): Whether the input pairs are symmetrized

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Encoded features for both views
        """
        img1 = view1["img"]
        img2 = view2["img"]

        feat1_list, feat2_list = [], []

        if symmetrized:
            # Computing half of forward pass'
            # modified in conjunction with UFM for not copying the images again.
            # used to be: feat1, feat2 = self._encode_image_pairs(img1[::2], img2[::2], data_norm_type=view1["data_norm_type"])
            # be very carefult with this!!!
            feat1_list_, feat2_list_ = self._encode_image_pairs(
                img1[::2], img2[::2], data_norm_type=view1["data_norm_type"]
            )

            for feat1, feat2 in zip(feat1_list_, feat2_list_):
                feat1, feat2 = interleave(feat1, feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
        else:
            feat1_list, feat2_list = self._encode_image_pairs(img1, img2, data_norm_type=view1["data_norm_type"])

        return feat1_list, feat2_list

    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks.

        Args:
            view1 (Dict[str, Any]): Input view 1 containing:
                - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
                - instance (List[int]): List of instance indices, or id of the input image
                - data_norm_type (str): Data normalization type, see uniception.models.encoders.IMAGE_NORMALIZATION_DICT
                - symmetrized (bool): Whether input pairs are symmetrized
            view2 (Dict[str, Any]): Input view 2 with same structure as view1

        Returns:
        - UFMOutputInterface (Dict[str, Any]): Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - covisibility [Optional] (Dict[str, torch.Tensor]): Covisibility output
            - [Optional] mask
            - [Optional] logits
        """

        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1_list, feat2_list = self._encode_symmetrized(view1, view2, view1["symmetrized"])

        # Pass the features through the info_sharing
        info_sharing_input = MultiViewTransformerInput(features=[feat1_list[-1], feat2_list[-1]])

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
            info_sharing_input
        )

        info_sharing_outputs = {
            "1": [
                feat1_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].float().contiguous(),
            ],
            "2": [
                feat2_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].float().contiguous(),
            ],
        }

        result = UFMOutputInterface()

        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", torch.float32):
            # run the collected info_sharing features through the prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)

            if "flow" in head_output1:
                # output is flow only
                result.flow = UFMFlowFieldOutput(flow_output=head_output1["flow"].value)

            if "flow_cov" in head_output1:
                result.flow.flow_covariance = head_output1["flow_cov"].covariance
                result.flow.flow_covariance_inv = head_output1["flow_cov"].inv_covariance
                result.flow.flow_covariance_log_det = head_output1["flow_cov"].log_det

            if "non_occluded_mask" in head_output1:
                result.covisibility = UFMMaskFieldOutput(
                    mask=head_output1["non_occluded_mask"].mask,
                    logits=head_output1["non_occluded_mask"].logits,
                )

        return result

    def _downstream_head(self, head_num, decout, img_shape):
        """
        Run the respective prediction heads.

        Args:
            head_num (Union[int, str]): Head number identifier (1)
            decout (Dict[str, List[torch.Tensor]]): Decoded features from info sharing
            img_shape (Tuple[int, int]): Target image shape (H, W)

        Returns:
            Dict[str, Any]: Head output containing predictions from adaptors
        """
        head = getattr(self, f"head{head_num}")
        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type in ["dpt", "moge_conv"]:
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This method guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
            Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer with keys:
                - "encoder": Parameters from the feature encoder
                - "info_sharing": Parameters from the information sharing module
                - "output_head": Parameters from the prediction head
        """

        return {
            "encoder": torch.nn.ParameterList(self.encoder.parameters()),
            "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
            "output_head": torch.nn.ParameterList(self.head1.parameters()),
        }


class UniFlowMatchConfidence(UniFlowMatch):
    """
    UniFlowMatch model with uncertainty estimation capabilities.

    This extends the base UniFlowMatch model by adding an uncertainty estimation head
    that can predict flow covariance, keypoint confidence, and other uncertainty-related
    outputs alongside the main flow predictions.
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any],
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",  # only dual+single is supported
        # Information sharing configurations
        info_sharing_str: str = "global_attention",
        info_sharing_kwargs: Dict[str, Any] = {},
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Uncertainty Heads & Adaptors
        detach_uncertainty_head: bool = True,
        uncertainty_head_type: str = "dpt",
        uncertainty_head_kwargs: Dict[str, Any] = {},
        uncertainty_adaptors_kwargs: Dict[str, Any] = {},
        # Load Pretrained Weights
        pretrained_backbone_checkpoint_path: Optional[str] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        # Inference Settings
        inference_resolution: Optional[Tuple[int, int]] = (560, 420),  # WH
        *args,
        **kwargs,
    ):
        """
        Initialize the UniFlowMatchConfidence model.

        Args:
            encoder_str (str): Encoder identifier string
            encoder_kwargs (Dict[str, Any]): Encoder configuration parameters
            info_sharing_and_head_structure (str): Info sharing structure, only "dual+single" supported
            info_sharing_str (str): Info sharing method identifier
            info_sharing_kwargs (Dict[str, Any]): Info sharing configuration parameters
            head_type (str): Main prediction head type
            feature_head_kwargs (Dict[str, Any]): Main feature head configuration
            adaptors_kwargs (Dict[str, Any]): Main adaptors configuration
            detach_uncertainty_head (bool): Whether to detach gradients for uncertainty head
            uncertainty_head_type (str): Uncertainty head type, currently only "dpt" supported
            uncertainty_head_kwargs (Dict[str, Any]): Uncertainty head configuration
            uncertainty_adaptors_kwargs (Dict[str, Any]): Uncertainty adaptors configuration
            pretrained_backbone_checkpoint_path (Optional[str]): Path to pretrained backbone checkpoint
            pretrained_checkpoint_path (Optional[str]): Path to pretrained full model checkpoint
            inference_resolution (Optional[Tuple[int, int]]): Target resolution for inference (W, H)
            *args: Additional arguments passed to parent classes
            **kwargs: Additional keyword arguments passed to parent classes
        """
        UniFlowMatch.__init__(
            self,
            encoder_str=encoder_str,
            encoder_kwargs=encoder_kwargs,
            info_sharing_and_head_structure=info_sharing_and_head_structure,
            info_sharing_str=info_sharing_str,
            info_sharing_kwargs=info_sharing_kwargs,
            head_type=head_type,
            feature_head_kwargs=feature_head_kwargs,
            adaptors_kwargs=adaptors_kwargs,
            pretrained_checkpoint_path=pretrained_backbone_checkpoint_path,
            inference_resolution=inference_resolution,
            *args,
            **kwargs,
        )

        # initialize uncertainty heads
        assert uncertainty_head_type == "dpt", "Only DPT is supported for uncertainty head now"

        self.uncertainty_head = self._initialize_prediction_heads(
            uncertainty_head_type, uncertainty_head_kwargs, uncertainty_adaptors_kwargs
        )
        self.uncertainty_adaptors = self._initialize_adaptors(uncertainty_adaptors_kwargs)

        assert pretrained_checkpoint_path is None, "Pretrained weights are not supported for now"

        self.detach_uncertainty_head = detach_uncertainty_head

    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks with uncertainty estimation.

        Args:
            view1 (Dict[str, Any]): Input view 1 containing:
                - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
                - instance (List[int]): List of instance indices, or id of the input image
                - data_norm_type (str): Data normalization type
                - symmetrized (bool): Whether input pairs are symmetrized
            view2 (Dict[str, Any]): Input view 2 with same structure as view1

        Returns:
        - UFMOutputInterace (Dict[str, Any]): Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - covisibility [Optional] (Dict[str, torch.Tensor]): Covisibiltiy output
            - [Optional] mask
            - [Optional] logits
        """

        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1_list, feat2_list = self._encode_symmetrized(view1, view2, view1["symmetrized"])

        # Pass the features through the info_sharing
        info_sharing_input = MultiViewTransformerInput(features=[feat1_list[-1], feat2_list[-1]])

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
            info_sharing_input
        )

        info_sharing_outputs = {
            "1": [
                feat1_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].float().contiguous(),
            ],
            "2": [
                feat2_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].float().contiguous(),
            ],
        }

        info_sharing_outputs_detached = {
            "1": [
                feat1_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].detach().float().contiguous(),
            ],
            "2": [
                feat2_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].detach().float().contiguous(),
            ],
        }

        result = UFMOutputInterface()

        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", torch.float32):
            # run the collected info_sharing features through the prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
            head_output_uncertainty = self._downstream_head(
                "uncertainty",
                info_sharing_outputs_detached if self.detach_uncertainty_head else info_sharing_outputs,
                shape1,
            )

            result.flow = UFMFlowFieldOutput(
                flow_output=head_output1["flow"].value,
            )

            if "flow_cov" in head_output_uncertainty:
                result.flow.flow_covariance = head_output_uncertainty["flow_cov"].covariance
                result.flow.flow_covariance_inv = head_output_uncertainty["flow_cov"].inv_covariance
                result.flow.flow_covariance_log_det = head_output_uncertainty["flow_cov"].log_det

            if "keypoint_confidence" in head_output_uncertainty:
                result.keypoint_confidence = head_output_uncertainty["keypoint_confidence"].value.squeeze(1)

            if "non_occluded_mask" in head_output_uncertainty:
                result.covisibility = UFMMaskFieldOutput(
                    mask=head_output_uncertainty["non_occluded_mask"].mask,
                    logits=head_output_uncertainty["non_occluded_mask"].logits,
                )

        return result

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This method guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
            Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer with keys:
                - "encoder": Parameters from the feature encoder
                - "info_sharing": Parameters from the information sharing module
                - "output_head": Parameters from the main prediction head
                - "uncertainty_head": Parameters from the uncertainty estimation head
        """

        return {
            "encoder": torch.nn.ParameterList(self.encoder.parameters()),
            "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
            "output_head": torch.nn.ParameterList(self.head1.parameters()),
            "uncertainty_head": torch.nn.ParameterList(self.uncertainty_head.parameters()),
        }

    def _downstream_head(self, head_num, decout, img_shape):
        """
        Run the respective prediction heads (main or uncertainty).

        Args:
            head_num (Union[int, str]): Head number identifier (1) or "uncertainty" for uncertainty head
            decout (Dict[str, List[torch.Tensor]]): Decoded features from info sharing
            img_shape (Tuple[int, int]): Target image shape (H, W)

        Returns:
            Dict[str, Any]: Head output containing predictions from adaptors
        """
        # if self.info_sharing_and_head_structure == "dual+single":

        head = getattr(self, f"head{head_num}") if head_num != "uncertainty" else self.uncertainty_head

        head_num = head_num if head_num != "uncertainty" else 1  # uncertainty head is always from branch 1

        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type in ["dpt", "moge_conv"]:
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)


class UniFlowMatchClassificationRefinement(UniFlowMatch):
    """
    The variant of UniFlowMatch with local classification-based refinement.

    This model extends UniFlowMatch by adding a classification head that performs
    local refinement of flow predictions using attention-based mechanisms. It can
    also optionally include uncertainty estimation and U-Net features for enhanced
    performance.
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any],
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",  # only dual+single is supported
        # Information sharing configurations
        info_sharing_str: str = "global_attention",
        info_sharing_kwargs: Dict[str, Any] = {},
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Uncertainty Heads & Adaptors
        detach_uncertainty_head: bool = True,
        uncertainty_head_type: str = "dpt",
        uncertainty_head_kwargs: Dict[str, Any] = {},
        uncertainty_adaptors_kwargs: Dict[str, Any] = {},
        # Classification Heads & Adaptors
        temperature: float = 4,
        use_unet_feature: bool = False,
        use_unet_batchnorm: bool = False,
        classification_head_type: str = "patch_mlp",
        classification_head_kwargs: Dict[str, Any] = {},
        feature_combine_method: str = "conv",
        # Refinement Range
        refinement_range: int = 5,
        # Load Pretrained Weights
        pretrained_backbone_checkpoint_path: Optional[str] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        # Inference Settings
        inference_resolution: Optional[Tuple[int, int]] = (560, 420),  # WH
        *args,
        **kwargs,
    ):
        """
        Initialize the UniFlowMatchClassificationRefinement model.

        Args:
            encoder_str (str): Encoder identifier string
            encoder_kwargs (Dict[str, Any]): Encoder configuration parameters
            info_sharing_and_head_structure (str): Info sharing structure, only "dual+single" supported
            info_sharing_str (str): Info sharing method identifier
            info_sharing_kwargs (Dict[str, Any]): Info sharing configuration parameters
            head_type (str): Main prediction head type
            feature_head_kwargs (Dict[str, Any]): Main feature head configuration
            adaptors_kwargs (Dict[str, Any]): Main adaptors configuration
            detach_uncertainty_head (bool): Whether to detach gradients for uncertainty head
            uncertainty_head_type (str): Uncertainty head type, currently only "dpt" supported
            uncertainty_head_kwargs (Dict[str, Any]): Uncertainty head configuration
            uncertainty_adaptors_kwargs (Dict[str, Any]): Uncertainty adaptors configuration
            temperature (float): Temperature parameter for attention softmax in classification refinement
            use_unet_feature (bool): Whether to use additional U-Net features for refinement
            use_unet_batchnorm (bool): Whether to use batch normalization in U-Net, helps with convergence
            classification_head_type (str): Classification head type, currently only "patch_mlp" supported
            classification_head_kwargs (Dict[str, Any]): Classification head configuration
            feature_combine_method (str): Method for combining features ("conv" or "modulate")
            refinement_range (int): Local patch size for refinement (must be odd)
            pretrained_backbone_checkpoint_path (Optional[str]): Path to pretrained backbone checkpoint
            pretrained_checkpoint_path (Optional[str]): Path to pretrained full model checkpoint
            inference_resolution (Optional[Tuple[int, int]]): Target resolution for inference (W, H)
            *args: Additional arguments passed to parent classes
            **kwargs: Additional keyword arguments passed to parent classes
        """
        UniFlowMatch.__init__(
            self,
            encoder_str=encoder_str,
            encoder_kwargs=encoder_kwargs,
            info_sharing_and_head_structure=info_sharing_and_head_structure,
            info_sharing_str=info_sharing_str,
            info_sharing_kwargs=info_sharing_kwargs,
            head_type=head_type,
            feature_head_kwargs=feature_head_kwargs,
            adaptors_kwargs=adaptors_kwargs,
            pretrained_checkpoint_path=pretrained_backbone_checkpoint_path,
            inference_resolution=inference_resolution,
            *args,
            **kwargs,
        )

        # initialize uncertainty heads
        assert classification_head_type == "patch_mlp", "Only DPT is supported for uncertainty head now"
        self.classification_head_type = classification_head_type

        self.classification_head = self._initialize_classification_head(classification_head_kwargs)

        self.refinement_range = refinement_range
        self.temperature = temperature

        assert pretrained_checkpoint_path is None, "Pretrained weights are not supported for now"

        self.use_unet_feature = use_unet_feature

        self.feature_combine_method = feature_combine_method

        # Unet experiment
        if self.use_unet_feature:
            self.use_unet_batchnorm = use_unet_batchnorm
            self.unet_feature = UNet(
                in_channels=3, out_channels=16, features=[64, 128, 256, 512], use_batch_norm=self.use_unet_batchnorm
            )

            self.conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

            if self.feature_combine_method == "conv":
                self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
            elif self.feature_combine_method == "modulate":
                self.conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)

        default_attention_bias = torch.zeros(self.refinement_range * self.refinement_range)
        self.classification_bias = nn.Parameter(default_attention_bias)

        # initialize uncertainty heads
        if len(uncertainty_head_kwargs) > 0:
            assert uncertainty_head_type == "dpt", "Only DPT is supported for uncertainty head now"

            self.uncertainty_head = self._initialize_prediction_heads(
                uncertainty_head_type, uncertainty_head_kwargs, uncertainty_adaptors_kwargs
            )
            self.uncertainty_adaptors = self._initialize_adaptors(uncertainty_adaptors_kwargs)

            assert pretrained_checkpoint_path is None, "Pretrained weights are not supported for now"

            self.detach_uncertainty_head = detach_uncertainty_head

    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks with classification-based refinement.

        Args:
            view1 (Dict[str, Any]): Input view 1 containing:
                - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
                - instance (List[int]): List of instance indices, or id of the input image
                - data_norm_type (str): Data normalization type
                - symmetrized (bool): Whether input pairs are symmetrized
            view2 (Dict[str, Any]): Input view 2 with same structure as view1

        Returns:
        - Dict[str, Any]: Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - covisibility [Optional] (Dict[str, torch.Tensor]): Covisibility output
            - [Optional] mask
            - [Optional] logits
          - classification [Optional]: Probability and targets of the classification head
        """

        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1_list, feat2_list = self._encode_symmetrized(view1, view2, view1["symmetrized"])

        # Pass the features through the info_sharing
        info_sharing_input = MultiViewTransformerInput(features=[feat1_list[-1], feat2_list[-1]])

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
            info_sharing_input
        )

        info_sharing_outputs = {
            "1": [
                feat1_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].float().contiguous(),
            ],
            "2": [
                feat2_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].float().contiguous(),
            ],
        }

        info_sharing_outputs_detached = {
            "1": [
                feat1_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].detach().float().contiguous(),
            ],
            "2": [
                feat2_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].detach().float().contiguous(),
            ],
        }

        # optionally inference for U-Net Features
        if self.use_unet_feature:
            unet_feat1 = self.unet_feature(view1["img"])
            unet_feat2 = self.unet_feature(view2["img"])

        result = UFMOutputInterface()
        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", torch.float32):
            # run the collected info_sharing features through the prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
            flow_prediction = head_output1["flow"].value

            if hasattr(self, "uncertainty_head"):
                # run the uncertainty head
                head_output_uncertainty = self._downstream_head(
                    "uncertainty",
                    info_sharing_outputs_detached if self.detach_uncertainty_head else info_sharing_outputs,
                    shape1,
                )

                if "flow_cov" in head_output_uncertainty:
                    result.flow.flow_covariance = head_output_uncertainty["flow_cov"].covariance
                    result.flow.flow_covariance_inv = head_output_uncertainty["flow_cov"].inv_covariance
                    result.flow.flow_covariance_log_det = head_output_uncertainty["flow_cov"].log_det

                if "keypoint_confidence" in head_output_uncertainty:
                    result.keypoint_confidence = head_output_uncertainty["keypoint_confidence"].value.squeeze(1)

                if "non_occluded_mask" in head_output_uncertainty:
                    result.covisibility = UFMMaskFieldOutput(
                        mask=head_output_uncertainty["non_occluded_mask"].mask,
                        logits=head_output_uncertainty["non_occluded_mask"].logits,
                    )

            # we run the classification head in the autocast environment bacause it is not regression
            if self.classification_head_type == "patch_mlp":
                # concatenate the last encoder feature with final info_sharing feature

                # use the first encoder feature, because it captures more low-level information, which is needed
                # for refinement of the regressed flow.
                classification_feat_1 = torch.cat(
                    [feat1_list[0].float().contiguous(), info_sharing_outputs["1"][-1]], dim=1
                )
                classification_feat_2 = torch.cat(
                    [feat2_list[0].float().contiguous(), info_sharing_outputs["2"][-1]], dim=1
                )

                classification_input = PredictionHeadInput(
                    torch.cat([classification_feat_1, classification_feat_2], dim=0)
                )

                classification_features = self.classification_head(classification_input).decoded_channels

                if self.use_unet_feature:
                    if self.feature_combine_method == "conv":
                        combined_features = torch.cat(
                            [classification_features, torch.cat([unet_feat1, unet_feat2], dim=0)], dim=1
                        )

                        combined_features = self.conv1(combined_features)
                        combined_features = nn.functional.relu(combined_features)
                        combined_features = self.conv2(combined_features)
                    elif self.feature_combine_method == "modulate":
                        combined_features = classification_features * torch.tanh(
                            torch.cat([unet_feat1, unet_feat2], dim=0)
                        )
                        combined_features = self.conv2(combined_features)

                    classification_features = combined_features

                classification_features0, classification_features1 = classification_features.chunk(2, dim=0)

            # refine the flow prediction with features from the classification head
            for i in range(1):
                residual, log_softmax_attention = self.classification_refinement(
                    flow_prediction, classification_features
                )
                flow_prediction = flow_prediction + residual

        # Fill in the result
        # WARNING: based on how the residual is computed, flow_prediction will have gradient cancelled by mathematics,
        # so there will be no supervision to the flow prediction at all. We need to use specialized loss function to
        # supervise the regression_flow_output.
        result.flow = UFMFlowFieldOutput(
            flow_output=flow_prediction,
        )

        result.classification_refinement = UFMClassificationRefinementOutput(
            regression_flow_output=flow_prediction,
            residual=residual,
            log_softmax=log_softmax_attention,
            feature_map_0=classification_features0,
            feature_map_1=classification_features1,
        )

        return result

    # @torch.compile()
    def classification_refinement(self, flow_prediction, classification_features) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use correlation between self feature and features around a local patch of the initial flow prediction
        to refine the flow prediction using attention-based mechanisms.

        Args:
            flow_prediction (torch.Tensor): Initial flow prediction tensor with shape (B, 2, H, W)
            classification_features (torch.Tensor): Classification features tensor with shape (2*B, C, H, W)
                where the first B samples correspond to view1 and the second B samples to view2

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - residual (torch.Tensor): Flow residual to be added to initial prediction, shape (B, 2, H, W)
                - log_softmax_attention (torch.Tensor): Log softmax attention weights, shape (B, H, W, P, P)
                  where P is the refinement_range
        """

        classification_features1, classification_features2 = classification_features.chunk(2, dim=0)

        neighborhood_features, neighborhood_flow_residual = self.obtain_neighborhood_features(
            flow_estimation=flow_prediction, other_features=classification_features2, local_patch=self.refinement_range
        )

        residual, log_softmax_attention = self.compute_refinement_attention(
            classification_features1, neighborhood_features, neighborhood_flow_residual
        )

        return residual, log_softmax_attention

    def compute_refinement_attention(self, classification_features1, neighborhood_features, neighborhood_flow_residual):
        """
        Compute the attention weights for flow refinement using feature correlation.

        This method performs attention computation between query features (classification_features1)
        and key features (neighborhood_features) to generate attention weights for flow residual values.

        Args:
            classification_features1 (torch.Tensor): Query features from view1 with shape (B, C, H, W)
            neighborhood_features (torch.Tensor): Key features from neighborhood sampling with shape (B, H, W, P, P, C)
                where P is the refinement_range
            neighborhood_flow_residual (torch.Tensor): Flow residual values for each neighborhood position
                with shape (B, H, W, P, P, 2)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - residual (torch.Tensor): Weighted flow residual with shape (B, 2, H, W)
                - log_softmax_attention (torch.Tensor): Log softmax attention weights with shape (B, H, W, P, P)
        """

        B, C, H, W = classification_features1.shape
        P = self.refinement_range

        # reshape Q to B, H, W, 1, 1, C
        classification_features1 = classification_features1.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)

        # reshape K to B, H, W, 1, P^2, C
        assert neighborhood_features.shape[0] == B
        assert neighborhood_features.shape[1] == H
        assert neighborhood_features.shape[2] == W
        assert neighborhood_features.shape[3] == P
        assert neighborhood_features.shape[4] == P
        assert neighborhood_features.shape[5] == C

        neighborhood_features = neighborhood_features.reshape(B * H * W, P * P, C)

        # reshape V to B, H, W, 1, P^2, 2
        neighborhood_flow_residual = neighborhood_flow_residual.reshape(-1, P * P, 2)

        # compute the attention
        attention_score = (
            torch.matmul(classification_features1, neighborhood_features.permute(0, 2, 1)) / self.temperature
        )
        attention_score = attention_score + self.classification_bias

        attention = torch.nn.functional.softmax(attention_score, dim=-1)
        log_softmax_attention = torch.nn.functional.log_softmax(attention_score, dim=-1)

        # compute the weighted sum
        residual = torch.matmul(attention, neighborhood_flow_residual)

        # reshape the residual to B, H, W, 2, then B, 2, H, W
        residual = residual.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        return residual, log_softmax_attention.reshape(B, H, W, P, P)

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        # if self.info_sharing_and_head_structure == "dual+single":

        head = getattr(self, f"head{head_num}") if head_num != "uncertainty" else self.uncertainty_head

        head_num = head_num if head_num != "uncertainty" else 1  # uncertainty head is always from branch 1

        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type in ["dpt", "moge_conv"]:
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def obtain_neighborhood_features(
        self, flow_estimation: torch.Tensor, other_features: torch.Tensor, local_patch: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the other features according to flow estimation to extract neighborhood features.

        This method samples features from the second view at locations determined by the flow estimation
        plus a local neighborhood around each predicted flow vector. It uses bilinear interpolation
        to sample features at non-integer coordinates.

        Args:
            flow_estimation (torch.Tensor): Flow prediction tensor with shape (B, 2, H, W)
            other_features (torch.Tensor): Features from the second view with shape (B, C, H, W)
            local_patch (int): Size of the local patch for neighborhood sampling (must be odd), default 5

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - expected_output (torch.Tensor): Sampled neighborhood features with shape (B, H, W, P, P, C)
                  where P is the local_patch size
                - neighborhood_grid_xy_residual (torch.Tensor): Local coordinate offsets with shape (B, H, W, P, P, 2)

        Raises:
            AssertionError: If local_patch is not an odd number
        """

        assert local_patch % 2 == 1, "local_patch should be odd number"

        P = local_patch
        R = (P - 1) // 2
        B, C, H, W = other_features.shape

        device = other_features.device

        # expected_output = torch.zeros(B, H, W, P, P, C, device=other_features.device, dtype=torch.float32)

        neighborhood_grid_ij: torch.Tensor

        i_local, j_local = torch.meshgrid(
            torch.arange(-R, R + 1, device=device), torch.arange(-R, R + 1, device=device), indexing="ij"
        )
        ij_local = torch.stack((i_local, j_local), dim=0)  # 2, P, P tensor

        # compute the indices of the fetch
        base_grid_xy = get_meshgrid_torch(W=W, H=H, device=device).permute(2, 0, 1).reshape(1, 2, H, W)

        target_coordinate_xy_float = flow_estimation + base_grid_xy
        target_coordinate_xy = target_coordinate_xy_float.view(B, 2, H, W, 1, 1)
        target_coordinate_ij = target_coordinate_xy[:, [1, 0], ...]

        # compute the neighborhood grid
        neighborhood_grid_ij = target_coordinate_ij + ij_local.view(1, 2, 1, 1, P, P)

        grid_for_sample = neighborhood_grid_ij[:, [1, 0], ...].permute(0, 2, 3, 4, 5, 1).reshape(B, H, W * P * P, 2)
        grid_for_sample = (grid_for_sample + 0.5) / torch.tensor([W, H], device=device).view(1, 1, 1, 2)
        grid_for_sample = grid_for_sample * 2 - 1

        expected_output = torch.nn.functional.grid_sample(
            other_features, grid=grid_for_sample, mode="bicubic", padding_mode="zeros", align_corners=False
        ).view(B, C, H, W, P, P)

        # transform BCHWPP to BHWPPC
        expected_output = expected_output.permute(0, 2, 3, 4, 5, 1)

        neighborhood_grid_xy_residual = ij_local[[1, 0], ...].view(1, 2, 1, 1, P, P).to(device).float()
        neighborhood_grid_xy_residual = neighborhood_grid_xy_residual.permute(0, 2, 3, 4, 5, 1).float()

        return expected_output, neighborhood_grid_xy_residual

    def _initialize_classification_head(self, classification_head_kwargs: Dict[str, Any]):
        """
        Initialize classification head for flow refinement.

        Args:
            classification_head_kwargs (Dict[str, Any]): Classification head configuration parameters

        Returns:
            nn.Module: Initialized classification head module

        Raises:
            ValueError: If unsupported classification head type is specified
        """

        if self.classification_head_type == "patch_mlp":
            return MLPFeature(**classification_head_kwargs)
        else:
            raise ValueError(f"Classification head type {self.classification_head_type} not supported.")

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This method guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
            Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer with keys:
                - "encoder": Parameters from the feature encoder
                - "info_sharing": Parameters from the information sharing module
                - "output_head": Parameters from the main prediction head
                - "classification_head": Parameters from the classification refinement head
                - "unet_feature": Parameters from U-Net features (if use_unet_feature=True)
                - "uncertainty_head": Parameters from uncertainty head (if present)
        """

        if self.use_unet_feature:
            params_dict = {
                "encoder": torch.nn.ParameterList(self.encoder.parameters()),
                "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
                "output_head": torch.nn.ParameterList(self.head1.parameters()),
                "classification_head": torch.nn.ParameterList(self.classification_head.parameters()),
                "unet_feature": torch.nn.ParameterList(
                    list(self.unet_feature.parameters())
                    + list(self.conv1.parameters())
                    + list(self.conv2.parameters())
                    + [self.classification_bias]
                ),
            }
        else:
            params_dict = {
                "encoder": torch.nn.ParameterList(self.encoder.parameters()),
                "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
                "output_head": torch.nn.ParameterList(self.head1.parameters()),
                "classification_head": torch.nn.ParameterList(self.classification_head.parameters()),
            }

        if hasattr(self, "uncertainty_head"):
            params_dict["uncertainty_head"] = torch.nn.ParameterList(self.uncertainty_head.parameters())

        return params_dict


