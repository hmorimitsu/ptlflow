"""UFM: A Simple Path towards Unified Dense Correspondence with Flow.

This model is a port of the official UFM (UniFlowMatch) implementation
(https://github.com/UniFlowMatch/UFM) to the PTLFlow interface. The UFM model directly regresses
pixel displacement (flow) fields between two images and can be used both for optical flow
estimation and for wide-baseline matching.

The original code and the vendored copies in the `uniflowmatch`, `uniception` and `layers`
folders are licensed under BSD-3-Clause (see the LICENSE file). Note that, as stated by the
original repository, the pretrained checkpoints inherit the licenses of the training datasets and
may not be used for commercial purposes (CC BY-NC-SA 4.0).

Reference: Zhang et al., "UFM: A Simple Path towards Unified Dense Correspondence with Flow", 2025.

Code: https://github.com/UniFlowMatch/UFM
Paper: https://arxiv.org/abs/2506.09278
"""

import torch

from ptlflow.utils.registry import register_model

from ..base_model.base_model import BaseModel
from .uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from .uniflowmatch.models import (
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)

# Architectural configurations of each released model variant, copied from the config.json file
# available in each checkpoint repository at https://huggingface.co/infinity1096.
_VARIANT_CONFIGS = {
    "base": {'adaptors_kwargs': {'flow': {'class': 'FlowAdaptor',
                              'kwargs': {'base_shape': [224, 224],
                                         'flow_mean': [0.0, 0.0],
                                         'flow_std': [25, 25],
                                         'name': 'flow',
                                         'output_normalized_coordinate': False,
                                         'scale_strategy': 'scale_both'}}},
 'detach_uncertainty_head': False,
 'encoder_kwargs': {'data_norm_type': 'dinov2',
                    'name': 'dinov2_encoder',
                    'patch_size': 14,
                    'size': 'large',
                    'with_registers': False},
 'encoder_str': 'dinov2',
 'feature_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                         'hooks': [0, 1, 2, 3],
                                         'input_feature_dims': [1024, 768, 768, 768],
                                         'layer_dims': [96, 192, 384, 768],
                                         'output_width_ratio': 1,
                                         'patch_size': 14,
                                         'pretrained_checkpoint_path': None,
                                         'use_bn': False},
                         'dpt_processor': {'hidden_dims': [128, 128],
                                           'input_feature_dim': 256,
                                           'output_dim': 2,
                                           'pretrained_checkpoint_path': None}},
 'head_type': 'dpt',
 'inference_resolution': [560, 420],
 'info_sharing_and_head_structure': 'dual+single',
 'info_sharing_kwargs': {'attn_drop': 0.0,
                         'custom_positional_encoding': None,
                         'depth': 12,
                         'dim': 768,
                         'drop_path': 0.0,
                         'gradient_checkpointing': False,
                         'indices': [5, 8],
                         'init_values': None,
                         'input_embed_dim': 1024,
                         'max_num_views': 2,
                         'mlp_ratio': 4.0,
                         'name': 'global_attention',
                         'norm_intermediate': True,
                         'num_heads': 12,
                         'pretrained_checkpoint_path': None,
                         'proj_drop': 0.0,
                         'qk_norm': False,
                         'qkv_bias': True,
                         'size': 'base',
                         'use_rand_idx_pe_for_non_reference_views': False},
 'info_sharing_str': 'global_attention',
 'pretrained_backbone_checkpoint_path': None,
 'pretrained_checkpoint_path': None,
 'uncertainty_adaptors_kwargs': {'non_occluded_mask': {'class': 'MaskAdaptor',
                                                       'kwargs': {'name': 'non_occluded_mask'}}},
 'uncertainty_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                             'hooks': [0, 1, 2, 3],
                                             'input_feature_dims': [1024, 768, 768, 768],
                                             'layer_dims': [96, 192, 384, 768],
                                             'output_width_ratio': 1,
                                             'patch_size': 14,
                                             'pretrained_checkpoint_path': None,
                                             'use_bn': False},
                             'dpt_processor': {'hidden_dims': [128, 128],
                                               'input_feature_dim': 256,
                                               'output_dim': 1,
                                               'pretrained_checkpoint_path': None}},
 'uncertainty_head_type': 'dpt'},
    "refine": {'adaptors_kwargs': {'flow': {'class': 'FlowAdaptor',
                              'kwargs': {'base_shape': [224, 224],
                                         'flow_mean': [0.0, 0.0],
                                         'flow_std': [25, 25],
                                         'name': 'flow',
                                         'output_normalized_coordinate': False,
                                         'scale_strategy': 'scale_both'}}},
 'classification_head_kwargs': {'bias': True,
                                'input_feature_dim': '1024 + 768',
                                'mlp_ratio': 2,
                                'output_dim': 16,
                                'patch_size': 14,
                                'pretrained_checkpoint_path': None},
 'classification_head_type': 'patch_mlp',
 'detach_uncertainty_head': True,
 'encoder_kwargs': {'data_norm_type': 'dinov2',
                    'indices': [6, 23],
                    'name': 'dinov2_encoder',
                    'patch_size': 14,
                    'size': 'large',
                    'with_registers': False},
 'encoder_str': 'dinov2',
 'feature_combine_method': 'conv',
 'feature_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                         'hooks': [0, 1, 2, 3],
                                         'input_feature_dims': [1024, 768, 768, 768],
                                         'layer_dims': [96, 192, 384, 768],
                                         'output_width_ratio': 1,
                                         'patch_size': 14,
                                         'pretrained_checkpoint_path': None,
                                         'use_bn': False},
                         'dpt_processor': {'hidden_dims': [128, 128],
                                           'input_feature_dim': 256,
                                           'output_dim': 2,
                                           'pretrained_checkpoint_path': None}},
 'head_type': 'dpt',
 'inference_resolution': [560, 420],
 'info_sharing_and_head_structure': 'dual+single',
 'info_sharing_kwargs': {'attn_drop': 0.0,
                         'custom_positional_encoding': None,
                         'depth': 12,
                         'dim': 768,
                         'drop_path': 0.0,
                         'gradient_checkpointing': False,
                         'indices': [5, 8],
                         'init_values': None,
                         'input_embed_dim': 1024,
                         'max_num_views': 2,
                         'mlp_ratio': 4.0,
                         'name': 'global_attention',
                         'norm_intermediate': True,
                         'num_heads': 12,
                         'pretrained_checkpoint_path': None,
                         'proj_drop': 0.0,
                         'qk_norm': False,
                         'qkv_bias': True,
                         'size': 'base',
                         'use_rand_idx_pe_for_non_reference_views': False},
 'info_sharing_str': 'global_attention',
 'pretrained_backbone_checkpoint_path': None,
 'pretrained_checkpoint_path': None,
 'refinement_range': 7,
 'temperature': 4,
 'uncertainty_adaptors_kwargs': {'non_occluded_mask': {'class': 'MaskAdaptor',
                                                       'kwargs': {'name': 'non_occluded_mask'}}},
 'uncertainty_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                             'hooks': [0, 1, 2, 3],
                                             'input_feature_dims': [1024, 768, 768, 768],
                                             'layer_dims': [96, 192, 384, 768],
                                             'output_width_ratio': 1,
                                             'patch_size': 14,
                                             'pretrained_checkpoint_path': None,
                                             'use_bn': False},
                             'dpt_processor': {'hidden_dims': [128, 128],
                                               'input_feature_dim': 256,
                                               'output_dim': 1,
                                               'pretrained_checkpoint_path': None}},
 'uncertainty_head_type': 'dpt',
 'use_unet_feature': True},
    "base-980": {'adaptors_kwargs': {'flow': {'class': 'FlowAdaptor',
                              'kwargs': {'base_shape': [224, 224],
                                         'flow_mean': [0.0, 0.0],
                                         'flow_std': [25, 25],
                                         'name': 'flow',
                                         'output_normalized_coordinate': False,
                                         'scale_strategy': 'scale_both'}}},
 'detach_uncertainty_head': False,
 'encoder_kwargs': {'data_norm_type': 'dinov2',
                    'name': 'dinov2_encoder',
                    'patch_size': 14,
                    'size': 'large',
                    'with_registers': False},
 'encoder_str': 'dinov2',
 'feature_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                         'hooks': [0, 1, 2, 3],
                                         'input_feature_dims': [1024, 768, 768, 768],
                                         'layer_dims': [96, 192, 384, 768],
                                         'output_width_ratio': 1,
                                         'patch_size': 14,
                                         'pretrained_checkpoint_path': None,
                                         'use_bn': False},
                         'dpt_processor': {'hidden_dims': [128, 128],
                                           'input_feature_dim': 256,
                                           'output_dim': 2,
                                           'pretrained_checkpoint_path': None}},
 'head_type': 'dpt',
 'inference_resolution': [980, 644],
 'info_sharing_and_head_structure': 'dual+single',
 'info_sharing_kwargs': {'attn_drop': 0.0,
                         'custom_positional_encoding': None,
                         'depth': 12,
                         'dim': 768,
                         'drop_path': 0.0,
                         'gradient_checkpointing': False,
                         'indices': [5, 8],
                         'init_values': None,
                         'input_embed_dim': 1024,
                         'max_num_views': 2,
                         'mlp_ratio': 4.0,
                         'name': 'global_attention',
                         'norm_intermediate': True,
                         'num_heads': 12,
                         'pretrained_checkpoint_path': None,
                         'proj_drop': 0.0,
                         'qk_norm': False,
                         'qkv_bias': True,
                         'size': 'base',
                         'use_rand_idx_pe_for_non_reference_views': False},
 'info_sharing_str': 'global_attention',
 'pretrained_backbone_checkpoint_path': None,
 'pretrained_checkpoint_path': None,
 'uncertainty_adaptors_kwargs': {'non_occluded_mask': {'class': 'MaskAdaptor',
                                                       'kwargs': {'name': 'non_occluded_mask'}}},
 'uncertainty_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                             'hooks': [0, 1, 2, 3],
                                             'input_feature_dims': [1024, 768, 768, 768],
                                             'layer_dims': [96, 192, 384, 768],
                                             'output_width_ratio': 1,
                                             'patch_size': 14,
                                             'pretrained_checkpoint_path': None,
                                             'use_bn': False},
                             'dpt_processor': {'hidden_dims': [128, 128],
                                               'input_feature_dim': 256,
                                               'output_dim': 1,
                                               'pretrained_checkpoint_path': None}},
 'uncertainty_head_type': 'dpt'},
    "refine-980": {'adaptors_kwargs': {'flow': {'class': 'FlowAdaptor',
                              'kwargs': {'base_shape': [224, 224],
                                         'flow_mean': [0.0, 0.0],
                                         'flow_std': [25, 25],
                                         'name': 'flow',
                                         'output_normalized_coordinate': False,
                                         'scale_strategy': 'scale_both'}}},
 'classification_head_kwargs': {'bias': True,
                                'input_feature_dim': '1024 + 768',
                                'mlp_ratio': 2,
                                'output_dim': 16,
                                'patch_size': 14,
                                'pretrained_checkpoint_path': None},
 'classification_head_type': 'patch_mlp',
 'detach_uncertainty_head': True,
 'encoder_kwargs': {'data_norm_type': 'dinov2',
                    'indices': [6, 23],
                    'name': 'dinov2_encoder',
                    'patch_size': 14,
                    'size': 'large',
                    'with_registers': False},
 'encoder_str': 'dinov2',
 'feature_combine_method': 'modulate',
 'feature_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                         'hooks': [0, 1, 2, 3],
                                         'input_feature_dims': [1024, 768, 768, 768],
                                         'layer_dims': [96, 192, 384, 768],
                                         'output_width_ratio': 1,
                                         'patch_size': 14,
                                         'pretrained_checkpoint_path': None,
                                         'use_bn': False},
                         'dpt_processor': {'hidden_dims': [128, 128],
                                           'input_feature_dim': 256,
                                           'output_dim': 2,
                                           'pretrained_checkpoint_path': None}},
 'head_type': 'dpt',
 'inference_resolution': [980, 644],
 'info_sharing_and_head_structure': 'dual+single',
 'info_sharing_kwargs': {'attn_drop': 0.0,
                         'custom_positional_encoding': None,
                         'depth': 12,
                         'dim': 768,
                         'drop_path': 0.0,
                         'gradient_checkpointing': False,
                         'indices': [5, 8],
                         'init_values': None,
                         'input_embed_dim': 1024,
                         'max_num_views': 2,
                         'mlp_ratio': 4.0,
                         'name': 'global_attention',
                         'norm_intermediate': True,
                         'num_heads': 12,
                         'pretrained_checkpoint_path': None,
                         'proj_drop': 0.0,
                         'qk_norm': False,
                         'qkv_bias': True,
                         'size': 'base',
                         'use_rand_idx_pe_for_non_reference_views': False},
 'info_sharing_str': 'global_attention',
 'pretrained_backbone_checkpoint_path': None,
 'pretrained_checkpoint_path': None,
 'refinement_range': 5,
 'temperature': 4,
 'uncertainty_adaptors_kwargs': {'non_occluded_mask': {'class': 'MaskAdaptor',
                                                       'kwargs': {'name': 'non_occluded_mask'}}},
 'uncertainty_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                             'hooks': [0, 1, 2, 3],
                                             'input_feature_dims': [1024, 768, 768, 768],
                                             'layer_dims': [96, 192, 384, 768],
                                             'output_width_ratio': 1,
                                             'patch_size': 14,
                                             'pretrained_checkpoint_path': None,
                                             'use_bn': False},
                             'dpt_processor': {'hidden_dims': [128, 128],
                                               'input_feature_dim': 256,
                                               'output_dim': 1,
                                               'pretrained_checkpoint_path': None}},
 'uncertainty_head_type': 'dpt',
 'use_unet_batchnorm': False,
 'use_unet_feature': True},
    "base-dinov2l-init": {'adaptors_kwargs': {'flow': {'class': 'FlowAdaptor',
                              'kwargs': {'base_shape': [224, 224],
                                         'flow_mean': [0.0, 0.0],
                                         'flow_std': [25, 25],
                                         'name': 'flow',
                                         'output_normalized_coordinate': False,
                                         'scale_strategy': 'scale_both'}}},
 'detach_uncertainty_head': False,
 'encoder_kwargs': {'data_norm_type': 'dinov2',
                    'keep_first_n_layers': 12,
                    'name': 'dinov2_encoder',
                    'patch_size': 14,
                    'size': 'large',
                    'with_registers': False},
 'encoder_str': 'dinov2',
 'feature_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                         'hooks': [0, 1, 2, 3],
                                         'input_feature_dims': [1024, 1024, 1024, 1024],
                                         'layer_dims': [96, 192, 384, 768],
                                         'output_width_ratio': 1,
                                         'patch_size': 14,
                                         'pretrained_checkpoint_path': None,
                                         'use_bn': False},
                         'dpt_processor': {'hidden_dims': [128, 128],
                                           'input_feature_dim': 256,
                                           'output_dim': 2,
                                           'pretrained_checkpoint_path': None}},
 'head_type': 'dpt',
 'inference_resolution': [560, 420],
 'info_sharing_and_head_structure': 'dual+single',
 'info_sharing_kwargs': {'attn_drop': 0.0,
                         'custom_positional_encoding': None,
                         'depth': 12,
                         'dim': 1024,
                         'drop_path': 0.0,
                         'gradient_checkpointing': False,
                         'indices': [4, 8],
                         'init_values': 1e-05,
                         'input_embed_dim': 1024,
                         'max_num_views_for_pe': 2,
                         'mlp_ratio': 4.0,
                         'name': 'global_attention',
                         'norm_intermediate': True,
                         'num_heads': 16,
                         'pretrained_checkpoint_path': '/ocean/projects/cis220039p/mdt2/shared/uniception_checkpoints/info_sharing/dino_init/gat_init_w_dinov2_vitl_layers_12_to_24.pth',
                         'proj_drop': 0.0,
                         'qk_norm': False,
                         'qkv_bias': True,
                         'size': 'large',
                         'use_rand_idx_pe_for_non_reference_views': False},
 'info_sharing_str': 'global_attention',
 'pretrained_backbone_checkpoint_path': None,
 'pretrained_checkpoint_path': None,
 'uncertainty_adaptors_kwargs': {'non_occluded_mask': {'class': 'MaskAdaptor',
                                                       'kwargs': {'name': 'non_occluded_mask'}}},
 'uncertainty_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                             'hooks': [0, 1, 2, 3],
                                             'input_feature_dims': [1024, 1024, 1024, 1024],
                                             'layer_dims': [96, 192, 384, 768],
                                             'output_width_ratio': 1,
                                             'patch_size': 14,
                                             'pretrained_checkpoint_path': None,
                                             'use_bn': False},
                             'dpt_processor': {'hidden_dims': [128, 128],
                                               'input_feature_dim': 256,
                                               'output_dim': 1,
                                               'pretrained_checkpoint_path': None}},
 'uncertainty_head_type': 'dpt'},
    "base-dinov2g-init": {'adaptors_kwargs': {'flow': {'class': 'FlowAdaptor',
                              'kwargs': {'base_shape': [224, 224],
                                         'flow_mean': [0.0, 0.0],
                                         'flow_std': [25, 25],
                                         'name': 'flow',
                                         'output_normalized_coordinate': False,
                                         'scale_strategy': 'scale_both'}}},
 'detach_uncertainty_head': False,
 'encoder_kwargs': {'data_norm_type': 'dinov2',
                    'keep_first_n_layers': 24,
                    'name': 'dinov2_encoder',
                    'patch_size': 14,
                    'size': 'giant',
                    'with_registers': True},
 'encoder_str': 'dinov2',
 'feature_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                         'hooks': [0, 1, 2, 3],
                                         'input_feature_dims': [1536, 1536, 1536, 1536],
                                         'layer_dims': [96, 192, 384, 768],
                                         'output_width_ratio': 1,
                                         'patch_size': 14,
                                         'pretrained_checkpoint_path': None,
                                         'use_bn': False},
                         'dpt_processor': {'hidden_dims': [128, 128],
                                           'input_feature_dim': 256,
                                           'output_dim': 2,
                                           'pretrained_checkpoint_path': None}},
 'head_type': 'dpt',
 'inference_resolution': [560, 420],
 'info_sharing_and_head_structure': 'dual+single',
 'info_sharing_kwargs': {'attn_drop': 0.0,
                         'custom_positional_encoding': None,
                         'depth': 16,
                         'dim': 1536,
                         'drop_path': 0.0,
                         'gradient_checkpointing': False,
                         'indices': [7, 12],
                         'init_values': 1e-05,
                         'input_embed_dim': 1536,
                         'max_num_views_for_pe': 2,
                         'mlp_layer': 'SwiGLU',
                         'mlp_ratio': 4.0,
                         'name': 'global_attention',
                         'norm_intermediate': True,
                         'num_heads': 24,
                         'proj_drop': 0.0,
                         'qk_norm': False,
                         'qkv_bias': True,
                         'size': 'giant',
                         'use_rand_idx_pe_for_non_reference_views': False},
 'info_sharing_str': 'global_attention',
 'pretrained_backbone_checkpoint_path': None,
 'pretrained_checkpoint_path': None,
 'uncertainty_adaptors_kwargs': {'non_occluded_mask': {'class': 'MaskAdaptor',
                                                       'kwargs': {'name': 'non_occluded_mask'}}},
 'uncertainty_head_kwargs': {'dpt_feature': {'feature_dim': 256,
                                             'hooks': [0, 1, 2, 3],
                                             'input_feature_dims': [1536, 1536, 1536, 1536],
                                             'layer_dims': [96, 192, 384, 768],
                                             'output_width_ratio': 1,
                                             'patch_size': 14,
                                             'pretrained_checkpoint_path': None,
                                             'use_bn': False},
                             'dpt_processor': {'hidden_dims': [128, 128],
                                               'input_feature_dim': 256,
                                               'output_dim': 1,
                                               'pretrained_checkpoint_path': None}},
 'uncertainty_head_type': 'dpt'},
}

# The variants that use the classification refinement architecture. The remaining ones use the
# confidence architecture (flow + covisibility heads).
_REFINE_VARIANTS = ("refine", "refine-980")


class UFM(BaseModel):
    """UFM model, adapted to the PTLFlow interface.

    The model is inference-oriented: no loss function is provided (the original training code is
    kept in a separate branch of the official repository).

    Inputs are expected to follow the PTLFlow conventions, i.e., images with shape [B, T, 3, H, W]
    in the [0, 1] range with BGR channel order. Each consecutive image pair is processed
    independently. The images are converted to RGB and normalized with the DINOv2 statistics, and
    each pair is resized to the trained resolution of the chosen variant (following the original
    inference procedure) before predicting the flow and covisibility mask.

    Args:
        variant (str): Which model variant to use. One of ("base", "refine", "base-980",
            "refine-980", "base-dinov2l-init", "base-dinov2g-init"). The variant must match the
            checkpoint given to --ckpt_path, since each variant has a different architecture.
    """

    # All the original checkpoints are named model.safetensors inside their HuggingFace
    # repositories. Since the local cache name is deduced from the url by default, all the
    # variants would collide into a single model.safetensors file. Therefore, each entry is
    # given as a (url, file_name) tuple, where file_name is the unique name used to store and
    # search for the checkpoint in the local cache. These are the names registered in
    # docs/source/models/checkpoint_list.csv.
    pretrained_checkpoints = {
        "base": (
            "https://huggingface.co/infinity1096/UFM-Base/resolve/main/model.safetensors",
            "ufm-base.safetensors",
        ),
        "base-980": (
            "https://huggingface.co/infinity1096/UFM-Base-980/resolve/main/model.safetensors",
            "ufm-base-980.safetensors",
        ),
        "base-dinov2l-init": (
            "https://huggingface.co/infinity1096/UFM-Base-DINOv2L-init/resolve/main/model.safetensors",
            "ufm-base-dinov2l-init.safetensors",
        ),
        "base-dinov2g-init": (
            "https://huggingface.co/infinity1096/UFM-Base-DINOv2G-init/resolve/main/model.safetensors",
            "ufm-base-dinov2g-init.safetensors",
        ),
    }

    def __init__(
        self,
        variant: str = "base",
        **kwargs,
    ):
        super().__init__(output_stride=14, loss_fn=None, **kwargs)

        if variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown UFM variant: {variant}. Choose one from {list(_VARIANT_CONFIGS.keys())}"
            )
        self.variant = variant

        model_cls = (
            UniFlowMatchClassificationRefinement
            if variant in _REFINE_VARIANTS
            else UniFlowMatchConfidence
        )
        self.model = model_cls(**_VARIANT_CONFIGS[variant])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Remap the keys of the official UFM checkpoints to the containment layout used here.

        The official checkpoints store the parameters without any prefix, whereas this model keeps
        the UFM network inside the self.model attribute.

        Checkpoints from the refine variants contain extra modules (classification_head,
        unet_feature, etc.) that only exist in those variants. Therefore, before loading, this
        method checks that the architecture of the checkpoint matches the architecture of the
        instantiated variant, raising an informative error when they diverge. In this way,
        loading can remain strict, and genuine loading problems are not silently ignored.
        """
        state_dict = {
            (k if k.startswith("model.") else "model." + k): v
            for k, v in state_dict.items()
        }
        if strict:
            self._check_variant_matches_checkpoint(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as err:
            raise RuntimeError(
                f"Failed to load the UFM checkpoint into the model variant {self.variant!r}. "
                "Each UFM checkpoint can only be loaded into a model instantiated with the "
                "matching variant, so verify that the value given to --model.variant corresponds "
                "to the checkpoint given to --ckpt_path.\n"
                f"Original error:\n{err}"
            ) from err

    def _check_variant_matches_checkpoint(self, state_dict) -> None:
        """Raise an informative error if the checkpoint belongs to another variant family.

        The refine variants (refine and refine-980) are the only ones with classification
        refinement modules. The presence (or absence) of these modules in the checkpoint is used
        to detect when the checkpoint belongs to a different family than the instantiated
        variant.
        """
        ckpt_has_refine_modules = any(
            k.startswith("model.classification_head.") for k in state_dict
        )
        if ckpt_has_refine_modules != (self.variant in _REFINE_VARIANTS):
            if ckpt_has_refine_modules:
                raise ValueError(
                    f"The given checkpoint contains the classification refinement modules "
                    f"(classification_head, unet_feature, etc.), but they do not exist in the "
                    f"instantiated variant {self.variant!r}. This checkpoint most likely "
                    f"corresponds to one of the refine variants "
                    f"{list(_REFINE_VARIANTS)}. Instantiate the model with the matching variant, "
                    f"e.g., with --model.variant refine."
                )
            else:
                raise ValueError(
                    f"The instantiated variant {self.variant!r} expects the classification "
                    f"refinement modules (classification_head, unet_feature, etc.), but they are "
                    f"not present in the given checkpoint. This checkpoint most likely "
                    f"corresponds to one of the base variants "
                    f"{[v for v in _VARIANT_CONFIGS if v not in _REFINE_VARIANTS]}. "
                    f"Instantiate the model with the matching variant, "
                    f"e.g., with --model.variant base."
                )

    def forward(self, inputs):
        """Forward the inputs through the network and produce the predictions.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Expected keys:
            - 'images': torch.Tensor, shape [B, T, 3, H, W] in the [0, 1] range, BGR channel order.

        Returns
        -------
        Dict[str, torch.Tensor]
            - 'flows': 5D tensor [B, T-1, 2, H, W] with the predicted flows at the input scale.
            - 'confs': 5D tensor [B, T-1, 1, H, W] with the predicted covisibility masks in
              [0, 1] (1 = fully covisible).
        """
        inputs = self._normalize_inputs(inputs)
        images = inputs["images"]

        assert images.ndim == 5, "images must have 5 dimensions in the BTHWC order"
        assert images.shape[2] == 3, "images must have 3 channels"
        assert images.shape[1] >= 2, "At least two images are required to estimate the flow"

        # PTLFlow provides images in the [0, 1] range with BGR channel order, whereas UFM expects
        # RGB images normalized with the DINOv2 statistics. The normalization follows the same
        # operation order as the official UFM inference code, so that the results are numerically
        # identical to the original model.
        images = images.flip([-3]).float()
        norm = IMAGE_NORMALIZATION_DICT[self.model.encoder.data_norm_type]
        mean = norm.mean.view(1, 3, 1, 1).to(device=images.device)
        std = norm.std.view(1, 3, 1, 1).to(device=images.device)
        images = (images - mean) / std

        flows = []
        confs = []
        for i in range(images.shape[1] - 1):
            result = self.model.predict_correspondences_batched(
                source_image=images[:, i].contiguous(),
                target_image=images[:, i + 1].contiguous(),
                data_norm_type=self.model.encoder.data_norm_type,
            )
            flows.append(result.flow.flow_output)
            if result.covisibility is not None:
                confs.append(result.covisibility.mask.unsqueeze(1))

        outputs = {"flows": torch.stack(flows, dim=1)}
        if len(confs) > 0:
            outputs["confs"] = torch.stack(confs, dim=1)
        return self._maybe_return_simple(outputs)


class UFMRefine(UFM):
    """UFM model with the classification refinement architecture.

    This variant is only compatible with the "refine" and "refine-980" variants, which are the
    only ones that include the classification refinement modules (classification_head,
    unet_feature, etc.).
    """
    pretrained_checkpoints = {
        "refine": (
            "https://huggingface.co/infinity1096/UFM-Refine/resolve/main/model.safetensors",
            "ufm-refine.safetensors",
        ),
        "refine-980": (
            "https://huggingface.co/infinity1096/UFM-Refine-980/resolve/main/model.safetensors",
            "ufm-refine-980.safetensors",
        ),
    }

    def __init__(self, variant="refine", **kwargs):
        super().__init__(variant=variant, **kwargs)




@register_model
class ufm(UFM):
    pass


@register_model
class ufm_refine(UFMRefine):
    pass
