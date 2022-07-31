import os
import math
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .setrans_ablation import RandPosEmbedder, SinuPosEmbedder, ZeroEmbedder, MultiHeadFeatTrans
from .utils import print0
torch.set_printoptions(sci_mode=False)

bb2_stage_dims = {  'raft-small':   [32, 32,  64,  96,   128],   
                    'raft-basic':   [64, 64,  96,  128,  256],   
                    'resnet34':     [64, 64,  128, 256,  512],
                    'resnet50':     [64, 256, 512, 1024, 2048],
                    'resnet101':    [64, 256, 512, 1024, 2048],
                    'resibn101':    [64, 256, 512, 1024, 2048],   # resibn: resnet + IBN layers
                    'eff-b0':       [16, 24,  40,  112,  1280],   # input: 224
                    'eff-b1':       [16, 24,  40,  112,  1280],   # input: 240
                    'eff-b2':       [16, 24,  48,  120,  1408],   # input: 260
                    'eff-b3':       [24, 32,  48,  136,  1536],   # input: 300
                    'eff-b4':       [24, 32,  56,  160,  1792],   # input: 380
                    'i3d':          [64, 192, 480, 832,  1024]    # input: 224
                 }

# Can also be implemented using torch.meshgrid().
def gen_all_indices(shape, device):
    indices = torch.arange(shape.numel(), device=device).view(shape)

    out = []
    for dim_size in reversed(shape):
        out.append(indices % dim_size)
        indices = torch.div(indices, dim_size, rounding_mode='trunc')
    return torch.stack(tuple(reversed(out)), len(shape))

# drop_path and DropPath are copied from timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SETransConfig(object):
    def __init__(self):
        self.feat_dim       = -1
        self.in_feat_dim    = -1
        # self.backbone_type  = 'eff-b4'          # resnet50, resnet101, resibn101, eff-b1~b4
        # self.bb_stage_idx   = 4                 # Last stage of the five stages. Index starts from 0.
        # self.set_backbone_type(self.backbone_type)
        # self.use_pretrained = True        

        # Positional encoding settings.
        self.pos_dim           = 2
        self.pos_code_weight   = 1
        
        # Architecture settings
        # Number of modes in the expansion attention block.
        # When doing ablation study of multi-head, num_modes means num_heads, 
        # to avoid introducing extra config parameters.
        self.num_modes = 4
        self.tie_qk_scheme  = 'shared'          # shared, loose, or none.
        self.trans_output_type  = 'private'     # shared or private.
        self.act_fun = F.gelu

        self.attn_clip = 100
        self.attn_diag_cycles = 1000
        self.base_initializer_range = 0.02
        
        self.qk_have_bias = False
        # Without the bias term, V projection often performs better.
        self.v_has_bias = False
        # Add an identity matrix (*0.02*query_idbias_scale) to query/key weights
        # to make a bias towards identity mapping.
        # Set to 0 to disable the identity bias.
        self.query_idbias_scale = 10
        self.feattrans_lin1_idbias_scale = 10

        # Pooling settings
        self.pool_modes_feat  = 'softmax'       # softmax, max, mean, or none.

        # Randomness settings
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.2
        self.drop_path_prob = 0                 # Drop path reduces performance greatly.
        self.pos_code_type          = 'bias'
        self.ablate_multihead       = False
        self.out_attn_probs_only    = False
        # When out_attn_scores_only, dropout is not applied to attention scores.
        self.out_attn_scores_only   = False
        self.attn_mask_radius = -1
        
    def set_backbone_type(self, args):
        if self.try_assign(args, 'backbone_type'):
            self.bb_stage_dims  = bb2_stage_dims[self.backbone_type]
            self.in_feat_dim    = self.bb_stage_dims[-1]
    
    # return True if any parameter is successfully set, and False if none is set.
    def try_assign(self, args, *keys):
        is_successful = False
        
        for key in keys:
            if key in args:
                if isinstance(args, dict):
                    self.__dict__[key] = args[key]
                else:
                    self.__dict__[key] = args.__dict__[key]
                is_successful = True
                
        return is_successful

    def update_config(self, args):
        self.set_backbone_type(args)
        self.try_assign(args, 'use_pretrained', 'apply_attn_stage', 'num_modes', 
                              'trans_output_type', 'base_initializer_range', 
                              'pos_code_type', 'ablate_multihead', 'attn_clip', 'attn_diag_cycles', 
                              'tie_qk_scheme', 'feattrans_lin1_idbias_scale', 'qk_have_bias', 'v_has_bias',
                              # out_attn_probs_only/out_attn_scores_only are only True for the optical flow correlation block.
                              'out_attn_probs_only', 'out_attn_scores_only', 
                              'in_feat_dim', 'pos_bias_radius')
        
        if self.try_assign(args, 'out_feat_dim'):
            self.feat_dim   = self.out_feat_dim
        else:
            self.feat_dim   = self.in_feat_dim
            
        if 'dropout_prob' in args and args.dropout_prob >= 0:
            self.hidden_dropout_prob          = args.dropout_prob
            self.attention_probs_dropout_prob = args.dropout_prob
            print0("Dropout prob: %.2f" %(args.dropout_prob))
            
CONFIG = SETransConfig()


# =================================== SETrans Initialization ====================================#
class SETransInitWeights(nn.Module):
    """ An abstract class to handle weights initialization """
    def __init__(self, config, *inputs, **kwargs):
        super(SETransInitWeights, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
            type(module.weight)      # <class 'torch.nn.parameter.Parameter'>
            type(module.weight.data) # <class 'torch.Tensor'>
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            base_initializer_range  = self.config.base_initializer_range
            module.weight.data.normal_(mean=0.0, std=base_initializer_range)
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def tie_qk(module):
    if isinstance(module, CrossAttFeatTrans) \
            and module.tie_qk_scheme != 'none' and module.tie_qk_scheme != None:
        module.tie_qk()

def add_identity_bias(module):
    if isinstance(module, CrossAttFeatTrans) or isinstance(module, ExpandedFeatTrans):
        module.add_identity_bias()
                        
#====================================== SETrans Shared Modules ========================================#

class MMSharedMid(nn.Module):
    def __init__(self, config):
        super(MMSharedMid, self).__init__()
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        self.shared_linear  = nn.Linear(self.feat_dim, self.feat_dim)
        self.mid_act_fn     = config.act_fun
        # This dropout is not presented in huggingface transformers.
        # Added to conform with lucidrains and rwightman's implementations.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    # x: [B0, 1792*4, U]
    def forward(self, x):
        # shape_4d: [B0, 4, 1792, U].
        shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        # x_4d: [B0, 4, U, 1792].
        x_4d        = x.view(shape_4d).permute([0, 1, 3, 2])

        x_trans     = self.shared_linear(x_4d)
        x_act       = self.mid_act_fn(x_trans)
        x_drop      = self.dropout(x_act)
        
        # restore the original shape
        x_drop      = x_drop.permute([0, 1, 3, 2]).reshape(x.shape)

        return x_drop

# MMPrivateOutput/MMSharedOutput <- MMandedFeatTrans <- CrossAttFeatTrans
# MM***Output has a shortcut (residual) connection.
class MMPrivateOutput(nn.Module):
    def __init__(self, config):
        super(MMPrivateOutput, self).__init__()
        self.num_modes  = config.num_modes
        self.feat_dim   = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        # Each group (mode) is applied a linear transformation, respectively.
        self.group_linear = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1792*4, U]
    def forward(self, x, shortcut):
        x        = self.group_linear(x)
        # x_comb: [B0, 1792*4, U]. Residual connection.
        x_comb   = x + shortcut
        shape_4d = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        # x_comb_4d, x_drop_4d: [B0, 4, U, 1792].
        x_comb_4d = x.view(shape_4d).permute([0, 1, 3, 2])
        x_drop_4d = self.dropout(x_comb_4d)
        x_normed  = self.resout_norm_layer(x_drop_4d)
        return x_normed

# MMPrivateOutput/MMSharedOutput <- MMandedFeatTrans <- CrossAttFeatTrans
# MM***Output has a shortcut (residual) connection.
class MMSharedOutput(nn.Module):
    # feat_dim_allmode is not used. Just to keep the ctor arguments the same as MMPrivateOutput.
    def __init__(self, config):
        super(MMSharedOutput, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim  = config.feat_dim
        self.shared_linear = nn.Linear(self.feat_dim, self.feat_dim)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1792*4, U] or [B0, 4, U, 1792]
    def forward(self, x, shortcut):
        # shape_4d: [B0, 4, 1792, U].
        shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        if len(x.shape) == 3:
            x_4d    = x.view(shape_4d).permute([0, 1, 3, 2])
        else:
            x_4d    = x
        if len(shortcut.shape) == 3:
            shortcut_4d = shortcut.view(shape_4d).permute([0, 1, 3, 2])
        else:
            shortcut_4d = shortcut

        # x_4d, shortcut_4d: [B0, 4, U, 1792].
        x_trans     = self.shared_linear(x_4d)
        # x_4d, x_comb: [B0, 4, U, 1792]. Residual connection.
        x_comb      = x_trans + shortcut_4d
        x_drop      = self.dropout(x_comb)
        x_normed    = self.resout_norm_layer(x_drop)
        return x_normed

# group_dim: the tensor dimension that corresponds to the multiple groups.
class LearnedSoftAggregate(nn.Module):
    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim  = group_dim
        # num_feat = 1: element-wise score function & softmax.
        # num_feat > 1: the linear score function is applied to the last dim (features) of the input tensor. 
        self.num_feat   = num_feat
        self.feat2score = nn.Linear(num_feat, 1)
        self.keepdim    = keepdim

    def forward(self, x, score_basis=None):
        # Assume the last dim of x is the feature dim.
        if score_basis is None:
            score_basis = x
        
        if self.num_feat == 1:
            mode_scores = self.feat2score(score_basis.unsqueeze(-1)).squeeze(-1)
        else:
            mode_scores = self.feat2score(score_basis)
        attn_probs  = mode_scores.softmax(dim=self.group_dim)
        x_aggr      = (x * attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
        return x_aggr

# ExpandedFeatTrans <- CrossAttFeatTrans.
# ExpandedFeatTrans has a residual connection.
class ExpandedFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(ExpandedFeatTrans, self).__init__()
        self.config = config
        self.name = name
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.num_modes = config.num_modes
        self.feat_dim_allmode = self.feat_dim * self.num_modes
        # first_linear is the value projection in other transformer implementations.
        # The output of first_linear will be divided into num_modes groups.
        # first_linear is always 'private' for each group, i.e.,
        # parameters are not shared (parameter sharing makes no sense).
        self.first_linear = nn.Linear(self.in_feat_dim, self.feat_dim_allmode, bias=config.v_has_bias)
            
        self.base_initializer_range = config.base_initializer_range
        self.has_FFN        = getattr(config, 'has_FFN', True)
        self.has_input_skip = getattr(config, 'has_input_skip', False)
        self.drop_path = DropPath(config.drop_path_prob) if config.drop_path_prob > 0. else nn.Identity()

        print0("{}: v_has_bias: {}, has_FFN: {}, has_input_skip: {}".format(
              self.name, config.v_has_bias, self.has_FFN, self.has_input_skip))
              
        self.pool_modes_keepdim = False
        self.pool_modes_feat = config.pool_modes_feat

        if self.pool_modes_feat == 'softmax':
            agg_basis_feat_dim = self.feat_dim

            # group_dim = 1, i.e., features will be aggregated across the modes.
            self.feat_softaggr = LearnedSoftAggregate(agg_basis_feat_dim, group_dim=1,
                                                      keepdim=self.pool_modes_keepdim)

        if self.has_FFN:
            self.intermediate = MMSharedMid(self.config)

            if config.trans_output_type == 'shared':
                self.output = MMSharedOutput(config)
            elif config.trans_output_type == 'private':
                self.output = MMPrivateOutput(config)

        # Have to ensure U1 == U2.
        if self.has_input_skip:
            self.input_skip_coeff = Parameter(torch.ones(1))
            self.skip_layer_norm  = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
            
    def add_identity_bias(self):
        if self.config.feattrans_lin1_idbias_scale > 0:
            # first_linear dimension is num_modes * feat_dim.
            # If in_feat_dim == feat_dim, only add identity bias to the first mode.
            # If in_feat_dim > feat_dim, expand to more modes until all in_feat_dim dimensions are covered.
            identity_weight = torch.diag(torch.ones(self.feat_dim)) * self.base_initializer_range \
                              * self.config.feattrans_lin1_idbias_scale
            # Only bias the weight of the first mode.
            # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
            self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] = \
                self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] * 0.5 + identity_weight

    # input_feat is usually key_feat.
    # input_feat: [3, 4416, 128]. attention_probs: [3, 4, 4416, 4416]. 
    def forward(self, input_feat, attention_probs):
        # input_feat: [B, U2, 1792], mm_first_feat: [B, Units, 1792*4]
        # B: batch size, U2: number of the 2nd group of units,
        # IF: in_feat_dim, could be different from feat_dim, due to layer compression
        # (different from squeezed attention).
        B, U2, IF = input_feat.shape
        U1 = attention_probs.shape[2]
        F = self.feat_dim
        M = self.num_modes
        mm_first_feat = self.first_linear(input_feat)
        # mm_first_feat after transpose: [B, 1792*4, U2]
        mm_first_feat = mm_first_feat.transpose(1, 2)

        # mm_first_feat_4d: [B, 4, U2, 1792]
        mm_first_feat_4d = mm_first_feat.view(B, M, F, U2).transpose(2, 3)

        # attention_probs:      [B, 4, U1, U2]. On sintel: [1, 4, 7040, 7040]
        # mm_first_feat_fusion: [B, 4, U2, F]. On sintel: [1, 4, 7040, 256]
        mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)
        mm_first_feat_fusion_3d = mm_first_feat_fusion.transpose(2, 3).reshape(B, M*F, U1)
        mm_first_feat = mm_first_feat_fusion_3d

        if self.has_FFN:
            # mm_mid_feat:   [B, 1792*4, U1]. Group linear & gelu of mm_first_feat.
            mm_mid_feat   = self.intermediate(mm_first_feat)
            # mm_last_feat:  [B, 4, U1, 1792]. Group/shared linear & residual & Layernorm
            mm_last_feat = self.output(mm_mid_feat, mm_first_feat)
            mm_trans_feat = mm_last_feat
        else:
            mm_trans_feat = mm_first_feat_fusion
            
        if self.pool_modes_feat == 'softmax':
            trans_feat = self.feat_softaggr(mm_trans_feat)
        elif self.pool_modes_feat == 'max':
            trans_feat = mm_trans_feat.max(dim=1)[0]
        elif self.pool_modes_feat == 'mean':
            trans_feat = mm_trans_feat.mean(dim=1)
        elif self.pool_modes_feat == 'none':
            trans_feat = mm_trans_feat

        # Have to ensure U1 == U2.
        if self.has_input_skip:
            trans_feat = self.input_skip_coeff * input_feat + self.drop_path(trans_feat)
            trans_feat = self.skip_layer_norm(trans_feat)
            
        # trans_feat: [B, U1, 1792]
        return trans_feat

class CrossAttFeatTrans(SETransInitWeights):
    def __init__(self, config, name):
        super(CrossAttFeatTrans, self).__init__(config)
        self.config     = config
        self.name       = name
        self.num_modes  = config.num_modes
        self.in_feat_dim    = config.in_feat_dim
        self.feat_dim       = config.feat_dim
        self.attention_mode_dim = self.in_feat_dim // self.num_modes   # 448
        # att_size_allmode: 512 * modes
        self.att_size_allmode = self.num_modes * self.attention_mode_dim
        self.query = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.key   = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.base_initializer_range = config.base_initializer_range

        self.out_attn_scores_only   = config.out_attn_scores_only
        self.out_attn_probs_only    = config.out_attn_probs_only
        self.ablate_multihead       = config.ablate_multihead

        # out_attn_scores_only / out_attn_probs_only implies no FFN nor V projection.
        if self.out_attn_scores_only or self.out_attn_probs_only:
            self.out_trans  = None
            if self.num_modes > 1:
                # Each attention value is a scalar. So num_feat = 1.
                self.attn_softaggr = LearnedSoftAggregate(1, group_dim=1, keepdim=True)
            
        elif self.ablate_multihead:
            self.out_trans  = MultiHeadFeatTrans(config, name + "-out_trans")
        else:
            self.out_trans  = ExpandedFeatTrans(config,  name + "-out_trans")

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.tie_qk_scheme    = config.tie_qk_scheme
        print0("{}: in_feat_dim: {}, feat_dim: {}, modes: {}, qk_have_bias: {}".format(
              self.name, self.in_feat_dim, self.feat_dim, self.num_modes, config.qk_have_bias))

        # if using SlidingPosBiases2D, then add positional embeddings in CrossAttFeatTrans.forward().
        if config.pos_code_type == 'bias':
            self.pos_code_weight = config.pos_code_weight
            print0("Positional biases weight: {:.3}".format(self.pos_code_weight))
        else:
            self.pos_code_weight = 1
            
        self.attn_clip    = config.attn_clip
        if 'attn_diag_cycles' in config.__dict__:
            self.attn_diag_cycles   = config.attn_diag_cycles
        else:
            self.attn_diag_cycles   = 1000
        self.max_attn    = 0
        self.clamp_count = 0
        self.call_count  = 0
        self.apply(self.init_weights)
        self.apply(tie_qk)
        # tie_qk() has to be executed after weight initialization.
        self.apply(add_identity_bias)
        
    # if tie_qk_scheme is not None, it overrides the initialized self.tie_qk_scheme
    def tie_qk(self, tie_qk_scheme=None):
        # override config.tie_qk_scheme
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme

        if self.tie_qk_scheme == 'shared':
            self.key.weight = self.query.weight
            if self.key.bias is not None:
                self.key.bias = self.query.bias

        elif self.tie_qk_scheme == 'loose':
            self.key.weight.data.copy_(self.query.weight)
            if self.key.bias is not None:
                self.key.bias.data.copy_(self.query.bias)

    def add_identity_bias(self):
        identity_weight = torch.diag(torch.ones(self.attention_mode_dim)) * self.base_initializer_range \
                          * self.config.query_idbias_scale
        repeat_count = self.in_feat_dim // self.attention_mode_dim
        identity_weight = identity_weight.repeat([1, repeat_count])
        # only bias the weight of the first mode
        # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
        self.key.weight.data[:self.attention_mode_dim] = \
            self.key.weight.data[:self.attention_mode_dim] * 0.5 + identity_weight

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_modes, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # pos_biases: [1, 1, U1, U2].
    def forward(self, query_feat, key_feat=None, pos_biases=None, attention_mask=None):
        # query_feat: [B, U1, 1792]
        # if key_feat == None: self attention.
        if key_feat is None:
            key_feat = query_feat
        # mixed_query_layer, mixed_key_layer: [B, U1, 1792], [B, U2, 1792]
        mixed_query_layer = self.query(query_feat)
        mixed_key_layer   = self.key(key_feat)
        # query_layer, key_layer: [B, 4, U1, 448], [B, 4, U2, 448]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B0, 4, U1, 448] [B0, 4, 448, U2]
        attention_scores = attention_scores / math.sqrt(self.attention_mode_dim)  # [B0, 4, U1, U2]

        #if self.call_count == 0:
        #    print0(f"{self.name} query: {list(query_feat.shape)}, attn: {list(attention_scores.shape)}")

        with torch.no_grad():
            curr_max_attn = attention_scores.max().item()
            curr_avg_attn = attention_scores.abs().mean().item()

        if curr_max_attn > self.max_attn:
            self.max_attn = curr_max_attn

        if curr_max_attn > self.attn_clip:
            attention_scores = torch.clamp(attention_scores, -self.attn_clip, self.attn_clip)
            self.clamp_count += 1
        
        self.call_count += 1
        if self.training:
            if self.call_count % self.attn_diag_cycles == 0:
                print0("max-attn: {:.2f}, avg-attn: {:.2f}, clamp-count: {}".format(self.max_attn, curr_avg_attn, self.clamp_count))
                self.max_attn    = 0
                self.clamp_count = 0

        if pos_biases is not None:
            #[B0, 8, U1, U2] = [B0, 8, U1, U2]  + [1, 1, U1, U2].
            attention_scores = attention_scores + self.pos_code_weight * pos_biases
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # When out_attn_scores_only, dropout is not applied to attention scores.
        if self.out_attn_scores_only:
            if self.num_modes > 1:
                # [3, num_modes=4, 4500, 4500] => [3, 1, 4500, 4500]
                attention_scores = self.attn_softaggr(attention_scores)
            # attention_scores = self.att_dropout(attention_scores)
            return attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # self.attention_probs = attention_probs

        # lucidrains doesn't have this dropout but rwightman has. Will keep it.
        attention_probs = self.att_dropout(attention_probs)     #[B0, 4, U1, U2]

        if self.out_attn_probs_only:
            # [6, 4, 4500, 4500]
            return attention_probs

        else:
            # out_feat: [B0, U1, 1792], in the same size as query_feat.
            out_feat      = self.out_trans(key_feat, attention_probs)
            return out_feat

class SelfAttVisPosTrans(nn.Module):
    def __init__(self, config, name):
        nn.Module.__init__(self)
        self.config = copy.copy(config)
        self.name = name
        self.out_attn_only = config.out_attn_scores_only or config.out_attn_probs_only
        self.attn_mask_radius   = config.attn_mask_radius
        self.setrans = CrossAttFeatTrans(self.config, name)
        self.vispos_encoder = SETransInputFeatEncoder(self.config)

    def forward(self, x):
        coords = gen_all_indices(x.shape[2:], device=x.device)
        if self.attn_mask_radius > 0:
            coords2 = coords.reshape(-1, 2)
            coords_diff = coords2.unsqueeze(0) - coords2.unsqueeze(1)
            attn_mask = (coords_diff.abs().max(dim=2)[0] > self.attn_mask_radius).float()
            attn_mask = (attn_mask * -1e9).unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = None
 
        coords = coords.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        x_vispos, pos_biases = self.vispos_encoder(x, coords, return_pos_biases=True)

                   
        # if out_attn_scores_only/out_attn_probs_only, 
        # then x_trans is an attention matrix in the shape of (query unit number, key unit number)
        # otherwise, output features are in the same shape as the query features.
        # key features are recombined to get new query features by matmul(attention_probs, V(key features))
        #             frame1 frame2
        # x_vispos, x_trans: [4, 2852, 256]
        # Here key_feat is omitted (None), i.e., key_feat = query_feat = x_vispos.
        x_trans = self.setrans(x_vispos, pos_biases=pos_biases, attention_mask=attn_mask)

        # Save f2 attention for visualization
        if self.name == 'F2 transformer' and 'SAVEF2' in os.environ:
            # save the attention scores
            f2_attention_probs = self.setrans.attention_probs.detach().cpu()
            # [B0, 4, U1, U2] => [B0, U1, U2]
            f2_attention_probs = f2_attention_probs.mean(dim=1, keepdim=False)
            f2_savepath = os.environ['SAVEF2']
            batch, C, h1, w1 = x.shape
            f2attn = f2_attention_probs.reshape(batch, h1, w1, h1, w1)
            torch.save(f2attn, f2_savepath)
            print0(f"F2 attention tensor saved to {f2_savepath}")

        # reshape x_trans to the input shape.
        if not self.out_attn_only:
            x_trans_shape = x_trans.shape
            x_trans = x_trans.permute(0, 2, 1).reshape(x.shape)
            
        return x_trans

# =================================== SETrans BackBone Components ==============================#

class LearnedSinuPosEmbedder(nn.Module):
    def __init__(self, pos_dim, pos_embed_dim, omega=1, affine=True):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_fc = nn.Linear(self.pos_dim, self.pos_embed_dim, bias=True)
        self.pos_mix_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        self.omega = omega
        print0("Learnable Sinusoidal positional encoding")
        
    def forward(self, pos_normed):
        pos_embed_sum = 0
        pos_embed0 = self.pos_fc(pos_normed)
        pos_embed_sin = torch.sin(self.omega * pos_embed0[:, :, 0::2])
        pos_embed_cos = torch.cos(self.omega * pos_embed0[:, :, 1::2])
        # Interlace pos_embed_sin and pos_embed_cos.
        pos_embed_mix = torch.stack((pos_embed_sin, pos_embed_cos), dim=3).view(pos_embed0.shape)
        pos_embed_out = self.pos_mix_norm_layer(pos_embed_mix)

        return pos_embed_out

class SlidingPosBiases2D(nn.Module):
    def __init__(self, pos_dim=2, pos_bias_radius=7, max_pos_size=(200, 200)):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        # biases: [15, 15]
        pos_bias_shape = [ pos_bias_radius * 2 + 1 for i in range(pos_dim) ]
        self.biases = Parameter(torch.zeros(pos_bias_shape))
        # Currently only feature maps with a 2D spatial shape (i.e., 2D images) are supported.
        if self.pos_dim == 2:
            all_h1s, all_w1s, all_h2s, all_w2s = [], [], [], []
            for i in range(max_pos_size[0]):
                i_h1s, i_w1s, i_h2s, i_w2s = [], [], [], []
                for j in range(max_pos_size[1]):
                    h1s, w1s, h2s, w2s = torch.meshgrid(torch.tensor(i), torch.tensor(j), 
                                                        torch.arange(i, i+2*R+1), torch.arange(j, j+2*R+1))
                    i_h1s.append(h1s)
                    i_w1s.append(w1s)
                    i_h2s.append(h2s)
                    i_w2s.append(w2s)
                                                  
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
            
            all_h1s = torch.cat(all_h1s, dim=0)
            all_w1s = torch.cat(all_w1s, dim=0)
            all_h2s = torch.cat(all_h2s, dim=0)
            all_w2s = torch.cat(all_w2s, dim=0)
        else:
            breakpoint()

        # Put indices on GPU to speed up. 
        # But if without persistent=False, they will be saved to checkpoints, 
        # making the checkpoints unnecessarily huge.
        self.register_buffer('all_h1s', all_h1s, persistent=False)
        self.register_buffer('all_w1s', all_w1s, persistent=False)
        self.register_buffer('all_h2s', all_h2s, persistent=False)
        self.register_buffer('all_w2s', all_w2s, persistent=False)
        print0(f"Sliding-window Positional Biases, r: {R}, max size: {max_pos_size}")
        
    def forward(self, feat_shape, device):
        R = self.R
        spatial_shape = feat_shape[-self.pos_dim:]
        # [H, W, H, W] => [H+2R, W+2R, H+2R, W+2R].
        padded_pos_shape  = list(spatial_shape) + [ 2*R + spatial_shape[i] for i in range(self.pos_dim) ]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=device)
        
        if self.pos_dim == 2:
            H, W = spatial_shape
            all_h1s = self.all_h1s[:H, :W]
            all_w1s = self.all_w1s[:H, :W]
            all_h2s = self.all_h2s[:H, :W]
            all_w2s = self.all_w2s[:H, :W]
            padded_pos_biases[(all_h1s, all_w1s, all_h2s, all_w2s)] = self.biases
                
        # Remove padding. [H+2R, W+2R, H+2R, W+2R] => [H, W, H, W].
        pos_biases = padded_pos_biases[:, :, R:-R, R:-R]
            
        return pos_biases
        
class SETransInputFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim         = config.in_feat_dim  # 256
        self.pos_embed_dim    = self.feat_dim
        self.dropout          = nn.Dropout(config.hidden_dropout_prob)
        self.comb_norm_layer  = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        self.pos_code_type    = config.pos_code_type
        
        # if using SlidingPosBiases2D, do not add positional embeddings here.
        if config.pos_code_type != 'bias':
            self.pos_code_weight = config.pos_code_weight
            print0("Positional embedding weight: {:.3}".format(self.pos_code_weight))
        else:
            self.pos_code_weight   = 0
            
        # Box position encoding. no affine, but could have bias.
        # 2 channels => 1792 channels
        if config.pos_code_type == 'lsinu':
            self.pos_coder = LearnedSinuPosEmbedder(config.pos_dim, self.pos_embed_dim, omega=1, affine=False)
        elif config.pos_code_type == 'rand':
            self.pos_coder = RandPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.pos_code_type == 'sinu':
            self.pos_coder = SinuPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.pos_code_type == 'zero':
            self.pos_coder = ZeroEmbedder(self.pos_embed_dim)
        elif config.pos_code_type == 'bias':
            self.pos_coder = SlidingPosBiases2D(config.pos_dim, config.pos_bias_radius)

        self.cached_pos_code   = None
        self.cached_feat_shape = None
        
    # Cache the pos_code and feat_shape to avoid unnecessary generation time.
    # This is only used during inference. During training, pos_code is always generated each time it's used.
    # Otherwise the cached pos_code cannot receive proper gradients.
    def pos_code_lookup_cache(self, vis_feat_shape, device, voxels_pos_normed):
        if self.pos_code_type == 'bias':
            # Cache miss for 'bias' type of positional codes.
            if self.training or self.cached_pos_code is None or self.cached_feat_shape != vis_feat_shape:
                self.cached_pos_code    = self.pos_coder(vis_feat_shape, device)
                self.cached_feat_shape  = vis_feat_shape    \
            # else: self.cached_pos_code exists, and self.cached_feat_shape == vis_feat_shape.
            # Just return the cached pos_code.
        else:
            # Cache miss for all other type of positional codes.
            if self.training or self.cached_pos_code is None or self.cached_feat_shape != voxels_pos_normed.shape:
                self.cached_pos_code    = self.pos_coder(voxels_pos_normed)
                self.cached_feat_shape  = voxels_pos_normed.shape
            # else: self.cached_pos_code exists, and self.cached_feat_shape == voxels_pos_normed.shape.
            # Just return the cached pos_code.
        return self.cached_pos_code

    # return: [B0, num_voxels, 256]
    def forward(self, vis_feat, voxels_pos, return_pos_biases=True):
        # vis_feat:  [8, 256, 46, 62]
        batch, dim, ht, wd  = vis_feat.shape

        if self.pos_code_type != 'bias':
            # voxels_pos: [8, 46, 62, 2]
            voxels_pos_normed = voxels_pos / voxels_pos.max()
            # voxels_pos_normed: [B0, num_voxels, 2]
            # pos_embed:         [B0, num_voxels, 256]
            voxels_pos_normed = voxels_pos_normed.view(batch, ht * wd, -1)
            pos_embed   = self.pos_code_lookup_cache(vis_feat.shape, vis_feat.device, voxels_pos_normed)
            pos_biases  = None
        else:
            pos_embed   = 0
            # SlidingPosBiases2D() may be a bit slow. So only generate when necessary.
            if return_pos_biases:
                # pos_biases: [1, 1, H, W, H, W]
                pos_biases  = self.pos_code_lookup_cache(vis_feat.shape, vis_feat.device, None)
                # pos_biases: [1, 1, H*W, H*W]
                pos_biases  = pos_biases.reshape(1, 1, ht*wd, ht*wd)
            else:
                # Simply discard pos_biases. Used when encoding the 2nd frame.
                # As for cross-frame attention, only one group of pos_biases is required 
                # (added to the cross-frame attentio scores).
                # When encoding the 1st frame, pos_biases is already returned, no need 
                # another group of pos_biases.  
                pass

        vis_feat    = vis_feat.view(batch, dim, ht * wd).transpose(1, 2)
            
        feat_comb   = vis_feat + self.pos_code_weight * pos_embed
        feat_normed = self.comb_norm_layer(feat_comb)
        feat_normed = self.dropout(feat_normed)
                  
        if return_pos_biases:
            return feat_normed, pos_biases
        else:
            return feat_normed
