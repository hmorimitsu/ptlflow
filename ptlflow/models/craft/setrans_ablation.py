import math

import torch
import torch.nn as nn

def positionalencoding2d(pos_embed_dim, height, width):
    """
    :param pos_embed_dim: dimension of the model embeddings
    :param height: height of the positions
    :param width: width of the positions
    :return: height * width * pos_embed_dim matrix
    """
    if pos_embed_dim % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(pos_embed_dim))
    pe = torch.zeros(pos_embed_dim, height, width)
    # Each dimension use half of pos_embed_dim
    pos_embed_dim = int(pos_embed_dim / 2)
    div_term = torch.exp(torch.arange(0., pos_embed_dim, 2) *
                         -(math.log(10000.0) / pos_embed_dim))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:pos_embed_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:pos_embed_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[pos_embed_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[pos_embed_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = pe.permute(1, 2, 0)
    return pe
    
class RandPosEmbedder(nn.Module):
    def __init__(self, pos_dim, pos_embed_dim, shape, affine):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        height, width = shape
        self.pos_embed = nn.Embedding(height * width, pos_embed_dim)
        self.pos_embed_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        print("Random discrete embedder for positional encoding ablation created")
        
    def forward(self, pos_normed):
        B, N, D = pos_normed.shape
        pos_embed_1 = self.pos_embed.weight
        pos_embed_out_1 = self.pos_embed_norm_layer(pos_embed_1)
        pos_embed_out = pos_embed_out_1.unsqueeze(0).repeat((B, 1, 1))
        return pos_embed_out

class SinuPosEmbedder(nn.Module):
    def __init__(self, pos_dim, pos_embed_dim, shape, affine):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_embed = positionalencoding2d(pos_embed_dim, shape[0], shape[1])
        self.pos_embed = self.pos_embed.cuda().reshape((shape[0] * shape[1], pos_embed_dim))
        print("Sinu embedder for positional encoding ablation created")

    def forward(self, pos_normed):
        B, N, D = pos_normed.shape
        pos_embed_out = self.pos_embed.unsqueeze(0).repeat((B, 1, 1))
        return pos_embed_out

class ZeroEmbedder(nn.Module):
    def __init__(self, pos_embed_dim):
        super().__init__()
        self.pos_embed_dim = pos_embed_dim
        print("Zero embedder for positional encoding ablation created")
        
    def forward(self, pos_normed):
        B, N, D = pos_normed.shape
        zero_pos_embed = torch.zeros(B, N, self.pos_embed_dim, requires_grad=False).cuda()
        return zero_pos_embed
        
# MM*Mid, MM*Output are the same as in segtran. Just for use by ExpandedFeatTrans.
class MMPrivateMid(nn.Module):
    def __init__(self, config):
        super(MMPrivateMid, self).__init__()
        # Use 1x1 convolution as a group linear layer.
        # Equivalent to each group going through a respective nn.Linear().
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.group_linear   = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.mid_act_fn     = config.act_fun

    def forward(self, x):
        x_trans = self.group_linear(x)      # [B0, 1024*8, 50] -> [B0, 1024*8, 50]
        x_act   = self.mid_act_fn(x_trans)  # [B0, 1024*8, 50]
        return x

class MMSharedMid(nn.Module):
    def __init__(self, config):
        super(MMSharedMid, self).__init__()
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.shared_linear  = nn.Linear(self.feat_dim, self.feat_dim)
        self.mid_act_fn     = config.act_fun

    # x: [B0, 1024*8, 50] or [B0, 8, 50, 1024]
    def forward(self, x):
        if len(x.shape) == 3:
            # shape_4d: [B0, 8, 1024, 50].
            shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
            # x_4d: [B0, 8, 50, 1024].
            x_4d        = x.view(shape_4d).permute([0, 1, 3, 2])
            reshaped    = True
        else:
            x_4d        = x
            reshaped    = False

        x_trans         = self.shared_linear(x_4d)
        x_act           = self.mid_act_fn(x_trans)

        if reshaped:
            # restore the original shape
            x_act       = x_act.permute([0, 1, 3, 2]).reshape(x.shape)

        return x_act

# MMPrivateOutput/MMSharedOutput <- ExpandedFeatTrans <- SelfAttFeatTrans <- SegtranFusionEncoder.
# MM***Output has a shortcut (residual) connection.
class MMPrivateOutput(nn.Module):
    def __init__(self, config):
        super(MMPrivateOutput, self).__init__()
        self.num_modes  = config.num_modes
        self.feat_dim   = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        self.group_linear = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1024*8, 50]
    def forward(self, x, shortcut):
        x        = self.group_linear(x)
        # x_comb: [B0, 1024*8, 50]. Residual connection.
        x_comb   = x + shortcut
        shape_4d = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        # x_comb_4d, x_drop_4d: [B0, 8, 50, 1024].
        x_comb_4d = x.view(shape_4d).permute([0, 1, 3, 2])
        x_drop_4d = self.dropout(x_comb_4d)
        x_normed = self.resout_norm_layer(x_drop_4d)
        return x_normed

# MMPrivateOutput/MMSharedOutput <- ExpandedFeatTrans <- SelfAttFeatTrans <- SegtranFusionEncoder.
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

    # x, shortcut: [B0, 1024*8, 50] or [B0, 8, 50, 1024]
    def forward(self, x, shortcut):
        # shape_4d: [B0, 8, 1024, 50].
        shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        if len(x.shape) == 3:
            x_4d    = x.view(shape_4d).permute([0, 1, 3, 2])
        else:
            x_4d    = x
        if len(shortcut.shape) == 3:
            shortcut_4d = shortcut.view(shape_4d).permute([0, 1, 3, 2])
        else:
            shortcut_4d = shortcut

        # x_4d, shortcut_4d: [B0, 8, 50, 1024].
        x_trans     = self.shared_linear(x_4d)
        # x_4d, x_comb: [B0, 8, 50, 1024]. Residual connection.
        x_comb      = x_trans + shortcut_4d
        x_drop      = self.dropout(x_comb)
        x_normed    = self.resout_norm_layer(x_drop)
        return x_normed

# MultiHeadFeatTrans <- SelfAttFeatTrans.
# MultiHeadFeatTrans has a residual connection.
# We "misuse" num_modes for num_heads, to avoid introducing extra config parameters.
class MultiHeadFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(MultiHeadFeatTrans, self).__init__()
        self.config = config
        self.name = name
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.num_modes = config.num_modes
        self.feat_dim_onehead = self.feat_dim // self.num_modes
        self.feat_dim_allhead = self.feat_dim_onehead * self.num_modes
        # first_linear is the value projection in other transformer implementations.
        # The output of first_linear will be divided into num_modes groups.
        # first_linear is always 'private' for each group, i.e.,
        # parameters are not shared (parameter sharing makes no sense).
        self.first_linear = nn.Linear(self.in_feat_dim, self.feat_dim_allhead)
        
        print("%s: pool_modes_feat=concat, trans_output_type=%s" % \
                (self.name, config.trans_output_type))

        # Disable multiple modes for intermediate and output layers.
        config.num_modes = 1
        self.intermediate = MMSharedMid(self.config)

        if config.trans_output_type == 'shared':
            self.output = MMSharedOutput(config)
        elif config.trans_output_type == 'private':
            self.output = MMPrivateOutput(config)

        self.apply_attn_early = config.apply_attn_early

    def add_identity_bias(self):
        if self.config.feattrans_lin1_idbias_scale > 0:
            identity_weight = torch.diag(torch.ones(self.feat_dim)) * self.config.initializer_range \
                              * self.config.feattrans_lin1_idbias_scale
            # Only bias the weight of the first mode.
            # The total initial "weight mass" in each row is reduced by 1024*0.02*0.5.
            self.first_linear.weight.data[:self.feat_dim] = \
                self.first_linear.weight.data[:self.feat_dim] * 0.5 + identity_weight

    def forward(self, input_feat, attention_probs, attention_scores):
        # input_feat: [B0, 50, 1024], mm_first_feat: [B0, 50, 1024*8]
        mm_first_feat = self.first_linear(input_feat)
        # mm_first_feat_act after permute: [B0, 1024*8, 50]
        mm_first_feat = mm_first_feat.permute(0, 2, 1)

        if self.apply_attn_early:
            # shape_4d: [B0, 8, 1024, 50]
            shape_4d    = ( mm_first_feat.shape[0], self.num_modes, self.feat_dim_onehead, mm_first_feat.shape[2] )
            # mm_first_feat_4d: [B0, 8, 50, 1024]
            mm_first_feat_4d = mm_first_feat.view(shape_4d).permute([0, 1, 3, 2])
            mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)
            mm_first_feat_fusion_3d = mm_first_feat_fusion.permute([0, 1, 3, 2]).reshape(mm_first_feat.shape)
            mm_first_feat = mm_first_feat_fusion_3d

        # mm_mid_feat:   [B0, 1024*8, 50]. Group linear & gelu of mm_first_feat.
        mm_mid_feat   = self.intermediate(mm_first_feat)
        # mm_last_feat:  [B0, 8, 50, 1024]. Group/shared linear & residual & Layernorm
        mm_last_feat = self.output(mm_mid_feat, mm_first_feat)

        if (attention_probs is not None) and (not self.apply_attn_early):
            # matmul(t1, t2): (h1, w1), (w1, w2) => (h1, w2)
            # [B0, 8, 50, 50][B0, 8, 50, 1024] -> mm_trans_feat: [B0, 8, 50, 1024]
            mm_trans_feat = torch.matmul(attention_probs, mm_last_feat)
        else:
            mm_trans_feat = mm_last_feat

        trans_feat = mm_trans_feat.squeeze(1)

        # trans_feat: [B0, 50, 1024],   if pool_modes_feat != 'none',
        # or          [B0, 8, 50, 1024] if pool_modes_feat == 'none'.
        return trans_feat
