import torch
import torch.nn as nn
from timm.models.layers import DropPath

from timm.models.layers import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

try:
    from torch.utils.cpp_extension import load
    print("[Start compiling NAT]")
    nattenav_cuda = load(
        'nattenav_cuda', ['core/cuda/nattenav_cuda.cpp', 'core/cuda/nattenav_cuda_kernel.cu'], verbose=True)
    print("[Finished 1/2]")
    nattenqkrpb_cuda = load(
        'nattenqkrpb_cuda', ['core/cuda/nattenqkrpb_cuda.cpp', 'core/cuda/nattenqkrpb_cuda_kernel.cu'], verbose=False)
    print("[Finished 2/2]")
except:
    print("Failed to load nat cuda")
    exit()

class NATTENAVFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = nattenav_cuda.forward(
                attn, 
                value)[0]
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_attn, d_value = outputs
        return d_attn, d_value


class NATTENQKRPBFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_cuda.forward(
                query,
                key,
                rpb.contiguous())[0]
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb

class selfattentionlayer_nat(nn.Module):
    def __init__(self, cfg):
        super(selfattentionlayer_nat, self).__init__()
        dropout = cfg.dropout
        droppath = cfg.droppath

        self.cfg = cfg
        kernel_size = 11
        qk_dim = cfg.attn_dim
        self.num_heads = qk_dim // 16
        self.expand_factor = cfg.expand_factor

        self.context_proj = nn.Linear(cfg.encoder_latent_dim, cfg.vert_c_dim, bias=True)
        self.context_norm = nn.LayerNorm(cfg.encoder_latent_dim)
        self.norm1 = nn.LayerNorm(cfg.cost_latent_dim+cfg.vert_c_dim)
        self.norm1_v = nn.LayerNorm(cfg.cost_latent_dim)
        self.norm2 = nn.LayerNorm(cfg.cost_latent_dim)

        self.q, self.k, self.v = nn.Linear(cfg.cost_latent_dim+cfg.vert_c_dim, qk_dim, bias=True), nn.Linear(cfg.cost_latent_dim+cfg.vert_c_dim, qk_dim, bias=True), nn.Linear(cfg.cost_latent_dim, qk_dim, bias=True)

        self.proj = nn.Linear(cfg.attn_dim+cfg.cost_latent_dim, cfg.cost_latent_dim, bias=True)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(cfg.cost_latent_dim, cfg.cost_latent_dim*cfg.expand_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.cost_latent_dim*cfg.expand_factor, cfg.cost_latent_dim),
            nn.Dropout(dropout)
        )

        self.dim = qk_dim
        self.rpb = nn.Parameter(torch.zeros(self.num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        self.scale = 16 ** -0.5
    def forward(self, x, size, context):
        
        B, HW, C = x.shape
        B_context, C_context, H, W = context.shape
        
        short_cut = x
        context = context.reshape(B_context, C_context, HW).permute(0, 2, 1)
        context = self.context_proj(self.context_norm(context))
        context = context.repeat(B//context.shape[0], 1, 1)
        qk = torch.cat([x, context], dim=-1)
        qk = self.norm1(qk)
        v = self.norm1_v(x)

        q = (self.q(qk) * self.scale).reshape(B, H, W, self.num_heads, 16).permute(0, 3, 1, 2, 4)
        k = self.k(qk).reshape(B, H, W, self.num_heads, 16).permute(0, 3, 1, 2, 4)
        v = self.v(v).reshape(B, H, W, self.num_heads, 16).permute(0, 3, 1, 2, 4)

        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0,2,3,1,4).reshape(B, HW, -1)
        x = self.proj(torch.cat([x, short_cut],dim=2))
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

from .twins import Block

class NATwins(nn.Module):
    def __init__(self, cfg):
        super(NATwins, self).__init__()
        
        self.cfg = cfg
        self.dim = dim = cfg.cost_latent_dim
        self.num_heads = num_heads = 8
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = cfg.droppath
        drop_rate = cfg.dropout
        attn_drop_rate=0.

        self.local_block = selfattentionlayer_nat(cfg)
        self.global_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=1, with_rpe=True, vert_c_dim=cfg.vert_c_dim, encoder_latent_dim=cfg.encoder_latent_dim)


    def forward(self, x, size, context):
        
        x = self.local_block(x, size, context)
        x = self.global_block(x, size, context)

        return x
