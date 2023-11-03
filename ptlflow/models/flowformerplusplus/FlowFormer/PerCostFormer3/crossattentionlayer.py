import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .attention import BroadMultiHeadAttention

class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token, size=None, ids_keep=None):
        """
            x: [BH1W1, H3W3, D]
        """
        
        if ids_keep is not None:
            tgt_token = torch.gather(tgt_token, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, tgt_token.shape[-1]))

        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)

        x = self.multi_head_attn(q, k, v)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

class CrossAttentionLayer_two_level(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer_two_level, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def compute_indices(self, size):
        H, W = size
        hs = [(0, int(H/2)), (int(H/2), H)]
        ws = [(0, int(W/3)), (int(W/3), int(W/3)*2), (int(W/3)*2, W)]
        return [(*h, *w) for h in hs for w in ws]


    def forward(self, query, tgt_token, size):
        """
            x: [BH1W1, H3W3, D]
            q ([1, 8, 128])  k torch.Size([5704, 48, 128]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        B, HW, C = k.shape
        H, W = size

        res = []
        res.append(self.multi_head_attn(q[:, :2, :], k, v))

        k = k.reshape(B, H, W, C)
        v = v.reshape(B, H, W, C)

        indices = self.compute_indices(size)
        for idx, corners in enumerate(indices):
            h0, h1, w0, w1 = corners
            res.append(self.multi_head_attn(q[:, 2+idx:3+idx, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))

        x = torch.cat(res, dim=1)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

class CrossAttentionLayer_convk3s2(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer_convk3s2, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

        # coarse level
        self.down_sample = nn.Conv2d(query_token_dim, query_token_dim, 3, 2, 1)
        self.norm1_coarse = nn.LayerNorm(query_token_dim)
        self.norm2_coarse = nn.LayerNorm(query_token_dim)
        self.q_coarse, self.k_coarse, self.v_coarse = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)
        self.proj_coarse = nn.Linear(v_dim, query_token_dim)
        self.ffn_coarse = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def compute_indices_fine(self, size):
        H, W = size
        hs = [(0, int(H/2)), (int(H/2), H)]
        ws = [(0, int(W/3)), (int(W/3), int(W/3)*2), (int(W/3)*2, W)]
        return [(*h, *w) for h in hs for w in ws]
    
    def compute_indices_coarse(self, size):
        H, W = size
        hs = [(0, int(H/2)), (int(H/2), H)]
        ws = [(0, int(W/2)), (int(W/2), W)]
        return [(*h, *w) for h in hs for w in ws]

    def forward(self, query, tgt_token, size):
        """
            x: [BH1W1, H3W3, D]
            q ([1, 8, 128])  k torch.Size([5704, 48, 128]
        """
        assert query.shape[1] == 10, "Error: k3s2 version, query num must be 10"

        short_cut = query
        query = self.norm1(query[:, :6, :])

        # fine level
        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        B, HW, C = k.shape
        H, W = size

        res_fine = []

        k = k.reshape(B, H, W, C)
        v = v.reshape(B, H, W, C)

        indices = self.compute_indices_fine(size)
        for idx, corners in enumerate(indices):
            h0, h1, w0, w1 = corners
            res_fine.append(self.multi_head_attn(q[:, idx:idx+1, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))

        x = torch.cat(res_fine, dim=1)

        x = short_cut[:, :6, :] + self.proj_drop(self.proj(x))

        x_fine = x + self.drop_path(self.ffn(self.norm2(x)))

        # coarse_level
        tgt_token_coarse = tgt_token.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        tgt_token_coarse = self.down_sample(tgt_token_coarse).permute(0, 2, 3, 1)
        H, W = tgt_token_coarse.shape[1:3]

        query = self.norm1_coarse(short_cut[:, 6:10, :])
        q, k, v = self.q_coarse(query), self.k_coarse(tgt_token_coarse), self.v_coarse(tgt_token_coarse)
        res_coarse = []

        indices = self.compute_indices_coarse((H, W))
        for idx, corners in enumerate(indices):
            h0, h1, w0, w1 = corners
            res_coarse.append(self.multi_head_attn(q[:, idx:idx+1, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))

        x = torch.cat(res_coarse, dim=1)

        x = short_cut[:, 6:10, :] + self.proj_drop(self.proj_coarse(x))

        x_coarse = x + self.drop_path(self.ffn_coarse(self.norm2_coarse(x)))

        x = torch.cat([x_fine, x_coarse], dim=1)
        
        return x

class CrossAttentionLayer_two_level_rep(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer_two_level_rep, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def compute_indices(self, size):
        H, W = size
        hs = [(0, int(H/2)), (int(H/2), H)]
        ws = [(0, int(W/3)), (int(W/3), int(W/3)*2), (int(W/3)*2, W)]
        return [(*h, *w) for h in hs for w in ws]


    def forward(self, query, tgt_token, size):
        """
            x: [BH1W1, H3W3, D]
            q ([1, 8, 128])  k torch.Size([5704, 48, 128]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        B, HW, C = k.shape
        H, W = size

        res = []
        
        k = k.reshape(B, H, W, C)
        v = v.reshape(B, H, W, C)

        indices = self.compute_indices(size)

        h0, h1, w0, w1 = indices[0]
        res.append(self.multi_head_attn(q[:, 0:1, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))
        h0, h1, w0, w1 = indices[1]
        res.append(self.multi_head_attn(q[:, 1:2, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))

        for idx, corners in enumerate(indices):
            h0, h1, w0, w1 = corners
            res.append(self.multi_head_attn(q[:, 2+idx:3+idx, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))

        x = torch.cat(res, dim=1)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

class CrossAttentionLayer_34(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer_34, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def compute_indices(self, size):
        H, W = size
        hs = [(0, int(H/3)), (int(H/3), int(H/3)*2), (int(H/3)*2, H)]
        ws = [(0, int(W/4)), (int(W/4), int(W/4)*2), (int(W/4)*2, int(W/4)*3), (int(W/4)*3, W)]
        return [(*h, *w) for h in hs for w in ws]


    def forward(self, query, tgt_token, size):
        """
            x: [BH1W1, H3W3, D]
            q ([1, 8, 128])  k torch.Size([5704, 48, 128]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        B, HW, C = k.shape
        H, W = size

        res = []

        k = k.reshape(B, H, W, C)
        v = v.reshape(B, H, W, C)

        indices = self.compute_indices(size)
        for idx, corners in enumerate(indices):
            h0, h1, w0, w1 = corners
            res.append(self.multi_head_attn(q[:, idx:idx+1, :], k[:, h0:h1, w0:w1, :].reshape(B, -1, C), v[:, h0:h1, w0:w1, :].reshape(B, -1, C)))

        x = torch.cat(res, dim=1)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x
