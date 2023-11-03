import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from ..encoders import twins_svt_large, convnext_large
from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder

class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        
        H1, W1, H2, W2 = cfg.pic_size
        H_offset = (H1-H2) // 2
        W_offset = (W1-W2) // 2
        cfg.H_offset = H_offset
        cfg.W_offset = W_offset

        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain, del_layers=cfg.del_layers)
        elif "jh" in cfg.cnet:
            self.context_encoder = twins_svt_large_jihao(pretrained=self.cfg.pretrain, del_layers=cfg.del_layers, version=cfg.cnet)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        elif cfg.cnet == 'convnext':
            self.context_encoder = convnext_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'nat':
            self.context_encoder = nat_base(pretrained=self.cfg.pretrain)

        if cfg.pretrain_mode:
            print("[In pretrain mode, freeze context encoder]")
            for param in self.context_encoder.parameters():
                param.requires_grad = False


    def forward(self, image1, image2, mask=None, output=None, flow_init=None):
        if self.cfg.pretrain_mode:
            loss = self.pretrain_forward(image1, image2, mask=mask, output=output)
            return loss
        else:
            # Following https://github.com/princeton-vl/RAFT/
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

            data = {}
            
            context, _ = self.context_encoder(image1)
            context_quater = None

            cost_memory, cost_patches, feat_s_quater, feat_t_quater = self.memory_encoder(image1, image2, data, context)

            flow_predictions = self.memory_decoder(cost_memory, context, context_quater, feat_s_quater, feat_t_quater, data, flow_init=flow_init, cost_patches=cost_patches)

            return flow_predictions
    
    def pretrain_forward(self, image1, image2, mask=None, output=None, flow_init=None):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        H_offset = self.cfg.H_offset
        W_offset = self.cfg.W_offset
        H2, W2 = self.cfg.pic_size[2:]
        
        image1_inner = image1[:, :, H_offset:H_offset+H2, W_offset:W_offset+W2]
        image2_inner = image2[:, :, H_offset:H_offset+H2, W_offset:W_offset+W2]
        
        data = {}
        
        context, _ = self.context_encoder(image1_inner)
            
        cost_memory, cost_patches = self.memory_encoder.pretrain_forward(image1, image2, image1_inner, image2_inner, data, context , mask=mask)

        loss = self.memory_decoder.pretrain_forward(cost_memory, context, data, flow_init=flow_init, cost_patches=cost_patches)

        return loss
