import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from ..encoders import twins_svt_large, convnext_Xlarge_4x, convnext_base_2x
from .corr import CorrBlock, OLCorrBlock, AlternateCorrBlock
from ...utils.utils import bilinear_sampler, coords_grid, upflow8
from .gma import Attention, Aggregate

from torchvision.utils import save_image

autocast = torch.cuda.amp.autocast

class MOFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = hdim = self.cfg.feat_dim // 2
        self.context_dim = cdim = self.cfg.feat_dim // 2

        cfg.corr_radius = 4

        # feature network, context network, and update block
        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn='instance')
        elif cfg.cnet == 'convnext_Xlarge_4x':
            print("[Using convnext_Xlarge_4x as context encoder]")
            self.cnet = convnext_Xlarge_4x(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'convnext_base_2x':
            print("[Using convnext_base_2x as context encoder]")
            self.cnet = convnext_base_2x(pretrained=self.cfg.pretrain)
        
        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        elif cfg.fnet == 'convnext_Xlarge_4x':
            print("[Using convnext_Xlarge_4x as feature encoder]")
            self.fnet = convnext_Xlarge_4x(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'convnext_base_2x':
            print("[Using convnext_base_2x as feature encoder]")
            self.fnet = convnext_base_2x(pretrained=self.cfg.pretrain)

        hidden_dim_ratio = 256 // cfg.feat_dim        

        if self.cfg.Tfusion == 'stack':
            print("[Using stack.]")
            self.cfg.cost_heads_num = 1
            from .stack import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128//hidden_dim_ratio)
        # elif self.cfg.Tfusion == 'resstack':
        #     print("[Using resstack.]")
        #     self.cfg.cost_heads_num = 1
        #     from .resstack import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
        #     self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128)
        # elif self.cfg.Tfusion == 'stackcat':
        #     print("[Using stackcat.]")
        #     self.cfg.cost_heads_num = 1
        #     from .stackcat import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
        #     self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128)
        

        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        gma_down_ratio = 256 // cfg.feat_dim

        self.att = Attention(args=self.cfg, dim=128//hidden_dim_ratio, heads=1, max_pos_size=160, dim_head=128//hidden_dim_ratio)

        if self.cfg.context_3D:
            print("[Using 3D Conv on context feature.]")
            self.context_3D = nn.Sequential(
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
            )

    def initialize_flow(self, img, bs, down_ratio):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(bs, H // down_ratio, W // down_ratio).to(img.device)
        coords1 = coords_grid(bs, H // down_ratio, W // down_ratio).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    
    def upsample_flow_4x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)
    
    def upsample_flow_2x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 2 * H, 2 * W)
    
    

    def forward(self, images, data={}, flow_init=None):

        down_ratio = self.cfg.down_ratio

        B, N, _, H, W = images.shape

        images = 2 * (images / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.cfg.mixed_precision):
            fmaps = self.fnet(images.reshape(B*N, 3, H, W)).reshape(B, N, -1, H//down_ratio, W//down_ratio)
        fmaps = fmaps.float()

        if self.cfg.corr_fn == "default":
            corr_fn = CorrBlock
        elif self.cfg.corr_fn == "efficient":
            corr_fn = AlternateCorrBlock
        forward_corr_fn = corr_fn(fmaps[:, 1:N-1, ...].reshape(B*(N-2), -1, H//down_ratio, W//down_ratio), fmaps[:, 2:N, ...].reshape(B*(N-2), -1, H//down_ratio, W//down_ratio), num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        backward_corr_fn = corr_fn(fmaps[:, 1:N-1, ...].reshape(B*(N-2), -1, H//down_ratio, W//down_ratio), fmaps[:, 0:N-2, ...].reshape(B*(N-2), -1, H//down_ratio, W//down_ratio), num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)

        with autocast(enabled=self.cfg.mixed_precision):
            cnet = self.cnet(images[:, 1:N-1, ...].reshape(B*(N-2), 3, H, W))
            if self.cfg.context_3D:
                #print("!@!@@#!@#!@")
                cnet = cnet.reshape(B, N-2, -1, H//2, W//2).permute(0, 2, 1, 3, 4)
                cnet = self.context_3D(cnet) + cnet
                #print(cnet.shape)
                cnet = cnet.permute(0, 2, 1, 3, 4).reshape(B*(N-2), -1, H//down_ratio, W//down_ratio)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)
        
        forward_coords1, forward_coords0 = self.initialize_flow(images[:, 0, ...], bs=B*(N-2), down_ratio=down_ratio)
        backward_coords1, backward_coords0 = self.initialize_flow(images[:, 0, ...], bs=B*(N-2), down_ratio=down_ratio)

        flow_predictions = [] # forward flows followed by backward flows

        motion_hidden_state = None

        for itr in range(self.cfg.decoder_depth):
            
            forward_coords1 = forward_coords1.detach()
            backward_coords1 = backward_coords1.detach()

            forward_corr = forward_corr_fn(forward_coords1)
            backward_corr = backward_corr_fn(backward_coords1)

            forward_flow = forward_coords1 - forward_coords0
            backward_flow = backward_coords1 - backward_coords0
            
            with autocast(enabled=self.cfg.mixed_precision):
                net, motion_hidden_state, up_mask, delta_flow = self.update_block(net, motion_hidden_state, inp, forward_corr, backward_corr, forward_flow, backward_flow, forward_coords0, attention, bs=B)

            forward_up_mask, backward_up_mask = torch.split(up_mask, [down_ratio**2*9, down_ratio**2*9], dim=1)

            forward_coords1 = forward_coords1 + delta_flow[:, 0:2, ...]
            backward_coords1 = backward_coords1 + delta_flow[:, 2:4, ...]

            # upsample predictions
            if down_ratio == 4:
                forward_flow_up = self.upsample_flow_4x(forward_coords1-forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow_4x(backward_coords1-backward_coords0, backward_up_mask)
            elif down_ratio == 2:
                forward_flow_up = self.upsample_flow_2x(forward_coords1-forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow_2x(backward_coords1-backward_coords0, backward_up_mask)
            elif down_ratio == 8:
                forward_flow_up = self.upsample_flow(forward_coords1-forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow(backward_coords1-backward_coords0, backward_up_mask)

            flow_predictions.append(torch.cat([forward_flow_up.reshape(B, N-2, 2, H, W), backward_flow_up.reshape(B, N-2, 2, H, W)], dim=1))

        if self.training:
            return flow_predictions
        else:
            return flow_predictions[-1], flow_predictions[-1]
