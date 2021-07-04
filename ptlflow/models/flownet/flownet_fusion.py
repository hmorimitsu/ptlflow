'''
Portions of this code copyright 2017, Clement Pinard
'''
from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import init

from .submodules import *
from .flownet_base import FlowNetBase


class FlowNetFusion(FlowNetBase):
    def __init__(self,
                 args: Namespace):
        args.loss_start_scale = 1
        args.loss_num_scales = 3
        super(FlowNetFusion, self).__init__(args)

        self.conv0   = conv(self.args.batch_norm,  11,   64)
        self.conv1   = conv(self.args.batch_norm,  64,   64, stride=2)
        self.conv1_1 = conv(self.args.batch_norm,  64,   128)
        self.conv2   = conv(self.args.batch_norm,  128,  128, stride=2)
        self.conv2_1 = conv(self.args.batch_norm,  128,  128)

        self.deconv1 = deconv(128,32)
        self.deconv0 = deconv(162,16)

        self.inter_conv1 = i_conv(self.args.batch_norm,  162,   32)
        self.inter_conv0 = i_conv(self.args.batch_norm,  82,   16)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        x = inputs['images']

        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2       = self.predict_flow2(out_conv2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        
        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1       = self.predict_flow1(out_interconv1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        
        concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0       = self.predict_flow0(out_interconv0)

        outputs = {}

        if self.training:
            outputs['flow_preds'] = [flow0.float(),flow1.float(),flow2.float()]
            outputs['flows'] = flow0[:, None]
        else:
            outputs['flows'] = flow0[:, None]

        return outputs
