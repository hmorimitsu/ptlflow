from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1,with_bn=True):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1

        if with_bn:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
        else:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation,with_bn=False)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
        self.downsample = downsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        bias = not with_bn

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        bias = not with_bn
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True, levels=4):
        super(pyramidPooling, self).__init__()
        self.levels = levels

        self.paths = []
        for i in range(levels):
            self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in np.linspace(1,min(h,w)//2,self.levels,dtype=int):
            k_sizes.append((int(h/pool_size), int(w/pool_size)))
            strides.append((int(h/pool_size), int(w/pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]

        pp_sum = x

        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
            pp_sum = pp_sum + 1./self.levels*out
        pp_sum = self.relu(pp_sum/2.)

        return pp_sum

class pspnet(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True,groups=1):
        super(pspnet, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1))
        self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1)

        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)
            self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.interpolate(conv6, [conv5.size()[2],conv5.size()[3]],mode='bilinear', align_corners=False)
        concat5 = torch.cat((conv5,self.upconv6[1](conv6x)),dim=1)
        conv5 = self.iconv5(concat5) 

        conv5x = F.interpolate(conv5, [conv4.size()[2],conv4.size()[3]],mode='bilinear', align_corners=False)
        concat4 = torch.cat((conv4,self.upconv5[1](conv5x)),dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.interpolate(conv4, [rconv3.size()[2],rconv3.size()[3]],mode='bilinear', align_corners=False)
        concat3 = torch.cat((rconv3,self.upconv4[1](conv4x)),dim=1)
        conv3 = self.iconv3(concat3) 

        conv3x = F.interpolate(conv3, [pool1.size()[2],pool1.size()[3]],mode='bilinear', align_corners=False)
        concat2 = torch.cat((pool1,self.upconv3[1](conv3x)),dim=1)
        conv2 = self.iconv2(concat2) 

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)
            return proj6,proj5,proj4,proj3,proj2
        else:
            return conv6, conv5, conv4, conv3, conv2


class pspnet_s(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True,groups=1):
        super(pspnet_s, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        #self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=False),
        #                             conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
        #                                         padding=1, stride=1))
        #self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
        #                                         padding=1, stride=1)

        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)
            #self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.interpolate(conv6, [conv5.size()[2],conv5.size()[3]],mode='bilinear', align_corners=False)
        concat5 = torch.cat((conv5,self.upconv6[1](conv6x)),dim=1)
        conv5 = self.iconv5(concat5) 

        conv5x = F.interpolate(conv5, [conv4.size()[2],conv4.size()[3]],mode='bilinear', align_corners=False)
        concat4 = torch.cat((conv4,self.upconv5[1](conv5x)),dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.interpolate(conv4, [rconv3.size()[2],rconv3.size()[3]],mode='bilinear', align_corners=False)
        concat3 = torch.cat((rconv3,self.upconv4[1](conv4x)),dim=1)
        conv3 = self.iconv3(concat3) 

        #conv3x = F.interpolate(conv3, [pool1.size()[2],pool1.size()[3]],mode='bilinear', align_corners=False)
        #concat2 = torch.cat((pool1,self.upconv3[1](conv3x)),dim=1)
        #conv2 = self.iconv2(concat2) 

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
        #    proj2 = self.proj2(conv2)
        #    return proj6,proj5,proj4,proj3,proj2
            return proj6,proj5,proj4,proj3
        else:
        #    return conv6, conv5, conv4, conv3, conv2
            return conv6, conv5, conv4, conv3