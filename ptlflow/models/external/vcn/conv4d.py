import pdb
import torch.nn as nn
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
from torch.nn import Conv2d

def conv4d(data,filters,bias=None,permute_filters=True,use_half=False):
    """
    This is done by stacking results of multiple 3D convolutions, and is very slow.
    Taken from https://github.com/ignacio-rocco/ncnet
    """
    b,c,h,w,d,t=data.size()

    data=data.permute(2,0,1,3,4,5).contiguous() # permute to avoid making contiguous inside loop    
        
    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters=filters.permute(2,0,1,3,4,5).contiguous() # permute to avoid making contiguous inside loop    

    c_out=filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(h,b,c_out,w,d,t),requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(h,b,c_out,w,d,t),requires_grad=data.requires_grad)
    
    padding=filters.size(0)//2
    if use_half:
        Z=Variable(torch.zeros(padding,b,c,w,d,t).half())
    else:
        Z=Variable(torch.zeros(padding,b,c,w,d,t))
    
    if data.is_cuda:
        Z=Z.cuda(data.get_device())    
        output=output.cuda(data.get_device())
        
    data_padded = torch.cat((Z,data,Z),0)
    

    for i in range(output.size(0)): # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i,:,:,:,:,:]=F.conv3d(data_padded[i+padding,:,:,:,:,:], 
                                     filters[padding,:,:,:,:,:], bias=bias, stride=1, padding=padding)
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1,padding+1):
            output[i,:,:,:,:,:]=output[i,:,:,:,:,:]+F.conv3d(data_padded[i+padding-p,:,:,:,:,:], 
                                                             filters[padding-p,:,:,:,:,:], bias=None, stride=1, padding=padding)
            output[i,:,:,:,:,:]=output[i,:,:,:,:,:]+F.conv3d(data_padded[i+padding+p,:,:,:,:,:], 
                                                             filters[padding+p,:,:,:,:,:], bias=None, stride=1, padding=padding)

    output=output.permute(1,2,0,3,4,5).contiguous()
    return output

class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True): 
        # stride, dilation and groups !=1 functionality not tested 
        stride=1
        dilation=1
        groups=1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _quadruple(0), groups, bias)  
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor, 
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters=pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data=self.weight.data.permute(2,0,1,3,4,5).contiguous()
        self.use_half=False
    #    self.isbias = bias
    #    if not self.isbias:
    #        self.bn = torch.nn.BatchNorm1d(out_channels)


    def forward(self, input):
        out = conv4d(input, self.weight, bias=self.bias,permute_filters=not self.pre_permuted_filters,use_half=self.use_half) # filters pre-permuted in constructor
    #    if not self.isbias:
    #        b,c,u,v,h,w = out.shape
    #        out = self.bn(out.view(b,c,-1)).view(b,c,u,v,h,w)
        return out

class fullConv4d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        super(fullConv4d, self).__init__()
        self.conv = Conv4d(in_channels, out_channels, kernel_size, bias=bias, pre_permuted_filters=pre_permuted_filters)
        self.isbias = bias
        if not self.isbias:
            self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, input):
        out = self.conv(input)
        if not self.isbias:
            b,c,u,v,h,w = out.shape
            out = self.bn(out.view(b,c,-1)).view(b,c,u,v,h,w)
        return out

class butterfly4D(torch.nn.Module):
    '''
    butterfly 4d
    '''
    def __init__(self, fdima, fdimb, withbn=True, full=True,groups=1):
        super(butterfly4D, self).__init__()
        self.proj = nn.Sequential(projfeat4d(fdima, fdimb, 1, with_bn=withbn,groups=groups),
                                  nn.ReLU(inplace=True),)
        self.conva1 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)
        self.conva2 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(2,1,1),full=full,groups=groups)
        self.convb3 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
        self.convb2 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)
        self.convb1 = sepConv4dBlock(fdimb,fdimb,with_bn=withbn, stride=(1,1,1),full=full,groups=groups)

    #@profile
    def forward(self,x):
        out = self.proj(x)
        b,c,u,v,h,w = out.shape # 9x9

        out1 = self.conva1(out) # 5x5, 3
        _,c1,u1,v1,h1,w1 = out1.shape

        out2 = self.conva2(out1) # 3x3, 9
        _,c2,u2,v2,h2,w2 = out2.shape

        out2 = self.convb3(out2) # 3x3, 9

        tout1 = F.interpolate(out2.view(b,c,u2,v2,-1),(u1,v1,h2*w2),mode='trilinear', align_corners=False).view(b,c,u1,v1,h2,w2) # 5x5
        tout1 = F.interpolate(tout1.view(b,c,-1,h2,w2),(u1*v1,h1,w1),mode='trilinear', align_corners=False).view(b,c,u1,v1,h1,w1) # 5x5
        out1 = tout1 + out1
        out1 = self.convb2(out1)

        tout = F.interpolate(out1.view(b,c,u1,v1,-1),(u,v,h1*w1),mode='trilinear', align_corners=False).view(b,c,u,v,h1,w1)
        tout = F.interpolate(tout.view(b,c,-1,h1,w1),(u*v,h,w),mode='trilinear', align_corners=False).view(b,c,u,v,h,w)
        out = tout + out
        out = self.convb1(out)

        return out



class projfeat4d(torch.nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride, with_bn=True,groups=1):
        super(projfeat4d, self).__init__()
        self.with_bn = with_bn
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, 1, (stride,stride,1), padding=0,bias=not with_bn,groups=groups)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self,x):
        b,c,u,v,h,w = x.size()
        x = self.conv1(x.view(b,c,u,v,h*w))
        if self.with_bn:
            x = self.bn(x)
        _,c,u,v,_ = x.shape
        x = x.view(b,c,u,v,h,w)
        return x

class sepConv4d(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 3D convolutions
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1), with_bn=True, ksize=3, full=True,groups=1):
        super(sepConv4d, self).__init__()
        bias = not with_bn
        self.isproj = False
        self.stride = stride[0]
        expand = 1

        if with_bn:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0,groups=groups),
                                          nn.BatchNorm2d(out_planes))
            if full:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=(1,self.stride,self.stride), bias=bias, padding=(0,ksize//2,ksize//2),groups=groups),
                                           nn.BatchNorm3d(in_planes))
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=1,                           bias=bias, padding=(0,ksize//2,ksize//2),groups=groups),
                                           nn.BatchNorm3d(in_planes))
            self.conv2 = nn.Sequential(nn.Conv3d(in_planes, in_planes*expand, (ksize,ksize,1), stride=(self.stride,self.stride,1), bias=bias, padding=(ksize//2,ksize//2,0),groups=groups),
                                       nn.BatchNorm3d(in_planes*expand))
        else:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0,groups=groups)
            if full:
                self.conv1 = nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=(1,self.stride,self.stride), bias=bias, padding=(0,ksize//2,ksize//2),groups=groups)
            else:
                self.conv1 = nn.Conv3d(in_planes*expand, in_planes, (1,ksize,ksize), stride=1,                           bias=bias, padding=(0,ksize//2,ksize//2),groups=groups)
            self.conv2 = nn.Conv3d(in_planes, in_planes*expand, (ksize,ksize,1), stride=(self.stride,self.stride,1), bias=bias, padding=(ksize//2,ksize//2,0),groups=groups)
        self.relu = nn.ReLU(inplace=True)
        
    #@profile
    def forward(self,x):
        b,c,u,v,h,w = x.shape
        x = self.conv2(x.view(b,c,u,v,-1)) # WTA convolution over (u,v)
        b,c,u,v,_ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b,c,-1,h,w)) # spatial convolution over (x,y)
        b,c,_,h,w = x.shape

        if self.isproj:
            x = self.proj(x.view(b,c,-1,w))
        x = x.view(b,-1,u,v,h,w)
        return x


class sepConv4dBlock(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 2D convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1), with_bn=True, full=True,groups=1):
        super(sepConv4dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            if full:
                self.downsample = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn,ksize=1, full=full,groups=groups)
            else:
                self.downsample = projfeat4d(in_planes, out_planes,stride[0], with_bn=with_bn,groups=groups)
        self.conv1 = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn, full=full ,groups=groups)
        self.conv2 = sepConv4d(out_planes, out_planes,(1,1,1), with_bn=with_bn, full=full,groups=groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    #@profile
    def forward(self,x):
        out = self.relu1(self.conv1(x))
        if self.downsample:
            x = self.downsample(x)
        out = self.relu2(x + self.conv2(out))
        return out


##import torch.backends.cudnn as cudnn
##cudnn.benchmark = True
#import time
##im = torch.randn(9,64,9,160,224).cuda()
##net = torch.nn.Conv3d(64, 64, 3).cuda()
##net = Conv4d(1,1,3,bias=True,pre_permuted_filters=True).cuda()
##net = sepConv4dBlock(2,2,stride=(1,1,1)).cuda()
#
##im = torch.randn(1,16,9,9,96,320).cuda()
##net = sepConv4d(16,16,with_bn=False).cuda()
#
##im = torch.randn(1,16,81,96,320).cuda()
##net = torch.nn.Conv3d(16,16,(1,3,3),padding=(0,1,1)).cuda()
#
##im = torch.randn(1,16,9,9,96*320).cuda()
##net = torch.nn.Conv3d(16,16,(3,3,1),padding=(1,1,0)).cuda()
#
##im = torch.randn(10000,10,9,9).cuda()
##net = torch.nn.Conv2d(10,10,3,padding=1).cuda()
#
##im = torch.randn(81,16,96,320).cuda()
##net = torch.nn.Conv2d(16,16,3,padding=1).cuda()
#c=   int(16 *1)
#cp = int(16 *1)
#h=int(96  *4)
#w=int(320 *4)
#k=3
#im = torch.randn(1,c,h,w).cuda()
#net = torch.nn.Conv2d(c,cp,k,padding=k//2).cuda()
#
#im2 = torch.randn(cp,k*k*c).cuda()
#im1 = F.unfold(im, (k,k), padding=k//2)[0]
# 
#
#net(im)
#net(im)
#torch.mm(im2,im1)
#torch.mm(im2,im1)
#torch.cuda.synchronize()
#beg = time.time()
#for i in range(100):
#    net(im)
#    #im1 = F.unfold(im, (k,k), padding=k//2)[0]
#    torch.mm(im2,im1)
#torch.cuda.synchronize()
#print('%f'%((time.time()-beg)*10.))