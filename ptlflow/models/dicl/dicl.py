from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler

from .loss_functions import MultiScale_UP
from ..base_model.base_model import BaseModel


class DICL_MODULE(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self):
        super(DICL_MODULE, self).__init__()

        self.match = nn.Sequential(
                        BasicConv(64, 96, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(96, 128, kernel_size=3, stride=2,    padding=1),   # down by 1/2
                        BasicConv(128, 128, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(128, 64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),)

    def forward(self, x):
        x = self.match(x)
        return x
      



class FlowEntropy(nn.Module):
    # Compute entro from matching cost
    def __init__(self):
        super(FlowEntropy, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x, 1)
        B,U,V,H,W = x.shape
        x = x.view(B,-1,H,W)
        x = F.softmax(x,dim=1).view(B,U,V,H,W)
        global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
        global_entropy /= np.log(x.shape[1]*x.shape[2])
        return global_entropy

class FlowRegression(nn.Module):
    # 2D soft argmin/argmax
    def __init__(self, maxU, maxV, flow_reg_by_max):
        super(FlowRegression, self).__init__()
        self.maxU = maxU
        self.maxV = maxV
        self.flow_reg_by_max = flow_reg_by_max

    def forward(self, x):
        assert(x.is_contiguous() == True)
        sizeU = 2*self.maxU+1
        sizeV = 2*self.maxV+1
        x = x.squeeze(1)
        B,_,_,H,W = x.shape

        # displacement along u 
        dispU = torch.reshape(torch.arange(-self.maxU, self.maxU+1, dtype=torch.float32),[1,sizeU,1,1,1]).to(x.device)
        dispU = dispU.expand(B, -1, sizeV, H,W).contiguous()
        dispU = dispU.view(B,sizeU*sizeV , H,W)

        # displacement along v
        dispV = torch.reshape(torch.arange(-self.maxV, self.maxV+1, dtype=torch.float32),[1,1,sizeV,1,1]).to(x.device)
        dispV = dispV.expand(B,sizeU, -1, H,W).contiguous()
        dispV = dispV.view(B,sizeU*sizeV,H,W)
            
        x = x.view(B,sizeU*sizeV,H,W)

        if self.flow_reg_by_max:
            x = F.softmax(x,dim=1)
        else:
            x = F.softmin(x,dim=1)

        flowU = (x*dispU).sum(dim=1)
        flowV = (x*dispV).sum(dim=1)
        flow  = torch.cat((flowU.unsqueeze(1),flowV.unsqueeze(1)),dim=1)
        return flow


class DAP(nn.Module):
    def __init__(self, md=3, dap_by_temperature=False):
        # Displacement-aware projection layer
        # implemented as a 1x1 2D conv
        super(DAP, self).__init__()
        self.dap_by_temperature = dap_by_temperature
        dimC = (2*md+1)**2
        self.dap_layer = BasicConv(dimC, dimC, kernel_size=1, padding=0, stride=1, bn=False, relu=False)
        if self.dap_by_temperature:
            self.dap_layer = BasicConv(dimC, 1, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

    def forward(self, x):
        x = x.squeeze(1)
        bs,du,dv,h,w = x.shape
        x = x.view(bs,du*dv,h,w)

        if self.dap_by_temperature:
            temp = self.dap_layer(x)+1e-6
            x = x*temp
        else:
            x = self.dap_layer(x)

        return x.view(bs,du,dv,h,w).unsqueeze(1)  



class DICLBase(BaseModel):
    def __init__(self, args):
        super(DICLBase,self).__init__(
            args=args,
            loss_fn=MultiScale_UP(
                loss_type=args.loss_type,
                weight=args.loss_weight,
                valid_range=args.loss_valid_range,
                removezero=args.loss_removezero,
                use_valid_range=args.loss_use_valid_range
            ),
            output_stride=128
        )
        self.feature = FeatureGA()

        if self.args.dap_layer:
            self.dap_layer6 = DAP(md=self.args.search_range[4], dap_by_temperature=self.args.dap_by_temperature)
            self.dap_layer5 = DAP(md=self.args.search_range[3], dap_by_temperature=self.args.dap_by_temperature)
            self.dap_layer4 = DAP(md=self.args.search_range[2], dap_by_temperature=self.args.dap_by_temperature)
            self.dap_layer3 = DAP(md=self.args.search_range[1], dap_by_temperature=self.args.dap_by_temperature)
            self.dap_layer2 = DAP(md=self.args.search_range[0], dap_by_temperature=self.args.dap_by_temperature)
        else:
            self.dap_layer6 = self.dap_layer5 = self.dap_layer4 = self.dap_layer3 = self.dap_layer2 = None

        self.entropy = FlowEntropy()

        # matching net, with the same arch at each pyramid level
        # level 6->2:  1/64,1/32,1/16,1/8,1/4
        self.matching6 = DICL_MODULE()
        self.matching5 = DICL_MODULE()
        self.matching4 = DICL_MODULE()
        self.matching3 = DICL_MODULE()
        self.matching2 = DICL_MODULE()

        # search range, e.g., [-3,3]
        # the search range for FlowRegression should be aligned with that when computing matching cost
        self.flow6 = FlowRegression(self.args.search_range[4],self.args.search_range[4], self.args.flow_reg_by_max)
        self.flow5 = FlowRegression(self.args.search_range[3],self.args.search_range[3], self.args.flow_reg_by_max)
        self.flow4 = FlowRegression(self.args.search_range[2],self.args.search_range[2], self.args.flow_reg_by_max)
        self.flow3 = FlowRegression(self.args.search_range[1],self.args.search_range[1], self.args.flow_reg_by_max)
        self.flow2 = FlowRegression(self.args.search_range[0],self.args.search_range[0], self.args.flow_reg_by_max)


        if args.ctf_context: 
            # If you would use context network as introduced in PWCNet
            self.context_net2 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64, 128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context_net3 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context_net4 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 64 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context_net5 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 64, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            # We remove the last several layers on pyramid level 6,5,4 considering their inputs' resolution
            # though this does not have an obvious effect
            self.context_net6 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64 , 64, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if self.args.dap_init_by_id:
            # init the dap layer kernel by identity matrix
            if self.dap_layer6 is not None:
                nn.init.eye_(self.dap_layer6.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer5 is not None:
                nn.init.eye_(self.dap_layer5.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer4 is not None:
                nn.init.eye_(self.dap_layer4.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer3 is not None:
                nn.init.eye_(self.dap_layer3.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer2 is not None:
                nn.init.eye_(self.dap_layer2.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
                
    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ctf_context', action='store_true')
        parser.add_argument('--ctf_context_only_flow2', action='store_true')
        parser.add_argument('--cuda_cost', action='store_true')
        parser.add_argument('--dap_by_temperature', action='store_true')
        parser.add_argument('--dap_layer', action='store_true')
        parser.add_argument('--dap_init_by_id', action='store_true')
        parser.add_argument('--flow_reg_by_max', action='store_true')
        parser.add_argument('--remove_warp_hole', action='store_true')
        parser.add_argument('--scale_context', type=float, nargs='+', default=(0.03125, 0.0625, 0.125, 0.25, 0.5))
        parser.add_argument('--search_range', type=int, nargs='+', default=(3, 3, 3, 3, 3))
        parser.add_argument('--sup_raw_flow', action='store_true')
        parser.add_argument('--loss_removezero', action='store_true')
        parser.add_argument('--loss_type', type=str, default='L1')
        parser.add_argument('--loss_use_valid_range', action='store_true')
        parser.add_argument('--loss_weight', type=float, nargs='+', default=(1.0,0.75,0.50,0.25,0.25))
        parser.add_argument(
            '--loss_valid_range', type=float, nargs='+', default=((372,372),(360,360),(336,336),(288,288),(192,192)))
        
        return parser


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask, mask

    def forward(self, inputs):
        # left and right images normalized
        x = inputs['images'][:,0] * 2 - 1
        y = inputs['images'][:,1] * 2 - 1

        
        # feature extraction
        _,x2,x3,x4,x5,x6 = self.feature(x)       
        _,y2,y3,y4,y5,y6 = self.feature(y)

        # compute flow for level 6
        cost6 = self.compute_cost(x6,y6,self.matching6,self.args.search_range[4],self.args.search_range[4])
        g6 = F.interpolate(x, [x.shape[-2]//64, x.shape[-1]//64], mode='bilinear',align_corners=True)
        if self.dap_layer6 is not None: cost6 = self.dap_layer6(cost6)
        flow6 = self.flow6(cost6)
        if self.args.ctf_context:
            if self.args.sup_raw_flow: raw_flow6 = flow6
            entro6 = self.entropy(cost6)
            # input to context network
            # including pred flow, entro, left img feat, left img
            feat6 = torch.cat((flow6.detach(),entro6.detach(),x6,g6),dim=1)
            flow6 = flow6 + self.context_net6(feat6)*self.args.scale_context[0]
        up_flow6 = 2.0*F.interpolate(flow6, [x5.shape[2],x5.shape[3]], mode='bilinear',align_corners=True)
        up_flow6 = up_flow6.detach()

         # compute flow for level 5
        warp5,_ = self.warp(y5,up_flow6)
        cost5 = self.compute_cost(x5,warp5,self.matching5,self.args.search_range[3],self.args.search_range[3])
        g5 = F.interpolate(x, [x.shape[-2]//32, x.shape[-1]//32], mode='bilinear',align_corners=True)
        if self.dap_layer5 is not None: cost5 = self.dap_layer5(cost5)
        flow5 = self.flow5(cost5) + up_flow6
        if self.args.ctf_context:
            if self.args.sup_raw_flow: raw_flow5 = flow5
            entro5 = self.entropy(cost5)
            feat5 = torch.cat((flow5.detach(),entro5.detach(),x5,g5),dim=1) 
            flow5 = flow5 + self.context_net5(feat5)*self.args.scale_context[1]
        up_flow5 = 2.0*F.interpolate(flow5, [x4.shape[2],x4.shape[3]], mode='bilinear',align_corners=True)
        up_flow5 = up_flow5.detach()

         # compute flow for level 4
        warp4,_ = self.warp(y4,up_flow5)
        cost4 = self.compute_cost(x4,warp4,self.matching4,self.args.search_range[2],self.args.search_range[2])
        g4 = F.interpolate(x, [x.shape[-2]//16, x.shape[-1]//16], mode='bilinear',align_corners=True)
        if self.dap_layer4 is not None: cost4 = self.dap_layer4(cost4)
        flow4 = self.flow4(cost4) + up_flow5
        if self.args.ctf_context:
            if self.args.sup_raw_flow: raw_flow4 = flow4
            entro4 = self.entropy(cost4)
            feat4 = torch.cat((flow4.detach(),entro4.detach(),x4,g4),dim=1) 
            flow4 = flow4 + self.context_net4(feat4)*self.args.scale_context[2]
        up_flow4 = 2.0*F.interpolate(flow4, [x3.shape[2],x3.shape[3]], mode='bilinear',align_corners=True)
        up_flow4 = up_flow4.detach()

         # compute flow for level 3
        warp3,_ = self.warp(y3,up_flow4)
        cost3 = self.compute_cost(x3,warp3,self.matching3,self.args.search_range[1],self.args.search_range[1])
        g3 = F.interpolate(x, [x.shape[-2]//8, x.shape[-1]//8], mode='bilinear',align_corners=True)
        if self.dap_layer3 is not None: cost3 = self.dap_layer3(cost3)
        flow3 = self.flow3(cost3) + up_flow4
        if self.args.ctf_context:
            if self.args.sup_raw_flow: raw_flow3 = flow3
            entro3 = self.entropy(cost3)
            feat3 = torch.cat((flow3.detach(),entro3.detach(),x3,g3),dim=1) 
            flow3 = flow3 + self.context_net3(feat3)*self.args.scale_context[3]
        up_flow3 = 2.0*F.interpolate(flow3, [x2.shape[2],x2.shape[3]], mode='bilinear',align_corners=True)
        up_flow3 = up_flow3.detach()

         # compute flow for level 2
        warp2,_ = self.warp(y2,up_flow3)
        cost2 = self.compute_cost(x2,warp2,self.matching2,self.args.search_range[0],self.args.search_range[0])
        g2 = F.interpolate(x, [x.shape[-2]//4, x.shape[-1]//4], mode='bilinear',align_corners=True)
        if self.dap_layer2 is not None: cost2 = self.dap_layer2(cost2)
        flow2 = self.flow2(cost2) + up_flow3

        if self.args.ctf_context or self.args.ctf_context_only_flow2:
            if self.args.sup_raw_flow: raw_flow2 = flow2
            entro2 = self.entropy(cost2)
            feat2 = torch.cat((flow2.detach(),entro2.detach(),x2,g2),dim=1) 
            flow2 = flow2 + self.context_net2(feat2)*self.args.scale_context[4]
        # sds

        flow_out = F.interpolate(flow2, size=x.shape[-2:], mode='bilinear', align_corners=True)[:, None]
        flow_out[:, :, 0] = flow_out[:, :, 0] * (float(x.shape[-1]) / flow2.shape[-1])
        flow_out[:, :, 1] = flow_out[:, :, 1] * (float(x.shape[-2]) / flow2.shape[-2])

        output = {
            'flows': flow_out
        }
        if self.training:
            if self.args.sup_raw_flow:
                output['flow_preds'] = [flow2, raw_flow2, flow3, raw_flow3, flow4, raw_flow4, flow5, raw_flow5, flow6,raw_flow6]
            else:
                output['flow_preds'] = [flow2, flow3, flow4, flow5, flow6]
        return output
        
    def compute_cost(self, x,y,matchnet,maxU,maxV):
        sizeU = 2*maxU+1
        sizeV = 2*maxV+1
        b,c,height,width = x.shape

        with torch.cuda.device_of(x):
            # init cost as tensor matrix
            cost = x.new().resize_(x.size()[0], 2*c, 2*maxU+1,2*maxV+1, height,  width).zero_()


        if self.args.cuda_cost:
            # CUDA acceleration
            corr = SpatialCorrelationSampler(kernel_size=1,patch_size=(int(1+2*3),int(1+2*3)),stride=1,padding=0,dilation_patch=1)
            cost = corr(x, y)
        else:
            for i in range(2*maxU+1):
                ind = i-maxU
                for j in range(2*maxV+1):
                    indd = j-maxV
                    # for each displacement hypothesis, we construct a feature map as the input of matching net
                    # here we hold them together for parallel processing later
                    cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                    cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]

        if self.args.remove_warp_hole:
            # mitigate the effect of holes (may be raised by occ)
            valid_mask = cost[:,c:,...].sum(dim=1)!=0
            valid_mask = valid_mask.detach()
            cost = cost*valid_mask.unsqueeze(1).float()

        # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
        cost = cost.permute([0,2,3,1,4,5]).contiguous() 
        # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
        cost = cost.view(x.size()[0]*sizeU*sizeV,c*2, x.size()[2], x.size()[3])
        # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
        cost = matchnet(cost)
        cost = cost.view(x.size()[0],sizeU,sizeV,1, x.size()[2],x.size()[3])
        cost = cost.permute([0,3,1,2,4,5]).contiguous() 

        # (B, U, V, H, W)
        return cost



######################## Part of the Model #############################################


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if self.use_bn: self.bn = nn.BatchNorm2d(out_channels)
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.deconv = deconv
        if deconv:
            kernel = 4
        else:
            kernel = 3

        self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert(x.size() == rem.size())

        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class FeatureGA(nn.Module):
    def __init__(self):
        super(FeatureGA, self).__init__()
        # feature backbone
        # adopted from GANet (https://github.com/feihuzhang/GANet)
        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5a = BasicConv(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv6a = BasicConv(160, 192, kernel_size=3, stride=2, padding=1)

        self.deconv6a = Conv2x(192, 160, deconv=True)
        self.deconv5a = Conv2x(160, 128, deconv=True)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)
        self.conv5b = Conv2x(128, 160)
        self.conv6b = Conv2x(160, 192)

        self.deconv6b = Conv2x(192,160, deconv=True)
        self.outconv_6 = BasicConv(160, 32, kernel_size=3,  padding=1)

        self.deconv5b = Conv2x(160,128, deconv=True)
        self.outconv_5 = BasicConv(128, 32, kernel_size=3,  padding=1)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.outconv_4 = BasicConv(96, 32, kernel_size=3,  padding=1)

        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.outconv_3 = BasicConv(64, 32, kernel_size=3,  padding=1)

        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.outconv_2 = BasicConv(48, 32, kernel_size=3,  padding=1)


    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x

        x = self.deconv6a(x,rem5)
        rem5 = x
        x = self.deconv5a(x,rem4)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)

        x = self.deconv6b(x, rem5)
        x6 = self.outconv_6(x)


        x = self.deconv5b(x, rem4)
        x5 = self.outconv_5(x)

        x = self.deconv4b(x, rem3)
        x4 = self.outconv_4(x)

        x = self.deconv3b(x, rem2)
        x3 = self.outconv_3(x)

        x = self.deconv2b(x, rem1)
        x2 = self.outconv_2(x)


        return None, x2, x3, x4,x5,x6


class DICL(DICLBase):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dicl-chairs-fdc24e2f.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dicl-kitti-4813ccab.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dicl-sintel-fa9fc259.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dicl-things-20bc573d.ckpt'
    }

    def __init__(self, args):
        args.ctf_context = True
        args.dap_layer = True
        args.dap_init_by_id = True
        args.flow_reg_by_max = True
        args.remove_warp_hole = True
        super(DICL, self).__init__(args)
