import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MultiScale_UP(nn.Module):
    def __init__(self,loss_type='L1',weight=[1.,0.5,0.25],valid_range=None,removezero=False,use_valid_range=False):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.valid_range = valid_range
        self.removezero = removezero
        self.use_valid_range = use_valid_range

    def forward(self, preds, inputs):
        # Multi scale Loss, where flow upsampling is automatically computed here
        loss = 0
        loss_list = []

        target = inputs['flows'][:, 0]
        output = preds['flow_preds']
        extra_mask = inputs.get('extra_mask')
        b, _, h, w = target.size()
        
        for i, cur_output in enumerate(output):
            # Compute loss for each level
            realflow = F.interpolate(cur_output, (h,w), mode='bilinear', align_corners=True)
            realflow[:,0,:,:] = realflow[:,0,:,:]*(w/cur_output.shape[3])
            realflow[:,1,:,:] = realflow[:,1,:,:]*(h/cur_output.shape[2])

            with torch.no_grad():
                if i==0: epe = realEPE(realflow,target,extra_mask=extra_mask)

            if self.loss_type=='L2':
                lossvalue = torch.norm(realflow-target,p=2,dim=1)
            elif self.loss_type=='robust':
                lossvalue = ((realflow-target).abs().sum(dim=1)+1e-8)
                lossvalue = lossvalue**0.4
            elif self.loss_type=='L1':
                lossvalue = (realflow-target).abs().sum(dim=1)
            else:
                raise NotImplementedError

            if self.use_valid_range and self.valid_range is not None:
                # Filter out the pixels whose gt flow is out of our search range
                with torch.no_grad():
                    mask = (target[:,0,:,:].abs()<=self.valid_range[i][1]) & (target[:,1,:,:].abs()<=self.valid_range[i][0])
            else:
                with torch.no_grad():
                    mask = torch.ones(target[:,0,:,:].shape).type_as(target)

            lossvalue = lossvalue*mask.float() 
            
            if extra_mask is not None:
                val = extra_mask > 0
                lossvalue = lossvalue[val]
                cur_loss = lossvalue.mean()*self.weight[i]
                assert lossvalue.shape[0] == extra_mask.sum()
            else:
                cur_loss = lossvalue.mean()*self.weight[i]

            loss+=cur_loss
            loss_list.append(cur_loss)

        loss = loss/len(output)
        return {
            'loss': loss,
            'loss_list': loss_list,
            'epe': epe
        }


#################################################

def random_select_points(x,y,x_,y_,samples=10):
    idx=torch.randperm(x.shape[0])
    x=x[idx[:samples],:]
    y=y[idx[:samples],:]
    x_=x_[idx[:samples],:]
    y_=y_[idx[:samples],:]
    return x,y,x_,y_

def subspace_loss_batch(flow):

    # https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Unsupervised_Deep_Epipolar_Flow_for_Stationary_or_Dynamic_Scenes_CVPR_2019_paper.pdf
    B, _, H, W = flow.size()
    xx = Variable(torch.arange(0, W).view(1,-1).repeat(H,1).cuda())
    yy = Variable(torch.arange(0, H).view(-1,1).repeat(1,W).cuda())
    grid_x = xx.view(1,1,H,W).repeat(B,1,1,1).float()
    grid_y = yy.view(1,1,H,W).repeat(B,1,1,1).float()
    
    flow_u = flow[:,0,:,:].unsqueeze(1)
    flow_v = flow[:,1,:,:].unsqueeze(1)
    
    pos_x = grid_x + flow_u
    pos_y = grid_y + flow_v

    inside_x = (pos_x <= (W-1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (H-1)) & (pos_y >= 0.0)
    
    inside = inside_x & inside_y 
    
    loss = 0

    least_num = 2000 

    list_X =[]
    list_X_ = []
    for i in range(B):
        grid_x_i = grid_x[i,:,:,:]
        grid_y_i = grid_y[i,:,:,:]
        pos_x_i = pos_x[i,:,:,:]
        pos_y_i = pos_y[i,:,:,:]
        inside_i= inside[i,:,:,:]
        
        if inside_i.sum()>least_num:
            x  = torch.masked_select(grid_x_i, inside_i).view(-1,1)
            y  = torch.masked_select(grid_y_i, inside_i).view(-1,1)
            x_ = torch.masked_select(pos_x_i, inside_i).view(-1,1)
            y_ = torch.masked_select(pos_y_i, inside_i).view(-1,1)
            x, y, x_, y_ = random_select_points(x,y,x_,y_,samples=least_num)
            o  = torch.ones_like(x)
            x, y, x_, y_ = x/W, y/W, x_/W, y_/W
            X  = torch.cat((x,x,x,y,y,y,o,o,o),1).permute(1,0)  
            X_ = torch.cat((x_,y_,o,x_,y_,o,x_,y_,o),1).permute(1,0)
            list_X.append(X.unsqueeze(0))
            list_X_.append(X_.unsqueeze(0))

    all_X = torch.cat(list_X)
    all_X_ = torch.cat(list_X_)
    M = all_X*all_X_
    lambda1 = 10
    MTM = lambda1 * torch.matmul(M.permute(0,2,1),M)
    I = torch.eye(MTM.size()[1]).type_as(MTM).unsqueeze(0).repeat(B,1,1)
    MTM_inverse = torch.inverse((I + MTM))
    C = torch.matmul(MTM_inverse,MTM)
    C2 = C**2
    loss1 = torch.sum(C2.view(-1,1),dim=0)

    loss2 = lambda1 * torch.sum(((torch.matmul(M,C)-M)**2).view(-1,1),dim=0)

    loss +=  (loss1 + loss2)

    return loss/B


def EPE_flow(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE_flow(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE_flow(output, target)
        return [lossvalue, epevalue]




########################################################


def warp(x, flo):
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
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
        
    return output*mask
      
def warp_(self, tensorInput, tensorFlow):
    if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
        self.tensorPartial = tensorFlow.new_ones(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))
        # end

    if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        # end

    tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

    tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask


def EPE(input_flow, target_flow, sparse=False, mean=True,extra_mask=None):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]

    if extra_mask is not None:
        EPE_map = EPE_map[extra_mask.byte()]

    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def realEPE(output, target, sparse=False, valid_range=None,extra_mask=None,use_valid_range=False):
    b, _, h, w = target.size()
    upsampled_output = output
    if use_valid_range and valid_range is not None:
        mask = (target[:,0,:,:].abs()<=valid_range[1]) & (target[:,1,:,:].abs()<=valid_range[0])
        mask = mask.unsqueeze(1).expand(-1,2,-1,-1).float()
        upsampled_output = upsampled_output*mask
        target = target*mask
    return EPE(upsampled_output, target, sparse, mean=True,extra_mask=extra_mask)