'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math

def EPE(input_flow, target_flow):
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
        output = output['flow_preds']
        target = target['flows'][:, 0]
        lossvalue = self.loss(output, target)
        return lossvalue

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        output = output['flow_preds']
        target = target['flows'][:, 0]
        lossvalue = self.loss(output, target)
        return lossvalue

class MultiScale(nn.Module):
    def __init__(self, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE']

    def forward(self, output, target):
        output = output['flow_preds']
        target = target['flows'][:, 0]
        lossvalue = 0

        if type(output) is tuple or type(output) is list:
            target = self.div_flow * target
            for i, flow_pred_ in enumerate(output):
                target_ = self.multiScales[i](target)
                lossvalue += self.loss_weights[i]*self.loss(flow_pred_, target_)
        else:
            lossvalue += self.loss(output, target)
        return lossvalue
