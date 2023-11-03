import torch
import torch.nn as nn
import timm
import numpy as np
#from .twins_ft import _twins_svt_large_jihao

class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True, del_layers=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)

        if del_layers:
            del self.svt.head
            del self.svt.patch_embeds[2]
            del self.svt.patch_embeds[2]
            del self.svt.blocks[2]
            del self.svt.blocks[2]
            del self.svt.pos_block[2]
            del self.svt.pos_block[2]
               
    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == 0:
                x_16 = x.clone()
            if i == layer-1:
                break
        
        return x
    
    def extract_ml_features(self, x, data=None, layer=2):
        res = []
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            x, size = embed(x)
            if i == layer-1:
                x1 = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer-1:
                break
        
        return x1, x
    
    def compute_params(self):
        num = 0

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            
            for param in embed.parameters():
                num += np.prod(param.size())
            for param in blocks.parameters():
                num += np.prod(param.size())
            for param in pos_blk.parameters():
                num += np.prod(param.size())
            for param in drop.parameters():
                num += np.prod(param.size())
            if i == 1:
                break
        return num

class convnext_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.convnext = timm.create_model('convnext_large', pretrained=pretrained)
    
    def forward(self, x, data=None, layer=2):

        x = self.convnext.stem(x)
        x = self.convnext.stages[0](x)
        x = self.convnext.stages[1](x)
        return x
    
    def compute_params(self):
        num = 0

        for param in self.convnext.stem.parameters():
            num += np.prod(param.size())
        for param in self.convnext.stages[0].parameters():
            num += np.prod(param.size())
        for param in self.convnext.stages[1].parameters():
            num += np.prod(param.size())

        return num

class convnext_Xlarge_4x(nn.Module):
    def __init__(self, pretrained=True, del_layers=True):
        super().__init__()
        self.convnext = timm.create_model('convnext_xlarge_in22k', pretrained=pretrained)

        # self.convnext.stem[0].stride = (2, 2)
        # self.convnext.stem[0].padding = (1, 1)

        if del_layers:
            del self.convnext.head
            del self.convnext.stages[1]
            del self.convnext.stages[1]
            del self.convnext.stages[1]
        
        # print(self.convnext)
            
    
    def forward(self, x, data=None, layer=2):

        x = self.convnext.stem(x)
        x = self.convnext.stages[0](x)
        return x

class convnext_base_2x(nn.Module):
    def __init__(self, pretrained=True, del_layers=True):
        super().__init__()
        self.convnext = timm.create_model('convnext_base_in22k', pretrained=pretrained)

        self.convnext.stem[0].stride = (2, 2)
        self.convnext.stem[0].padding = (1, 1)

        if del_layers:
            del self.convnext.head
            del self.convnext.stages[1]
            del self.convnext.stages[1]
            del self.convnext.stages[1]
        
        # print(self.convnext)
            
    
    def forward(self, x, data=None, layer=2):

        x = self.convnext.stem(x)
        x = self.convnext.stages[0](x)
        return x


if __name__ == "__main__":
    m = convnext_Xlarge_2x()
    input = torch.randn(2, 3, 64, 64)
    out = m(input)
    print(out.shape)
