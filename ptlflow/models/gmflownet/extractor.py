import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class IPTHeadEncoder(nn.Module):
    """docstring for IPTHead"""
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(IPTHeadEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        half_out_dim = max(output_dim // 2, 64)
        self.layer1 = ResidualBlock(64, half_out_dim, self.norm_fn, stride=2)
        self.layer2 = ResidualBlock(half_out_dim, output_dim, self.norm_fn, stride=2)

        # # output convolution; this can solve mixed memory warning, not know why
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class BasicConvEncoder(nn.Module):
    """docstring for BasicConvEncoder"""
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicConvEncoder, self).__init__()
        self.norm_fn = norm_fn

        half_out_dim = max(output_dim // 2, 64)

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(half_out_dim)
            self.norm3 = nn.BatchNorm2d(output_dim)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm2 = nn.InstanceNorm2d(half_out_dim)
            self.norm3 = nn.InstanceNorm2d(output_dim)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, half_out_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(half_out_dim, output_dim, kernel_size=3, stride=2, padding=1)

        # # output convolution; this can solve mixed memory warning, not know why
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x = F.relu(self.norm3(self.conv3(x)), inplace=True)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x




class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = self.d_model // self.n_head

        self.w_qs = nn.Linear(self.d_model, self.d_model, bias=False)  # TODO: enable bias
        self.w_ks = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_vs = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc = nn.Linear(self.d_model, self.d_model)

        # self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5) # TODO

        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):
        ''' 
           q: shape of N*len*C
        '''
        d_head, n_head = self.d_head, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_head)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_head)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_head)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.d_head)
        attn = self.dropout(F.softmax(attn, dim=-1))
        q_updated = torch.matmul(attn, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_updated = q_updated.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_updated = self.dropout(self.fc(q_updated))
        # q_updated += residual

        # q_updated = self.layer_norm(q_updated)

        return q_updated, attn



class AnchorEncoderBlock(nn.Module):
  
  def __init__(self, anchor_dist, d_model, num_heads, d_ff, dropout=0.):
    super().__init__()

    self.anchor_dist = anchor_dist
    self.half_anchor_dist = anchor_dist // 2

    self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm_1 = nn.LayerNorm(d_model)

    self.FFN = nn.Sequential(
          nn.Linear(d_model, d_ff),
          nn.ReLU(),
          nn.Linear(d_ff, d_model),
        )

    self.layer_norm_2 = nn.LayerNorm(d_model)
  
  
  def forward(self, inputs):
    ''' 
        inputs: batches with N*C*H*W
    '''
    N, C, H, W = inputs.shape

    x = inputs
    anchors = inputs[:,:, self.half_anchor_dist::self.anchor_dist, 
                            self.half_anchor_dist::self.anchor_dist].clone()

    # flatten feature maps
    x = x.reshape(N, C, H*W).transpose(-1,-2)
    anchors = anchors.reshape(N, C, anchors.shape[2]* anchors.shape[3]).transpose(-1,-2)

    # two-stage multi-head self-attention
    anchors_new = self.dropout(self.selfAttn(anchors, x, x)[0])
    residual = self.dropout(self.selfAttn(x, anchors_new, anchors_new)[0])

    norm_1 = self.layer_norm_1(x + residual)
    x_linear = self.dropout(self.FFN(norm_1))
    x_new = self.layer_norm_2(norm_1 + x_linear)

    outputs = x_new.transpose(-1,-2).reshape(N, C, H, W)
    return outputs



class EncoderBlock(nn.Module):
  
  def __init__(self, d_model, num_heads, d_ff, dropout=0.):
    super().__init__()

    self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm_1 = nn.LayerNorm(d_model)

    self.FFN = nn.Sequential(
          nn.Linear(d_model, d_ff),
          nn.ReLU(),
          nn.Linear(d_ff, d_model),
        )

    self.layer_norm_2 = nn.LayerNorm(d_model)
  
  
  def forward(self, x):
    ''' 
        x: input batches with N*C*H*W
    '''
    N, C, H, W = x.shape

    # update x
    x = x.reshape(N, C, H*W).transpose(-1,-2)

    residual = self.dropout(self.selfAttn(x, x, x)[0])
    norm_1 = self.layer_norm_1(x + residual)
    x_linear = self.dropout(self.FFN(norm_1))
    x_new = self.layer_norm_2(norm_1 + x_linear)

    outputs = x_new.transpose(-1,-2).reshape(N, C, H, W)
    return outputs



class ReduceEncoderBlock(nn.Module):
  
  def __init__(self, d_model, num_heads, d_ff, dropout=0.):
    super().__init__()

    self.reduce = nn.Sequential(
          nn.Conv2d(d_model, d_model, 2, 2),
          nn.Conv2d(d_model, d_model, 2, 2)
        )
    # self.reduce = nn.Sequential(
    #       nn.AvgPool2d(16, 16)
    #     )

    self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm_1 = nn.LayerNorm(d_model)

    self.FFN = nn.Sequential(
          nn.Linear(d_model, d_ff),
          nn.ReLU(),
          nn.Linear(d_ff, d_model)
        )

    self.layer_norm_2 = nn.LayerNorm(d_model)
  
  
  def forward(self, x):
    ''' 
        x: input batches with N*C*H*W
    '''
    N, C, H, W = x.shape
    x_reduced = self.reduce(x)

    # update x
    x = x.reshape(N, C, H*W).transpose(-1,-2)
    x_reduced = x_reduced.reshape(N, C, -1).transpose(-1,-2)

    # print('x ', x.shape)
    # print('x_reduced ', x_reduced.shape)
    # exit()

    residual = self.dropout(self.selfAttn(x, x_reduced, x_reduced)[0])

    norm_1 = self.layer_norm_1(x + residual)
    x_linear = self.dropout(self.FFN(norm_1))
    x_new = self.layer_norm_2(norm_1 + x_linear)

    outputs = x_new.transpose(-1,-2).reshape(N, C, H, W)
    return outputs



def window_partition(x, window_size):
  """
  Args:
      x: (B, H, W, C)
      window_size (int): window size

  Returns:
      windows: (num_windows*B, window_size, window_size, C)
  """
  B, H, W, C = x.shape
  x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
  windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
  return windows


def window_reverse(windows, window_size, H, W):
  """
  Args:
      windows: (num_windows*B, window_size, window_size, C)
      window_size (int): Window size
      H (int): Height of image
      W (int): Width of image

  Returns:
      x: (B, H, W, C)
  """
  B = int(windows.shape[0] / (H * W / window_size / window_size))
  x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
  x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
  return x


class LayerEncoderBlock(nn.Module):
  
    def __init__(self, win_size, d_model, num_heads, d_ff, dropout=0.):
        super().__init__()

        self.win_size = win_size
        self.down_factor = 4
        self.unfold_stride = int(self.win_size//self.down_factor)

        self.stride_list = [math.floor(win_size/self.down_factor**idx) for idx in range(8) if win_size/self.down_factor**idx >= 1]
        # [16, 4, 1]

        self.reduce = nn.Sequential(
              nn.AvgPool2d(self.down_factor, self.down_factor)
            )

        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.crossAttn = MultiHeadAttention(d_model, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)
        self.layerNormSelf = nn.LayerNorm(d_model)
        self.layerNormCross = nn.LayerNorm(d_model)

        self.FFN = nn.Sequential(
              nn.Linear(d_model, d_ff),
              nn.ReLU(),
              nn.Linear(d_ff, d_model)
            )

        self.layer_norm_out = nn.LayerNorm(d_model)


    def Circular_pad2D(self, x, pad_right, pad_bottom):
        '''
            x: (N, H, W, C)
            x_pad: (N, H_pad, W_pad, C)
        '''
        N, H, W, C = x.shape

        H_pad = H + pad_bottom
        W_pad = W + pad_right

        H_repeat = math.ceil(H_pad/H)
        W_repeat = math.ceil(W_pad/W)
        x_repeat = x.repeat(1, H_repeat, W_repeat, 1)

        x_pad = x_repeat[:, :H_pad, :W_pad, :]
        return x_pad


    def pad_fit_win(self, x, win_size):
        N, H, W, C = x.shape

        W_ = math.ceil(W/win_size)*win_size
        H_ = math.ceil(H/win_size)*win_size
        padRight = W_ - W
        padBottom = H_ - H

        x_pad = self.Circular_pad2D(x, padRight, padBottom) # N*H_*W_*C
        return x_pad


    def self_attention(self, x):
        '''
            x: (N, H, W, C)
            out: (N, H, W, C)
        '''
        N, H, W, C = x.shape
        x_pad = self.pad_fit_win(x, self.win_size) # N*H_*W_*C
        _, H_, W_, _ = x_pad.shape

        # x_pad = F.pad(x.permute(xxx), (0, padRight, 0, padBottom), mode='reflect') # N*C*H_*W_

        x_window = window_partition(x_pad, self.win_size) # (num_win*B, win_size, win_size, C)
        x_window = x_window.view(-1, self.win_size*self.win_size, C) # (num_win*B, win_size*win_size, C)

        # self-attention
        residual = self.dropout(self.selfAttn(x_window, x_window, x_window)[0])
        residual = residual.view(-1, self.win_size, self.win_size, C)
        residual = window_reverse(residual, self.win_size, H_, W_) # (N, H_, W_, C)

        out = x_pad + residual
        out = out[:, :H, :W, :]
        return out


    def cross_attention(self, query, keyVal):
        '''
            query: (N, qH, qW, C)
            keyVal: (N, kH, kW, C)
            out: (N, qH, qW, C)
        '''
        _, qH, qW, C = query.shape
        _, kH, kW, C = keyVal.shape

        # print('in query ', query.shape)
        # print('in keyVal ', keyVal.shape)
        # print('-')

        query = self.pad_fit_win(query, self.win_size) # N*H_*W_*C
        _, qH_, qW_, C = query.shape

        query_win = window_partition(query, self.win_size)
        query_win = query_win.view(-1, self.win_size*self.win_size, C) # (num_win*B, win_size*win_size, C)

        # pad and unfold keyVal
        kW_ = (math.ceil(kW/self.unfold_stride) - 1)*self.unfold_stride + self.win_size
        kH_ = (math.ceil(kH/self.unfold_stride) - 1)*self.unfold_stride + self.win_size
        padRight = kW_ - kW
        padBottom = kH_ - kH

        keyVal_pad = self.Circular_pad2D(keyVal, padRight, padBottom)
        keyVal = F.unfold(keyVal_pad.permute(0, 3, 1, 2), self.win_size, stride=self.unfold_stride) # (N, C*win_size*win_size, num_win)
        keyVal = keyVal.permute(0,2,1).reshape(-1, C, self.win_size*self.win_size).permute(0,2,1) # (num_win*B, win_size*win_size, C)

        # print('win query ', query_win.shape)
        # print('win keyVal ', keyVal.shape)
        # print('-')

        residual = self.dropout(self.crossAttn(query_win, keyVal, keyVal)[0])
        residual = residual.view(-1, self.win_size, self.win_size, C)
        residual = window_reverse(residual, self.win_size, qH_, qW_) # (N, H, W, C)

        out = query + residual
        out = out[:, :qH, :qW, :]
        return out

  
    def forward(self, x):
        ''' 
            x: input batches with N*C*H*W
        '''
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # N*H*W*C
        x = self.pad_fit_win(x, self.win_size) # pad

        # layered self-attention
        layerAttnList = []
        strideListLen = len(self.stride_list)
        for idx in range(strideListLen):
            x_attn = self.self_attention(x) # built-in shortcut
            x_attn = self.layerNormSelf(x_attn)
            layerAttnList.append(x_attn)

            if idx < strideListLen - 1:
                x = self.reduce(x_attn.permute(0, 3, 1 ,2)) # N*C*H*W
                x = x.permute(0, 2, 3, 1) # N*H*W*C

        # layered cross-attention
        KeyVal = layerAttnList[-1]
        for idx in range(strideListLen-1, 0, -1):
            Query = layerAttnList[idx-1]
            Query = self.cross_attention(Query, KeyVal) # built-in shortcut
            Query = self.layerNormCross(Query)

            KeyVal = Query

        Query = Query[:, :H, :W, :]  # unpad

        q_residual = self.dropout(self.FFN(Query))
        x_new = self.layer_norm_out(Query + q_residual)

        outputs = x_new.permute(0, 3, 1, 2)
        return outputs



class BasicLayerEncoderBlock(nn.Module):
  
    def __init__(self, win_size, d_model, num_heads, d_ff, dropout=0.):
        super().__init__()

        self.win_size = win_size
        self.down_factor = 2
        self.unfold_stride = int(self.win_size//self.down_factor)

        self.stride_list = [math.floor(win_size/self.down_factor**idx) for idx in range(8) if win_size/self.down_factor**idx >= 1]
        # [16, 8, 4, 2, 1]

        self.reduce = nn.Sequential(
              nn.AvgPool2d(self.down_factor, self.down_factor)
            )

        self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
        self.crossAttn = MultiHeadAttention(d_model, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)
        self.layerNormSelf = nn.LayerNorm(d_model)
        self.layerNormCross = nn.LayerNorm(d_model)

        self.FFN = nn.Sequential(
              nn.Linear(d_model, d_ff),
              nn.ReLU(),
              nn.Linear(d_ff, d_model)
            )

        self.layer_norm_out = nn.LayerNorm(d_model)


    def Circular_pad2D(self, x, pad_right, pad_bottom):
        '''
            x: (N, H, W, C)
            x_pad: (N, H_pad, W_pad, C)
        '''
        N, H, W, C = x.shape

        H_pad = H + pad_bottom
        W_pad = W + pad_right

        H_repeat = math.ceil(H_pad/H)
        W_repeat = math.ceil(W_pad/W)
        x_repeat = x.repeat(1, H_repeat, W_repeat, 1)

        x_pad = x_repeat[:, :H_pad, :W_pad, :]
        return x_pad


    def pad_fit_win(self, x, win_size):
        N, H, W, C = x.shape

        W_ = math.ceil(W/win_size)*win_size
        H_ = math.ceil(H/win_size)*win_size
        padRight = W_ - W
        padBottom = H_ - H

        x_pad = self.Circular_pad2D(x, padRight, padBottom) # N*H_*W_*C
        return x_pad


    def self_attention(self, x):
        '''
            x: (N, H, W, C)
            out: (N, H, W, C)
        '''
        N, H, W, C = x.shape
        x_pad = self.pad_fit_win(x, self.win_size) # N*H_*W_*C
        _, H_, W_, _ = x_pad.shape

        # x_pad = F.pad(x.permute(xxx), (0, padRight, 0, padBottom), mode='reflect') # N*C*H_*W_

        x_window = window_partition(x_pad, self.win_size) # (num_win*B, win_size, win_size, C)
        x_window = x_window.view(-1, self.win_size*self.win_size, C) # (num_win*B, win_size*win_size, C)

        # self-attention
        residual = self.dropout(self.selfAttn(x_window, x_window, x_window)[0])
        residual = residual.view(-1, self.win_size, self.win_size, C)
        residual = window_reverse(residual, self.win_size, H_, W_) # (N, H_, W_, C)

        out = x_pad + residual
        out = out[:, :H, :W, :]
        return out


    def cross_attention(self, query, keyVal, query_win_size):
        '''
            query: (N, qH, qW, C)
            keyVal: (N, kH, kW, C)
            out: (N, qH, qW, C)
        '''
        _, qH, qW, C = query.shape

        query_win = window_partition(query, query_win_size)
        query_win = query_win.view(-1, query_win_size*query_win_size, C) # (num_win*B, win_size*win_size, C)

        keyWinSize = query_win_size // 2
        keyVal_win = window_partition(keyVal, keyWinSize)
        keyVal_win = keyVal_win.view(-1, keyWinSize*keyWinSize, C) # (num_win*B, win_size*win_size, C)

        residual = self.dropout(self.crossAttn(query_win, keyVal_win, keyVal_win)[0])
        residual = residual.view(-1, query_win_size, query_win_size, C)
        residual = window_reverse(residual, query_win_size, qH, qW) # (N, H, W, C)

        out = query + residual
        return out

  
    def forward(self, x):
        ''' 
            x: input batches with N*C*H*W
        '''
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # N*H*W*C
        x = self.pad_fit_win(x, self.win_size) # pad

        # layered self-attention
        layerAttnList = []
        strideListLen = len(self.stride_list)
        for idx in range(strideListLen):
            x_attn = self.self_attention(x) # built-in shortcut
            x_attn = self.layerNormSelf(x_attn)
            layerAttnList.append(x_attn)

            if idx < strideListLen - 1:
                x = self.reduce(x_attn.permute(0, 3, 1 ,2)) # N*C*H*W
                x = x.permute(0, 2, 3, 1) # N*H*W*C

        # layered cross-attention
        KeyVal = layerAttnList[-1]
        for idx in range(strideListLen-1, 0, -1):
            Query = layerAttnList[idx-1]
            QueryWinSize = self.stride_list[idx-1]

            Query = self.cross_attention(Query, KeyVal, QueryWinSize) # built-in shortcut
            Query = self.layerNormCross(Query)

            KeyVal = Query

        Query = Query[:, :H, :W, :]  # unpad

        q_residual = self.dropout(self.FFN(Query))
        x_new = self.layer_norm_out(Query + q_residual)

        outputs = x_new.permute(0, 3, 1, 2)
        return outputs




class PositionalEncoding(nn.Module):
  
    def __init__(self, d_model, dropout=0.):
        super().__init__()

        self.max_len = 256
        self.d_model = d_model

        self._update_PE_table(self.max_len, self.d_model//2)


    def _update_PE_table(self, max_len, d_model):
        self.PE_table = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2).float()/d_model)

        self.PE_table[:, 0::2] = torch.sin(pos/denominator)
        self.PE_table[:, 1::2] = torch.cos(pos/denominator)


    def forward(self, x):
        ''' x: image batches with N*C*H*W '''

        N, C, H, W = x.shape
        max_hw = max(H, W)

        if max_hw > self.max_len or self.d_model != C:
            self.max_len = max_hw
            self.d_model = C

            self._update_PE_table(self.max_len, self.d_model//2)

        if self.PE_table.device != x.device:
          self.PE_table = self.PE_table.to(x.device)

        h_pos_emb = self.PE_table[:H, :].unsqueeze(1).repeat(1, W, 1) # H*W*C/2
        w_pos_emb = self.PE_table[:W, :].unsqueeze(0).repeat(H, 1, 1) # H*W*C/2
        pos_emb = torch.cat([h_pos_emb, w_pos_emb], dim=-1
                    ).permute([2,0,1]).unsqueeze(0).repeat(N,1,1,1) # N*C*H*W

        output = x + pos_emb
        return output



class TransformerEncoder(nn.Module):
  
  def __init__(self, anchor_dist, num_blocks, d_model, num_heads, d_ff, dropout=0.):
    super().__init__()

    self.anchor_dist = anchor_dist

    blocks_list = []
    for idx in range(num_blocks):
      # blocks_list.append( AnchorEncoderBlock(anchor_dist, d_model, num_heads, d_ff, dropout) )
      # blocks_list.append( EncoderBlock(d_model, num_heads, d_ff, dropout) )
      blocks_list.append( ReduceEncoderBlock(d_model, num_heads, d_ff, dropout) )
    #   blocks_list.append( BasicLayerEncoderBlock(anchor_dist, d_model, num_heads, d_ff, dropout) )

    self.blocks = nn.Sequential(*blocks_list)

    self.posEmbedding = PositionalEncoding(d_model, dropout)
  

  def forward(self, x):
    x_w_pos = self.posEmbedding(x)
    x_updated = self.blocks(x_w_pos)

    return x_updated



class RawInputTransEncoder(nn.Module):
  
  def __init__(self, anchor_dist, num_blocks, d_model, num_heads, d_ff, dropout=0.):
    super().__init__()

    self.anchor_dist = anchor_dist

    self.linear = nn.Conv2d(3, d_model, 8, 8)

    blocks_list = []
    for idx in range(num_blocks):
      # blocks_list.append( AnchorEncoderBlock(anchor_dist, d_model, num_heads, d_ff, dropout) )
      # blocks_list.append( EncoderBlock(d_model, num_heads, d_ff, dropout) )
      # blocks_list.append( ReduceEncoderBlock(d_model, num_heads, d_ff, dropout) )
      blocks_list.append( LayerEncoderBlock(anchor_dist, d_model, num_heads, d_ff, dropout) )

    self.blocks = nn.Sequential(*blocks_list)

    self.posEmbedding = PositionalEncoding(d_model, dropout)
  
    # initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if m.weight is not None:
          nn.init.constant_(m.weight, 1)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)


  def forward(self, x):
    # if input is list, combine batch dimension
    is_list = isinstance(x, tuple) or isinstance(x, list)
    if is_list:
        batch_dim = x[0].shape[0]
        x = torch.cat(x, dim=0)

    x = self.linear(x)
    x_w_pos = self.posEmbedding(x)
    x_updated = self.blocks(x_w_pos)

    if is_list:
        x_updated = torch.split(x_updated, [batch_dim, batch_dim], dim=0)

    return x_updated



class GlobalLocalBlock(nn.Module):
  
  def __init__(self, anchor_dist, d_model, num_heads, out_dim, dropout=0., stride=1):
    super().__init__()

    self.anchor_dist = anchor_dist
    self.half_anchor_dist = anchor_dist // 2
    self.d_model = d_model
    self.out_dim = out_dim

    self.selfAttn = MultiHeadAttention(d_model, num_heads, dropout)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm_1 = nn.LayerNorm(d_model)

    self.resBlock_1 = ResidualBlock(d_model, d_model, norm_fn='instance', stride=stride)
    self.change_channel = nn.Linear(d_model, out_dim)
    self.resBlock_2 = ResidualBlock(out_dim, out_dim, norm_fn='instance', stride=1)
  
    self.posEmbedding = PositionalEncoding(d_model, dropout)


  def forward(self, inputs):
    ''' 
        inputs: batches with N*H*W*C
    '''

    # local update 1
    x = self.resBlock_1(inputs)
    x = self.posEmbedding(x)
    anchors = x[:,:, self.half_anchor_dist::self.anchor_dist, 
                            self.half_anchor_dist::self.anchor_dist].clone()

    # flatten feature maps
    N, C, H, W = x.shape
    x = x.reshape(N, C, H*W).transpose(-1,-2)
    anchors = anchors.reshape(N, C, anchors.shape[2]* anchors.shape[3]).transpose(-1,-2)

    # gloabl update with two-stage multi-head self-attention
    anchors_new = self.dropout(self.selfAttn(anchors, x, x)[0])
    residual = self.dropout(self.selfAttn(x, anchors_new, anchors_new)[0])
    norm_1 = self.layer_norm_1(x + residual)

    # local update 2
    norm_1 = self.change_channel(norm_1)
    norm_1 = norm_1.transpose(-1,-2).reshape(N, self.out_dim, H, W)
    outputs = self.resBlock_2(norm_1)

    return outputs



class GlobalLocalEncoder(nn.Module):
  
    def __init__(self, anchor_dist, output_dim, dropout=0.):
        super().__init__()

        self.anchor_dist = anchor_dist
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = GlobalLocalBlock(self.anchor_dist, 64, 2, 96, dropout, stride=2)
        self.layer2 = GlobalLocalBlock(self.anchor_dist, 96, 3, 96, dropout, stride=1)
        self.layer3 = GlobalLocalBlock(self.anchor_dist//2, 96, 4, 128, dropout, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    # def _make_layer(self, in_dim, out_dim, dropout=0., stride=1):
    #     layer1 = GlobalLocalBlock(self.anchor_dist, in_dim, in_dim//32, out_dim, dropout=0., stride=stride)
    #     layer2 = GlobalLocalBlock(self.anchor_dist, out_dim, out_dim//32, out_dim, dropout=0., stride=1)
    #     layers = (layer1, layer2)
        
    #     return nn.Sequential(*layers)


    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.relu1(self.norm1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x