import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import BasicBlock, conv1x1, ConvNextBlock
from .layers.block import Block
from .layers.attention import MemEffAttention


class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """

    def __init__(
        self,
        input_dim=3,
        output_dim=256,
        ratio=1.0,
        downsample=8,
        num_blocks=2,
        initial_dim=64,
        block_dims=[64, 128, 256],
        norm_layer=nn.BatchNorm2d,
        init_weight=False,
        pretrain="resnet34",
        return_all_feat=False,
    ):
        super().__init__()
        # Config
        block = BasicBlock
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(
            input_dim, initial_dim, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        if pretrain == "resnet34":
            n_block = [3, 4, 6]
        elif pretrain == "resnet18":
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(
            block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0]
        )  # 1/2
        self.layer2 = self._make_layer(
            block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1]
        )  # 1/4
        if num_blocks == 3:
            self.layer3 = self._make_layer(
                block,
                block_dims[2],
                stride=2 if downsample == 8 else 1,
                norm_layer=norm_layer,
                num=n_block[2],
            )  # 1/4 or 1/8
        self.final_conv = conv1x1(
            block_dims[2] if num_blocks == 3 else block_dims[1], output_dim
        )
        self.num_blocks = num_blocks
        self.return_all_feat = return_all_feat
        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            from torchvision.models import (
                resnet18,
                ResNet18_Weights,
                resnet34,
                ResNet34_Weights,
            )

            if self.pretrain == "resnet18":
                pretrained_dict = resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1
                ).state_dict()
            else:
                pretrained_dict = resnet34(
                    weights=ResNet34_Weights.IMAGENET1K_V1
                ).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == "conv1.weight":
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        output_all_scales = []
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        if self.return_all_feat:
            output_all_scales.append(x)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        if self.num_blocks == 3:
            for i in range(len(self.layer3)):
                x = self.layer3[i](x)
        # Output
        output = self.final_conv(x)
        if self.return_all_feat:
            output_all_scales.append(output)
            return output_all_scales
        return output


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channels=324, dim=128):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(corr_channels, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class MFMotionEncoder(nn.Module):
    def __init__(self, corr_channels, dim=128):
        super(MFMotionEncoder, self).__init__()
        cor_planes = corr_channels * 2
        self.convc1 = nn.Conv2d(cor_planes, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2 * 2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2 * 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class RAFTUpdateBlock(nn.Module):
    def __init__(
        self,
        corr_channels=324,
        hdim=128,
        cdim=128,
        downsample_factor=8,
        num_blocks=2,
        predict_visconf=False,
        iter_visconf=False,
        use_temporal_attn=False,
    ):
        # net: hdim, inp: cdim
        super(RAFTUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channels=corr_channels, dim=cdim)
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(
                ConvNextBlock(
                    (
                        2 * cdim + hdim + 2
                        if iter_visconf and predict_visconf
                        else 2 * cdim + hdim
                    ),
                    hdim,
                )
            )
        self.refine = nn.ModuleList(self.refine)
        self.flow_head = FlowHead(hdim, hidden_dim=hdim * 2, output_dim=2)
        self.mask = nn.Sequential(
            nn.Conv2d(hdim, 2 * hdim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * hdim, downsample_factor**2 * 9, 1, padding=0),
        )

        self.iter_visconf = iter_visconf
        if predict_visconf:
            self.visconf_head = nn.Sequential(
                nn.Conv2d(hdim, 2 * hdim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * hdim, 2, kernel_size=3, padding=1),
            )

        # temporal attention
        self.use_temporal_attn = use_temporal_attn
        if use_temporal_attn:
            self.temporal_attn_refine = nn.ModuleList()
            for i in range(num_blocks):
                self.temporal_attn_refine.append(
                    Block(
                        dim=hdim,
                        num_heads=4,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        proj_bias=True,
                        ffn_bias=True,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=0.0,
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        attn_class=MemEffAttention,
                        qk_norm=False,
                        fused_attn=False,
                        rope=None,
                    )
                )

                # initialize the last layers of attention and mlp to zero
                self.temporal_attn_refine[i].attn.proj.weight.data.zero_()
                self.temporal_attn_refine[i].attn.proj.bias.data.zero_()
                self.temporal_attn_refine[i].mlp.fc2.weight.data.zero_()
                self.temporal_attn_refine[i].mlp.fc2.bias.data.zero_()

    def forward(self, net, inp, corr, flow, visconf=None, batch_size=None):
        motion_features = self.encoder(flow, corr)
        if self.iter_visconf and visconf is not None:
            inp = torch.cat([inp, motion_features, visconf], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)

        if self.use_temporal_attn:
            assert batch_size is not None

        for i, blk in enumerate(self.refine):
            net = blk(torch.cat([net, inp], dim=1))

            if self.use_temporal_attn:
                c, h, w = net.shape[1:]

                net = net.reshape(batch_size, -1, *net.shape[1:])  # [B, T-1, C, H, W]
                net = (
                    net.permute(0, 3, 4, 1, 2)
                    .contiguous()
                    .reshape(batch_size * h * w, -1, c)
                )  # [B*H*W, T-1, C]
                net = self.temporal_attn_refine[i](net)  # [B*H*W, T-1, C]
                net = (
                    net.reshape(batch_size, h, w, -1, c)
                    .permute(0, 3, 4, 1, 2)
                    .contiguous()
                    .reshape(-1, c, h, w)
                )

        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        if self.iter_visconf and visconf is not None:
            delta_visconf = self.visconf_head(net)
        else:
            delta_visconf = None
        return net, mask, delta_flow, delta_visconf
