"""
Paper:      PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model
Url:        https://arxiv.org/abs/2204.02681
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dsconv import Conv_BN_ACT


class PPLiteSeg(nn.Module):
    def __init__(self, backbone, num_class=1):
        super().__init__()
        decoder_dim = [32, 64, 128]
        # decoder_dim = [64, 96, 128]
        dim_list = backbone.dim_list

        self.backbone = backbone
        self.sppm = SPPM(dim_list[-1], decoder_dim[-1])
        self.decoder = FLD(dim_list, decoder_dim, num_class)

    def forward(self, x):
        size = x.size()[2:]
        x2, x3, x4 = self.backbone(x)[1:]
        x4 = self.sppm(x4)
        x = self.decoder(x2, x3, x4, size)

        return x


def make_pool_layer(in_channels, out_channels, pool_size):
    return nn.Sequential(nn.AdaptiveAvgPool2d(pool_size),
                         Conv_BN_ACT(in_channels, out_channels, 1))


class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hid_channels = int(in_channels // 4)
        self.pool1 = make_pool_layer(in_channels, hid_channels, 1)
        self.pool2 = make_pool_layer(in_channels, hid_channels, 2)
        self.pool3 = make_pool_layer(in_channels, hid_channels, 4)
        self.conv = Conv_BN_ACT(hid_channels, out_channels, kernel_size=3, padding=1)
        # self.conv = conv3x3(hid_channels, out_channels)

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.pool3(x), size, mode='bilinear', align_corners=True)
        x = self.conv(x1 + x2 + x3)
        return x


class FLD(nn.Module):
    def __init__(self, in_dim, out_dim, num_class):
        super().__init__()
        self.fusion1 = UAFM(in_dim[2], out_dim[-1], out_dim[1])

        self.fusion2 = UAFM(in_dim[1], out_dim[1], out_dim[0])

        self.seg_head = Conv_BN_ACT(out_dim[0], num_class, 3, 1, 1)

    def forward(self, x1, x2, x3, size):
        x = self.fusion1(x2, x3)
        x = self.fusion2(x1, x)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class UAFM(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super().__init__()

        self.conv = conv1x1(in_channels, hidden_size)
        self.attention = SpatialAttentionModule()
        self.fuse = Conv_BN_ACT(hidden_size, out_channels, 3, 1, 1)

    def forward(self, x_low, x_high):
        size = x_low.size()[2:]
        x_low = self.conv(x_low)
        x_up = F.interpolate(x_high, size, mode='bilinear', align_corners=True)
        alpha = self.attention(x_up, x_low)
        x = alpha * x_up + (1 - alpha) * x_low
        x = self.fuse(x)

        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = conv1x1(4, 1)

    def forward(self, x_up, x_low):
        mean_up = torch.mean(x_up, dim=1, keepdim=True)
        max_up, _ = torch.max(x_up, dim=1, keepdim=True)
        mean_low = torch.mean(x_low, dim=1, keepdim=True)
        max_low, _ = torch.max(x_low, dim=1, keepdim=True)
        x = self.conv(torch.cat([mean_up, max_up, mean_low, max_low], dim=1))
        x = torch.sigmoid(x)  # [N, 1, H, W]

        return x



class ChannelAttentionModule(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = conv1x1(4*out_channels, out_channels)

    def forward(self, x_up, x_low):
        avg_up = self.avg_pool(x_up)
        max_up = self.max_pool(x_up)
        avg_low = self.avg_pool(x_low)
        max_low = self.max_pool(x_low)
        x = self.conv(torch.cat([avg_up, max_up, avg_low, max_low], dim=1))
        x = torch.sigmoid(x)    # [N, C, 1, 1]

        return x



# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
