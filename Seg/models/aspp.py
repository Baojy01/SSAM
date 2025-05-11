import torch.nn as nn
import torch
import torch.nn.functional as F
from .dsconv import DSC_BN_ACT, Conv_BN_ACT


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=(1, 2, 3)):
        super().__init__()

        self.stage1 = DSC_BN_ACT(in_channels, out_channels, 1)
        self.stage2 = DSC_BN_ACT(in_channels, out_channels, 3, padding=dilation[0], dilation=dilation[0])
        self.stage3 = DSC_BN_ACT(in_channels, out_channels, 3, padding=dilation[1], dilation=dilation[1])
        self.stage4 = DSC_BN_ACT(in_channels, out_channels, 3, padding=dilation[2], dilation=dilation[2])
        self.stage5 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    DSC_BN_ACT(in_channels, out_channels, 1))

        self.conv = DSC_BN_ACT(5 * out_channels, out_channels, 1)

        self.__init_weight()

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x)
        x5 = self.stage5(x)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=True)

        x = self.conv(torch.cat([x1, x2, x3, x4, x5], dim=1))
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


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