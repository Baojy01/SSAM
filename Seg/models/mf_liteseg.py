import torch.nn as nn
import torch.nn.functional as F
import torch

from .aspp import ASPP, SPPM
from .dsconv import DSC_BN_ACT
from .module import AttBlock


class MFLiteSeg(nn.Module):
    def __init__(self, backbone, n_classes=4, pretrain_weight=' ', pre_train=False):
        super(MFLiteSeg, self).__init__()

        self.backbone = backbone
        if pre_train:
            state_dict = torch.load(pretrain_weight)
            backbone.load_state_dict(state_dict)
        dim_list = backbone.dim_list

        out_ch = [32, 48, 64]

        # self.aspp = SPPM(dim_list[-1], out_ch[-1])

        self.att = nn.Sequential(nn.Conv2d(dim_list[-1], out_ch[-1], 1),
                                 AttBlock(out_ch[-1]))

        self.seg_head = nn.Sequential(DSC_BN_ACT(dim_list[0] + sum(out_ch), out_ch[0], 3, 1, 1),
                                      nn.Conv2d(out_ch[0], n_classes, kernel_size=1))

        self.fuse1 = Fuse(dim_list[2], out_ch[1], out_ch[-1])
        self.fuse2 = Fuse(dim_list[1], out_ch[0], out_ch[-1])

        for m in self.seg_head:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        # x4 = self.aspp(x4)
        x4 = self.att(x4)
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x3 = self.fuse1(x3, x4, x1.size()[2:])

        x2 = self.fuse2(x2, x4, x1.size()[2:])

        out = torch.cat((x1, x2, x3, x4), dim=1)

        out = self.seg_head(out)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)

        return out


class Fuse(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()

        self.fuse = DSC_BN_ACT(in_channels + hidden_channels, out_channels, 3, 1, 1)

    def forward(self, x_low, x_high, size):
        x_up = F.interpolate(x_high, x_low.size()[2:], mode='bilinear', align_corners=True)
        x = self.fuse(torch.cat([x_low, x_up], dim=1))
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x
