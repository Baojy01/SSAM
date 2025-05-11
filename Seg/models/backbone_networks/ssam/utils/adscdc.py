import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class AdaptiveDynamicKernel(nn.Module):
    def __init__(self, dim, kernel_size, num_groups=4, reduction=4, bias=False):
        super().__init__()
        self.padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(kernel_size)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(dim // reduction),
                                  nn.ReLU(),
                                  nn.Conv2d(dim // reduction, num_groups * dim, kernel_size=1, bias=False)
                                  )

        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(num_groups, dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x))  # B, G*C, k, k
        scale = rearrange(scale, 'B (G C) H W -> B G C H W', C=C)  # B, G, C, k, k
        scale = torch.softmax(scale, dim=1)

        weight = torch.sum(scale * self.weight, dim=1)  # B, C, k, k
        weight = rearrange(weight, 'B C H W -> (B C) 1 H W')  # B*C, 1, k, k

        if self.bias is not None:
            scale = self.proj(F.adaptive_avg_pool2d(x, output_size=1))  # B, G*C, 1, 1
            scale = rearrange(scale, 'B (G C) H W -> B G (C H W)', C=C)  # B, G, C
            scale = torch.softmax(scale, dim=1)
            bias = torch.sum(scale * self.bias, dim=1).flatten(0)  # B*C
        else:
            bias = None

        return weight, bias


class CAM(nn.Module):
    def __init__(self, dim, reduction=4):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SCDConv(nn.Module):
    """
    Salient Central Difference Depthwise Convolution
    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.attn = CAM(dim)

    def forward(self, x):
        theta = self.attn(x)

        w_sum = self.conv.weight.sum(dim=[2, 3], keepdim=True)  # C, C/g, 1, 1
        w_center = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        w_center = w_center[:, :, None, None]  # center of self.conv.weight,  C, C/g, 1, 1

        out_center = F.conv2d(x, w_center, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        weight_diff = self.conv.weight - F.pad(w_sum, [self.kernel_size // 2] * 4, "constant", 0)  # convert to k*k kernel

        out_diff = F.conv2d(x, weight_diff, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)

        out = out_diff + theta * out_center

        return out


class ADSCDConv(nn.Module):
    """
    Depthwise Adaptive Dynamic Salient Central Differential Convolution
    """

    def __init__(self, dim, kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.adk = AdaptiveDynamicKernel(dim, kernel_size, bias=bias)
        self.attn = CAM(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        theta = self.attn(x).reshape(-1, 1, 1, 1)  # B, C, 1, 1 -> B*C, 1, 1, 1
        weight, bias = self.adk(x)  # B*C, 1, k, k; B*C

        w_sum = weight.sum(dim=[2, 3], keepdim=True)  # B*C, 1, 1, 1
        w_center = weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        w_center = w_center[:, :, None, None]  # center of weight,  B*C, 1, 1, 1

        weight_s = weight - F.pad(w_sum - theta * w_center, [self.kernel_size // 2] * 4, "constant", 0)  # convert to k*k kernel

        out = F.conv2d(x.reshape(1, -1, H, W), weight_s, bias, stride=self.stride, padding=self.kernel_size//2, groups=B*C)
        out = out.reshape(B, C, H, W)
        return out
