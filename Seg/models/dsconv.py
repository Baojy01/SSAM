import torch.nn as nn


class DSC_BN_ACT(nn.Module):
    """'
    Depthwise Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                                  nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        if self.kernel_size == 1:
            out = self.proj(x)
        else:
            out = self.proj(self.dwc(x))

        return out


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
                                   nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        out = self.convs(x)
        return out