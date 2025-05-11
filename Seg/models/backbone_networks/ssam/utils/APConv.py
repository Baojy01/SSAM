import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class PConv(nn.Module):
    """ Pinwheel-shaped Convolution using the Asymmetric Padding method. """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        mid = out_channels // 4
        p = [(kernel_size, 0, 1, 0), (0, kernel_size, 0, 1), (0, 1, kernel_size, 0), (1, 0, 0, kernel_size)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(in_channels, mid, (1, kernel_size), s=stride, p=0, g=mid)
        self.ch = Conv(in_channels, mid, (kernel_size, 1), s=stride, p=0, g=mid)
        
        self.cat = nn.Sequential(Conv(out_channels, out_channels, 2, s=1, p=0, g=out_channels),
                                 nn.Conv2d(out_channels, out_channels, 1))

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
