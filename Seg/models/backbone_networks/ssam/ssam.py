import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from .torch_dwt import DWT2D
from .utils import DropPath, DAConv, ADSCDConv, PConv
from .utils import act_layer, norm_layer


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='BN', act_type=None):
        super(Downsample, self).__init__()
        assert out_channels % in_channels == 0

        self.proj = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, groups=in_channels, bias=False)
        self.norm = norm_layer(out_channels, norm_type)
        self.act = act_layer(act_type)

        self.dwt = DWT2D()

    def forward(self, x):
        # out = self.dwt(x)
        out = torch.stack(self.dwt(x), dim=2)
        out = rearrange(out, 'B C n H W -> B (C n) H W')
        out = self.act(self.norm(self.proj(out)))

        return out


class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='BN', act_type='GELU'):
        super(ConvStem, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_channels//2, norm_type),
            act_layer(act_type),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_channels, norm_type),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class MPConv(nn.Module):
    def __init__(self, dim, norm_type='BN', act_type='GELU'):
        super().__init__()
        assert dim % 4 == 0
        d = dim // 4
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(nn.Sequential(nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False),
                                            norm_layer(d, norm_type)))

        self.act = act_layer(act_type)

    def forward(self, x):
        x_list = torch.chunk(x, 4, dim=1)
        out = []
        y = x_list[0]
        for i, conv in enumerate(self.convs):
            if i == 0:
                y = x_list[i]
            else:
                y = y + x_list[i]
            y = conv(y)
            out.append(y)
        out = self.act(torch.cat(out, dim=1))
        return out


class MBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4, drop_path=0., norm_type='BN', act_type='GELU'):
        super(MBlock, self).__init__()

        hidden_dim = int(dim * expand_ratio)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                norm_layer(dim, norm_type),
                act_layer(act_type),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                norm_layer(dim, norm_type),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
                norm_layer(hidden_dim, norm_type),
                act_layer(act_type),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
                norm_layer(hidden_dim, norm_type),
                act_layer(act_type),
                nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
                norm_layer(dim, norm_type),
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        out = x + self.drop_path(self.conv(x))

        return out


class MPBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4, drop_path=0., norm_type='BN', act_type='GELU'):
        super().__init__()
        assert expand_ratio > 1

        hidden_dim = int(dim * expand_ratio)

        self.convs = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            norm_layer(hidden_dim, norm_type),
            act_layer(act_type),
            MPConv(hidden_dim, norm_type, act_type),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            norm_layer(dim, norm_type)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = x + self.drop_path(self.convs(x))

        return out


class SAMM(nn.Module):
    # Spatial - adaptive Modulation Module(SAMM)
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0
        d = dim // 4
        self.pc = PConv(d, d)
        self.dac_x = DAConv(d, d, groups=d, morph=0)
        self.dac_y = DAConv(d, d, groups=d, morph=1)
        self.dc = ADSCDConv(d, kernel_size=3)
        self.convs = nn.ModuleList([self.pc, self.dac_x, self.dac_y, self.dc])
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x_list = torch.chunk(x, 4, dim=1)
        out_list = [conv(y) for y, conv in zip(x_list, self.convs)]
        out = torch.cat(out_list, dim=1)
        out = self.act(self.proj(out)) * x
        return out


class AFMM(nn.Module):
    # Adaptive Frequency Mixing Module (AFMM)
    def __init__(self, channels, groups=4):
        super(AFMM, self).__init__()
        assert channels % groups == 0

        self.groups = groups
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, groups, channels // groups, channels // groups))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, groups, channels // groups, channels // groups))

        self.b1 = nn.Parameter(self.scale * torch.randn(2, groups, channels // groups))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, groups, channels // groups))

    def forward(self, x):
        B, C, H, W = x.shape

        out = fft.rfft2(x.float(), dim=(2, 3), norm="ortho")
        origin_ffted = out
        out = rearrange(out, 'B (g d) H W -> B g d H W', g=self.groups)

        o1_real = torch.einsum('b k i h w, k i o -> b k o h w', out.real, self.w1[0]) - torch.einsum(
            'b k i h w, k i o -> b k o h w', out.imag, self.w1[1]) + self.b1[0, :, :, None, None]
        o1_imag = torch.einsum('b k i h w, k i o -> b k o h w', out.imag, self.w1[0]) + torch.einsum(
            'b k i h w, k i o -> b k o h w', out.real, self.w1[1]) + self.b1[1, :, :, None, None]

        o1_real, o1_imag = F.relu(o1_real), F.relu(o1_imag)

        o2_real = torch.einsum('b k i h w, k i o -> b k o h w', o1_real, self.w2[0]) - torch.einsum(
            'b k i h w, k i o -> b k o h w', o1_imag, self.w2[1]) + self.b2[0, :, :, None, None]
        o2_imag = torch.einsum('b k i h w, k i o -> b k o h w', o1_imag, self.w2[0]) + torch.einsum(
            'b k i h w, k i o -> b k o h w', o1_real, self.w2[1]) + self.b2[1, :, :, None, None]

        out = torch.stack([o2_real, o2_imag], dim=-1)
        out = torch.view_as_complex(out.float())
        out = rearrange(out, 'B h d H W -> B (h d) H W', h=self.groups)

        out = out * origin_ffted
        out = fft.irfft2(out, s=(H, W), dim=(2, 3), norm="ortho")

        return out


class SSAM(nn.Module):
    # Spatial-Spectral Adaptive Modulation (SSAM)
    def __init__(self, dim, drop_path=0., layer_scale_init_value=None, norm_type='LN', post_norm=False):
        super(SSAM, self).__init__()

        self.samm = SAMM(dim)
        self.afmm = AFMM(dim)

        self.norm1 = norm_layer(dim, norm_type)
        self.norm2 = norm_layer(dim, norm_type)

        if layer_scale_init_value is not None and type(layer_scale_init_value) in [int, float]:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1))
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1))
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.post_norm = post_norm

    def forward(self, x):

        if self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.samm(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.afmm(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.samm(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.afmm(self.norm2(x)))
        else:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.samm(x)))
                x = x + self.drop_path(self.norm2(self.afmm(x)))
            else:
                x = x + self.drop_path(self.samm(self.norm1(x)))
                x = x + self.drop_path(self.afmm(self.norm2(x)))

        return x


def make_layers(in_channels, out_channels, layers, expand_ratio, dpr_list, stride=1,
                layer_scale_init_value=1e-5, block_type='C'):
    assert block_type in ['C', 'T']
    blocks = []
    down = Downsample(in_channels, out_channels) if stride == 2 else nn.Identity()
    blocks.append(down)

    if block_type == 'C':
        for block_idx in range(layers):
            blocks.append(MPBlock(out_channels, expand_ratio, dpr_list[block_idx]))

    elif block_type == 'T':
        for block_idx in range(layers):
            blocks.append(SSAM(out_channels, dpr_list[block_idx], layer_scale_init_value))

    return nn.Sequential(*blocks)


def split_list(init_list: list, split: list):
    count = 0
    out_list = []
    for i in split:
        sub_list = init_list[count:i + count]
        out_list.append(sub_list)
        count += i

    return out_list


class SSAMNet(nn.Module):
    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [2, 4, 4, 2],
                         'embed_dims': [32, 64, 128, 256],
                         'strides': [1, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_path_rate': 0,
                         'layer_scale_init_value': None}),

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [2, 4, 4, 2],
                         'embed_dims': [48, 96, 192, 384],
                         'strides': [1, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_path_rate': 0,
                         'layer_scale_init_value': None}),

        **dict.fromkeys(['b', 'base', 'B'],
                        {'layers': [2, 4, 4, 2],
                         'embed_dims': [64, 128, 256, 512],
                         'strides': [1, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_path_rate': 0,
                         'layer_scale_init_value': None}),

        **dict.fromkeys(['l', 'large', 'L'],
                        {'layers': [2, 4, 4, 2],
                         'embed_dims': [96, 192, 384, 768],
                         'strides': [1, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_path_rate': 0,
                         'layer_scale_init_value': None})
    }

    def __init__(self, arch='tiny'):
        super().__init__()

        if isinstance(arch, str):
            assert arch in self.arch_settings, f'Unavailable arch, please choose from {set(self.arch_settings)} or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, f'The arch dict must have "layers" and "embed_dims", but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        expand_ratio = arch['expand_ratio']
        strides = arch['strides']
        drop_path_rate = arch['drop_path_rate']
        layer_scale_init_value = arch['layer_scale_init_value']

        dpr = [*np.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule
        dpr_list = split_list(dpr, layers)
        self.dim_list = embed_dims

        self.conv_stem = ConvStem(3, embed_dims[0])

        self.layers1 = make_layers(embed_dims[0], embed_dims[0], layers[0], expand_ratio[0], dpr_list[0],
                                   strides[0], layer_scale_init_value, 'C')

        self.layers2 = make_layers(embed_dims[0], embed_dims[1], layers[1], expand_ratio[1], dpr_list[1],
                                   strides[1], layer_scale_init_value, 'C')

        self.layers3 = make_layers(embed_dims[1], embed_dims[2], layers[2], expand_ratio[2], dpr_list[2],
                                   strides[2], layer_scale_init_value, 'T')

        self.layers4 = make_layers(embed_dims[2], embed_dims[3], layers[3], expand_ratio[3], dpr_list[3],
                                   strides[3], layer_scale_init_value, 'T')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_stem(x)

        x1 = self.layers1(x)

        x2 = self.layers2(x1)

        x3 = self.layers3(x2)

        x4 = self.layers4(x3)

        return x1, x2, x3, x4


def SSAMNet_Tiny():
    model = SSAMNet(arch='t')
    return model


def SSAMNet_Small():
    model = SSAMNet(arch='s')
    return model


def SSAMNet_Base():
    model = SSAMNet(arch='b')
    return model


def SSAMNet_Large():
    model = SSAMNet(arch='l')
    return model
