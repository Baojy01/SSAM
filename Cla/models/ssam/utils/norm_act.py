import torch.nn as nn
import torch
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, dim, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (dim,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



def act_layer(act_type: str = 'GELU'):
    if act_type == 'ReLU':
        act = nn.ReLU(inplace=True)
    elif act_type == 'SiLU':
        act = nn.SiLU(inplace=True)
    elif act_type == 'GELU':
        act = nn.GELU()
    elif act_type == 'Hardswish':
        act = nn.Hardswish(inplace=True)
    elif act_type == 'LeakyReLU':
        act = nn.LeakyReLU(inplace=True)
    elif act_type is None:
        act = nn.Identity()
    else:
        raise NotImplementedError(f'act_layer does not support {act_layer}')
    return act


def norm_layer(dim, norm_type: str = 'GN'):
    if norm_type == 'BN':
        norm = nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        norm = nn.GroupNorm(1, dim)
    elif norm_type == 'LN':
        norm = LayerNorm(dim)
    elif norm_type is None:
        norm = nn.Identity()
    else:
        raise NotImplementedError(f'norm_type does not support {norm_type}')

    return norm
