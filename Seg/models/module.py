import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)     

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = torch.chunk(self.qkv(x), 3, dim=1)
        q, k, v = map(lambda y: rearrange(y, 'B (h d) H W -> B h (H W) d', h=self.heads), (q, k, v))
        q = q * self.scale

        att = einsum('b h x d, b h y d -> b h x y', q, k)
        att = F.softmax(att, dim=-1)
        out = einsum('b h x y, b h y d -> b h x d', att, v)
        out = rearrange(out, 'B h (H W) d -> B (h d) H W', H=H)
        out = self.proj(out)
        return out


class ConvGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)

        self.dwc = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)

        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwc(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super(AttBlock, self).__init__()

        self.att = Attention(dim, heads)
        self.ffn = ConvGLU(dim, 2*dim)
        
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
