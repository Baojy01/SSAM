from typing import Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import _get_filters, _construct_2d_filter, _pad1d, _pad2d, _remove_pad, _construct_2d_filter_low

"""
PyTorch implementation for one level 1d and 2d dwt/idwt. This code references to:
Repo: https://github.com/v0lta/PyTorch-Wavelet-Toolbox
Paper: ptwt - The PyTorch Wavelet Toolbox. JMLR, 2024. http://jmlr.org/papers/v25/23-0636.html
"""


def dwt(x: torch.Tensor, wave: str = 'db1', mode: str = 'zero'):
    """
    Performs a 1 level 1D Discrete  wavelet transform of an image. The results are
    same as the MATLAB  function "dwt".

    Args:
        x (torch.Tensor): Input with shape B, C, L.
        wave (str): The wavelet to use.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodic'. The padding scheme. Default: 'zero'.
    Returns:
        List of coefficients with shape: B, C, L/2. The ordering in these 2 coefficients is: (A, D).
    """
    assert len(x.shape) == 3
    C = x.shape[1]
    dec_lo, dec_hi, _, _ = _get_filters(wave, flip=True, device=x.device, dtype=x.dtype)  # 1, k
    res = _pad1d(x, wave, mode=mode)
    x_lo = F.conv1d(res, dec_lo.repeat(C, 1, 1), stride=2, groups=C)
    x_hi = F.conv1d(res, dec_hi.repeat(C, 1, 1), stride=2, groups=C)

    return x_lo, x_hi


def idwt(x: Union[Tuple[torch.Tensor], List[torch.Tensor]], wave: str = 'db1'):
    """
    Performs a 1d-DWT Inverse reconstruction of an image.
    NOTE: Just for 1-level DWT reconstruction.

    Args:
        x: tuple of lowpass and bandpass coefficients with shape: N, C, L.
        wave (str): The wavelet to use.
    Returns:
        Reconstructed input with shape: N, C, 2L.
    """
    assert len(x) == 2 and x[0].shape == x[1].shape and len(x[0].shape) == 3

    C = x[0].shape[1]
    _, _, rec_lo, rec_hi = _get_filters(wave, flip=False, device=x[0].device, dtype=x[0].dtype)
    rec_filter = torch.stack([rec_lo, rec_hi], 0)  # 2, 1, k
    res_lo = torch.stack(x, dim=2)  # B, C, 2, L
    res_lo = rearrange(res_lo, 'B C n L -> (B C) n L')  # B C 2 L  -> B*C 2 L

    res_lo = F.conv_transpose1d(res_lo, rec_filter, stride=2)  # B*C 1 L
    res_lo = rearrange(res_lo.squeeze(1), '(B C) L -> B C L', C=C)  # B*C 1 L -> B*C L

    # remove the padding
    L = (2 * rec_filter.shape[-1] - 3) // 2
    res_lo = _remove_pad(res_lo, [L] * 2)

    return res_lo


def dwt2d(x: torch.Tensor, wave: str = 'db1', mode: str = 'zero'):
    """
    Performs a 1 level 2D Discrete  wavelet transform of an image. The results are
    same as the MATLAB  function "dwt2".

    Args:
        x (torch.Tensor): Input with shape N, C, H, W.
        wave (str): The wavelet to use.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodic'. The padding scheme. Default: 'zero'.
    Returns:
        List of coefficients with shape: B, C, H/2, W/2. The ordering in these 4
        coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
    """
    assert len(x.shape) == 4
    C = x.shape[1]
    dec_lo, dec_hi, _, _ = _get_filters(wave, flip=True, device=x.device, dtype=x.dtype)
    dec_filter = _construct_2d_filter(dec_lo, dec_hi)  # 4 1 k k

    res_ll = _pad2d(x, wave, mode=mode)

    res_ll = rearrange(res_ll, 'B C H W -> (B C) 1 H W')  # B C H W ->  B*C 1 H W
    res_ll = F.conv2d(res_ll, dec_filter, stride=2)  # B*C 1 H W ->  B*C 4 H W

    out = torch.chunk(res_ll, 4, dim=1)  # B*C 1 H W
    out = [rearrange(y.squeeze(1), '(B C) H W -> B C H W', C=C) for y in out]

    return out


def dwt_low(x: torch.Tensor, wave: str = 'db1', mode: str = 'zero'):
    assert len(x.shape) == 4
    C = x.shape[1]
    dec_lo, _, _, _ = _get_filters(wave, flip=True, device=x.device, dtype=x.dtype)
    dec_filter = _construct_2d_filter_low(dec_lo).repeat(C, 1, 1, 1)  # C 1 k k
    res_ll = _pad2d(x, wave, mode=mode)
    out = F.conv2d(res_ll, dec_filter, stride=2, groups=C)

    return out


def idwt2d(x: Union[Tuple[torch.Tensor], List[torch.Tensor]], wave: str = 'db1'):
    """
    Performs a 2d-DWT Inverse reconstruction of an image.
    NOTE: Just for 1-level DWT reconstruction.

    Args:
        x: tuple of lowpass and bandpass coefficients with shape: N, C, H, W.
        wave (str): The wavelet to use.
    Returns:
        Reconstructed input with shape: N, C, 2H, 2W.
    """
    assert len(x) == 4 and len(x[0].shape) == 4
    assert x[0].shape == x[1].shape == x[2].shape == x[3].shape

    C = x[0].shape[1]
    _, _, rec_lo, rec_hi = _get_filters(wave, flip=False, device=x[0].device, dtype=x[0].dtype)
    rec_filter = _construct_2d_filter(rec_lo, rec_hi)

    res_ll = torch.stack(x, dim=2)  # B C 4 H W
    res_ll = rearrange(res_ll, 'B C n H W -> (B C) n H W')  # B C 4 H W  -> B*C 4 H W

    res_ll = F.conv_transpose2d(res_ll, rec_filter, stride=2)  # B*C 4 H W  -> B*C 1 H W
    res_ll = rearrange(res_ll.squeeze(1), '(B C) H W -> B C H W', C=C)  # B*C 1 H W -> B*C H W  -> B C H W

    L = (2 * rec_filter.shape[-1] - 3) // 2
    res_ll = _remove_pad(res_ll, [L] * 4)

    return res_ll


class DWT1D(nn.Module):
    """
    Performs a 1 level 1D Discrete  wavelet transform of an image.

    Args:
        wave (str): The wavelet to use.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodic'. The padding scheme. Default: 'zero'.
    """

    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        self.wave = wave
        self.mode = mode

    def forward(self, x):
        """
        Args:
            x (tensor): Input with shape :B, C, L

        Returns:
            List of coefficients for each scale with shape: B, C, L/2. The ordering in
            these 2 coefficients is: (A, D).
        """
        x_lo, x_hi = dwt(x, self.wave, self.mode)

        return x_lo, x_hi


class IDWT1D(nn.Module):
    """
    Performs a 1d-DWT Inverse reconstruction of an image.
    NOTE: Just for 1-level DWT reconstruction.

    Args:
        wave (str): The wavelet to use.
    """

    def __init__(self, wave='db1'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        """
        Args:
            x (A, D): tuple or list of lowpass and bandpass coefficients  with shape : N, C, L.
        Returns:
            Reconstructed input with shape: N, C, 2L
        """
        res_lo = idwt(x, self.wave)

        return res_lo


class DWT2D(nn.Module):
    """
    Performs a 1 level 2D Discrete  wavelet transform of an image.

    Args:
        wave (str): The wavelet to use.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodic'. The padding scheme. Default: 'zero'.
    """

    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        self.wave = wave
        self.mode = mode

    def forward(self, x):
        """
        Args:
            x (tensor): Input with shape :B, C, H, W

        Returns:
            List of coefficients with shape: B, C, H/2, W/2. The ordering in these 4
            coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        x_a, x_h, x_v, x_d = dwt2d(x, self.wave, self.mode)
        return x_a, x_h, x_v, x_d


class IDWT2D(nn.Module):
    """
    Performs a 2d-DWT Inverse reconstruction of an image.
    NOTE: Just for 1-level DWT reconstruction.

    Args:
        wave (str): The wavelet to use.
    """

    def __init__(self, wave='db1'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        """
        Args:
            x (A, H, V, D): tuple or list of lowpass and bandpass coefficients with shape : N, C, H, W.
        Returns:
            Reconstructed input with shape: N, C, 2H, 2W.
        """
        res_ll = idwt2d(x.chunk(4, dim=1), self.wave)
        return res_ll


class DWT_Low(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        self.wave = wave
        self.mode = mode

    def forward(self, x):
        out = dwt_low(x, self.wave, self.mode)

        return out
