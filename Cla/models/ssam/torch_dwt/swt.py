"""This module implements stationary wavelet transforms."""

from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import _get_filters, _construct_2d_filter

"""
PyTorch implementation for one level 1d and 2d swt/iswt. This code references to:
Repo: https://github.com/v0lta/PyTorch-Wavelet-Toolbox
Paper: ptwt - The PyTorch Wavelet Toolbox. JMLR, 2024. http://jmlr.org/papers/v25/23-0636.html
"""


def _circular_pad(x: torch.Tensor, padding_dimensions: Union[Tuple[int], List[int]]):
    """Pad a tensor in circular mode, more than once if needed."""
    trailing_dimension = x.shape[-1]

    # if every padding dimension is smaller than or equal the trailing dimension,
    # we do not need to manually wrap
    if not any(
            padding_dimension > trailing_dimension
            for padding_dimension in padding_dimensions
    ):
        return F.pad(x, padding_dimensions, mode="circular")

    # repeat to pad at maximum trailing dimensions until all padding dimensions are zero
    while any(padding_dimension > 0 for padding_dimension in padding_dimensions):
        # reduce every padding dimension to at maximum trailing dimension width
        reduced_padding_dimensions = [
            min(trailing_dimension, padding_dimension)
            for padding_dimension in padding_dimensions
        ]
        # pad using reduced dimensions,
        # which will never throw the circular wrap error
        x = F.pad(x, reduced_padding_dimensions, mode="circular")
        # remove the pad width that was just padded, and repeat
        # if any pad width is greater than zero
        padding_dimensions = [
            max(padding_dimension - trailing_dimension, 0)
            for padding_dimension in padding_dimensions
        ]

    return x


def swt(x: torch.Tensor, wave: str):
    """Compute a one-level 1d stationary wavelet transform. The results are
    same as the MATLAB  function "swt".

    Args:
        x (torch.Tensor): The input data with shape: B, C, L.
        wave (str): The wavelet to use. Default: 'periodic' padding.
    Returns:
          List of coefficients with shape: B, C, L. The ordering in these 2 coefficients is: (A, D).
    """
    assert len(x.shape) == 3
    C = x.shape[1]
    dec_lo, dec_hi, _, _ = _get_filters(wave, flip=True, device=x.device, dtype=x.dtype)
    filt_len = dec_lo.shape[-1]

    padl, padr = filt_len // 2 - 1, filt_len // 2
    res = _circular_pad(x, [padl, padr])
    x_lo = F.conv1d(res, dec_lo.repeat(C, 1, 1), stride=1, groups=C)
    x_hi = F.conv1d(res, dec_hi.repeat(C, 1, 1), stride=1, groups=C)

    return x_lo, x_hi


def iswt(x: Union[Tuple[torch.Tensor], List[torch.Tensor]], wave: str):
    """Invert a one level 1d stationary wavelet transform.

    Args:
        x (Tuple[torch.Tensor]): The coefficients as computed by the swt function with shape: N, C, L.
        wave (str): The wavelet to use.
    Returns:
        A reconstruction of the original swt input with shape: N, C, L.
    """
    assert len(x) == 2 and x[0].shape == x[1].shape and len(x[0].shape) == 3

    C = x[0].shape[1]
    _, _, rec_lo, rec_hi = _get_filters(wave, flip=False, device=x[0].device, dtype=x[0].dtype)
    filt_len = rec_lo.shape[-1]
    rec_filter = torch.stack([rec_lo, rec_hi], 0)

    res_lo = torch.stack(x, dim=2)  # B, C, 2, L
    res_lo = rearrange(res_lo, 'B C n L -> (B C) n L')  # B C 2 L  -> B*C 2 L

    padl, padr = filt_len // 2, filt_len // 2 - 1
    res_lo = _circular_pad(res_lo, [padl, padr])
    res_lo = F.conv_transpose1d(res_lo, rec_filter, groups=2, padding=(padl + padr))  # B*C 2 L
    res_lo = torch.mean(res_lo, dim=1, keepdim=False)  # B*C L
    res_lo = rearrange(res_lo, '(B C) L -> B C L', C=C)  # B*C L -> B C L

    return res_lo


def swt2d(x: torch.Tensor, wave: str):
    """Compute a one-level 2d stationary wavelet transform. The results are
    same as the MATLAB  function "swt2".

    Args:
        x (torch.Tensor): The input data with shape: B, C, H, W.
        wave (str): The wavelet to use. Default: 'periodic' padding.
    Returns:
        List of coefficients with shape: B, C, H, W. The ordering in these 4
        coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
    """
    assert len(x.shape) == 4
    C = x.shape[1]
    dec_lo, dec_hi, _, _ = _get_filters(wave, flip=True, device=x.device, dtype=x.dtype)
    dec_filter = _construct_2d_filter(dec_lo, dec_hi)  # 4 1 k k

    filt_len = dec_filter.shape[-1]
    padl, padr = filt_len // 2 - 1, filt_len // 2
    pad_t, pad_b = filt_len // 2 - 1, filt_len // 2

    res = _circular_pad(x, [padl, padr, pad_t, pad_b])

    res = rearrange(res, 'B C H W -> (B C) 1 H W')  # B C H W ->  B*C 1 H W
    res = F.conv2d(res, dec_filter, stride=1)  # B*C 1 H W ->  B*C 4 H W

    out = torch.chunk(res, 4, dim=1)  # B*C 1 H W
    out = [rearrange(y.squeeze(1), '(B C) H W -> B C H W', C=C) for y in out]

    return out


def iswt2d(x: Union[Tuple[torch.Tensor], List[torch.Tensor]], wave: str):
    """Invert a one level 2d stationary wavelet transform.

    Args:
        x (Tuple[torch.Tensor]): The coefficients as computed by the swt function.
        wave (str): The wavelet to use.
    Returns:
        A reconstruction of the original swt input.
    """
    assert len(x) == 4 and len(x[0].shape) == 4
    assert x[0].shape == x[1].shape == x[2].shape == x[3].shape

    C = x[0].shape[1]
    _, _, rec_lo, rec_hi = _get_filters(wave, flip=False, device=x[0].device, dtype=x[0].dtype)
    rec_filter = _construct_2d_filter(rec_lo, rec_hi)

    filt_len = rec_lo.shape[-1]

    res_ll = torch.stack(x, dim=2)  # B C 4 H W
    res_ll = rearrange(res_ll, 'B C n H W -> (B C) n H W')  # B C 4 H W  -> B*C 4 H W

    padl, padr = filt_len // 2, filt_len // 2 - 1
    pad_t, pad_b = filt_len // 2, filt_len // 2 - 1
    res_ll = _circular_pad(res_ll, [padl, padr, pad_t, pad_b])
    res_ll = F.conv_transpose2d(res_ll, rec_filter, groups=4, padding=(pad_t + pad_b, padl + padr))  # B*C 4 H W
    res_ll = torch.mean(res_ll, dim=1, keepdim=False)  # B*C H W
    res_ll = rearrange(res_ll, '(B C) H W -> B C H W', C=C)  # B*C H W -> B C H W

    return res_ll


class SWT1D(nn.Module):
    """
    Performs a one level 1D Stationary  wavelet transform of an image.

    Args:
        wave (str): The wavelet to use.
    """

    def __init__(self, wave='db1'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        """
        Args:
            x (tensor): Input with shape: B, C, L

        Returns:
            List of coefficients for each scale with shape: B, C, L. The ordering in
            these 2 coefficients is: (A, D).
        """
        x_lo, x_hi = swt(x, self.wave)

        return x_lo, x_hi


class ISWT1D(nn.Module):
    """
    Performs a 1d-SWT Inverse reconstruction of an image.
    NOTE: Just for 1-level SWT reconstruction.

    Args:
        wave (str): The wavelet to use.
    """

    def __init__(self, wave='db1'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        """
        Args:
            x (A, D): tuple or list of lowpass and bandpass coefficients  with shape: N, C, L.
        Returns:
            Reconstructed input with shape: N, C, 2L
        """
        res_lo = iswt(x, self.wave)

        return res_lo


class SWT2D(nn.Module):
    """
    Performs a one level 2D Stationary wavelet transform of an image.

    Args:
        wave (str): The wavelet to use.
    """

    def __init__(self, wave='db1'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        """
        Args:
            x (tensor): Input with shape: B, C, H, W

        Returns:
            List of coefficients with shape: B, C, H, W. The ordering in these 4
            coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        x_a, x_h, x_v, x_d = swt2d(x, self.wave)
        # out = torch.stack([x_a, x_h, x_v, x_d], dim=2)
        out = torch.cat([x_a, x_h, x_v, x_d], dim=1)
        return out


class ISWT2D(nn.Module):
    """
    Performs a 2d-SWT Inverse reconstruction of an image.
    NOTE: Just for 1-level SWT reconstruction.

    Args:
        wave (str): The wavelet to use.
    """

    def __init__(self, wave='db1'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        """
        Args:
            x (A, H, V, D): tuple or list of lowpass and bandpass coefficients with shape: N, C, H, W.
        Returns:
            Reconstructed input with shape: N, C, 2H, 2W.
        """
        res_ll = iswt2d(x, self.wave)
        return res_ll
