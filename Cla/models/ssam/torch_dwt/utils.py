from typing import Tuple, List, Union

import pywt
import torch
import torch.nn.functional as F


def _translate_mode(mode: str):
    if mode == "zero":
        return "constant"
    elif mode == "reflect":
        return mode
    elif mode == "periodic":
        return "circular"
    elif mode == "symmetric":
        return mode
    raise ValueError(f"Padding mode not supported: {mode}")


def _get_pad(data_len: int, filter_len: int):
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filter_len (int): The size of the used filter.

    Returns:
        A tuple (padr, padl). The first entry specifies how many numbers
        to attach on the right. The second entry covers the left side.
    """
    # pad to ensure we see all filter positions and for pywt compatibility.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filter_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filter_len - 1)/2)
    # => floor([data_len + total_pad - filter_len]/2) + 1
    #    = floor((data_len + filter_len - 1)/2)
    # (data_len + total_pad - filter_len) + 2 = data_len + filter_len - 1
    # total_pad = 2*filter_len - 3

    # we pad half of the total required padding on each side.
    padr = (2 * filter_len - 3) // 2
    padl = (2 * filter_len - 3) // 2

    # pad to even single length.
    padr += data_len % 2

    return padr, padl


def _pad_symmetric_1d(data: torch.Tensor, pad_list: Tuple[int, int]):
    padl, padr = pad_list
    dim_len = data.shape[0]
    if padl > dim_len or padr > dim_len:
        if padl > dim_len:
            data = _pad_symmetric_1d(data, (dim_len, 0))
            padl = padl - dim_len
        if padr > dim_len:
            data = _pad_symmetric_1d(data, (0, dim_len))
            padr = padr - dim_len
        return _pad_symmetric_1d(data, (padl, padr))
    else:
        cat_list = [data]
        if padl > 0:
            cat_list.insert(0, data[:padl].flip(0))
        if padr > 0:
            cat_list.append(data[-padr::].flip(0))
        return torch.cat(cat_list, dim=0)


def _pad_symmetric(data: torch.Tensor, pad_lists: List[Tuple[int, int]]):
    if len(data.shape) < len(pad_lists):
        raise ValueError("not enough dimensions to pad.")

    dims = len(data.shape) - 1
    for pos, pad_list in enumerate(pad_lists[::-1]):
        current_axis = dims - pos
        data = data.transpose(0, current_axis)
        data = _pad_symmetric_1d(data, pad_list)
        data = data.transpose(current_axis, 0)
    return data


def _outer(a: torch.Tensor, b: torch.Tensor):
    """Torch implementation of outer in numpy  for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def _construct_2d_filter(lo: torch.Tensor, hi: torch.Tensor):
    """Construct two-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        Stacked 2d-filters of dimension
        [filters_no, 1, height, width].
        The four filters are ordered ll, lh, hl, hh.

    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filters = torch.stack([ll, lh, hl, hh], 0)
    filters = filters.unsqueeze(1)
    return filters


def _construct_2d_filter_low(lo: torch.Tensor):
    ll = _outer(lo, lo)
    filters = ll.unsqueeze(0)
    return filters


def _pad1d(data: torch.Tensor, wave: str, mode: str = None):
    """Pad the input signal to make the fwt matrix work.
    The padding assumes a future step will transform the last axis.
    Args:
        data (torch.Tensor): Input data ``[batch_size, 1, time]``
        wave (str): The name of a pywt wavelet.
        mode: The desired padding mode for extending the signal along the edges.
            Defaults to "zero".
    Returns:
        A PyTorch tensor with the padded input data
    """
    wavelet = pywt.Wavelet(wave)

    if mode is None:
        mode = "zero"
    pad_mode = _translate_mode(mode)
    padr, padl = _get_pad(data.shape[-1], len(wavelet))

    if pad_mode == "symmetric":
        data_pad = _pad_symmetric(data, [(padl, padr)])
    else:
        data_pad = F.pad(data, [padl, padr], mode=pad_mode)
    return data_pad


def _pad2d(data: torch.Tensor, wave: str, mode: str = None):
    """Pad data for the 2d FWT.
    This function pads along the last two axes.
    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wave (str): The name of a pywt wavelet.
        mode (str): The desired padding mode for extending the signal along the edges.
            Support 'zero', 'symmetric', 'reflect' or 'periodic'. Default: 'zero'
    Returns:
        The padded output tensor.
    """
    wavelet = pywt.Wavelet(wave)

    pad_bottom, pad_top = _get_pad(data.shape[-2], len(wavelet))
    pad_right, pad_left = _get_pad(data.shape[-1], len(wavelet))

    if mode is None:
        mode = "zero"
    pad_mode = _translate_mode(mode)

    if pad_mode == "symmetric":
        data_pad = _pad_symmetric(data, [(pad_top, pad_bottom), (pad_left, pad_right)])
    else:
        data_pad = F.pad(data, [pad_left, pad_right, pad_top, pad_bottom], mode=pad_mode)
    return data_pad


def _create_tensor(data, flip: bool, device: torch.device, dtype: torch.dtype):
    if flip:
        if isinstance(data, torch.Tensor):
            return data.flip(-1).unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(data[::-1], device=device, dtype=dtype).unsqueeze(0)
    else:
        if isinstance(data, torch.Tensor):
            return data.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(data, device=device, dtype=dtype).unsqueeze(0)


def _get_filters(wave: str, flip: bool, device: torch.device, dtype: torch.dtype = torch.float32):
    """Convert input wavelet to filter tensors.
    Args:
        wave (str): The name of a pywt wavelet.
        flip (bool): Flip filters left-right, if true.
    Returns:
        A tuple (dec_lo, dec_hi, rec_lo, rec_hi) containing the four filter tensors
    """
    wavelet = pywt.Wavelet(wave)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo, flip, device, dtype)
    dec_hi_tensor = _create_tensor(dec_hi, flip, device, dtype)
    rec_lo_tensor = _create_tensor(rec_lo, flip, device, dtype)
    rec_hi_tensor = _create_tensor(rec_hi, flip, device, dtype)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _remove_pad(x: torch.Tensor, pad_list: Union[List[int], Tuple[int]]):
    assert len(pad_list) == 2 or len(pad_list) == 4
    if len(pad_list) == 2:
        assert len(x.shape) == 3  # B C L
        pad_l, pad_r = pad_list
        if pad_l > 0:
            x = x[..., pad_l:]
        if pad_r > 0:
            x = x[..., :-pad_r]
    elif len(pad_list) == 4:
        assert len(x.shape) == 4  # B C H W
        pad_l, pad_r, pad_t, pad_b = pad_list
        if pad_t > 0:
            x = x[..., pad_t:, :]
        if pad_b > 0:
            x = x[..., :-pad_b, :]
        if pad_l > 0:
            x = x[..., pad_l:]
        if pad_r > 0:
            x = x[..., :-pad_r]
    else:
        ValueError('Unsupported pad_list input!')

    return x
