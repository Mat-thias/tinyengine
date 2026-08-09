# ----------------------------------------------------------------------
# Project: Right In-Place Convolution
# Title:   right_inplace.py
#
# Reference papers:
#    Yet to be published
# Contact authors:
#  - Opegbemi Matthias Busoye, busoyeopegbemimatthias@gmail.com
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

"""
Right in-place convolution -- workspace gap
============================================================

Sizes the gap Delta by which a right-inplace kernel shifts its input before
computing, so the output written from the base of the buffer can never
overtake the input still being read. The generated call carries Delta as its
last argument and the allocator reserves Delta + input for the layer.

Notation
--------
    C_in, C_out         input / output channel counts        (cin, cout)
    H_in, W_in          input spatial dims                    (hin, win)
    H_p, W_p            padded input spatial dims
    H_out, W_out        output spatial dims                  (hout, wout)
    k_h, k_w            kernel height / width
    S_h, S_w            stride                                (s_h, s_w)
    d_h, d_w            dilation
    p_h, p_w            padding
    Delta               workspace gap (in scalars)

Public API
----------
    out_dims(...)                    (H_p, W_p, H_out, W_out)
    right_inplace_gap_unpadded_ext(...)  Delta, dense (unpadded) layout
    right_inplace_gap_unpadded(...)      square-kernel / symmetric wrapper

The `_ext` function takes the full per-axis parameter set; the non-`ext`
wrapper assumes square kernels and symmetric stride / dilation / padding.
"""

import math


# ==========================================================================
# Dimension helper
# ==========================================================================

def out_dims(hin, win, k_h, k_w, s_h, s_w, d_h, d_w, p_h, p_w):
    """Return (H_p, W_p, H_out, W_out) for the given convolution parameters."""
    hin_p = hin + 2 * p_h
    win_p = win + 2 * p_w
    k_eff_h = d_h * (k_h - 1) + 1
    k_eff_w = d_w * (k_w - 1) + 1
    hout = (hin_p - k_eff_h) // s_h + 1
    wout = (win_p - k_eff_w) // s_w + 1
    return hin_p, win_p, hout, wout


# ==========================================================================
# Right-inplace, dense (padding-not-materialized) layout
# ==========================================================================

def _clamp(a, x, b):
    return min(max(a, x), b)


def right_inplace_gap_unpadded_ext(
    cin, cout, k_h, k_w, hin, win,
    s_h=1, s_w=1, d_h=1, d_w=1, p_h=0, p_w=0
):
    """
    Safe gap for right-inplace convolution under a DENSE layout: padding is
    not materialized, so only the hin*win*cin real input pixels occupy
    workspace (packed at gap + (r*win+c)*cin), and kernel taps landing in
    the padding region read/write nothing. Contrast with right_inplace_gap_ext,
    where padding is stored like any other input pixel and D(r,c) is affine
    everywhere; here the padding boundary makes D(r,c) piecewise-linear
    (a "clamp" kink where the receptive field crosses from padding into real
    input, and another where it exits back into padding), so instead of 4
    corners we max over every candidate row/column where a kink can occur.

    Requires p_h < k_eff_h and p_w < k_eff_w (matches
    validate_right_inplace_unpadded_ext's precondition).
    """
    s_h = s_h or 1
    s_w = s_w or 1
    d_h = d_h or 1
    d_w = d_w or 1
    p_h = p_h or (k_h - 1) * d_h // 2
    p_w = p_w or (k_w - 1) * d_w // 2

    hin_p, win_p, hout, wout = out_dims(
        hin, win, k_h, k_w,
        s_h, s_w, d_h, d_w, p_h, p_w
    )
    if hout <= 0 or wout <= 0:
        return 0

    def D_row(r):
        return r * wout * cout - _clamp(0, r * s_h - p_h, hin) * win * cin

    def D_col(c):
        return c * cout - _clamp(0, c * s_w - p_w, win) * cin

    r_c = {0, hout - 1}
    for v in (p_h / s_h, (p_h + hin - 1) / s_h):
        r_c.add(int(math.floor(v)))
        r_c.add(int(math.ceil(v)))
    r_c = [r for r in r_c if 0 <= r <= hout - 1]

    c_c = {0, wout - 1}
    for v in (p_w / s_w, (p_w + win - 1) / s_w):
        c_c.add(int(math.floor(v)))
        c_c.add(int(math.ceil(v)))
    c_c = [c for c in c_c if 0 <= c <= wout - 1]

    return max(cout, max(D_row(r) for r in r_c) + max(D_col(c) for c in c_c) + cout)


def right_inplace_gap_unpadded(cin, cout, k, hin, win, stride=1, dilation=1, padding=0):
    """Square-kernel / symmetric wrapper around right_inplace_gap_unpadded_ext."""
    return right_inplace_gap_unpadded_ext(
        cin, cout, k, k, hin, win,
        stride, stride, dilation, dilation, padding, padding
    )
