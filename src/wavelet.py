#! /usr/bin/env python

"""
Helpers for Constructing Wavelet Bases

author: krissankaran@stanford.edu
date: 03/18/2018
"""

import pywt
import numpy as np
from plotnine import ggplot, geom_point, aes
import pandas as pd


def wavelet_basis(x, name="db3", resolution=2 ** 10):
    """Evaluate Wavelet basis at x positions

    :param x: Positions at which to evaluate wavelet basis functions
    :param name: Name of the wavelet family to use (default Daubechies 3)
    :param resolution: How finely to evaluate the wavelet basis numerically
        (default to 2 ** 10).
    :return z: A numpy array whose rows are time indices and columns are
        wavelet basis functions.

    >>> x = np.random.random(100)
    >>> z = wavelet_basis(x)
    >>> (ggplot(
      pd.DataFrame({
        "x": x,
        "z": z[:, 10]
      })) +
      geom_point(
      aes(x = "x", y = "z")
     ))
    """
    x_stretch = (x - min(x)) / (max(x) - min(x)) * resolution
    x_floor = np.floor(x_stretch).astype("Int64")
    x_diff = x_stretch - x_floor

    w = pywt.Wavelet(name)
    z = []
    wd_obj = pywt.wavedec(np.zeros(resolution), w)
    for scale_ix in range(len(wd_obj)):
        for loc_ix in range(len(wd_obj[scale_ix])):
            wd_obj[scale_ix][loc_ix] = 1
            wr_obj = pywt.waverec (wd_obj, w)
            wd_obj[scale_ix][loc_ix] = 0

            wr_obj = np.append(wr_obj, wr_obj[-2:]) # so interpolation doesn't run out of bounds
            z.append((1 - x_diff) * wr_obj[x_floor] + x_diff * wr_obj[x_floor + 1])

    return np.stack(z).T
