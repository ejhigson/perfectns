#!/usr/bin/env python
"""
Contains helper functions for the 'gaussian_cached' prior.

This is needed as the 'gaussian' prior suffers from overflow errors and
numerical instability for very low values of X, which are reached in
high dimensional problems.
"""

import warnings
import numpy as np
import nestcheck.io_utils as iou
import perfectns.maths_functions as mf


def interp_r_logx_dict(n_dim, prior_scale, **kwargs):
    """
    Generate a dictionary containing arrays of logx and r values for use in
    interpolation, as well as a record of the settings used.

    Parameters
    ----------
    n_dim: int
        Number of dimensions.
    prior_scale: float
        Gaussian prior's standard deviation.
    logx_min: float
        Values for interpolation are generated between logx_min and logx_max.
    save_dict: bool, optional
        Whether or not to cache the output dictionary.
    cache_dir: str, optional
        Directory in which to cache output if save_dict is True.
    interp_density: float
        Number of points to include in interpolation arrays per unit of logx
        (the number of points is cast to int so this can be a fraction).
    logx_max: float, optional
        Values for interpolation are generated between logx_min and logx_max.

    Returns
    -------
    interp_dict: dict
    """
    logx_min = kwargs.pop('logx_min')  # no default, must specify
    save_dict = kwargs.pop('save_dict', True)
    cache_dir = kwargs.pop('cache_dir', 'cache')
    interp_density = kwargs.pop('interp_density')  # no default, must specify
    if n_dim > 1000 and logx_min >= -4500:
        warnings.warn(
            ('Interp_r_logx_dict: WARNING: n_dim=' + str(n_dim) + ': '
             'for very high dimensions, depending on the likelihood, you may '
             'need to lower logx_min=' + str(logx_min)), UserWarning)
    if n_dim < 50:
        warnings.warn(
            ('Interp_r_logx_dict: WARNING: n_dim=' + str(n_dim) + ': '
             'for n_dim<50 the "gaussian" prior works fine and you should '
             'use it instead of the "gaussian_cached" prior'), UserWarning)
    # use a smaller logx_max as the point at which the logx at which the
    # scipy method (scipy.special.gammainc) fails is lower in lower
    # dimensions. logx_max=-10 will mean the gaussian_cached prior works
    # for all n_dim>2.
    if n_dim < 100:
        logx_max = kwargs.pop('logx_max', -10)
    elif n_dim < 250:
        logx_max = kwargs.pop('logx_max', -100)
    else:
        logx_max = kwargs.pop('logx_max', -200)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    save_name = (cache_dir + '/interp_gauss_prior_' + str(n_dim) + 'd_' +
                 str(prior_scale) + 'rmax_' + str(logx_min) + 'xmin_' +
                 str(logx_max) + 'xmax_' + str(interp_density) + 'id')
    try:
        interp_dict = iou.pickle_load(save_name)
    except (OSError, IOError):  # Python 2 and 3 compatable
        print(save_name)
        print('Interp file not found - try generating new data')
        r_max = mf.gaussian_r_given_logx(logx_max, prior_scale, n_dim)
        # Iteratively reduce r_min until its corresponding logx value is less
        # than logx_min. This process depends only on mpmath functions which
        # can handle arbitrarily small numbers.
        r_min = r_max
        while mf.gaussian_logx_given_r(r_min, prior_scale, n_dim) > logx_min:
            r_min /= 2.0
        r_array = np.linspace(r_min, r_max,
                              int((logx_max - logx_min) * interp_density))
        logx_array = mf.gaussian_logx_given_r(r_array, prior_scale, n_dim)
        interp_dict = {'interp_density': interp_density,
                       'logx_min': logx_min,
                       'n_dim': n_dim,
                       'prior_scale': prior_scale,
                       'r_array': r_array,
                       'logx_array': logx_array}
        if save_dict:
            iou.pickle_save(interp_dict, save_name)
    return interp_dict
