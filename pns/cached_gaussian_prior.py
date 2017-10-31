#!/usr/bin/python
"""Contains useful helper functions shared by many other modules."""


import numpy as np
# from numba import jit
from pynverse import inversefunc
import pns.save_load_utils as slu
import pns.maths_functions as mf


# Maths functions


def pynverse_gaussian_r_given_logx(logx, sigma, n_dim):
    """tbc"""
    r = inversefunc(mf.gaussian_logx_given_r, args=(sigma, n_dim),
                    y_values=logx)
    return float(r)


def interp_r_logx_dict(n_dim, prior_scale, **kwargs):
    """tbc"""
    logx_min = kwargs.get('logx_min', -4500)
    if n_dim > 1000 and logx_min >= -4500:
        print("Interp_r_logx_dict: WARNING: n_dim=" + str(n_dim) + ": "
              "for very high dimensions, depending on the likelihood, you may "
              "need to lower logx_min=" + str(logx_min))
    if n_dim < 100:
        print("Interp_r_logx_dict: WARNING: n_dim=" + str(n_dim) + ": "
              "for n_dim<100 the 'gaussian' prior works fine and you should "
              "use it instead of the 'gaussian_cached' prior")
        # use a smaller logx_max as the point at which the logx at which the
        # scipy method (scipy.special.gammainc) fails is lower in lower.
        # dimensions. logx_max=-10 will mean the gaussian_cached prior works
        # for all n_dim>2.
        logx_max = kwargs.get('logx_max', -10)
    else:
        logx_max = kwargs.get('logx_max', -200)
    interp_density = kwargs.get('interp_density', 10)
    version = 'v01'
    save_name = 'interp_gauss_prior_' + version + "_" + str(n_dim) + 'd_' + \
                str(prior_scale) + 'rmax_' + str(logx_min) + 'xmin_' + \
                str(logx_max) + 'xmax_' + str(interp_density) + 'id'
    try:
        interp_dict = slu.pickle_load(save_name)
        return interp_dict
    except OSError:  # "FileNotFoundError is a subclass of OSError
        print(save_name)
        print("Interp file not found - try generating new data")
        r_max = mf.scipy_gaussian_r_given_logx(logx_max, prior_scale, n_dim)
        r_min = pynverse_gaussian_r_given_logx(logx_min, prior_scale, n_dim)
        # guard against any errors in interpolation
        while mf.mpmath_gaussian_logx_given_r(r_min, prior_scale,
                                              n_dim) > logx_min:
            # print("increasing interp range", r_min,
            #       mf.mpmath_gaussian_logx_given_r(r_min, prior_scale, n_dim))
            r_min /= 2.0
        r_array = np.linspace(r_min, r_max,
                              int((logx_max - logx_min) * interp_density))
        logx_array = mf.gaussian_logx_given_r(r_array, prior_scale, n_dim)
        interp_dict = {'version': version,
                       'interp_density': interp_density,
                       'logx_min': logx_min,
                       'n_dim': n_dim,
                       'prior_scale': prior_scale,
                       'r_array': r_array,
                       'logx_array': logx_array}
        slu.pickle_save(interp_dict, save_name)
        return interp_dict


def test_scipy(logx, n_dim, sigma=10):
    r = mf.scipy_gaussian_r_given_logx(logx, sigma, n_dim)
    return mf.gaussian_logx_given_r(r, sigma, n_dim)


def get_logx_scipy_fail(n_dim, sigma):
    toll = 10 ** -8
    logx = -1.0
    test = test_scipy(logx, n_dim, sigma=sigma) / logx
    while 1. - toll < test < 1. + toll:
        logx -= 0.1
        test = test_scipy(logx, n_dim, sigma=sigma) / logx
    return 25  # add a healthy margin for error
