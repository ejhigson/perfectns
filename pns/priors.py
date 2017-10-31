#!/usr/bin/python
"""
Object containing spherically symetric likelihood and prior.
"""

import numpy as np
from scipy import interpolate
import pns.maths_functions as mf
import pns.cached_gaussian_prior as cgp


class uniform(object):

    """Spherically symetric uniform prior."""

    def __init__(self, prior_scale):
        self.prior_scale = prior_scale

    def logx_given_r(self, r, n_dim):
        return mf.nsphere_logx_given_r(r, self.prior_scale, n_dim)

    def r_given_logx(self, logx, n_dim):
        return mf.nsphere_r_given_logx(logx, self.prior_scale, n_dim)


class gaussian(object):

    """Spherically symetric uniform prior."""

    def __init__(self, prior_scale):
        self.prior_scale = prior_scale

    def logx_given_r(self, r, n_dim):
        return mf.gaussian_logx_given_r(r, self.prior_scale, n_dim)

    def r_given_logx(self, logx, n_dim):
        return mf.gaussian_r_given_logx(logx, self.prior_scale, n_dim)


class gaussian_cached(object):

    """
    Spherically symetric uniform prior.
    The scipy inverse gamma function is not numerially stable so cache
    r_given_logx by using the mpmath logx_given_r and linearly
    interpolating.
    """

    def __init__(self, prior_scale, **kwargs):
        self.prior_scale = prior_scale
        # if n_dim is specified we can cache the interpolation now.
        # Otherwise wait until it is called.
        if 'n_dim' in kwargs:
            self.interp_d = cgp.interp_r_logx_dict(kwargs['n_dim'],
                                                   self.prior_scale)
            self.interp_f = interpolate.interp1d(self.interp_d['logx_array'],
                                                 self.interp_d['r_array'])
        else:
            self.interp_d = {'n_dim': None, 'prior_scale': None}

    def logx_given_r(self, r, n_dim):
        return mf.gaussian_logx_given_r(r, self.prior_scale, n_dim)

    def r_given_logx(self, logx, n_dim):
        self.check_cache(n_dim)
        try:
            if isinstance(logx, np.ndarray):
                r = np.zeros(logx.shape)
                ind = np.where(logx <= self.interp_d['logx_array'].max())[0]
                r[ind] = self.interp_f(logx[ind])
                ind = np.where(logx > self.interp_d['logx_array'].max())[0]
                r[ind] = mf.scipy_gaussian_r_given_logx(logx[ind],
                                                        self.prior_scale,
                                                        n_dim)
                assert np.count_nonzero(r) == r.shape[0], \
                    "r contains zeros! r = " + str(r)
                return r
            else:
                if logx <= self.interp_d['logx_array'].max():
                    return self.interp_f(logx)
                else:
                    return mf.scipy_gaussian_r_given_logx(logx,
                                                          self.prior_scale,
                                                          n_dim)
        except ValueError:
            print("ValueError in r_given_logx for gaussian prior")
            print("logx_interp is ", self.interp_d['logx_array'])
            print("input logx is", logx)

    def check_cache(self, n_dim):
        # Check that the input dimension matches that of the cached
        # interpolation function, and if needed recalculate.
        if (n_dim != self.interp_d['n_dim']) or (self.prior_scale !=
                                                 self.interp_d['prior_scale']):
            print("re-cache prior: input (n_dim, self.prior_scale) = (" +
                  str(n_dim) + ", " + str(self.prior_scale) + ") =! cached " +
                  "(n_dim, prior_scale) = (" + str(self.interp_d['n_dim']) +
                  ", " + str(self.prior_scale) + ")")
            self.interp_d = cgp.interp_r_logx_dict(n_dim, self.prior_scale)
            self.interp_f = interpolate.interp1d(self.interp_d['logx_array'],
                                                 self.interp_d['r_array'])
