#!/usr/bin/python
"""
Object containing spherically symetric likelihood and prior.
"""

import pns.maths_functions as mf


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
