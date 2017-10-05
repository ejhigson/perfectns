#!/usr/bin/python
"""
Object representing spherically symetric likelihoods.
"""

import pns.maths_functions as mf
import numpy as np


class gaussian(object):

    """Spherically symetric gaussian likelihood."""

    def __init__(self, likelihood_scale=1.0):
        self.likelihood_scale = likelihood_scale

    def logl_given_r(self, r, n_dim):
        return mf.log_gaussian_given_r(r, self.likelihood_scale, n_dim)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        return mf.r_given_log_gaussian(logl, self.likelihood_scale, n_dim)

    def logz_analytic(self, prior, n_dim):
        assert type(prior).__name__ in ['uniform', 'gaussian'], \
            ('No logz_analytic set up for ' + type(self).__name__ +
             ' likelihoods and ' + type(prior).__name__ + ' priors.')
        if type(prior).__name__ == "uniform":
            # logZ = -log volume of uniform prior + correction for truncation
            logvol = mf.nsphere_logvol(n_dim, radius=prior.prior_scale)
            # the fraction of the gaussian
            return -logvol + mf.gaussian_logx_given_r(prior.prior_scale,
                                                      self.likelihood_scale,
                                                      n_dim)
        elif type(prior).__name__ == "gaussian":
            # See "Products and convolutions of Gaussian probability density
            # functions" (P Bromiley, 2003) page 3 for a derivation of this
            # result
            return (-n_dim / 2.) * np.log(2 * np.pi *
                                          (self.likelihood_scale ** 2 +
                                           prior.prior_scale ** 2))


class exp_power(object):

    """Spherically symetric exponential power likelihood."""

    def __init__(self, likelihood_scale=1, power=2):
        self.likelihood_scale = likelihood_scale
        self.power = power

    def logl_given_r(self, r, n_dim):
        return mf.log_exp_power_given_r(r, self.likelihood_scale,
                                        n_dim, b=self.power)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        return mf.r_given_log_exp_power(logl, self.likelihood_scale,
                                        n_dim, b=self.power)
