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
        assert type(prior).__name__ in ['uniform', 'gaussian'], 'No logz_analytic set up for ' + type(self).__name__ + ' likelihoods and ' + type(prior).__name__ + ' priors.'
        if type(prior).__name__ == "uniform":
            return - mf.nsphere_logvol(n_dim, radius=prior.prior_scale) + mf.gaussian_logx_given_r(prior.prior_scale, self.likelihood_scale, n_dim)
        elif type(self.prior).__name__ == "gaussian":
            # See "Products and convolutions of Gaussian probability density functions" (P Bromiley, 2003) page 3 for a derivation of this result
            self.logz_analytic = (-self.n_dim / 2.) * np.log(2 * np.pi * (self.likelihood_scale ** 2 + prior.prior_scale ** 2))
