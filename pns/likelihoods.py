#!/usr/bin/python
"""
Object representing spherically symetric likelihoods.
"""

import pns.maths_functions as mf


class gaussian(object):

    """Spherically symetric gaussian likelihood."""

    def __init__(self, likelihood_scale=1.0):
        self.likelihood_scale = likelihood_scale

    # functions of likelihoods

    def logl_given_r(self, r, n_dim):
        return mf.log_gaussian_given_r(r, self.likelihood_scale, n_dim)

    def r_given_logl(self, logl, n_dim):
        return mf.r_given_log_gaussian(logl, self.likelihood_scale, n_dim)
