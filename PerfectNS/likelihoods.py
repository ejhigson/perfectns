#!/usr/bin/env python
"""
Classes representing spherically symmetric likelihoods.

Each likelihood class must contain a member function giving the log likelihood
as a function of the radial coordinate r = |theta| and the number of dimensions

    def logl_given_r(self, r, n_dim):
        ...

The number of dimensions is not stored in this class but in the
PerfectNSSettings object to ensure it is the same for the likelihood and the
prior.

Likelihood classes may also optionally contain the inverse function

    def r_given_logl(self, logl, n_dim):
        ...

(although this is not needed to generate nested sampling runs). Another
optional function is the analytic value of the log evidence for some
given prior and dimension, which is useful for checking results:

    def logz_analytic(self, prior, n_dim):
        ...
"""

import PerfectNS.maths_functions as mf
import numpy as np


class gaussian(object):

    """Spherically symmetric Gaussian likelihood."""

    def __init__(self, likelihood_scale=1.0):
        self.likelihood_scale = likelihood_scale

    def logl_given_r(self, r, n_dim):
        """
        Parameters
        ----------
        r: float or numpy array
        n_dim: int
        Returns
        -------
        logl: same type and size as r
        """
        return mf.log_gaussian_given_r(r, self.likelihood_scale, n_dim)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        """
        Parameters
        ----------
        logl: float or numpy array
        n_dim: int
        Returns
        -------
        r: same type and size as logl
        """
        return mf.r_given_log_gaussian(logl, self.likelihood_scale, n_dim)

    def logz_analytic(self, prior, n_dim):
        """
        Returns analytic value of the log evidence for the input prior and
        dimension if known.
        If not set up for this prior an AssertionError is thrown (this is
        caught in functions which check analytical values where they are
        available).
        """
        assert type(prior).__name__ in ['uniform', 'gaussian'], \
            ('No logz_analytic set up for ' + type(self).__name__ +
             ' likelihoods and ' + type(prior).__name__ + ' priors.')
        if type(prior).__name__ == 'uniform':
            # logZ = -log volume of uniform prior + correction for truncation
            logvol = mf.nsphere_logvol(n_dim, radius=prior.prior_scale)
            # To find how much of the likelihood's mass lies within the uniform
            # prior we can reuse the gaussian_logx_given_r function which is
            # used for Gaussian priors.
            return -logvol + mf.gaussian_logx_given_r(prior.prior_scale,
                                                      self.likelihood_scale,
                                                      n_dim)
        elif type(prior).__name__ == 'gaussian':
            # See 'Products and convolutions of Gaussian probability density
            # functions' (P Bromiley, 2003) page 3 for a derivation of this
            # result
            return (-n_dim / 2.) * np.log(2 * np.pi *
                                          (self.likelihood_scale ** 2 +
                                           prior.prior_scale ** 2))


class exp_power(object):

    """
    Spherically symmetric exponential power likelihood.
    When power=2, this is the same as the Gaussian likelihood.
    """

    def __init__(self, likelihood_scale=1, power=2):
        self.likelihood_scale = likelihood_scale
        self.power = power

    def logl_given_r(self, r, n_dim):
        """
        Parameters
        ----------
        r: float or numpy array
        n_dim: int
        Returns
        -------
        logl: same type and size as r
        """
        return mf.log_exp_power_given_r(r, self.likelihood_scale,
                                        n_dim, b=self.power)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        """
        Parameters
        ----------
        logl: float or numpy array
        n_dim: int
        Returns
        -------
        r: same type and size as logl
        """
        return mf.r_given_log_exp_power(logl, self.likelihood_scale,
                                        n_dim, b=self.power)


class cauchy(object):

    """Spherically symmetric Cauchy likelihood."""

    def __init__(self, likelihood_scale=1):
        self.likelihood_scale = likelihood_scale

    def logl_given_r(self, r, n_dim):
        """
        Parameters
        ----------
        r: float or numpy array
        n_dim: int
        Returns
        -------
        logl: same type and size as r
        """
        return mf.log_cauchy_given_r(r, self.likelihood_scale, n_dim)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        """
        Parameters
        ----------
        logl: float or numpy array
        n_dim: int
        Returns
        -------
        r: same type and size as logl
        """
        return mf.r_given_log_cauchy(logl, self.likelihood_scale, n_dim)
