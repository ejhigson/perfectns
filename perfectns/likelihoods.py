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

import numpy as np
import perfectns.maths_functions as mf


class Gaussian(object):

    """Spherically symmetric Gaussian likelihood."""

    def __init__(self, likelihood_scale=1.0):
        """Store Gaussian likelihood's sigma."""
        self.likelihood_scale = likelihood_scale

    def logl_given_r(self, r, n_dim):
        """
        Get loglikelihood values for input radial coordinates.

        Parameters
        ----------
        r: float or numpy array
            Radial coordinates.
        n_dim: int
            Number of dimensions.

        Returns
        -------
        logl: same type and size as r
            Loglikelihood values.
        """
        return mf.log_gaussian_given_r(r, self.likelihood_scale, n_dim)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        """
        Get the radial coordinates corresponding to the input loglikelihood
        values.

        Parameters
        ----------
        logl: float or numpy array
            Loglikelihood values.
        n_dim: int
            Number of dimensions.

        Returns
        -------
        r: same type and size as logl
            Radial coordinates.
        """
        return mf.r_given_log_gaussian(logl, self.likelihood_scale, n_dim)

    def logz_analytic(self, prior, n_dim):
        """
        Returns analytic value of the log evidence for the input prior and
        dimension if it is available.

        If not set up for this prior an AssertionError is thrown (this is
        caught in functions which check analytical values where they are
        available).

        Parameters
        ----------
        prior: object
        n_dim: int
            Number of dimensions.

        Returns
        -------
        float
            Analytic value of log Z for this likelihood given the prior and
            number of dimensions.
        """
        assert type(prior).__name__ in ['Uniform', 'Gaussian'], \
            ('No logz_analytic set up for ' + type(self).__name__ +
             ' likelihoods and ' + type(prior).__name__ + ' priors.')
        if type(prior).__name__ == 'Uniform':
            # logZ = -log volume of uniform prior + correction for truncation
            logvol = mf.nsphere_logvol(n_dim, radius=prior.prior_scale)
            # To find how much of the likelihood's mass lies within the uniform
            # prior we can reuse the gaussian_logx_given_r function which is
            # used for Gaussian priors.
            return -logvol + mf.gaussian_logx_given_r(prior.prior_scale,
                                                      self.likelihood_scale,
                                                      n_dim)
        else:
            assert type(prior).__name__ == 'Gaussian'
            # See 'Products and convolutions of Gaussian probability density
            # functions' (P Bromiley, 2003) page 3 for a derivation of this
            # result
            return (-n_dim / 2.) * np.log(2 * np.pi *
                                          (self.likelihood_scale ** 2 +
                                           prior.prior_scale ** 2))


class ExpPower(object):

    """
    Spherically symmetric exponential power likelihood.
    When power=2, this is the same as the Gaussian likelihood.
    """

    def __init__(self, likelihood_scale=1, power=2):
        """Save the likelihood scale and power."""
        self.likelihood_scale = likelihood_scale
        self.power = power

    def logl_given_r(self, r, n_dim):
        """
        Get loglikelihood values for input radial coordinates.

        Parameters
        ----------
        r: float or numpy array
            Radial coordinates.
        n_dim: int
            Number of dimensions.

        Returns
        -------
        logl: same type and size as r
            Loglikelihood values.
        """
        return mf.log_exp_power_given_r(r, self.likelihood_scale,
                                        n_dim, b=self.power)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        """
        Get the radial coordinates corresponding to the input loglikelihood
        values.

        Parameters
        ----------
        logl: float or numpy array
            Loglikelihood values.
        n_dim: int
            Number of dimensions.

        Returns
        -------
        r: same type and size as logl
            Radial coordinates.
        """
        return mf.r_given_log_exp_power(logl, self.likelihood_scale,
                                        n_dim, b=self.power)


class Cauchy(object):

    """Spherically symmetric Cauchy likelihood."""

    def __init__(self, likelihood_scale=1):
        """Save the likelihood scale."""
        self.likelihood_scale = likelihood_scale

    def logl_given_r(self, r, n_dim):
        """
        Get loglikelihood values for input radial coordinates.

        Parameters
        ----------
        r: float or numpy array
            Radial coordinates.
        n_dim: int
            Number of dimensions.

        Returns
        -------
        logl: same type and size as r
            Loglikelihood values.
        """
        return mf.log_cauchy_given_r(r, self.likelihood_scale, n_dim)

    # Optional functions

    def r_given_logl(self, logl, n_dim):
        """
        Get the radial coordinates corresponding to the input loglikelihood
        values.

        Parameters
        ----------
        logl: float or numpy array
            Loglikelihood values.
        n_dim: int
            Number of dimensions.

        Returns
        -------
        r: same type and size as logl
            Radial coordinates.
        """
        return mf.r_given_log_cauchy(logl, self.likelihood_scale, n_dim)
