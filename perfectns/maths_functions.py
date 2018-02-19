#!/usr/bin/env python
"""
Generic mathematical functions.
"""


import numpy as np
import scipy
import scipy.stats
import scipy.special
import scipy.misc
import mpmath


def gaussian_r_given_logx(logx, sigma, n_dim):
    """
    Returns r coordinate corresponding to logx values for a Gaussian prior with
    the specificed standard deviation and dimension.

    This uses scipy.special.gammaincinv and requires exponentiating logx, so
    numerical errors occur with very low logx values.
    """
    exponent = scipy.special.gammaincinv(n_dim / 2., np.exp(logx))
    return np.sqrt(2 * exponent * sigma ** 2)


def gaussian_logx_given_r(r, sigma, n_dim):
    """
    Returns logx coordinate corresponding to r values for a Gaussian prior with
    the specificed standard deviation and dimension

    Uses mpmath package for arbitary precision.
    """
    exponent = 0.5 * (r / sigma) ** 2
    if isinstance(r, np.ndarray):  # needed to ensure output is numpy array
        logx = np.zeros(r.shape)
        for i, expo in enumerate(exponent):
            logx[i] = float(mpmath.log(mpmath.gammainc(n_dim / 2., a=0, b=expo,
                                                       regularized=True)))
        return logx
    else:
        return float(mpmath.log(mpmath.gammainc(n_dim / 2., a=0, b=exponent,
                                                regularized=True)))


def analytic_logx_terminate(settings):
    """
    Find logx_terminate analytically by assuming all likelihood at very low
    X approximately equals the maximum likelihood.
    This approximation breaks down in very high dimensions.
    """
    # use r=0 rather than logx=-np.inf as the latter causes numerical problems
    logl_max = settings.logl_given_r(0)
    if settings.logz_analytic() is not None:
        return logx_terminate_bound(logl_max, settings.termination_fraction,
                                    settings.logz_analytic())


def logx_terminate_bound(logl_max, termination_fraction, logz_analytic):
    """
    Find a lower bound logx_terminate analytically by assuming all likelihood
    at very low X approximately equals the maximum likelihood. This
    approximation breaks down in very high dimensions and the true logx
    terminate required will be larger.
    We want:
    Z_term = termination_fraction * Z_analytic
           = int_0^Xterm L(X) dX
           approx= Xterm L_max,
    so
    logx_term = log(termination_fraction) + log(Z_analytic) - logl_max
    """
    return np.log(termination_fraction) + logz_analytic - logl_max


def sample_nsphere_shells_beta(r, n_dim, n_sample=None):
    """
    Given some 1d numpy array of radial coordinates r, return a numpy array
    of sample coordinates on the hyperspherical shells with each radial
    coordinate.

    Sample single parameters on n_dim-dimensional sphere independently, as
    as described in Section 3.2 of 'Sampling Errors in nested sampling
    parameter estimation' (Higson et al. 2017).

    NB if each parameter is sampled independently and combined into a vector,
    that vector will not have a magnitude r.

    The n_sample argument can be used to set the number of parameters for which
    samples are returned (by symmetry they all have the same distribution).
    This is useful for saving memory in high dimensions.

    This function is much quicker than sample_nsphere_shells_normal when n_dim
    is high and n_sample is low.
    """
    assert isinstance(r, np.ndarray), 'input r must be a numpy array'
    if n_sample is None:
        n_sample = n_dim
    thetas = np.sqrt(np.random.beta(0.5, (n_dim - 1) * 0.5,
                                    size=(r.shape[0], n_sample)))
    # randomly select + or -
    thetas *= (-1) ** (np.random.randint(0, 2, size=thetas.shape))
    # multiply by r
    thetas *= r[:, None]
    return thetas


def sample_nsphere_shells_normal(r, n_dim, n_sample=None):
    """
    Given some 1d numpy array of radial coordinates r, return a numpy array
    of sample coordinates on the hyperspherical shells with each radial
    coordinate.

    This works by using the symmetry of the normal distribution to sample
    isotropically, then normalising each set of samples to lie on a spherical
    shell of the correct radius.

    The n_sample argument can be used to set the number of parameters for which
    samples are returned (by symmetry they all have the same distribution).
    This is useful for saving memory in high dimensions.
    """
    if n_sample is None:
        n_sample = n_dim
        assert n_sample <= n_dim, 'so far only set up for nsample <= ndim'
    thetas = np.random.normal(size=(r.shape[0], n_dim))
    # calculate normalisation so sum_i(theta_i^2) = r^2 for each row
    norm = r / np.sqrt(np.sum(thetas ** 2, axis=1))
    # only return n_sample columns
    thetas = thetas[:, :n_sample]
    # normalise each column
    thetas *= norm[:, None]
    return thetas


def sample_nsphere_shells(r, n_dim, n_sample):
    """
    Wrapper calling either sample_nsphere_shells_normal or
    sample_nsphere_shells_beta depending on the dimension and
    number of samples required.
    """
    if n_dim >= 100 and n_sample <= 2:
        return sample_nsphere_shells_beta(r, n_dim, n_sample)
    else:
        return sample_nsphere_shells_normal(r, n_dim, n_sample)


def nsphere_r_given_logx(logx, r_max, n_dim):
    """
    Finds r coordinates given input logx values for a uniform prior within an
    n-dimensional sphere co-centred with a spherically symmetric likelihood.
    This will return an answer of the same type (float or numpy array) as the
    input {logx}.
    """
    r = np.exp(logx / n_dim) * r_max
    return r


def nsphere_logx_given_r(r, r_max, n_dim):
    """
    Finds logx assuming the prior is an n-dimensional sphere co-centred with a
    spherically symmetric likelihood.
    This will return an answer of the same type (float or numpy array) as the
    input {r}.
    """
    logx = (np.log(r) - np.log(r_max)) * n_dim
    return logx


def nsphere_logvol(dim, radius=1.0):
    """
    Returns the natural log of the hypervolume of a unit nsphere of specified
    dimension. Useful for very high dimensions. Formulae is from
    https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
    """
    return ((np.log(radius) * dim) + (np.log(np.pi) * (dim / 2.0)) -
            (scipy.special.gammaln((dim / 2.0) + 1.0)))


def log_gaussian_given_r(r, sigma, n_dim):
    """
    Returns the natural log of a normalised, uncorrelated Gaussian likelihood
    with equal variance in all n_dim dimensions.
    """
    logl = -0.5 * ((r ** 2) / (sigma ** 2))
    # normalise
    logl -= n_dim * np.log(sigma)
    logl -= np.log(2 * np.pi) * (n_dim / 2.0)
    return logl


def log_exp_power_given_r(r, sigma, n_dim, b=0.5):
    """
    Returns the natural log of an exponential power distribution.
    This equals a Gaussian distribution when b=1.
    """
    logl = -0.5 * (((r ** 2) / (sigma ** 2)) ** b)
    # normalise
    logl += np.log(n_dim)
    logl += scipy.special.gammaln((n_dim) / 2.0)
    logl -= np.log(np.pi) * (n_dim / 2.0)
    logl -= n_dim * np.log(sigma)
    logl -= np.log(2) * (1 + 0.5 * b)
    logl -= scipy.special.gammaln(1. + ((n_dim) / (2 * b)))
    return logl


def r_given_log_exp_power(logl, sigma, n_dim, b=0.5):
    """
    Returns the natural log of an exponential power distribution.
    This equals a gaussian distribution when b=1.
    """
    # remove normalisation constant
    exponent = logl - np.log(n_dim)
    exponent -= scipy.special.gammaln((n_dim) / 2.0)
    exponent += np.log(np.pi) * (n_dim / 2.0)
    exponent += n_dim * np.log(sigma)
    exponent += np.log(2) * (1 + 0.5 * b)
    exponent += scipy.special.gammaln(1. + ((n_dim) / (2 * b)))
    # rearrange
    exponent = (-2. * exponent) ** (1. / b)
    logl = np.sqrt(exponent * (sigma ** 2))
    return logl


def r_given_log_gaussian(logl, sigma, n_dim):
    """
    Returns the radius of a given logl for a normalised,  uncorrelated Gaussian
    with equal variance in all n_dim dimensions.
    """
    # remove normalisation constant
    exponent = logl + n_dim * np.log(sigma)
    exponent += np.log(2 * np.pi) * (n_dim / 2.0)
    # rearrange
    r = np.sqrt(-2 * exponent) * sigma
    return r


def log_cauchy_given_r(r, sigma, n_dim):
    """
    Returns the natural log of a normalised,  uncorrelated Cauchy distribution
    with 1 degree of freedom.
    """
    logl = (-(1 + n_dim) / 2) * np.log(1 + ((r ** 2) / (sigma ** 2)))
    logl += scipy.special.gammaln((1.0 + n_dim) / 2.0)
    logl -= np.log(np.pi) * (n_dim + 1.0) / 2.0  # NB gamma(0.5) = sqrt(pi)
    logl += (-n_dim) * np.log(sigma)
    return logl


def r_given_log_cauchy(logl, sigma, n_dim):
    """
    Returns the radius for a given logl of a normalised,  uncorrelated
    Cauchy distribution with 1 degree of freedom.
    """
    exponent = logl
    # remove normalisation constant
    exponent -= scipy.special.gammaln((1.0 + n_dim) / 2.0)
    exponent += np.log(np.pi) * (n_dim + 1.0) / 2.0  # NB gamma(0.5) = sqrt(pi)
    exponent -= (-n_dim) * np.log(sigma)
    # remove power
    exponent /= -(n_dim + 1.0) / 2.0
    # rearrange
    exponent = np.exp(exponent) - 1.0
    r_squared = exponent * (sigma ** 2)
    return np.sqrt(r_squared)


def array_ratio_std(values_n, sigmas_n, values_d, sigmas_d):
    """
    Gives error on the ratio of 2 floats or 2 1dimensional arrays given their
    values and errors assuming the errors are uncorrelated.
    This assumes covariance = 0. _n and _d denote the numerator and
    denominator.
    """
    return (values_n / values_d) * (((sigmas_n / values_n) ** 2 +
                                     (sigmas_d / values_d) ** 2)) ** 0.5
