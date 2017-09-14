#!/usr/bin/python
"""Contains useful helper functions shared by many other modules."""


import numpy as np
import scipy
import scipy.stats
import scipy.special
import mpmath
from pynverse import inversefunc
# perfect nested sampling modules
import pns.utils as utils


# Maths functions

def pynverse_gaussian_r_given_logx(logx, sigma, n_dim):
    return inversefunc(gaussian_logx_given_r, args=(sigma, n_dim), y_values=logx)


def scipy_gaussian_r_given_logx(logx, sigma, n_dim):
    exponent = scipy.special.gammaincinv(n_dim / 2., np.exp(logx))
    return np.sqrt(2 * exponent * sigma ** 2)


def get_interp_r_given_logx_dict_fixed(n_dim, prior_scale, logx_scipy_fail):
    version = 'v03'
    logx_min = -4500
    logx_max = logx_scipy_fail + 1
    pointsfactor = 20
    save_name = version + '_interpfixed_' + str(n_dim) + 'd_' + str(prior_scale) + 'rmax'
    try:
        interp_dict = utils.pickle_load(save_name, print_time=False)
        # print("successfully loaded")
        return interp_dict
    except OSError:  # "FileNotFoundError is a subclass of OSError. Must use OSError for except.
        print("Interp file not found - try generating new data")
        r_max = scipy_gaussian_r_given_logx(logx_max, prior_scale, n_dim)
        r_min = pynverse_gaussian_r_given_logx(logx_min, prior_scale, n_dim)
        # guard against any errors in interpolation
        while mpmath_gaussian_logx_given_r(r_min, prior_scale, n_dim) > logx_min:
            print("increasing interp range", r_min, mpmath_gaussian_logx_given_r(r_min, prior_scale, n_dim))
            r_min /= 2.0
        r_array = np.linspace(r_min, r_max, int(2 + logx_max - logx_min) * pointsfactor)
        logx_array = gaussian_logx_given_r(r_array, prior_scale, n_dim)
        interp_dict = {'version': version,
                       'pointsfactor': pointsfactor,
                       'logx_min': logx_min,
                       'logx_max': logx_max,
                       'r_array': r_array,
                       'logx_array': logx_array}
        utils.pickle_save(interp_dict, save_name)
        return interp_dict


def get_interp_r_given_logx_dict(n_dim, prior_scale, logl_max, logz_analytic, termination_fraction, pointsfactor, extend_factor, likelihood_name):
    version = 'v01'
    save_name = version + '_interp_' + likelihood_name + '_' + str(n_dim) + 'd_' + str(prior_scale) + 'rmax_' + str(termination_fraction) + 'term_' + str(pointsfactor) + 'points_' + str(extend_factor) + 'ef'
    # print("get_interp_r_given_logx_dict: " + save_name)
    # try:
    #     interp_dict = utils.pickle_load(save_name)
    #     print("successfully loaded")
    #     return interp_dict
    # except OSError:  # "FileNotFoundError is a subclass of OSError. Must use OSError for except.
    #     print("File not found - try generating new data")
    logx_min = -3000.
    # logx_min = extend_factor * logx_terminate_bound(logl_max, termination_fraction, logz_analytic)
    logx_scipy_fail = get_logx_scipy_fail(n_dim, prior_scale)
    r_min = pynverse_gaussian_r_given_logx(logx_min - 1, prior_scale, n_dim)
    # guard against any errors in interpolation
    while mpmath_gaussian_logx_given_r(r_min, prior_scale, n_dim) > logx_min:
        print("increasing interp range", r_min, mpmath_gaussian_logx_given_r(r_min, prior_scale, n_dim))
        r_min /= 5.0
    r_max = gaussian_r_given_logx(logx_scipy_fail + 1, prior_scale, n_dim, logx_scipy_fail)  # make logx_max a bit bigger to make sure it falls before logx_scipy_fail
    r_array = np.linspace(r_min, r_max, int(2 + logx_scipy_fail - logx_min) * pointsfactor)
    logx_array = gaussian_logx_given_r(r_array, prior_scale, n_dim)
    interp_dict = {'version': version,
                   'pointsfactor': pointsfactor,
                   'termination_fraction': termination_fraction,
                   'extend_factor': extend_factor,
                   'logx_min': logx_min,
                   'logx_scipy_fail': logx_scipy_fail,
                   'r_array': r_array,
                   'logx_array': logx_array}
    utils.pickle_save(interp_dict, save_name, path="data/", print_time=False)
    return interp_dict


# def gaussian_r_given_logx(logx, sigma, n_dim):
#     test = abs(test_scipy(logx, n_sim, sigma=sigma) / logx)
#     r = np.zeros(logx.shape)
#     tollerance = 10 ** -4
#     ind_pass = np.where(test < 1. - tollerance and test

# def gaussian_r_given_logx(logx, sigma, n_dim):
#     test = abs(test_scipy(logx, n_sim, sigma=sigma) / logx)


def gaussian_r_given_logx(logx, shrinkage_ratio, n_dim, logx_scipy_fail):
    if isinstance(logx, float) or isinstance(logx, int):
        if logx < logx_scipy_fail:
            r = pynverse_gaussian_r_given_logx(logx, shrinkage_ratio, n_dim)
            # r = np.interp(logx, logx_interp, r_interp)
        else:
            r = scipy_gaussian_r_given_logx(logx, shrinkage_ratio, n_dim)
    else:
        r = np.zeros(logx.shape)
        for i, logx_i in enumerate(logx):
            if logx_i < logx_scipy_fail:
                r[i] = pynverse_gaussian_r_given_logx(logx[i], shrinkage_ratio, n_dim)
                # r[i] = np.interp(logx[i], logx_interp, r_interp)
            else:
                r[i] = scipy_gaussian_r_given_logx(logx[i], shrinkage_ratio, n_dim)
    return r


def scipy_gaussian_logx_given_r(r, sigma, n_dim):
    exponent = 0.5 * (r / sigma) ** 2
    return np.log(scipy.special.gammainc(n_dim / 2., exponent))  # - scipy.special.gammaln(n_dim / 2.)


def mpmath_gaussian_logx_given_r(r, sigma, n_dim):
    exponent = 0.5 * (r / sigma) ** 2
    return float(mpmath.log(mpmath.gammainc(n_dim / 2., a=0, b=exponent, regularized=True)))


def gaussian_logx_given_r(r, sigma, n_dim):
    if isinstance(r, np.ndarray):  # needed to ensure output is numpy array
        logx = np.zeros(r.shape)
        for i, r_i in enumerate(r):
            logx[i] = mpmath_gaussian_logx_given_r(r_i, sigma, n_dim)
        return logx
    else:
        return mpmath_gaussian_logx_given_r(r, sigma, n_dim)


def test_pynverse(logx, n_dim, sigma=10):
    r = pynverse_gaussian_r_given_logx(logx, sigma, n_dim)
    return gaussian_logx_given_r(r, sigma, n_dim)


def test_scipy(logx, n_dim, sigma=10):
    r = scipy_gaussian_r_given_logx(logx, sigma, n_dim)
    return gaussian_logx_given_r(r, sigma, n_dim)


def get_logx_scipy_fail(n_dim, sigma):
    toll = 10 ** -8
    logx = -1.0
    test = test_scipy(logx, n_dim, sigma=sigma) / logx
    while 1. - toll < test < 1. + toll:
        logx -= 0.1
        test = test_scipy(logx, n_dim, sigma=sigma) / logx
    return logx + 25  # add a healthy margin for error


def analytic_logx_terminate(settings):
    """Find logx_terminate analytically by assuming all likelihood at very low X approximately equals the maximum likelihood.
    This approximation breaks down in very high dimensions."""
    logl_max = settings.likelihood_prior.logl_given_r(0)  # use r=0 rather than logx = -np.inf as the latter causes numerical problems
    return logx_terminate_bound(logl_max, settings.zv_termination_fraction, settings.likelihood_prior.logz_analytic)


def logx_terminate_bound(logl_max, zv_termination_fraction, logz_analytic):
    """
    Find a lower bound logx_terminate analytically by assuming all likelihood at very low X approximately equals the maximum likelihood.
    This approximation breaks down in very high dimensions and the true logx terminate required will be larger.
    We want Z_term = zv_termination_fraction * Z_analytic = int_0^Xterm L(X) dX approx= Xterm L_max
    so logx_term = log(zv_termination_fraction) + log(Z_analytic) - logl_max
    """
    return np.log(zv_termination_fraction) + logz_analytic - logl_max


def sample_nsphere_shells_beta(r, n_dim, n_sample, mu=0):
    # sample single parameters on n_dim-dimensional sphere independently
    # as described in section 4 of my errors in nested sampling paper
    # NB this will not look correct when plotted in many dimensions as the radius of n points independently sampled on the radius R hypersphere do not nessesarily have a radius r!
    thetas = np.sqrt(np.random.beta(0.5, (n_dim - 1) * 0.5, size=(r.shape[0], n_sample)))
    thetas *= (-1) ** (np.random.randint(0, 2, size=thetas.shape))  # randomly select + or -
    for i in range(n_sample):
        thetas[:, i] *= r
        if isinstance(mu, float) or isinstance(mu, int):
            thetas[:, i] += mu
        else:
            thetas[:, i] += mu[i]
    return thetas


def sample_nsphere_shells(r, n_dim, n_sample, mu=0):
    assert n_sample <= n_dim, "so far only set up for nsample <= ndim"
    # sample single parameters on n_dim-dimensional sphere independently
    # as described in section 4 of my errors in nested sampling paper
    thetas = np.random.normal(size=(r.shape[0], n_dim))
    for i in range(r.shape[0]):
        thetas[i, :] /= np.sqrt(np.sum(thetas[i, :] ** 2))
    thetas = thetas[:, :n_sample]
    for i in range(n_sample):
        thetas[:, i] *= r
        if isinstance(mu, float) or isinstance(mu, int):
            thetas[:, i] += mu
        else:
            thetas[:, i] += mu[i]
    return thetas


def nsphere_r_given_logx(logx, r_max, n_dim):
    """
    Finds r assuming the prior is an n-dimensional sphere co-centred with a spherically symetric likelihood.
    This will return an answer of the same type (float or numpy array) as the input {logx}.
    """
    r = np.exp(logx / n_dim) * r_max
    return r


def nsphere_logx_given_r(r, r_max, n_dim):
    """
    Finds logx assuming the prior is an n-dimensional sphere co-centred with a spherically symetric likelihood.
    This will return an answer of the same type (float or numpy array) as the input {r}.
    """
    logx = (np.log(r) - np.log(r_max)) * n_dim
    return logx


def nsphere_vol(dim, radius=1.0):
    """Returns hypervolume of a unit nsphere of specified dimension."""
    # From https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
    return ((np.pi ** (dim / 2.0)) / scipy.special.gamma((dim / 2.0) + 1.0)) * (radius ** dim)


def nsphere_logvol(dim, radius=1.0):
    """Returns the natural log of the hypervolume of a unit nsphere of specified dimension. Useful for very high dimensions."""
    # From https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
    return (np.log(radius) * dim) + (np.log(np.pi) * (dim / 2.0)) - (scipy.special.gammaln((dim / 2.0) + 1.0))


def log_gaussian_given_r(r, sigma, n_dim):
    """Returns the natural log of a normalised,  uncorrelated gaussian with equal variance in all n_dim dimensions."""
    logl = -0.5 * ((r ** 2) / (sigma ** 2))
    # normalise
    logl -= n_dim * np.log(sigma)
    logl -= np.log(2 * np.pi) * (n_dim / 2.0)
    return logl


def log_laplace_given_r(r, sigma, n_dim):
    """Returns the natural log of a normalised,  uncorrelated laplacian with equal variance in all n_dim dimensions."""
    logl = np.log(scipy.special.k0(np.sqrt(2) * r / sigma))
    # normalise
    logl -= n_dim * np.log(sigma)
    logl -= np.log(np.pi) * (n_dim / 2.)
    logl -= scipy.special.gammaln(n_dim / 2.)
    logl -= np.log(2) * ((n_dim / 2.) - 1)
    return logl


def log_bessel_given_r(r, sigma_in, n_dim, q):
    """Returns the natural log of a normalised,  uncorrelated bessel with equal variance in all n_dim dimensions.
    Equals Laplace for q=0 and sigma_in = 1 / sqrt(2)
    """
    sigma = sigma_in
    logl = np.log(scipy.special.kn(q, r / sigma))
    logl += q * np.log(r)
    # normalise
    logl -= (n_dim + q) * np.log(sigma)
    logl -= np.log(np.pi) * (n_dim / 2.)
    logl -= scipy.special.gammaln(q + (q + (n_dim / 2.)))
    logl -= np.log(2) * (q + (n_dim / 2.) - 1)
    return logl


def log_logistic_given_r(r, sigma, n_dim):
    """Returns the natural log of a logistic function. NB not normalised."""
    logl = -0.5 * ((r ** 2) / (sigma ** 2))
    if isinstance(r, np.ndarray):
        for i, r_i in enumerate(r):
            logl[i] -= 2 * log_sum_given_logs(np.asarray([0.0, -0.5 * ((r_i ** 2) / (sigma ** 2))]))
    else:
        logl -= 2 * log_sum_given_logs(np.asarray([0.0, -0.5 * ((r ** 2) / (sigma ** 2))]))

    # normalise
    logl -= n_dim * np.log(sigma)
    logl -= np.log(2 * np.pi) * (n_dim / 2.0)
    return logl


def r_given_log_logistic(logl, sigma, n_dim):
    """Returns the radius of a given logl for a logistic function. NB not normalised."""
    # remove normalisation constant
    c = logl + n_dim * np.log(sigma)
    c += np.log(2 * np.pi) * (n_dim / 2.0)
    # we now have C = x / (1 + x ** 2), where x = exp(exponent)
    # only real solutions when 1 - 2C > 0
    # in this case x = 0 or x = (1 - 2c) / c
    # log(x) = log(1 - 2c) - log(c)
    exponent = log_subtract(0, np.log(2 * c)) - np.log(c)
    r = np.sqrt(-2 * exponent) * sigma
    return r


def log_exp_power_given_r(r, sigma, n_dim, b=0.5):
    """Returns the natural log of an exponential power distribution.
    This equals a gaussian distribution when b=1.
    See http://aurelie.boisbunon.free.fr/downloads/loisSS.pdf for more details
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
    """Returns the natural log of an exponential power distribution.
    This equals a gaussian distribution when b=1.
    See http://aurelie.boisbunon.free.fr/downloads/loisSS.pdf for more details
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
    """Returns the radius of a given logl for a normalised,  uncorrelated gaussian with equal variance in all n_dim dimensions."""
    # remove normalisation constant
    exponent = logl + n_dim * np.log(sigma)
    exponent += np.log(2 * np.pi) * (n_dim / 2.0)
    # rearrange
    r = np.sqrt(-2 * exponent) * sigma
    return r


def log_cauchy_given_r(r, sigma, n_dim):
    """Returns the natural log of a normalised,  uncorrelated Cauchy distribution with 1 degree of freedom."""
    # see https://en.wikipedia.org/wiki/Cauchy_distribution#Multivariate_Cauchy_distribution
    # NB gamma(0.5) = sqrt(pi)
    logl = (-(1 + n_dim) / 2) * np.log(1 + ((r ** 2) / (sigma ** 2)))
    logl += scipy.special.gammaln((1.0 + n_dim) / 2.0)
    logl -= np.log(np.pi) * (n_dim + 1.0) / 2.0  # NB gamma(0.5) = sqrt(pi)
    logl += (-n_dim) * np.log(sigma)
    return logl


def r_given_log_cauchy(logl, sigma, n_dim):
    """Returns the radius for a given logl of a normalised,  uncorrelated Cauchy distribution with 1 degree of freedom."""
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


def entropy_num_samples(w):
    """
    Return the entropy of a set of weighted samples from their weights.
    This takes the absolute value of weights first.
    If np.sum(w) is not 1 the function rerenormalises it to 1 for the calculation.
    """
    if w.shape[0] == 0:
        return 0
    else:
        # assert w.min() >= 0.0, "Weights must be non-negative!"
        w = np.absolute(w)
        w = w / np.sum(w)
        w = w[np.where(w > 0)]
        return np.exp((-1.0) * np.sum(w * np.log(w)))


def log_sum_given_logs(logs):
    """
    Returns the log of the sum of values given a list, touple or np array of values.
    See https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/ for more details.
    """
    if isinstance(logs, tuple) or isinstance(logs, list):
        logs = np.asarray(logs)
    logsum = logs.max()
    logsum += np.log(np.sum(np.exp(logs - logs.max())))
    return logsum


def log_subtract(loga, logb):
    """
    Returns log(a-b) given loga and logb.
    See https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/ for more details.
    """
    assert loga >= logb, "log_subtract: a-b is negative for loga=" + str(loga) + " and logb=" + str(logb)
    return loga + np.log(1 - np.exp(logb - loga))


# helper functions using settings

def r_given_lp(lp, settings):
    """
    Returns r(paramaters).
    Assume lp is 2d.
    """
    assert lp.ndim == 2, "set up only for 2d lp array"
    return settings.likelihood_prior.r_given_theta(lp[:, 1:])


# Stats functions:


def stats_rows(array, reduce_std=1.0):
    """Get the mean, std, skew and kurtosis of a 1d array or of each row of a 2d array"""
    # Reduce std argument reduces the standard deviation by a factor of
    # 1/sqrt(reduce_std) - this is used for estimating stds on the mean.
    assert array.ndim == 1 or array.ndim == 2,  "Input must be a 1 or 2d array"
    if array.ndim == 1:
        stats_output = np.zeros(4)
        stats_output[0] = np.mean(array)
        stats_output[1] = np.std(array, ddof=1)
        stats_output[2] = scipy.stats.skew(array, bias=False)
        stats_output[3] = scipy.stats.kurtosis(array)
        # reduce std for use in splitting
        stats_output[1] = stats_output[1] / np.sqrt(float(reduce_std))
    elif array.ndim == 2:
        stats_output = np.zeros((array.shape[0], 4))
        for i in range(0, array.shape[0]):
            stats_output[i, 0] = np.mean(array[i, :])
            stats_output[i, 1] = np.std(array[i, :], ddof=1)
            stats_output[i, 2] = scipy.stats.skew(array[i, :], bias=False)
            stats_output[i, 3] = scipy.stats.kurtosis(array[i, :])
        # reduce std for use in splitting
        stats_output[:, 1] = stats_output[:, 1] / np.sqrt(float(reduce_std))
    return stats_output


def sigma_given_cdf(cdf):
    """Returns sigma from median given an array of cdf values \in [0,1]"""
    # commented out 23/3/17 # assert -0.0001 <= cdf and cdf <= 1.0001, "cdf=" + str(cdf) + " must be between 1 and 0"
    # commented out 23/3/17 # if cdf >= 1.0 or cdf <= 0.0:
    # commented out 23/3/17 #     return -100.0
    # commented out 23/3/17 # else:
    # The error function is defined in [-1,+1] mapping to [-inf,+inf].
    # We want to map a CDF to the number of sigma from the median - i.e. from [0,+1] to [0,+inf].
    # Hence we need the argument (2*cdf - 1), and to take abs to get a positive answer.
    sigma_temp = abs(scipy.special.erfinv((cdf * 2) - 1))
    # erf is definted in terms of
    # int e^(-t^2)
    # which corresponds to a gaussian with sigma = 1/sqrt(2). So we must multiply sigma_temp by sqrt(2)
    return np.sqrt(2) * sigma_temp


def kde_cdf_given_values(x, support):
    """
    Uses kernel density estimation to output an array of cdf values given some data points x and a support.

    Positional arguments:
    x -- array of data point values.
    support -- array of values on which to calculate the cdf.
    """
    bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
    kernels = []
    for x_i in x:
        kernel = scipy.stats.norm(x_i, bandwidth).pdf(support)
        kernels.append(kernel)
    density = np.sum(kernels, axis=0)
    density /= scipy.integrate.trapz(density, support)
    cdf = np.zeros(density.shape[0])
    cdf[0] = density[0]
    for i in range(1, density.shape[0]):
        cdf[i] = cdf[i - 1] + density[i]
    cdf = cdf / np.sum(density)
    # adjust for rounding errors
    assert cdf.max() <= 1.0001, "cdf must be approx <= 1.0. cdf=" + str(cdf) + ", density=" + str(density)
    cdf[np.where(cdf > 1.0)] = 1.0
    return cdf


def n_std_difference(values_1, sigmas_1, values_2, sigmas_2):
    """
    Gives the number of standard deviations difference between two floats or 1d arrays.
    Assumes the errors are uncorrelated.
    """
    sigmas = np.sqrt(sigmas_1 ** 2 + sigmas_2 ** 2)
    return (values_1 - values_2) / sigmas


def array_ratio_std(values_n, sigmas_n, values_d, sigmas_d):
    """
    Gives error on the ratio of 2 floats or 2 1dimensional arrays given their values and errors assuming the errors are uncorrelated.
    The sqare root of the variance formulae here is used http://stats.stackexchange.com/questions/19576/variance-of-the-reciprocal-ii/19580#19580
    This assumes covariance = 0. _n and _d denote the numerator and denominator.
    """
    return (values_n / values_d) * np.sqrt((sigmas_n / values_n) ** 2 + (sigmas_d / values_d) ** 2)
