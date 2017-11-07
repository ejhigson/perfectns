#!/usr/bin/python
"""
Contains classes representing quantities which can be calculated from nested
sampling run.

Each estimator class should contain a mandatory member function returning the
value of the estimator for a nested sampling run:

    def estimator(self, logw, ns_run):
        ...

They may also optionally contain a function giving its analytical value for
some given set of calculation settings (for use in checking results):

    def analytical(self, settings):
        ...

as well as helper functions.
Estimators should also contain class variables:

    name: str
        used for results tables.
    latex_name: str
        used for plotting results diagrams.

"""

import numpy as np
import pandas as pd
import scipy
import scipy.misc  # for scipy.misc.logsumexp
import pns.maths_functions as mf


# Estimators
# ----------


class logzEstimator(object):

    """Natural log of Bayesian evidence."""

    name = 'logz'
    latex_name = r'$\mathrm{log} \mathcal{Z}$'

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        return scipy.misc.logsumexp(logw)

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return settings.logz_analytic()


class zEstimator(object):

    """Bayesian evidence."""

    name = 'z'
    latex_name = '$\mathcal{Z}$'

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        return np.exp(scipy.misc.logsumexp(logw))

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return np.exp(settings.logz_analytic())


class nSamplesEstimator(object):

    """Numer of samples in run."""

    name = 'n_samples'
    latex_name = '\# samples'

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        return logw.shape[0]


class rMeanEstimator:

    """Mean of |theta| (the radial distance from the centre)."""

    name = 'r'
    latex_name = '$|\\theta|$'

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        w_relative = np.exp(logw - logw.max())
        return (np.sum(w_relative * ns_run['r']) / np.sum(w_relative))

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return 0

    def min(self, settings):
        return 0

    def ftilde(self, logx, settings):
        return settings.r_given_logx(logx)


class rCredEstimator(object):

    """One-tailed credible interval on the value of |theta|."""

    def __init__(self, probability):
        assert 1 > probability > 0, 'credible interval probability = ' + \
            str(probability) + ' must be between 0 and 1'
        self.name = 'rc_' + str(probability)
        self.probability = probability
        # format percent without trailing zeros
        percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
        self.latex_name = '$\mathrm{C.I.}_{' + percent_str + '\%}(|\\theta|)$'

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        # get sorted array of p1 values with their posterior weight
        wr = np.zeros((logw.shape[0], 2))
        wr[:, 0] = np.exp(logw - logw.max())
        wr[:, 1] = ns_run['r']
        wr = wr[np.argsort(wr[:, 1], axis=0)]
        # calculate cdf
        cdf = np.zeros(wr.shape[0])
        cdf[0] = wr[0, 0] * 0.5
        for i, _ in enumerate(wr[1:, 0]):
            cdf[i + 1] = cdf[i] + 0.5 * (wr[i, 0] + wr[i + 1, 0])
        cdf = cdf / np.sum(wr[:, 0])
        # linearly interpolate value
        return np.interp(self.probability, cdf, wr[:, 1])


class paramMeanEstimator(object):

    """
    Mean of a single parameter (single component of theta).
    By symmetry all parameters have the same distribution.
    """

    def __init__(self, param_ind=1):
        self.param_ind = param_ind
        self.name = 't' + str(param_ind)
        self.latex_name = ('$\\overline{\\theta_{\hat{' + str(param_ind) +
                           '}}}$')

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        w_relative = np.exp(logw - logw.max())
        return ((np.sum(w_relative * ns_run['theta'][:, self.param_ind - 1])
                / np.sum(w_relative)))

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return 0.

    def ftilde(self, logx, settings):
        return np.zeros(logx.shape)


class paramCredEstimator(object):

    """
    One-tailed credible interval on the value of a single parameter (component
    of theta).
    By symmetry all parameters have the same distribution.
    """

    def __init__(self, probability, param_ind=1):
        assert 1 > probability > 0, 'credible interval probability = ' + \
            str(probability) + ' must be between 0 and 1'
        self.param_ind = param_ind
        self.name = 't' + str(param_ind) + 'c_' + str(probability)
        self.probability = probability
        param_str = '\\theta_{\hat{' + str(param_ind) + '}}'
        if probability == 0.5:
            self.latex_name = '$\mathrm{median}(' + param_str + ')$'
        else:
            # format percent without trailing zeros
            percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
            self.latex_name = ('$\mathrm{C.I.}_{' + percent_str +
                               '\%}(' + param_str + ')$')

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        wp = np.zeros((logw.shape[0], 2))
        wp[:, 0] = np.exp(logw - logw.max())
        wp[:, 1] = ns_run['theta'][:, self.param_ind - 1]
        wp = wp[np.argsort(wp[:, 1], axis=0)]
        # calculate cdf
        cdf = np.zeros(wp.shape[0])
        cdf[0] = wp[0, 0] * 0.5
        for i, _ in enumerate(wp[1:, 0]):
            cdf[i + 1] = cdf[i] + 0.5 * (wp[i, 0] + wp[i + 1, 0])
        cdf = cdf / np.sum(wp[:, 0])
        # linearly interpolate value
        return np.interp(self.probability, cdf, wp[:, 1])

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        if self.probability == 0.5:
            # by symmetry the median of any parameter given spherically
            # symmetric likelihoods and priors cocentred on zero is zero
            return 0
        else:
            assert type(settings.likelihood).__name__ == 'gaussian', \
                "so far only set up for Gaussian likelihoods"
            assert type(settings.prior).__name__ in ['gaussian',
                                                     'gaussian_cached'], \
                "so far only set up for Gaussian priors"
            # the product of two gaussians is another gaussian with sigma:
            sigma = ((settings.likelihood.likelihood_scale ** -2) +
                     (settings.prior.prior_scale ** -2)) ** -0.5
            # find number of sigma from the mean by inverting the CDF of the
            # normal distribution.
            # cdf(x) = (1/2) + (1/2) * error_function(x / sqrt(2))
            z = scipy.special.erfinv((self.probability * 2) - 1) * np.sqrt(2)
            return z * sigma


class paramSquaredMeanEstimator:

    """
    Mean of the square of single parameter (second moment of its posterior
    distribution).
    By symmetry all parameters have the same distribution.
    """

    def __init__(self, param_ind=1):
        self.param_ind = param_ind
        self.name = 't' + str(param_ind) + 'squ'
        self.latex_name = ('$\\overline{\\theta^2_{\hat{' + str(param_ind) +
                           '}}}$')

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        w_relative = np.exp(logw - logw.max())  # protect against overflow
        w_relative /= np.sum(w_relative)
        return np.sum(w_relative *
                      (ns_run['theta'][:, self.param_ind - 1] ** 2))

    def ftilde(self, logx, settings):
        """
        ftilde(X) is mean of f(theta) on the isolikelihood contour
        L(theta) = L(X).
        """
        # by symmetry at each (hyper)spherical isolikelihood contour:
        r = settings.r_given_logx(logx)
        return r ** 2 / settings.n_dim

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return check_by_integrating(self.ftilde, settings)


# Functions for checking estimator results
# ----------------------------------------


def get_true_estimator_values(estimator_list, settings):
    """
    Return a pandas data frame of the correct values for the estimators in
    estimator_list given the likelihood and prior in settings. If there is no
    method for calculating the values set up yet they are set to np.nan.
    """
    output = {}
    for est in estimator_list:
        try:
            output[est.name] = est.analytical(settings)
        except (AttributeError, AssertionError):
            output[est.name] = np.nan
    return pd.DataFrame(output, index=['true values'])


def check_by_integrating(ftilde, settings):
    """
    Return the analytical value of the estimator using numerical
    integration.

    Chopin and Robert (2010) show that the expectation of some function
    f(theta) is given by the integral

        int L(X) X ftilde(X) dX / Z,

    where ftilde(X) is mean of f(theta) on the isolikelihood contour
    L(theta) = L(X).
    """
    logx_terminate = mf.analytic_logx_terminate(settings)
    assert logx_terminate is not None, \
        'logx_terminate function not set up for current settings'
    result = scipy.integrate.quad(check_integrand, logx_terminate,
                                  0.0, args=(ftilde, settings))
    return result[0] / np.exp(settings.logz_analytic())


def check_integrand(logx, ftilde, settings):
    """
    Helper function to return integrand L(X) X ftilde(X) for checking
    estimator values by numerical intergration.
    """
    # returns L(X) X ftilde(X) for integrating dlogx
    # NB this must be normalised by a factor V / Z
    return (np.exp(settings.logl_given_logx(logx) + logx)
            * ftilde(logx, settings))
