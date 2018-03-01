#!/usr/bin/env python
"""
Contains classes representing quantities which can be calculated from nested
sampling run.

Each estimator class should contain a mandatory member function returning the
value of the estimator for a nested sampling run:

    def __call__(self, ns_run, logw=None, simulate=False):
        ...

This allows logw to be provided if many estimators are being calculated from
the same run so logw is only calculated once. Otherwise logw is calculated from
the run if required.

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
import scipy
import perfectns.maths_functions as mf
import nestcheck.analyse_run as ar


# Estimators
# ----------


class LogZ(object):

    """Natural log of Bayesian evidence."""

    name = 'logz'
    latex_name = r'$\mathrm{log} \mathcal{Z}$'

    @staticmethod
    def __call__(ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        return scipy.special.logsumexp(logw)

    @staticmethod
    def analytical(settings):
        """Returns analytical value of estimator given settings."""
        return settings.logz_analytic()


class Z(object):

    """Bayesian evidence."""

    name = 'z'
    latex_name = r'$\mathcal{Z}$'

    @staticmethod
    def __call__(ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        return np.exp(scipy.special.logsumexp(logw))

    @staticmethod
    def analytical(settings):
        """Returns analytical value of estimator given settings."""
        return np.exp(settings.logz_analytic())


class CountSamples(object):

    """Number of samples in run."""

    name = 'n_samples'
    latex_name = r'\# samples'

    @staticmethod
    def __call__(ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        return ns_run['logl'].shape[0]


class RMean:

    """Mean of |theta| (the radial distance from the centre)."""

    min_value = 0

    def __init__(self, from_theta=False):
        self.name = 'r'
        self.latex_name = r'$\overline{|\theta|}$'
        self.from_theta = from_theta

    def __call__(self, ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        w_relative = np.exp(logw - logw.max())
        if self.from_theta:
            # If run contains a dims_to_sample setting, check that samples from
            # every dimension are included
            try:
                assert (ns_run['settings']['dims_to_sample'] ==
                        ns_run['settings']['n_dim']), "Cannot work out radius!"
            except KeyError:
                pass
            r = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
        else:
            r = ns_run['r']
        return np.sum(w_relative * r) / np.sum(w_relative)

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return check_by_integrating(self.ftilde, settings)

    @staticmethod
    def ftilde(logx, settings):
        """
        ftilde(X) is mean of f(theta) on the iso-likelihood contour
        L(theta) = L(X).
        """
        return settings.r_given_logx(logx)


class RCred(object):

    """One-tailed credible interval on the value of |theta|."""

    def __init__(self, probability, from_theta=False):
        assert 1 > probability > 0, 'credible interval probability = ' + \
            str(probability) + ' must be between 0 and 1'
        self.name = 'rc_' + str(probability)
        self.probability = probability
        self.from_theta = from_theta
        # format percent without trailing zeros
        percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
        self.latex_name = r'$\mathrm{C.I.}_{' + percent_str + r'\%}(|\theta|)$'

    def __call__(self, ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        # get sorted array of r values with their posterior weight
        wr = np.zeros((logw.shape[0], 2))
        wr[:, 0] = np.exp(logw - logw.max())
        if self.from_theta:
            # If run contains a dims_to_sample setting, check that samples from
            # every dimension are included
            try:
                assert (ns_run['settings']['dims_to_sample'] ==
                        ns_run['settings']['n_dim']), "Cannot work out radius!"
            except KeyError:
                pass
            wr[:, 1] = np.sqrt(np.sum(ns_run['theta'] ** 2, axis=1))
        else:
            wr[:, 1] = ns_run['r']
        wr = wr[np.argsort(wr[:, 1], axis=0)]
        # calculate cumulative distribution function (cdf)
        # Adjust by subtracting 0.5 * weight of first point to correct skew
        # - otherwise we need cdf=1 to return the last value but will return
        # the smallest value if cdf<the fractional weight of the first point.
        # This should not much matter as typically points' relative weights
        # will be very small compared to self.probability or
        # 1-self.probability.
        cdf = np.cumsum(wr[:, 0]) - (wr[0, 0] / 2)
        cdf /= np.sum(wr[:, 0])
        # calculate cdf
        # linearly interpolate value
        return np.interp(self.probability, cdf, wr[:, 1])


class ParamMean(object):

    """
    Mean of a single parameter (single component of theta).
    By symmetry all parameters have the same distribution.
    """

    def __init__(self, param_ind=1):
        self.param_ind = param_ind
        self.name = 'theta' + str(param_ind)
        self.latex_name = (r'$\overline{\theta_{\hat{' + str(param_ind) +
                           '}}}$')

    def __call__(self, ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        w_relative = np.exp(logw - logw.max())
        return ((np.sum(w_relative * ns_run['theta'][:, self.param_ind - 1])
                 / np.sum(w_relative)))

    @staticmethod
    def analytical(settings):
        """Returns analytical value of estimator given settings."""
        return 0.

    @staticmethod
    def ftilde(logx, settings):
        """
        ftilde(X) is mean of f(theta) on the iso-likelihood contour
        L(theta) = L(X).
        """
        return np.zeros(logx.shape)


class ParamCred(object):

    """
    One-tailed credible interval on the value of a single parameter (component
    of theta).
    By symmetry all parameters have the same distribution.
    """

    def __init__(self, probability, param_ind=1):
        assert 1 > probability > 0, 'credible interval probability = ' + \
            str(probability) + ' must be between 0 and 1'
        self.param_ind = param_ind
        self.name = 'theta' + str(param_ind) + 'c_' + str(probability)
        self.probability = probability
        param_str = r'\theta_{\hat{' + str(param_ind) + '}}'
        if probability == 0.5:
            self.name = 'Median(theta' + str(param_ind) + ')'
            self.latex_name = r'$\mathrm{median}(' + param_str + ')$'
        else:
            self.name = 'theta' + str(param_ind) + 'c_' + str(probability)
            # format percent without trailing zeros
            percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
            self.latex_name = (r'$\mathrm{C.I.}_{' + percent_str +
                               r'\%}(' + param_str + ')$')

    def __call__(self, ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        # get sorted array of parameter values with their posterior weight
        wp = np.zeros((logw.shape[0], 2))
        wp[:, 0] = np.exp(logw - logw.max())
        wp[:, 1] = ns_run['theta'][:, self.param_ind - 1]
        wp = wp[np.argsort(wp[:, 1], axis=0)]
        # calculate cumulative distribution function (cdf)
        # Adjust by subtracting 0.5 * weight of first point to correct skew
        # - otherwise we need cdf=1 to return the last value but will return
        # the smallest value if cdf<the fractional weight of the first point.
        # This should not much matter as typically points' relative weights
        # will be very small compared to self.probability or
        # 1-self.probability.
        cdf = np.cumsum(wp[:, 0]) - (wp[0, 0] / 2)
        cdf /= np.sum(wp[:, 0])
        # linearly interpolate value
        return np.interp(self.probability, cdf, wp[:, 1])

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        if self.probability == 0.5:
            # by symmetry the median of any parameter given spherically
            # symmetric likelihoods and priors co-centred on zero is zero
            return 0
        else:
            assert type(settings.likelihood).__name__ == 'Gaussian', \
                "so far only set up for Gaussian likelihoods"
            assert type(settings.prior).__name__ in ['Gaussian',
                                                     'GaussianCached'], \
                "so far only set up for Gaussian priors"
            # the product of two Gaussians is another Gaussian with sigma:
            sigma = ((settings.likelihood.likelihood_scale ** -2) +
                     (settings.prior.prior_scale ** -2)) ** -0.5
            # find numCer of sigma from the mean by inverting the CDF of the
            # normal distribution.
            # CDF(x) = (1/2) + (1/2) * error_function(x / sqrt(2))
            zscore = (scipy.special.erfinv((self.probability * 2) - 1)
                      * np.sqrt(2))
            return zscore * sigma


class ParamSquaredMean:

    """
    Mean of the square of single parameter (second moment of its posterior
    distribution).
    By symmetry all parameters have the same distribution.
    """

    min_value = 0

    def __init__(self, param_ind=1):
        self.param_ind = param_ind
        self.name = 'theta' + str(param_ind) + 'squ'
        self.latex_name = (r'$\overline{\theta^2_{\hat{' + str(param_ind) +
                           '}}}$')

    def __call__(self, ns_run, logw=None, simulate=False):
        """Returns estimator value for run."""
        if logw is None:
            logw = ar.get_logw(ns_run, simulate=simulate)
        w_relative = np.exp(logw - logw.max())  # protect against overflow
        w_relative /= np.sum(w_relative)
        return np.sum(w_relative *
                      (ns_run['theta'][:, self.param_ind - 1] ** 2))

    @staticmethod
    def ftilde(logx, settings):
        """
        ftilde(X) is mean of f(theta) on the iso-likelihood contour
        L(theta) = L(X).
        """
        # by symmetry at each (hyper)spherical iso-likelihood contour:
        r = settings.r_given_logx(logx)
        return r ** 2 / settings.n_dim

    def analytical(self, settings):
        """Returns analytical value of estimator given settings."""
        return check_by_integrating(self.ftilde, settings)


# Functions for checking estimator results
# ----------------------------------------


def get_true_estimator_values(estimators, settings):
    """
    Calculates analytic values for estimators given the likelihood and
    prior in settings. If there is no method for calculating the values
    then np.nan is returned.

    Parameters
    ----------
    estimators: estimator object or list of estimator objects
    settings: PerfectNSSettings object

    Returns:
    --------
    output: np.array of size (len(estimators),) if estimators is a list, float
        otherwise.
    """
    if isinstance(estimators, list):
        output = np.zeros(len(estimators))
        for i, est in enumerate(estimators):
            try:
                output[i] = est.analytical(settings)
            except (AttributeError, AssertionError):
                output[i] = np.nan
        return output
    else:
        try:
            return estimators.analytical(settings)
        except (AttributeError, AssertionError):
            return np.nan


def check_by_integrating(ftilde, settings):
    """
    Return the analytical value of the estimator using numerical
    integration.

    Chopin and Robert (2010) show that the expectation of some function
    f(theta) is given by the integral

        int L(X) X ftilde(X) dX / Z,

    where ftilde(X) is mean of f(theta) on the iso-likelihood contour
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
    estimator values by numerical integration.
    Note that the integral must be normalised by multiplying by a factor (1/Z).
    """
    # returns L(X) X ftilde(X) for integrating dlogx
    return (np.exp(settings.logl_given_logx(logx) + logx)
            * ftilde(logx, settings))
