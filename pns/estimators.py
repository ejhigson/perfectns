#!/usr/bin/python
"""
Contains classes representing estimators f(theta) for use on nested sampling
output and analytical analysis.

Each estimator class should contain a mandatory member function returning the
value of the estimator for a nested sampling run:

    def estimator(self, logw, ns_run):
        ...

and may also contain a function giving its analytical value for some given set
of calculation settings (for use in checking results):

    def analytical(self, settings):
        ...

Estimators should also contain class variables:

    name: str
        used for results tables.
    latex_name: str
        used for plotting results diagrams.

"""

import numpy as np
import scipy
import scipy.misc  # for scipy.misc.logsumexp
# perfect nested sampling modules
import pns.maths_functions as mf


# Estimators
# ----------
# A class is used to hold the functions required for each estimator f(theta)


class logzEstimator(object):

    """Log of Bayesian evidence."""

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


class n_samplesEstimator(object):

    """Numer of samples in run."""

    name = 'n_samples'
    latex_name = '\# samples'

    def estimator(self, logw, ns_run):
        """Returns estimator value for run."""
        return logw.shape[0]


class rEstimator:

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


class rconfEstimator(object):

    def __init__(self, fraction):
        assert fraction < 1.0 and fraction > 0, ("conf interval fraction = " +
                                                 str(fraction) +
                                                 " must be between 0 and 1")
        self.name = 'rc_' + str(fraction)
        self.fraction = fraction
        # format percent without trailing zeros
        percent_str = ('%f' % (fraction * 100)).rstrip('0').rstrip('.')
        self.latex_name = '$\mathrm{C.I.}_{' + percent_str + '\%}(|\\theta|)$'

    def min(self, settings):
        return 0

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
        return np.interp(self.fraction, cdf, wr[:, 1])


class theta1Estimator(object):

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


class theta1confEstimator(object):

    def __init__(self, fraction, param_ind=1):
        assert 1.0 > fraction > 0, \
            "conf interval fraction = " + str(fraction) + " not <1 and >0"
        self.param_ind = param_ind
        self.name = 't' + str(param_ind) + 'c_' + str(fraction)
        self.fraction = fraction
        param_str = '\\theta_{\hat{' + str(param_ind) + '}}'
        if fraction == 0.5:
            self.latex_name = '$\mathrm{median}(' + param_str + ')$'
        else:
            # format percent without trailing zeros
            percent_str = ('%f' % (fraction * 100)).rstrip('0').rstrip('.')
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
        return np.interp(self.fraction, cdf, wp[:, 1])


class theta1squaredEstimator:

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


def check_estimator_values(funcs_list, settings):
    """
    Return an array of the analytical values of the estimators in
    funcs_list for the provided settings. If the analytical values
    are not available they are set to np.nan.
    """
    output = np.zeros(len(funcs_list))
    for i, f in enumerate(funcs_list):
        try:
            output[i] = f.analytical(settings)
        except (AttributeError, AssertionError):
            output[i] = np.nan
    return output


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
        "logx_terminate function not set up for current settings"
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
