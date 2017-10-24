#!/usr/bin/python
"""
Contains classes representing estimators f(theta) for use on nested sampling
output and analytical analysis.
"""

import numpy as np
import scipy
import scipy.misc  # for scipy.misc.logsumexp
# perfect nested sampling modules
import pns.maths_functions as mf


# Checking estimator results
# --------------------------


def check_integrand(logx, func, settings):
    """
    Helper function to return integrand L(X) X ftilde(X) for checking
    estimator values by numerical intergration.
    """
    # returns L(X) X ftilde(X) for integrating dlogx
    # NB this must be normalised by a factor V / Z
    return (np.exp(settings.logl_given_logx(logx) + logx)
            * func.ftilde(logx, settings))


def check_estimator_values(funcs_list, settings, return_int_errors=False):
    output = np.zeros(len(funcs_list))
    errors = np.zeros(len(funcs_list))
    for i, f in enumerate(funcs_list):
        try:
            output[i] = f.analytical(settings)
            # print(f.name + " from analytical")
        except (AttributeError, AssertionError):
            try:
                # We want to calculate (V / Z) int L(X) X ftilde(X),
                # where the V factor comes from integrating normalised
                # distributions in arbitary prior volumes.
                # See Keeton (2011) section on the canonical gaussian for more
                # details.
                logx_terminate = mf.analytic_logx_terminate(settings)
                assert logx_terminate is not None, \
                    "logx_terminate function not set up for current settings"
                result = scipy.integrate.quad(check_integrand, logx_terminate,
                                              0.0, args=(f, settings))
                output[i] = result[0] / np.exp(settings.logz_analytic())
                errors[i] = result[1] / np.exp(settings.logz_analytic())
                # print(f.name + " from integrating = " + str(result[0]) + "
                # with tollerance " + str(result[1]))
            except (AttributeError, AssertionError):
                output[i] = np.nan
                errors[i] = np.nan
    if return_int_errors:
        return output, errors
    else:
        return output


# Estimators
# ----------
# A class is used to hold the functions required for each estimator f(theta)


class logzEstimator(object):

    name = 'logz'
    latex_name = '$\log \mathcal{Z}$'

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        return scipy.misc.logsumexp(logw)

    def analytical(self, settings):
        return settings.logz_analytic()


class zEstimator(object):

    name = 'z'
    latex_name = '$\mathcal{Z}$'

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        return np.exp(scipy.misc.logsumexp(logw))

    def analytical(self, settings):
        return np.exp(settings.logz_analytic())


class n_samplesEstimator(object):

    name = 'n_samples'
    latex_name = '\# samples'

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        return logw.shape[0]


class rEstimator:

    name = 'r'
    latex_name = '$|\\theta|$'

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        w_relative = np.exp(logw - logw.max())
        return (np.sum(w_relative * r) / np.sum(w_relative))

    def analytical(self, settings):
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

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        # get sorted array of p1 values with their posterior weight
        wr = np.zeros((r.shape[0], 2))
        wr[:, 0] = np.exp(logw - logw.max())
        wr[:, 1] = r
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

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        w_relative = np.exp(logw - logw.max())
        return ((np.sum(w_relative * theta[:, self.param_ind - 1])
                / np.sum(w_relative)))

    def analytical(self, settings):
        return 0.

    def ftilde(self, logx, settings):
        return np.zeros(logx.shape)


class theta1confEstimator(object):

    def __init__(self, fraction, param_ind=1):
        assert fraction < 1.0 and fraction > 0, ("conf interval fraction = " +
                                                 str(fraction) +
                                                 " must be between 0 and 1")
        self.param_ind = param_ind
        self.name = 't' + str(param_ind) + 'c_' + str(fraction)
        self.fraction = fraction
        # format percent without trailing zeros
        percent_str = ('%f' % (fraction * 100)).rstrip('0').rstrip('.')
        self.latex_name = ('$\mathrm{C.I.}_{' + percent_str +
                           '\%}(\\theta_{\hat{' + str(param_ind) + '}})$')

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        wp = np.zeros((r.shape[0], 2))
        wp[:, 0] = np.exp(logw - logw.max())
        wp[:, 1] = theta[:, self.param_ind - 1]
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

    def estimator(self, logw=None, logl=None, r=None, theta=None):
        w_relative = np.exp(logw - logw.max())
        return ((np.sum(w_relative * (theta[:, self.param_ind - 1] ** 2)) /
                np.sum(w_relative)))

    def ftilde(self, logx, settings):
        # by symmetry at each (hyper)spherical isolikelihood contour:
        r = settings.r_given_logx(logx)
        return r ** 2 / settings.n_dim
