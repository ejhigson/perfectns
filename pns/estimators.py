#!/usr/bin/python
"""Contains classes representing estimators f(theta) for use on nested sampling output and analytical analysis."""

import numpy as np
import scipy
# perfect nested sampling modules
import pns.maths_functions as mf


# Checking estimator results
# --------------------------


def check_integrand(logx, func, settings):
    """Helper function to return integrand L(X) X ftilde(X) for checking estimator values by numerical intergration."""
    # returns L(X) X ftilde(X) for integrating dlogx
    # NB this must be normalised by a factor V / Z
    return np.exp(settings.likelihood_prior.logl_given_logx(logx) + logx) * func.ftilde(np.exp(logx), settings)


def check_estimator_values(funcs_list, settings, return_int_errors=False, force_integration=False):
    output = np.zeros(len(funcs_list))
    errors = np.zeros(len(funcs_list))
    for i, f in enumerate(funcs_list):
        # Check for discontinuities in the function:
        # These can be given to scipy.integrate.quad using the points argument (default value is None).
        if hasattr(f, "integrate_points"):
            points = f.integrate_points(settings)
        else:
            points = None
        # use the force_integration option to give the integration method priority over the analytic method
        if force_integration:
            try:
                # we want (V / Z) int L(X) X ftilde(X)
                # where the V factor comes from integrating normalised distributions in arbitary prior volumes.
                # See Keeton (2011) section on the canonical gaussian for more details.
                logx_terminate = mf.analytic_logx_terminate(settings)
                result = scipy.integrate.quad(check_integrand, logx_terminate, 0.0, args=(f, settings), points=points)
                output[i] = result[0] / np.exp(settings.likelihood_prior.logz_analytic)
                errors[i] = result[1] / np.exp(settings.likelihood_prior.logz_analytic)
                # print(f.name + " from integrating = " + str(result[0]) + " with tollerance " + str(result[1]))
            except (AttributeError, AssertionError):
                try:
                    output[i] = f.analytical(settings)
                    # print(f.name + " from analytical")
                except (AttributeError, AssertionError):
                    output[i] = np.nan
                    errors[i] = np.nan
        else:
            try:
                output[i] = f.analytical(settings)
                # print(f.name + " from analytical")
            except (AttributeError, AssertionError):
                try:
                    # We want to calculate (V / Z) int L(X) X ftilde(X),
                    # where the V factor comes from integrating normalised distributions in arbitary prior volumes.
                    # See Keeton (2011) section on the canonical gaussian for more details.
                    logx_terminate = mf.analytic_logx_terminate(settings)
                    result = scipy.integrate.quad(check_integrand, logx_terminate, 0.0, args=(f, settings), points=points)
                    output[i] = result[0] / np.exp(settings.likelihood_prior.logz_analytic)
                    errors[i] = result[1] / np.exp(settings.likelihood_prior.logz_analytic)
                    # print(f.name + " from integrating = " + str(result[0]) + " with tollerance " + str(result[1]))
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

    name = 'z_scaled'
    latex_name = '$\log \mathcal{Z}$'

    def estimator(self, settings, logw=None, logl=None, r=None, theta=None):
        return mf.log_sum_given_logs(logw)

    def analytical(self, settings):
        try:
                return settings.logz_analytic
        except AttributeError:
            return None


class rEstimator:

    name = 'r'
    latex_name = '$|\\theta|$'

    def estimator(self, settings, logw=None, logl=None, r=None, theta=None):
        w_relative = np.exp(logw - logw.max())
        return (np.sum(w_relative * r) / np.sum(w_relative))

    def analytical(self, settings):
        return 0

    def min(self, settings):
        return 0


class rconfEstimator(object):

    def __init__(self, fraction):
        assert fraction < 1.0, "confidence interval fraction = " + str(fraction) + " must be < 1.0"
        assert fraction > 0, "confidence interval fraction = " + str(fraction) + " must be > 0"
        self.name = 'rconf_' + str(fraction)
        self.fraction = fraction
        # self.latex_name = '$\Theta_{P(|\\theta_{\hat{1}}|<\Theta)=' + str(fraction * 100) + '\%}$'
        percent_str = ('%f' % (fraction * 100)).rstrip('0').rstrip('.')  # format percent without trailing zeros as per http://stackoverflow.com/questions/2440692/formatting-floats-in-python-without-superfluous-zeros
        self.latex_name = '$\mathrm{C.I.}_{' + percent_str + '\%}(|\\theta|)$'

    def min(self, settings):
        return 0

    def estimator(self, settings, logw=None, logl=None, r=None, theta=None):
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
        self.name = 'theta' + str(param_ind)
        self.latex_name = '$\\overline{\\theta_{\hat{' + str(param_ind) + '}}}$'

    def estimator(self, settings, logw=None, logl=None, r=None, theta=None):
        w_relative = np.exp(logw - logw.max())
        return (np.sum(w_relative * theta[:, self.param_ind - 1]) / np.sum(w_relative))

    def analytical(self, settings):
        return 0.


class theta1confEstimator(object):

    def __init__(self, fraction, param_ind=1):
        assert fraction < 1.0, "confidence interval fraction = " + str(fraction) + " must be < 1.0"
        assert fraction > 0, "confidence interval fraction = " + str(fraction) + " must be > 0"
        self.param_ind = param_ind
        self.name = 'theta' + str(param_ind) + 'conf_' + str(fraction)
        self.fraction = fraction
        percent_str = ('%f' % (fraction * 100)).rstrip('0').rstrip('.')  # format percent without trailing zeros as per http://stackoverflow.com/questions/2440692/formatting-floats-in-python-without-superfluous-zeros
        self.latex_name = '$\mathrm{C.I.}_{' + percent_str + '\%}(\\theta_{\hat{' + str(param_ind) + '}})$'

    def estimator(self, settings, logw=None, logl=None, r=None, theta=None):
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
    # N.B. this is var about the likelihood mean and is therefore not affected by the mean

    def __init__(self, param_ind=1):
        self.param_ind = param_ind
        self.name = 'theta' + str(param_ind) + 'squared'
        self.latex_name = '$\\overline{\\theta_{\hat{' + str(param_ind) + '}}}$'
        self.latex_name = '$\\overline{\\theta^2_{\hat{' + str(param_ind) + '}}}$'

    def estimator(self, settings, logw=None, logl=None, r=None, theta=None):
        w_relative = np.exp(logw - logw.max())
        return (np.sum(w_relative * (theta[:, self.param_ind - 1] ** 2)) / np.sum(w_relative))
