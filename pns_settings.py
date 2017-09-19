#!/usr/bin/python
"""Defines a class to hold variables and functions describing the problem's likelihood and priors."""

import numpy as np
import pns.maths_functions as mf
import pns.priors as priors
import pns.likelihoods as likelihoods


class PerfectNestedSamplingSettings:

    prior_scale = 10
    n_dim = 5
    prior = priors.uniform(10)
    likelihood = likelihoods.gaussian(1)
    dims_to_sample = 1
    # nested sampling settings
    zv_termination_fraction = 0.0001  # do not write in standard form as it messes with file names
    # dynamic settings
    nlive_1 = 5
    nlive_2 = 1
    n_calls_frac = 0
    dynamic_fraction = 0.9
    dynamic_keep_final_point = True

    # functions of priors

    def logx_given_r(self, r):
        return self.prior.logx_given_r(r, self.n_dim)

    def r_given_logx(self, logx):
        return self.prior.r_given_logx(logx, self.n_dim)

    # functions of likelihoods

    def logl_given_r(self, r):
        return self.likelihood.logl_given_r(r, self.n_dim)

    def r_given_logl(self, logl):
        return self.likelihood.r_given_logl(logl, self.n_dim)

    # generic functions

    def logl_given_logx(self, logx):
        return self.logl_given_r(self.r_given_logx(logx))

    def logx_given_logl(self, logl):
        return self.logx_given_r(self.r_given_logl(logl))

    def sample_contours(self, logx):
        """Samples {theta} given isolikelihood contours {logx} as 1D numpy array."""
        r = self.r_given_logx(logx)
        return mf.sample_nsphere_shells(r, self.n_dim, self.dims_to_sample)
        # return np.hstack((samples, np.reshape(logx, (logx.shape[0], 1))))

    def logz_analytic(self):
        if type(self.likelihood).__name__ == "gaussian" and type(self.prior).__name__ == "uniform":
            return - mf.nsphere_logvol(self.n_dim, radius=self.prior.prior_scale) + mf.gaussian_logx_given_r(self.prior.prior_scale, self.likelihood.likelihood_scale, self.n_dim)
        elif type(self.likelihood).__name__ == "gaussian" and type(self.prior).__name__ == "gaussian":
            # See "Products and convolutions of Gaussian probability density functions" (P Bromiley, 2003) page 3 for a derivation of this result
            self.logz_analytic = (-self.n_dim / 2.) * np.log(2 * np.pi * (self.likelihood.likelihood_scale ** 2 + self.prior.prior_scale ** 2))

    # def get_settings_dict(self):
    #     """
    #     Returns a dictionary which contains all settings, including both class and instance variables.
    #     This is needed as the __dict__ magic method does not give class variables.
    #     """
    #     members = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
    #     dict = {}
    #     for i in members:
    #         if not inspect.ismethod(getattr(self, i)) and i not in ["likelihood_prior", "nlive_1", "nlive_2", "n_calls_frac", "dynamic_fraction", "dynamic_keep_final_point"]:  # remove classes and methods that cannot be pickled
    #             dict[i] = getattr(self, i)
    #     return dict
    #
#    def set_var(self, dictionary):
#        """Sets all variables to values from an input dictionary."""
#        for key, value in dictionary.items():
#            setattr(self, key, value)
#
#        return dict
