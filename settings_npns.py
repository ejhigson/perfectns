#!/usr/bin/python
"""Defines a class to hold variables and functions describing the problem's likelihood and priors."""


import npns.maths_functions as mf
import npns.priors as priors
import npns.likelihoods as likelihoods


class NestedSamplingSettings:

    prior_scale = 10
    n_dim = 5
    prior = priors.uniform(10)
    likelihood = likelihoods.gaussian(1)
    dims_to_sample = 1
    # nested sampling settings
    derived_parameters = []  # "uniform"]  # these are added as extra columns in lp
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

    # def get_dynamic_settings(self, dynamic_zp_weight):
    #     assert dynamic_zp_weight >= 0 and dynamic_zp_weight <= 1, "dynamic_zp_weight = " + str(dynamic_zp_weight) + " must be in [0,1]"
    #     dnsd = {
    #         "nlive_1": self.nlive_1,
    #         "nlive_2": self.nlive_2,
    #         "n_calls_frac": self.n_calls_frac,
    #         "dynamic_keep_final_point": self.dynamic_keep_final_point,
    #     }
    #     dnsd["importance_fraction"] = (self.dynamic_fraction, self.dynamic_fraction, dynamic_zp_weight)
    #     return dnsd

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
