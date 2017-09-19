#!/usr/bin/python
"""Defines a class to hold variables and functions describing the problem's likelihood and priors."""

import copy
import pns.maths_functions as mf
import pns.priors as priors
import pns.likelihoods as likelihoods


class PerfectNestedSamplingSettings(object):

    def __init__(self, **kwargs):

        default_settings = {
            # likelihood and prior settings
            # -----------------------------
            'n_dim': 5,
            'prior': priors.uniform(10),
            'likelihood': likelihoods.gaussian(1),
            # calculation settings
            # --------------------
            'nlive': 100,
            'dims_to_sample': 1,
            'zv_termination_fraction': 0.0001,  # do not write in standard form as it messes with file names
            'dynamic_goal': None,
            # dynamic nested sampling settings - only used if dynamic_goal is not None
            'nlive_1': 5,
            'nlive_2': 1,
            'n_calls_frac': 0,
            'dynamic_fraction': 0.9,
            'dynamic_keep_final_point': True,
            # from func args
            'tuned_dynamic_p': False,
            'n_calls_max': None
        }

        for (setting_name, default_value) in default_settings.items():
            setattr(self, setting_name, kwargs.get(setting_name, default_value))

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
        try:
            return self.likelihood.logz_analytic(self.prior, self.n_dim)
        except (AttributeError, AssertionError):
            print('No logz_analytic set up for ' + type(self.likelihood).__name__ + " likelihoods and " + type(self.prior).__name__ + " priors.")

    def get_settings_dict(self):
        """
        Returns a dictionary which contains all settings, including both class and instance variables.
        This is needed as the __dict__ magic method does not give class variables.
        """
        settings_dict = copy.deepcopy(self.__dict__)
        # replace the likelihood and prior objects with information about them in the form of strings and dicts so the output can be saved with pickle
        settings_dict['likelihood'] = type(self.likelihood).__name__
        settings_dict['likelihood_args'] = self.likelihood.__dict__
        settings_dict['prior'] = type(self.prior).__name__
        settings_dict['prior_args'] = self.prior.__dict__
        return settings_dict

#    def set_var(self, dictionary):
#        """Sets all variables to values from an input dictionary."""
#        for key, value in dictionary.items():
#            setattr(self, key, value)
#
#        return dict
