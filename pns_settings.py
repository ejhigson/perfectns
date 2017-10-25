#!/usr/bin/python
"""
Defines base class which holds settings for performing perfect nested sampling.
This inludes likelihood and prior objects as well as the parameters controlling
how the calculation is performed - for example the number of live points and
whether or not dynamic nested sampling is to be used.
"""

import copy
import pns.priors as priors
import pns.likelihoods as likelihoods


class PerfectNestedSamplingSettings(object):

    """
    Perfect Nested Sampling Settings

    Keyword Arguments
    -----------------
    n_dim: int
        Dimension of the likelihood and prior.

    prior: class
        A prior object containing functions logx_given_r(r, n_dim) and
        r_given_logx(logx, n_dim) for some spherically symmetric prior
        pi(theta) = pi(r) centred on r=0. See likelihoods.py for more
        details.
    likelihood: class
        A likelihood object. See likelihoods.py for more details.

    nlive: int
        The number of live points.
        Increasing nlive increases the number of samples taken (~ O(nlive) )
        and therefore increases accuracy and runtime.
    """

    def __init__(self, **kwargs):

        default_settings = {
            # likelihood and prior settings
            # -----------------------------
            'n_dim': 10,
            'prior': priors.gaussian(10),
            # 'prior': priors.gaussian_cached(10, n_dim=10),
            'likelihood': likelihoods.gaussian(1),
            # calculation settings
            # --------------------
            'nlive': 200,
            'data_version': 'v03',
            'dims_to_sample': 1,
            'zv_termination_fraction': 0.001,
            # dynamic nested sampling settings
            # only used if dynamic_goal is not None.
            'dynamic_goal': None,
            'nlive_1': 5,
            'nlive_2': 2,
            'n_calls_frac': 0,
            'dynamic_fraction': 0.9,
            'dynamic_keep_final_point': True,
            # from func args
            'tuned_dynamic_p': False,
            'n_calls_max': None
        }

        for (setting_name, default_value) in default_settings.items():
            setattr(self,
                    setting_name,
                    kwargs.get(setting_name, default_value))

    # functions of the spherically symmetric prior

    def logx_given_r(self, r):
        """
        param r: radial coordinate(s)
        type r: float, numpy array
        returns logx: the prior's corresponding logx coordinates
        rtype: float, numpy array (same type as input)
        """
        return self.prior.logx_given_r(r, self.n_dim)

    def r_given_logx(self, logx):
        """
        param logx: logx coordinate(s)
        type logx: float, numpy array
        returns r: the prior's corresponding radial coordinates
        rtype: float, numpy array (same type as input)
        """
        return self.prior.r_given_logx(logx, self.n_dim)

    # functions of the spherically symmetric likelihood

    def logl_given_r(self, r):
        """
        param r: radial coordinate(s)
        type r: float, numpy array
        returns logl: the loglikelihoods corresponding to the input r values
        rtype: float, numpy array (same type as input)
        """
        return self.likelihood.logl_given_r(r, self.n_dim)

    def r_given_logl(self, logl):
        """
        param logl: loglikelihood value(s)
        type logl: float, numpy array
        returns r: the radial coordiantes corresponding to the input
        loglikelihood values
        rtype: float, numpy array (same type as input)
        """
        return self.likelihood.r_given_logl(logl, self.n_dim)

    # functions of both the likelihood and the prior

    def logl_given_logx(self, logx):
        """
        param logx: logx coordinate(s)
        type logx: float, numpy array
        returns logl: the loglikelihoods corresponding to the input logx values
        rtype: float, numpy array (same type as input)
        """
        return self.logl_given_r(self.r_given_logx(logx))

    def logx_given_logl(self, logl):
        """
        param logl: loglikelihood value(s)
        type logl: float, numpy array
        returns logx: the prior's corresponding logx coordinates
        rtype: float, numpy array (same type as input)
        """
        return self.logx_given_r(self.r_given_logl(logl))

    def logz_analytic(self):
        """
        If available gives an analytically calculated value for the log
        evidence for the likelihood and prior (useful for checking results).

        This functionality is stored in the likelihood object. If it has not
        been set up then an error message is printed and nothing is returned.
        """
        try:
            return self.likelihood.logz_analytic(self.prior, self.n_dim)
        except (AttributeError, AssertionError):
            print('No logz_analytic set up for ' +
                  type(self.likelihood).__name__ +
                  " likelihoods and " + type(self.prior).__name__ + " priors.")

    def get_settings_dict(self):
        """
        Returns a dictionary containing settings information. The names and
        parameters of the likelihoods and priors are stored instead of the
        objects themselves so they can be saved with pickle.
        """
        settings_dict = copy.deepcopy(self.__dict__)
        # Replace the likelihood and prior objects with information about them
        # in the form of strings and dicts so the output can be saved with
        # pickle.
        settings_dict['likelihood'] = type(self.likelihood).__name__
        settings_dict['likelihood_args'] = self.likelihood.__dict__
        settings_dict['prior'] = type(self.prior).__name__
        settings_dict['prior_args'] = self.prior.__dict__
        return settings_dict
