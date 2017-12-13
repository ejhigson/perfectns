#!/usr/bin/env python
"""
Defines base class which holds settings for performing perfect nested sampling.
This includes likelihood and prior objects as well as the parameters
controlling how the calculation is performed - for example the number of live
points and whether or not dynamic nested sampling is to be used.
"""

import copy
import PerfectNS.priors as priors
import PerfectNS.likelihoods as likelihoods


class PerfectNSSettings(object):

    """
    Controls how Perfect Nested Sampling is performed.

    For more details of the standard nested sampling and dynamic nested
    sampling algorithms see:

    1) 'Dynamic nested sampling: an improved algorithm for nested  sampling
    parameter estimation and evidence calculation' (Higson et al.
    2017).
    2) 'Sampling errors in nested sampling parameter estimation' (Higson et al.
    2017)

    Parameters
    ----------
    n_dim: int
        Dimension of the likelihood and prior.
    prior: object
        A prior object. See priors.py for more details.
    likelihood: object
        A likelihood object. See likelihoods.py for more details.
    nlive_const: int
        The number of live points for standard nested sampling.
    termination_fraction: float
        Standard nested sampling runs will terminate when the posterior mass
        remaining (estimated from the live points) is less than
        termination_fraction times the posterior mass (evidence) in the dead
        points.
    dynamic_goal: float or None
        Determines is dynamic nested sampling is used, and if so how
        improvements in evidence and parameter estimation accuracy are
        weighted.
    n_samples_max: int or None
        Number of samples after which dynamic nested sampling will terminate.
        If this is None the number of samples to use is chosen using
        nlive_const.
    ninit: int
        Number of live points used in dynamic nested sampling for the initial
        exploratory standard nested sampling run.
    nbatch: int
        Number of threads dynamic nested sampling adds before each
        recalculation of points' importances to the calculation.
    dynamic_fraction: float
        Dynamic nested sampling adds more samples wherever points' importances
        are greater than dynamic_fraction times the largest importance.
    tuned_dynamic_p: bool
        Determines if dynamic nested sampling is tuned for a specific parameter
        estimation problem.
    """

    def __init__(self, **kwargs):

        default_settings = {
            # likelihood and prior settings
            # -----------------------------
            'n_dim': 10,
            'prior': priors.gaussian(10),
            'likelihood': likelihoods.gaussian(1),
            # calculation settings
            # --------------------
            'nlive_const': 200,
            'dims_to_sample': 1,
            'termination_fraction': 0.001,
            # Dynamic nested sampling settings - only used if dynamic_goal is
            # not None.
            'dynamic_goal': None,
            'n_samples_max': None,
            'ninit': 5,
            'nbatch': 1,
            'dynamic_fraction': 0.9,
            'tuned_dynamic_p': False
        }

        for (setting_name, default_value) in default_settings.items():
            setattr(self,
                    setting_name,
                    kwargs.get(setting_name, default_value))

    # functions of the spherically symmetric prior

    def logx_given_r(self, r):
        """
        Parameters
        ----------
        r: float or numpy array
            radial coordinate(s)
        Returns
        -------
        logx: same type and size as r
        """
        return self.prior.logx_given_r(r, self.n_dim)

    def r_given_logx(self, logx):
        """
        Parameters
        ----------
        logx: float or numpy array
        Returns
        -------
        r: same type and size as logx
        """
        return self.prior.r_given_logx(logx, self.n_dim)

    # functions of the spherically symmetric likelihood

    def logl_given_r(self, r):
        """
        Parameters
        ----------
        r: float or numpy array
            radial coordinate(s)
        Returns
        -------
        logl: same type and size as r
        """
        return self.likelihood.logl_given_r(r, self.n_dim)

    def r_given_logl(self, logl):
        """
        Parameters
        ----------
        logl: float or numpy array
        Returns
        -------
        r: same type and size as logl
        """
        return self.likelihood.r_given_logl(logl, self.n_dim)

    # functions of both the likelihood and the prior

    def logl_given_logx(self, logx):
        """
        Parameters
        ----------
        logx: float or numpy array
        Returns
        -------
        logl: same type and size as logx
        """
        return self.logl_given_r(self.r_given_logx(logx))

    def logx_given_logl(self, logl):
        """
        Parameters
        ----------
        logl: float or numpy array
        Returns
        -------
        logx: same type and size as logl
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
        if type(self.prior).__name__ == 'gaussian_cached':
            settings_dict['prior_args'] = {'prior_scale':
                                           self.prior.prior_scale}
        else:
            settings_dict['prior_args'] = self.prior.__dict__
        return settings_dict
