#!/usr/bin/env python
"""
Test the perfectns module installation.
"""

import unittest
import copy
import numpy as np
import matplotlib
import perfectns.settings
import perfectns.estimators as e
import perfectns.cached_gaussian_prior
import perfectns.likelihoods as likelihoods
import perfectns.nested_sampling as ns
import perfectns.results_tables as rt
import perfectns.priors as priors
import perfectns.plots
import nestcheck.analyse_run as ar
# from matplotlib.testing.decorators import cleanup
# import os
# import nestcheck.io_utils as iou


class TestPerfectNS(unittest.TestCase):

    """Container for module tests."""

    def setUp(self):
        """
        Set up list of estimator objects and settings for each test.
        Use all the estimators in the module in each case, and choose settings
        so the tests run quickly.
        """
        self.estimator_list = [e.LogZ(),
                               e.Z(),
                               e.ParamMean(),
                               e.ParamSquaredMean(),
                               e.ParamCred(0.5),
                               e.ParamCred(0.84),
                               e.RMean(),
                               # e.RMean(from_theta=True),
                               e.RCred(0.84)]
        self.settings = perfectns.settings.PerfectNSSettings()
        self.settings.likelihood = likelihoods.Gaussian(likelihood_scale=1)
        self.settings.prior = priors.Gaussian(prior_scale=10)
        self.settings.n_dim = 2
        self.settings.dims_to_sample = 2
        self.settings.nlive_const = 20
        self.settings.dynamic_goal = None

    def test_dynamic_results_table(self):
        """
        Test generating a table comparing dynamic and standard nested sampling;
        this covers a lot of the perfectns module's functionality.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        # Need parallelise=False for coverage module to give correct answers
        dynamic_table = rt.get_dynamic_results(5, [0, 0.5, 1],
                                               self.estimator_list,
                                               self.settings,
                                               load=False,
                                               parallelise=False)
        # Try it again with parallelise=True to cover parallel parts
        dynamic_table = rt.get_dynamic_results(5, [0, 1],
                                               self.estimator_list,
                                               self.settings,
                                               load=False,
                                               parallelise=True)
        # The first row of the table contains analytic calculations of the
        # estimators' values given the likelihood and prior. These are not
        # available for RCred.
        for est in self.estimator_list:
            if est.name != e.RCred(0.84).name:
                self.assertTrue(~np.isnan(dynamic_table.loc['true values',
                                                            est.name]))
        # None of the other values in the table should be NaN:
        self.assertFalse(np.any(np.isnan(dynamic_table.values[1:, :])))

    def test_bootstrap_results_table(self):
        """
        Generate a table showing sampling error estimates using the bootstrap
        method.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        # Need parallelise=False for coverage module to give correct answers
        bootstrap_table = rt.get_bootstrap_results(3, 10,
                                                   self.estimator_list,
                                                   self.settings,
                                                   n_run_ci=2,
                                                   n_simulate_ci=100,
                                                   add_sim_method=True,
                                                   cred_int=0.95,
                                                   load=False,
                                                   ninit_sep=False,
                                                   parallelise=False)
        # The first row of the table contains analytic calculations of the
        # estimators' values given the likelihood and prior which have already
        # been tested in test_dynamic_results_table.
        # None of the other values in the table should be NaN:
        self.assertFalse(np.any(np.isnan(bootstrap_table.values[1:, :])))

    def test_standard_ns_exp_power_likelihood_gaussian_prior(self):
        """Check the exp_power likelihood, as well as some functions in
        analyse_run."""
        # np.random.seed(0)
        settings = copy.deepcopy(self.settings)
        settings.likelihood = likelihoods.ExpPower(likelihood_scale=1,
                                                   power=2)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.logz_analytic()
        ns_run = ns.generate_ns_run(settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        # print(values)
        self.assertFalse(np.any(np.isnan(values)))

    def test_standard_ns_cauchy_likelihood_gaussian_prior(self):
        """Check the Cauchy likelihood."""
        settings = copy.deepcopy(self.settings)
        # settings.n_dim = 10
        settings.likelihood = likelihoods.Cauchy(likelihood_scale=1)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.logz_analytic()
        ns_run = ns.generate_ns_run(settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))

    def test_standard_ns_gaussian_likelihood_uniform_prior(self):
        """Check the uniform prior."""
        settings = copy.deepcopy(self.settings)
        settings.prior = priors.Uniform(prior_scale=10)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.logz_analytic()
        ns_run = ns.generate_ns_run(settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))

    def test_cached_gaussian_prior(self):
        """Check the cached_gaussian prior."""
        settings = copy.deepcopy(self.settings)
        # test initialisation with and without specifying n_dim
        settings.prior = priors.GaussianCached(prior_scale=10,
                                               save_dict=False)
        settings.prior = priors.GaussianCached(
            prior_scale=10, save_dict=False, n_dim=settings.n_dim)
        # check the argument options and messages for interp_r_logx_dict
        self.assertRaises(
            TypeError, perfectns.cached_gaussian_prior.interp_r_logx_dict,
            2000, 10, unexpected=0)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.get_settings_dict()
        ns_run = ns.generate_ns_run(settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))

    def test_plot_rel_posterior_mass(self):
        fig = perfectns.plots.plot_rel_posterior_mass(
            [perfectns.likelihoods.Gaussian(1),
             perfectns.likelihoods.ExpPower(1, 2)],
            perfectns.priors.Gaussian(1),
            [2], np.linspace(-10, 0, 100))
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertRaises(
            TypeError, perfectns.plots.plot_rel_posterior_mass,
            [perfectns.likelihoods.Gaussian(1),
             perfectns.likelihoods.ExpPower(1, 2)],
            perfectns.priors.Gaussian(1),
            [2], np.linspace(-10, 0, 100), unexpected=0)

    def test_plot_dynamic_nlive(self):
        fig = perfectns.plots.plot_dynamic_nlive(
            [None, 0, 1, 1], self.settings, n_run=2,
            tuned_dynamic_ps=[False, False, False, True])
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Test ymax and the fallback for normalising analytic lines when the
        # dynamic goal which is meant to mirror them is not present
        fig = perfectns.plots.plot_dynamic_nlive(
            [None], self.settings, n_run=2,
            tuned_dynamic_ps=[True], ymax=1000)
        # Test unexpected kwargs check
        self.assertRaises(
            TypeError, perfectns.plots.plot_dynamic_nlive,
            [None, 0, 1, 1], self.settings, n_run=2,
            tuned_dynamic_ps=[False, False, False, True], unexpected=0)

    def test_plot_parameter_logx_diagram(self):
        for ftheta in [e.ParamMean(), e.ParamSquaredMean(), e.RMean()]:
            fig = perfectns.plots.plot_parameter_logx_diagram(
                self.settings, ftheta, x_points=50, y_points=50)
            self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Test unexpected kwargs check
        self.assertRaises(
            TypeError, perfectns.plots.plot_parameter_logx_diagram,
            self.settings, ftheta, x_points=50, y_points=50, unexpected=0)

    def test_settings(self):
        self.assertRaises(
            TypeError, perfectns.settings.PerfectNSSettings, unexpected=0)
        settings = copy.deepcopy(self.settings)
        settings.dynamic_goal = 1
        settings.nbatch += 1
        settings.nlive_const = None
        settings.tuned_dynamic_p = True
        settings.n_samples_max = 100
        settings.save_name()


if __name__ == '__main__':
    unittest.main()
