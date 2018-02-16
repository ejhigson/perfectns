#!/usr/bin/env python
"""
Test the perfectns module installation.
"""

import os
import unittest
import numpy as np
import perfectns.settings
import perfectns.estimators as e
import perfectns.likelihoods as likelihoods
import perfectns.nested_sampling as ns
import perfectns.results_tables as rt
import perfectns.priors as priors
import nestcheck.analyse_run as ar
import nestcheck.io_utils as iou


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
                               e.RCred(0.84)]
        self.settings = perfectns.settings.PerfectNSSettings()
        self.settings.n_dim = 2
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
        self.settings.likelihood = likelihoods.Gaussian(likelihood_scale=1)
        self.settings.prior = priors.Gaussian(prior_scale=10)
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
        self.settings.likelihood = likelihoods.Gaussian(likelihood_scale=1)
        self.settings.prior = priors.Gaussian(prior_scale=10)
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
        np.random.seed(0)
        self.settings.exp_power = likelihoods.ExpPower(likelihood_scale=1,
                                                       power=2)
        self.settings.prior = priors.Gaussian(prior_scale=10)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        print(values)
        self.assertFalse(np.any(np.isnan(values)))

    def test_standard_ns_cauchy_likelihood_gaussian_prior(self):
        """Check the Cauchy likelihood."""
        self.settings.n_dim = 10
        self.settings.likelihood = likelihoods.Cauchy(likelihood_scale=1)
        self.settings.prior = priors.Gaussian(prior_scale=10)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))

    def test_standard_ns_gaussian_likelihood_uniform_prior(self):
        """Check the uniform prior."""
        self.settings.likelihood = likelihoods.Gaussian(likelihood_scale=1)
        self.settings.prior = priors.Uniform(prior_scale=10)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))

    def test_standard_ns_gaussian_likelihood_cached_gaussian_prior(self):
        """Check the cached_gaussian prior."""
        self.settings.likelihood = likelihoods.Gaussian(likelihood_scale=1)
        self.settings.prior = priors.GaussianCached(prior_scale=10,
                                                    save_dict=False)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))

    def test_save_load_utils(self):
        """Check the input output functions."""
        filename = iou.data_save_name(self.settings, 1)
        testdata = np.random.random(5)
        iou.pickle_save(testdata, filename, extension='.pkl')
        testdata_out = iou.pickle_load(filename, extension='.pkl')
        os.remove(filename + '.pkl')
        self.assertTrue(np.array_equal(testdata, testdata_out))


if __name__ == '__main__':
    unittest.main()
