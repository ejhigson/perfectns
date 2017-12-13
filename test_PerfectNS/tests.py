#!/usr/bin/env python
"""
Test the PerfectNS module installation.
"""

import unittest
import numpy as np
import PerfectNS.settings
import PerfectNS.estimators as e
import PerfectNS.likelihoods as likelihoods
import PerfectNS.nested_sampling as ns
import PerfectNS.results_tables as rt
import PerfectNS.priors as priors
import PerfectNS.analyse_run as ar


class TestPerfectNS(unittest.TestCase):

    def setUp(self):
        """
        Set up list of estimator objects and settings for each test.
        Use all the estimators in the module in each case, and choose settings
        so the tests run quickly.
        """
        self.estimator_list = [e.logzEstimator(),
                               e.zEstimator(),
                               e.paramMeanEstimator(),
                               e.paramSquaredMeanEstimator(),
                               e.paramCredEstimator(0.5),
                               e.paramCredEstimator(0.84),
                               e.rMeanEstimator(),
                               e.rCredEstimator(0.84)]
        self.settings = PerfectNS.settings.PerfectNSSettings()
        self.settings.n_dim = 2
        self.settings.nlive_const = 20
        self.settings.dynamic_goal = None

    def test_dynamic_results_table(self):
        """
        Test generating a table comparing dynamic and standard nested sampling;
        this covers a lot of the PerfectNS module's functionality.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        self.settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
        self.settings.prior = priors.gaussian(prior_scale=10)
        dynamic_table = rt.get_dynamic_results(5, [0, 1],
                                               self.estimator_list,
                                               self.settings,
                                               parallelise=True)
        # The first row of the table contains analytic calculations of the
        # estimators' values given the likelihood and prior. These are not
        # available for rCredEstimator.
        for est in self.estimator_list:
            if est.name != e.rCredEstimator(0.84).name:
                self.assertTrue(~np.isnan(dynamic_table.loc['true values',
                                                            est.name]))
        # None of the other values in the table should be NaN:
        self.assertTrue(np.all(~np.isnan(dynamic_table.values[1:, :])))

    def test_bootstrap_results_table(self):
        """
        Generate a table showing sampling error estimates using the bootstrap
        method.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        self.settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
        self.settings.prior = priors.gaussian(prior_scale=10)
        bootstrap_table = rt.get_bootstrap_results(3, 10,
                                                   self.estimator_list,
                                                   self.settings,
                                                   n_run_ci=2,
                                                   n_simulate_ci=100,
                                                   add_sim_method=True,
                                                   cred_int=0.95,
                                                   ninit_sep=False,
                                                   parallelise=True)
        # The first row of the table contains analytic calculations of the
        # estimators' values given the likelihood and prior which have already
        # been tested in test_dynamic_results_table.
        # None of the other values in the table should be NaN:
        self.assertTrue(np.all(~np.isnan(bootstrap_table.values[1:, :])))

    def test_standard_ns_exp_power_likelihood_gaussian_prior(self):
        """Check the exp_power likelihood."""
        self.settings.exp_power = likelihoods.exp_power(likelihood_scale=1,
                                                        power=2)
        self.settings.prior = priors.gaussian(prior_scale=10)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertTrue(np.all(~np.isnan(values)))

    def test_standard_ns_cauchy_likelihood_gaussian_prior(self):
        """Check the Cauchy likelihood."""
        self.settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
        self.settings.prior = priors.gaussian(prior_scale=10)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertTrue(np.all(~np.isnan(values)))

    def test_standard_ns_gaussian_likelihood_uniform_prior(self):
        """Check the uniform prior."""
        self.settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
        self.settings.prior = priors.uniform(prior_scale=10)
        ns_run = ns.generate_ns_run(self.settings)
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertTrue(np.all(~np.isnan(values)))


if __name__ == '__main__':
    unittest.main()
