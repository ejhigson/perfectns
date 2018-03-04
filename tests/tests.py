#!/usr/bin/env python
"""
Test the perfectns module installation.
"""

import os
import shutil
import unittest
import copy
import numpy as np
import numpy.testing
import pandas as pd
import matplotlib
import perfectns.settings
import perfectns.estimators as e
import perfectns.cached_gaussian_prior
import perfectns.likelihoods as likelihoods
import perfectns.nested_sampling as ns
import perfectns.results_tables as rt
import perfectns.maths_functions
import perfectns.priors as priors
import perfectns.plots
import nestcheck.analyse_run as ar
import nestcheck.data_processing as dp


class TestPerfectNS(unittest.TestCase):

    """Container for module tests."""

    def setUp(self):
        """
        Set up list of estimator objects and settings for each test.
        Use all the estimators in the module in each case, and choose settings
        so the tests run quickly.
        """
        self.cache_dir = 'cache_tests/'
        assert not os.path.exists(self.cache_dir[:-1]), \
            ('Directory ' + self.cache_dir[:-1] + ' exists! Tests use this ' +
             'dir to check caching then delete it afterwards, so the path ' +
             'should be left empty.')
        self.estimator_list = [e.LogZ(),
                               e.Z(),
                               e.ParamMean(),
                               e.ParamSquaredMean(),
                               e.ParamCred(0.5),
                               e.ParamCred(0.84),
                               e.RMean(from_theta=True),
                               e.RCred(0.84, from_theta=True)]
        self.settings = perfectns.settings.PerfectNSSettings()
        self.settings.likelihood = likelihoods.Gaussian(likelihood_scale=1)
        self.settings.prior = priors.Gaussian(prior_scale=10)
        self.settings.dims_to_sample = 2
        self.settings.n_dim = 2
        self.settings.nlive_const = 20
        self.settings.dynamic_goal = None

    def tearDown(self):
        # Remove any caches saved by the tests
        try:
            shutil.rmtree(self.cache_dir[:-1])
        except FileNotFoundError:
            pass

    def test_nestcheck_run_format(self):
        """
        Check perfectns runs are compatable with the nestcheck run format
        (excepting their additional 'logx' and 'r' keys).
        """
        settings = copy.deepcopy(self.settings)
        for dynamic_goal in [None, 0, 0.5, 1]:
            settings.dynamic_goal = dynamic_goal
            run = ns.generate_ns_run(settings)
            del run['logx']
            del run['r']
            dp.check_ns_run(run)

    def test_dynamic_results_table(self):
        """
        Test generating a table comparing dynamic and standard nested sampling;
        this covers a lot of the perfectns module's functionality.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        # Need parallelise=False for coverage module to give correct answers
        dynamic_table = rt.get_dynamic_results(
            5, [0, 0.25, 1, 1], self.estimator_list, self.settings, load=True,
            save=True, cache_dir=self.cache_dir,
            parallelise=False, tuned_dynamic_ps=[False, False, False, True])
        # Uncomment below line to update values if they change for a known
        # reason
        # dynamic_table.to_pickle('tests/dynamic_table_test_values.pkl')
        # Check the values of every row for the theta1 estimator
        test_values = pd.read_pickle('tests/dynamic_table_test_values.pkl')
        numpy.testing.assert_allclose(dynamic_table.values, test_values.values,
                                      rtol=1e-13)
        # None of the other values in the table should be NaN:
        self.assertFalse(np.any(np.isnan(dynamic_table.values)))
        # Check the kwargs checking
        self.assertRaises(TypeError, rt.get_dynamic_results, 5, [0],
                          self.estimator_list, self.settings, unexpected=1)

    def test_bootstrap_results_table(self):
        """
        Generate a table showing sampling error estimates using the bootstrap
        method.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        # Need parallelise=False for coverage module to give correct answers
        bootstrap_table = rt.get_bootstrap_results(5, 10,
                                                   self.estimator_list,
                                                   self.settings,
                                                   n_run_ci=2,
                                                   n_simulate_ci=100,
                                                   add_sim_method=True,
                                                   cred_int=0.95,
                                                   load=True, save=True,
                                                   cache_dir=self.cache_dir,
                                                   ninit_sep=True,
                                                   parallelise=False)
        # The first row of the table contains analytic calculations of the
        # estimators' values given the likelihood and prior which have already
        # been tested in test_dynamic_results_table.
        # None of the other values in the table should be NaN:
        self.assertFalse(np.any(np.isnan(bootstrap_table.values[1:, :])))
        # Check the kwargs checking
        self.assertRaises(TypeError, rt.get_bootstrap_results, 3, 10,
                          self.estimator_list, self.settings, unexpected=1)

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
        self.assertRaises(
            TypeError, priors.GaussianCached,
            prior_scale=10, unexpected=0)
        settings.prior = priors.GaussianCached(
            prior_scale=10, save_dict=True, n_dim=settings.n_dim,
            cache_dir=self.cache_dir,
            interp_density=10, logx_min=-30)
        # Test inside and outside cached regime (logx<-10).
        # Need fairly low number of places
        for logx in [-1, -11]:
            self.assertAlmostEqual(
                settings.logx_given_logl(settings.logl_given_logx(logx)),
                logx, places=3)
        # Test array version of the function too
        logx = np.asarray([-2])
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(logx)[0]),
            logx[0], places=12)
        settings.get_settings_dict()
        # Generate NS run using get_run_data to check it checks the cache
        # before submitting process to parallel apply
        ns_run = ns.get_run_data(settings, 1, load=False, save=False)[0]
        values = ar.run_estimators(ns_run, self.estimator_list)
        self.assertFalse(np.any(np.isnan(values)))
        # check the argument options and messages for interp_r_logx_dict
        self.assertRaises(
            TypeError, perfectns.cached_gaussian_prior.interp_r_logx_dict,
            2000, 10, logx_min=-100, interp_density=1, unexpected=0)

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
            tuned_dynamic_ps=[False, False, False, True],
            save=False, load=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Test ymax and the fallback for normalising analytic lines when the
        # dynamic goal which is meant to mirror them is not present
        fig = perfectns.plots.plot_dynamic_nlive(
            [None], self.settings, n_run=2,
            save=False, load=False,
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
        # Test warning for estimators without CDF
        perfectns.plots.cdf_given_logx(e.LogZ(), np.zeros(1), np.zeros(1),
                                       self.settings)
        # Test unexpected kwargs check
        self.assertRaises(
            TypeError, perfectns.plots.plot_parameter_logx_diagram,
            self.settings, e.ParamMean(), x_points=50, y_points=50,
            unexpected=0)

    def test_settings(self):
        self.assertRaises(
            TypeError, perfectns.settings.PerfectNSSettings, unexpected=0)
        # check all the if statements in settings.save_name() for unusual
        # settings
        settings = copy.deepcopy(self.settings)
        settings.dynamic_goal = 1
        settings.nbatch += 1
        settings.nlive_const = None
        settings.tuned_dynamic_p = True
        settings.n_samples_max = 100
        settings.dynamic_fraction = 0.8
        settings.dims_to_sample = 2
        settings.save_name()
        self.assertRaises(TypeError, settings.__setattr__, 'unexpected', 1)

    def test_estimators(self):
        # Check analytic values
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.LogZ(), self.settings),
            -6.4529975832506050, places=10)
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.Z(), self.settings),
            1.5757915157613399e-03, places=10)
        self.assertEqual(
            e.get_true_estimator_values(e.ParamMean(), self.settings), 0)
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.ParamSquaredMean(), self.settings),
            9.9009851517647807e-01, places=10)
        self.assertEqual(
            e.get_true_estimator_values(e.ParamCred(0.5), self.settings), 0)
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.ParamCred(0.84), self.settings),
            9.8952257789120635e-01, places=10)
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.RMean(), self.settings),
            1.2470645289408879e+00, places=10)
        self.assertTrue(np.isnan(
            e.get_true_estimator_values(e.RCred(0.84), self.settings)))
        self.assertTrue(np.isnan(
            e.get_true_estimator_values([e.RCred(0.84)], self.settings)[0]))
        # Check calculating the radius from theta: when points in theta have
        # coordinates (1, 1) the radius should be sqrt(2)
        run_dict_temp = {'theta': np.full((2, 2), 1),
                         'r': np.full((2,), np.sqrt(2)),
                         'logl': np.full((2,), 0.),
                         'nlive_array': np.full((2,), 5.),
                         'settings': {'dims_to_sample': 2, 'n_dim': 2}}
        logw_temp = np.zeros(2)
        self.assertEqual(e.RMean(from_theta=True)(
            run_dict_temp, logw=logw_temp), np.sqrt(2))
        # Check without deriving r from theta
        self.assertEqual(e.RMean(from_theta=False)(
            run_dict_temp, logw=logw_temp), np.sqrt(2))
        # Check RCred
        e.RCred(0.84, from_theta=True)(run_dict_temp, logw=logw_temp)
        e.RCred(0.84, from_theta=False)(run_dict_temp, logw=logw_temp)
        # Check logw=None
        e.RMean(from_theta=False)(run_dict_temp, logw=None)
        e.RCred(0.84, from_theta=False)(run_dict_temp, logw=None)
        # Check CountSamples estimator is working ok
        self.assertEqual(e.CountSamples()({'logl': np.zeros(10)}), 10)

    def test_maths_functions(self):
        # By default only used in high dim so manually test with dim=100
        perfectns.maths_functions.sample_nsphere_shells(
            np.asarray([1]), 100, n_sample=1)
        # Check handling of n_sample=None
        self.assertEqual(
            perfectns.maths_functions.sample_nsphere_shells_normal(
                np.asarray([1]), 2, n_sample=None).shape, (1, 2))
        self.assertEqual(
            perfectns.maths_functions.sample_nsphere_shells_beta(
                np.asarray([1]), 2, n_sample=None).shape, (1, 2))

    def test_nested_sampling(self):
        settings = copy.deepcopy(self.settings)
        settings.dynamic_goal = 0
        settings.n_samples_max = None
        settings.nlive_const = 10
        ns.generate_dynamic_run(settings)
        # test saving and loading
        settings.n_samples_max = 100
        ns.get_run_data(settings, 1, save=True, load=True,
                        check_loaded_settings=True, cache_dir=self.cache_dir)
        # test loading and checking settings
        ns.get_run_data(settings, 1, save=True, load=True,
                        check_loaded_settings=True, cache_dir=self.cache_dir)
        # test loading and checking settings when settings are not the same
        # this only works for changing a setting which dosnt affect the save
        # name
        settings.n_samples_max += 1
        ns.get_run_data(settings, 1, save=True, load=True,
                        check_loaded_settings=True, cache_dir=self.cache_dir)
        # test unexpected kwargs check
        self.assertRaises(TypeError, ns.get_run_data, self.settings, 1,
                          unexpected=1)
        # check returning None when keep_final_point is False and thread is
        # empty
        ns.generate_single_thread(self.settings, -10 ** -150,
                                  0, keep_final_point=False)
        # for checking with exact=True
        ns.z_importance(np.random.random(10), np.full((10), 5), exact=True)
        # for checking importance condition when the final point is one of the
        # ones with high importance
        ns.min_max_importance(np.full(2, 1), np.random.random((2, 3)),
                              settings)


if __name__ == '__main__':
    unittest.main()
