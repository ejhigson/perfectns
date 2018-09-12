#!/usr/bin/env python
"""
Test the perfectns package installation.
"""

import os
import shutil
import unittest
import warnings
import numpy as np
import numpy.testing
import matplotlib
import nestcheck.ns_run_utils
import perfectns.settings
import perfectns.estimators as e
import perfectns.cached_gaussian_prior
import perfectns.likelihoods as likelihoods
import perfectns.nested_sampling as ns
import perfectns.results_tables as rt
import perfectns.maths_functions
import perfectns.priors as priors
import perfectns.plots

ESTIMATOR_LIST = [e.LogZ(),
                  e.Z(),
                  e.ParamMean(),
                  e.ParamSquaredMean(),
                  e.ParamCred(0.5),
                  e.ParamCred(0.84),
                  e.RMean(from_theta=True),
                  e.RCred(0.84, from_theta=True)]
TEST_CACHE_DIR = 'cache_tests'
TEST_DIR_EXISTS_MSG = ('Directory ' + TEST_CACHE_DIR + ' exists! Tests use '
                       'this dir to check caching then delete it afterwards, '
                       'so the path should be left empty.')


class TestNestedSampling(unittest.TestCase):

    def setUp(self):
        """Check TEST_CACHE_DIR does not already exist."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG

    def tearDown(self):
        """Remove any caches created by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_nestcheck_run_format(self):
        """
        Check perfectns runs are compatable with the nestcheck run format
        (excepting their additional 'logx' and 'r' keys).
        """
        settings = get_minimal_settings()
        for dynamic_goal in [None, 0, 0.5, 1]:
            settings.dynamic_goal = dynamic_goal
            run = ns.generate_ns_run(settings)
            del run['logx']
            del run['r']
            del run['settings']
            del run['random_seed']
            try:
                nestcheck.ns_run_utils.check_ns_run(run)
            except AttributeError:
                # check_ns_run moved from nestcheck.data_processing to
                # nestcheck.ns_run_utils in v0.1.8, so this is needed to
                # maintain compatibility with earlier versions
                pass

    def test_get_run_data_caching(self):
        settings = get_minimal_settings()
        settings.dynamic_goal = None
        settings.n_samples_max = 100
        ns.get_run_data(settings, 1, save=True, load=True,
                        check_loaded_settings=True, cache_dir=TEST_CACHE_DIR)
        # test loading and checking settings
        ns.get_run_data(settings, 1, save=True, load=True,
                        check_loaded_settings=True, cache_dir=TEST_CACHE_DIR)
        # test loading and checking settings when settings are not the same
        # this only works for changing a setting which dosnt affect the save
        # name
        settings.dynamic_goal = 0
        ns.get_run_data(settings, 1, save=True, load=True,
                        check_loaded_settings=True, cache_dir=TEST_CACHE_DIR)
        settings.n_samples_max += 1
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            ns.get_run_data(
                settings, 1, save=True, load=True, check_loaded_settings=True,
                cache_dir=TEST_CACHE_DIR)
            self.assertEqual(len(war), 1)

    def test_get_run_data_unexpected_kwarg(self):
        settings = get_minimal_settings()
        self.assertRaises(TypeError, ns.get_run_data, settings, 1,
                          unexpected=1)

    def test_no_point_thread(self):
        """
        Check generate_single_thread returns None when keep_final_point is
        False and thread is empty.
        """
        settings = get_minimal_settings()
        self.assertIsNone(ns.generate_single_thread(
            settings, -(10 ** -150), 0, keep_final_point=False))

    def test_exact_z_importance(self):
        """Check z_importance with exact=True."""
        imp_exact = ns.z_importance(np.asarray([1., 2.]), np.asarray([5, 5]),
                                    exact=True)
        self.assertEqual(imp_exact[0], 1.)
        self.assertAlmostEqual(imp_exact[1], 0.99082569, places=6)

    def test_min_max_importance(self):
        """
        Check importance condition when the final point is one of the
        ones with high importance.
        """
        settings = get_minimal_settings()
        samples = np.random.random((2, 3))
        loglmm, logxmm = ns.min_max_importance(np.full(2, 1), samples,
                                               settings)
        self.assertEqual(loglmm[1], samples[-1, 0])
        self.assertEqual(logxmm[1], samples[-1, 2])

    def test_tuned_p_importance(self):
        theta = np.random.random((5, 1))
        w_rel = np.full(5, 1)
        imp = np.abs(theta - np.mean(theta))[:, 0]
        imp /= imp.max()
        self.assertTrue(np.array_equal(
            ns.p_importance(theta, w_rel, tuned_dynamic_p=True), imp))


class TestEstimators(unittest.TestCase):

    """
    Test estimators: largely checking the get_true_estimator_values output
    as the functions used for analysing nested sampling runs are mostly
    imported from nestcheck which has its own tests.
    """

    def setUp(self):
        """Get some settings for the get_true_estimator_values tests."""
        self.settings = get_minimal_settings()

    def test_true_logz_value(self):
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.LogZ(), self.settings),
            -6.4529975832506050, places=10)

    def test_true_z_value(self):
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.Z(), self.settings),
            1.5757915157613399e-03, places=10)

    def test_true_param_mean_value(self):
        self.assertEqual(
            e.get_true_estimator_values(e.ParamMean(), self.settings), 0)

    def test_true_param_mean_squared_value(self):
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.ParamSquaredMean(), self.settings),
            9.9009851517647807e-01, places=10)

    def test_true_param_cred_value(self):
        self.assertEqual(
            e.get_true_estimator_values(e.ParamCred(0.5), self.settings), 0)
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.ParamCred(0.84), self.settings),
            9.8952257789120635e-01, places=10)

    def test_true_r_mean_value(self):
        self.assertAlmostEqual(
            e.get_true_estimator_values(e.RMean(), self.settings),
            1.2470645289408879e+00, places=10)

    def test_true_r_cred_value(self):
        self.assertTrue(np.isnan(
            e.get_true_estimator_values(e.RCred(0.84), self.settings)))
        # Test with a list argument as well to cover list version of true
        # get_true_estimator_values
        self.assertTrue(np.isnan(
            e.get_true_estimator_values([e.RCred(0.84)], self.settings)[0]))

    def test_r_not_from_theta(self):
        run_dict_temp = {'theta': np.full((2, 2), 1),
                         'r': np.full((2,), np.sqrt(2)),
                         'logl': np.full((2,), 0.),
                         'nlive_array': np.full((2,), 5.),
                         'settings': {'dims_to_sample': 2, 'n_dim': 2}}
        self.assertAlmostEqual(e.RMean(from_theta=False)(
            run_dict_temp, logw=None), np.sqrt(2), places=10)
        self.assertAlmostEqual(e.RCred(0.84, from_theta=False)(
            run_dict_temp, logw=None), np.sqrt(2), places=10)

    def test_count_samples(self):
        self.assertEqual(e.CountSamples()({'logl': np.zeros(10)}), 10)


class TestMathsFunctions(unittest.TestCase):

    def test_analytic_logx_terminate(self):
        """Check None is returned when the likelihood is not set up."""
        settings = get_minimal_settings()
        settings.likelihood = likelihoods.ExpPower(2)
        self.assertIsNone(
            perfectns.maths_functions.analytic_logx_terminate(settings))

    def test_nsphere_sampling(self):
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


class TestSettings(unittest.TestCase):

    def test_settings_unexpected_arg(self):
        self.assertRaises(
            TypeError, perfectns.settings.PerfectNSSettings, unexpected=0)

    def test_settings_save_name(self):
        settings = perfectns.settings.PerfectNSSettings()
        settings.dynamic_goal = 1
        settings.nbatch = 2
        settings.nlive_const = None
        settings.tuned_dynamic_p = True
        settings.n_samples_max = 100
        settings.dynamic_fraction = 0.8
        settings.dims_to_sample = 2
        self.assertIsInstance(settings.save_name(), str)

    def test_settings_unexpected_attr(self):
        settings = perfectns.settings.PerfectNSSettings()
        self.assertRaises(TypeError, settings.__setattr__, 'unexpected', 1)


class TestLikelihoods(unittest.TestCase):

    def test_standard_ns_exp_power_likelihood_gaussian_prior(self):
        """Check the exp_power likelihood, as well as some functions in
        analyse_run."""
        settings = get_minimal_settings()
        settings.likelihood = likelihoods.ExpPower(likelihood_scale=1,
                                                   power=2)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.logz_analytic()
        ns_run = ns.generate_ns_run(settings)
        values = nestcheck.ns_run_utils.run_estimators(ns_run, ESTIMATOR_LIST)
        self.assertFalse(np.any(np.isnan(values)))

    def test_standard_ns_cauchy_likelihood_gaussian_prior(self):
        """Check the Cauchy likelihood."""
        settings = get_minimal_settings()
        settings.likelihood = likelihoods.Cauchy(likelihood_scale=1)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.logz_analytic()
        ns_run = ns.generate_ns_run(settings)
        values = nestcheck.ns_run_utils.run_estimators(ns_run, ESTIMATOR_LIST)
        self.assertFalse(np.any(np.isnan(values)))


class TestPriors(unittest.TestCase):

    def setUp(self):
        """Check TEST_CACHE_DIR does not already exist."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG

    def tearDown(self):
        """Remove any caches created by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_standard_ns_gaussian_likelihood_uniform_prior(self):
        """Check the uniform prior."""
        settings = get_minimal_settings()
        settings.prior = priors.Uniform(prior_scale=10)
        self.assertAlmostEqual(
            settings.logx_given_logl(settings.logl_given_logx(-1.0)),
            -1.0, places=12)
        settings.logz_analytic()
        ns_run = ns.generate_ns_run(settings)
        values = nestcheck.ns_run_utils.run_estimators(ns_run, ESTIMATOR_LIST)
        self.assertFalse(np.any(np.isnan(values)))

    def test_cached_gaussian_prior(self):
        """Check the cached_gaussian prior."""
        settings = get_minimal_settings()
        self.assertRaises(
            TypeError, priors.GaussianCached,
            prior_scale=10, unexpected=0)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            settings.prior = priors.GaussianCached(
                prior_scale=10, save_dict=True, n_dim=settings.n_dim,
                cache_dir=TEST_CACHE_DIR,
                interp_density=10, logx_min=-30)
            self.assertEqual(len(war), 1)
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
        values = nestcheck.ns_run_utils.run_estimators(ns_run, ESTIMATOR_LIST)
        self.assertFalse(np.any(np.isnan(values)))
        # check the argument options and messages for interp_r_logx_dict
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            self.assertRaises(
                TypeError, perfectns.cached_gaussian_prior.interp_r_logx_dict,
                2000, 10, logx_min=-100, interp_density=1, unexpected=0)
            self.assertEqual(len(war), 1)
        self.assertRaises(
            TypeError, perfectns.cached_gaussian_prior.interp_r_logx_dict,
            200, 10, logx_min=-100, interp_density=1, unexpected=0)


class TestPlotting(unittest.TestCase):

    def test_plot_dynamic_nlive(self):
        settings = get_minimal_settings()
        fig = perfectns.plots.plot_dynamic_nlive(
            [None, 0, 1, 1], settings, n_run=2,
            tuned_dynamic_ps=[False, False, False, True],
            save=False, load=False)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Test ymax and the fallback for normalising analytic lines when the
        # dynamic goal which is meant to mirror them is not present
        fig = perfectns.plots.plot_dynamic_nlive(
            [None], settings, n_run=2,
            save=False, load=False,
            tuned_dynamic_ps=[True], ymax=1000)

    def test_plot_parameter_logx_diagram(self):
        settings = get_minimal_settings()
        for ftheta in [e.ParamMean(), e.ParamSquaredMean(), e.RMean()]:
            fig = perfectns.plots.plot_parameter_logx_diagram(
                settings, ftheta, x_points=50, y_points=50)
            self.assertIsInstance(fig, matplotlib.figure.Figure)
        # Test warning for estimators without CDF
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            perfectns.plots.cdf_given_logx(e.LogZ(), np.zeros(1), np.zeros(1),
                                           settings)
            self.assertEqual(len(war), 1)
        # Test unexpected kwargs check
        self.assertRaises(
            TypeError, perfectns.plots.plot_parameter_logx_diagram,
            settings, e.ParamMean(), x_points=50, y_points=50,
            unexpected=0)

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


class TestDynamicResultsTables(unittest.TestCase):

    def setUp(self):
        """Check TEST_CACHE_DIR does not already exist."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG

    def tearDown(self):
        """Remove any caches created by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_dynamic_results_table_values(self):
        """
        Test generating a table comparing dynamic and standard nested sampling;
        this covers a lot of the perfectns package's functionality.

        Tests of the expected values relies on default seeding of runs in
        get_run_data using numpy.random.seed - this should be stable over
        different platforms but may be worth checking if errors occur.
        """
        settings = get_minimal_settings()
        n_run = 5
        dynamic_goals = [0, 0.25, 1, 1]
        tuned_dynamic_ps = [False, False, False, True]
        dynamic_table = rt.get_dynamic_results(
            n_run, dynamic_goals, ESTIMATOR_LIST, settings, load=True,
            save=True, cache_dir=TEST_CACHE_DIR,
            parallel=True, tuned_dynamic_ps=tuned_dynamic_ps)
        # Check merged dynamic results function
        merged_df = rt.merged_dynamic_results(
            [(settings.n_dim, settings.prior.prior_scale)],
            [settings.likelihood], settings, ESTIMATOR_LIST,
            dynamic_goals=dynamic_goals, n_run=n_run,
            cache_dir=TEST_CACHE_DIR, tuned_dynamic_ps=tuned_dynamic_ps,
            load=True, save=False)
        self.assertTrue(np.array_equal(
            merged_df.values, dynamic_table.values))
        # Check numerical values in dynamic_table
        self.assertFalse(np.any(np.isnan(dynamic_table.values)))
        # Check the values for one column (those for RMean)
        expected_rmean_vals = np.asarray(
            [1.05159345, 0.05910616, 1.09315952, 0.08192338, 1.14996638, 0.11357112,
             1.24153945, 0.09478196, 1.24436994, 0.07220817, 0.13216539, 0.04672752,
             0.18318625, 0.06476612, 0.25395275, 0.08978585, 0.21193890, 0.07493172,
             0.16146238, 0.05708557, 0.52053477, 0.52053477, 0.27085052, 0.27085052,
             0.38887867, 0.38887867, 0.67002776, 0.67002776])
        numpy.testing.assert_allclose(
            dynamic_table[e.RMean(from_theta=True).latex_name].values,
            expected_rmean_vals, rtol=1e-7,
            err_msg=('this relies on numpy.random.seed being consistent - '
                     'this should be true but is perhaps worth checking for '
                     'your platform.'))

    def test_dynamic_results_table_unexpected_kwargs(self):
        settings = get_minimal_settings()
        # Run some of the code in merged_dynamic_results which is missed with
        # different options
        self.assertRaises(TypeError, rt.merged_dynamic_results, [(1000, 10)],
                          [likelihoods.ExpPower()],
                          settings, ESTIMATOR_LIST, unexpected=1)


class TestBootstrapResultsTables(unittest.TestCase):

    def setUp(self):
        """Check TEST_CACHE_DIR does not already exist."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG

    def tearDown(self):
        """Remove any caches created by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_bootstrap_results_table_values(self):
        """
        Generate a table showing sampling error estimates using the bootstrap
        method.

        As the numerical values produced are stochastic we just test that the
        function runs ok and does not produce NaN values - this should be
        sufficient.
        """
        np.random.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            bs_df = rt.get_bootstrap_results(
                5, 10, ESTIMATOR_LIST, get_minimal_settings(), n_run_ci=2,
                n_simulate_ci=10, add_sim_method=True, cred_int=0.95, load=True,
                save=True, cache_dir=TEST_CACHE_DIR, ninit_sep=True,
                parallel=False)
        # Check numerical values in dynamic_table:
        # The first row of the table contains analytic calculations of the
        # estimators' values given the likelihood and prior which have already
        # been tested in test_dynamic_results_table.
        # None of the other values in the table should be NaN:
        self.assertFalse(np.any(np.isnan(bs_df.values[1:, :])))
        # Check the values for one column (those for RMean)
        expected_rmean_vals = np.asarray(
            [1.05159345e+00, 5.91061598e-02, 1.32165391e-01, 4.67275222e-02,
             9.74212313e-01, 3.95418293e-01, 4.45773150e+01, 1.57604609e+01,
             1.26559404e+00, 4.81565225e-01, 3.14517691e+01, 1.11198796e+01,
             1.44503362e+00, 1.20619710e-01, 6.00000000e+01, 1.00000000e+02])
        numpy.testing.assert_allclose(
            bs_df[e.RMean(from_theta=True).latex_name].values,
            expected_rmean_vals, rtol=1e-7,
            err_msg=('this relies on numpy.random.seed being consistent - '
                     'this should be true but is perhaps worth checking for '
                     'your platform.'))

    def test_bootstrap_results_table_unexpected_kwargs(self):
        settings = get_minimal_settings()
        self.assertRaises(TypeError, rt.get_bootstrap_results, 3, 10,
                          ESTIMATOR_LIST, settings, unexpected=1)


# Helper functions
# ----------------


def get_minimal_settings():
    """
    Get a perfectns settings object with a minimal number of live points so
    that tests run quickly.
    """
    settings = perfectns.settings.PerfectNSSettings()
    settings.dims_to_sample = 2
    settings.n_dim = 2
    settings.nlive_const = 5
    settings.ninit = 2
    settings.dynamic_goal = None
    return settings


if __name__ == '__main__':
    unittest.main()
