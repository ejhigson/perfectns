#!/usr/bin/python
"""
Runs some tests of perfect_nested_sampling functionality.
"""

import pandas as pd
import PerfectNestedSampling.settings
import PerfectNestedSampling.estimators as e
import PerfectNestedSampling.likelihoods as likelihoods
import PerfectNestedSampling.nested_sampling as ns
import PerfectNestedSampling.results_tables as rt
import PerfectNestedSampling.priors as priors
import PerfectNestedSampling.analyse_run as ar
# import PerfectNestedSampling.parallelised_wrappers as pw
pd.set_option('display.width', 200)


print('\nTesting perfect_nested_sampling:')
print('--------------------------------\n')

print('Creating a PerfectNestedSamplingSettings object')
settings = PerfectNestedSampling.settings.PerfectNestedSamplingSettings()
print('Setting the likelihood, prior and number of dimensions')
print('(you can change the default settings in \
PerfectNestedSampling/settings.py)')
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
settings.prior = priors.gaussian(prior_scale=1)
settings.n_dim = 3

print('Perform standard nested sampling')
settings.dynamic_goal = None  # specifies standard nested sampling
standard_ns_run = ns.generate_ns_run(settings)
print('Perform dynamic nested sampling')
settings.dynamic_goal = 1  # dynamic nested sampling
dynamic_ns_run = ns.generate_ns_run(settings)
print('Calculate the log Bayesian evidence and some parameter \
estimation quantities')
estimator_list = [e.logzEstimator(),
                  e.paramMeanEstimator(),
                  e.paramSquaredMeanEstimator(),
                  e.paramCredEstimator(0.5),
                  e.paramCredEstimator(0.84)]
single_run_tests = e.get_true_estimator_values(estimator_list, settings)
single_run_tests.loc['standard run'] = ar.run_estimators(standard_ns_run,
                                                         estimator_list)
single_run_tests.loc['dynamic run'] = ar.run_estimators(dynamic_ns_run,
                                                        estimator_list)
print(single_run_tests)

print('\nCompare dynamic and standard nested sampling:')
print('---------------------------------------------\n')
settings.nlive_const = 100  # return to default nlive_const for multi-run tests
n_runs = 10


dynamic_tests = rt.get_dynamic_results(n_runs, [0, 1],
                                       estimator_list, settings,
                                       parallelise=True)
print(dynamic_tests)

print('\nCheck bootstrap error estimates:')
print('--------------------------------\n')

bootstrap_tests = rt.get_bootstrap_results(n_runs, 20,
                                           estimator_list, settings,
                                           n_simulate_ci=100,
                                           add_sim_method=True,
                                           n_run_ci=2,
                                           cred_int=0.95,
                                           ninit_sep=False,
                                           parallelise=True)
print(bootstrap_tests)
