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
import PerfectNestedSampling.maths_functions as mf
import PerfectNestedSampling.parallelised_wrappers as pw
pd.set_option('display.width', 200)


print('-----------------------')
print('Perfect Nested Sampling')
print('-----------------------')

print('\nGenerate nested sampling runs:\n')

print('First create a PerfectNestedSamplingSettings object.')
settings = PerfectNestedSampling.settings.PerfectNestedSamplingSettings()
print('Set the likelihood, prior and number of dimensions; you can change \
the default settings in settings.py')
print('Here we use a 3d spherically symmetric Gaussian likelihood with \
sigma=1 and a Gaussian prior with sigma=10')
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
settings.prior = priors.gaussian(prior_scale=10)
settings.n_dim = 3

print('Perform standard nested sampling.')
settings.dynamic_goal = None  # specifies standard nested sampling
standard_ns_run = ns.generate_ns_run(settings)
print('Perform dynamic nested sampling.')
settings.dynamic_goal = 0.5  # dynamic nested sampling
dynamic_ns_run = ns.generate_ns_run(settings)

print('\nEvidence and parameter estimation calculations:\n')

print('We can now perform evidence calculation and parameter estimation using \
the nested sampling runs we have just generated.')
estimator_list = [e.logzEstimator(),
                  e.paramMeanEstimator(),
                  e.paramSquaredMeanEstimator(),
                  e.paramCredEstimator(0.5),
                  e.paramCredEstimator(0.84)]
print('Here we calculate:')
print('\t- the log evidence')
print('\t- the mean of the first parameter theta1')
print('\t- second moment of theta1')
print('\t- the median of theta1')
print('\t- and 85% one-tailed credibile interval on theta1')
print('By symmetry the posterior distribituon for any of the n_dim parameters \
is the same.')
print('In these cases the posterior distribution is known so we can calculate \
the true values analytically and check our results.')
single_run_tests = e.get_true_estimator_values(estimator_list, settings)
single_run_tests.loc['standard run'] = ar.run_estimators(standard_ns_run,
                                                         estimator_list)
single_run_tests.loc['dynamic run'] = ar.run_estimators(dynamic_ns_run,
                                                        estimator_list)
print(single_run_tests)

print('\nEstimate calculation errors:\n')

print('We can estimate the numerical uncertainties on these results by \
caclulating the standard deviation of the sampling errors distributions for \
each run using resampling.')

single_run_tests.loc['standard unc'] = ar.run_std_bootstrap(standard_ns_run,
                                                            estimator_list,
                                                            n_simulate=200)
single_run_tests.loc['dynamic unc'] = ar.run_std_bootstrap(dynamic_ns_run,
                                                           estimator_list,
                                                           n_simulate=200)

print(single_run_tests.loc[['standard unc', 'dynamic unc']])

print('The calculation results should agree with the true values to within \
+- 2 standard deviations of the calculated sampling errors distributions \
most of the time.')

print('\nGenerate and analyse many runs in parallel:\n')

n_runs = 100
print('Generate ' + str(n_runs) + ' runs in parallel using the \
concurrent.futures module.')
run_list = pw.generate_runs(settings, n_runs, parallelise=True)
print('Calculate estimators for each run in parallel.')
values = pw.func_on_runs(ar.run_estimators, run_list, estimator_list,
                         parallelise=True)
print('Calculate the mean and standard deviation of calculation results.')
estimator_names = [est.name for est in estimator_list]
multi_run_tests = mf.get_df_row_summary(values, estimator_names)
print(multi_run_tests.loc[['mean', 'std']])


print('\nCompare dynamic and standard nested sampling performance:')
print('---------------------------------------------------------')

print(rt.get_dynamic_results.__doc__)
dynamic_tests = rt.get_dynamic_results(n_runs, [0, 1],
                                       estimator_list, settings,
                                       parallelise=True)
print(dynamic_tests)

print('\nCompare bootstrap error estimates to observed results distribution:')
print('-------------------------------------------------------------------')

settings.dynamic_goal = 0.5
print(rt.get_bootstrap_results.__doc__)
bootstrap_tests = rt.get_bootstrap_results(n_runs, 20,
                                           estimator_list, settings,
                                           n_simulate_ci=100,
                                           add_sim_method=True,
                                           n_run_ci=2,
                                           cred_int=0.95,
                                           ninit_sep=False,
                                           parallelise=True)
print(bootstrap_tests)
