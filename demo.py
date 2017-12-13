#!/usr/bin/env python
"""
Demonstrate the PerfectNS module.
"""

import pandas as pd
import PerfectNS.settings
import PerfectNS.estimators as e
import PerfectNS.likelihoods as likelihoods
import PerfectNS.nested_sampling as ns
import PerfectNS.results_tables as rt
import PerfectNS.priors as priors
import PerfectNS.analyse_run as ar
import PerfectNS.maths_functions as mf
import PerfectNS.parallelised_wrappers as pw
pd.set_option('display.width', 200)


print('-----------------------')
print('Perfect Nested Sampling')
print('-----------------------')

print('\nGenerate nested sampling runs:\n')

print('First create a PerfectNSSettings object.')
SETTINGS = PerfectNS.settings.PerfectNSSettings()
print('Set the likelihood, prior and number of dimensions; you can change \
the default settings in settings.py')
SETTINGS.likelihood = likelihoods.gaussian(likelihood_scale=1)
SETTINGS.prior = priors.gaussian(prior_scale=10)
SETTINGS.n_dim = 10
print('Here we use a 10d spherically symmetric Gaussian likelihood with \
sigma=1 and a Gaussian prior with sigma=10')

print('Perform standard nested sampling.')
SETTINGS.dynamic_goal = None  # specifies standard nested sampling
standard_ns_run = ns.generate_ns_run(SETTINGS)
print('Perform dynamic nested sampling.')
SETTINGS.dynamic_goal = 0.5  # dynamic nested sampling
dynamic_ns_run = ns.generate_ns_run(SETTINGS)

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
print('\t- and 84% one-tailed credible interval on theta1')
print('By symmetry the posterior distribution for any of the n_dim parameters \
is the same.')
print('In this case we can calculate the posterior distribution and the true \
values of these quantities given the posterior analytically and check our \
results.')
single_run_tests = e.get_true_estimator_values(estimator_list, SETTINGS)
single_run_tests.loc['standard run'] = ar.run_estimators(standard_ns_run,
                                                         estimator_list)
single_run_tests.loc['dynamic run'] = ar.run_estimators(dynamic_ns_run,
                                                        estimator_list)
print(single_run_tests)

print('\nEstimate calculation errors:\n')

print('We can estimate the numerical uncertainties on these results by \
calculating the standard deviation of the sampling errors distributions for \
each run using resampling.')

single_run_tests.loc['standard unc'] = ar.run_std_bootstrap(standard_ns_run,
                                                            estimator_list,
                                                            n_simulate=200)
single_run_tests.loc['dynamic unc'] = ar.run_std_bootstrap(dynamic_ns_run,
                                                           estimator_list,
                                                           n_simulate=200)
print(single_run_tests.loc[['standard unc', 'dynamic unc']])

print('\nGenerate and analyse many runs in parallel:\n')

n_runs = 100
print('Generate ' + str(n_runs) + ' runs in parallel using the \
concurrent.futures module.')
run_list = pw.generate_runs(SETTINGS, n_runs, parallelise=True)
print('Calculate estimators for each run in parallel.')
values = pw.func_on_runs(ar.run_estimators, run_list, estimator_list,
                         parallelise=True)
print('Calculate the mean and standard deviation of calculation results.')
estimator_names = [est.name for est in estimator_list]
multi_run_tests = mf.get_df_row_summary(values, estimator_names)
print(multi_run_tests.loc[['mean', 'std']])


print('\nCompare dynamic and standard nested sampling performance:')
print('---------------------------------------------------------')

n_runs = 200
print(rt.get_dynamic_results.__doc__)
print('Lets now compare the performance of dynamic and standard nested \
sampling, using the 10d Gaussian likelihood and prior.')
print('This is the same code that was used for Table 1 of the dynamic nested \
sampling paper, although we only use ' + str(n_runs) + ' runs instead of \
5000.')
print('Tables 1, 2 and 3 can also be replicated by changing the settings.')
print('See the paper for more details.')
SETTINGS.likelihood = likelihoods.gaussian(likelihood_scale=1)
SETTINGS.prior = priors.gaussian(prior_scale=10)
SETTINGS.n_dim = 10
estimator_list = [e.logzEstimator(),
                  e.paramMeanEstimator(),
                  e.paramCredEstimator(0.5),
                  e.paramCredEstimator(0.84)]
dynamic_tests = rt.get_dynamic_results(n_runs, [0, 1],
                                       estimator_list, SETTINGS,
                                       parallelise=True)
print(dynamic_tests)

print('Looking at the final row of the table, you should see that dynamic \
nested sampling targeted at parameter estimation (dynamic goal=1) has an \
efficiency gain (equivalent computational speedup) for parameter estimation \
(columns other than logz) of factor of around 3 to 4 compared to standard \
nested sampling')

print('\nCompare bootstrap error estimates to observed results distribution:')
print('-------------------------------------------------------------------')

print(rt.get_bootstrap_results.__doc__)

n_runs = 100
print('Lets check if the bootstrap estimates of parameter estimation sampling \
errors are accurate, using a 3d Gaussian likelihood and Gaussian prior.')
print('This is the same code that was used for Table 5 of the dynamic nested \
sampling paper, although we only use ' + str(n_runs) + ' runs instead of \
5000.')
print('See the paper for more details.')

SETTINGS.likelihood = likelihoods.gaussian(likelihood_scale=1)
SETTINGS.prior = priors.gaussian(prior_scale=10)
SETTINGS.n_dim = 3
estimator_list = [e.paramMeanEstimator(),
                  e.paramSquaredMeanEstimator(),
                  e.paramCredEstimator(0.5),
                  e.paramCredEstimator(0.84)]
SETTINGS.dynamic_goal = 1

bootstrap_tests = rt.get_bootstrap_results(n_runs, 200,
                                           estimator_list, SETTINGS,
                                           n_run_ci=int(n_runs / 5),
                                           n_simulate_ci=1000,
                                           add_sim_method=False,
                                           cred_int=0.95,
                                           ninit_sep=False,
                                           parallelise=True)
print(bootstrap_tests)


print('You should see that the ratio of the bootstrap error estimates to \
the standard deviation of results (row 4) has values close to 1.')
