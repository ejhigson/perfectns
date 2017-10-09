#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import pandas as pd
import numpy as np
import pns_settings
import pns.estimators as e
import pns.analysis_utils as au
import pns.parallelised_wrappers as pw
import pns.maths_functions as mf
import pns.likelihoods as likelihoods
settings = pns_settings.PerfectNestedSamplingSettings()
# settings.likelihood = likelihoods.exp_power(likelihood_scale=1, power=2)
# settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
pd.set_option('display.width', 200)

# settings
# --------
n_repeats = 2000
n_simulate = 50
n_simulate_ci = n_simulate * 4
settings.dynamic_goal = 1
settings.n_dim = 3
cred_int = 0.95
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]
e_names = []
for est in estimator_list:
    e_names.append(est.name)

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

run_list = pw.get_run_data(settings, n_repeats)
# get std from repeats
rep_values = pw.func_on_runs(au.run_estimators, run_list, estimator_list)
rep_df = mf.get_df_row_summary(rep_values, e_names)
# results = pd.DataFrame([rep_df.loc['std'], rep_df.loc['std_unc']],
#                        index=['repeats std', 'repeats std_unc'])
results = rep_df.set_index('repeats ' + rep_df.index.astype(str))
# get bootstrap std estimate
bs_values = pw.func_on_runs(au.run_std_bootstrap, run_list,
                            estimator_list, n_simulate=n_simulate)
bs_df = mf.get_df_row_summary(bs_values, e_names)
results.loc['bs std'] = bs_df.loc['mean']
results.loc['bs std_unc'] = bs_df.loc['mean_unc']
# get bootstrap CI estimates
bs_cis = pw.func_on_runs(au.run_ci_bootstrap, run_list, estimator_list,
                         n_simulate=n_simulate_ci, cred_int=cred_int)
bs_ci_df = mf.get_df_row_summary(bs_cis, e_names)
results.loc['bs ' + str(cred_int) + ' CI'] = bs_ci_df.loc['mean']
results.loc['bs ' + str(cred_int) + ' CI_unc'] = bs_ci_df.loc['mean_unc']
# add +- 1 std coverage
max_value = rep_df.loc['mean'].values + results.loc['bs std'].values
min_value = rep_df.loc['mean'].values - results.loc['bs std'].values
coverage = np.zeros(rep_values.shape[0])
for i, _ in enumerate(coverage):
    ind = np.where((rep_values[i, :] > min_value[i]) &
                   (rep_values[i, :] < max_value[i]))
    coverage[i] = ind[0].shape[0] / rep_values.shape[1]
results.loc['bs +-1std cov'] = coverage
# add conf interval coverage
max_value = results.loc['bs ' + str(cred_int) + ' CI'].values
ci_coverage = np.zeros(rep_values.shape[0])
for i, _ in enumerate(coverage):
    ind = np.where(rep_values[i, :] < max_value[i])
    ci_coverage[i] = ind[0].shape[0] / rep_values.shape[1]
results.loc['bs ' + str(cred_int) + ' CI cov'] = ci_coverage
# # get std from simulation estimate
# sim_values = pw.func_on_runs(au.run_std_simulate, run_list, estimator_list,
#                              n_simulate=n_simulate)
# sim_df = mf.get_df_row_summary(sim_values, e_names)
# results.loc['sim std'] = sim_df.loc['mean']
# results.loc['sim std_unc'] = sim_df.loc['mean_unc']
results = mf.df_unc_rows_to_cols(results)
print(results)

# load_results = True
# save_name = slu.data_save_name(settings, n_repeats)
# save_file = 'data/' + save_name + '.dat'
#
# if load_results:
#     try:
#         dr = pd.read_pickle('data/' + save_name + '.dat')
#         print("Loading results from file:\n" + save_file)
#     except OSError:
#         load_results = False
# if not load_results:
#     dr = rg.get_dynamic_results(n_repeats, dynamic_goals, estimator_list,
#                                 settings, tuned_dynamic_ps=tuned_dynamic_ps)
#     dr.to_pickle(save_file)
#     print("Results saved to:\n" + save_file)
#
# print(dr)
