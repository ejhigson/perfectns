#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import pandas as pd
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
settings.dynamic_goal = 1
settings.n_dim = 3
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]
e_names = []
for est in estimator_list:
    e_names.append(est.name)

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

print("Standard NS Run")
s_run_list = pw.get_run_data(settings, n_repeats)
# get std from repeats
rep_values = pw.func_on_runs(au.run_estimators, s_run_list, estimator_list)
rep_df = mf.get_df_row_summary(rep_values, e_names)
results = pd.DataFrame([rep_df.loc['std'], rep_df.loc['std_unc']],
                       index=['repeats std', 'repeats std_unc'])
# get std from bootstrap estimate
bs_values = pw.func_on_runs(au.run_std_bootstrap, s_run_list, estimator_list,
                            n_simulate=n_simulate)
bs_df = mf.get_df_row_summary(bs_values, e_names)
results.loc['bs std'] = bs_df.loc['mean']
results.loc['bs std_unc'] = bs_df.loc['mean_unc']
# get std from simulation estimate
sim_values = pw.func_on_runs(au.run_std_simulate, s_run_list, estimator_list,
                             n_simulate=n_simulate)
sim_df = mf.get_df_row_summary(sim_values, e_names)
results.loc['sim std'] = sim_df.loc['mean']
results.loc['sim std_unc'] = sim_df.loc['mean_unc']
results = mf.df_unc_rows_to_cols(results, ["repeats std", "bs std", "sim std"])
print(results)
# # print("Dynamic NS Run")
# import pns.parallelised_wrappers as pw
# import pns.analysis_utils as au
# import pns.maths_functions as mf
# settings.dynamic_goal = 1
# d_run_list = pw.get_run_data(settings, 10)
# values = pw.func_on_runs(au.run_estimators, d_run_list, estimator_list)
# d_df = mf.get_df_row_summary(values, e_names)
# print(d_df)

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
