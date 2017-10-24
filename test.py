#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import pandas as pd
import pns_settings
import pns.estimators as e
import pns.likelihoods as likelihoods
# import pns.results_generation as rg
import pns.analysis_utils as au
import pns.parallelised_wrappers as pw
settings = pns_settings.PerfectNestedSamplingSettings()
# settings.likelihood = likelihoods.exp_power(likelihood_scale=1, power=2)
# settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
pd.set_option('display.width', 200)

# settings
# --------
n_run = 2
settings.dynamic_goal = 1
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]
e_names = []
for est in estimator_list:
    e_names.append(est.name)

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

runs = pw.generate_runs(settings, n_run, parallelise=False)

lrxp_a = au.vstack_sort_array_list(runs[0][1])
logl_a = lrxp_a[:, 0]
lrxp_b = au.vstack_sort_array_list(runs[1][1])
logl_b = lrxp_b[:, 0]
nlive_a = au.get_nlive(runs[0][0], logl_a)
nlive_b = au.get_nlive(runs[1][0], logl_b)
nlive_merge_test = au.merge_nlive(logl_a, nlive_a, logl_b, nlive_b)

run_merge = [{'settings': settings.get_settings_dict()}, runs[0][1] + runs[1][1]]
run_merge[0]['settings']['nlive'] *= 2
run_merge[0]['settings']['nlive_1'] *= 2
run_merge[0]['settings']['nlive_2'] *= 2
run_merge[0]['thread_logl_min_max'] = runs[0][0]['thread_logl_min_max'] + runs[1][0]['thread_logl_min_max']
lrxp_m = au.vstack_sort_array_list(run_merge[1])
nlive_merge = au.get_nlive(run_merge[0], lrxp_m[:, 0])
# results = rg.get_bootstrap_results(n_run, n_simulate, estimator_list,
#                                    settings,
#                                    n_simulate_ci=n_simulate_ci,
#                                    cred_int=cred_int)
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
