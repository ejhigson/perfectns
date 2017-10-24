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
import pns.parallelised_wrappers as pw
settings = pns_settings.PerfectNestedSamplingSettings()
# settings.likelihood = likelihoods.exp_power(likelihood_scale=1, power=2)
# settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
pd.set_option('display.width', 200)

# settings
# --------
n_run = 4
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

