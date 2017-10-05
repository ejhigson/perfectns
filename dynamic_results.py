#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import pandas as pd
import pns_settings
import pns.estimators as e
import pns.save_load_utils as slu
# import pns.analysis_utils as au
# import pns.parallelised_wrappers as pw
# import pns.maths_functions as mf
import pns.results_generation as rg
import pns.likelihoods as likelihoods
settings = pns_settings.PerfectNestedSamplingSettings()
pd.set_option('display.width', 200)

# Settings
# --------
# settings.likelihood = likelihoods.exp_power(likelihood_scale=1, power=2)
settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
n_repeats = 500
dynamic_goals = [None, 0, 0.25, 1]
tuned_dynamic_ps = [False] * len(dynamic_goals)
load_results = True
save_name = slu.data_save_name(settings, n_repeats, dynamic_test=dynamic_goals)
save_file = 'data/' + save_name + '.dat'
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1confEstimator(0.84),
                  # e.rconfEstimator(0.84)
                  e.theta1squaredEstimator()]
if type(settings.likelihood).__name__ == "cauchy":
    dynamic_goals = [None, 0, 1, 1]
    tuned_dynamic_ps = [False] * (len(dynamic_goals) - 1) + [True]
e_names = []
for est in estimator_list:
    e_names.append(est.name)

print("True est values")
print(e_names)
print(e.check_estimator_values(estimator_list, settings))


if load_results:
    try:
        results = pd.read_pickle('data/' + save_name + '.dat')
        print("Loading results from file:\n" + save_file)
    except OSError:
        load_results = False
if not load_results:
    results = rg.get_dynamic_results(n_repeats, dynamic_goals, estimator_list,
                                     settings,
                                     tuned_dynamic_ps=tuned_dynamic_ps)
    results.to_pickle(save_file)
    print("Results saved to:\n" + save_file)

print(results)
