#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import pandas as pd
import pns_settings
import pns.estimators as e
# import pns.analysis_utils as au
# import pns.parallelised_wrappers as pw
# import pns.maths_functions as mf
import pns.results_generation as rg
import pns.likelihoods as likelihoods
settings = pns_settings.PerfectNestedSamplingSettings()
pd.set_option('display.width', 200)

# Shared settings
# --------
settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
n_runs = 10
load = True
save = True
settings.n_dim = 3
settings.dynamic_goal = 1
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

# dynamic results
# ---------------
run_dynamic_results = True
dynamic_goals = [None, 0, 0.25, 1]
tuned_dynamic_ps = [False] * len(dynamic_goals)
if type(settings.likelihood).__name__ == "cauchy":
    dynamic_goals = [None, 0, 1, 1]
    tuned_dynamic_ps = [False] * (len(dynamic_goals) - 1) + [True]
if run_dynamic_results:
    print("Running dynamic results:")
    dynamic_results = rg.get_dynamic_results(n_runs, dynamic_goals,
                                             estimator_list, settings,
                                             tuned_dynamic_ps=tuned_dynamic_ps)
    print(dynamic_results)

# bootstrap results
# -----------------
run_bootstrap_results = True
n_simulate = 50
n_simulate_ci = n_simulate * 4
cred_int = 0.95
e_names = []
for est in estimator_list:
    e_names.append(est.name)
if run_bootstrap_results:
    print("Running bootstrap results:")
    bootstrap_results = rg.get_bootstrap_results(n_runs, n_simulate,
                                                 estimator_list, settings,
                                                 n_simulate_ci=n_simulate_ci,
                                                 cred_int=cred_int)
    print(bootstrap_results)
