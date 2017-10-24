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


# Which results do you want to run?
# ---------------------------------
run_bootstrap_results = False
run_dynamic_results = True
# Shared settings
# --------
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
n_runs = 100
load = True
save = True
settings.n_dim = 3
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

# dynamic results
# ---------------
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
n_simulate = 200
n_simulate_ci = 1000
n_run_ci = 100
add_sim_method = False
settings.dynamic_goal = None
cred_int = 0.95
e_names = []
for est in estimator_list:
    e_names.append(est.name)
if run_bootstrap_results:
    print("Running bootstrap results:")
    bootstrap_results = rg.get_bootstrap_results(n_runs, n_simulate,
                                                 estimator_list, settings,
                                                 n_simulate_ci=n_simulate_ci,
                                                 add_sim_method=add_sim_method,
                                                 n_run_ci=n_run_ci,
                                                 cred_int=cred_int)
    print(bootstrap_results)
