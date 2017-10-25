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
import pns.priors as priors
settings = pns_settings.PerfectNestedSamplingSettings()
pd.set_option('display.width', 200)


# Settings
# --------
settings.prior = priors.gaussian(10)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
n_runs = 6
load = True
save = True
settings.n_dim = 10
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]
dynamic_goals = [None, 0, 0.25, 1]
tuned_dynamic_ps = [False] * len(dynamic_goals)
if type(settings.likelihood).__name__ == "cauchy":
    dynamic_goals = [None, 0, 1, 1]
    tuned_dynamic_ps = [False] * (len(dynamic_goals) - 1) + [True]

# Run program
# -----------
print("True est values")
print(e.check_estimator_values(estimator_list, settings))
print("Running dynamic results:")
dynamic_results = rg.get_dynamic_results(n_runs, dynamic_goals,
                                         estimator_list, settings,
                                         tuned_dynamic_ps=tuned_dynamic_ps,
                                         parallelise=True)
print(dynamic_results)
