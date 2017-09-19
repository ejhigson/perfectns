#!/usr/bin/python
"""Contains the functions which perform nested sampling given input from settings. These are all called from within the wrapper function nested_sampling(settings)."""

import pns_settings
import pns.estimators as e
import pns.analysis_utils as au
import pns.parallelised_wrappers as pw
settings = pns_settings.PerfectNestedSamplingSettings()
estimator_list = [e.logzEstimator(), e.theta1Estimator(), e.theta1squaredEstimator(), e.theta1confEstimator(0.5), e.theta1confEstimator(0.84), e.theta1confEstimator(0.975), e.rconfEstimator(0.84)]

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

print("Standard NS Run")
run_list = pw.generate_runs(settings, 10)
stats, all_values = pw.func_on_runs(au.run_estimators, run_list, estimator_list)
print(stats)


print("Dynamic NS Run")
settings.dynamic_goal = 1
run_list = pw.generate_runs(settings, 10)
stats, all_values = pw.func_on_runs(au.run_estimators, run_list, estimator_list)
print(stats)
