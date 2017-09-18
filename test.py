#!/usr/bin/python
"""Contains the functions which perform nested sampling given input from settings. These are all called from within the wrapper function nested_sampling(settings)."""

import pns.nested_sampling as ns
import pns_settings
import pns.dynamic_nested_sampling as dns
import pns.estimators as e
import pns.analysis_utils as au
settings = pns_settings.PerfectNestedSamplingSettings()
estimator_list = [e.logzEstimator(), e.theta1Estimator(), e.theta1squaredEstimator(), e.theta1confEstimator(0.5), e.theta1confEstimator(0.84), e.theta1confEstimator(0.975), e.rconfEstimator(0.84)]

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

print("Standard NS Run")
r = ns.generate_standard_run(10, settings)


print("Dynamic NS Run")
d = dns.generate_dynamic_run(10, 1, settings)

print("Get some results")

ests = au.run_estimators(d, estimator_list, settings)
