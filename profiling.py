#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import cProfile
import pandas as pd
import pns_settings
# import pns.estimators as e
import pns.likelihoods as likelihoods
import pns.estimators as e
import pns.nested_sampling as ns
import pns.analysis_utils as au
# import pns.results_generation as rg
settings = pns_settings.PerfectNestedSamplingSettings()
# settings.likelihood = likelihoods.exp_power(likelihood_scale=1, power=2)
# settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
pd.set_option('display.width', 200)

# settings
# --------
settings.dynamic_goal = 1
settings.n_dim = 40
run = ns.perfect_nested_sampling(settings)
lrxp = au.merge_lrxp(run[1])
estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.84)]
# cProfile.run('ns.perfect_nested_sampling(settings)')
cProfile.run('au.get_nlive(run[0], lrxp[:, 0])')
