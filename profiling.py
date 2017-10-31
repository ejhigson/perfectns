#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import cProfile
import re
import pstats
import pandas as pd
import numpy as np
import pns_settings
# import pns.estimators as e
import pns.likelihoods as likelihoods
import pns.estimators as e
import pns.nested_sampling as ns
import pns.analysis_utils as au
import pns.priors as priors
# import pns.results_generation as rg
settings = pns_settings.PerfectNestedSamplingSettings()
# settings.likelihood = likelihoods.exp_power(likelihood_scale=1, power=2)
# settings.likelihood = likelihoods.cauchy(likelihood_scale=1)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
np.core.arrayprint._line_width = 400
np.set_printoptions(precision=5, suppress=True)
pd.set_option('display.width', 200)

# settings
# --------
settings.dynamic_goal = None
settings.n_dim = 300
# settings.prior = priors.gaussian_cached(10, n_dim=settings.n_dim)
settings.prior = priors.gaussian_cached(10)
prof = cProfile.run('ns.perfect_nested_sampling(settings)', 'data/restats')
p = pstats.Stats('data/restats')
p.strip_dirs().sort_stats('tottime').print_stats(20)
# run = ns.perfect_nested_sampling(settings)
# lrxp = run['lrxtnp']
# estimator_list = [e.logzEstimator(),
#                   e.theta1Estimator(),
#                   e.theta1squaredEstimator(),
#                   e.theta1confEstimator(0.84)]
# cProfile.run('au.get_nlive(run, lrxp[:, 0])')
# cProfile.run('au.run_std_bootstrap(run, estimator_list, n_simulate=200)')
# cProfile.run('au.run_ci_bootstrap(run, estimator_list, n_simulate=2000, cred_int=0.95)')
