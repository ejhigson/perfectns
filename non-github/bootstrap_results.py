#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import pandas as pd
import pns.settings
import pns.estimators as e
# import pns.analysis_utils as au
# import pns.parallelised_wrappers as pw
# import pns.maths_functions as mf
import pns.results_generation as rg
import pns.likelihoods as likelihoods
# import pns.save_load_utils as slu
import pns.priors as priors
settings = pns.settings.PerfectNestedSamplingSettings()
pd.set_option('display.width', 200)


# Settings
# --------
parallelise = True
load = True
save = True
n_run_ci = 500
n_runs = 5000
n_simulate = 200
n_simulate_ci = 1000
# perfect nested sampling calc type:
add_sim_method = True
cred_int = 0.95
settings.dynamic_goal = None
settings.prior = priors.gaussian(10)
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
ninit_sep = False
settings.n_dim = 3
estimator_list = [e.logzEstimator(),
                  e.paramMeanEstimator(),
                  e.paramSquaredMeanEstimator(),
                  e.paramCredEstimator(0.5),
                  e.paramCredEstimator(0.84)]
e_names = []
for est in estimator_list:
    e_names.append(est.name)

# run results
# -----------
print("Running bootstrap results:")
bootstrap_results = rg.get_bootstrap_results(n_runs, n_simulate,
                                             estimator_list, settings,
                                             load=load, save=save,
                                             n_simulate_ci=n_simulate_ci,
                                             add_sim_method=add_sim_method,
                                             n_run_ci=n_run_ci,
                                             cred_int=cred_int,
                                             ninit_sep=ninit_sep,
                                             parallelise=parallelise)
print(bootstrap_results)
# latex_df = slu.latex_format_df(bootstrap_results, cols=None, rows=None,
#                                dp_list=None)
# print(latex_df)
