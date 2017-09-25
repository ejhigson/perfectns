#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import numpy as np
import pns_settings
import pns.estimators as e
# import pns.save_load_utils as slu
# import pns.analysis_utils as au
# import pns.parallelised_wrappers as pw
# import pns.maths_functions as mf
import pns.save_load_utils as slu
import pns.results_generation as rg
settings = pns_settings.PerfectNestedSamplingSettings()
np.set_printoptions(precision=5, suppress=True, linewidth=400)

estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1confEstimator(0.84),
                  e.theta1squaredEstimator(),
                  e.rconfEstimator(0.84)]

e_names = []
for est in estimator_list:
    e_names.append(est.name)

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

# print("Standard NS Run")
# s_run_list = pw.get_run_data(settings, 10)
# values = pw.func_on_runs(au.run_estimators, s_run_list, estimator_list)
# s_df = mf.get_df_row_summary(values, e_names)
# print(s_df)

# print("Dynamic NS Run")

# settings.dynamic_goal = 1
# d_run_list = pw.get_run_data(settings, 10)
# values = pw.func_on_runs(au.run_estimators, d_run_list, estimator_list)
# d_df = mf.get_df_row_summary(values, e_names)
# print(d_df)

n_repeats = 500
dynamic_goals = [None, 0, 0.25, 1]

dr = rg.get_dynamic_results(n_repeats, dynamic_goals, estimator_list, settings)

override_dg = "dynamic_results"
for dg in dynamic_goals:
    override_dg += "_" + str(dg)

save_name = slu.data_save_name(settings, n_repeats, override_dg=override_dg)
dr.to_pickle('data/' + save_name + '.dat')
