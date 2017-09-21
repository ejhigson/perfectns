#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import numpy as np
import pns_settings
import pns.estimators as e
import pns.analysis_utils as au
import pns.parallelised_wrappers as pw
settings = pns_settings.PerfectNestedSamplingSettings()
np.set_printoptions(precision=5, suppress=True, linewidth=400)

# def timing_decorator(some_function):
# 
#     """
#     Outputs the time a function takes
#     to execute.
#     """
# 
#     def wrapper():
#         t1 = time.time()
#         result = some_function()
#         t2 = time.time()
#         return "Time it took to run the function: " + str((t2 - t1)) + "\n"
#     return wrapper
# 
# 
# def timeFunc(func):
# 
#     def time(*args, **kw):
#         t_start = time.time()
#         result = method(*args, **kw)
#         t_end = time.time()
#         print('%r (%r, %r) %2.2f sec' % \
#               (method.__name__, args, kw, t_end - t_start))
#         return result
#     return timed


estimator_list = [e.logzEstimator(),
                  e.theta1Estimator(),
                  e.theta1squaredEstimator(),
                  e.theta1confEstimator(0.5),
                  e.theta1confEstimator(0.84),
                  e.theta1confEstimator(0.975),
                  e.rconfEstimator(0.84)]

print("True est values")
print(e.check_estimator_values(estimator_list, settings))

print("Standard NS Run")
s_run_list = pw.generate_runs(settings, 10, parallelise=False)
stats, all_values = pw.func_on_runs(au.run_estimators,
                                    s_run_list,
                                    estimator_list)
print(stats)


print("Dynamic NS Run")
settings.dynamic_goal = 1
d_run_list = pw.generate_runs(settings, 10)
stats, all_values = pw.func_on_runs(au.run_estimators,
                                    d_run_list,
                                    estimator_list)
print(stats)
