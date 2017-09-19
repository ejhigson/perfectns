#!/usr/bin/python
"""Wrapper functions for running nested sampling in parallel and in batches."""

import time
import concurrent.futures
import numpy as np
from tqdm import tqdm
# perfect nested sampling modules
import pns.maths_functions as mf
import pns.nested_sampling as ns


# Parallelised wrappers
# ---------------------
# def generate_dynamic_run(dynamic_goal, settings, nlive_const=None, n_calls_max=None):

def generate_runs(settings, n_repeat, tqdm_leave=True, print_time=True, n_process=None, parallelise=True):
    if print_time:
        start_time = time.time()
    run_list = []
    if parallelise is False:
        print("Warning: generate_runs not parallelised!")
        for i in tqdm(range(n_repeat), leave=tqdm_leave):
            run_list.append(ns.perfect_nested_sampling(settings))
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_process)  # if n_process is None this defaults to num processors of machine * 5
        futures = []
        for i in range(n_repeat):
                futures.append(pool.submit(ns.perfect_nested_sampling, settings))
        for i, result in tqdm(enumerate(concurrent.futures.as_completed(futures)), leave=tqdm_leave, total=len(futures)):
                run_list.append(result.result())
        del futures
        del pool
    if print_time:
        end_time = time.time()
        print("generate_runs took %.3f seconds" % (end_time - start_time))
    return run_list


# Perform single run function on list of ns runs
# ----------------------------------


def func_on_runs(single_run_func, run_list, estimator_list, **kwargs):
    """
    Performs input analysis function on a list of nested sampling runs.
    Parallelised by default.
    """
    n_process = kwargs.get('n_process', None)
    parallelise = kwargs.get('parallelise', True)
    use_tqdm = kwargs.get('use_tqdm', True)
    # use_tqdm = kwargs.get('use_tqdm', utils.in_ipython())  # defaults to using tqdm if in ipython and not otherwise
    tqdm_leave = kwargs.get('tqdm_leave', False)
    print_time = kwargs.get('print_time', True)
    return_list = kwargs.get('return_list', False)
    results_list = []
    start_time = time.time()
    if parallelise:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_process)  # if n_process is None this defaults to num processors of machine * 5
        futures = []
        for i in range(len(run_list)):
            futures.append(pool.submit(single_run_func, run_list[i], estimator_list, **kwargs))
        if use_tqdm:
            for i, result in tqdm(enumerate(concurrent.futures.as_completed(futures)), leave=tqdm_leave, total=len(futures)):
                results_list.append(result.result())
        else:
            for i, result in enumerate(concurrent.futures.as_completed(futures)):
                results_list.append(result.result())
        del futures
        del pool
    else:
        print("Warning: func_on_runs not parallelised!")
        if use_tqdm:
            for i in tqdm(range(len(run_list)), leave=tqdm_leave):
                results_list.append(single_run_func(run_list[i], estimator_list, **kwargs))
        else:
            for i in range(len(run_list)):
                results_list.append(single_run_func(run_list[i], estimator_list, **kwargs))
    if print_time:
        end_time = time.time()
        print(single_run_func.__name__ + " took %.3f seconds" % (end_time - start_time))
    if return_list:
        return results_list
    else:
        all_values = np.zeros((len(estimator_list), len(run_list)))
        for i, result in enumerate(results_list):
            all_values[:, i] = result
        stats = mf.stats_rows(all_values)
        return stats, all_values


def func_on_runs_batch(single_run_func, run_list, estimator_list, **kwargs):
    """
    Wrapper which splits run_list into n_batch parts and uses func_on_runs on each seperately.
    Faster for large lists as futures objects take up a lot of memory.
    """
    n_run = len(run_list)
    start_time = time.time()
    n_batch = kwargs.get('n_batch', int((n_run - 1) / 1000.0) + 1)  # set default number of batches
    assert n_run % n_batch == 0, "n_run=" + str(n_run) + " must be divisable by n_batch=" + str(n_batch)
    batch_len = int(n_run / n_batch)
    all_values = np.zeros((len(estimator_list), n_run))
    for i in range(n_batch):
        _, all_values[:, i * batch_len:(i + 1) * batch_len] = func_on_runs(single_run_func, run_list[i * batch_len:(i + 1) * batch_len], estimator_list, print_time=False, **kwargs)
    stats = mf.stats_rows(all_values)
    end_time = time.time()
    print("func_on_runs_batch using " + single_run_func.__name__ + " and " + str(n_run) + " runs took %.3f seconds" % (end_time - start_time))
    return stats, all_values


# def get_run_data(n_repeat, nlive, settings, dynamic_goal=None, n_calls_max=None, tqdm_leave=False, print_time=True, n_process=None, parallelise=True, load=True, save=True, tuned_dynamic_p=False):
#     save_name = utils.data_save_name(n_repeat, dynamic_goal, nlive, settings, n_calls_max=n_calls_max, tuned_dynamic_p=tuned_dynamic_p)
#     print("get_run_data: " + save_name)
#     if load:
#         print("Loading threads")
#         try:
#             data = utils.pickle_load(save_name)
#         except OSError:  # "FileNotFoundError is a subclass of OSError. Must use OSError for except.
#             print("File not found - try generating new data")
#             load = False
#         else:
#             # this section ensures we use the settings from the loaded data not the current ones when there is a clash
#             if test_settings_dict(data[0], settings):
#                 print("Loaded settings = current settings")
#                 if dynamic_goal is not None:
#                     dynamic_settings = settings.get_dynamic_settings(dynamic_goal)
#                     print("dynamic settings:", dynamic_settings)
#                     for key in dynamic_settings:
#                         if dynamic_settings[key] != data[1][0][key]:
#                             print("Loaded dnsd is not equal to current settings! key, loaded value, settings value are:")
#                             print(key, data[1][0][key], dynamic_settings[key])
#                             load = False
#             else:
#                 print("Loaded settings =")
#                 print(data[0])
#                 print("are not equal to current settings =")
#                 print(settings.get_settings_dict())
#                 del data
#                 load = False
#     if not load:
#         print("Generate new runs")
#         data = [settings.get_settings_dict()]
#         data = data + generate_runs(n_repeat, nlive, dynamic_goal, settings, n_calls_max=n_calls_max, tqdm_leave=tqdm_leave, print_time=print_time, n_process=n_process, parallelise=parallelise, tuned_dynamic_p=tuned_dynamic_p)
#         if save:
#             print("Saving runs")
#             utils.pickle_save(data, save_name)
#     return data[1:]


# helper functions
# ----------------

def test_settings_dict(dictionary, settings):
    """Test if settings in a loaded dictionary are the same as those currently in the settings module."""
    answer = True
    for key, value in settings.get_settings_dict().items():
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, dictionary[key]):
                print(key, "current=" + str(value) + " != loaded=" + str(dictionary[key]))
                answer = False
        else:
            if not value == dictionary[key]:
                print(key, "current=" + str(value) + " != loaded=" + str(dictionary[key]))
                answer = False
        return answer
