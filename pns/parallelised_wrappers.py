#!/usr/bin/python
"""Wrapper functions for running nested sampling in parallel and in batches."""

import concurrent.futures
import numpy as np
from tqdm import tqdm
# perfect nested sampling modules
import pns.nested_sampling as ns
import pns.save_load_utils as slu


# Parallelised wrappers
# ---------------------

@slu.timing_decorator
def generate_runs(settings, n_repeat, tqdm_leave=True, n_process=None,
                  parallelise=True):
    """
    Generate n_repeat nested sampling runs in parallel
    """
    run_list = []
    if parallelise is False:
        print("Warning: generate_runs not parallelised!")
        for _ in tqdm(range(n_repeat), leave=tqdm_leave):
            run_list.append(ns.perfect_nested_sampling(settings))
    else:
        # if n_process is None this defaults to num processors of machine * 5
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_process)
        futures = []
        for _ in range(n_repeat):
            futures.append(pool.submit(ns.perfect_nested_sampling, settings))
        for _, fut in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                           leave=tqdm_leave, total=len(futures)):
            run_list.append(fut.result())
        del futures
        del pool
    return run_list


# Perform single run function on list of ns runs
# ----------------------------------


@slu.timing_decorator
def func_on_runs(single_run_func, run_list, estimator_list, **kwargs):
    """
    Performs input analysis function on a list of nested sampling runs.
    Parallelised by default.
    """
    n_process = kwargs.get('n_process', None)
    parallelise = kwargs.get('parallelise', True)
    tqdm_leave = kwargs.get('tqdm_leave', False)
    results_list = []
    if parallelise:
        # if n_process is None this defaults to num processors of machine * 5
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_process)
        futures = []
        for run in run_list:
            futures.append(pool.submit(single_run_func, run,
                                       estimator_list, **kwargs))
        for _, fut in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                           leave=tqdm_leave, total=len(futures)):
            results_list.append(fut.result())
        del futures
        del pool
    else:
        print("Warning: func_on_runs not parallelised!")
        for i in tqdm(range(len(run_list)), leave=tqdm_leave):
            results_list.append(single_run_func(run_list[i],
                                                estimator_list, **kwargs))
    all_values = np.zeros((len(estimator_list), len(run_list)))
    for i, result in enumerate(results_list):
        all_values[:, i] = result
    return all_values


def get_run_data(settings, n_repeat, tqdm_leave=False, n_process=None,
                 parallelise=True, load=True, save=True):
    """
    Check if data has already been cashed and if not rerun and save it.
    """
    save_name = slu.data_save_name(settings, n_repeat)
    print("get_run_data: " + save_name)
    if load:
        # print("Loading threads")
        try:
            data = slu.pickle_load(save_name)
        except OSError:  # "FileNotFoundError is a subclass of OSError
            print("File not found - try generating new data")
            load = False
        else:
            # ensure the loaded settings match the current settings
            if settings.get_settings_dict() == data[0][0]['settings']:
                print("Loaded settings = current settings")
                # print(settings.get_settings_dict())
            else:
                print("Loaded settings =")
                print(data[0][0]['settings'])
                print("are not equal to current settings =")
                print(settings.get_settings_dict())
                del data
                load = False
    if not load:
        print("Generate new runs")
        data = generate_runs(settings, n_repeat, tqdm_leave=tqdm_leave,
                             n_process=n_process, parallelise=parallelise)
        if save:
            print("Saving runs")
            slu.pickle_save(data, save_name)
    return data[1:]
