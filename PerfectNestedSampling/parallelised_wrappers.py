#!/usr/bin/python
"""
Wrapper functions for generating nested sampling runs and performing functions
on them in parallel, using the concurrent.futures module.
"""

import concurrent.futures
import numpy as np
from tqdm import tqdm
# perfect nested sampling modules
import PerfectNestedSampling.nested_sampling as ns
import PerfectNestedSampling.save_load_utils as slu


@slu.timing_decorator
def func_on_runs(single_run_func, run_list, estimator_list, **kwargs):
    """
    Performs input analysis function on a list of nested sampling runs.
    Parallelised by default.
    """
    max_worker = kwargs.get('max_worker', None)
    parallelise = kwargs.get('parallelise', True)
    results_list = []
    print('func_on_runs: calculating ' + single_run_func.__name__ + ' for ' +
          str(len(run_list)) + ' runs')
    if parallelise:
        # if max_worker is None this defaults to number of processors on the
        # machine * 5
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_worker)
        futures = []
        for run in run_list:
            futures.append(pool.submit(single_run_func, run,
                                       estimator_list, **kwargs))
        for _, fut in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                           leave=False, total=len(futures)):
            results_list.append(fut.result())
        del futures
        del pool
    else:
        print('Warning: func_on_runs not parallelised!')
        for i in tqdm(range(len(run_list)), leave=False):
            results_list.append(single_run_func(run_list[i],
                                                estimator_list, **kwargs))
    all_values = np.zeros((len(estimator_list), len(run_list)))
    for i, result in enumerate(results_list):
        all_values[:, i] = result
    return all_values


@slu.timing_decorator
def generate_runs(settings, n_repeat, max_worker=None, parallelise=True):
    """
    Generate n_repeat nested sampling runs in parallel.

    Parameters
    ----------
    settings: PerfectNestedSamplingSettings object
    n_repeat: int
        Number of nested sampling runs to generate.
    parallelise: bool, optional
        Should runs be generated in parallel
    max_worker: int or None, optional
        Number of processes. If max_worker is None then
        concurrent.futures.ProcessPoolExecutor defaults to 5 * the number
        processors of the machine.

    Returns
    -------
    run_list
        list of n_repeat nested sampling runs.
    """
    run_list = []
    if parallelise is False:
        print('Warning: generate_runs not parallelised!')
        for _ in tqdm(range(n_repeat), leave=False):
            run_list.append(ns.generate_ns_run(settings))
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_worker)
        futures = []
        for _ in range(n_repeat):
            futures.append(pool.submit(ns.generate_ns_run, settings))
        for _, fut in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                           leave=False, total=len(futures)):
            run_list.append(fut.result())
        del futures
        del pool
    return run_list


def get_run_data(settings, n_repeat, **kwargs):
    """
    Tests if runs with the specified settings are already cached. If not
    the runs are generated and saved.

    Parameters
    ----------
    settings: PerfectNestedSamplingSettings object
    n_repeat: int
        Number of nested sampling runs to generate.
    parallelise: bool, optional
        Should runs be generated in parallel
    max_worker: int or None
        Number of processes. If max_worker is None then
        concurrent.futures.ProcessPoolExecutor defaults to 5 * the number
        processors of the machine.
    load: bool
        Should previously saved runs be loaded? If False, new runs are
        generated.
    save: bool
        Should any new runs generated be saved?
    overwrite_existing: bool, optional
        if a file exists already but we generate new run data, should we
        overwrite the existing file when saved?
    check_loaded_settings: bool, optional
        if we load a cached file, should we check if the loaded file's settings
        match the current settings (and generate fresh runs if they do not)?

    Returns
    -------
    run_list
        list of n_repeat nested sampling runs.
    """
    parallelise = kwargs.get('parallelise', True)
    max_worker = kwargs.get('max_worker', None)
    load = kwargs.get('load', True)
    save = kwargs.get('save', True)
    overwrite_existing = kwargs.get('overwrite_existing', False)
    check_loaded_settings = kwargs.get('check_loaded_settings', False)
    save_name = slu.data_save_name(settings, n_repeat)
    if load:
        print('get_run_data: ' + save_name)
        try:
            data = slu.pickle_load(save_name)
        except OSError:  # FileNotFoundError is a subclass of OSError
            print('File not found - try generating new data')
            load = False
        except EOFError:
            print('EOFError loading file - try generating new data and '
                  'overwriting current file')
            load = False
            overwrite_existing = True
        if check_loaded_settings:
            # Assume all runs in the loaded list have the same settings, in
            # which case we only need check the first one.
            if settings.get_settings_dict() == data[0]['settings']:
                print('Loaded settings = current settings')
            else:
                print('Loaded settings =')
                print(data[0]['settings'])
                print('are not equal to current settings =')
                print(settings.get_settings_dict())
                del data
                load = False
    if not load:
        data = generate_runs(settings, n_repeat, max_worker=max_worker,
                             parallelise=parallelise)
        if save:
            print('Generated new runs: saving to ' + save_name)
            slu.pickle_save(data, save_name,
                            overwrite_existing=overwrite_existing)
    return data
