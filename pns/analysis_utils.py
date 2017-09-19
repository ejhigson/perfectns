#!/usr/bin/python
"""Module for functions used to analyse the nested sampling run's outputs"""

import numpy as np
import random  # for shuffling lists
# perfect nested sampling modules
# import pns.utils as utils
import pns.maths_functions as mf


# Helper functions
# ----------------


def get_logx(logl, nlive_array, simulate=False):
    """Returns a logx vector showing the expected or simulated logx positions of points."""
    logx = np.zeros(logl.shape[0])
    assert isinstance(nlive_array, np.ndarray), "warning! nlive = " + str(nlive_array) + " is not an array"
    assert nlive_array.min() > 0, "nlive_array contains zeros: " + str(nlive_array)
    if simulate:
        logx[0] = np.log(np.random.random()) / nlive_array[0]
        for i, _ in enumerate(logx[1:]):
            logx[i + 1] = logx[i] + (np.log(np.random.random()) / nlive_array[i + 1])
    else:
        logx[0] = -1.0 / nlive_array[0]
        for i, _ in enumerate(logx[1:]):
            logx[i + 1] = logx[i] - 1.0 / nlive_array[i + 1]
    return logx


def get_logw(logl, nlive_array, simulate=False):
    logx_inc_start = np.zeros(logl.shape[0] + 1)
    # find X value for each point
    logx_inc_start[1:] = get_logx(logl, nlive_array, simulate=simulate)
    logw = np.zeros(logl.shape[0])
    for i, _ in enumerate(logw[:-1]):
        logw[i] = np.log(0.5) + mf.log_subtract(logx_inc_start[i], logx_inc_start[i + 2])
    logw[0] = mf.log_sum_given_logs([logw[0], np.log(0.5) + mf.log_subtract(logx_inc_start[0], logx_inc_start[1])])  # assign extra prior vol outside the first point to the first point logw[0]
    logw[-1] = logw[-2]  # approximate the final prior volume as equal to the one before
    logw += logl
    return logw


def get_n_calls(ns_run):
    """Returns the number of likelihood calls in a nested sampling run"""
    n_calls = 0
    if isinstance(ns_run[0], np.ndarray):  # true if nlive constant (ns_run is list of threads)
        for thread in ns_run:
            if thread is not None:
                n_calls += thread.shape[0]
    elif isinstance(ns_run[0], dict):  # true for dynamic ns
        for thread in ns_run[1]:
            if thread is not None:
                n_calls += thread.shape[0]
    return n_calls


def get_lp_nlive(ns_run):
    lp = combine_lp_list(ns_run[1])
    if 'nlive_array' in ns_run[0]:
        nlive_array = ns_run[0]['nlive_array']
    else:
        # if nlive_array is not already stored for the run, find it iteratively using the minimum and maximum logls of the slices
        nlive_array = np.zeros(lp.shape[0])
        assert len(ns_run[0]['thread_logl_min_max']) == len(ns_run[1]), "length of threads list not equal to length of logl_min_max list"
        for logl_mm in ns_run[0]['thread_logl_min_max']:
            if len(logl_mm) == 3:
                incriment = logl_mm[2]
            else:
                incriment = 1
            if logl_mm[0] is None:
                if logl_mm[1] is None:
                    nlive_array += incriment
                else:
                    ind = np.where(lp[:, 0] <= logl_mm[1])[0]
                    nlive_array[ind] += incriment
            else:
                if logl_mm[1] is None:
                    ind = np.where(lp[:, 0] > logl_mm[0])[0]
                    nlive_array[ind] += incriment
                else:
                    ind = np.where((lp[:, 0] > logl_mm[0]) & (lp[:, 0] <= logl_mm[1]))[0]
                    nlive_array[ind] += incriment
        if nlive_array.min() < 1:
            loglmax = np.zeros(len(ns_run[0]['thread_logl_min_max']))
            for i, lmm in enumerate(ns_run[0]['thread_logl_min_max']):
                loglmax[i] = lmm[1]
            print(lp[-1, :], loglmax.max(), lp[-1, 0] == loglmax.max())
            print(lp[:, 0])
            assert nlive_array.min() > 0, "nlive_array contains zeros: " + str(nlive_array) + "final logl_min_max = " + str(ns_run[0]['thread_logl_min_max'][-1]) + " and final logl is " + str(ns_run[1][-1][-1, 0]) + " and equality test=" + str(ns_run[1][-1][-1, 0] == ns_run[0]['thread_logl_min_max'][-1][-1])
    return lp, nlive_array


def combine_lp_list(lp_list):
    output = None
    for i in range(len(lp_list)):
        if lp_list[i] is not None:
            if output is not None:
                output = np.vstack((output, lp_list[i]))
            else:
                output = lp_list[i]
    if output is not None:
        output = output[np.argsort(output[:,  0])]
    return output


def get_estimators(lrxp, logw, estimator_list):
    output = np.zeros(len(estimator_list))
    for i, f in enumerate(estimator_list):
        output[i] = f.estimator(logw=logw, logl=lrxp[:, 0], r=lrxp[:, 1], theta=lrxp[:, 3:])
    return output


# Functions for output from a single nested sampling run
# ------------------------------------------------------
# These all have arguments (ns_run, estimator_list, **kwargs)


def run_estimators(ns_run, estimator_list, **kwargs):
    simulate = kwargs.get('simulate', False)
    lrxp, nlive = get_lp_nlive(ns_run)
    logw = get_logw(lrxp[:, 0], nlive, simulate=simulate)
    return get_estimators(lrxp, logw, estimator_list)


# Std dev estimators


def run_std_simulate(ns_run, estimator_list, **kwargs):
    # NB simulate must be True and analytical_w must be False so these are not taken from kwargs
    n_simulate = kwargs["n_simulate"]  # No default value - if not specified should return an error.
    return_values = kwargs.get('return_values', False)
    assert return_values is True or return_values is False, "return_values = " + str(return_values) + " must be True or False"
    all_values = np.zeros((len(estimator_list), n_simulate))
    lp, nlive = get_lp_nlive(ns_run)
    for i in range(0, n_simulate):
            logw = get_logw(lp[:, 0], nlive, simulate=True)
            all_values[:, i] = get_estimators(lp, logw, estimator_list)
    stats = mf.stats_rows(all_values)
    if return_values is False:
        return stats[:, 1]
    elif return_values is True:
        return stats[:, 1], all_values


def run_std_bootstrap(ns_run, estimator_list, **kwargs):
    # assert isinstance(ns_run[0], np.ndarray), "So far only set up for nlive = constant"
    simulate = kwargs.get('simulate', False)
    n_simulate = kwargs["n_simulate"]  # No default value - if not specified should return an error.
    return_values = kwargs.get('return_values', False)
    credible_interval = kwargs.get('credible_interval', False)
    assert return_values is True or return_values is False, "return_values = " + str(return_values) + " must be True or False"
    bootstrap_values = np.zeros((len(estimator_list), n_simulate))
    if isinstance(ns_run[0], np.ndarray):  # run is just a list of threads
        threads = ns_run
    else:
        threads = ns_run[1]
    for i in range(0, n_simulate):
        threads_temp = []
        logl_min_max_temp = []
        #     ns_run_temp = []
        # elif "thread_logl_min_max" in ns_run[0]:  # dynamic ns_run
        #     ns_run_temp = [{"thread_logl_min_max": []}, []]
        # else:  # standard ns run with live points included
        #     ns_run_temp = [{"nlive": ns_run[0]["nlive"]}, []]
        # run is just list of threads
        if isinstance(ns_run[0], np.ndarray):  # run is just a list of threads
            for j in range(len(threads)):
                ind = random.randint(0, len(threads) - 1)
                threads_temp.append(threads[ind])
            ns_run_temp = threads_temp
        else:
            if "nlive_1" in ns_run[0]:  # dynamic run
                # first resample initial threads going all the way through the run
                for j in range(0, ns_run[0]["nlive_1"]):
                    ind = random.randint(0, ns_run[0]["nlive_1"] - 1)
                    threads_temp.append(threads[ind])
                # now resample the remaining threads
                for j in range(ns_run[0]["nlive_1"], len(threads)):
                    ind = random.randint(ns_run[0]["nlive_1"], len(threads) - 1)
                    threads_temp.append(threads[ind])
            else:  # standard run
                for j in range(len(threads)):
                    ind = random.randint(0, len(threads) - 1)
                    threads_temp.append(threads[ind])
                    if "thread_logl_min_max" in ns_run[0]:
                        logl_min_max_temp.append(ns_run[0]["thread_logl_min_max"][ind])
            # make resampled threads into a run
            if "thread_logl_min_max" in ns_run[0]:
                ns_run_temp = [{"thread_logl_min_max": logl_min_max_temp}, threads_temp]
            else:  # standard ns run with live points included
                nlive_temp = np.zeros(get_n_calls(threads_temp)) + len(threads)
                for k in range(1, len(threads)):
                    nlive_temp[-k] = k
                ns_run_temp = [{"nlive": nlive_temp}, threads_temp]
        lp_temp, nlive_temp = get_lp_nlive(threads_temp)
        logw_temp = get_logw(lp_temp[:, 0], nlive_temp, simulate=simulate)
        bootstrap_values[:, i] = get_estimators(lp_temp, logw_temp, estimator_list)
        # bootstrap_values[:, i] = (n_simulate * expected_estimators) - ((n_simulate - 1) * temp_estimators)
        del ns_run_temp
    if credible_interval is False:
        stats = mf.stats_rows(bootstrap_values)
        output = stats[:, 1]
    else:
        # estimate specificed confidence interval
        # formulae for alpha CI on estimator T = 2 T(x) - G^{-1}(T(x*))
        # where G is the CDF of the bootstrap resamples
        assert min(credible_interval, 1. - credible_interval) * n_simulate > 1, "n_simulate=" + str(n_simulate) + " is not big enough to calculate the bootstrap " + str(credible_interval) + " CI"
        lp, nlive = get_lp_nlive(ns_run)
        # the expected values must NOT be simulated
        logw = get_logw(lp[:, 0], nlive, simulate=False)
        expected_estimators = get_estimators(lp, logw, estimator_list)
        output = expected_estimators * 2
        cdf = (np.asarray(range(bootstrap_values.shape[1])) + 0.5) / bootstrap_values.shape[1]
        for i, _ in enumerate(output):
            output[i] -= np.interp(1. - credible_interval, cdf, np.sort(bootstrap_values[i, :]))
        # print(bootstrap_values, credible_interval, cdf, output)
    if return_values is False:
        return output
    elif return_values is True:
        return output, bootstrap_values
