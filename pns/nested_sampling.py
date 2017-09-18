#!/usr/bin/python
"""Generates nested sampling runs and threads."""

import numpy as np
import copy
# perfect nested sampling modules
import pns.maths_functions as mf
import pns.analysis_utils as au


def generate_standard_run(nlive, settings, return_logl_min_max=False):
    # Reset the random seed to avoid repeated results when multiprocessing. For more info see:
    # http://stackoverflow.com/questions/29854398/seeding-random-number-generators-in-parallel-programs
    np.random.seed()
    threads = [None] * nlive
    live = np.zeros((nlive, 3))
    live[:, 2] = np.log(np.random.random(live.shape[0]))
    live[:, 1] = settings.r_given_logx(live[:, 2])
    live[:, 0] = settings.logl_given_r(live[:, 1])
    # termination condition variables
    logx_i = 0.0
    logz_dead = -np.inf
    logz_live = mf.log_sum_given_logs(live[:, 0]) + logx_i
    t = np.exp(-1.0 / nlive)
    logtrapz = np.log(0.5 * ((t ** -1) - t))  # factor for trapizium rule of geometric series
    while logz_live - np.log(settings.zv_termination_fraction) > logz_dead:  # or logdelta > logz_dead:
        # add to dead points
        dying_ind = np.where(live[:, 0] == live[:, 0].min())[0][0]
        if threads[dying_ind] is None:
            threads[dying_ind] = copy.deepcopy(live[dying_ind, :])  # must deep copy or this changes when the live points are updated
            # must reshape so threads is a 2d array even if it only has one row to avoid index errors
            threads[dying_ind] = np.reshape(threads[dying_ind], (1, threads[dying_ind].shape[0]))
        else:
            threads[dying_ind] = np.vstack((threads[dying_ind], live[dying_ind, :]))
        # update dead evidence estimates
        logx_i += -1.0 / nlive
        logz_dead = mf.log_sum_given_logs((logz_dead, live[dying_ind, 0] + logtrapz + logx_i))
        # add new point
        live[dying_ind, 2] += np.log(np.random.random())
        live[dying_ind, 1] = settings.r_given_logx(live[dying_ind, 2])
        live[dying_ind, 0] = settings.logl_given_r(live[dying_ind, 1])
        logz_live = mf.log_sum_given_logs(live[:, 0]) + logx_i - np.log(nlive)
    for i, _ in enumerate(threads):
        # add remaining live points to end of threads
        if threads[i] is None:
            # must reshape so threads is a 2d array even if it only has one row to avoid index errors
            threads[i] = np.reshape(live[i, :], (1, live.shape[1]))
        else:
            threads[i] = np.vstack((threads[i], live[i, :]))
        # add parameters
        threads[i] = np.hstack([threads[i], settings.sample_contours(threads[i][:, 2])])
    if return_logl_min_max:  # return data on threads for use as part of a dynamic run
        logl_min_max_list = []
        for i, _ in enumerate(threads):
            logl_min_max_list.append([None, live[i, 0]])
        return threads, logl_min_max_list
    else:  # return nlive vector for inclusion of live points in threads
        n_calls = au.get_n_calls(threads)
        nlive_array = np.zeros(n_calls) + nlive
        for i in range(1, nlive + 1):
            nlive_array[-i] = i
        return [{'nlive': nlive_array}, threads]
    # return [{'nlive': nlive_array, "logz_scaled_dead": logz_dead - settings.logz_analytic}, threads]


# Single thread helper functions
# ------------------------------
# used for analytic termination and for dynamic nested sampling


def generate_thread_logx(logx_end, logx_start=0, keep_final_point=True):
    """Generate x co-ordinates of a new nested sampling thread (single live point run)."""
    logx_list = [logx_start + np.log(np.random.random())]
    while logx_list[-1] > logx_end:
        logx_list.append(logx_list[-1] + np.log(np.random.random()))
    if not keep_final_point:
        del logx_list[-1]  # remove the point which violates the hard termination condition
    return logx_list


def generate_single_thread(settings, logx_end, logx_start=0, keep_final_point=True, reset_random_seed=True):
    """Make lp array for single thread of problem specified in settings."""
    if reset_random_seed:
        np.random.seed()  # needed to avoid repeated results when multiprocessing - see http://stackoverflow.com/questions/29854398/seeding-random-number-generators-in-parallel-programs
    if logx_end is None:
        logx_end = settings.logx_terminate
    assert logx_start > logx_end, "generate_single_thread: logx_start=" + str(logx_start) + " <= logx_end=" + str(logx_end)
    logx_list = generate_thread_logx(logx_end, logx_start=logx_start, keep_final_point=keep_final_point)
    if len(logx_list) == 0:
        return None
    else:
        lrx = np.zeros((len(logx_list), 3))
        lrx[:, 2] = np.asarray(logx_list)
        lrx[:, 1] = settings.r_given_logx(lrx[:, 2])
        lrx[:, 0] = settings.logl_given_r(lrx[:, 1])
        return np.hstack([lrx, settings.sample_contours(lrx[:, 2])])
