#!/usr/bin/python
"""Module for functions used to analyse the nested sampling run's outputs"""

import numpy as np
import scipy.misc  # for scipy.misc.logsumexp
# from numba import jit
import pns.maths_functions as mf


# Access functions which interface directly with ns runs
# ------------------------------------------------------

def samples_array_given_run(run):
    """
    Converts information on samples in a nested sampling run dictionary into a
    numpy array representation. This allows fast addition of more samples and
    recalculation of nlive.

    Parameters
    ----------
    run: dict
        Nested sampling run dictionary.
        Contains keys: 'logl', 'r', 'logx', 'thread_label', 'nlive_array',
        'theta'

    Returns
    -------
    samples: numpy array
        Numpy array containing columns
        [logl, r, logx, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    """
    samples = np.zeros((run['theta'].shape[0], 5 + run['theta'].shape[1]))
    samples[:, 0] = run['logl']
    samples[:, 1] = run['r']
    samples[:, 2] = run['logx']
    samples[:, 3] = run['thread_labels']
    # Calculate 'change in nlive' after each step
    samples[:-1, 4] = np.diff(run['nlive_array'])
    samples[-1, 4] = -1  # nlive drops to zero after final point
    samples[:, 5:] = run['theta']
    return samples


def dict_given_samples_array(samples, thread_min_max):
    """
    Converts an array of information about samples back into a dictionary.

    Parameters
    ----------
    samples: numpy array
        Numpy array containing columns
        [logl, r, logx, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    nlive_0: int
        The number of threads which begin by sampling from the whole prior.
        I.e. the number of live points at the first sample. Needed to
        calculate nlive_array from the differences in nlive at each step.

    Returns
    -------
    run: dict
        Nested sampling run dictionary.
        Contains keys: 'logl', 'r', 'logx', 'thread_label', 'nlive_array',
        'theta'
    """
    nlive_0 = sum(np.isnan(thread_min_max[:, 0]))
    nlive_array = np.zeros(samples.shape[0]) + nlive_0
    nlive_array[1:] += np.cumsum(samples[:-1, 4])
    assert nlive_array.min() > 0, 'nlive contains 0s or negative values!\n' \
        'nlive_array = ' + str(nlive_array)
    assert nlive_array[-1] == 1, 'final point in nlive_array should be 1!\n' \
        'nlive_array = ' + str(nlive_array)
    samples_dict = {'logl': samples[:, 0],
                    'r': samples[:, 1],
                    'logx': samples[:, 2],
                    'thread_labels': samples[:, 3],
                    'nlive_array': nlive_array,
                    'thread_min_max': thread_min_max,
                    'theta': samples[:, 5:]}
    return samples_dict


def get_run_threads(ns_run):
    """
    Get the individual threads for a nested sampling run.
    """
    samples = samples_array_given_run(ns_run)
    n_threads = ns_run['thread_min_max'].shape[0]
    threads = []
    for i in range(1, n_threads + 1):
        threads.append(samples[np.where(samples[:, 3] == i)])
        # delete changes in nlive due to other threads in the run
        threads[-1][:, 4] = 0
        threads[-1][-1, 4] = -1
    return threads


def get_nlive_thread_min_max(run):
    """
    Calculates the local number of live points for each sample using likelihood
    values and the thread_min_max array.
    """
    nlive_array = np.zeros(run['logl'].shape[0])
    lmm_ar = run['thread_min_max']
    # no min logl
    for r in lmm_ar[np.isnan(lmm_ar[:, 0])]:
        nlive_array[np.where(r[1] >= run['logl'])] += 1
    # no max logl
    for r in lmm_ar[np.isnan(lmm_ar[:, 1])]:
        nlive_array[np.where(run['logl'] > r[0])] += 1
    # both min and max logl
    for r in lmm_ar[~np.isnan(lmm_ar[:, 0]) & ~np.isnan(lmm_ar[:, 1])]:
        indexes = np.where((r[1] >= run['logl']) & (run['logl'] > r[0]))
        nlive_array[indexes] += 1
    assert lmm_ar[np.isnan(lmm_ar[:, 0]) &
                  np.isnan(lmm_ar[:, 1])].shape[0] == 0, \
        'Should not have threads with neither start nor end logls'
    # If nlive_array contains zeros then print info and throw error
    assert nlive_array.min() > 0, 'nlive contains 0s or negative values!\n' \
        'nlive_array = ' + str(nlive_array)
    assert nlive_array[-1] == 1, 'final point in nlive_array should be 1!\n' \
        'nlive_array = ' + str(nlive_array)
    return nlive_array


def bootstrap_resample_run(ns_run, threads, ninit_sep=False):
    """
    Bootstrap resamples threads of nested sampling run, returning a new
    (resampled) nested sampling run.
    """
    n_threads = len(threads)
    if ns_run['settings']['dynamic_goal'] is not None and ninit_sep:
        ninit = ns_run['settings']['ninit']
        inds = np.random.randint(0, ninit, ninit)
        inds = np.append(inds, np.random.randint(ninit, n_threads,
                                                 n_threads - ninit))
    else:
        inds = np.random.randint(0, n_threads, n_threads)
    threads_temp = [threads[i] for i in inds]
    thread_min_max_temp = ns_run['thread_min_max'][inds]
    # construct lrxtnp array from the threads, including an updated nlive
    lrxtnp_temp = threads_temp[0]
    for t in threads_temp[1:]:
        lrxtnp_temp = np.vstack((lrxtnp_temp, t))
    lrxtnp_temp = lrxtnp_temp[np.argsort(lrxtnp_temp[:, 0])]
    # update the changes in live points column for threads which start part way
    # through the run. These are only present in dynamic nested sampling.
    logl_starts = thread_min_max_temp[:, 0]
    for logl_start in logl_starts[~np.isnan(logl_starts)]:
        ind = np.where(lrxtnp_temp[:, 0] == logl_start)[0]
        if ind.shape == (1,):
            # If the point at which this thread started is present exactly
            # once in this bootstrap replication:
            lrxtnp_temp[ind[0], 4] += 1
        elif ind.shape == (0,):
            # If the point with the likelihood at which the thread started
            # is not present in this particular bootstrap replication,
            # approximate it with the point with the nearest likelihood.
            ind_closest = np.argmin(np.abs(lrxtnp_temp[:, 0] - logl_start))
            lrxtnp_temp[ind_closest, 4] += 1
        else:
            # If the point at which this thread started is present multiple
            # times in this bootstrap replication, select one at random to
            # incriment nlive on.
            lrxtnp_temp[np.random.choice(ind), 4] += 1
    # make run
    ns_run_temp = dict_given_samples_array(lrxtnp_temp, thread_min_max_temp)
    ns_run_temp['settings'] = ns_run['settings']
    return ns_run_temp


# Helper functions
# ----------------

def get_logw(logl, nlive_array, simulate=False):
    """
    tbc
    """
    logx_inc_start = np.zeros(logl.shape[0] + 1)
    # find X value for each point (start is at logX=0)
    logx_inc_start[1:] = get_logx(nlive_array, simulate=simulate)
    logw = np.zeros(logl.shape[0])
    # vectorized trapezium rule:
    logw[:-1] = mf.log_subtract(logx_inc_start[:-2], logx_inc_start[2:])
    logw -= np.log(2)  # divide by 2 as per trapezium rule formulae
    # assign all prior volume between X=0 and the first point to logw[0]
    logw[0] = scipy.misc.logsumexp([logw[0], np.log(0.5) +
                                    mf.log_subtract(logx_inc_start[0],
                                                    logx_inc_start[1])])
    logw[-1] = logw[-2]  # approximate final element as equal to the one before
    logw += logl
    return logw


def get_logx(nlive, simulate=False):
    """
    Returns a logx vector showing the expected or simulated logx positions of
    points.
    """
    assert nlive.min() > 0, 'nlive contains zeros or negative values!' \
        'nlive = ' + str(nlive)
    if simulate:
        logx_steps = np.log(np.random.random(nlive.shape)) / nlive
    else:
        logx_steps = -1 * (nlive ** -1)
    return np.cumsum(logx_steps)


# Functions for output from a single nested sampling run
# ------------------------------------------------------
# These all have arguments (ns_run, estimator_list, **kwargs)


def run_estimators(run, estimator_list, **kwargs):
    """
    Calculates values of list of estimators for a single nested sampling run.
    """
    simulate = kwargs.get('simulate', False)
    logw = get_logw(run['logl'], run['nlive_array'], simulate=simulate)
    output = np.zeros(len(estimator_list))
    for i, f in enumerate(estimator_list):
        output[i] = f.estimator(logw, run)
    return output


def run_std_simulate(run, estimator_list, **kwargs):
    """
    Calculates simulated weight standard deviation estimates for a single
    nested sampling run.
    """
    # NB simulate must be True and analytical_w must be False so these are not
    # taken from kwargs
    n_simulate = kwargs['n_simulate']  # No default, must specify
    return_values = kwargs.get('return_values', False)
    all_values = np.zeros((len(estimator_list), n_simulate))
    for i in range(0, n_simulate):
        all_values[:, i] = run_estimators(run, estimator_list, simulate=True)
    stds = np.zeros(all_values.shape[0])
    for i, _ in enumerate(stds):
        stds[i] = np.std(all_values[i, :], ddof=1)
    if return_values:
        return stds, all_values
    else:
        return stds


def run_std_bootstrap(ns_run, estimator_list, **kwargs):
    """
    Calculates bootstrap standard deviation estimates
    for a single nested sampling run.
    """
    ninit_sep = kwargs.get('ninit_sep', True)
    n_simulate = kwargs['n_simulate']  # No default, must specify
    threads = get_run_threads(ns_run)
    bs_values = np.zeros((len(estimator_list), n_simulate))
    for i in range(0, n_simulate):
        ns_run_temp = bootstrap_resample_run(ns_run, threads,
                                             ninit_sep=ninit_sep)
        bs_values[:, i] = run_estimators(ns_run_temp, estimator_list)
        del ns_run_temp
    stds = np.zeros(bs_values.shape[0])
    for j, _ in enumerate(stds):
        stds[j] = np.std(bs_values[j, :], ddof=1)
    return stds


def run_ci_bootstrap(ns_run, estimator_list, **kwargs):
    """
    Calculates bootstrap confidence interval estimates for a single nested
    sampling run.
    """
    ninit_sep = kwargs.get('ninit_sep', True)
    n_simulate = kwargs['n_simulate']  # No default, must specify
    cred_int = kwargs['cred_int']   # No default, must specify
    assert min(cred_int, 1. - cred_int) * n_simulate > 1, \
        'n_simulate = ' + str(n_simulate) + ' is not big enough to ' \
        'calculate the bootstrap ' + str(cred_int) + ' CI'
    threads = get_run_threads(ns_run)
    bs_values = np.zeros((len(estimator_list), n_simulate))
    for i in range(0, n_simulate):
        ns_run_temp = bootstrap_resample_run(ns_run, threads,
                                             ninit_sep=ninit_sep)
        bs_values[:, i] = run_estimators(ns_run_temp, estimator_list)
        del ns_run_temp
    # estimate specificed confidence intervals
    # formulae for alpha CI on estimator T = 2 T(x) - G^{-1}(T(x*))
    # where G is the CDF of the bootstrap resamples
    expected_estimators = run_estimators(ns_run, estimator_list)
    cdf = ((np.asarray(range(bs_values.shape[1])) + 0.5) /
           bs_values.shape[1])
    ci_output = expected_estimators * 2
    for i, _ in enumerate(ci_output):
        ci_output[i] -= np.interp(1. - cred_int, cdf,
                                  np.sort(bs_values[i, :]))
    return ci_output
