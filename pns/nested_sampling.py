#!/usr/bin/python
"""Generates nested sampling runs and threads."""

import copy
import numpy as np
import scipy.misc  # for scipy.misc.logsumexp
# perfect nested sampling modules
import pns.maths_functions as mf
import pns.analysis_utils as au


def perfect_nested_sampling(settings):
    if settings.dynamic_goal is None:
        return generate_standard_run(settings)
    else:
        return generate_dynamic_run(settings)


def generate_standard_run(settings, nlive_const=None):
    if nlive_const is None:
        nlive_const = settings.nlive
    # Reset the random seed to avoid repeated results when multiprocessing.
    np.random.seed()
    live_lrxtn = np.zeros((nlive_const, 5))
    live_lrxtn[:, 3] = np.arange(nlive_const) + 1  # thread number
    live_lrxtn[:, 2] = np.log(np.random.random(live_lrxtn.shape[0]))
    live_lrxtn[:, 1] = settings.r_given_logx(live_lrxtn[:, 2])
    live_lrxtn[:, 0] = settings.logl_given_r(live_lrxtn[:, 1])
    # termination condition variables
    logx_i = 0.0
    logz_dead = -np.inf
    logz_live = (scipy.misc.logsumexp(live_lrxtn[:, 0]) + logx_i -
                 np.log(nlive_const))
    t = np.exp(-1.0 / nlive_const)
    # Calculate factor for trapizium rule of geometric series
    logtrapz = np.log(0.5 * ((t ** -1) - t))
    # start the array of dead points
    dead_points = []
    while logz_live - np.log(settings.zv_termination_fraction) > logz_dead:
        # add to dead points
        ind = np.where(live_lrxtn[:, 0] == live_lrxtn[:, 0].min())[0][0]
        dead_points.append(copy.deepcopy(live_lrxtn[ind, :]))
        # update dead evidence estimates
        logx_i += -1.0 / nlive_const
        logz_dead = scipy.misc.logsumexp((logz_dead, live_lrxtn[ind, 0] +
                                          logtrapz + logx_i))
        # add new point
        live_lrxtn[ind, 2] += np.log(np.random.random())
        live_lrxtn[ind, 1] = settings.r_given_logx(live_lrxtn[ind, 2])
        live_lrxtn[ind, 0] = settings.logl_given_r(live_lrxtn[ind, 1])
        logz_live = (scipy.misc.logsumexp(live_lrxtn[:, 0]) + logx_i -
                     np.log(nlive_const))
    lrxtn = np.array(dead_points)
    # add remaining live points
    # at -1 to the "change in nlive" column for the final point in each thread
    live_lrxtn[:, 4] = -1
    # include final live points (sorted in likelihood order)
    lrxtn = np.vstack((lrxtn, live_lrxtn[np.argsort(live_lrxtn[:, 0])]))
    # add parameters
    theta = mf.sample_nsphere_shells(lrxtn[:, 1], settings.n_dim,
                                     settings.dims_to_sample)
    run = {'settings': settings.get_settings_dict(),
           'lrxtnp': np.hstack((lrxtn, theta))}
    # Add data on threads' beginnings and ends. Each starts by sampling the
    # whole prior and ends on one of the final live poins.
    run['thread_min_max'] = np.zeros((nlive_const, 2))
    run['thread_min_max'][:, 0] = np.nan
    run['thread_min_max'][:, 1] = live_lrxtn[:, 0]
    return run


# Make dynamic ns run:
# --------------------


def generate_dynamic_run(settings):
    """
    Generate a dynamic nested sampling run.
    Outputs a list starting with dnsd, followed by a list of live points for
    each value of fractions.
    Output dnsd contains additional list array "nlive" and list of minimum and
    maximum logls of threads "thread_min_max".
    The order of the "thread_min_max" list corresponds to the order of the
    threads.
    An entry "[none, none]" is a thread which runs over the whole range X=0 to
    X=X_terminate.
    """
    assert 1 >= settings.dynamic_goal >= 0, "dynamic_goal = " + \
        str(settings.dynamic_goal) + " should be between 0 and 1"
    np.random.seed()  # needed to avoid repeated results when multiprocessing
    # Step 1: run all the way through with limited number of threads
    run = generate_standard_run(settings, nlive_const=settings.nlive_1)
    n_calls = run['lrxtnp'].shape[0]
    if settings.n_calls_max is None:
        # estimate number of likelihood calls available
        n_calls_max = settings.nlive * n_calls / settings.nlive_1
        # Reduce by small factor so dynamic ns uses fewer likelihood calls than
        # normal ns. This factor is a function of the dynamic goal as typically
        # evidence calculations have longer attitional threads than parameter
        # estimation calculations.
        n_calls_max *= (settings.nlive - settings.nlive_2 *
                        (1.5 - 0.5 * settings.dynamic_goal)) / settings.nlive
    else:
        n_calls_max = settings.n_calls_max
    # Step 2: sample the peak until we run out of likelihood calls
    while n_calls < n_calls_max:
        importance = point_importance(run, settings)
        logl_min_max, logx_min_max = min_max_importance(importance,
                                                        run['lrxtnp'],
                                                        settings)
        nlive_2_count = 0
        while nlive_2_count < settings.nlive_2:
            nlive_2_count += 1
            # make new thread
            thread_label = run['thread_min_max'].shape[0] + 1
            thread = generate_single_thread(settings,
                                            logx_min_max[1],
                                            thread_label,
                                            logx_start=logx_min_max[0],
                                            keep_final_point=True)
            # update run
            if not np.isnan(logl_min_max[0]):
                start_ind = np.where(run['lrxtnp'][:, 0] == logl_min_max[0])[0]
                # check there is exactly one point with the likelihood at which
                # the new thread starts, and note that nlive increases by 1
                assert start_ind.shape == (1,)
                run['lrxtnp'][start_ind, 4] += 1
            run['lrxtnp'] = np.vstack((run['lrxtnp'], thread))
            lmm = np.asarray([logl_min_max[0], thread[-1, 0]])
            run['thread_min_max'] = np.vstack((run['thread_min_max'], lmm))
        # sort array and update n_calls in preparation for the next run
        run['lrxtnp'] = run['lrxtnp'][np.argsort(run['lrxtnp'][:, 0])]
        n_calls = run['lrxtnp'].shape[0]
    return run


# Single thread helper functions
# ------------------------------
# used for analytic termination and for dynamic nested sampling


def generate_thread_logx(logx_end, logx_start=0, keep_final_point=True):
    """
    Generate x co-ordinates of a new nested sampling thread (single live point
    run).
    """
    logx_list = [logx_start + np.log(np.random.random())]
    while logx_list[-1] > logx_end:
        logx_list.append(logx_list[-1] + np.log(np.random.random()))
    if not keep_final_point:
        del logx_list[-1]  # remove point which violates termination condition
    return logx_list


def generate_single_thread(settings, logx_end, thread_label, logx_start=0,
                           keep_final_point=True):
    """Make lp array for single thread of problem specified in settings."""
    if logx_end is None:
        logx_end = settings.logx_terminate
    assert logx_start > logx_end, "generate_single_thread: logx_start=" + \
        str(logx_start) + " <= logx_end=" + str(logx_end)
    logx_list = generate_thread_logx(logx_end, logx_start=logx_start,
                                     keep_final_point=keep_final_point)
    if not logx_list:  # PEP8 method for testing if sequence is empty
        return None
    else:
        lrxtn = np.zeros((len(logx_list), 5))
        lrxtn[:, 3] = thread_label
        lrxtn[:, 2] = np.asarray(logx_list)
        lrxtn[:, 1] = settings.r_given_logx(lrxtn[:, 2])
        lrxtn[:, 0] = settings.logl_given_r(lrxtn[:, 1])
        # set change in nlive to -1 where thread ends (zero elsewhere)
        lrxtn[-1, 4] = -1
        theta = mf.sample_nsphere_shells(lrxtn[:, 1],
                                         settings.n_dim,
                                         settings.dims_to_sample)
        return np.hstack([lrxtn, theta])


# Dynamic NS helper functions
# ----------------


def point_importance(run, settings, simulate=False):
    nlive = au.get_nlive(run)
    logw = au.get_logw(run['lrxtnp'][:, 0], nlive, simulate=simulate)
    # subtract logw.max() to avoids numerical errors with very small numbers
    w_relative = np.exp(logw - logw.max())
    if settings.dynamic_goal == 0:
        return z_importance(w_relative, nlive)
    elif settings.dynamic_goal == 1:
        return p_importance(run['lrxtnp'], w_relative,
                            tuned_dynamic_p=settings.tuned_dynamic_p)
    else:
        imp_z = z_importance(w_relative, nlive)
        imp_p = p_importance(run['lrxtnp'], w_relative,
                             tuned_dynamic_p=settings.tuned_dynamic_p)
        importance = (imp_z / np.sum(imp_z)) * (1.0 - settings.dynamic_goal)
        importance += (imp_p / np.sum(imp_p)) * settings.dynamic_goal
        return importance / importance.max()


def z_importance(w, nlive, exact=False):
    importance = np.cumsum(w)
    importance = importance.max() - importance
    if exact:
        importance *= (((nlive ** 2) - 3) * (nlive ** 1.5))
        importance /= (((nlive + 1) ** 3) * ((nlive + 2) ** 1.5))
        importance += w * (nlive ** 0.5) / ((nlive + 2) ** 1.5)
    else:
        importance *= 1.0 / nlive
    return importance / importance.max()


def p_importance(lrxp, w, tuned_dynamic_p=False, tuning_type='theta1'):
    if tuned_dynamic_p is False:
        return w / w.max()
    else:
        assert tuning_type == 'theta1', 'so far only set up for theta1'
        if tuning_type == 'theta1':
            # extract theta1 values from lrxp
            f = lrxp[:, 5]
        # calculate importance in proportion to difference between f values and
        # the calculation mean.
        fabs = np.absolute(f - (np.sum(f * w) / np.sum(w)))
        importance = fabs * w
        return importance / importance.max()


def min_max_importance(importance, lrxp, settings):
    assert settings.dynamic_fraction > 0. and settings.dynamic_fraction < 1., \
        "min_max_importance: settings.dynamic_fraction = " + \
        str(settings.dynamic_fraction) + " must be in [0, 1]"
    # where to start the additional threads:
    high_importance_inds = np.where(importance > settings.dynamic_fraction)[0]
    if high_importance_inds[0] == 0:  # start from sampling the whole prior
        logl_min = np.nan
        logx_min = 0
    else:
        logl_min = lrxp[:, 0][high_importance_inds[0] - 1]
        # use lookup to avoid float errors and to not need inverse function
        ind = np.where(lrxp[:, 0] == logl_min)[0]
        assert ind.shape == (1,), \
            "Should be one unique match for logl=logl_min=" + str(logl_min) + \
            ". Instead we have matches at indexes " + str(ind) + \
            " of the lrxp array (shape " + str(lrxp.shape) + ")"
        logx_min = lrxp[ind[0], 2]
    # where to end the additional threads:
    if high_importance_inds[-1] == lrxp[:, 0].shape[0] - 1:
        logl_max = lrxp[-1, 0]
        logx_max = lrxp[-1, 2]
    else:
        logl_max = lrxp[:, 0][(high_importance_inds[-1] + 1)]
        # use lookup to avoid float errors and to not need inverse function
        ind = np.where(lrxp[:, 0] == logl_max)[0]
        assert ind.shape == (1,), \
            "Should be one unique match for logl=logl_max=" + str(logl_max) + \
            ".\n Instead we have matches at indexes " + str(ind) + \
            " of the lrxp array (shape " + str(lrxp.shape) + ")"
        logx_max = lrxp[ind[0], 2]
    return [logl_min, logl_max], [logx_min, logx_max]
