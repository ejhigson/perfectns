#!/usr/bin/python
"""Generates nested sampling runs and threads."""

import numpy as np
import scipy.misc  # for scipy.misc.logsumexp
import copy
# perfect nested sampling modules
import pns.maths_functions as mf
import pns.analysis_utils as au


def perfect_nested_sampling(settings):
    if settings.dynamic_goal is None:
        return generate_standard_run(settings)
    else:
        return generate_dynamic_run(settings)


def generate_standard_run(settings, nlive_const=None,
                          return_logl_min_max=False):
    if nlive_const is None:
        nlive_const = settings.nlive
    # Reset the random seed to avoid repeated results when multiprocessing.
    np.random.seed()
    threads = [None] * nlive_const
    live = np.zeros((nlive_const, 3))
    live[:, 2] = np.log(np.random.random(live.shape[0]))
    live[:, 1] = settings.r_given_logx(live[:, 2])
    live[:, 0] = settings.logl_given_r(live[:, 1])
    # termination condition variables
    logx_i = 0.0
    logz_dead = -np.inf
    logz_live = scipy.misc.logsumexp(live[:, 0]) + logx_i
    t = np.exp(-1.0 / nlive_const)
    # Calculate factor for trapizium rule of geometric series
    logtrapz = np.log(0.5 * ((t ** -1) - t))
    while logz_live - np.log(settings.zv_termination_fraction) > logz_dead:
        # add to dead points
        dying_ind = np.where(live[:, 0] == live[:, 0].min())[0][0]
        if threads[dying_ind] is None:
            # Must deep copy or this changes when the live points are updated.
            threads[dying_ind] = copy.deepcopy(live[dying_ind, :])
            # Must reshape so threads is a 2d array even if it only has one row
            # to avoid index errors,
            threads[dying_ind] = np.reshape(threads[dying_ind],
                                            (1, threads[dying_ind].shape[0]))
        else:
            threads[dying_ind] = np.vstack((threads[dying_ind],
                                           live[dying_ind, :]))
        # update dead evidence estimates
        logx_i += -1.0 / nlive_const
        logz_dead = scipy.misc.logsumexp((logz_dead, live[dying_ind, 0] +
                                          logtrapz + logx_i))
        # add new point
        live[dying_ind, 2] += np.log(np.random.random())
        live[dying_ind, 1] = settings.r_given_logx(live[dying_ind, 2])
        live[dying_ind, 0] = settings.logl_given_r(live[dying_ind, 1])
        logz_live = (scipy.misc.logsumexp(live[:, 0]) + logx_i -
                     np.log(nlive_const))
    for i, _ in enumerate(threads):
        # add remaining live points to end of threads
        if threads[i] is None:
            # Must reshape so threads is a 2d array even if it only has one row
            # to avoid index errors
            threads[i] = np.reshape(live[i, :], (1, live.shape[1]))
        else:
            threads[i] = np.vstack((threads[i], live[i, :]))
        # add parameters
        theta = mf.sample_nsphere_shells(threads[i][:, 1], settings.n_dim,
                                         settings.dims_to_sample)
        threads[i] = np.hstack([threads[i], theta])
    if return_logl_min_max:
        # return data on threads for use as part of a dynamic run
        logl_min_max_list = []
        for i, _ in enumerate(threads):
            logl_min_max_list.append([None, live[i, 0]])
        return threads, logl_min_max_list
    else:  # return nlive vector for inclusion of live points in threads
        n_calls = 0
        for t in threads:
            n_calls += t.shape[0]
        nlive_array = np.zeros(n_calls) + nlive_const
        for i in range(1, nlive_const + 1):
            nlive_array[-i] = i
        return [{'nlive_array': nlive_array,
                 'settings': settings.get_settings_dict()}, threads]


# Make dynamic ns run:
# --------------------


def generate_dynamic_run(settings):
    """
    Generate a dynamic nested sampling run.
    Outputs a list starting with dnsd, followed by a list of live points for
    each value of fractions.
    Output dnsd contains additional list array "nlive" and list of minimum and
    maximum logls of threads "thread_logl_min_max".
    The order of the "thread_logl_min_max" list corresponds to the order of the
    threads.
    An entry "[none, none]" is a thread which runs over the whole range X=0 to
    X=X_terminate.
    """
    assert 1 >= settings.dynamic_goal >= 1, "dynamic_goal = " + \
        str(settings.dynamic_goal) + " should be between 0 and 1"
    np.random.seed()  # needed to avoid repeated results when multiprocessing
    # Step 1: run all the way through with limited number of threads
    run = [{'settings': settings.get_settings_dict()}]
    threads, logl_min_max = generate_standard_run(settings,
                                                  nlive_const=settings.nlive_1,
                                                  return_logl_min_max=True)
    run.append(threads)
    run[0]['thread_logl_min_max'] = logl_min_max
    n_calls = au.get_n_calls(run)
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
        logl_min_max, logx_min_max, n_calls = logl_min_max_given_fraction(run, settings)
        nlive_2_count = 0
        if settings.dynamic_keep_final_point:
            n_calls_itr = 0
            while n_calls_itr < n_calls_max * settings.n_calls_frac or nlive_2_count < settings.nlive_2:
                nlive_2_count += 1
                run[1].append(generate_single_thread(settings, logx_min_max[1],
                              logx_start=logx_min_max[0],
                              keep_final_point=settings.dynamic_keep_final_point))
                n_calls_itr += run[1][-1].shape[0]
                run[0]['thread_logl_min_max'].append([logl_min_max[0],
                                                      run[1][-1][-1, 0]])
        else:
            # Make many threads in a single array with a single logl_min_max to
            # speed stuff up.
            logx = []
            while (len(logx) < n_calls_max * settings.n_calls_frac) or \
                  (nlive_2_count < settings.nlive_2):
                nlive_2_count += 1
                logx += generate_thread_logx(logx_min_max[1],
                            logx_start=logx_min_max[0],
                            keep_final_point=settings.dynamic_keep_final_point)
            # make thread
            lrx = np.zeros((len(logx), 3))
            lrx[:, 2] = np.asarray(logx)
            lrx[:, 1] = settings.r_given_logx(lrx[:, 2])
            lrx[:, 0] = settings.logl_given_r(lrx[:, 1])
            theta = mf.sample_nsphere_shells(lrx[:, 1], settings.n_dim, settings.dims_to_sample)
            run[1].append(np.hstack([lrx, theta]))
            logl_min_max.append(nlive_2_count)
            run[0]['thread_logl_min_max'].append(logl_min_max)
    lrxp = au.vstack_sort_array_list(run[1])
    run[0]['nlive_array'] = au.get_nlive(run[0], lrxp[:, 0])
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


def generate_single_thread(settings, logx_end, logx_start=0,
                           keep_final_point=True):
    """Make lp array for single thread of problem specified in settings."""
    if logx_end is None:
        logx_end = settings.logx_terminate
    assert logx_start > logx_end, "generate_single_thread: logx_start=" + \
        str(logx_start) + " <= logx_end=" + str(logx_end)
    logx_list = generate_thread_logx(logx_end, logx_start=logx_start,
                                     keep_final_point=keep_final_point)
    if len(logx_list) == 0:
        return None
    else:
        lrx = np.zeros((len(logx_list), 3))
        lrx[:, 2] = np.asarray(logx_list)
        lrx[:, 1] = settings.r_given_logx(lrx[:, 2])
        lrx[:, 0] = settings.logl_given_r(lrx[:, 1])
        theta = mf.sample_nsphere_shells(lrx[:, 1],
                                         settings.n_dim,
                                         settings.dims_to_sample)
        return np.hstack([lrx, theta])


# Dynamic NS helper functions
# ----------------


def point_importance(lp, nlive, settings, simulate=False):
    logw = au.get_logw(lp[:, 0], nlive, simulate=simulate)
    # subtract logw.max() to avoids numerical errors with very small numbers
    w_relative = np.exp(logw - logw.max())
    if settings.dynamic_goal == 0:
        return z_importance(w_relative, nlive)
    elif settings.dynamic_goal == 1:
        return p_importance(lp, w_relative,
                            tuned_dynamic_p=settings.tuned_dynamic_p)
    else:
        imp_z = z_importance(w_relative, nlive)
        imp_p = p_importance(lp, w_relative,
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
            f = lrxp[:, 3]
        # calculate importance in proportion to difference between f values and
        # the calculation mean.
        fabs = np.absolute(f - (np.sum(f * w) / np.sum(w)))
        importance = fabs * w
        return importance / importance.max()


def logl_min_max_given_fraction(run, settings):
    assert settings.dynamic_fraction > 0. and settings.dynamic_fraction < 1., \
        "logl_min_max_given_fraction: settings.dynamic_fraction = " + \
        str(settings.dynamic_fraction) + " must be in [0, 1]"
    lrxp = au.vstack_sort_array_list(run[1])
    nlive = au.get_nlive(run[0], lrxp[:, 0])
    n_calls = lrxp.shape[0]
    importance = point_importance(lrxp, nlive, settings)
    # where to start the additional threads:
    high_importance_inds = np.where(importance > settings.dynamic_fraction)[0]
    if high_importance_inds[0] == 0:  # start from sampling the whole prior
        logl_min = None
        logx_min = 0
    else:
        logl_min = lrxp[:, 0][high_importance_inds[0] - 1]
        # use lookup to avoid float errors and to not need inverse function
        ind = np.where(lrxp[:, 0] == logl_min)[0]
        assert ind.shape[0] == 1, \
            "Should be one unique match for logl=logl_min=" + str(logl_min) + \
            ". Instead we have matches at indexes " + str(ind) + \
            " of the lrxp array (shape " + str(lrxp.shape) + ")"
        logx_min = lrxp[ind[0], 2]
    # where to end the additional threads:
    if high_importance_inds[-1] == lrxp[:, 0].shape[0] - 1:
        if settings.dynamic_keep_final_point:
            logl_max = lrxp[-1, 0]
            logx_max = lrxp[-1, 2]
        else:
            # If this is the last point and we are not keeping final points
            # then allow samples to go a bit further.
            # Here we use the biggest shrinkage (smallest number) of 3 randomly
            # generated shrinkage ratios to model what would happen if we were
            # keeping points.
            logx_max = lrxp[-1, 2] + np.log(np.min(np.random.random(3)))
            logl_max = settings.likelihood_prior.logl_given_logx(logx_max)
    else:
        logl_max = lrxp[:, 0][(high_importance_inds[-1] + 1)]
        # use lookup to avoid float errors and to not need inverse function
        ind = np.where(lrxp[:, 0] == logl_max)[0]
        assert ind.shape[0] == 1, \
            "Should be one unique match for logl=logl_max=" + str(logl_max) + \
            ".\n Instead we have matches at indexes " + str(ind) + \
            " of the lrxp array (shape " + str(lrxp.shape) + ")"
        logx_max = lrxp[ind[0], 2]
    return [logl_min, logl_max], [logx_min, logx_max], n_calls
