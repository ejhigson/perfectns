#!/usr/bin/python
"""Generates nested sampling runs and threads."""

import copy
import numpy as np
import scipy.misc  # for scipy.misc.logsumexp
# perfect nested sampling modules
import pns.maths_functions as mf
import pns.analysis_utils as au


def perfect_nested_sampling(settings):
    """
    Performs perfect nested sampling calculation and returns a nested sampling
    run in the form of a dictionary.

    This function is just a wrapper around the
    generate_standard_run (performs standard nested sampling) and
    generate_dynamic_run (performs dynamic nested samplign) which are chosen
    depending on the input settings.

    Parameters
    ----------
    settings: PerfectNestedSamplingSettings object

    Returns
    -------
    dict
        Nested sampling run dictionary containing information about the run's
        posterior samples and a record of the settings used. These are as
        seperate arrays giving values for points in order of increasing
        likelihood.
        Keys:
            'settings': dict recording settings used.
            'logl': 1d array of log likelihoods.
            'r': 1d array of radial coordinates.
            'logx': 1d array of logx coordinates.
            'theta': 2d array, each row is sample coordinate. The numer of
                     co-ordinates saved is controlled by
                     settings.dims_to_sample.
            'nlive_array': 1d array of the local number of live points at each
                           sample.
            'thread_min_max': 2d array containing likelihoods at which each
                              nested sampling thread begins and ends.
            'thread_labels': 1d array listing the threads each sample belongs
                              to.
    """
    if settings.dynamic_goal is None:
        return generate_standard_run(settings)
    else:
        return generate_dynamic_run(settings)


def generate_standard_run(settings, is_dynamic_initial_run=False):
    """
    Performs standard nested sampling using the likelihood and prior specified
    in settings.

    For more details see "Sampling errors in nested sampling parameter
    estimation" (Higson et al. 2017).

    The run terminates when the estiamted posterior mass contained in the live
    points is less than settings.zv_termination_fraction. The evidence in the
    remaining live points is estimated as

        Z_{live} = average likelihood of live points * prior volume remaining

    Parameters
    ----------
    settings: PerfectNestedSamplingSettings object

    Returns
    -------
    run: dict
        Nested sampling run dictionary containing information about the run's
        posterior samples and a record of the settings used. See docstring for
        perfect_nested_sampling for more details.
    """
    # Reset the random seed to avoid repeated results when multiprocessing.
    np.random.seed()
    if is_dynamic_initial_run:
        nlive_const = settings.ninit
    else:
        nlive_const = settings.nlive_const
    # Sample live points as a 2-dimensional array with columns:
    # [loglikelihood, radial coordinate, logx coordinate, thread label]
    live_points = np.zeros((nlive_const, 4))
    # Thread labels are 1 to nlive_const
    live_points[:, 3] = np.arange(nlive_const) + 1
    live_points[:, 2] = np.log(np.random.random(live_points.shape[0]))
    live_points[:, 1] = settings.r_given_logx(live_points[:, 2])
    live_points[:, 0] = settings.logl_given_r(live_points[:, 1])
    # termination condition variables
    logx_i = 0.0
    logz_dead = -np.inf
    logz_live = (scipy.misc.logsumexp(live_points[:, 0]) + logx_i -
                 np.log(nlive_const))
    # Calculate factor for trapizium rule of geometric series
    t = np.exp(-1.0 / nlive_const)
    logtrapz = np.log(0.5 * ((t ** -1) - t))
    # start the array of dead points
    dead_points_list = []
    while logz_live - np.log(settings.zv_termination_fraction) > logz_dead:
        # add to dead points
        ind = np.where(live_points[:, 0] == live_points[:, 0].min())[0][0]
        dead_points_list.append(copy.deepcopy(live_points[ind, :]))
        # update dead evidence estimates
        logx_i += -1.0 / nlive_const
        logz_dead = scipy.misc.logsumexp((logz_dead, live_points[ind, 0] +
                                          logtrapz + logx_i))
        # add new point
        live_points[ind, 2] += np.log(np.random.random())
        live_points[ind, 1] = settings.r_given_logx(live_points[ind, 2])
        live_points[ind, 0] = settings.logl_given_r(live_points[ind, 1])
        logz_live = (scipy.misc.logsumexp(live_points[:, 0]) + logx_i -
                     np.log(nlive_const))
    points = np.array(dead_points_list)
    # add remaining live points (sorted by increacing likelihood)
    points = np.vstack((points, live_points[np.argsort(live_points[:, 0])]))
    # Create a dictionary representing the nested sampling run
    run = {'settings': settings.get_settings_dict(),
           'logl': points[:, 0],
           'r': points[:, 1],
           'logx': points[:, 2],
           'thread_labels': points[:, 3]}
    # add array of parameter values sampled from the hyperspheres corresponding
    # to the radial coordinate of each point.
    run['theta'] = mf.sample_nsphere_shells(run['r'], settings.n_dim,
                                            settings.dims_to_sample)
    # Add an array of the local numbe of live points - this equals nlive_const
    # until the run terminates, at which point it reduces by 1 as each thread
    # ends.
    run['nlive_array'] = np.zeros(run['logl'].shape[0]) + nlive_const
    for i in range(1, nlive_const):
        run['nlive_array'][-i] = i
    # Get array of data on threads' beginnings and ends. Each starts by
    # sampling the whole prior and ends on one of the final live poins.
    run['thread_min_max'] = np.zeros((nlive_const, 2))
    run['thread_min_max'][:, 0] = np.nan
    run['thread_min_max'][:, 1] = live_points[:, 0]
    return run


# Make dynamic ns run:
# --------------------


def generate_dynamic_run(settings):
    """
    Generate a dynamic nested sampling run.
    For details of the dynamic nested sampling algorithm, see "Dynamic nested
    sampling: an improved algorithm for nested sampling parameter estimation
    and evidence calculation" (Higson et al. 2017).

    The run terminates when the number of samples reachs some limit
    settings.n_samples_max. If this is not set, the function will estimate the
    number of samples that a standard nested sampling run with
    settings.nlive_const would use from the number of samples in the initial
    exploratory run.

    Parameters
    ----------
    settings: PerfectNestedSamplingSettings object
        settings.dynamic_goal controls whether the algorithm aims to increase
        parameter estimation accuracy (dynamic_goal=1), evidence accuracy
        (dynamic_goal=0) or places some weight on both.

    Returns
    -------
    dict
        Nested sampling run dictionary containing information about the run's
        posterior samples and a record of the settings used. See docstring for
        perfect_nested_sampling for more details.
    """
    assert 1 >= settings.dynamic_goal >= 0, "dynamic_goal = " + \
        str(settings.dynamic_goal) + " should be between 0 and 1"
    np.random.seed()  # needed to avoid repeated results when multiprocessing
    # Step 1: inital exploratory standard ns run with ninit live points
    # ----------------------------------------------------------------
    standard_run = generate_standard_run(settings, is_dynamic_initial_run=True)
    # create "samples" array with columns:
    # [logl, r, logx, thread label, change in nlive, params]
    samples = au.samples_array_given_run(standard_run)
    thread_min_max = standard_run['thread_min_max']
    n_samples = samples.shape[0]
    if settings.n_samples_max is None:
        print("dns: settings.n_samples_max is None: estimate from nlive_const")
        # estimate number of likelihood calls available
        n_samples_max = settings.nlive_const * n_samples / settings.ninit
        # Reduce by small factor so dynamic ns uses fewer likelihood calls than
        # normal ns. This factor is a function of the dynamic goal as typically
        # evidence calculations have longer attitional threads than parameter
        # estimation calculations.
        n_samples_max *= 1 - ((1.5 - 0.5 * settings.dynamic_goal) *
                              (settings.nbatch / settings.nlive_const))
    else:
        n_samples_max = settings.n_samples_max
    # Step 2: add samples whereever they are most useful
    # --------------------------------------------------
    while n_samples < n_samples_max:
        importance = point_importance(samples, thread_min_max, settings)
        logl_min_max, logx_min_max = min_max_importance(importance,
                                                        samples,
                                                        settings)
        for _ in range(settings.nbatch):
            # make new thread
            thread_label = thread_min_max.shape[0] + 1
            thread = generate_single_thread(settings,
                                            logx_min_max[1],
                                            thread_label,
                                            logx_start=logx_min_max[0],
                                            keep_final_point=True)
            # update run
            if not np.isnan(logl_min_max[0]):
                start_ind = np.where(samples[:, 0] == logl_min_max[0])[0]
                # check there is exactly one point with the likelihood at which
                # the new thread starts, and note that nlive increases by 1
                assert start_ind.shape == (1,)
                samples[start_ind, 4] += 1
            samples = np.vstack((samples, thread))
            lmm = np.asarray([logl_min_max[0], thread[-1, 0]])
            thread_min_max = np.vstack((thread_min_max, lmm))
        # sort array and update n_samples in preparation for the next iteration
        samples = samples[np.argsort(samples[:, 0])]
        n_samples = samples.shape[0]
    # To compute nlive from the changes in nlive at each step, first find nlive
    # for the first point (= the number of threads which sample from the entire
    # prior)
    run = au.dict_given_samples_array(samples, thread_min_max)
    run['settings'] = settings.get_settings_dict()
    return run


# Dynamic NS helper functions
# ------------------------------


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


def point_importance(samples, thread_min_max, settings, simulate=False):
    run_dict = au.dict_given_samples_array(samples, thread_min_max)
    logw = au.get_logw(run_dict['logl'], run_dict['nlive_array'],
                       simulate=simulate)
    # subtract logw.max() to avoids numerical errors with very small numbers
    w_relative = np.exp(logw - logw.max())
    if settings.dynamic_goal == 0:
        return z_importance(w_relative, run_dict['nlive_array'])
    elif settings.dynamic_goal == 1:
        return p_importance(run_dict['theta'], w_relative,
                            tuned_dynamic_p=settings.tuned_dynamic_p)
    else:
        imp_z = z_importance(w_relative, run_dict['nlive_array'])
        imp_p = p_importance(run_dict['theta'], w_relative,
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


def p_importance(theta, w, tuned_dynamic_p=False, tuning_type='theta1'):
    if tuned_dynamic_p is False:
        return w / w.max()
    else:
        assert tuning_type == 'theta1', 'so far only set up for theta1'
        if tuning_type == 'theta1':
            # extract theta1 values from lrxp
            f = theta[:, 0]
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
