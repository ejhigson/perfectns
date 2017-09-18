#!/usr/bin/python
"""Implements dynamic nested sampling."""

import numpy as np
# perfect nested sampling modules
import pns.analysis_utils as au
import pns.nested_sampling as ns


# Make dynamic ns run:
# --------------------

def get_dynamic_settings(dynamic_zp_weight, settings):
    assert dynamic_zp_weight >= 0 and dynamic_zp_weight <= 1, "dynamic_zp_weight = " + str(dynamic_zp_weight) + " must be in [0,1]"
    dnsd = {
        "nlive_1": settings.nlive_1,
        "nlive_2": settings.nlive_2,
        "n_calls_frac": settings.n_calls_frac,
        "dynamic_keep_final_point": settings.dynamic_keep_final_point,
    }
    dnsd["importance_fraction"] = (settings.dynamic_fraction, settings.dynamic_fraction, dynamic_zp_weight)
    return dnsd


def generate_dynamic_run(nlive_const, dynamic_goal, settings, n_calls_max=None, tuned_dynamic_p=False):
    """
    Generate a dynamic nested sampling run.
    Outputs a list starting with dnsd, followed by a list of live points for each value of fractions.
    Output dnsd contains additional list array "nlive" and list of minimum and maximum logls of threads "thread_logl_min_max".
    The order of the "thread_logl_min_max" list corresponds to the order of the threads.
    An entry "[none, none]" is a thread which runs over the whole range X=0 to X=X_terminate.
    """
    np.random.seed()  # needed to avoid repeated results when multiprocessing - see http://stackoverflow.com/questions/29854398/seeding-random-number-generators-in-parallel-programs
    # Step 1: run all the way through with limited number of threads
    run = [get_dynamic_settings(dynamic_goal, settings)]
    threads, logl_min_max = ns.generate_standard_run(run[0]["nlive_1"], settings, return_logl_min_max=True)
    run.append(threads)
    run[0]['thread_logl_min_max'] = logl_min_max
    n_calls = au.get_n_calls(run)
    if n_calls_max is None:
        # estimate number of likelihood calls available
        n_calls_max = nlive_const * n_calls / run[0]["nlive_1"]
        # reduce by small factor so dynamic ns uses fewer likelihood calls than normal ns
        # this factor is a function of the dynamic goal as typically evidence calculations have longer attitional threads than parameter estimation calculations
        n_calls_max *= (nlive_const - run[0]["nlive_2"] * (1.5 - 0.5 * dynamic_goal)) / nlive_const
    # Step 2: sample the peak until we run out of likelihood calls
    while n_calls < n_calls_max:
        logl_min_max, logx_min_max, n_calls = logl_min_max_given_fraction(run[0]['importance_fraction'], run, settings, tuned_dynamic_p=tuned_dynamic_p)
        nlive_2_count = 0
        if settings.dynamic_keep_final_point:
            n_calls_itr = 0
            while n_calls_itr < n_calls_max * run[0]['n_calls_frac'] or nlive_2_count < run[0]['nlive_2']:
                nlive_2_count += 1
                run[1].append(ns.generate_single_thread(settings, logx_min_max[1], logx_start=logx_min_max[0], keep_final_point=settings.dynamic_keep_final_point, reset_random_seed=False))
                n_calls_itr += run[1][-1].shape[0]
                run[0]['thread_logl_min_max'].append([logl_min_max[0], run[1][-1][-1, 0]])
        else:  # make many threads in a single array with a single logl_min_max to speed stuff up
            logx = []
            while len(logx) < n_calls_max * run[0]['n_calls_frac'] or nlive_2_count < run[0]['nlive_2']:
                nlive_2_count += 1
                logx += ns.generate_thread_logx(logx_min_max[1], logx_start=logx_min_max[0], keep_final_point=settings.dynamic_keep_final_point)
            run[1].append(ns.sample_parameters(np.asarray(logx), settings))
            logl_min_max.append(nlive_2_count)
            run[0]['thread_logl_min_max'].append(logl_min_max)
        # print("n_calls=" + str(n_calls) + ", n_calls_max=" + str(n_calls_max))
    lp, nlive = au.get_lp_nlive(run)
    run[0]['nlive'] = nlive
    return run


# Helper functions
# ----------------

def point_importance(lp, nlive, dynamic_zp_weight, settings, tuned_dynamic_p=False, simulate=False):
    w = au.get_w(lp[:, 0], nlive, settings, simulate=simulate)
    if dynamic_zp_weight == 0:
        return z_importance(w, nlive)
    elif dynamic_zp_weight == 1:
        return p_importance(lp, w, tuned_dynamic_p=tuned_dynamic_p)
    else:
        importance_z = z_importance(w, nlive)
        importance_p = p_importance(lp, w, tuned_dynamic_p=tuned_dynamic_p)
        importance = (importance_z / np.sum(importance_z)) * (1.0 - dynamic_zp_weight)
        importance += (importance_p / np.sum(importance_p)) * dynamic_zp_weight
        return importance / importance.max()


def z_importance(w, nlive, exact=False):
    importance = np.cumsum(w)
    importance = importance.max() - importance
    if exact:
        importance = importance * (((nlive ** 2) - 3) * (nlive ** 1.5)) / (((nlive + 1) ** 3) * ((nlive + 2) ** 1.5))
        importance += w * (nlive ** 0.5) / ((nlive + 2) ** 1.5)
    else:
        importance *= 1.0 / nlive
    return importance / importance.max()


def p_importance(lp, w, tuned_dynamic_p=False):
    if tuned_dynamic_p is True:
        f = lp[:, 1]
        fabs = np.absolute(f - (np.sum(f * w) / np.sum(w)))
        importance = fabs * w
        # n = int(np.ceil(w.shape[0] / 100.0))
        # importance = np.zeros(w.shape[0])
        # for i, _ in enumerate(importance):
        #     nmin = i - n
        #     nmax = i + n + 1
        #     if nmin < 0:
        #         nmin = 0
        #     if nmax > importance.shape[0] - 1:
        #         nmax = importance.shape[0] - 1
        #     importance[i] = np.mean(fabs[nmin:nmax]) * w[i]
        return importance / importance.max()
    else:
        return w / w.max()


def logl_min_max_given_fraction(fraction, run, settings, tuned_dynamic_p=False):
    assert fraction[0] > 0. and fraction[0] < 1., "logl_min_max_given_fraction: importance_fraction = " + str(fraction) + " must have min max fractions in [0, 1]"
    assert fraction[1] > 0. and fraction[1] < 1., "logl_min_max_given_fraction: importance_fraction = " + str(fraction) + " must have min max fractions in [0, 1]"
    lp, nlive = au.get_lp_nlive(run)
    n_calls = lp.shape[0]
    importance = point_importance(lp, nlive, fraction[2], settings, tuned_dynamic_p=tuned_dynamic_p)
    # where to start the additional threads:
    ind_start = np.where(importance > fraction[0])[0]
    if ind_start[0] == 0:  # start from sampling the whole prior
        logl_min = None
        logx_min = 0
    else:
        logl_min = lp[:, 0][ind_start[0] - 1]
        # use lookup to avoid float errors and to not need inverse function
        ind = np.where(lp[:, 0] == logl_min)[0]
        assert ind.shape[0] == 1, "Should be one unique match for logl=logl_min=" + str(logl_min) + ". Instead we have matches at indexes " + str(ind) + " of the lp array (shape " + str(lp.shape) + ")"
        logx_min = lp[ind[0], 2]
    # where to start the additional threads:
    ind_end = np.where(importance > fraction[1])[0]
    if ind_end[-1] == lp[:, 0].shape[0] - 1:
        if settings.dynamic_keep_final_point:
            logl_max = lp[-1, 0]
            logx_max = lp[-1, 2]
        else:
            # if this is the last point and we are not keeping final points we need to allow samples to go a bit further.
            # here we use the biggest shrinkage (smallest number) of 3 randomly generated shrinkage ratios to model what would happen if we were keeping points.
            logx_max = lp[-1, 2] + np.log(np.min(np.random.random(3)))
            logl_max = settings.likelihood_prior.logl_given_logx(logx_max)
    else:
        logl_max = lp[:, 0][(ind_end[-1] + 1)]
        # use lookup to avoid float errors and to not need inverse function
        ind = np.where(lp[:, 0] == logl_max)[0]
        assert ind.shape[0] == 1, "Should be one unique match for logl=logl_min=" + str(logl_max) + ".\n Instead we have matches at indexes " + str(ind) + " of the lp array (shape " + str(lp.shape) + ")"
        logx_max = lp[ind[0], 2]
    return [logl_min, logl_max], [logx_min, logx_max], n_calls
