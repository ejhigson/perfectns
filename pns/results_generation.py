#!/usr/bin/python
"""Functions for generating results."""

import pandas as pd
import numpy as np
# perfect nested sampling modules
import pns.save_load_utils as slu
import pns.parallelised_wrappers as pw
import pns.analysis_utils as au
import pns.maths_functions as mf
import pns.estimators as e


def get_dynamic_results(n_run, dynamic_goals, funcs_in, settings, **kwargs):
    """
    Generate results using different dynamic goals and output a pandas data
    frame containing standard deviations and performance gains.
    """
    load = kwargs.get('load', True)
    save = kwargs.get('save', True)
    parallelise = kwargs.get('parallelise', True)
    reduce_n_calls_max_frac = kwargs.get('reduce_n_calls_max_frac', 0.02)
    tuned_dynamic_ps = kwargs.get('tuned_dynamic_ps', None)
    # make save_name
    extra_root = "dynamic_test"
    for dg in dynamic_goals:
        extra_root += "_" + str(dg)
    save_file = slu.data_save_name(settings, n_run, extra_root=extra_root,
                                   include_dg=False)
    save_file = 'data/' + save_file + '.dat'
    # try loading results
    if load:
        try:
            results = pd.read_pickle(save_file)
            print("Loading results from file:\n" + save_file)
            return results
        except OSError:
            pass
    # start function
    # --------------
    # get info on the number of samples taken in each run as well
    funcs_list = [e.n_samplesEstimator()] + funcs_in
    func_names = []
    for func in funcs_list:
        func_names.append(func.name)
    df_dict = {}
    values_list = []
    for i, dynamic_goal in enumerate(dynamic_goals):
        settings.dynamic_goal = dynamic_goal
        if tuned_dynamic_ps is not None:
            settings.tuned_dynamic_p = tuned_dynamic_ps[i]
        print("dynamic_goal = " + str(settings.dynamic_goal))
        run_list = pw.get_run_data(settings, n_run, parallelise=parallelise,
                                   load=load, save=save)
        values = pw.func_on_runs(au.run_estimators, run_list, funcs_list)
        if dynamic_goal is None:
            key_i = "standard"
        else:
            key_i = "dyn " + str(settings.dynamic_goal)
            if settings.tuned_dynamic_p is True:
                key_i += " tuned"
        df = mf.get_df_row_summary(values, func_names)
        df_dict[key_i] = df
        if (settings.dynamic_goal is None and settings.n_calls_max is None
                and i == 0):
            n_calls_max = int(df['n_samples']['mean'] *
                              (1.0 - reduce_n_calls_max_frac))
            print("given standard used " + str(df['n_samples']['mean']) +
                  " calls, set n_calls_max=" + str(n_calls_max))
        values_list.append(values)
        del run_list
    # analyse data
    # ------------
    # find performance gain (proportional to ratio of errors squared)
    for key, df in df_dict.items():
        std_ratio = df_dict["standard"].loc["std"] / df.loc["std"]
        std_ratio_unc = mf.array_ratio_std(df_dict["standard"].loc["std"],
                                           df_dict["standard"].loc["std_unc"],
                                           df.loc["std"],
                                           df.loc["std_unc"])
        df_dict[key].loc["gain"] = std_ratio ** 2
        df_dict[key].loc["gain_unc"] = 2 * std_ratio * std_ratio_unc
    # make uncertainties appear in seperate columns
    for key, df in df_dict.items():
        df_dict[key] = mf.df_unc_rows_to_cols(df)
        df_dict[key]["dynamic_goal"] = [key] * df_dict[key].shape[0]
    results = pd.concat(df_dict.values())
    # make the calc column catagorical with a custom ordering
    results['calc_type'] = pd.Categorical(results.index,
                                          ['mean', 'std', 'gain'])
    results.sort_values(["calc_type", "dynamic_goal"], inplace=True)
    del results['calc_type']
    # put the dynamic goal column first
    cols = list(results)
    cols.insert(0, cols.pop(cols.index('dynamic_goal')))
    cols.insert(1, cols.pop(cols.index('n_samples')))
    cols.insert(2, cols.pop(cols.index('n_samples_unc')))
    results = results.loc[:, cols]
    if save:
        results.to_pickle(save_file)
        print("Results saved to:\n" + save_file)
    return results


def get_bootstrap_results(n_run, n_simulate, estimator_list, settings,
                          **kwargs):
    """
    tbc
    """
    load = kwargs.get('load', True)
    save = kwargs.get('save', True)
    add_sim_method = kwargs.get('add_sim_method', False)
    n_simulate_ci = kwargs.get('n_simulate_ci', n_simulate)
    n_run_ci = kwargs.get('n_run_ci', n_run)
    cred_int = kwargs.get('cred_int', 0.95)
    # make save_name
    extra_root = "bootstrap_results_" + str(n_simulate) + "nsim"
    save_file = slu.data_save_name(settings, n_run, extra_root=extra_root)
    save_file = 'data/' + save_file + '.dat'
    # try loading results
    if load:
        try:
            results = pd.read_pickle(save_file)
            print("Loading results from file:\n" + save_file)
            return results
        except OSError:
            pass
    # start function
    e_names = []
    for est in estimator_list:
        e_names.append(est.name)
    # generate runs
    run_list = pw.get_run_data(settings, n_run, save=save, load=load)
    # get std from repeats
    rep_values = pw.func_on_runs(au.run_estimators, run_list, estimator_list)
    rep_df = mf.get_df_row_summary(rep_values, e_names)
    results = rep_df.set_index('repeats ' + rep_df.index.astype(str))
    # get bootstrap std estimate
    bs_values = pw.func_on_runs(au.run_std_bootstrap, run_list,
                                estimator_list, n_simulate=n_simulate)
    bs_df = mf.get_df_row_summary(bs_values, e_names)
    results.loc['bs std'] = bs_df.loc['mean']
    results.loc['bs std_unc'] = bs_df.loc['mean_unc']
    if add_sim_method:
        # get std from simulation estimate
        sim_values = pw.func_on_runs(au.run_std_simulate, run_list,
                                     estimator_list, n_simulate=n_simulate)
        sim_df = mf.get_df_row_summary(sim_values, e_names)
        results.loc['sim std'] = sim_df.loc['mean']
        results.loc['sim std_unc'] = sim_df.loc['mean_unc']
    # get bootstrap CI estimates
    bs_cis = pw.func_on_runs(au.run_ci_bootstrap, run_list[:n_run_ci],
                             estimator_list, n_simulate=n_simulate_ci,
                             cred_int=cred_int)
    bs_ci_df = mf.get_df_row_summary(bs_cis, e_names)
    results.loc['bs ' + str(cred_int) + ' CI'] = bs_ci_df.loc['mean']
    results.loc['bs ' + str(cred_int) + ' CI_unc'] = bs_ci_df.loc['mean_unc']
    # add +- 1 std coverage
    max_value = rep_df.loc['mean'].values + results.loc['bs std'].values
    min_value = rep_df.loc['mean'].values - results.loc['bs std'].values
    coverage = np.zeros(rep_values.shape[0])
    for i, _ in enumerate(coverage):
        ind = np.where((rep_values[i, :] > min_value[i]) &
                       (rep_values[i, :] < max_value[i]))
        coverage[i] = ind[0].shape[0] / rep_values.shape[1]
    results.loc['bs +-1std cov'] = coverage
    # add conf interval coverage
    max_value = results.loc['bs ' + str(cred_int) + ' CI'].values
    ci_coverage = np.zeros(rep_values.shape[0])
    for i, _ in enumerate(coverage):
        ind = np.where(rep_values[i, :] < max_value[i])
        ci_coverage[i] = ind[0].shape[0] / rep_values.shape[1]
    results.loc['bs ' + str(cred_int) + ' CI cov'] = ci_coverage
    results = mf.df_unc_rows_to_cols(results)
    if save:
        results.to_pickle(save_file)
        print("Results saved to:\n" + save_file)
    return results
