#!/usr/bin/python
"""
Functions used to generate the results in:
'Dynamic nested sampling: an improved algorithm for nested sampling parameter
estimation and evidence calculation' (Higson et al. 2017).
"""

import pandas as pd
import numpy as np
# perfect nested sampling modules
import PerfectNestedSampling.save_load_utils as slu
import PerfectNestedSampling.parallelised_wrappers as pw
import PerfectNestedSampling.analysis_utils as au
import PerfectNestedSampling.maths_functions as mf
import PerfectNestedSampling.estimators as e


@slu.timing_decorator
def get_dynamic_results(n_run, dynamic_goals, funcs_in, settings, **kwargs):
    """
    Generate results tables showing the standard deviations of the results of
    repeated calculations and efficiency gains (ratios of variances of results
    calculations) from different dynamic goals.

    Results are output in pandas data frames, and are also saved in latex
    format in a .txt file.

    This function was used for Tables 1, 2, 3 and 4, as well as to generate the
    results shown in figures 6 and 7 of 'Dynamic nested sampling: an improved
    algorithm for nested sampling parameter estimation and evidence
    calculation' (Higson et al. 2017).
    """
    load = kwargs.get('load', True)
    save = kwargs.get('save', True)
    save_dir = kwargs.get('save_dir', 'data')
    parallelise = kwargs.get('parallelise', True)
    tuned_dynamic_ps = kwargs.get('tuned_dynamic_ps', None)
    # make save_name
    extra_root = 'dynamic_test'
    for dg in dynamic_goals:
        extra_root += '_' + str(dg)
    save_root = slu.data_save_name(settings, n_run, extra_root=extra_root,
                                   include_dg=False)
    save_file = save_dir + '/' + save_root + '.dat'
    print('Running get_dynamic_results: save file is')
    print(save_file)
    # try loading results
    if load:
        try:
            results = pd.read_pickle(save_file)
            print('Loading results from file:\n' + save_file)
            return results
        except OSError:
            pass
    # start function
    # --------------
    # get info on the number of samples taken in each run as well
    estimator_list = [e.nSamplesEstimator()] + funcs_in
    func_names = []
    for func in estimator_list:
        func_names.append(func.name)
    df_dict = {}
    values_list = []
    if type(settings.prior).__name__ == 'gaussian_cached':
        settings.prior.check_cache(settings.n_dim)
    method_names = []
    for i, dynamic_goal in enumerate(dynamic_goals):
        # set up settings
        settings.dynamic_goal = dynamic_goal
        if tuned_dynamic_ps is not None:
            settings.tuned_dynamic_p = tuned_dynamic_ps[i]
        print('dynamic_goal = ' + str(settings.dynamic_goal))
        # if we have already done the standard calculation, set n_samples_max
        # for dynamic calculations so it is slightly smaller than the number
        # of samples the standard calculation used (for comparison of
        # performance)
        if settings.dynamic_goal is not None and 'standard' in df_dict:
            n_samples_max = df_dict['standard']['n_samples']['mean']
            # This factor is a function of the dynamic goal as typically
            # evidence calculations have longer attitional threads than
            # parameter estimation calculations.
            n_samples_max *= 1 - ((1.5 - 0.5 * settings.dynamic_goal) *
                                  (settings.nbatch / settings.nlive_const))
            n_samples_max = int(n_samples_max)
            print('given standard used ' +
                  str(df_dict['standard']['n_samples']['mean']) +
                  ' calls, set n_samples_max=' + str(n_samples_max))
            settings.n_samples_max = n_samples_max
        # get a name for this calculation method
        if dynamic_goal is None:
            method_names.append('standard')
        else:
            method_names.append('dyn ' + str(settings.dynamic_goal))
            if settings.tuned_dynamic_p is True:
                method_names[-1] += ' tuned'
        # generate runs and get results
        run_list = pw.get_run_data(settings, n_run, parallelise=parallelise,
                                   load=load, save=save)
        values = pw.func_on_runs(au.run_estimators, run_list, estimator_list,
                                 parallelise=parallelise)
        df = mf.get_df_row_summary(values, func_names)
        df_dict[method_names[-1]] = df
        values_list.append(values)
        del run_list
    # analyse data
    # ------------
    # find performance gain (proportional to ratio of errors squared)
    for key, df in df_dict.items():
        std_ratio = df_dict['standard'].loc['std'] / df.loc['std']
        std_ratio_unc = mf.array_ratio_std(df_dict['standard'].loc['std'],
                                           df_dict['standard'].loc['std_unc'],
                                           df.loc['std'],
                                           df.loc['std_unc'])
        df_dict[key].loc['gain'] = std_ratio ** 2
        df_dict[key].loc['gain_unc'] = 2 * std_ratio * std_ratio_unc
        # We want to see the number of samples (not its std or gain), so set
        # every row of n_samples column equal to the mean number of samples
        df_dict[key]['n_samples']['std'] = df['n_samples']['mean']
        df_dict[key]['n_samples']['std_unc'] = df['n_samples']['mean_unc']
        df_dict[key]['n_samples']['gain'] = df['n_samples']['mean']
        df_dict[key]['n_samples']['gain_unc'] = df['n_samples']['mean_unc']
    for key, df in df_dict.items():
        # make uncertainties appear in seperate columns
        df_dict[key] = mf.df_unc_rows_to_cols(df)
        # edit keys to show method names
        df_dict[key] = df_dict[key].set_index(df_dict[key].index + ' ' + key)
    results = pd.concat(df_dict.values())
    # Add true values to test that nested sampling is working correctly - these
    # should be close to the mean calculation values
    results = results.append(e.get_true_estimator_values(estimator_list,
                                                         settings))
    # Sort the rows and columns into the order needed for the paper
    row_order = ['true values']
    for pref in ['mean', 'std', 'gain']:
        for mn in method_names:
            row_order.append(pref + ' ' + mn)
    results = results.reindex(row_order)
    cols = list(results)
    cols.insert(0, cols.pop(cols.index('n_samples')))
    cols.insert(1, cols.pop(cols.index('n_samples_unc')))
    results = results.loc[:, cols]
    if save:
        # save the results data frame
        results.to_pickle(save_file)
        print('Results saved to:\n' + save_file)
        # save results in latex format
        latex_save_file = save_dir + '/' + save_root + '_latex.txt'
        latex_df = slu.latex_format_df(results, cols=None, rows=None,
                                       dp_list=None)
        with open(latex_save_file, 'w') as text_file:
            print(latex_df.to_latex(), file=text_file)
    return results


@slu.timing_decorator
def get_bootstrap_results(n_run, n_simulate, estimator_list, settings,
                          **kwargs):
    """
    Generate results tables showing the standard deviations of the results of
    repeated calculations and estimated sampling errors from bootstrap
    resampling.

    Results are output in pandas data frames, and are also saved in latex
    format in a .txt file.

    This function was used for Table 5 in 'Dynamic nested sampling: an improved
    algorithm for nested sampling parameter estimation and evidence
    calculation' (Higson et al. 2017).
    """
    load = kwargs.get('load', True)
    save = kwargs.get('save', True)
    save_dir = kwargs.get('save_dir', 'data')
    ninit_sep = kwargs.get('ninit_sep', False)
    parallelise = kwargs.get('parallelise', True)
    add_sim_method = kwargs.get('add_sim_method', False)
    n_simulate_ci = kwargs.get('n_simulate_ci', n_simulate)
    n_run_ci = kwargs.get('n_run_ci', n_run)
    cred_int = kwargs.get('cred_int', 0.95)
    # make save_name
    extra_root = ('bootstrap_results_' + str(n_simulate) + 'nsim_' +
                  str(ninit_sep) + 'sep')
    save_root = slu.data_save_name(settings, n_run, extra_root=extra_root)
    save_file = save_dir + '/' + save_root + '.dat'
    print('Running get_bootstrap_results: save file is')
    print(save_file)
    # try loading results
    if load:
        try:
            results = pd.read_pickle(save_file)
            print('Loading results from file:\n' + save_file)
            return results
        except OSError:
            pass
    # start function
    e_names = []
    for est in estimator_list:
        e_names.append(est.name)
    # generate runs
    run_list = pw.get_run_data(settings, n_run, save=save, load=load,
                               parallelise=parallelise)
    # Add true values to test that nested sampling is working correctly - these
    # should be close to the mean calculation values
    results = e.get_true_estimator_values(estimator_list, settings)
    # get mean and std of repeated calculations
    rep_values = pw.func_on_runs(au.run_estimators, run_list, estimator_list,
                                 parallelise=parallelise)
    rep_df = mf.get_df_row_summary(rep_values, e_names)
    results = results.append(rep_df.set_index('repeats ' +
                                              rep_df.index.astype(str)))
    # get bootstrap std estimate
    bs_values = pw.func_on_runs(au.run_std_bootstrap, run_list,
                                estimator_list, n_simulate=n_simulate,
                                parallelise=parallelise)
    bs_df = mf.get_df_row_summary(bs_values, e_names)
    # Get the mean bootstrap std estimate as a fraction of the std measured
    # from repeated calculations.
    results.loc['bs std'] = bs_df.loc['mean'] / results.loc['repeats std']
    bs_std_ratio_unc = mf.array_ratio_std(bs_df.loc['mean'],
                                          bs_df.loc['mean_unc'],
                                          results.loc['repeats std'],
                                          results.loc['repeats std_unc'])
    results.loc['bs std_unc'] = bs_std_ratio_unc
    # multiply by 100 to express as a percentage
    results.loc['bs var'] = 100 * bs_df.loc['std'] / bs_df.loc['mean']
    results.loc['bs var_unc'] = 100 * bs_df.loc['std_unc'] / bs_df.loc['mean']
    if add_sim_method:
        # get std from simulation estimate
        sim_values = pw.func_on_runs(au.run_std_simulate, run_list,
                                     estimator_list, n_simulate=n_simulate,
                                     parallelise=parallelise)
        sim_df = mf.get_df_row_summary(sim_values, e_names)
        # get mean simulation std estimate a ratio to the std from repeats
        results.loc['sim std'] = (sim_df.loc['mean'] /
                                  results.loc['repeats std'])
        sim_std_ratio_unc = mf.array_ratio_std(sim_df.loc['mean'],
                                               sim_df.loc['mean_unc'],
                                               results.loc['repeats std'],
                                               results.loc['repeats std_unc'])
        results.loc['sim std_unc'] = sim_std_ratio_unc
        # multiply by 100 to express as a percentage
        results.loc['sim var'] = 100 * sim_df.loc['std'] / sim_df.loc['mean']
        results.loc['sim var_unc'] = 100 * (sim_df.loc['std_unc'] /
                                            sim_df.loc['mean'])
    # get bootstrap CI estimates
    bs_cis = pw.func_on_runs(au.run_ci_bootstrap, run_list[:n_run_ci],
                             estimator_list, n_simulate=n_simulate_ci,
                             cred_int=cred_int, parallelise=parallelise)
    bs_ci_df = mf.get_df_row_summary(bs_cis, e_names)
    results.loc['bs ' + str(cred_int) + ' CI'] = bs_ci_df.loc['mean']
    results.loc['bs ' + str(cred_int) + ' CI_unc'] = bs_ci_df.loc['mean_unc']
    # add coverage for +- 1 bootstrap std estimate
    max_value = rep_df.loc['mean'].values + bs_df.loc['mean'].values
    min_value = rep_df.loc['mean'].values - bs_df.loc['mean'].values
    coverage = np.zeros(rep_values.shape[0])
    for i, _ in enumerate(coverage):
        ind = np.where((rep_values[i, :] > min_value[i]) &
                       (rep_values[i, :] < max_value[i]))
        coverage[i] = ind[0].shape[0] / rep_values.shape[1]
    # multiply by 100 to express as a percentage
    results.loc['bs +-1std cov'] = coverage * 100
    # add conf interval coverage
    max_value = results.loc['bs ' + str(cred_int) + ' CI'].values
    ci_coverage = np.zeros(rep_values.shape[0])
    for i, _ in enumerate(coverage):
        ind = np.where(rep_values[i, :] < max_value[i])
        ci_coverage[i] = ind[0].shape[0] / rep_values.shape[1]
    # multiply by 100 to express as a percentage
    results.loc['bs ' + str(cred_int) + ' CI cov'] = ci_coverage * 100
    results = mf.df_unc_rows_to_cols(results)
    if save:
        # save the results data frame
        results.to_pickle(save_file)
        print('Results saved to:\n' + save_file)
        # save results in latex format
        latex_save_file = save_dir + '/' + save_root + '_latex.txt'
        latex_df = slu.latex_format_df(results, cols=None, rows=None,
                                       dp_list=None)
        with open(latex_save_file, 'w') as text_file:
            print(latex_df.to_latex(), file=text_file)
    return results
