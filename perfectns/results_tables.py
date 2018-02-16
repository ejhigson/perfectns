#!/usr/bin/env python
"""
Functions used to generate the results in:
'Dynamic nested sampling: an improved algorithm for nested sampling parameter
estimation and evidence calculation' (Higson et al. 2017).
"""

import pandas as pd
import numpy as np
import nestcheck.io_utils as iou
import nestcheck.analyse_run as ar
import perfectns.parallelised_wrappers as pw
import perfectns.maths_functions as mf
import perfectns.estimators as e


@iou.timing_decorator
def get_dynamic_results(n_run, dynamic_goals_in, estimator_list_in, settings,
                        **kwargs):
    """
    Generate data frame showing the standard deviations of the results of
    repeated calculations and efficiency gains (ratios of variances of results
    calculations) from different dynamic goals. To make the comparison fair,
    for dynamic nested sampling settings.n_samples_max is set to slightly below
    the mean number of samples used by standard nested sampling.

    This function was used for Tables 1, 2, 3 and 4, as well as to generate the
    results shown in figures 6 and 7 of 'Dynamic nested sampling: an improved
    algorithm for nested sampling parameter estimation and evidence
    calculation' (Higson et al. 2017). See the paper for a more detailed
    description.

    Parameters
    ----------
    n_run: int
        how many runs to use
    dynamic_goals_in: list of floats
        which dynamic goals to test
    estimator_list_in: list of estimator objects
    settings: PerfectNSSettings object
    load: bool, optional
        should run data and results be loaded if available?
    save: bool, optional
        should run data and results be saved?
    overwrite_existing: bool, optional
        if a file exists already but we generate new run data, should we
        overwrite the existing file when saved?
    parallelise: bool, optional
    tuned_dynamic_ps: list of bools, same length as dynamic_goals_in, optional
    max_workers: int or None, optional
        Number of processes.
        If max_workers is None then concurrent.futures.ProcessPoolExecutor
        defaults to using the number of processors of the machine.
        N.B. If max_workers=None and running on supercomputer clusters with
        multiple nodes, this may default to the number of processors on a
        single node and therefore there will be no speedup from multiple
        nodes (must specify manually in this case).

    Returns
    -------
    results: pandas data frame
        results data frame.
        Contains two columns for each estimator - the second column (with
        '_unc' appended to the title) shows the numerical uncertainty in the
        first column.
        Contains rows:
            true values: analytical values of estimators for this likelihood
                and posterior if available
            mean [dynamic goal]: mean calculation result for standard nested
                sampling and dynamic nested sampling with each input dynamic
                goal.
            std [dynamic goal]: standard deviation of results for standard
                nested sampling and dynamic nested sampling with each input
                dynamic goal.
            gain [dynamic goal]: the efficiency gain (computational speedup)
                from dynamic nested sampling compared to standard nested
                sampling. This equals (variance of standard results) /
                (variance of dynamic results); see the dynamic nested
                sampling paper for more details.
    """
    load = kwargs.pop('load', False)
    save = kwargs.pop('save', False)
    save_dir = kwargs.pop('save_dir', 'data')
    max_workers = kwargs.pop('max_workers', None)
    parallelise = kwargs.pop('parallelise', True)
    tuned_dynamic_ps = kwargs.pop('tuned_dynamic_ps', None)
    overwrite_existing = kwargs.pop('overwrite_existing', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    # Store the input settings.n_samples_max as we are going to edit it
    n_samples_max_in = settings.n_samples_max
    # First we run a standard nested sampling run for comparison:
    dynamic_goals = [None] + dynamic_goals_in
    if tuned_dynamic_ps is not None:
        tuned_dynamic_ps = [False] + tuned_dynamic_ps
    # make save_name
    extra_root = 'dynamic_test'
    for dg in dynamic_goals:
        extra_root += '_' + str(dg)
    save_root = iou.data_save_name(settings, n_run, extra_root=extra_root,
                                   include_dg=False)
    save_file = save_dir + '/' + save_root + '.pkl'
    # try loading results
    if load:
        try:
            results = pd.read_pickle(save_file)
            print('get_dynamic_results: loading results from\n' + save_file)
            return results
        except OSError:
            pass
    # start function
    # --------------
    # get info on the number of samples taken in each run as well
    estimator_list = [e.CountSamples()] + estimator_list_in
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
        # of samples the standard calculation used to ensure a fair comparison
        # of performance. Otherwise dynamic nested sampling will end up using
        # more samples than standard nested sampling as it does not terminate
        # until after the number of samples is greater than n_samples_max.
        if settings.dynamic_goal is not None and 'standard' in df_dict:
            n_samples_max = df_dict['standard'].loc['mean', 'n_samples']
            # This factor is a function of the dynamic goal as typically
            # evidence calculations have longer additional threads than
            # parameter estimation calculations.
            reduce_factor = 1 - ((1.5 - 0.5 * settings.dynamic_goal) *
                                 (settings.nbatch / settings.nlive_const))
            settings.n_samples_max = int(n_samples_max * reduce_factor)
        # get a name for this calculation method
        if dynamic_goal is None:
            method_names.append('standard')
        else:
            method_names.append('dynamic ' + str(settings.dynamic_goal))
            if settings.tuned_dynamic_p is True:
                method_names[-1] += ' tuned'
        # generate runs and get results
        run_list = pw.get_run_data(settings, n_run, parallelise=parallelise,
                                   load=load, save=save,
                                   max_workers=max_workers,
                                   overwrite_existing=overwrite_existing)
        values = pw.func_on_runs(ar.run_estimators, run_list, estimator_list,
                                 max_workers=max_workers,
                                 parallelise=parallelise)
        df = mf.get_df_row_summary(values, func_names)
        df_dict[method_names[-1]] = df
        values_list.append(values)
        del run_list
    # Restore settings.n_samples_max to its original value to ensure the
    # function does not edit the settings object
    settings.n_samples_max = n_samples_max_in
    # analyse data
    # ------------
    # find performance gain (proportional to ratio of errors squared)
    for key, df in df_dict.items():
        std_ratio = df_dict['standard'].loc['std'] / df.loc['std']
        std_ratio_unc = mf.array_ratio_std(df_dict['standard'].loc['std'],
                                           df_dict['standard'].loc['std_unc'],
                                           df.loc['std'],
                                           df.loc['std_unc'])
        df_dict[key].loc['efficiency gain'] = std_ratio ** 2
        df_dict[key].loc['efficiency gain_unc'] = 2 * std_ratio * std_ratio_unc
        # We want to see the number of samples (not its std or gain), so set
        # every row of n_samples column equal to the mean number of samples
        df_dict[key].loc['std', 'n_samples'] = df.loc['mean', 'n_samples']
        df_dict[key].loc['efficiency gain', 'n_samples'] = df.loc['mean',
                                                                  'n_samples']
    for key, df in df_dict.items():
        # make uncertainties appear in separate columns
        df_dict[key] = mf.df_unc_rows_to_cols(df)
        # Delete uncertainty on mean number of samples as not interested in it
        del df_dict[key]['n_samples_unc']
        # edit keys to show method names
        df_dict[key] = df_dict[key].set_index(df_dict[key].index + ' ' + key)
    # Combine all results into one data frame
    results = pd.concat(df_dict.values())
    # Get true values to test that nested sampling is working correctly - the
    # mean calculation values should be close to these
    true_values = e.get_true_estimator_values(estimator_list, settings)
    for est in estimator_list:
        true_values[est.name + '_unc'] = 0
    # Get the correct column order before concatenating true_values otherwise
    # column order is messed up
    col_order = list(results)
    col_order.insert(0, col_order.pop(col_order.index('n_samples')))
    # add true values and reorder
    results = pd.concat((results, true_values))
    results = results.loc[:, col_order]
    # Sort the rows into the order we want for the paper
    row_order = ['true values']
    for pref in ['mean', 'std', 'efficiency gain']:
        for name in method_names:
            row_order.append(pref + ' ' + name)
    results = results.reindex(row_order)
    # get rid of the 'efficiency gain standard' row as it compares standard
    # nested sampling to its self and so always has value 1.
    results.drop('efficiency gain standard', inplace=True)
    if save:
        # save the results data frame
        print('get_dynamic_results: saving results to\n' + save_file)
        results.to_pickle(save_file)
    return results


@iou.timing_decorator
def get_bootstrap_results(n_run, n_simulate, estimator_list, settings,
                          **kwargs):
    """
    Generate data frame showing the standard deviations of the results of
    repeated calculations and estimated sampling errors from bootstrap
    resampling.

    This function was used for Table 5 in 'Dynamic nested sampling: an improved
    algorithm for nested sampling parameter estimation and evidence
    calculation' (Higson et al. 2017). See the paper for more details.

    Parameters
    ----------
    n_run: int
        how many runs to use
    n_simulate: int
        how many times to resample the nested sampling run in each bootstrap
        standard deviation estimate.
    estimator_list: list of estimator objects
    settings: PerfectNSSettings object
    load: bool, optional
        should run data and results be loaded if available?
    save: bool, optional
        should run data and results be saved?
    parallelise: bool, optional
    add_sim_method: bool, optional
        should we also calculate standard deviations using the simulated
        weights method for comparison with bootstrap resampling? This method is
        inaccurate for parameter estimation.
    n_simulate_ci: int, optional
        how many times to resample the nested sampling run in each bootstrap
        credible interval estimate. These may require more simulations than the
        standard deviation estimate.
    n_run_ci: int, optional
        how many runs to use for each credible interval estimate. You may want
        to set this to lower than n_run if n_simulate_ci is large as otherwise
        the credible interval estimate may take a long time.
    cred_int: float, optional
        one-tailed credible interval to calculate
    max_workers: int or None, optional
        Number of processes.
        If max_workers is None then concurrent.futures.ProcessPoolExecutor
        defaults to using the number of processors of the machine.
        N.B. If max_workers=None and running on supercomputer clusters with
        multiple nodes, this may default to the number of processors on a
        single node and therefore there will be no speedup from multiple
        nodes (must specify manually in this case).

    Returns
    -------
    results: pandas data frame
        results data frame.
        Contains two columns for each estimator - the second column (with
        '_unc' appended to the title) shows the numerical uncertainty in the
        first column.
        Contains rows:
            true values: analytical values of estimators for this likelihood
                and posterior if available
            repeats mean: mean calculation result
            repeats std: standard deviation of calculation results
            bs std / repeats std: mean bootstrap standard deviation estimate as
                a fraction of the standard deviation of repeated results.
            bs estimate % variation: standard deviation of bootstrap estimates
                as a percentage of the mean estimate.
            [only if add sim method is True]:
                sim std / repeats std: as for 'bs std / repeats std' but with
                    simulation method standard deviation estimates.
                sim estimate % variation: as for 'bs estimate % variation' but
                    with simulation method standard deviation estimates.
            bs [cred_int] CI: mean bootstrap credible interval estimate.
            bs +-1std % coverage: % of calculation results falling within +- 1
                mean bootstrap standard deviation estimate of the mean.
            bs [cred_int] CI % coverage: % of calculation results which are
                less than the mean bootstrap credible interval estimate.
    """
    load = kwargs.pop('load', False)
    save = kwargs.pop('save', False)
    max_workers = kwargs.pop('max_workers', None)
    save_dir = kwargs.pop('save_dir', 'data')
    ninit_sep = kwargs.pop('ninit_sep', False)
    parallelise = kwargs.pop('parallelise', True)
    add_sim_method = kwargs.pop('add_sim_method', False)
    n_simulate_ci = kwargs.pop('n_simulate_ci', n_simulate)
    n_run_ci = kwargs.pop('n_run_ci', n_run)
    cred_int = kwargs.pop('cred_int', 0.95)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    # make save_name
    extra_root = ('bootstrap_results_' + str(n_simulate) + 'nsim_' +
                  str(ninit_sep) + 'sep')
    save_root = iou.data_save_name(settings, n_run, extra_root=extra_root)
    save_file = save_dir + '/' + save_root + '.pkl'
    # try loading results
    if load:
        try:
            results = pd.read_pickle(save_file)
            print('get_bootstrap_results: loading results from\n' + save_file)
            return results
        except OSError:
            pass
    # start function
    e_names = []
    for est in estimator_list:
        e_names.append(est.name)
    # generate runs
    run_list = pw.get_run_data(settings, n_run, save=save, load=load,
                               max_workers=max_workers,
                               parallelise=parallelise)
    # Add true values to test that nested sampling is working correctly - these
    # should be close to the mean calculation values
    results = e.get_true_estimator_values(estimator_list, settings)
    results.loc['true values_unc'] = 0
    # get mean and std of repeated calculations
    rep_values = pw.func_on_runs(ar.run_estimators, run_list, estimator_list,
                                 max_workers=max_workers,
                                 parallelise=parallelise)
    rep_df = mf.get_df_row_summary(rep_values, e_names)
    results = results.append(rep_df.set_index('repeats ' +
                                              rep_df.index.astype(str)))
    # get bootstrap std estimate
    bs_values = pw.func_on_runs(ar.run_std_bootstrap, run_list,
                                estimator_list, n_simulate=n_simulate,
                                max_workers=max_workers,
                                parallelise=parallelise)
    bs_df = mf.get_df_row_summary(bs_values, e_names)
    # Get the mean bootstrap std estimate as a fraction of the std measured
    # from repeated calculations.
    results.loc['bs std / repeats std'] = (bs_df.loc['mean'] /
                                           results.loc['repeats std'])
    bs_std_ratio_unc = mf.array_ratio_std(bs_df.loc['mean'],
                                          bs_df.loc['mean_unc'],
                                          results.loc['repeats std'],
                                          results.loc['repeats std_unc'])
    results.loc['bs std / repeats std_unc'] = bs_std_ratio_unc
    # multiply by 100 to express as a percentage
    results.loc['bs estimate % variation'] = 100 * (bs_df.loc['std'] /
                                                    bs_df.loc['mean'])
    results.loc['bs estimate % variation_unc'] = 100 * (bs_df.loc['std_unc'] /
                                                        bs_df.loc['mean'])
    if add_sim_method:
        # get std from simulation estimate
        sim_values = pw.func_on_runs(ar.run_std_simulate, run_list,
                                     estimator_list, n_simulate=n_simulate,
                                     max_workers=max_workers,
                                     parallelise=parallelise)
        sim_df = mf.get_df_row_summary(sim_values, e_names)
        # get mean simulation std estimate a ratio to the std from repeats
        results.loc['sim std / repeats std'] = (sim_df.loc['mean'] /
                                                results.loc['repeats std'])
        sim_std_ratio_unc = mf.array_ratio_std(sim_df.loc['mean'],
                                               sim_df.loc['mean_unc'],
                                               results.loc['repeats std'],
                                               results.loc['repeats std_unc'])
        results.loc['sim std / repeats std_unc'] = sim_std_ratio_unc
        # multiply by 100 to express as a percentage
        results.loc['sim estimate % variation'] = 100 * (sim_df.loc['std'] /
                                                         sim_df.loc['mean'])
        results.loc['sim estimate % variation_unc'] = (100 *
                                                       sim_df.loc['std_unc'] /
                                                       sim_df.loc['mean'])
    # get bootstrap CI estimates
    bs_cis = pw.func_on_runs(ar.run_ci_bootstrap, run_list[:n_run_ci],
                             estimator_list, n_simulate=n_simulate_ci,
                             max_workers=max_workers,
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
    results.loc['bs +-1std % coverage'] = coverage * 100
    # set uncertainty on empirical measurement of coverage to zero
    results.loc['bs +-1std % coverage_unc'] = 0
    # add credible interval coverage
    max_value = results.loc['bs ' + str(cred_int) + ' CI'].values
    ci_coverage = np.zeros(rep_values.shape[0])
    for i, _ in enumerate(coverage):
        ind = np.where(rep_values[i, :] < max_value[i])
        ci_coverage[i] = ind[0].shape[0] / rep_values.shape[1]
    # multiply by 100 to express as a percentage
    results.loc['bs ' + str(cred_int) + ' CI % coverage'] = ci_coverage * 100
    # set uncertainty on empirical measurement of coverage to zero
    results.loc['bs ' + str(cred_int) + ' CI % coverage_unc'] = 0
    results = mf.df_unc_rows_to_cols(results)
    if save:
        # save the results data frame
        print('get_bootstrap_results: results saved to\n' + save_file)
        results.to_pickle(save_file)
    return results
