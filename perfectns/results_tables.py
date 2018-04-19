#!/usr/bin/env python
"""
Functions used to generate results tables.

Used for results in 'Dynamic nested sampling: an improved algorithm for nested
sampling parameter estimation and evidence calculation' (Higson et al., 2017).
"""

import copy
import pandas as pd
import numpy as np
import nestcheck.io_utils as iou
import nestcheck.ns_run_utils
import nestcheck.error_analysis
import nestcheck.parallel_utils as pu
import nestcheck.pandas_functions as pf
import perfectns.nested_sampling as ns
import perfectns.priors as priors
import perfectns.estimators as e


@iou.timing_decorator
def get_dynamic_results(n_run, dynamic_goals_in, estimator_list_in,
                        settings_in, **kwargs):
    """
    Generate data frame showing the standard deviations of the results of
    repeated calculations and efficiency gains (ratios of variances of results
    calculations) from different dynamic goals. To make the comparison fair,
    for dynamic nested sampling settings.n_samples_max is set to slightly below
    the mean number of samples used by standard nested sampling.

    This function was used for Tables 1, 2, 3 and 4, as well as to generate the
    results shown in figures 6 and 7 of 'Dynamic nested sampling: an improved
    algorithm for nested sampling parameter estimation and evidence
    calculation' (Higson et al., 2017). See the paper for a more detailed
    description.

    Parameters
    ----------
    n_run: int
        how many runs to use
    dynamic_goals_in: list of floats
        which dynamic goals to test
    estimator_list_in: list of estimator objects
    settings_in: PerfectNSSettings object
    load: bool, optional
        should run data and results be loaded if available?
    save: bool, optional
        should run data and results be saved?
    overwrite_existing: bool, optional
        if a file exists already but we generate new run data, should we
        overwrite the existing file when saved?
    run_random_seeds: list, optional
        list of random seeds to use for generating runs.
    parallel: bool, optional
    cache_dir: str, optional
        Directory to use for caching.
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
        Contains rows:
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
    max_workers = kwargs.pop('max_workers', None)
    parallel = kwargs.pop('parallel', True)
    cache_dir = kwargs.pop('cache_dir', 'cache')
    overwrite_existing = kwargs.pop('overwrite_existing', True)
    run_random_seeds = kwargs.pop('run_random_seeds', list(range(n_run)))
    tuned_dynamic_ps = kwargs.pop('tuned_dynamic_ps',
                                  [False] * len(dynamic_goals_in))
    assert len(tuned_dynamic_ps) == len(dynamic_goals_in)
    for goal in dynamic_goals_in:
        assert goal is not None, \
            'Goals should be dynamic - standard NS already included'
    # Add a standard nested sampling run for comparison:
    dynamic_goals = [None] + dynamic_goals_in
    tuned_dynamic_ps = [False] + tuned_dynamic_ps
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Make a copy of the input settings to stop us editing them
    settings = copy.deepcopy(settings_in)
    # make save_name
    save_root = 'dynamic_test'
    for dg in dynamic_goals_in:
        save_root += '_' + str(dg).replace('.', '_')
    save_root += '_' + settings.save_name(include_dg=False)
    save_root += '_' + str(n_run) + 'reps'
    save_file = cache_dir + '/' + save_root + '.pkl'
    # try loading results
    if load:
        try:
            return pd.read_pickle(save_file)
        except OSError:
            print('Could not load file: ' + save_file)
    # start function
    # --------------
    # get info on the number of samples taken in each run as well
    estimator_list = [e.CountSamples()] + estimator_list_in
    est_names = [est.latex_name for est in estimator_list]
    method_names = []
    method_values = []
    assert dynamic_goals[0] is None, (
        'Need to start with standard ns to calculate efficiency gains')
    for i, dynamic_goal in enumerate(dynamic_goals):
        # set up settings
        settings.dynamic_goal = dynamic_goal
        settings.tuned_dynamic_p = tuned_dynamic_ps[i]
        # if we have already done the standard calculation, set n_samples_max
        # for dynamic calculations so it is slightly smaller than the number
        # of samples the standard calculation used to ensure a fair comparison
        # of performance. Otherwise dynamic nested sampling will end up using
        # more samples than standard nested sampling as it does not terminate
        # until after the number of samples is greater than n_samples_max.
        if i != 0 and settings.dynamic_goal is not None:
            assert dynamic_goals[0] is None
            assert isinstance(estimator_list[0], e.CountSamples)
            n_samples_max = np.mean(np.asarray([val[0] for val in
                                                method_values[0]]))
            # This factor is a function of the dynamic goal as typically
            # evidence calculations have longer additional threads than
            # parameter estimation calculations.
            reduce_factor = 1 - ((1.5 - 0.5 * settings.dynamic_goal) *
                                 (settings.nbatch / settings.nlive_const))
            settings.n_samples_max = int(n_samples_max * reduce_factor)
        print('dynamic_goal=' + str(settings.dynamic_goal),
              'n_samples_max=' + str(settings.n_samples_max))
        # get a name for this calculation method
        if dynamic_goal is None:
            method_names.append('standard')
        else:
            method_names.append('dynamic $G=' +
                                str(settings.dynamic_goal) + '$')
            if settings.tuned_dynamic_p is True:
                method_names[-1] += ' tuned'
        # generate runs and get results
        run_list = ns.get_run_data(settings, n_run, parallel=parallel,
                                   random_seeds=run_random_seeds,
                                   load=load, save=save,
                                   max_workers=max_workers,
                                   cache_dir=cache_dir,
                                   overwrite_existing=overwrite_existing)
        method_values.append(pu.parallel_apply(nestcheck.ns_run_utils.run_estimators, run_list,
                                               func_args=(estimator_list,),
                                               max_workers=max_workers,
                                               parallel=parallel))
    results = pf.efficiency_gain_df(method_names, method_values, est_names)
    if save:
        # save the results data frame
        print('get_dynamic_results: saving results to\n' + save_file)
        results.to_pickle(save_file)
    return results


@iou.timing_decorator
def merged_dynamic_results(dim_scale_list, likelihood_list, settings,
                           estimator_list, **kwargs):
    """
    Wrapper for running get_dynamic_results for many different likelihood,
    dimension and prior scales, and merging the output into a single
    data frame.
    See get_dynamic_results doccumentation for more details.


    Parameters
    ----------
    dim_scale_list: list of tuples
        (dim, prior_scale) pairs to run
    likelihood_list: list of likelihood objects
    settings_in: PerfectNSSettings object
    estimator_list: list of estimator objects
    n_run: int, optional
        number of runs for use with each setting.
    dynamic_goals_in: list of floats, optional
        which dynamic goals to test
    (remaining kwargs passed to get_dynamic_results)

    Returns
    -------
    results: pandas data frame
    """
    dynamic_goals = kwargs.pop('dynamic_goals', [0, 1])
    load = kwargs.pop('load', True)  # ensure default True for merged results
    save = kwargs.pop('save', True)  # ensure default True for merged results
    n_run = kwargs.pop('n_run', 1000)
    results_list = []
    for likelihood in likelihood_list:
        for n_dim, prior_scale in dim_scale_list:
            settings.n_dim = n_dim
            settings.likelihood = likelihood
            if n_dim >= 50:
                settings.prior = priors.GaussianCached(prior_scale=prior_scale)
            else:
                settings.prior = priors.Gaussian(prior_scale=prior_scale)
            like_lab = (type(settings.likelihood).__name__
                        .replace('ExpPower', 'Exp Power'))
            if type(settings.likelihood).__name__ == 'ExpPower':
                like_lab += (', $b=' + str(settings.likelihood.power)
                             .replace('0.75', r'\frac{3}{4}') + '$')
            print(like_lab, 'd=' + str(n_dim),
                  'prior_scale=' + str(prior_scale))
            df_temp = get_dynamic_results(
                n_run, dynamic_goals, estimator_list, settings, save=save,
                load=load, **kwargs)
            new_inds = ['likelihood', 'dimension $d$', r'$\sigma_\pi$']
            df_temp[new_inds[0]] = like_lab
            df_temp[new_inds[1]] = settings.n_dim
            df_temp[new_inds[2]] = settings.prior.prior_scale
            order = new_inds + list(df_temp.index.names)
            df_temp.set_index(new_inds, drop=True, append=True,
                              inplace=True)
            df_temp = df_temp.reorder_levels(order)
            results_list.append(df_temp)
    results = pd.concat(results_list)
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
    calculation' (Higson et al., 2017). See the paper for more details.

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
    parallel: bool, optional
    cache_dir: str, optional
        Directory to use for caching.
    add_sim_method: bool, optional
        should we also calculate standard deviations using the simulated
        weights method for comparison with bootstrap resampling? This method is
        inaccurate for parameter estimation.
    n_simulate_ci: int, optional
        how many times to resample the nested sampling run in each bootstrap
        credible interval estimate. These may require more simulations than the
        standard deviation estimate.
    run_random_seeds: list, optional
        list of random seeds to use for generating runs.
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
    ninit_sep = kwargs.pop('ninit_sep', True)
    parallel = kwargs.pop('parallel', True)
    cache_dir = kwargs.pop('cache_dir', 'cache')
    add_sim_method = kwargs.pop('add_sim_method', False)
    n_simulate_ci = kwargs.pop('n_simulate_ci', n_simulate)
    n_run_ci = kwargs.pop('n_run_ci', n_run)
    cred_int = kwargs.pop('cred_int', 0.95)
    run_random_seeds = kwargs.pop('run_random_seeds', list(range(n_run)))
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # make save_name
    save_root = ('bootstrap_results_' + str(n_simulate) + 'nsim_' +
                 str(ninit_sep) + 'sep')
    save_root += '_' + settings.save_name()
    save_root += '_' + str(n_run) + 'reps'
    save_file = cache_dir + '/' + save_root + '.pkl'
    # try loading results
    if load:
        try:
            return pd.read_pickle(save_file)
        except OSError:
            pass
    # start function
    est_names = [est.latex_name for est in estimator_list]
    # generate runs
    run_list = ns.get_run_data(settings, n_run, save=save, load=load,
                               random_seeds=run_random_seeds,
                               cache_dir=cache_dir,
                               max_workers=max_workers,
                               parallel=parallel)
    # sort in order of random seeds. This makes credible intervals results
    # reproducable even when only the first section of run_list is used.
    run_list = sorted(run_list, key=lambda r: r['random_seed'])
    rep_values = pu.parallel_apply(nestcheck.ns_run_utils.run_estimators, run_list,
                                   func_args=(estimator_list,),
                                   max_workers=max_workers,
                                   parallel=parallel)
    results = pf.summary_df_from_list(rep_values, est_names)
    new_index = ['repeats ' +
                 results.index.get_level_values('calculation type'),
                 results.index.get_level_values('result type')]
    results.set_index(new_index, inplace=True)
    results.index.rename('calculation type', level=0, inplace=True)
    # get bootstrap std estimate
    bs_values = pu.parallel_apply(nestcheck.error_analysis.run_std_bootstrap, run_list,
                                  func_args=(estimator_list,),
                                  func_kwargs={'n_simulate': n_simulate},
                                  max_workers=max_workers,
                                  parallel=parallel)
    bs_df = pf.summary_df_from_list(bs_values, est_names)
    # Get the mean bootstrap std estimate as a fraction of the std measured
    # from repeated calculations.
    results.loc[('bs std / repeats std', 'value'), :] = \
        (bs_df.loc[('mean', 'value')] / results.loc[('repeats std', 'value')])
    bs_std_ratio_unc = pf.array_ratio_std(
        bs_df.loc[('mean', 'value')],
        bs_df.loc[('mean', 'uncertainty')],
        results.loc[('repeats std', 'value')],
        results.loc[('repeats std', 'uncertainty')])
    results.loc[('bs std / repeats std', 'uncertainty'), :] = \
        bs_std_ratio_unc
    # Get the fractional variation of std estimates
    # multiply by 100 to express as a percentage
    results.loc[('bs estimate % variation', 'value'), :] = \
        100 * bs_df.loc[('std', 'value')] / bs_df.loc[('mean', 'value')]
    results.loc[('bs estimate % variation', 'uncertainty'), :] = \
        100 * bs_df.loc[('std', 'uncertainty')] / bs_df.loc[('mean', 'value')]
    if add_sim_method:
        # get std from simulation estimate
        sim_values = pu.parallel_apply(nestcheck.error_analysis.run_std_simulate, run_list,
                                       func_args=(estimator_list,),
                                       func_kwargs={'n_simulate': n_simulate},
                                       max_workers=max_workers,
                                       parallel=parallel)
        sim_df = pf.summary_df_from_list(sim_values, est_names)
        # Get the mean simulation std estimate as a fraction of the std
        # measured from repeated calculations.
        results.loc[('sim std / repeats std', 'value'), :] = \
            (sim_df.loc[('mean', 'value')] /
             results.loc[('repeats std', 'value')])
        sim_std_ratio_unc = pf.array_ratio_std(
            sim_df.loc[('mean', 'value')],
            sim_df.loc[('mean', 'uncertainty')],
            results.loc[('repeats std', 'value')],
            results.loc[('repeats std', 'uncertainty')])
        results.loc[('sim std / repeats std', 'uncertainty'), :] = \
            sim_std_ratio_unc
        # Get the fractional variation of std estimates
        # Multiply by 100 to express as a percentage
        results.loc[('sim estimate % variation', 'value'), :] = \
            100 * sim_df.loc[('std', 'value')] / sim_df.loc[('mean', 'value')]
        results.loc[('sim estimate % variation', 'uncertainty'), :] = \
            (100 * sim_df.loc[('std', 'uncertainty')] /
             sim_df.loc[('mean', 'value')])
    # get bootstrap CI estimates
    bs_cis = pu.parallel_apply(
        nestcheck.error_analysis.run_ci_bootstrap, run_list[:n_run_ci],
        func_args=(estimator_list,),
        func_kwargs={'n_simulate': n_simulate_ci,
                     'cred_int': cred_int,
                     'random_seeds': range(n_simulate_ci)},
        max_workers=max_workers, parallel=parallel)
    bs_ci_df = pf.summary_df_from_list(bs_cis, est_names)
    results.loc[('bs ' + str(cred_int) + ' CI', 'value'), :] = \
        bs_ci_df.loc[('mean', 'value')]
    results.loc[('bs ' + str(cred_int) + ' CI', 'uncertainty'), :] = \
        bs_ci_df.loc[('mean', 'uncertainty')]
    # add coverage for +- 1 bootstrap std estimate
    max_value = (results.loc[('repeats mean', 'value')].values
                 + bs_df.loc[('mean', 'value')].values)
    min_value = (results.loc[('repeats mean', 'value')].values
                 - bs_df.loc[('mean', 'value')].values)
    rep_values_array = np.stack(rep_values, axis=1)
    assert rep_values_array.shape == (len(estimator_list), n_run)
    coverage = np.zeros(rep_values_array.shape[0])
    for i, _ in enumerate(coverage):
        ind = np.where((rep_values_array[i, :] > min_value[i]) &
                       (rep_values_array[i, :] < max_value[i]))
        coverage[i] = ind[0].shape[0] / rep_values_array.shape[1]
    # multiply by 100 to express as a percentage
    results.loc[('bs +-1std % coverage', 'value'), :] = coverage * 100
    # add credible interval coverage
    max_value = results.loc[('bs ' + str(cred_int) + ' CI', 'value')].values
    ci_coverage = np.zeros(len(estimator_list))
    for i, _ in enumerate(coverage):
        ind = np.where(rep_values_array[i, :] < max_value[i])
        ci_coverage[i] = ind[0].shape[0] / rep_values_array.shape[1]
    # multiply by 100 to express as a percentage
    results.loc[('bs ' + str(cred_int) + ' CI % coverage', 'value'), :] = \
        (ci_coverage * 100)
    if save:
        # save the results data frame
        print('get_bootstrap_results: results saved to\n' + save_file)
        results.to_pickle(save_file)
    return results
