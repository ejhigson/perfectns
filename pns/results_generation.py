#!/usr/bin/python
"""Functions for generating results."""

import pandas as pd
# perfect nested sampling modules
import pns.parallelised_wrappers as pw
import pns.analysis_utils as au
import pns.maths_functions as mf
import pns.estimators as e


# results functions


def get_dynamic_results(n_run, dynamic_goals, funcs_list_in, settings,
                        load=True, save=True, parallelise=True,
                        reduce_n_calls_max_frac=0.02, tuned_dynamic_ps=None):
    """
    Generate results using different dynamic goals and output a pandas data
    frame containing standard deviations and performance gains.
    """
    values_list = []
    # get info on the number of samples too
    funcs_list = [e.n_samplesEstimator()] + funcs_list_in
    func_names = []
    df_dict = {}
    for func in funcs_list:
        func_names.append(func.name)
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
    calc_names = ['mean', 'std', 'gain']  # also controls row order
    for key, df in df_dict.items():
        df_values = df.loc[calc_names]
        df_uncs = df.loc[[s + "_unc" for s in calc_names]]
        # strip "_unc" suffix from row indexes
        df_uncs.rename(lambda s: s[:-4], inplace=True)
        # add "_unc" suffix to columns
        df_uncs = df_uncs.add_suffix('_unc')
        df_dict[key] = pd.concat([df_values, df_uncs], axis=1)
        df_dict[key] = df_dict[key].reindex_axis(sorted(df_dict[key].columns),
                                                 axis=1)
        df_dict[key]["dynamic_goal"] = [key] * df_dict[key].shape[0]
    results = pd.concat(df_dict.values())
    # make the calc column catagorical with a custom ordering
    results['calc_type'] = pd.Categorical(results.index, calc_names)
    results.sort_values(["calc_type", "dynamic_goal"], inplace=True)
    del results['calc_type']
    # put the dynamic goal column first
    cols = list(results)
    cols.insert(0, cols.pop(cols.index('dynamic_goal')))
    cols.insert(1, cols.pop(cols.index('n_samples')))
    cols.insert(2, cols.pop(cols.index('n_samples_unc')))
    results = results.loc[:, cols]
    return results
