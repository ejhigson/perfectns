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
                        reduce_n_calls_max_frac=0.02):
    values_list = []
    # get info on the number of samples too
    funcs_list = [e.n_samplesEstimator()] + funcs_list_in
    func_names = []
    df_dict = {}
    for func in funcs_list:
        func_names.append(func.name)
    for i, dynamic_goal in enumerate(dynamic_goals):
        settings.dynamic_goal = dynamic_goal
        run_list = pw.get_run_data(settings, n_run, parallelise=parallelise,
                                   load=load, save=save)
        values = pw.func_on_runs(au.run_estimators, run_list, funcs_list)
        df = mf.get_df_row_summary(values, func_names)
        df_dict[dynamic_goal] = df
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
    for key in df_dict:
        # find performance gain (proportional to ratio of errors squared)
        std_ratio = df_dict[None].loc["std"] / df_dict[key].loc["std"]
        std_ratio_unc = mf.array_ratio_std(df_dict[None].loc["std"],
                                           df_dict[None].loc["std_unc"],
                                           df_dict[key].loc["std"],
                                           df_dict[key].loc["std_unc"])
        df_dict[key].loc["gain"] = std_ratio ** 2
        df_dict[key].loc["gain_unc"] = 2 * std_ratio * std_ratio_unc
    for key in df_dict:
        df_dict[key]["dynamic_goal"] = [key] * df_dict[key].shape[0]
        df_dict[key]["calc_type"] = df_dict[key].index
    results = pd.concat(df_dict.values())
    # make the calc column catagorical with a custom ordering
    order = ["mean", "mean_unc", "std", "std_unc", "gain", "gain_unc"]
    results['calc_type'] = pd.Categorical(results['calc_type'], order)
    results.sort_values(["calc_type", "dynamic_goal"], inplace=True)
    # put the dynamic goal column first
    cols = list(results)
    cols.insert(0, cols.pop(cols.index('dynamic_goal')))
    results = results.loc[:, cols]
    return results
