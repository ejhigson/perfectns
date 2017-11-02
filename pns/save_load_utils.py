#!/usr/bin/python
"""Contains useful helper functions shared by many other modules."""


import os.path  # for saving and reading data
import numpy as np
import pandas as pd
import time
import pickle
from functools import wraps

# Saving and loading functions with pickle:


def timing_decorator(func):
    """
    Outputs the time a function takes to execute.
    """
    @wraps(func)
    def wrapper(*args, **kw):
        """
        Wrapper for printing execution time.
        """
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()
        print(func.__name__ + " took %.3f seconds" % (end_time - start_time))
        return result
    return wrapper


def data_save_name(settings, n_repeats, extra_root=None, include_dg=True):
    """
    Make a standard save name format for data with a given set of settings.
    """
    save_name = settings.data_version
    if extra_root is not None:
        save_name += "_" + str(extra_root)
    if include_dg:
        save_name += "_dg" + str(settings.dynamic_goal)
    save_name += "_" + str(settings.n_dim) + "d"
    # add likelihood and prior info
    save_name += "_" + type(settings.likelihood).__name__
    if type(settings.likelihood).__name__ == "exp_power":
        save_name += "_" + str(settings.likelihood.power)
    save_name += "_" + str(settings.likelihood.likelihood_scale)
    save_name += "_" + type(settings.prior).__name__
    save_name += "_" + str(settings.prior.prior_scale)
    save_name += "_" + str(settings.zv_termination_fraction) + "term"
    save_name += "_" + str(n_repeats) + "reps"
    save_name += "_" + str(settings.nlive_const) + "nlive"
    if settings.dynamic_goal is not None or include_dg is False:
        save_name += "_" + str(settings.ninit) + "ninit"
        if settings.nbatch != 1:
            save_name += "_" + str(settings.nbatch) + "nbatch"
    if settings.n_calls_max is not None and settings.nlive_const is None:
        save_name += "_" + str(settings.n_calls_max) + "callsmax"
    if settings.tuned_dynamic_p is True and settings.dynamic_goal is not None:
        save_name += "_tuned"
    save_name = save_name.replace(".", "_")
    save_name = save_name.replace("-", "_")
    return save_name


@timing_decorator
def pickle_save(data, name, path="data/", extension=".dat"):
    """Saves object with pickle,  appending name with the time file exists."""
    filename = path + name + extension
    if os.path.isfile(filename):
        filename = path + name + "_" + time.asctime().replace(" ", "_")
        filename += extension
        print("File already exists! Saving with time appended")
    print(filename)
    try:
        outfile = open(filename, 'wb')
        pickle.dump(data, outfile)
        outfile.close()
    except MemoryError:
        print("pickle_save could not save data due to memory error: exiting " +
              "without saving")


@timing_decorator
def pickle_load(name, path="data/", extension=".dat"):
    """Load data with pickle."""
    filename = path + name + extension
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def latex_sf(value, start_end_sf=[2, -2], dp=4):
    if value != 0:
        power = int(np.log10(abs(value)))
    else:
        power = 0
    if power >= start_end_sf[0] or power <= start_end_sf[1]:
        value = value * (10 ** (- power))
    else:
        power = False
    output = round(value, dp)
    if dp == 0:
        output = int(output)
    output = str(output)
    if power is not False and power != 0:
        output += "cdot 10{" + str(power) + "}"
    return output


def latex_form(value_in, error_in, start_end_sf=[2, -2], dp=4):
    try:
        if value_in == 0:
            power = 0
        else:
            power = int(np.log10(abs(value_in)))
        if power >= start_end_sf[0] or power <= start_end_sf[1]:
            value = value_in * (10 ** (- power))
        else:
            value = value_in
            power = 0
        output = '{:.{prec}f}'.format(value, prec=dp)
        # output = str(round(value, dp))
        error = error_in / (10 ** (power - dp))
        if error < 0:
            print("latex_form: warning: error on final digit=" + str(error) +
                  " < 0")
        if error == 0:
            output += "(0)"
        else:
            output += "("
            if error > 1:
                error_dp = 0
            else:
                error_dp = int(np.ceil((-1.0) * np.log10(error)))
            output += '{:.{prec}f}'.format(error, prec=error_dp)
            output += ")"
        if power is not False and power != 0:
            output += " cdot 10{" + str(power) + "}"
        return output
    except (ValueError, OverflowError):
        # print("latex_form: warning: ValueError on ", value_in, error_in)
        return str(value_in) + "(" + str(error_in) + ")"


def latex_form_percent(value_in, error_in, start_end_sf=[2, -2], dp=4):
    return latex_form(value_in * 100, error_in * 100,
                      start_end_sf=start_end_sf, dp=(dp - 2)) + "\\%"


def latex_format_df(df, cols=None, rows=None, dp_list=None):
    if cols is None:
        cols = [n for n in list(df.columns) if (len(n) <= 3 or
                                                n[-4:] != '_unc')]
    if rows is None:
        rows = list(df.index)
    if dp_list is None:
        dp_list = [4] * len(rows)
    latex_dict = {}
    for c in cols:
        latex_dict[c] = []
        for i, r in enumerate(rows):
            temp = latex_form(df[c][r], df[c + '_unc'][r], dp=dp_list[i],
                              start_end_sf=[4, -4])
            latex_dict[c].append(temp)
    latex_df = pd.DataFrame(latex_dict, index=rows)
    return latex_df
