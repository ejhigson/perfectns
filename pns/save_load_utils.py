#!/usr/bin/python
"""Contains useful helper functions shared by many other modules."""


import os.path  # for saving and reading data
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
    save_name += "_" + str(settings.nlive) + "nlive"
    if settings.dynamic_goal is not None or include_dg is False:
        save_name += "_" + str(settings.nlive_1) + "nlive1"
        save_name += "_" + str(settings.nlive_2) + "nlive2"
    if settings.n_calls_max is not None and settings.nlive is None:
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
