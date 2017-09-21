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
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()
        print(func.__name__ + " took %.3f seconds" % (end_time - start_time))
        return result
    return wrapper


def data_save_name(n_repeats, settings):
    save_name = settings.data_version + "_" + str(settings.n_dim) + "d_"
    save_name += str(settings.dynamic_goal) + "dg_"
    # add likelihood and prior info
    save_name += type(settings.likelihood).__name__
    save_name += str(settings.likelihood.likelihood_scale) + "_"
    save_name += type(settings.prior).__name__ + str(settings.prior_scale) + "_"
    save_name += str(settings.zv_termination_fraction) + "term_" + str(n_repeats) + "reps_"
    save_name += str(settings.nlive) + "nlive_"
    if settings.dynamic_goal is not None:
        save_name = settings.data_version + "_dynamic_" + str(settings.dynamic_goal) + "_" + str(settings.nlive_1) + "nlive1_" + str(settings.nlive_2) + "nlive2_" + save_name
    if settings.n_calls_max is not None and settings.nlive is None:
        save_name += "_" + str(settings.n_calls_max) + "callsmax"
    if settings.tuned_dynamic_p is True and settings.dynamic_goal is not None:
        save_name += "_tuned"
    save_name = save_name.replace(".", "_")
    save_name = save_name.replace("-", "_")
    return save_name


@timing_decorator
def pickle_save(data, name, path="data/", extension=".txt"):
    """Saves object with pickle,  appending name with the time file exists."""
    filename = path + name + extension
    if os.path.isfile(filename):
        filename = path + name + "_" + time.asctime().replace(" ", "_")
        filename += extension
        print("File already exists! Saving with time appended")
    print(filename)
    outfile = open(filename, 'wb')
    pickle.dump(data,  outfile)
    outfile.close()


@timing_decorator
def pickle_load(name, path="data/", extension=".txt", print_time=True):
    """Load data with pickle."""
    filename = path + name + extension
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data
