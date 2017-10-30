#!/usr/bin/python
"""
Contains the functions which perform nested sampling given input from settings.
These are all called from within the wrapper function
nested_sampling(settings).
"""

import numpy as np
import matplotlib.pyplot as plt
# perfect nested sampling modules
import pns.analysis_utils as au
import pns.estimators as e
import pns.parallelised_wrappers as pw
import pns_settings
import pns.likelihoods as likelihoods
settings = pns_settings.PerfectNestedSamplingSettings()
np.core.arrayprint._line_width = 400
np.set_printoptions(precision=5, suppress=True)

# settings
# --------
settings.likelihood = likelihoods.gaussian(likelihood_scale=1)
n_run = 10
settings.nlive = 200
settings.n_dim = 10
use_automatic_n_calls_max = True
load = True
save = True

# program
# -------
# Normal runs
# -------
# if type(settings.likelihood).__name__ == "cauchy":
#     dynamic_goals = [None, 1, 1]
# else:
# if dynamic_goals[-1] == 1 and dynamic_goals[-2] == 1:
if type(settings.likelihood).__name__ == "cauchy":
    dynamic_goals = [None, 0, 1, 1]
    tuned_dynamic_p = [False] * (len(dynamic_goals) - 1)
    tuned_dynamic_p += [True]
else:
    dynamic_goals = [None, 0, 0.25, 1]
    tuned_dynamic_p = [False] * len(dynamic_goals)
run_list = []
# work out n_calls_max from first set of runs
call_stats = np.zeros((len(dynamic_goals), 2))
n_calls_max = None
for i, _ in enumerate(dynamic_goals):
    print(dynamic_goals[i], tuned_dynamic_p[i])
    settings.dynamic_goal = dynamic_goals[i]
    settings.tuned_dynamic_p = tuned_dynamic_p[i]
    settings.n_calls_max = n_calls_max
    temp_runs = pw.get_run_data(settings, n_run, parallelise=True, load=load,
                                save=save)
    calls = pw.func_on_runs(au.run_estimators, temp_runs,
                            [e.n_samplesEstimator()])
    call_stats[i, 0] = np.mean(calls[0, :])
    call_stats[i, 1] = np.std(calls[0, :], ddof=1)
    if i == 0 and use_automatic_n_calls_max:
        n_calls_max = int(call_stats[0, 0] * (settings.nlive - 1) /
                          settings.nlive)
    print(dynamic_goals[i], len(temp_runs))
    run_list += temp_runs
# Output Settings
# ---------------
root = ("nlive_" + settings.data_version + "_" +
        type(settings.likelihood).__name__)
if any(tuned_dynamic_p):
    root += "_tuned"
if not use_automatic_n_calls_max:
    root += "_no_n_calls_max"
label_fontsize = 8
save = True
xmax = 0.0
nbin = 1000
# Plotting
# --------
plt.clf()
if type(settings.likelihood).__name__ == "cauchy":
    figsize = (6, 2.5)
else:
    figsize = (6, 3)
image = plt.figure(figsize=figsize)
ax = image.add_subplot(1, 1, 1)

n_calls = np.zeros(len(run_list))
logx_min = np.zeros(len(run_list))
# colors = ["k", "b", "g", "y", "m", "c", "r"]
# use default colors as per http://matplotlib.org/users/dflt_style_changes.html
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = []
colors.append(default_colors[2])  # green for standard ns
if (len(dynamic_goals) == 4 and type(settings.likelihood).__name__ != "cauchy"):
    colors.append(default_colors[8])  # pale yellow close to blue analytic line
    colors.append(default_colors[4])  # if required, pink
    colors.append(default_colors[9])  # pale blue
elif len(dynamic_goals) == 4 and type(settings.likelihood).__name__ == "cauchy":
    colors.append(default_colors[8])  # pale yellow close to blue analytic line
    colors.append(default_colors[9])  # pale blue
    colors.append(default_colors[4])  # if required, pink
elif len(dynamic_goals) == 3:
    colors.append(default_colors[9])  # pale blue
    colors.append(default_colors[8])  # pale yellow close blue analytic line
standard_nlive_max = 0
integrals = np.zeros(len(run_list))
for g, goal in enumerate(dynamic_goals):
    for i in range(n_run):
        run = run_list[g * n_run + i]
        nlive = au.get_nlive(run)
        n_calls[i] = nlive.shape[0]
        logx = settings.logx_given_logl(run['lrxtnp'][:, 0])
        logx[0] = 0  # to make lines extend all the way to the end
        logx_min[i] = logx[-1]  # for calculating plotting limits on x
        # for normalising analytic weight lines
        integrals[g * n_run + i] = -np.trapz(nlive, x=logx)
        if i == 1:
            if goal is None:
                label_i = "standard"
                standard_nlive_max = max(standard_nlive_max, nlive.max())
            else:
                label_i = "dynamic $G=" + str(goal) + "$"
                if tuned_dynamic_p[g] is True:
                    label_i = "tuned " + label_i
        else:
            label_i = ""
        if tuned_dynamic_p[g] is True:
            ax.plot(logx, nlive, linewidth=1, label=label_i, color=colors[g])
        else:
            ax.plot(logx, nlive, linewidth=1, label=label_i, color=colors[g])
# find analytic w
xmin = np.floor(logx_min.min())
logx = np.linspace(xmin, xmax, nbin)
logw_an = logx + settings.logl_given_logx(logx)
w_an = np.exp(logw_an - logw_an.max())
w_an /= np.trapz(w_an, x=logx)
# set limits
if type(settings.likelihood).__name__ == "gaussian" and settings.n_dim == 10:
    xmin = -35
    ymax = 1200
elif type(settings.likelihood).__name__ == "exp_power_2" and settings.n_dim == 10:
    xmin = -45
    ymax = 2000
elif type(settings.likelihood).__name__ == "exp_power_0_75" and settings.n_dim == 10:
    xmin = -30
    ymax = 1000
elif type(settings.likelihood).__name__ == "cauchy" and settings.n_dim == 10:
    xmin = -45
    ymax = max(standard_nlive_max * 4, w_an.max() * 1.05)
else:
    ymax = max(standard_nlive_max * 4, w_an.max() * 1.05)
ax.set_xlim([xmin, 0])
ax.set_ylim([0, ymax])
# print(mf.stats_rows(n_calls))
# print(n_calls)
if any(tuned_dynamic_p):
    w_an *= np.mean(integrals[n_run:2 * n_run])
else:
    w_an *= np.mean(integrals[-(1 + n_run):-1])
ax.plot(logx, w_an, linewidth=1.5, label="relative posterior mass",
        linestyle=":", color='k')
# # plot cumulative posterior mass
w_an_c = np.cumsum(w_an)
w_an_c /= np.trapz(w_an_c, x=logx)
w_an_c *= np.mean(integrals[n_run:2 * n_run])
ax.plot(logx, w_an_c, linewidth=1.5, linestyle="--", dashes=(2, 3),
        label="posterior mass remaining", color='darkblue')
if any(tuned_dynamic_p):
    # plot tuned mass
    w_tuned = (w_an * settings.r_given_logx(logx) *
               np.sqrt(1.0 / settings.n_dim))
    # remove any nans or infs as they screw up normalisation
    w_tuned[~np.isfinite(w_tuned)] = 0.0
    w_tuned /= np.trapz(w_tuned, x=logx)
    w_tuned *= np.mean(integrals[-(1 + n_run):-1])
    ax.plot(logx, w_tuned, linewidth=1.5, label="tuned importance",
            linestyle="-.", dashes=(2, 1.5, 1, 1.5), color='k')
ax.set_ylabel("number of live points", fontsize=(label_fontsize))
ax.set_xlabel("$\log X $", fontsize=(label_fontsize))
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(label_fontsize)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(label_fontsize)
ax.set_xlim([xmin, xmax])
if type(settings.likelihood).__name__ == "cauchy":
    loc = 2
else:
    loc = 1
ax.legend(fontsize=label_fontsize, loc=loc)
# Output plot
# ------------
if save is True:
    output_root = "plots/" + root + ".pdf"
    # Save as pdf
    print("Saving as pdf")
    print(output_root)
    plt.savefig(output_root, bbox_inches='tight',
                pad_inches=0.0)
plt.close("all")
