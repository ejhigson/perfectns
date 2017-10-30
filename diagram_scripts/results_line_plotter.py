#!/usr/bin/python
"""
tbc
"""

import numpy as np
import pandas as pd
# perfect nested sampling modules
import pns.save_load_utils as slu
import pns.likelihoods as likelihoods
import pns.estimators as e
import pns.priors as priors
import pns_settings
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
settings = pns_settings.PerfectNestedSamplingSettings()
pd.set_option('display.width', 200)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
label_fontsize = 8

# rc('axes', linewidth=1)

# Output Settings
# ---------------

nlive = 200
n_run = 100
settings.data_version = 'v05'
settings.nlive_2 = 2
plot_name = 'dynamic_test'
plot_type = 'n_dim'
plot_type = 'prior_scale'
if plot_type == 'prior_scale':
    likelihood_list = [likelihoods.gaussian(1),
                       likelihoods.exp_power(1, 0.75),
                       likelihoods.exp_power(1, 2)]
    estimator_list = [e.logzEstimator(),
                      e.theta1Estimator()]
else:
    estimator_list = [e.logzEstimator(),
                      e.theta1Estimator(),
                      e.theta1confEstimator(0.5),
                      e.theta1confEstimator(0.84)]
    likelihood_list = [likelihoods.gaussian(1)]
    likelihood_list = [likelihoods.exp_power(1, 0.75)]
    likelihood_list = [likelihoods.exp_power(1, 2)]

# begin script
# ------------
log = True
dynamic_goals = [None, 0, 1]
ind_to_plot = 'performance'  # performance gain
extra_root = "dynamic_test"
for dg in dynamic_goals:
    extra_root += "_" + str(dg)
if plot_type == 'n_dim':
    prior_list = [priors.gaussian_cached(10)]
    n_dim_list = [2, 5, 10, 30, 100, 300, 1000]
elif plot_type == 'prior_scale':
    n_dim_list = [2]
    rmax_list = [0.1, 0.3, 1, 3, 10, 30, 100]
    prior_list = [priors.gaussian(r) for r in rmax_list]

likelihood_results = []
likelihood_x_values = []
for likelihood in likelihood_list:
    results = []
    x_values = []
    settings.likelihood = likelihood
    for prior in prior_list:
        settings.prior = prior
        for n_dim in n_dim_list:
            settings.n_dim = n_dim
            try:

                save_root = slu.data_save_name(settings, n_run,
                                               extra_root=extra_root,
                                               include_dg=False)
                save_file = 'data/' + save_root + '.dat'
                print(save_file)
                results.append(pd.read_pickle(save_file))
                if plot_type == 'n_dim':
                    x_values.append(settings.n_dim)
                elif plot_type == 'prior_scale':
                    x_values.append(settings.prior.prior_scale)
                print('likelihood=', type(settings.likelihood).__name__,
                      'n_dim=', settings.n_dim,
                      'prior_scale=', settings.prior.prior_scale,
                      'version=', settings.data_version)
            except IOError:
                pass
    likelihood_results.append(results)
    likelihood_x_values.append(x_values)

save = True
nbin = 400
linestyles = ['solid', '--', ':']
# Plotting
# --------
plt.clf()
if plot_type == 'prior_scale':
    figsize = (6, 3)
else:
    figsize = (6, 2)
image = plt.figure(figsize=figsize)
ax = image.add_subplot(1, 1, 1)
for l, likelihood in enumerate(likelihood_list):
    assert len(likelihood_results[l]) == len(likelihood_x_values[l])
    # Do not plot lines where there is no data as this causes an error when
    # ax.legend() is called.
    if likelihood_results[l]:  # True if likelihood_results is not empty
        for dg in [0, 1]:
            for est in estimator_list:
                x_values = likelihood_x_values[l]
                y_values = []
                y_unc = []
                for df in likelihood_results[l]:
                    gain_results = df[df['dynamic_goal'] == "dyn " +
                                      str(dg)].loc['gain']
                    y_values.append(gain_results[est.name])
                    y_unc.append(gain_results[est.name + "_unc"])
                # Make a string to label the series on the plot
                label = '$G=' + str(dg) + '$: ' + est.latex_name
                if len(likelihood_list) != 1:
                    name = type(likelihood).__name__.title().replace('_', ' ')
                    if type(likelihood).__name__ == 'exp_power':
                        if likelihood.power == 0.75:
                            label = ('$b=\\frac{3}{4}$, ' + label)
                        else:
                            label = ('$b=' + str(likelihood.power) +
                                     '$, ' + label)
                    label = name + ', ' + label
                # plot
                ax.errorbar(x_values, y_values, yerr=y_unc, label=label,
                            linestyle=linestyles[l])
# set limits on axis
ax.set_ylim(bottom=0)
if log:
    ax.set_xscale('log')
if plot_type == 'prior_scale':
    ax.set_ylim([0, 15])
    # ax.set_ylim(bottom=0)
    ax.set_xlabel('$\\sigma_\\pi$', fontsize=(label_fontsize))
    # to line up y labels as the fact the n_dim y axis goes up to 10 makes
    # its tick labels wider
    ypad = 2
elif plot_type == 'n_dim':
    ax.set_xlim(left=1)
    ax.set_xlabel('dimension $d$', fontsize=(label_fontsize))
    # if likelihood_names == ["exp_power_0_75"]:
    #     ypad = 5  # to make up for ymax < 10 meaning labels are thinner
    # else:
    #     ypad = 2
ax.set_ylabel('dynamic efficiency gain', labelpad=ypad,
              fontsize=(label_fontsize))
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(label_fontsize)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(label_fontsize)
ax.legend(fontsize=label_fontsize, ncol=2)
# Output plot
# ------------
save_name = ('results_plotter_' + plot_name + '_' + settings.data_version +
             '_' + plot_type)
for likelihood in likelihood_list:
    save_name += '_' + type(likelihood).__name__
output_root = 'plots/' + save_name
output_root = output_root.replace('.', '_')
output_root = output_root.replace(',', '_')
# Save as pdf
print('Saving as pdf')
print(output_root + '.pdf')
plt.savefig(output_root + '.pdf', bbox_inches='tight',
            pad_inches=0.0)
plt.close('all')
