#!/usr/bin/python
"""
tbc
"""

import numpy as np
import pandas as pd
# from tqdm import tqdm
import matplotlib.pyplot as plt
# perfect nested sampling modules
import pns.save_load_utils as slu
import pns.likelihoods as likelihoods
import pns.estimators as e
import pns.priors as priors
import pns_settings
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
settings = pns_settings.PerfectNestedSamplingSettings()
pd.set_option('display.width', 200)
np.core.arrayprint._line_width = 400
np.set_printoptions(precision=4, suppress=True)
label_fontsize = 8

# rc('axes', linewidth=1)

# Output Settings
# ---------------

nlive = 200
n_run = 100
version = settings.data_version
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
                # print(save_file)
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
    # assert len(results) != 0, ('no results found. Last save_file was:\n' +
    #                            save_file)
#     n_method = results[-1][ind_to_plot].shape[0]
#     n_estimator = int(results[-1][ind_to_plot].shape[1] / 2)
#     performance = np.zeros((n_method, n_estimator, len(x_values), 3))
#     linestyles = ['solid', '--', ':']
#     method_labels = results[-1]['method_labels']
#     func_latex_names = results[-1]['func_latex_names']
#     print(performance.shape)
#     for m in range(n_method):
#         for est in range(n_estimator):
#             for i in range(len(x_values)):
#                 performance[m, est, i, 0] = x_values[i]
#                 performance[m, est, i, 1] = results[i][ind_to_plot][m, est * 2]
#                 performance[m, est, i, 2] = results[i][ind_to_plot][m, est * 2 + 1]
#     m = 1
#     if len(likelihoods) == 1:
#         # ests = range(0, n_estimator)
#         ests = [0, 1, 3, 4]  # remove ind 2 = povar
#     else:
#         ests = [0, 1]  # using 3 (CI 84%) instead of 1 (mean of thetaone) gives bigger perf gain but makes the plot not look so good
#     for est in ests:
#         if est == 0:
#             m = 1
#         else:
#             m = -1
#         if est == 0:
#             label = method_labels[m] + ': $\mathrm{log} \mathcal{Z}$'
#         elif est == 4:
#             label = method_labels[m] + ': $\mathrm{median(\\theta_{\hat{1}})}$'
#         else:
#             label = method_labels[m] + ': ' + func_latex_names[est]
#         if len(likelihood_names) != 1:
#             if likelihood_name == 'exp_power_2':
#                 label = 'Exp Power, $b=2$, ' + label
#             elif likelihood_name == 'exp_power_0_75':
#                 label = 'Exp Power, $b=\\frac{3}{4}$, ' + label
#             else:
#                 label = likelihood_name.replace('_', ' ').title() + ', ' + label
#         pl_list.append([performance[m, est, :, :], label, linestyles[lcount]])
    # pl_list.append([performance[n_method - 1, 1, :, :], likelihood_name.title() + ': ${\overline{\\theta}}_{\hat{1}}$, $G=1$'])
    # pl_list.append([performance[n_method - 1, 2, :, :], likelihood_name.title() + ': ${\mathrm{C.I.}}_{84\%}(\\theta_{\hat{1}})$, $G=1$'])

if True:
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
    for lcount, likelihood in enumerate(likelihood_list):
        for dg in dynamic_goals:
            if dg is not None:
                for est in estimator_list:
                    x_values = likelihood_x_values[lcount]
                    y_values = []
                    y_unc = []
                    for df in likelihood_results[lcount]:
                        gain_results = df[df['dynamic_goal'] == "dyn " +
                                          str(dg)].loc['gain']
                        print(est.name)
                        y_values.append(gain_results[est.name])
                        y_unc.append(gain_results[est.name + "_unc"])
                    label = 'SG=' + str(dg) + '$: ' + est.latex_name
                    if len(likelihood_list) != 1:
                        label = type(likelihood).__name__ + ' ' + label
                    ax.errorbar(x_values, y_values, yerr=y_unc, label=label,
                                linestyle=linestyles[lcount])
    # set limits on axis
    ax.set_ylim(bottom=0)
    if log:
        ax.set_xscale('log')
    if plot_type == 'prior_scale':
        ax.set_ylim([0, 15])
        # ax.set_ylim(bottom=0)
        ax.set_xlabel('$\sigma_\pi$', fontsize=(label_fontsize))
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
    # ax.legend()  # fontsize=label_fontsize, ncol=2)
    # Output plot
    # ------------
    save_name = ('results_plotter_' + plot_name + '_' + version + '_' +
                 plot_type)
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
