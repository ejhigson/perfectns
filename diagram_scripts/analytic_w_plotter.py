#!/usr/bin/env python
import numpy as np
# import maths_functions as mf
# import matplotlib
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pns.likelihoods_and_priors as lp
import math
import pns.maths_functions as mf
from matplotlib import rc  # , rcParams
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('axes', linewidth=0.5)  # default = 0.8
rc('ytick.major', width=0.5)  # default = 0.8
rc('ytick.major', size=2)  # default = 3.5
# rcParams.update({'figure.autolayout': True})


def w_given_logx(logx, likelihood_prior):
    logw = logw_given_logx(logx, likelihood_prior)
    for i, w_i in enumerate(logw):
        if math.isnan(w_i):
            logw[i] = -np.inf
    logw -= logw.max()  # must do this after removing nans or it sets everything to nan
    return np.exp(logw)


def logw_given_logx(logx, likelihood_prior):
    return logx + likelihood_prior.logl_given_logx(logx)  # likelihood_prior.logz_analytic


# Settings
# --------


# Output Settings
# ---------------
version = 'v15'
root = 'an_weights_' + version

label_fontsize = 8
pdf_dpi = 100
save = True

# Plotting
# --------
likelihood_list = ['gaussian', 'cauchy']
# likelihood_list = ['gaussian', 'exp_power_1', 'exp_power_0_75']
# Dynamic NS paper settings
dim = [2, 10]
if len(likelihood_list) == 2 and likelihood_list[-1] == "cauchy":
    xmin = -35
    figsize = (6, 1.5)
else:
    xmin = -35
    figsize = (6, 2)
# # errors paper settings
# dim = [2, 6, 10]
# figsize = (6, 2)
# xmin = -30
# shared settings
xmax = 0.0
plt.clf()
image = plt.figure(figsize=figsize)
ax = image.add_subplot(1, 1, 1)
w_temp_max = 0
logx = np.linspace(xmin, xmax, 10000)
rmax = 10
d_rmax_list = []
for d in dim:
    d_rmax_list += [(d, rmax)]
linestyles = ['solid', 'dashed', 'dotted']
linestyles += ['solid'] * len(likelihood_list)
z_term_frac = 10 ** -4
entropy_gain = np.zeros((len(likelihood_list), len(d_rmax_list)))
for k, d_rmax in enumerate(d_rmax_list):
    for j, likelihood_name in enumerate(likelihood_list):
        print(likelihood_name, d_rmax)
        likelihood_prior = lp.likelihood_prior(likelihood_name, 'gaussian', d_rmax[1], d_rmax[0])
        # z = mpmath.quad(lambda x: likelihood_prior.logl_given_logx(x), [-1000, 0])
        w_temp = w_given_logx(logx, likelihood_prior)
        w_temp[np.isnan(w_temp)] = 0.0
        print(np.trapz(w_temp, x=logx), np.exp(likelihood_prior.logz_analytic))
        w_temp /= np.trapz(w_temp, x=logx)
        w_cumsum = np.cumsum(w_temp)
        w_temp_term = w_temp[np.where(w_cumsum > 1. - z_term_frac)]
        entropy_gain[j, k] = w_temp_term.shape[0] / mf.entropy_num_samples(w_temp_term)
        w_temp_max = max(w_temp_max, w_temp.max())
        label = '$d = ' + str(d_rmax[0]) + '$'
        # label += ' $\sigma_\pi = ' + str(d_rmax[1]) + ' $'
        if len(likelihood_list) != 1:
            if likelihood_name == 'exp_power_2':
                label = 'Exp power' + ': $b=2$, ' + label
            elif likelihood_name == 'exp_power_0_75':
                label = 'Exp power' + ': $b=\\frac{3}{4}$, ' + label
            else:
                label = likelihood_name.replace('_', ' ').title() + ': ' + label
        ax.plot(logx, w_temp, linewidth=1, label=label, linestyle=linestyles[j])
print(likelihood_list)
print("entropy gain")
print(d_rmax_list)
print(entropy_gain)
ax.set_ylabel('relative posterior mass', fontsize=label_fontsize)
ax.set_xlabel('$\log X$', fontsize=(label_fontsize))
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(label_fontsize)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(label_fontsize)
ax.set_ylim([0, w_temp_max * 1.1])
ax.set_yticks([])
ax.set_xlim([xmin, xmax])
if len(likelihood_list) == 1:
    ncol = 1
else:
    ncol = 2
ax.legend(loc=2, ncol=ncol, prop={'size': label_fontsize})  # :fontsize=label_fontsize)
# Output plot
# ------------
if save is True:
    output_root = 'plots/' + root + '.pdf'
    # Save as pdf
    print('Saving as pdf')
    print(output_root)
    plt.savefig(output_root, bbox_inches='tight',
                pad_inches=0.0, dpi=pdf_dpi)
