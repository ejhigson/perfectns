#!/usr/bin/env python
"""
Plots the posterior mass as a function of logX for different posteriors.
"""
import numpy as np
import pns.priors as priors
import pns.likelihoods as likelihoods
import pns.maths_functions as mf
import matplotlib.pyplot as plt
from matplotlib import rc  # , rcParams
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('axes', linewidth=0.5)  # default = 0.8
rc('ytick.major', width=0.5)  # default = 0.8
rc('ytick.major', size=2)  # default = 3.5
# rcParams.update({'figure.autolayout': True})


def w_given_logx(logx, n_dim, likelihood, prior):
    logw = logw_given_logx(logx, n_dim, likelihood, prior)
    logw[np.isnan(logw)] = -np.inf
    logw -= logw.max()  # must do this after removing nans
    return np.exp(logw)


def logw_given_logx(logx, n_dim, likelihood, prior):
    r = prior.r_given_logx(logx, n_dim)
    logl = likelihood.logl_given_r(r, n_dim)
    return logx + logl


# Settings
# --------


# Output Settings
# ---------------
version = 'v02'
root = 'an_weights_' + version

label_fontsize = 8
pdf_dpi = 100
save = True
plot_r = True

# Plotting
# --------
# likelihood_list = [likelihoods.gaussian(1),
#                    likelihoods.exp_power(1, 0.75),
#                    likelihoods.exp_power(1, 2)]
likelihood_list = [likelihoods.exp_power(1, 2)]
# prior_list = [priors.gaussian(10), priors.gaussian_cached(10)]
# prior_list = [priors.gaussian(10), priors.gaussian_cached(10)]
prior_list = [priors.gaussian_cached(10)]
dim_list = [3, 10, 30, 100, 300, 1000]
figsize = (12, 4)
xmin = -4000
xmax = -0.001
plt.close('all')
plt.clf()
image, ax = plt.subplots(figsize=figsize)
if plot_r:
    ax2 = ax.twinx()
    ax2.set_ylim([0, 150])
# image = plt.figure(figsize=figsize)
# ax = image.add_subplot(1, 1, 1)
w_temp_max = 0
logx = np.linspace(xmin, xmax, int(np.ceil(xmax - xmin)) * 10)
linestyles = ['solid', 'dashed', 'dotted']
z_term_frac = 10 ** -10
entropy_gain = np.zeros((len(likelihood_list), len(prior_list), len(dim_list)))
for l, likelihood in enumerate(likelihood_list):
    for p, prior in enumerate(prior_list):
        for d, n_dim in enumerate(dim_list):
            label = (type(likelihood).__name__ + ' ' + type(prior).__name__ +
                     ' $d = ' + str(n_dim) + '$')
            # label += ' $\sigma_\pi = ' + str(d_rmax[1]) + ' $'
            # if len(likelihood_list) != 1:
            #     if likelihood_name == 'exp_power_2':
            #         label = 'Exp power' + ': $b=2$, ' + label
            #     elif likelihood_name == 'exp_power_0_75':
            #         label = 'Exp power' + ': $b=\\frac{3}{4}$, ' + label
            #     else:
            #         label = likelihood_name.replace('_', ' ').title() +
            #                 ': ' + label
            label = label.replace('_', '\_')
            print(label)
            w_temp = w_given_logx(logx, n_dim, likelihood, prior)
            # normalise so area under curve = 1 for purposes of graphing
            w_temp /= np.trapz(w_temp, x=logx)
            # calculate entropy
            w_cumsum = np.cumsum(w_temp)
            w_cumsum /= w_cumsum.max()
            w_cumsum = 1 - w_cumsum  # run iterates from high to low logx
            w_temp_term = w_temp[np.where(w_cumsum < 1. - z_term_frac)]
            entropy_gain[l, p, d] = (w_temp_term.shape[0] /
                                     mf.entropy_num_samples(w_temp_term))
            ax.plot(logx, w_temp, linewidth=1, label=label,
                    linestyle=linestyles[p + 1])
            w_temp_max = max(w_temp_max, w_temp.max())
            if plot_r and l == 0:
                label = ('r_given_logx ' + type(prior).__name__ +
                         ' $d = ' + str(n_dim) + '$')
                r = prior.r_given_logx(logx, n_dim)
                ax2.plot(logx, r, linewidth=1, label=label,
                         linestyle=linestyles[p + 1])
print("entropy gain")
print(entropy_gain)
ax.set_ylabel('relative posterior mass', fontsize=label_fontsize)
ax.set_xlabel('$\mathrm{log} X$', fontsize=(label_fontsize))
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
ax.legend(loc=2, ncol=ncol, prop={'size': label_fontsize})
# Output plot
# ------------
if save is True:
    output_root = 'plots/' + root + '.pdf'
    # Save as pdf
    print('Saving as pdf')
    print(output_root)
    plt.savefig(output_root, bbox_inches='tight',
                pad_inches=0.0, dpi=pdf_dpi)
