#!/usr/bin/env python
"""
Plotting functions.
"""

import warnings
import copy
import numpy as np
import scipy
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import perfectns.nested_sampling as ns
import perfectns.estimators as e
import nestcheck.plots


def plot_rel_posterior_mass(likelihood_list, prior, dim_list, logx, **kwargs):
    """
    Plot analytic distributions of the relative posterior mass for different
    likelihoods and dimensionalities as a function of logx (this is
    proportional to L(X)X, where L(X) is the likelihood).

    Parameters
    ----------
    logx: 1d numpy array
        Logx values at which to calculate posterior mass.
    likelihood_list: list of perfectns likelihood objects
        Likelihoods to plot.
    prior: perfectns prior object
    dim_list: list of ints
        Dimensions to plot for each likelihood.
    figsize: tuple, optional
        Size of figure in inches.
    linestyles: list, optional
        List of different linestyles to use for each likelihood.

    Returns
    -------
    fig: matplotlib figure
        Figure showing relative posterior masses of input likelihoods
    """
    linestyles = kwargs.pop('linestyles', ['solid', 'dashed', 'dotted'])
    figsize = kwargs.pop('figsize', (6.4, 2))
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    fig = plt.figure(figsize=figsize)
    for dim in dim_list:
        for nl, likelihood in enumerate(likelihood_list):
            if type(likelihood).__name__ == 'ExpPower':
                label = 'Exp Power: $b=' + str(likelihood.power) + '$,'
                label = label.replace('0.75', r'\frac{3}{4}')
            else:
                label = type(likelihood).__name__ + ':'
            label += ' $d=' + str(dim) + '$'
            logl = likelihood.logl_given_r(prior.r_given_logx(logx, dim), dim)
            w_rel = nestcheck.plots.rel_posterior_mass(logx, logl)
            plt.plot(logx, w_rel, linestyle=linestyles[nl], label=label)
    ax = plt.gca()
    ax.legend(ncol=2)
    ax.set_yticks([])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('relative posterior mass')
    ax.set_xlabel(r'$\mathrm{log} X$')
    ax.set_xlim(logx.min(), logx.max())
    return fig


def plot_dynamic_nlive(dynamic_goals, settings_in, **kwargs):
    """
    Plot the allocations of live points as a function of logX for different
    dynamic_goal settings.
    Plots also include analytically calculated distributions of relative
    posterior mass and relative posterior mass remaining.

    Parameters
    ----------
    dynamic_goals: list of ints or None
        dynamic_goal setting values to plot.
    settings_in: PerfectNSSettings object
    tuned_dynamic_ps: list of bools, optional
        tuned_dynamic_ps settings corresponding to each dynamic goal settings.
        Defaults to False for all dynamic goals.
    logx_min: float, optional
        Lower limit of logx axis. If not specified this is set to the lowest
        logx reached by any of the runs.
    load: bool, optional
        Should the nested sampling runs be loaded from cache if available?
    save: bool, optional
        Should the nested sampling runs be cached?
    ymax: bool, optional
        Maximum value for plot's nlive axis (yaxis).
    n_run: int, optional
        How many runs to plot for each dynamic goal.
    npoints: int, optional
        How many points to have in the logx array used to calculate and plot
        analytical weights.
    figsize: tuple, optional
        Size of figure in inches.

    Returns
    -------
    fig: matplotlib figure
    """
    tuned_dynamic_ps = kwargs.pop('tuned_dynamic_ps',
                                  [False] * len(dynamic_goals))
    save = kwargs.pop('save', True)
    load = kwargs.pop('load', True)
    npoints = kwargs.pop('npoints', 100)
    n_run = kwargs.pop('n_run', 10)
    # Confine settings edits to within this function
    settings = copy.deepcopy(settings_in)
    run_dict = {}
    # work out n_samples_max from first set of runs
    n_sample_stats = np.zeros((len(dynamic_goals), 2))
    method_names = []  # use list to store labels so order is preserved
    for i, dg in enumerate(dynamic_goals):
        print('dynamic_goal=' + str(dg))
        # Make label
        if dg is None:
            label = 'standard'
        else:
            label = 'dynamic $G=' + str(dg) + '$'
            if tuned_dynamic_ps[i] is True:
                label = 'tuned ' + label
        method_names.append(label)
        settings.dynamic_goal = dg
        settings.tuned_dynamic_p = tuned_dynamic_ps[i]
        temp_runs = ns.get_run_data(settings, n_run, parallel=True,
                                    load=load, save=save)
        n_samples = np.asarray([run['logl'].shape[0] for run in temp_runs])
        n_sample_stats[i, 0] = np.mean(n_samples)
        n_sample_stats[i, 1] = np.std(n_samples, ddof=1)
        if i == 0 and settings.n_samples_max is None:
            settings.n_samples_max = int(n_sample_stats[0, 0] *
                                         (settings.nlive_const - 1) /
                                         settings.nlive_const)
        run_dict[label] = temp_runs
        print('mean samples per run:', n_sample_stats[i, 0],
              'std:', n_sample_stats[i, 1])
    fig = nestcheck.plots.plot_run_nlive(
        method_names, run_dict, post_mass_norm='dynamic $G=1$',
        npoints=npoints, logx_given_logl=settings.logx_given_logl,
        logl_given_logx=settings.logl_given_logx,
        cum_post_mass_norm='dynamic $G=0$', **kwargs)
    # Plot the tuned posterior mass
    if 'tuned dynamic $G=1$' in method_names:
        ax = fig.axes[0]
        logx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], npoints)
        # Get expected magnitude of parameter
        # This is not defined for logx=0 so exclude final value of logx
        param_exp = settings.r_given_logx(logx[:-1]) / np.sqrt(settings.n_dim)
        # Tuned weight is the relative posterior mass times the expected
        # magnitude of the paramer being considered
        w_an = nestcheck.plots.rel_posterior_mass(
            logx, settings.logl_given_logx(logx))
        w_tuned = w_an[:-1] * param_exp
        w_tuned /= np.trapz(w_tuned, x=logx[:-1])
        # Get the normalising constant
        integrals = np.zeros(len(run_dict['tuned dynamic $G=1$']))
        for nr, run in enumerate(run_dict['tuned dynamic $G=1$']):
            logx_run = settings.logx_given_logl(run['logl'])
            logx[0] = 0  # to make lines extend all the way to the end
            # for normalising analytic weight lines
            integrals[nr] = -np.trapz(run['nlive_array'], x=logx_run)
        w_tuned *= np.mean(integrals)
        # Plot the tuned posterior mass
        ax.plot(logx[:-1], w_tuned, linewidth=2, label='tuned importance',
                linestyle='-.', dashes=(6, 1.5, 1, 1.5), color='k')
        ax.legend(ncol=3)
    return fig


def plot_parameter_logx_diagram(settings, ftheta, **kwargs):
    """
    Plots parameter estimation diagram of the type described in Section 3.1 and
    shown in Figure 3 of "Sampling errors in nested sampling parameter
    estimation" (Higson 2017).

    Parameters
    ----------
    settings: PerfectNSSettings object
    ftheta: estimator object
        function of parameters to plot
    ymin: float, optional
        y axis (ftheta) min.
    ymax: float, optional
        y axis (ftheta) max.
    ylabel: str, optional
        y axis (ftheta) label.
    logx_min: float, optional
        Lower limit of logx axis.
    x_points: int, optional
        How many logx points to use in the plots.
    y_points: int, optional
    figsize: tuple, optional
        Size of figure in inches.

    Returns
    -------
    fig: matplotlib figure
    """
    logx_min = kwargs.pop('logx_min', -16.0)
    figsize = kwargs.pop('figsize', (6, 2.5))
    x_points = kwargs.pop('x_points', 300)
    y_points = kwargs.pop('y_points', 300)
    ylab_def = r'$f(\theta)=' + ftheta.latex_name[1:]
    ylab_def = ylab_def.replace(r'\\overline', '').replace(r'\overline', '')
    ylabel = kwargs.pop('ylabel', ylab_def)
    # estimator specific defaults:
    ymax = kwargs.pop('ymax', 10)
    if hasattr(ftheta, 'min_value'):
        ymin = kwargs.pop('ymin', ftheta.min_value)
    else:
        ymin = kwargs.pop('ymin', -ymax)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Plotting settings
    max_sigma = 3.5
    contour_line_levels = [1, 2, 3]
    contour_linewidths = 0.5
    color_scheme = plt.get_cmap('Reds_r')
    contour_color_levels = np.arange(0, 4, 0.075)
    darkred = (200 / 255, 0, 0)  # match the color of the tikz evidence picture
    # Initialise figure
    # -----------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 40, 1],
                           height_ratios=[1, 5])
    gs.update(wspace=0.1, hspace=0.1)
    # plot weights
    # ------------
    weight = plt.subplot(gs[0, -2])
    x_setup = np.linspace(logx_min, 0, num=x_points)
    # Calculate posterior weight as a function of logx
    logw = settings.logl_given_logx(x_setup) + x_setup
    w_rel = np.exp(logw - logw.max())
    w_rel[0] = 0.0  # to make fill work
    w_rel[-1] = 0.0  # to make fill work
    plt.fill(x_setup, w_rel, color=darkred)
    weight.set_xticklabels([])
    weight.set_yticklabels([])
    weight.set_yticks([])
    weight.set_ylim([0.0, 1.1])
    weight.set_xlim([logx_min, 0])
    w_patch = matplotlib.patches.Patch(color=darkred,
                                       label='relative posterior mass')
    weight.legend(handles=[w_patch], loc=2 if settings.n_dim <= 4 else 1)
    # plot contours
    # -------------
    y_setup = np.linspace(ymin, ymax, num=y_points)
    x_grid, y_grid = np.meshgrid(x_setup, y_setup)
    # label with negative index for adding/removing extra posterior mean plot
    fgivenx = plt.subplot(gs[1, -2])
    cdf_fgivenx = cdf_given_logx(ftheta, y_grid, x_grid, settings)
    assert cdf_fgivenx.min() >= 0
    assert cdf_fgivenx.max() <= 1
    sigma_fgivenx = sigma_given_cdf(cdf_fgivenx)
    if not np.all(np.isinf(sigma_fgivenx)):
        # Plot the filled contours onto the axis ax
        cs1 = fgivenx.contourf(x_grid, y_grid, sigma_fgivenx,
                               cmap=color_scheme, levels=contour_color_levels,
                               vmin=0, vmax=max_sigma)
        for col in cs1.collections:  # remove annoying white lines
            col.set_edgecolor('face')
        # Plot some sigma-based contour lines
        cs2 = fgivenx.contour(
            x_grid, y_grid, sigma_fgivenx,
            colors='k',
            linewidths=contour_linewidths,
            levels=contour_line_levels, vmin=0, vmax=max_sigma
        )
    fgivenx.set_xlabel(r'$\mathrm{log} \, X $')
    fgivenx.set_yticklabels([])
    # Add ftilde if it is available
    if hasattr(ftheta, 'ftilde'):
        ftilde = ftheta.ftilde(x_setup, settings)
        fgivenx.plot(x_setup, ftilde, color='k', linestyle='--', dashes=(2, 3),
                     linewidth=1, label='$\\tilde{f}(X)$')
    # set y limits as grid goes below limits to make cdf to pdf calc work
    fgivenx.set_ylim([ymin, ymax])
    fgivenx.set_xlim([logx_min, 0])
    # plot posterior
    # --------------
    post_cdf, support = posterior_cdf(ftheta, y_setup, x_setup, settings)
    x_posterior, y_posterior = np.meshgrid(np.linspace(0, 1, num=2),
                                           support)
    z_posterior = sigma_given_cdf(np.vstack((post_cdf, post_cdf)).T)
    posterior = plt.subplot(gs[1, -3])
    # Plot the filled contours onto the axis ax
    cs1 = posterior.contourf(
        x_posterior, y_posterior, z_posterior,
        cmap=color_scheme,
        levels=contour_color_levels, vmin=0, vmax=max_sigma
    )
    for col in cs1.collections:  # remove annoying white lines
        col.set_edgecolor('face')
    # Plot some sigma-based contour lines
    cs2 = posterior.contour(x_posterior, y_posterior, z_posterior, colors='k',
                            linewidths=contour_linewidths,
                            levels=contour_line_levels, vmin=0, vmax=max_sigma)
    # add the posterior expectation if it exists
    posterior_mean = e.get_true_estimator_values(ftheta, settings)
    if not np.isnan(posterior_mean):
        fgivenx.plot([logx_min, 0], [posterior_mean, posterior_mean],
                     color='k', linestyle=':', dashes=(0.5, 1), linewidth=1,
                     label=r'$\mathrm{E}[f(\theta)|\mathcal{L},\pi]$')
        posterior.plot([0, 1], [posterior_mean, posterior_mean], color='k',
                       linestyle=':', linewidth=1,
                       label=r'$\mathrm{E}[f(\theta)|\mathcal{L},\pi]$')
        fgivenx.legend(loc=2)
    posterior.set_xticklabels([])
    posterior.set_xticks([])
    # add labelpad to avoid clash with xtick labels
    posterior.set_xlabel('posterior', labelpad=18)
    posterior.set_ylabel(ylabel)
    posterior.set_ylim([ymin, ymax])
    # plot Colorbar key
    # ----------------
    # place in seperate subplot
    colorbar_plot = plt.subplot(gs[1, -1])
    colorbar = plt.colorbar(cs1, cax=colorbar_plot, ticks=[1, 2, 3])
    colorbar.solids.set_edgecolor('face')
    colorbar.ax.set_yticklabels(
        [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
    colorbar.add_lines(cs2)
    return fig


def sigma_given_cdf(cdf):
    """
    Maps cdf values in [0,1] to number of standard deviations from the median.
    scipy.special.erfinv is defined in [-1,+1] mapping to [-inf,+inf].
    We want to map a CDF to the number of sigma from the median - i.e. from
    [0,+1] to [0,+inf] - so we need the argument (2*cdf - 1), and to take abs
    to get a positive answer.
    erf is definted in terms of int e^(-t^2) which corresponds to a Gaussian
    with sigma = 1/sqrt(2). So we must multiply sigma_temp by sqrt(2)
    """
    sigma_temp = abs(scipy.special.erfinv((cdf * 2) - 1))
    return np.sqrt(2) * sigma_temp


def cdf_given_logx(estimator, value, logx, settings):
    """
    Calculate CDF at where each column represents the CDF of the
    distribution of ftheta values on some iso-likelihood contour L = L(X).

    Parameters
    ----------
    estimator: estimator object
        Function whose values we are getting the CDF of.
    value: numpy array
        Function values at which to evaluate the CDF.
    logx: numpy array of same size and shape as value.
        Logx values specifying contours - we calculate the CDF on each contour.
    settings: PerfectNSSettings object

    Returns
    -------
    cdf: numpy array of same size and shape as value and logx
    """
    assert value.shape == logx.shape
    if estimator.__class__.__name__ == 'ParamMean':
        # From the sampling errors paper (Higson 2017) Section 4 the cdf of
        # p1^2 is a beta distribution
        r = settings.r_given_logx(logx)
        p1squared_cdf = scipy.stats.beta.cdf((value / r) ** 2, 0.5,
                                             (settings.n_dim - 1) / 2.)
        cdf = 0.5 + (0.5 * np.sign(value) * p1squared_cdf)
    elif estimator.__class__.__name__ == 'ParamSquaredMean':
        # From my errors paper section 4 this is just a beta distribution
        r = settings.r_given_logx(logx)
        cdf = scipy.stats.beta.cdf(value * (r ** -2), 0.5,
                                   (settings.n_dim - 1) / 2.)
    elif estimator.__class__.__name__ == 'RMean':
        cdf = np.zeros(logx.shape)
    else:
        warnings.warn(('cdf not available for ' + estimator.__class__.__name__
                       + '. Will use cdf=0 everywhere.'), UserWarning)
        cdf = np.zeros(logx.shape)
    assert cdf.min() >= 0, "cdf.min() = " + str(cdf.min()) + " < 0"
    assert cdf.max() <= 1, "cdf.max() = " + str(cdf.max()) + " > 1"
    return cdf


def posterior_cdf(estimator, values, logx, settings):
    """
    Calculates the 1d posterior cumulative distribution function (CDF) for some
    estimator given the likelihood and prior settings.

    Parameters
    ----------
    estimator: estimator object
        Function whose values we are getting the CDF of.
    value: 1d numpy array
        Function values at which to evaluate the CDF.
    logx: 1d numpy array
        Logx values over which to numericallly marginalise the probability
        distribution.
    settings: PerfectNSSettings object

    Returns
    -------
    cdf: numpy array of same size and shape as values
    """
    if estimator.__class__.__name__ == 'RMean':
        values_logx = settings.logx_given_r(values)
        log_pdf = values_logx + settings.logl_given_r(values)
        pdf = np.exp(log_pdf - log_pdf.max())
        cdf = np.cumsum(pdf)
        cdf /= cdf.max()
        assert cdf.min() >= 0, "cdf.min() = " + str(cdf.min()) + " < 0"
        assert cdf.max() <= 1, "cdf.max() = " + str(cdf.max()) + " > 1"
        return cdf, values
    else:
        # calculate pdf numberically from cdfs across different logx values
        # make numerical calculations extend beyond plot limits so cdfs at
        # edges of plot are approximately correct
        # it is important to recalculate at this step rather than just
        # using a bigger array for the fgivenx plot as that makes the .pdf
        # files bigger
        # increase ymax to ensure cdfs from trunkated pdfs are accurate
        ymin_temp = values.min() - (values.max() - values.min())
        ymax_temp = values.max() + (values.max() - values.min())
        y_setup_temp = np.linspace(ymin_temp, ymax_temp,
                                   num=values.shape[0] * 3)
        x_grid_temp, y_grid_temp = np.meshgrid(logx, y_setup_temp)
        cdf_fgivenx_temp = cdf_given_logx(estimator, y_grid_temp, x_grid_temp,
                                          settings)
        logw = settings.logl_given_logx(logx) + logx
        w_rel = np.exp(logw - logw.max())
        pdf_posterior = np.zeros(cdf_fgivenx_temp.shape[0])
        for i, _ in enumerate(pdf_posterior[:-1]):
            # as we have no pdf_givenx this is caclulated from differences
            # in cdf_fgivenx and therefor is 1 index smaller
            pdf_posterior[i] = np.sum((cdf_fgivenx_temp[i + 1, :] -
                                       cdf_fgivenx_temp[i, :]) * w_rel)
        assert pdf_posterior.min() >= 0
        # calculate the cdf numerically from the pdf
        cdf = np.cumsum(pdf_posterior) / np.sum(pdf_posterior)
        return np.clip(cdf, 0, 1), y_setup_temp
