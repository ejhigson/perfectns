# perfect_nested_sampling

Performs standard and dynamic nested sampling for spherically symmetric likelihoods and priors.
In these cases the algorithms can be followed perfectly without any implementation-specific errors, allowing testing.

## Introduction

Nested sampling (Skilling 2006) is a method for Bayesian computation which given some likelihood L(theta) and prior P(theta) will generate posterior samples and an estimate of the Bayesian evidence Z.

The algorithm works by sampling some number of points randomly from the prior then iteratively replacing the point with the lowest likelihood with another point sampled from the region of the prior with a higher likelihood.
Sampling within the likelihood constraint is numerically challenging and can only be done approximately by software such as MultiNest and PolyChord.
This module uses special cases where uncorrelated samples can be easily drawn from within some iso-likelihood contour to perform nested sampling perfectly.

This module uses only spherical likelihoods and priors L(r) and P(r), where the radial coordinate r = |theta|.
We further require that L(r) is a monotonically decreasing function of the radius r --- i.e. that the likelihood increases as r decreases to a maximum at r=0 (where all components of theta = 0).
This ensures all iso-likelihood contours are hyper-spherical shells.

Nested sampling works by statistically estimating the shrinkage in the fraction of the prior volume remaining X.
This module samples from the known distribution of X to generate new runs, then uses the prior to find the corresponding radial coordinates r(X) and the likelihood to find the likelihood values L(r(X)).
Parameter vectors theta_i for each point i are sampled from the hyper-spherical shells with radius r_i.

For details of standard and dynamic sampling and the terminology used here,
see 'Sampling errors in nested sampling parameter estimation' (Higson et
al. 2017) and 'Dynamic nested sampling: an improved algorithm for nested
sampling parameter estimation and evidence calculation' (Higson et al.
2017).


'Sampling errors in nested sampling parameter estimation' (Higson et al. 2017)
'Dynamic nested sampling: an improved algorithm for nested
sampling parameter estimation and evidence calculation' (Higson et al.
2017).

