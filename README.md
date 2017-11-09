Perfect Nested Sampling
=======================

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ejhigson/PerfectNestedSampling/LICENSE)


Performs dynamic nested sampling and standard nested sampling for spherically symmetric likelihoods and priors.
In these cases the algorithms can be followed "perfectly" (without implementation-specific errors from correlated sampling), making them ideal for studying nested sampling.


### Introduction

Nested sampling is a method for Bayesian computation which given some likelihood L(theta) and prior P(theta) will generate posterior samples and an estimate of the Bayesian evidence Z.
For more details see [Skilling's original nested sampling paper](https://projecteuclid.org/euclid.ba/1340370944) and the [dynamic nested sampling paper](https://arxiv.org/abs/1704.03459), which uses this code for its numerical tests.
The module also includes an implementation of nested sampling error estimates using resampling (described [here](https://arxiv.org/abs/1703.09701)).

Nested sampling works by sampling some number of points randomly from the prior then iteratively replacing the point with the lowest likelihood with another point sampled from the region of the prior with a higher likelihood.
Generating uncorrelated samples within the likelihood constraint is numerically challenging and can only be done approximately by software such as MultiNest and PolyChord.
This module uses special cases where uncorrelated samples can be easily drawn from within some iso-likelihood contour to perform nested sampling perfectly.

### Likelihoods and Priors

This module uses only spherical likelihoods and priors L(r) and P(r), where the radial coordinate r = |theta|.
We further require that L(r) is a monotonically decreasing function of the radius r --- i.e. that the likelihood increases as r decreases to a maximum at r=0 (where all components of theta = 0).
This ensures all iso-likelihood contours are hyper-spherical shells.

Nested statistically estimating the shrinkage in the fraction of the prior volume remaining X.
This module samples from the known distribution of prior volumes X to generate new runs, then uses the prior to find the corresponding radial coordinates r(X) and the likelihood to find the likelihood values L(r(X)).
Parameter vectors theta_i for each point i are sampled from the hyper-spherical shells with radius r_i.

New likelihoods and priors can be added `likelihoods.py` and `priors.py` respectively.


### Installation



### Examples

To see a demonstration, run

```
python test_PerfectNestedSampling.py
```
