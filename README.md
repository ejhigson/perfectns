perfectns
=========

[![Build Status](https://travis-ci.org/ejhigson/perfectns.svg?branch=master)](https://travis-ci.org/ejhigson/perfectns.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/ejhigson/perfectns/badge.svg?branch=master)](https://coveralls.io/github/ejhigson/perfectns?branch=master)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ejhigson/perfectns/LICENSE)


### Description

`perfectns` performs dynamic nested sampling and standard nested sampling for spherically symmetric likelihoods and priors, and analyses the samples produced.
These cases are ideal for studying nested sampling as the algorithms can be followed "perfectly" - i.e. without implementation-specific errors from correlated samples (see [Higson et al., 2018,](http://arxiv.org/abs/1804.06406) for a detailed discussion).
This package contains the code used to generate results in the [dynamic nested sampling paper (Higson et al., 2017b)](https://arxiv.org/abs/1704.03459) and provides an example implementation of the algorithm to accompany the paper.

##### Background

Nested sampling is a method for Bayesian computation which given some likelihood L(theta) and prior P(theta) will generate posterior samples and an estimate of the Bayesian evidence Z; for more details see [Higson et al. (2017a)](https://arxiv.org/abs/1703.09701) and [Higson et al. (2017b)](https://arxiv.org/abs/1704.03459).

Nested sampling works by sampling some number of points randomly from the prior then iteratively replacing the point with the lowest likelihood with another point sampled from the region of the prior with a higher likelihood.
Generating uncorrelated samples within the likelihood constraint is computationally challenging and can only be done approximately by software such as MultiNest and PolyChord.
This package uses special cases where uncorrelated samples can be easily drawn from within some iso-likelihood contour to perform nested sampling perfectly in the manner described by [Keeton (2010)](https://academic.oup.com/mnras/article/414/2/1418/977810).

##### Likelihoods and Priors

This package uses only spherical likelihoods and priors L(r) and P(r), where the radial coordinate r = |theta|.
Any perfect nested sampling evidence or parameter estimation calculation is equivalent to some problem with spherically symmetric likelihoods and priors (see Section 3 of [Higson et al. (2017a)](https://arxiv.org/abs/1703.09701) for more details), so with suitable choices of L(r), P(r) and the parameter estimation quantity a wide variety of tests can be performed with this package.
L(r) must be a monotonically decreasing function of the radius r, so that the likelihood increases as r decreases to a maximum at r=0.

Nested statistically estimating the shrinkage in the fraction of the prior volume remaining X.
This package generates nested sampling runs by sampling from the known distribution of prior volumes X, then using the prior to find the corresponding radial coordinates r(X) and the likelihood to find the likelihood values L(r(X)).
Parameter vectors theta_i for each point i are sampled from the hyper-spherical shell with radius r_i.

##### Implementation

The package uses a logarithmic number system for likelihoods and weights throughout to prevent overflow errors from extreme numerical values; this is particularly important in high dimensions.
Most of the computational work is done within the `numpy` package for computational efficiency.
The package also makes extensive use of the `nestcheck` nested sampling functions.

New likelihoods and priors can be added to `likelihoods.py` and `priors.py`, and new parameter estimation quantities can be added to `estimators.py`.
See the documentation in each file for more details.

### Getting Started

`perfectns` works for Python >= 3.5; For a list of its dependencies see `setup.py`..
To pip install `perfectns` and its dependencies:

```
pip install perfectns
```

You can check your installation is working using the test suite (this requires nose):

```
python setup.py test
```

##### Examples

See the `demo.ipynb` notebook contains a demonstration of `perfectns`' functionality.
