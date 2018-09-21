perfectns
=========

[![Build Status](https://travis-ci.org/ejhigson/perfectns.svg?branch=master)](https://travis-ci.org/ejhigson/perfectns.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/ejhigson/perfectns/badge.svg?branch=master)](https://coveralls.io/github/ejhigson/perfectns?branch=master)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ejhigson/perfectns/blob/master/LICENSE)

`perfectns` performs dynamic nested sampling and standard nested sampling for spherically symmetric likelihoods and priors, and analyses the samples produced.
These cases are ideal for studying nested sampling as the algorithms can be followed "perfectly" - i.e. without implementation-specific errors from correlated samples (see [Higson et al., 2018,](http://arxiv.org/abs/1804.06406) for a detailed discussion).
This package contains the code used to generate results in the dynamic nested sampling paper [(Higson et al., 2017b)](https://arxiv.org/abs/1704.03459).

## Background

Nested sampling is a method for Bayesian computation which given some likelihood L(theta) and prior P(theta) will generate posterior samples and an estimate of the Bayesian evidence Z; for more details see [Higson et al. (2017a)](https://arxiv.org/abs/1703.09701) and [Higson et al. (2017b)](https://arxiv.org/abs/1704.03459).

Nested sampling works by sampling some number of points randomly from the prior then iteratively replacing the point with the lowest likelihood with another point sampled from the region of the prior with a higher likelihood.
Generating uncorrelated samples within the likelihood constraint is computationally challenging and can only be done approximately by software such as MultiNest and PolyChord.
This package uses special cases where uncorrelated samples can be easily drawn from within some iso-likelihood contour to perform nested sampling perfectly in the manner described by [Keeton (2010)](https://academic.oup.com/mnras/article/414/2/1418/977810).

## Likelihoods and Priors

This package uses only spherical likelihoods and priors L(r) and P(r), where the radial coordinate r = |theta|.
Any perfect nested sampling evidence or parameter estimation calculation is equivalent to some problem with spherically symmetric likelihoods and priors (see Section 3 of [Higson et al. (2017a)](https://arxiv.org/abs/1703.09701) for more details), so with suitable choices of L(r), P(r) and the parameter estimation quantity a wide variety of tests can be performed with this package.
L(r) must be a monotonically decreasing function of the radius r, so that the likelihood increases as r decreases to a maximum at r=0.

Nested statistically estimating the shrinkage in the fraction of the prior volume remaining X.
This package generates nested sampling runs by sampling from the known distribution of prior volumes X, then using the prior to find the corresponding radial coordinates r(X) and the likelihood to find the likelihood values L(r(X)).
Parameter vectors theta_i for each point i are sampled from the hyper-spherical shell with radius r_i.

New likelihoods and priors can be added to `likelihoods.py` and `priors.py`, and new parameter estimation quantities can be added to `estimators.py`.
See the documentation in each file for more details.

## Install

`perfectns` works for Python >= 3.5; for a list of its dependencies see the `setup.py` file.
It can be installed with pip:

```
pip install perfectns
```

Alternatively, you can download the latest version of `perfectns` and install it by cloning the git repository:

```
git clone https://github.com/ejhigson/perfectns.git
cd perfectns
python setup.py install
```
Both of these methods also automatically install any dependencies which are not already satisfied by your system.

You can check your installation is working using the test suite (this requires `nose`) by running the following command in the install's root directory:

```
nosetests
```

To also get code coverage information (requires the `coverage` package)
```
nosetests --with-coverage --cover-erase --cover-package=perfectns
```

If all the tests pass, the install should be working.

## Demo and example use

See the `demo.ipynb` notebook [here](https://github.com/ejhigson/perfectns/blob/master/demos/demo.ipynb) for a demonstration of `perfectns`' functionality.

`perfectns` was used to produce many of the results and figures in the dynamic nested sampling paper; you can download the code at [https://github.com/ejhigson/dns](https://github.com/ejhigson/dns).
