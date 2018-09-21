Theory
======

Nested sampling is a method for Bayesian computation which given some likelihood L(theta) and prior P(theta) will generate posterior samples and an estimate of the Bayesian evidence Z. The algorithm works by sampling some number of points randomly from the prior then iteratively replacing the point with the lowest likelihood with another point sampled from the region of the prior with a higher likelihood.

Generating uncorrelated samples within the likelihood constraint is computationally challenging and can only be done approximately by software such as ``MultiNest`` and ``PolyChord``.
This can lead to additional errors; for example due to correlated samples or missing a mode in a multimodal posterior (see `Higson et al., (2018), <http://arxiv.org/abs/1804.06406>`_ for a detailed discussion).
``perfectns`` uses specially symmetric cases where uncorrelated samples can be easily drawn from within some iso-likelihood contour to perform nested sampling perfectly in the manner described by `Keeton (2010) <https://academic.oup.com/mnras/article/414/2/1418/977810>`_.

Likelihoods and Priors
----------------------

This package uses only spherical likelihoods and priors L(r) and P(r), where the radial coordinate r = abs(theta).
Any perfect nested sampling evidence or parameter estimation calculation is equivalent to some problem with spherically symmetric likelihoods and priors (see Section 3 of `Higson et al., 2018, <http://arxiv.org/abs/1804.06406>`_ for more details), so with suitable choices of L(r), P(r) and the parameter estimation quantity a wide variety of tests can be performed with this package.
L(r) must be a monotonically decreasing function of the radius r, so that the likelihood increases as r decreases to a maximum at r=0.

Nested statistically estimating the shrinkage in the fraction of the prior volume remaining X.
This package generates nested sampling runs by sampling from the known distribution of prior volumes X, then using the prior to find the corresponding radial coordinates r(X) and the likelihood to find the likelihood values L(r(X)).
Parameter vectors theta_i for each point i are sampled from the hyper-spherical shell with radius r_i.

New likelihoods and priors can be added to ``likelihoods.py`` and ``priors.py``, and new parameter estimation quantities can be added to ``estimators.py``.
See the documentation in each file for more details.
