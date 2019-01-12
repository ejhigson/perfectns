Theory
======

Nested sampling is a method for Bayesian computation which given some likelihood :math:`\mathcal{L}(\boldsymbol{\theta})` and prior :math:`\pi(\boldsymbol{\theta})` will generate posterior samples and an estimate of the Bayesian evidence :math:`\mathcal{Z}`. The algorithm works by sampling some number of points randomly from the prior then iteratively replacing the point with the lowest likelihood with another point sampled from the region of the prior with a higher likelihood.

Generating uncorrelated samples within the likelihood constraint is computationally challenging and can only be done approximately by software such as ``MultiNest`` and ``PolyChord``.
This can lead to additional errors; for example due to correlated samples or missing a mode in a multimodal posterior (see `Higson et al., (2019), <https://doi.org/10.1093/mnras/sty3090>`_ for a detailed discussion).
``perfectns`` uses specially symmetric cases where uncorrelated samples can be easily drawn from within some iso-likelihood contour to perform nested sampling perfectly in the manner described by `Keeton (2011) <https://doi.org/10.1111/j.1365-2966.2011.18474.x>`_.

Likelihoods and Priors
----------------------

``perfectns`` uses only spherical likelihoods and priors :math:`\mathcal{L}(r)` and :math:`\pi(r)`, where the radial coordinate :math:`r = | \boldsymbol{\theta} |`.
Any perfect nested sampling evidence or parameter estimation calculation is equivalent to some problem with spherically symmetric likelihoods and priors (see Section 3 of `Higson et al., 2018, <http://arxiv.org/abs/1804.06406>`_ for more details), so with suitable choices of :math:`\mathcal{L}(r)`, :math:`\pi(r)` and the parameter estimation quantity a wide variety of tests can be performed with this package.
:math:`\mathcal{L}(r)` must be a monotone decreasing function of the radius :math:`r`, so that the likelihood increases as r decreases to a maximum at :math:`r=0`.

Nested statistically estimating the shrinkage in the fraction of the prior volume remaining :math:`X`.
This package generates nested sampling runs by sampling from the known distribution of prior volumes :math:`X`, then using the prior to find the corresponding radial coordinates :math:`r(X)` and the likelihood to find the likelihood values :math:`\mathcal{L}(r(X))`.
Parameter vectors :math:`\boldsymbol{\theta}_i` for each point :math:`i` are sampled from the hyper-spherical shell with radius :math:`r_i`.

New likelihoods and priors can be added to ``likelihoods.py`` and ``priors.py``, and new parameter estimation quantities can be added to ``estimators.py``.
See the documentation in each file for more details.
