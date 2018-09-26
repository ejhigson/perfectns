---
title: '``perfectns``: perfect dynamic and standard nested sampling for spherically symmetric likelihoods and priors'
tags:
  - Python
  - nested sampling
  - dynamic nested sampling
  - Bayesian inference
authors:
  - name: Edward Higson
    orcid: 0000-0001-8383-4614
    affiliation: "1, 2"
affiliations:
 - name: Cavendish Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
 - name: Kavli Institute for Cosmology, Madingley Road, Cambridge, CB3 0HA, UK
   index : 2
date: 21 September 2018
bibliography: paper.bib
---

# Summary

Nested sampling [@Skilling2006] is a popular Monte Carlo method for computing Bayesian evidences and generating posterior samples given some likelihood and prior.
Both standard nested sampling and its more general variant dynamic nested sampling [@Higson2017b] require sampling randomly from the prior within a hard likelihood constraint.
This is a computationally challenging problem, and typically only be done approximately for practical problems.
Popular methods include rejection sampling, utilized by ``MultiNest`` [@Feroz2008; @Feroz2009; @Feroz2013], and slice sampling, which is used by ``PolyChord`` [@Handley2015a; @Handley2015b] and ``dyPolyChord`` [@Higson2018dypolychord; @Higson2017b].
However all such approximate techniques can lead to additional errors, for example due to correlated samples or missing a mode of a multimodal posterior; for more details see [@Higson2018a].

In order to test the statistical properties of the nested sampling algorithm or check numerical implementations, it is useful to follow the approach introduced by [@Keeton2011] and consider special cases where totally uncorrelated samples can be produced within hard likelihood constrains.
As a result the nested sampling algorithm can be performed perfectly; we term this *perfect nested sampling*.

``perfectns`` performs perfect nested sampling for spherically symmetric likelihoods and priors; its specialised design and ability to produce perfectly uncorrelated samples makes it highly effective for use with spherically symmetric problems.
Furthermore, such problems provides a rich source of test cases for assessing the capacity of other software implementations to perform the nested sampling algorithm accurately, and for use statistical research into nested sampling.
In fact, Section 3 of [@Higson2017a] shows that any perfect nested sampling calculation can in principle be transformed into a spherically symmetric form compatible with ``perfectns`` while retaining its statistical properties.
Such transformations can be used to generate a wider range of test cases, although it can be mathematically challenging and is not feasible for most practical problems.

``perfectns`` requires ``mpmath`` [@mpmath], ``matplotlib`` [@matplotlib], ``nestcheck`` [@Higson2018nestcheck; @Higson2018a; @Higson2017a], ``pandas`` [@pandas], ``numpy`` [@numpy] and ``scipy`` [@scipy].

``perfectns`` was used in the development of the dynamic nested sampling algorithm [@Higson2017b], and for making many of the numerical tests and plots in the dynamic nested sampling paper.
It was also used in testing ``dyPolyChord`` [@Higson2018dypolychord], and numerical tests and plots in [@Higson2017a] were made using earlier versions of the package.
The source code for ``perfectns`` has been archived to Zenodo [@zenodoperfectns].

# Acknowledgements

I am grateful to Will Handley, Anthony Lasenby and Mike Hobson for their help and advice in the research which lead to the creation of ``perfectns``.

# References
