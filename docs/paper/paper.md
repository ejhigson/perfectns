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
The algorithm requires sampling randomly from the prior within a hard likelihood constraint.
This is a computationally challenging problem, and for a general likelihood and prior can only be done approximately.
Popular methods include rejection sampling, used by ``MultiNest`` [@Feroz2008; @Feroz2009; @Feroz2013], and slice sampling, which is used by ``PolyChord`` [@Handley2015a; @Handley2015b] and ``dyPolyChord`` [@Higson2018dypolychord].
However all such approximate approaches can lead to additional errors, for example due to correlated samples or missing a mode of a multimodal posterior; a detailed discussion can be found in [@Higson2018a].

In order to test the statistical properties of the nested sampling algorithm or check numerical implementations, it is useful to follow the approach used by [@Keeton2010] and consider special cases where totally uncorrelated samples can be produced within hard likelihood constrains.
We term this `perfect nested sampling`.

``perfectns`` performs perfect nested sampling for spherically symmetric likelihoods and priors.
This provides a rich source of test cases for use in statistical research and for testing the ability of software to perform the nested sampling algorithm accurately.
In fact, Section 3 of [@Higson2017a] shows that any perfect nested sampling calculation can in principle be transformed into a spherically symmetric form compatible with ``perfectns`` while retaining its statistical properties.
``perfectns`` requires ``mpmath``[@mpmath], ``matplotlib``[@matplotlib], ``nestcheck`` [@Higson2018nestcheck; @Higson2018a; @Higson2017a], ``pandas`` [@pandas], ``numpy`` [@numpy] and ``scipy`` [@scipy].

``perfectns`` was used in the development of the dynamic nested sampling algorithm [@Higson2017b] and an earlier version was used in the development of the statistical tests in [@Higson2017a]. Numerical tests and plots in both papers were made using versions of the code.
The source code for ``perfectns`` has been archived to Zenodo [@zenodoperfectns].

# Acknowledgements

I am grateful to Will Handley for extensive help using ``PolyChord``, and to Anthony Lasenby and Mike Hobson for their help and advice in the research leading to dynamic nested sampling.

# References
