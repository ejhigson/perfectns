perfectns
=========

.. image:: https://travis-ci.org/ejhigson/perfectns.svg?branch=master
   :target: https://travis-ci.org/ejhigson/perfectns
.. image:: https://coveralls.io/repos/github/ejhigson/perfectns/badge.svg?branch=master
   :target: https://coveralls.io/github/ejhigson/perfectns?branch=master&service=github
.. image:: https://readthedocs.org/projects/perfectns/badge/?version=latest
   :target: http://perfectns.readthedocs.io/en/latest/?badge=latest
.. image:: http://joss.theoj.org/papers/10.21105/joss.00985/status.svg
   :target: https://doi.org/10.21105/joss.00985
.. image:: https://api.codeclimate.com/v1/badges/b04cc235c8f73870029c/maintainability
   :target: https://codeclimate.com/github/ejhigson/perfectns/maintainability
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ejhigson/perfectns/blob/master/LICENSE

Nested sampling (`Skilling, 2006 <https://projecteuclid.org/euclid.ba/1340370944>`_) is a popular numerical method for Bayesian computation, which simultaneously generates samples from the posterior distribution and an estimate of the Bayesian evidence for a given likelihood and prior.
Dynamic nested sampling (`Higson et al., 2019a <https://doi.org/10.1007/s11222-018-9844-0>`_) is a generalisation of the nested sampling algorithm which can provide order-of-magnitude increases in computational efficiency.

``perfectns`` performs dynamic nested sampling and standard nested sampling for spherically symmetric likelihoods and priors, and analyses the samples produced.
The spherical symmetry allows the nested sampling algorithm to be followed "perfectly" - i.e. without additional errors due to correlations between samples, which are present in other nested sampling software.

The specialised methods used by ``perfectns`` make it highly effective and reliable for spherically symmetric calculations.
The software is also intended for use in research into the statistical properties of nested sampling, and to provide a benchmark for testing the performance of nested sampling software packages used for practical problems - which rely on numerical techniques to produce approximately uncorrelated samples.
For details of the theory behind the software and of how it performs perfect nested sampling, see the see the `theory section <http://perfectns.readthedocs.io/en/latest/theory.html>`_ of the documentation.

To get started, see the `installation instructions <http://perfectns.readthedocs.io/en/latest/install.html>`_ and the `demo <http://perfectns.readthedocs.io/en/latest/demo.html>`_.
For more examples of ``perfectns``'s use, see the code used to make the results and figures in the dynamic nested sampling paper (https://github.com/ejhigson/dns).


Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   theory
   install
   demo
   api

Attribution
-----------

If ``perfectns`` is useful for your academic research, please cite the two papers introducing the software and the dynamic nested sampling algorithm. The BibTeX is:

.. code-block:: tex

    @article{Higson2018perfectns,
    author = {Higson, Edward},
    title = {perfectns: perfect dynamic and standard nested sampling for spherically symmetric likelihoods and priors},
    doi = {10.21105/joss.00985},
    journal = {Journal of Open Source Software},
    number = {30},
    pages = {985},
    volume = {3},
    year = {2018}}

    @article{Higson2019dynamic,
    author={Higson, Edward and Handley, Will and Hobson, Michael and Lasenby, Anthony},
    title={Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation},
    year={2019},
    journal={Statistics and Computing},
    doi={10.1007/s11222-018-9844-0},
    url={https://doi.org/10.1007/s11222-018-9844-0},
    archivePrefix={arXiv},
    arxivId={1704.03459}}


Changelog
---------

The changelog for each release can be found at https://github.com/ejhigson/perfectns/releases.

Contributions
-------------

Contributions are welcome! Development takes place on github:

- source code: https://github.com/ejhigson/perfectns;
- issue tracker: https://github.com/ejhigson/perfectns/issues.

When creating a pull request, please try to make sure the tests pass and use numpy-style docstrings.

If you have any questions or suggestions please get in touch (e.higson@mrao.cam.ac.uk).

Authors & License
-----------------

Copyright 2018 Edward Higson and contributors (MIT Licence).
