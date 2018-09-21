perfectns
=========

.. image:: https://travis-ci.org/ejhigson/perfectns.svg?branch=master
   :target: https://travis-ci.org/ejhigson/perfectns
.. image:: https://coveralls.io/repos/github/ejhigson/perfectns/badge.svg?branch=master
   :target: https://coveralls.io/github/ejhigson/perfectns?branch=master&service=github
.. image:: https://readthedocs.org/projects/perfectns/badge/?version=latest
   :target: http://perfectns.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://api.codeclimate.com/v1/badges/b04cc235c8f73870029c/maintainability
   :target: https://codeclimate.com/github/ejhigson/perfectns/maintainability
   :alt: Maintainability
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ejhigson/perfectns/blob/master/LICENSE

``perfectns`` performs dynamic nested sampling and standard nested sampling for spherically symmetric likelihoods and priors, and analyses the samples produced.
The spherical symmetry allows the nested sampling algorithm to be followed "perfectly" - i.e. without implementation-specific errors correlations between samples.
``perfectns`` is intended for use in research into the statistical properties of nested sampling, and to provide a benchmark for testing the performance of nested sampling software packages used for practical problems - which rely on numerical techniques to produce approximately uncorrelated samples.
For details of the theory behind the software and of how it performs perfect nested sampling, see the `theory section <http://perfectns.readthedocs.io/en/latest/theory.html>`_ of the documentation.

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

If this code is useful for your academic research, please cite the dynamic nested sampling paper. The BibTeX is:

.. code-block:: tex

    @article{Higson2017,
    author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
    title={Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation},
    journal={arXiv preprint arXiv:1704.03459},
    url={https://arxiv.org/abs/1704.03459},
    year={2017}}


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
