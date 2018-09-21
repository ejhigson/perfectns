.. _install:

Installation
============

``perfectns`` is compatible with python >=3.5; for a list of its dependencies see the ``setup.py`` file. Installation can be carried out with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

   pip install perfectns

Alternatively, you can download the latest version and install it by cloning `the github
repository <https://github.com/ejhigson/perfectns>`_:

.. code-block:: bash

    git clone https://github.com/ejhigson/perfectns.git
    cd perfectns
    python setup.py install

Note that the github repository may include new changes which have not yet been released on PyPI (and therefore will not be included if installing with pip).
Both of these methods also automatically install any of ``perfectns``'s dependencies which are not already satisfied by your system.

Tests
-----

You can run the test suite with `nose <http://nose.readthedocs.org/>`_. From the root ``perfectns`` directory, run:

.. code-block:: bash

    nosetests

To also get code coverage information (this requires the ``coverage`` package), use:

.. code-block:: bash

    nosetests --with-coverage --cover-erase --cover-package=perfectns

If all the tests pass, the install should be working.
