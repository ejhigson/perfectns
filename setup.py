#!/usr/bin/env python
"""perfectns setup."""
import os
import setuptools


def read_file(fname):
    """
    For using the README file as the long description.
    Taken from https://pythonhosted.org/an_example_pypi_project/setuptools.html
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(name='perfectns',
                 version='2.0.0',
                 description=('Dynamic and standard nested sampling '
                              'for spherically symmetric likelihoods and '
                              'priors.'),
                 long_description=read_file('README.md'),
                 url='https://github.com/ejhigson/perfectns',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 license='MIT',
                 keywords='nested-sampling dynamic-nested-sampling',
                 classifiers=[  # Optional
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Topic :: Scientific/Engineering :: Astronomy',
                     'Topic :: Scientific/Engineering :: Physics',
                     'Topic :: Scientific/Engineering :: Visualization',
                     'Topic :: Scientific/Engineering :: Information Analysis',
                 ],
                 packages=['perfectns'],
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'pandas',
                                   'matplotlib>=2.1.0',
                                   'mpmath',
                                   'nestcheck'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'])
