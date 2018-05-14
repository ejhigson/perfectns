#!/usr/bin/env python
"""
perfectns setup module.

Based on https://github.com/pypa/sampleproject/blob/master/setup.py.
"""
import os
import setuptools


def get_long_description():
    """Get PyPI long description from its own .rst file as PyPI does not render
    the README well."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, '.pypi_long_desc.rst')) as readme_file:
        long_description = readme_file.read()
    return long_description


setuptools.setup(name='perfectns',
                 version='2.0.1',
                 description=('Dynamic and standard nested sampling '
                              'for spherically symmetric likelihoods and '
                              'priors.'),
                 long_description=get_long_description(),
                 long_description_content_type='text/x-rst',
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
