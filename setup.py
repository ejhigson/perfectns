import os
import setuptools


def read_file(fname):
    """
    For using the README file as the long description.
    Taken from https://pythonhosted.org/an_example_pypi_project/setuptools.html
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(name='perfectns',
                 version='1.0.3',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 description=('Performs dynamic and standard nested sampling '
                              'for spherically symmetric likelihoods and '
                              'priors.'),
                 url='https://github.com/ejhigson/perfectns',
                 long_description=read_file('README.md'),
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'pandas',
                                   'mpmath',
                                   'tqdm>=4.11',
                                   'nestcheck',
                                   'futures'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'],
                 packages=['perfectns'])
