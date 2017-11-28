import os
from distutils.core import setup


def read_file(fname):
    """
    For using the README file as the long description.
    Taken from https://pythonhosted.org/an_example_pypi_project/setuptools.html
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='PerfectNS',
      version='1.0.0',
      author='Edward Higson',
      author_email='ejhigson@gmail.com',
      description=('Performs dynamic and standard nested sampling for '
                   'spherically symmetric likelihoods and priors.'),
      url='https://github.com/ejhigson/PerfectNS',
      long_description=read_file('README.md'),
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'mpmath',
                        'tqdm'],
      packages=['PerfectNS'])
