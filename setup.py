#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
from subprocess import check_call, PIPE
from os import path, unlink, environ, mkdir

cwd = path.join(path.dirname(__file__), 'gen2_analysis', 'data')
if not path.isdir(cwd):
    mkdir(cwd)

# if not 'ICECUBE_PASSWORD' in environ:
#     raise EnvironmentError('You need to set the environment variable ICECUBE_PASSWORD to the icecube user password.')
# check_call(['curl', '--fail', '-u', 'icecube:'+environ['ICECUBE_PASSWORD'], '-O', 'https://convey.icecube.wisc.edu/data/user/jvansanten/projects/2015/gen2_benchmark/data/archive.tar.gz'], cwd=cwd)
# check_call(['tar', 'xzf', 'archive.tar.gz'], cwd=cwd)
# unlink(path.join(path.dirname(__file__), 'gen2_analysis', 'data', 'archive.tar.gz'))

check_call(['curl', '--fail', '-o', 'minimal-archive.tar.gz', 'https://sandbox.zenodo.org/record/981577/files/minimal-archive.tar.gz?download=1'], cwd=cwd)
check_call(['tar', 'xzf', 'minimal-archive.tar.gz'], cwd=cwd)
unlink(path.join(path.dirname(__file__), 'gen2_analysis', 'data', 'minimal-archive.tar.gz'))

setup(name='gen2-analysis',
      version='0.1',
      description='IceCube-Gen2 benchmark analysis suite',
      author='Jakob van Santen',
      author_email='jvansanten@icecube.wisc.edu',
      url='http://icecube.wisc.edu/~jvansanten/gen2_analysis/',
      packages=find_packages(),
      package_data={'gen2_analysis': ['data/**/*']},
      entry_points={
          'console_scripts': [
              'gen2-figure-data = gen2_analysis.figures.cli:make_figure_data',
              'gen2-plot = gen2_analysis.figures.cli:make_figure',
          ]
      }
     )
