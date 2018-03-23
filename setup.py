#!/usr/bin/env python

from distutils.core import setup
from subprocess import check_call
from os import path

check_call(['rsync', '-avz', '--cvs-exclude', 'data.icecube.wisc.edu:/data/user/jvansanten/projects/2015/gen2_benchmark/data/', path.join(path.dirname(__file__), 'gen2_analysis', 'data')])

setup(name='gen2-analysis',
      version='0.1',
      description='IceCube-Gen2 benchmark analysis suite',
      author='Jakob van Santen',
      author_email='jvansanten@icecube.wisc.edu',
      url='http://icecube.wisc.edu/~jvansanten/gen2_analysis/',
      packages=['gen2_analysis'],
      package_data={'gen2_analysis': ['data/**/*']}
     )
