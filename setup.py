#!/usr/bin/env python

from distutils.core import setup
from subprocess import check_call, PIPE
from os import path, unlink, environ

if not 'ICECUBE_PASSWORD' in environ:
    raise EnvironmentError('You need to set the environment variable ICECUBE_PASSWORD to the icecube user password.')

cwd = path.join(path.dirname(__file__), 'gen2_analysis', 'data')
check_call(['curl', '--fail', '-u', 'icecube:'+environ['ICECUBE_PASSWORD'], '-O', 'http://convey.icecube.wisc.edu/data/user/jvansanten/projects/2015/gen2_benchmark/data/archive.tar.gz'], cwd=cwd)
check_call(['tar', 'xzf', 'archive.tar.gz'], cwd=cwd)
unlink(path.join(path.dirname(__file__), 'gen2_analysis', 'data', 'archive.tar.gz'))

setup(name='gen2-analysis',
      version='0.1',
      description='IceCube-Gen2 benchmark analysis suite',
      author='Jakob van Santen',
      author_email='jvansanten@icecube.wisc.edu',
      url='http://icecube.wisc.edu/~jvansanten/gen2_analysis/',
      packages=['gen2_analysis'],
      package_data={'gen2_analysis': ['data/**/*']}
     )
