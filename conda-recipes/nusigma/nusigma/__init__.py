from ._nusigma import *

# set path to table directory
from os.path import dirname, realpath, expandvars

basedir = dirname(realpath(expandvars(__file__))) + "/"
nusetup(basedir)
del dirname, realpath, nusetup
