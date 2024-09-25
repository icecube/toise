# set path to table directory
from os.path import dirname, expandvars, realpath

from ._nusigma import *

basedir = dirname(realpath(expandvars(__file__))) + "/"
nusetup(basedir)
del dirname, realpath, nusetup
