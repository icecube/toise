
import os
import numpy as np

data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'data'))

def center(x):
	return 0.5*(x[1:] + x[:-1])

def edge(x):
    """ returns bin edges with centers that match x. "inverse" of center(x).
    """
    c = center(x)
    return np.concatenate(([2*x[0]-c[0]], c, [2*x[-1]-c[-1]]))

class constants:
	annum = 365*24*3600.
	cm = 1e2
	cm2 = 1e4

class baseEnum(int):
	name = None
	values = {}
	def __repr__(self):
		return self.name

class metaEnum(type):
	"""Helper metaclass to return the class variables as a dictionary "
	"""

	def __new__(cls, classname, bases, classdict):
		""" Return a new class with a "values" attribute filled
		"""

		newdict = {"values":{}}

		for k in classdict.keys():
			if not (k.startswith('_') or k == 'name' or k == 'values'):
				val = classdict[k]
				member = baseEnum(val)
				member.name = k
				newdict['values'][val] = member
				newdict[k] = member
		
		# Tell each member about the values in the enum
		for k in newdict['values'].keys():
			newdict['values'][k].values = newdict['values']
		# Return a new class with the "values" attribute filled
		return type.__new__(cls, classname, bases, newdict)

class enum(baseEnum):
	"""This class mimicks the interface of boost-python-wrapped enums.
	
Inherit from this class to construct enumerated types that can
be passed to the I3Datatype, e.g.:

	class DummyEnummy(tableio.enum):
		Foo = 0
		Bar = 1
		Baz = 2

	desc = tableio.I3TableRowDescription()
	desc.add_field('dummy', tableio.I3Datatype(DummyEnummy), '', '')
"""
	__metaclass__ = metaEnum

class PDGCode(enum):
	PPlus       =       2212
	He4Nucleus  = 1000020040
	N14Nucleus  = 1000070140
	O16Nucleus  = 1000080160
	Al27Nucleus = 1000130270
	Fe56Nucleus = 1000260560
	NuE         =         12
	NuEBar      =        -12
	NuMu        =         14
	NuMuBar     =        -14
	NuTau       =         16
	NuTauBar    =        -16