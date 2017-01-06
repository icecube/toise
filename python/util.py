
import os
import numpy as np

if not 'I3_SRC' in os.environ:
	raise RuntimeError("I3_SRC must be defined in the environment! (run env-shell.sh)")

data_dir = os.path.join(os.environ['I3_SRC'], 'gen2_analysis', 'resources', 'data')

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
