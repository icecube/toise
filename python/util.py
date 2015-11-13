
import os

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'data')

def center(x):
	return 0.5*(x[1:] + x[:-1])