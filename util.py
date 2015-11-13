
import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def center(x):
	return 0.5*(x[1:] + x[:-1])