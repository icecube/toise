
import os

if not 'I3_SRC' in os.environ:
	raise RuntimeError("I3_SRC must be defined in the environment! (run env-shell.sh)")

data_dir = os.path.join(os.environ['I3_SRC'], 'gen2_analysis', 'resources', 'data')

def center(x):
	return 0.5*(x[1:] + x[:-1])
