
import numpy

def stepped_path(edges, bins):
	"""
	Create a stepped path suitable for histogramming
	
	:param edges: bin edges
	:param bins: bin contents
	"""
	if len(edges) != len(bins)+1:
		raise ValueError("edges must be 1 element longer than bins")
	
	x = numpy.zeros((2*len(edges)))
	y = numpy.zeros((2*len(edges)))
	
	x[0::2], x[1::2] = edges, edges
	y[1:-1:2], y[2::2] = bins, bins
	
	return x,y

def plot_profile2d(profile, x, y, levels=[68, 90, 99], colors='k', **kwargs):
	from scipy.stats import chi2
	import matplotlib.pyplot as plt
	
	xv = numpy.unique(profile[x])
	yv = numpy.unique(profile[y])
	shape = (xv.size, yv.size)
	
	ts = 2*(numpy.nanmax(profile['LLH'])-profile['LLH']).reshape(shape)
	pvalue = chi2.cdf(ts.T, 2)*100
	
	ax = plt.gca()
	cs = ax.contour(xv, yv, pvalue, levels=levels, colors=colors, **kwargs)
	if ax.get_xlabel() == '':
		ax.set_xlabel(x)
	if ax.get_ylabel() == '':
		ax.set_ylabel(y)
	return cs
