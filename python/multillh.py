import numpy, scipy.optimize

class NuisanceParam:
	"""
	NuisanceParam is a simple class implementing a free parameter in the
	fit, optionally constrained with either flat or Gaussian priors. This
	class adds only a knob and prior: it does not contribute in any way
	to the Poisson mean in any bin.
	"""
	def __init__(self, prior=1, gauserror = None, min=None, max=None):
		"""
		Seeds the minimizer with the value of prior, optionally
		constrained by a Gaussian prior around this value (gauserror)
		and/or by half- or fully-closed box constraints (min/max)
		"""
		self.seed = prior
		self.uncertainty = gauserror
		self.min = min
		self.max = max
	def prior(self, value, **kwargs):
		if self.uncertainty == None:
			return 0
		return -(value - self.seed)**2/(2.*self.uncertainty**2)

class Combination(object):
	def __init__(self, components):
		"""
		:param components: a dict of str: (component, livetime)
		"""
		self._components = components
		for label, (component, livetime) in self._components.items():
			if hasattr(component, 'min'):
				self.min = component.min
			if hasattr(component, 'max'):
				self.max = component.max
			if hasattr(component, 'seed'):
				self.seed = component.seed
	
	def prior(self, value, **kwargs):
		v = self._components.values()[0][0]
		if hasattr(v, 'prior'):
			return v.prior(value, **kwargs)
		else:
			return 0.
	
	def expectations(self, *args, **kwargs):
		exes = dict()
		for label, (component, livetime) in self._components.items():
			if hasattr(component.expectations, "__call__"):
				subex = component.expectations(*args, **kwargs)
			else:
				subex = component.expectations
			for k, v in subex.items():
				exes[label+'_'+k] = livetime*v
		return exes
	
	@property
	def bin_edges(self):
		edges = dict()
		for label, (component, livetime) in self._components.items():
			# everything returns a dict with 1 entry, 'tracks'. ew.
			edges[label+'_tracks'] = component.bin_edges

		return edges

	def differential_chunks(self, *args, **kwargs):
		generators = dict()
		for label, (component, livetime) in self._components.items():
			generators[label] = (component.differential_chunks(*args, **kwargs), livetime)
		while True:
			try:
				components = dict()
				for label, (generator, livetime) in generators.items():
					e_center, component = next(generator)
					components[label] = (component, livetime)
				combo = Combination(components)
				combo.energy_range = component.energy_range
				combo.energy_center = e_center
				yield combo
			except StopIteration:
				break

class LLHEval:
	"""
	Object containing a Poisson likelihood fit
	"""
	def __init__(self, data, unbinned = False):
		"""
		Initialize fit with data. This should be a dictionary from
		keys -- of any type -- shared with the fit components to 
		numpy arrays of arbitrary dimensionality containing histograms.

		If the keyword argument unbinned is set to True, data should
		instead be a dictionary mapping keys to a python list of numpy
		arrays, each describing the probability momemt function for one
		event, binned the same way as the simulated distributions.
		"""
		self.components = dict()
		self.data = data
		self.unbinned = unbinned
	def add_component(self, name, component):
		self.components[name] = component
	def bounds(self):
		"""
		Returns dictionary of tuples of the bounds on each parameter
		(None if unbounded)
		"""
		retval = dict()
		for c in self.components:
			min = 0
			max = None
			if hasattr(self.components[c], 'min'):
				min = self.components[c].min
			if hasattr(self.components[c], 'max'):
				max = self.components[c].max
			retval[c] = (min, max)
		return retval
	def expectations(self, **kwargs):
		"""
		Returns dictionary of numpy arrays in the same format as the
		input data given the parameter values specified in keyword
		arguments
		"""
		lamb = dict()
		for param in kwargs:
			c = self.components[param]
			if not hasattr(c, 'expectations'):
				continue
			if callable(c.expectations):
				expec = c.expectations(**kwargs)
			else:
				expec = c.expectations
			for prop in expec:
				llh_bit = kwargs[param]*expec[prop]
				if prop in lamb:
					lamb[prop] += llh_bit
				else:
					lamb[prop] = llh_bit
		return lamb
	def llh(self, **kwargs):
		"""
		Evaluates log-likelihood for parameter values specified in
		keyword arguments
		"""
		lamb = self.expectations(**kwargs)
		llh = 0
		for param in kwargs:
			if hasattr(self.components[param], 'prior'):
				llh += self.components[param].prior(kwargs[param], **kwargs)

		for prop in lamb:
			log_lambda = numpy.log(lamb[prop])
			log_lambda[numpy.isinf(log_lambda)] = 0
			if self.unbinned:
				norm = numpy.sum(lamb[prop])
				for event in self.data[prop]:
					llh += numpy.sum(event*lamb[prop])/norm
				llh -= norm
			else:
				llh += numpy.sum(self.data[prop]*log_lambda - lamb[prop])

		return llh
	
	def llh_contributions(self, **kwargs):
		"""
		Evaluates log-likelihood contributions for parameter values specified in
		keyword arguments. Priors are not included.
		"""
		lamb = self.expectations(**kwargs)
		llh = dict()
		for prop in lamb:
			log_lambda = numpy.log(lamb[prop])
			log_lambda[numpy.isinf(log_lambda)] = 0
			llh[prop] = self.data[prop]*log_lambda - lamb[prop]
		return llh
	
	def sat_llh(self, **kwargs):
		"""
		Evaluates LLH for a saturated Poisson model. Note
		that since this does not involve the parameters of the regular
		model, the resulting values do not include priors
		"""
		if self.unbinned:
			raise ValueError('Saturated LLH does not work in unbinned mode')

		llh = 0
		dof = 0
		for prop in self.data:
			log_lambda = numpy.log(self.data[prop])
			log_lambda[numpy.isinf(log_lambda)] = 0
			llh += numpy.sum(self.data[prop]*log_lambda - self.data[prop])
			dof += numpy.sum(self.data[prop] != 0)
		return (llh, dof+1)
	def fit(self, minimizer_params=dict(), **fixedparams):
		"""
		Return dictionary of best-fit values for all parameters using
		scipy's L-BFGS-B fitter. Keyword arguments are optional and
		are interpreted as assigning fixed values to the named
		parameters in the fit. All parameters not passed as keyword
		arguments will be free to float.
		
		If a parameter is fixed to an iterable value, it will be treated
		as a discrete parameter and minimized on a grid.
		"""
		from collections import Iterable
		import itertools
		
		freeparams = list(self.components.keys())
		discrete_params = []
		bounds = []
		seeds = []
		for param in fixedparams:
			if isinstance(fixedparams[param], Iterable) and not isinstance(fixedparams[param], str):
				discrete_params.append(param)
			if param in freeparams:
				freeparams.remove(param)
		for p in freeparams:
			bounds.append(self.bounds()[p])
			if hasattr(self.components[p], 'seed'):
				seeds.append(self.components[p].seed)
			else:
				seeds.append(0)

		def minllh(x):
			pdict = dict(zip(freeparams, x))
			pdict.update(fixedparams)
			return -self.llh(**pdict)
		
		if len(discrete_params) == 0:
			bestfit = scipy.optimize.fmin_l_bfgs_b(minllh, seeds, bounds=bounds, approx_grad=True, **minimizer_params)
			#print fixedparams['ice_model'], bestfit[1], bestfit[0]
			fixedparams.update(dict(zip(freeparams, bestfit[0])))
			return fixedparams
		else:
			bestllh = numpy.inf
			bestparams = dict(fixedparams)
			for points in itertools.product(*tuple(fixedparams[k] for k in discrete_params)):
				for k, p in zip(discrete_params, points):
					fixedparams[k] = p
				bestfit = scipy.optimize.fmin_l_bfgs_b(minllh, seeds, bounds=bounds, approx_grad=True)
				# print fixedparams['ice_model'], bestfit[1], bestllh, bestfit[0]
				if bestfit[1] < bestllh:
					bestparams = dict(fixedparams)
					bestparams.update(dict(zip(freeparams, bestfit[0])))
					bestllh = bestfit[1]
			# print '-->', bestparams['ice_model'], bestllh
			return bestparams
	def profile1d(self, param, values, minimizer_params=dict(), **fixedparams):
		"""
		Return a named numpy array with best-fit nuisance values and
		likelihood values for the values of param passed. Additional
		keywords are interpreted as fixed values of nuisance parameters.
		"""
		llhpoints = []
		for val in values:
			params = {param: val}
			params.update(fixedparams)
			if len(params) - len(self.components) == 0:
				fit = params
			else:
				fit = self.fit(minimizer_params=minimizer_params, **params)
			mlh = self.llh(**fit)
			llhpoints.append(tuple(list(fit.values()) + [mlh]))
		dkeys = list(fit.keys()) + ['LLH']
		dtypes = [float]*(len(list(fit.keys())) + 1)
		for i in range(len(dtypes)):
			if isinstance(llhpoints[-1][i], str):
				dtypes[i] = '|S32'
		return numpy.array(llhpoints, dtype=list(zip(dkeys, dtypes)))
	def profile2d(self, param1, values1, param2, values2, **fixedparams):
		"""
		Returns a 2D scan (see profile1d) of the likelihood space at
		all combinations of values passed.
		"""
		llhpoints = []
		for val1 in values1:
			for val2 in values2:
				params = {param1: val1, param2: val2}
				params.update(fixedparams)
				if len(params) - len(self.components) == 0:
					fit = params
				else:
					fit = self.fit(**params)
				mlh = self.llh(**fit)
				llhpoints.append(list(fit.values()) + [mlh])
		return numpy.asarray(llhpoints).view(dtype=list(zip(list(fit.keys()) + ['LLH'], [float]*(len(list(fit.keys())) + 1))))

def _pseudo_llh(components, poisson, **nominal):
	allh = LLHEval(None)
	allh.components = components
	for k in allh.components.keys():
		if not k in nominal:
			try:
				nominal[k] = getattr(components[k], 'seed')
			except AttributeError:
				nominal[k] = 1
	expectations = allh.expectations(**nominal)
	if poisson:
		pseudodata = dict()
		for tag in expectations:
			pseudodata[tag] = numpy.random.poisson(expectations[tag])
	else:
		pseudodata = expectations
	allh.data = pseudodata
	return allh

def pseudodata_llh(components, **hypothesis):
	"""
	Create a likelihood on a realization of the hypothesis
	"""
	
	return _pseudo_llh(components, True, **hypothesis)

def asimov_llh(components, **nominal):
	"""
	Create a likelihood on the average over all realizations of the hypothesis
	*nominal*, also known as the Asimov_ dataset. Likelihood ratios obtained
	this way are asymptotically equivalent to the median likelihood ratio over
	repeated realizations of the hypothesis.
	
	_Asimov: http://dx.doi.org/10.1140/epjc/s10052-011-1554-0
	"""
	
	return _pseudo_llh(components, False, **nominal)



