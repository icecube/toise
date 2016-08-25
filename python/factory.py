
import numpy
import os
import cPickle as pickle
from icecube.gen2_analysis import effective_areas, diffuse, pointsource, angular_resolution, grb, surface_veto, multillh, plotting
from icecube.gen2_analysis.util import data_dir, center

def make_key(opts, kwargs):
	key = dict(opts.__dict__)
	key.update(kwargs)
	for k, v in kwargs.items():
		if isinstance(v, numpy.ndarray):
			key[k] = (v[0], v[-1], len(v))
		else:
			key[k] = v
	
	return tuple(key.items())

def create_aeff(opts, **kwargs):

	cache_file = os.path.join(data_dir, 'cache', 'throughgoing_aeff')
	# try:
	# 	cache = pickle.load(open(cache_file))
	# except IOError:
	# 	cache = dict()
	# key = make_key(opts, kwargs)
	# if key in cache:
	# 	return cache[key]

	if opts.veto_area > 0:
		kwargs['veto_coverage'] = surface_veto.GeometricVetoCoverage(opts.geometry, opts.spacing, opts.veto_area)
	
	if opts.energy_threshold is not None:
		seleff = effective_areas.get_muon_selection_efficiency(opts.geometry, opts.spacing, opts.energy_threshold)
	else:
		seleff = effective_areas.get_muon_selection_efficiency(opts.geometry, opts.spacing)
	if opts.no_cuts:
		selection_efficiency = lambda emu, cos_theta: seleff(emu, cos_theta=0)
		# selection_efficiency = lambda emu, cos_theta: numpy.ones(emu.shape)
		# selection_efficiency = effective_areas.get_muon_selection_efficiency("IceCube", None)
	else:
		selection_efficiency = seleff
	
	for k in 'psi_bins', 'cos_theta':
		if k in kwargs:
			kwargs[k] = numpy.asarray(kwargs[k])
		elif hasattr(opts, k):
			kwargs[k] = numpy.asarray(getattr(opts, k))
	
	aeff = effective_areas.create_throughgoing_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing),
	    selection_efficiency=selection_efficiency,
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    psf=angular_resolution.get_angular_resolution(opts.geometry, opts.spacing, opts.angular_resolution_scale),
	    **kwargs)

	# cache[key] = aeff
	# pickle.dump(cache, open(cache_file, 'w'), 2)
	return aeff

def create_cascade_aeff(opts, **kwargs):

	# cache_file = os.path.join(data_dir, 'cache', 'cascade_aeff')
	# try:
	# 	cache = pickle.load(open(cache_file))
	# except IOError:
	# 	cache = dict()
	# key = make_key(opts, kwargs)
	# if key in cache:
	# 	return cache[key]

	for k in 'psi_bins', 'cos_theta':
		if k in kwargs:
			kwargs[k] = numpy.asarray(kwargs[k])
		elif hasattr(opts, k):
			kwargs[k] = numpy.asarray(getattr(opts, k))

	aeff = effective_areas.create_cascade_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing, channel='cascade'),
	    selection_efficiency=effective_areas.HESEishSelectionEfficiency(opts.geometry, opts.spacing, opts.cascade_energy_threshold),
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    # energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    psf=angular_resolution.get_angular_resolution(opts.geometry, opts.spacing, opts.angular_resolution_scale, channel='cascade'),
	    **kwargs)

	# cache[key] = aeff
	# pickle.dump(cache, open(cache_file, 'w'), 2)
	return aeff

class aeff_factory(object):
	"""
	Create effective areas, lazily
	"""
	def __init__(self):
		self._aeffs = dict()
		self._names = dict()
	
	def add(self, name, opts, **kwargs):
		opt_dict = dict(opts.__dict__)
		opt_dict.update(kwargs)
		key = hash(tuple(opt_dict.items()))
		
		self._names[name] = key
		self._aeffs[key] = (opts, kwargs)
	
		
	def _create(self, opts, **kwargs):
		aeffs = dict(tracks=create_aeff(opts,**kwargs))
		if opts.cascade_energy_threshold is not None:
			aeffs['cascades']=create_cascade_aeff(opts,**kwargs)
		return aeffs
	
	def __call__(self, name):
		if not name in self._names:
			raise KeyError("Unknown configuration '{}'".format(name))
		key = self._names[name]
		if isinstance(self._aeffs[key], tuple):
			opts, kwargs = self._aeffs[key]
			self._aeffs[key] = self._create(opts, **kwargs)
		return self._aeffs[key]
	
	@classmethod
	def get(cls):
		if not hasattr(cls, 'instance'):
			cls.instance = cls()
		return cls.instance

class component_bundle(object):
	def __init__(self, livetimes, component_factory, **kwargs):
		self.components = dict()
		self.livetimes = dict(livetimes)
		self.detectors = dict()
		for detector, livetime in livetimes.items():
			for channel, aeff in aeff_factory.get()(detector).items():
				key = detector + '_' + channel
				self.components[key] = component_factory(aeff)
				self.detectors[key] = detector
	def get_component(self, key, livetimes=None):
		if livetimes is None:
			livetimes = self.livetimes
		return multillh.Combination({k: (self.components[k][key], livetimes[self.detectors[k]]) for k in self.components})
	def get_components(self, livetimes=None):
			keys = set()
			for v in self.components.values():
				keys.update(v.keys())
			return {key : self.get_component(key, livetimes) for key in keys}

def make_options(**kwargs):
	import argparse
	defaults = dict(geometry='Sunflower', spacing=240, veto_area=75., angular_resolution_scale=1.,
	                cascade_energy_threshold=None, veto_threshold=1e5, energy_threshold=0, no_cuts=False,
	                livetime=1.)
	# icecube has no veto...yet
	if kwargs.get('geometry') == 'IceCube':
		defaults['veto_area'] = 0.
		defaults['veto_threshold'] = numpy.inf
	defaults.update(kwargs)
	return argparse.Namespace(**defaults)

def add_configuration(name, opts, **kwargs):
	"""
	Add an effective area calculation to the cache
	"""
	aeff_factory.get().add(name, opts, **kwargs)