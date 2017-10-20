
import numpy
import os
import cPickle as pickle
from . import effective_areas, diffuse, pointsource, angular_resolution, grb, surface_veto, multillh, plotting
from .util import data_dir, center

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
	
	seleff_kwargs = dict()
	if opts.energy_threshold is not None:
		seleff_kwargs['energy_threshold'] = opts.energy_threshold
	if opts.psf_class is not None:
		# assume that all PSF classes have equal effective area
		seleff_kwargs['scale'] = 1./opts.psf_class[1]
	if opts.efficiency_scale != 1:
		if seleff_kwargs.get('scale', 1) != 1:
			raise ValueError('psf_class and efficiency_scale are mutually exclusive')
		seleff_kwargs['scale'] = opts.efficiency_scale
	seleff = effective_areas.get_muon_selection_efficiency(opts.geometry, opts.spacing, **seleff_kwargs)
	
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

	if hasattr(opts, 'psf'):
		kwargs['psf'] = getattr(opts, 'psf')
	else:
		kwargs['psf'] = angular_resolution.get_angular_resolution(opts.geometry, opts.spacing, opts.angular_resolution_scale, opts.psf_class)
	
	neutrino_aeff = effective_areas.create_throughgoing_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing),
	    selection_efficiency=selection_efficiency,
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    **kwargs)
	
	bundle_aeff = effective_areas.create_bundle_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing),
	    selection_efficiency=selection_efficiency,
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    **kwargs)

	# cache[key] = aeff
	# pickle.dump(cache, open(cache_file, 'w'), 2)
	return neutrino_aeff, bundle_aeff

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
		# arguments for creating effective areas
		self._recipes = dict()
		# cached effective areas
		self._aeffs = dict()
	
	def set_kwargs(self, **kwargs):
		self._aeffs.clear()
		for opts, kw in self._recipes.values():
			kw.update(**kwargs)
	
	def add(self, name, opts, **kwargs):
		self._recipes[name] = (opts, kwargs)
		if name in self._aeffs:
			del self._aeffs[name]
	
	def _create(self, opts, **kwargs):
		aeffs = {}
		if opts.geometry == 'ARA':
			for k in 'psi_bins', 'cos_theta':
				if k in kwargs:
					kwargs[k] = numpy.asarray(kwargs[k])
				elif hasattr(opts, k):
					kwargs[k] = numpy.asarray(getattr(opts, k))
			kwargs['nstations'] = opts.nstations
			kwargs['depth'] = opts.depth
			aeffs['events'] = (effective_areas.create_ara_aeff(**kwargs), None)
		else:
			nu, mu = create_aeff(opts,**kwargs)
			aeffs['shadowed_tracks'] = (nu[0], mu[0])
			aeffs['unshadowed_tracks'] = (nu[1], mu[1])
			if opts.cascade_energy_threshold is not None:
                                print 'making cascades'
				aeffs['cascades']=(create_cascade_aeff(opts,**kwargs), None)
		return aeffs
	
	def __call__(self, name):
		if not name in self._recipes:
			raise KeyError("Unknown configuration '{}'".format(name))
		if not name in self._aeffs:
			opts, kwargs = self._recipes[name]
			self._aeffs[name] = self._create(opts, **kwargs)
		return self._aeffs[name]
	
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
	defaults = dict(geometry='Sunflower', spacing=240, veto_area=75., angular_resolution_scale=1., efficiency_scale=1.,
	                cascade_energy_threshold=None, veto_threshold=1e5, energy_threshold=0, no_cuts=False,
	                psf_class=(0,1),
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

set_kwargs = aeff_factory.get().set_kwargs

default_configs = {
	'IceCube' : dict(geometry='IceCube', spacing=125, cascade_energy_threshold=6e4, veto_area=1., veto_threshold=1e5),
	'Sunflower_240' : dict(geometry='Sunflower', spacing=240, cascade_energy_threshold=2e5, veto_area=75., veto_threshold=1e5),
	'ARA_37' : dict(geometry='ARA', nstations=37, depth=200),
	'ARA_200' : dict(geometry='ARA', nstations=200, depth=200),
	'ARA_300' : dict(geometry='ARA', nstations=300, depth=200),
	'IceCube_NoCasc' : dict(geometry='IceCube', spacing=125, veto_area=1., veto_threshold=1e5),
	'Sunflower_240_NoCasc' : dict(geometry='Sunflower', spacing=240, veto_area=75., veto_threshold=1e5),
	'KM3NeT' : dict(geometry='IceCube', spacing=125, veto_area=0., veto_threshold=None, angular_resolution_scale=0.2),
}
for k, config in default_configs.items():
	add_configuration(k, make_options(**config), cos_theta=numpy.linspace(-1, 1, 21))
