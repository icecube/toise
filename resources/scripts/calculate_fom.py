#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT icerec/trunk

import logging
logging.basicConfig(level='WARN')

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--geometry", choices=('IceCube', 'Sunflower', 'EdgeWeighted'), default='Sunflower')
parser.add_argument("--spacing", type=int, default=240)
parser.add_argument("--veto-area", type=float, default=25)
parser.add_argument("--veto-threshold", type=float, default=1e4)
parser.add_argument("--no-cuts", default=False, action="store_true")
parser.add_argument("--livetime", type=float, default=10.)
parser.add_argument("--energy-threshold", type=float, default=None)
parser.add_argument("--cascade-energy-threshold", type=float, default=None, help='Energy threshold for cascade detection. If None, no cascades.')
parser.add_argument("--sin-dec", type=float, default=None)
parser.add_argument("--livetimes", type=float, default=None, nargs=3)
parser.add_argument("--angular-resolution-scale", type=float, default=1.)
parser.add_argument("-o", "--outfile", default=None)

parser.add_argument("figure_of_merit")

def get_label(opts):
	name = {'survey_volume' : 'Survey volume',
	        'grb' : 'GRB discovery potential',
	        'gzk' : 'GZK discovery potential',
	        'galactic_diffuse' : 'Significance of galactic diffuse emission',
	        'diffuse_index'    : 'Astrophysical spectral index resolution',
	       }[opts.figure_of_merit]
	if opts.energy_threshold is not None:
		name += ' (above %s)' % plotting.format_energy('%d', opts.energy_threshold)
	return name

opts = parser.parse_args()
if opts.geometry == 'IceCube':
	opts.spacing = 125.

import sys, os
import numpy


from icecube.gen2_analysis import effective_areas, diffuse, pointsource, angular_resolution, grb, surface_veto, multillh, plotting
from icecube.gen2_analysis.util import data_dir, center

import cPickle as pickle
from clint.textui import progress

import warnings
warnings.filterwarnings("ignore")

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
	try:
		cache = pickle.load(open(cache_file))
	except IOError:
		cache = dict()
	key = make_key(opts, kwargs)
	if key in cache:
		return cache[key]

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

	aeff = effective_areas.create_throughgoing_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing),
	    selection_efficiency=selection_efficiency,
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    psf=angular_resolution.get_angular_resolution(opts.geometry, opts.spacing, opts.angular_resolution_scale),
	    psi_bins=numpy.sqrt(numpy.linspace(0, numpy.radians(2)**2, 100)),
	    **kwargs)

	cache[key] = aeff
	pickle.dump(cache, open(cache_file, 'w'), 2)
	return aeff

def create_cascade_aeff(opts, **kwargs):

	cache_file = os.path.join(data_dir, 'cache', 'cascade_aeff')
	try:
		cache = pickle.load(open(cache_file))
	except IOError:
		cache = dict()
	key = make_key(opts, kwargs)
	if key in cache:
		return cache[key]

	aeff = effective_areas.create_throughgoing_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing, channel='cascade'),
	    selection_efficiency=effective_areas.HESEishSelectionEfficiency(opts.geometry, opts.spacing, opts.cascade_energy_threshold),
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    # energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    psf=angular_resolution.get_angular_resolution(opts.geometry, opts.spacing, opts.angular_resolution_scale, channel='cascade'),
	    psi_bins=numpy.sqrt(numpy.linspace(0, numpy.radians(20)**2, 2)),
	    **kwargs)

	cache[key] = aeff
	pickle.dump(cache, open(cache_file, 'w'), 2)
	return aeff

def intflux(e, gamma=-2):
	return (e**(1+gamma))/(1+gamma)
def survey_distance(phi, L0=1e45):
	"""
	:param phi: E^-2 flux sensitivity in each declination band, in TeV/cm^2 s
	:param L0: standard candle luminosity, in erg/s
	"""
	phi = phi*(intflux(1e5) - intflux(1e-1)) # integrate from 100 GeV to 100 PeV
	phi *= (1e3/grb.units.GeV) # TeV / cm^2 s -> erg / cm^2 s
	dl = numpy.sqrt(L0/(4*numpy.pi*phi))
	dl *= grb.units.cm/1e3 # cm -> Gpc
	return numpy.where(numpy.isfinite(phi), dl, 0)
def survey_volume(sindec, phi, L0=1e45):
	"""
	:param sindec: sin(declination)
	:param phi: E^-2 flux sensitivity in each declination band, in TeV/cm^2 s
	:param L0: standard candle luminosity, in erg/s
	:returns: the volume in Gpc^3 in which standard candles of luminosity L0 would be detectable
	"""
	dl = survey_distance(phi, L0)
	return ((sindec.max()-sindec.min())*2*numpy.pi/3.)*((dl**3).mean())

def print_result(value, **kwargs):
	mapping = dict(opts.__dict__)
	mapping['veto_threshold'] /= 1e3
	mapping['name'] = get_label(opts)
	mapping['value'] = value
	if opts.veto_area > 0:
		mapping['veto_label'] = "veto: %(veto_area)2.0fkm^2/%(veto_threshold)3.0f TeV" % mapping
	else:
		mapping['veto_label'] = "(no veto)"
	line = '%(geometry)12s | %(spacing)dm | %(veto_label)20s | %(name)60s | %(value).2f' % mapping
	if len(kwargs) > 0:
		line += ' (%s)' % (', '.join(['%s=%.1f' % (k, v) for k, v in kwargs.items()]))
	print line

def get_expectations(llh, **nominal):
	exes = dict()
	for k, comp in llh.components.items():
		if k in nominal:
			continue
		if hasattr(comp, 'seed'):
			nominal[k] = comp.seed
		else:
			nominal[k] = 1
	for k, comp in llh.components.items():
		if hasattr(comp, 'expectations'):
			if callable(comp.expectations):
				ex = comp.expectations(**nominal)
			else:
				ex = comp.expectations
			exes[k] = {klass: nominal[k]*values for klass, values in ex.items()}
	return exes

if opts.figure_of_merit == 'survey_volume':
	
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 21))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	
	dp = []
	sindec = numpy.linspace(-1, 1, 21)
	for zi in xrange(20):
		ps = pointsource.SteadyPointSource(aeff, opts.livetime, zenith_bin=zi)
		bkg = atmo.point_source_background(zenith_index=zi)
		astro_bkg = astro.point_source_background(zenith_index=zi)
		
		diffuse = dict(atmo=bkg, astro=astro_bkg, gamma=gamma)
		fixed = dict(atmo=1, gamma=gamma.seed, astro=2)
		atmo.seed = 1
		dp.append(1e-12*pointsource.discovery_potential(ps, diffuse, **fixed))
		s = ps.expectations(**fixed)['tracks'].sum(axis=0)
		b = bkg.expectations['tracks'].sum(axis=0)
		if (~numpy.isnan(s/b)).any():
			pass
			# print s.cumsum()/s.sum()
			# print s/b
		# bkgs.append(bkg.expectations['tracks'][:,:20].sum())
	dp = numpy.array(dp)[::-1]
	
	volume = survey_volume(sindec, dp)
	
	print_result(volume)

elif opts.figure_of_merit == 'ps_time_evolution':
	
	livetime = opts.livetimes[0]
	cos_theta=numpy.linspace(-1, 1, 21)
	aeff = create_aeff(opts, cos_theta=cos_theta)
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, livetime, hard_veto_threshold=energy_threshold)
	prompt = diffuse.AtmosphericNu.prompt(aeff, livetime, hard_veto_threshold=energy_threshold)
	astro = diffuse.DiffuseAstro(aeff, livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	
	zi = cos_theta.searchsorted(-opts.sin_dec)-1
	
	ps = pointsource.SteadyPointSource(aeff, livetime, zenith_bin=zi)
	bkg = atmo.point_source_background(zenith_index=zi)
	astro_bkg = astro.point_source_background(zenith_index=zi)
		
	diffuse = dict(atmo=bkg, astro=astro_bkg, gamma=gamma)
	fixed = dict(atmo=1, gamma=gamma.seed, astro=2)
	atmo.seed = 1
	
	def scale_livetime(component, livetime):
		if hasattr(component, 'scale_livetime'):
			return component.scale_livetime(livetime)
		else:
			return component
	
	livetimes = numpy.linspace(*opts.livetimes)
	dps = numpy.zeros(livetimes.size)
	for i in progress.bar(range(livetimes.size)):
		lt = livetimes[i]
		dps[i] = 1e-12*pointsource.discovery_potential(scale_livetime(ps, lt), {k: scale_livetime(v, lt) for k,v in diffuse.items()}, **fixed)
	
	if opts.outfile is not None:
		numpy.savez(opts.outfile, livetime=livetimes, discovery_potential=dps,
		    sin_dec=opts.sin_dec)
	else:	
		import pylab
		pylab.plot(livetimes, dps)
		pylab.show()

elif opts.figure_of_merit == 'differential_discovery_potential':
	
	if opts.outfile is None:
		parser.error("You must supply an output file name")
	
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 20))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	
	values = dict()
	
	sindec = center(numpy.linspace(-1, 1, 20)[::-1])
	for zi in xrange(0, 19, 1):	
		ps = pointsource.SteadyPointSource(aeff, opts.livetime, zenith_bin=zi)
		atmo_bkg = atmo.point_source_background(zenith_index=zi)
		astro_bkg = astro.point_source_background(zenith_index=zi)
		if astro.expectations(gamma=gamma.seed)['tracks'][zi,:].sum() > 0:
			print astro_bkg.expectations(gamma=gamma.seed)['tracks'].sum()
			assert astro_bkg.expectations(gamma=gamma.seed)['tracks'].sum() > 0
		dps = []
		nses = []
		energies = []
		for ecenter, chunk in ps.differential_chunks(decades=1):
			energies.append(ecenter)
			diffuse = dict(atmo=atmo_bkg, astro=astro_bkg, gamma=gamma)
			fixed = dict(atmo=1, gamma=gamma.seed, astro=2)
			actual = pointsource.discovery_potential(chunk, diffuse, **fixed)
			components = dict(diffuse)
			components['ps'] = chunk
			allh = multillh.asimov_llh(components)
			total = pointsource.nevents(allh, ps=actual, **fixed)
			nb = pointsource.nevents(allh, ps=0, **fixed)
			ns = total-nb
			dps.append(1e-12*actual)
			nses.append(ns)
			
		values[sindec[zi]] = (energies, dps)
	with open(opts.outfile, 'w') as f:
		pickle.dump(values, f, 2)

elif opts.figure_of_merit == 'grb':
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 21))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	
	z = 2*numpy.ones(opts.livetime*170*2)
	t90 = numpy.ones(z.size)*45.1
	Eiso = 10**(53.5)*numpy.ones(z.size)

	pop = grb.GRBPopulation(aeff, z, Eiso)
	atmo_bkg = atmo.point_source_background(zenith_index=slice(None), livetime=t90.sum())
	astro_bkg = astro.point_source_background(zenith_index=slice(None), livetime=t90.sum())
	backgrounds = dict(atmo=atmo_bkg, astro=astro_bkg, gamma=gamma)
	fixed = dict(atmo=1, gamma=gamma.seed, astro=2)
	scale = pointsource.discovery_potential(pop, backgrounds, **fixed)
	
	components = dict(backgrounds)
	components['grb'] = pop
	
	exes = get_expectations(multillh.asimov_llh(components, grb=scale, **fixed), grb=scale, **fixed)
	nb = exes['atmo']['tracks'].sum() + exes['astro']['tracks'].sum()
	ns = exes['grb']['tracks'].sum()
	
	print_result(scale, nb=nb, ns=ns)

elif opts.figure_of_merit == 'gzk':
	
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 21))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	atmo.prior = lambda v: -(v-1)**2/(2*0.1**2)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt.min = 0.5
	prompt.max = 3
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	gzk = diffuse.AhlersGZK(aeff, opts.livetime)
	
	pev = numpy.where(aeff.bin_edges[2][1:] > 1e6)[0][0]
	ns = gzk.expectations()['tracks'].sum(axis=0)[pev:].sum()
	nb = astro.expectations(gamma=-2.3)['tracks'].sum(axis=0)[pev:].sum()
	baseline = 5*numpy.sqrt(nb)/ns
	
	components = dict(atmo=atmo, astro=astro, gamma=gamma)
	
	scale = pointsource.discovery_potential(gzk, components,
	    baseline=baseline, tolerance=1e-4, gamma=-2.3)
	
	components['gzk'] = gzk
	llh = multillh.asimov_llh(components)
	exes = get_expectations(llh, gzk=scale)
	nb = exes['atmo']['tracks'][:,pev:].sum() + exes['astro']['tracks'][:,pev:].sum()
	ns = exes['gzk']['tracks'][:,pev:].sum()

	print_result(scale, nb=nb, ns=ns)

elif opts.figure_of_merit == 'differential_diffuse':
	
	if opts.outfile is None:
		parser.error("You must supply an output file name")
	
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 21))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	atmo.prior = lambda v: -(v-1)**2/(2*0.1**2)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt.min = 0.5
	prompt.max = 3
	prompt.seed = 1
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 1
	gamma = multillh.NuisanceParam(-2, 0.5, min=-2.7, max=-1.7)
	
	energies = []
	dps = []
	i = 0
	for ecenter, chunk in astro.differential_chunks(decades=1):
		energies.append(ecenter)
		dps.append(1e-8*pointsource.discovery_potential(chunk, dict(atmo=atmo, prompt=prompt, gamma=gamma), atmo=1, prompt=1, gamma=-2))
	
	numpy.savetxt(opts.outfile, numpy.vstack((energies, dps)).T, header='# energy\tdiscovery flux [GeV cm-2 sr^-1 s^-1]')

elif opts.figure_of_merit == 'diffuse_index':
	
	cos_theta = numpy.linspace(-1, 1, 20)
	aeffs = dict(tracks=create_aeff(opts,cos_theta=cos_theta))
	if opts.cascade_energy_threshold is not None:
		aeffs['cascades']=create_cascade_aeff(opts,cos_theta=cos_theta)
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	components = dict()
	two_component = False and (opts.energy_threshold is not None)

	for k, aeff in aeffs.items():
		atmo = diffuse.AtmosphericNu.conventional(aeff, 1, hard_veto_threshold=energy_threshold)
		atmo.prior = lambda v: -(v-1)**2/(2*0.1**2)
		prompt = diffuse.AtmosphericNu.prompt(aeff, 1, hard_veto_threshold=energy_threshold)
		prompt.min = 0.5
		prompt.max = 3.
		if two_component:
			astro_lo = diffuse.DiffuseAstro(aeff.restrict_energy_range(0, opts.energy_threshold), 1, gamma_name='gamma_lo')
			astro_lo.seed = 2.
			astro = diffuse.DiffuseAstro(aeff.restrict_energy_range(opts.energy_threshold, numpy.inf), 1)
			astro.seed = 2.
			components[k] = dict(atmo=atmo, prompt=prompt, astro_lo=astro_lo, astro=astro)
		else:
			astro = diffuse.DiffuseAstro(aeff, 1)
			astro.seed = 2.
			components[k] = dict(atmo=atmo, prompt=prompt, astro=astro)
	def combine(key):
		return multillh.Combination({k: (components[k][key], opts.livetime) for k in components})

	comps = dict(conv=combine('atmo'), prompt=combine('prompt'), astro=combine('astro'), gamma=multillh.NuisanceParam(-2.3))
	if two_component:
		comps['astro_lo'] = combine('astro_lo')
		comps['gamma_lo'] = multillh.NuisanceParam(-2.3)
	llh = multillh.asimov_llh(comps)
	
	exes = get_expectations(llh)
	get_events = lambda d: sum(v.sum() for v in d.values())
	nb = get_events(exes['conv']) + get_events(exes['prompt'])
	ns = get_events(exes['astro'])

	from scipy import stats, optimize
	def find_limits(llh, critical_ts = 1**2, plotit=False):
		nom = {k:v.seed for k,v in llh.components.items()}
		print nom
		base = llh.llh(**nom)
		def ts_diff(gamma):
			alt = llh.fit(gamma=gamma)
			ts = -2*(llh.llh(**alt) - base) - critical_ts
			print alt, ts
			return ts
		g0 = -2.3
		try:
			lo = optimize.bisect(ts_diff, g0, g0+0.6, xtol=5e-3, rtol=1e-4)
		except ValueError:
			lo = g0+1
		try:
			hi = optimize.bisect(ts_diff, g0-0.8, g0, xtol=5e-3, rtol=1e-4)
		except ValueError:
			hi = g0-1
		
		if plotit:
			import pylab
			g = numpy.linspace(g0-0.5, g0+0.5, 21)
			pylab.plot(g, [ts_diff(g_) for g_ in g])
			color = pylab.gca().lines[-1].get_color()
	
			print (lo-hi)/2.
			pylab.axvline(lo, color=color)
			pylab.axvline(hi, color=color)
			pylab.show()
		
		return lo, hi
	
	lo, hi = find_limits(llh, plotit=False)
	
	print_result(abs(hi-lo)/2., ns=ns, nb=nb)

elif opts.figure_of_merit == 'galactic_diffuse':
	
	aeff = create_aeff(opts, cos_theta=16)
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	# atmo.prior = lambda v: -(v-1)**2/(2*0.1**2)
	atmo.min = 0.1
	atmo.max = 10
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt.min = 0.1
	prompt.max = 10
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2.05
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	fermi = diffuse.FermiGalacticEmission(aeff, livetime=opts.livetime)
	
	components = dict(atmo=atmo, charm=prompt, astro=astro, gamma=gamma, galactic=fermi)
	def ts(flux_norm, **fixed):
		"""
		Test statistic of flux_norm against flux norm=0
		"""
		allh = multillh.asimov_llh(components, galactic=flux_norm)
		if len(fixed) == len(components)-1:
			hypo, null = dict(fixed), dict(fixed)
			hypo['galactic'] = flux_norm
			null['galactic'] = 0
		else:
			hypo = allh.fit(galactic=flux_norm, **fixed)
			null = allh.fit(galactic=0, **fixed)
		return -2*(allh.llh(**null)-allh.llh(**hypo))
	fit_ts = ts(1, gamma=-2.5, charm=1)
	
	exes = get_expectations(multillh.asimov_llh(components, galactic=1))
	nb = exes['atmo']['tracks'].sum() + exes['charm']['tracks'].sum() + exes['astro']['tracks'].sum()
	ns = exes['galactic']['tracks'].sum()
	
	print_result(numpy.sqrt(fit_ts), nb=nb, ns=ns)

else:
	parser.error("Unknown figure of merit '%s'" % (opts.figure_of_merit))

