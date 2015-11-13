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
sys.path.append('/Users/jakob/Documents/IceCube/projects/2015/gen2_analysis')


import effective_areas

import diffuse
import pointsource
import angular_resolution
import grb
import surface_veto
import multillh
import plotting
from util import *

import cPickle as pickle

import warnings
warnings.filterwarnings("ignore")

def create_aeff(opts, **kwargs):
	
	cache_file = os.path.join(data_dir, 'cache', 'throughgoing_aeff')
	
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
	
	return effective_areas.create_throughgoing_aeff(
	    energy_resolution=effective_areas.get_energy_resolution(opts.geometry, opts.spacing),
	    selection_efficiency=selection_efficiency,
	    surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
	    energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
	    **kwargs)

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
	
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 41))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	psf = angular_resolution.get_angular_resolution(opts.geometry, opts.spacing)
	psi_bins = numpy.radians((numpy.linspace(0, 5, 91)))
	
	dp = []
	sindec = numpy.linspace(-1, 1, 21)
	for zi in xrange(20):
		ps = pointsource.SteadyPointSource(aeff, opts.livetime, zenith_bin=zi,
		    point_spread_function=psf, psi_bins=psi_bins)
		bkg = atmo.point_source_background(zenith_index=zi, psi_bins=psi_bins)
		astro_bkg = astro.point_source_background(zenith_index=zi, psi_bins=psi_bins)
		
		diffuse = dict(atmo=bkg, astro=astro_bkg, gamma=gamma)
		fixed = dict(atmo=1, gamma=gamma.seed, astro=2)
		atmo.seed = 1
		dp.append(1e-12*pointsource.discovery_potential(ps, diffuse, **fixed))
	dp = numpy.array(dp)[::-1]
	
	volume = survey_volume(sindec, dp)
	
	print_result(volume)

elif opts.figure_of_merit == 'differential_discovery_potential':
	
	if opts.outfile is None:
		parser.error("You must supply an output file name")
	
	aeff = create_aeff(opts, cos_theta=numpy.linspace(-1, 1, 21))
	energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90)
	atmo = diffuse.AtmosphericNu.conventional(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	prompt = diffuse.AtmosphericNu.prompt(aeff, opts.livetime, hard_veto_threshold=energy_threshold)
	astro = diffuse.DiffuseAstro(aeff, opts.livetime)
	astro.seed = 2
	gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
	psf = angular_resolution.get_angular_resolution(opts.geometry, opts.spacing)
	psi_bins = numpy.radians((numpy.linspace(0, 5, 91)))
	
	values = dict()
	
	sindec = numpy.linspace(-1, 1, 21)[::-1]
	for zi in xrange(0, 20, 1):	
		ps = pointsource.SteadyPointSource(aeff, opts.livetime, zenith_bin=zi,
		    point_spread_function=psf, psi_bins=psi_bins)
		atmo_bkg = atmo.point_source_background(zenith_index=zi, psi_bins=psi_bins)
		astro_bkg = astro.point_source_background(zenith_index=zi, psi_bins=psi_bins)
		if astro.expectations(gamma=gamma.seed)['tracks'][:,zi].sum() > 0:
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
	psf = angular_resolution.get_angular_resolution(opts.geometry, opts.spacing)
	psi_bins = numpy.radians((numpy.linspace(0, 5, 91)))

	z = 2*numpy.ones(opts.livetime*170*2)
	t90 = numpy.ones(z.size)*45.1
	Eiso = 10**(53.5)*numpy.ones(z.size)

	pop = grb.GRBPopulation(aeff, z, Eiso, psf, psi_bins)
	bkg = atmo.point_source_background(psi_bins, slice(None), livetime=t90.sum())
	scale = pointsource.discovery_potential(pop, dict(atmo=bkg), atmo=1.)
	
	exes = get_expectations(multillh.asimov_llh(dict(atmo=bkg, grb=pop)), grb=scale)
	nb = exes['atmo']['tracks'].sum()
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
	ns = gzk.expectations()['tracks'].sum(axis=1)[pev:].sum()
	nb = astro.expectations(gamma=-2.3)['tracks'].sum(axis=1)[pev:].sum()
	baseline = 5*numpy.sqrt(nb)/ns
	
	components = dict(atmo=atmo, astro=astro, gamma=gamma)
	scale = pointsource.discovery_potential(gzk, components,
	    baseline=baseline, tolerance=1e-4, gamma=-2.3)
	
	components['gzk'] = gzk
	llh = multillh.asimov_llh(components)
	exes = get_expectations(llh, gzk=scale)
	nb = exes['atmo']['tracks'][pev:,:].sum() + exes['astro']['tracks'][pev:,:].sum()
	ns = exes['gzk']['tracks'][pev:,:].sum()

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
	
	llh = multillh.asimov_llh(dict(conv=atmo, prompt=prompt, astro=astro, gamma=gamma), astro=2, gamma=-2.3)
	
	exes = get_expectations(llh)
	nb = exes['conv']['tracks'].sum() + exes['prompt']['tracks'].sum()
	ns = exes['astro']['tracks'].sum()

	from scipy import stats, optimize
	def find_limits(llh, critical_ts = 1**2, plotit=False):
		nom = {k:v.seed for k,v in llh.components.items()}
		base = llh.llh(**nom)
		def ts_diff(gamma):
			return -2*(llh.llh(**llh.fit(gamma=gamma)) - base) - critical_ts
		g0 = -2.3
		try:
			lo = optimize.bisect(ts_diff, g0, g0+0.6, xtol=1e-4, rtol=1e-4)
		except ValueError:
			lo = g0+1
		try:
			hi = optimize.bisect(ts_diff, g0-0.8, g0, xtol=1e-4, rtol=1e-4)
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
