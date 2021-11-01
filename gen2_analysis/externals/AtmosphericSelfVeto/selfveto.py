#!/usr/bin/env python

"""
Calculate the rate at which atmospheric neutrinos arrive at an underground
detector with no accompanying muons.

This script is nearly identical to the one distributed with http://arxiv.org/abs/1405.0525 .
"""

# Copyright (c) 2014, Jakob van Santen <jvansanten@icecube.wisc.edu>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy


def overburden(cos_theta, depth=1950, elevation=2400):
    """
    Overburden for a detector buried underneath a flat surface.

    :param cos_theta: cosine of zenith angle (in detector-centered coordinates)
    :param depth:     depth of detector (in meters below the surface)
    :param elevation: elevation of the surface above sea level

    :returns: an overburden [meters]
    """
    # curvature radius of the surface (meters)
    r = 6371315 + elevation
    # this is secrety a translation in polar coordinates
    return (numpy.sqrt(2*r*depth + (cos_theta*(r-depth))**2 - depth**2) - (r-depth)*cos_theta)


def minimum_muon_energy(distance, emin=1e3):
    """
    Minimum muon energy required to survive the given thickness of ice with at
    least emin TeV 50% of the time.

    :returns: minimum muon energy at the surface [GeV]
    """
    def polynomial(x, coefficients):
        return sum(c*x**i for i, c in enumerate(coefficients))
    coeffs = [[2.793, -0.476, 0.187],
              [2.069, -0.201, 0.023],
              [-2.689, 3.882]]
    a, b, c = (polynomial(numpy.log10(emin), c) for c in coeffs)
    return 10**polynomial(distance, (a, b/1e4, c/1e10))


def effective_costheta(costheta):
    """
    Effective local atmospheric density correction from [Chirkin]_.

    .. [Chirkin] D. Chirkin. Fluxes of atmospheric leptons at 600-GeV - 60-TeV. 2004. http://arxiv.org/abs/hep-ph/0407078
    """
    x = costheta
    p = [0.102573, -0.068287, 0.958633, 0.0407253, 0.817285]
    return numpy.sqrt((x**2 + p[0]**2 + p[1]*x**p[2] + p[3]*x**p[4])/(1 + p[0]**2 + p[1] + p[3]))


class fpe_context(object):
    """
    Temporarily modify floating-point exception handling
    """

    def __init__(self, **kwargs):
        self.new_kwargs = kwargs

    def __enter__(self):
        self.old_kwargs = numpy.seterr(**self.new_kwargs)

    def __exit__(self, *args):
        numpy.seterr(**self.old_kwargs)


elbert_params = {
    'elbert': {'a': 14.5,
               'p1': 0.757+1,
               'p2': 5.25,
               'p3': 1},
    'mu':    {'a': 49.41898933142626,
              'p1': 0.62585930096346309+1,
              'p2': 4.9382653076505525,
              'p3': 0.58038589096897897},
    'numu':   {'a': 79.918830537201231,
               'p1': 0.46284463423687988+1,
               'p2': 4.3799061061862314,
               'p3': 0.31657956163506323},
    'nue':    {'a': 0.548,
               'p1': 0.669+1,
               'p2': 8.05,
               'p3': 0.722},
    'charm':  {'a': 780.35285355003532/1e6,
               'p1': -0.39555243513109928+1,
               'p2': 7.3461490462825703,
               'p3': 0.76688386541155051}
}


def elbert_yield(emin, primary_energy, primary_mass, cos_theta, kind='mu', differential=False):
    """
    Evaluate the lepton yield of the given lepton family per shower for showers
    of a given type.

    :param emin:           lepton energy at which to evaluate
    :param primary_energy: total primary energy
    :param primary_mass:   primary atomic number
    :param cos_theta:      cosine of primary zenith angle
    :param kind:           family of parameterization (mu, numu, nue, or charm)
    :param differential:   if True, evaluate dN/dE at emin. Otherwise,
                           evaluate N(>emin).

    :returns: either a differential yield [1/GeV] or cumulative yield [number]
    """
    params = elbert_params[kind]
    a = params['a']
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']

    En = primary_energy/primary_mass
    x = emin/En

    if kind == 'charm':
        decay_prob = 1.
    else:
        decay_prob = 1./(En*effective_costheta(cos_theta))

    with fpe_context(all='ignore'):
        icdf = numpy.where(x >= 1, 0., a*primary_mass *
                           decay_prob*x**(-p1)*(1-x**p3)**p2)
        if differential:
            icdf *= (1./En)*numpy.where(x >= 1, 0.,
                                        (p1/x + p2*p3*x**(p3-1)/(1-x**p3)))

    return icdf


class ParticleType(object):
    PPlus = 14
    He4Nucleus = 402
    N14Nucleus = 1407
    Al27Nucleus = 2713
    Fe56Nucleus = 5626


def gaisser_flux(energy, ptype):
    """
    Evaluate the [Gaisser]_ H3a parameterization of the cosmic ray flux.

    :param energy: total energy of primary nucleus [GeV]
    :param ptype: particle type code 

    :returns: particle flux [particles/(GeV m^2 sr s)]

    .. [Gaisser] T. K. Gaisser. Spectrum of cosmic-ray nucleons, kaon production, and the atmospheric muon charge ratio. Astroparticle Physics, 35(12):801--806, 2012. ISSN 0927-6505. doi: 10.1016/j.astropartphys.2012.02.010.
    """
    if ptype < 100:
        z = 1
    else:
        z = ptype % 100

    codes = sorted([v for v in list(ParticleType.__dict__.values()) if isinstance(
        v, int)])
    idx = codes.index(ptype)

    # normalizations for each element
    norm = [
        [7860., 3550., 2200., 1430., 2120.],
        [20]*2 + [13.4]*3,
        [1.7]*2 + [1.14]*3,
    ]
    # spectral indices
    gamma = [
            [2.66, 2.58, 2.63, 2.67, 2.63],
            [2.4]*5,
            [2.4]*5
    ]
    # cutoff rigitity
    rigidity = [
        4e6, 30e6, 2e9
    ]

    return sum(n[idx]*energy**(-g[idx])*numpy.exp(-energy/(r*z)) for n, g, r in zip(norm, gamma, rigidity))


def logspace(start, stop, num):
    """
    A version of numpy.logspace that takes array arguments
    """
    # Add a trailing dimension to the inputs
    start = numpy.asarray(start)[..., None]
    stop = numpy.asarray(stop)[..., None]
    num = int(num)
    step = (stop-start)/float(num-1)
    y = (numpy.core.numeric.arange(0, num)*step + start).squeeze()
    return 10**y


def response_function(enu, emu, cos_theta, kind='numu'):
    """
    Evaluate the response function (contributions from each chunk of the cosmic
    ray flux) for leptons of the given energy and zenith angle

    :param enu:       lepton energy
    :param emu:       minimum muon energy needed for veto [GeV at surface]
    :param cos_theta: cosine of zenith angle
    :param kind:      lepton type for which to evaluate the passing rate:
                      "numu" for muon neutrinos from pion/kaon decay, "nue" for
                      electron neutrinos from kaon decay, or "charm" for either
                      flavor from charmed meson decay

    :returns: a tuple (response, muonyield, energy_per_nucleon)
    """
    # make everything an array
    enu, emu, cos_theta = list(map(numpy.asarray, (enu, emu, cos_theta)))
    shape = numpy.broadcast(enu, emu, cos_theta).shape
    # contributions to the differential neutrino flux from chunks of the
    # primary spectrum for each element
    contrib = numpy.zeros(shape+(5, 100))
    # mean integral muon yield from same chunks
    muyield = numpy.zeros(shape+(5, 100))
    energy_per_nucleon = logspace(numpy.log10(enu), numpy.log10(enu)+3, 101)
    ptypes = [getattr(ParticleType, pt) for pt in ('PPlus',
              'He4Nucleus', 'N14Nucleus', 'Al27Nucleus', 'Fe56Nucleus')]
    A = [[pt/100, 1][pt == ParticleType.PPlus] for pt in ptypes]
    for i, (ptype, a) in enumerate(zip(ptypes, A)):
        # primary energies that contribute to the neutrino flux at given energy
        penergy = a*energy_per_nucleon
        # width of energy bins
        de = numpy.diff(penergy)
        # center of energy bins
        pe = penergy[..., :-1] + de/2.
        # hobo-integrate the flux
        weights = gaisser_flux(pe, int(ptype))*de
        # neutrino yield from each chunk of the primary flux
        ey = elbert_yield(enu[..., None], pe, a,
                          cos_theta[..., None], kind=kind, differential=True)
        contrib[..., i, :] = ey*weights
        muyield[..., i, :] = elbert_yield(
            emu[..., None], pe, a, cos_theta[..., None], kind='mu', differential=False)

    return contrib, muyield, energy_per_nucleon[..., :-1] + numpy.diff(energy_per_nucleon)/2.


def get_bin_edges(energy_grid):
    half_width = 10**(numpy.diff(numpy.log10(energy_grid))[0]/2)
    return numpy.concatenate((energy_grid/half_width, [energy_grid[-1]*half_width]))


def get_bin_width(energy_grid):
    return numpy.diff(get_bin_edges(energy_grid))


def extract_yields(yield_record, types):
    return sum((yield_record[t] for t in types))


def interpolate_integral_yield(log_N, log_e, log_emin):
    interp = numpy.interp(log_emin, log_e, log_N)
    if numpy.isfinite(interp):
        return numpy.exp(interp)
    else:
        return 0.


def integral_yield(e_min, e_grid, diff_yield):
    edges = get_bin_edges(e_grid)
    de = numpy.diff(edges)
    intyield = de*diff_yield
    axis = 1
    index = [slice(None)]*diff_yield.ndim
    index[axis] = slice(None, None, -1)
    index = tuple(index)
    # cumulative sum from the right
    N = intyield[index].cumsum(axis=axis)[index]

    return numpy.apply_along_axis(interpolate_integral_yield, 1,
                                  numpy.log(N), numpy.log(edges[1:]), numpy.log(e_min))


def mceq_response_function(fname, emu, nu_types, prompt_muons=False):
    """
    Evaluate the response function (contributions from each chunk of the cosmic
    ray flux) for leptons of the given energy and zenith angle

    :param fname:     path to a pickle file containing yields
    :param emu:       minimum muon energy needed for veto [GeV at surface]
    :param nu_types:  a list of lepton families for which to evaluate the response

    :returns: a tuple (response, muonyield, energy_per_nucleon)
    """
    bundle = numpy.load(fname)
    e_grid = bundle['e_grid']

    contrib = numpy.empty((e_grid.size, 5, e_grid.size))
    muyield = numpy.empty((e_grid.size, 5, e_grid.size))
    ptypes = [getattr(ParticleType, pt) for pt in ('PPlus',
              'He4Nucleus', 'N14Nucleus', 'Al27Nucleus', 'Fe56Nucleus')]
    A = [[pt/100, 1][pt == ParticleType.PPlus] for pt in ptypes]
    elements = ['H', 'He', 'N', 'Al', 'Fe']
    for i, (ptype, element, a) in enumerate(zip(ptypes, elements, A)):
        primary_energy = e_grid*a
        de = get_bin_width(primary_energy)
        # hobo-integrate the flux
        weights = gaisser_flux(primary_energy, int(ptype))*de
        ey = extract_yields(bundle['yields'][element], nu_types)
        # contributions to the neutrino flux from showers of given energy
        contrib[:, i, :] = (weights[:, None]*ey).T
        # average number of muons per shower above threshold
        muy = extract_yields(bundle['yields'][element], [
                             'total_mu-', 'total_mu+'])
        if not prompt_muons:
            muy -= extract_yields(bundle['yields']
                                  [element], ['pr_mu-', 'pr_mu+'])
        muyield[:, i, :] = integral_yield(emu, e_grid, muy)[None, :]
    return contrib, muyield, e_grid


def uncorrelated_passing_rate(enu, emu, cos_theta, kind='numu'):
    """
    Calculate the probability that neutrinos of the given energy and type will
    be accompanied by at least one muon from an unrelated branch of the shower.

    :param enu:       neutrino energy
    :param emu:       minimum muon energy needed for veto [GeV at surface]
    :param cos_theta: cosine of zenith angle
    :param kind:      neutrino type for which to evaluate the passing rate:
                      "numu" for muon neutrinos from pion/kaon decay, "nue" for
                      electron neutrinos from kaon decay, or "charm" for either
                      flavor from charmed meson decay
    """
    # get contributions to the differential neutrino flux from chunks of
    # the cosmic ray spectrum in each element
    contrib, muyield = response_function(enu, emu, cos_theta, kind)[:2]
    # normalize contributions over all primaries
    contrib /= contrib.sum(axis=(-1, -2), keepdims=True)
    # weight contributions by probability of having zero muons. if that
    # probability is always 1, then this returns 1 by construction
    return (numpy.exp(-muyield)*contrib).sum(axis=(-1, -2))


def mceq_uncorrelated_passing_rate(fname, emu, nu_types, prompt_muons=False):
    # get contributions to the differential neutrino flux from chunks of
    # the cosmic ray spectrum in each element
    contrib, muyield = mceq_response_function(
        fname, emu, nu_types, prompt_muons)[:2]
    # normalize contributions over all primaries
    contrib /= contrib.sum(axis=(-1, -2), keepdims=True)
    # weight contributions by probability of having zero muons. if that
    # probability is always 1, then this returns 1 by construction
    return (numpy.exp(-muyield)*contrib).sum(axis=(-1, -2))


def analytic_numu_flux(enu, cos_theta, emu=None):
    """
    Calculate the differential flux of muon-neutrinos from pion and kaon decay
    in air showers, optionally restricting the solution to the part of the
    decay phase space where the lab-frame energy of the muon is above the given
    threshold [Schoenert]_. See [Lipari]_ for a detailed derivation of the
    cascade-equation solution.

    :param enu:       neutrino energy at which to evaluate [GeV]
    :param cos_theta: cosine of zenith angle in detector-centered coordinates
    :param emu:       if not None, calculate only the fraction of the flux
                      where the partner muon has at least the given energy.
                      Otherwise, calculate the total flux.

    :returns: the neutrino flux as a fraction of the primary nucleon flux at
              evaluated at the given energy.

    .. [Schoenert] S. Schoenert, T. K. Gaisser, E. Resconi, and O. Schulz. Vetoing atmospheric neutrinos in a high energy neutrino telescope. Phys. Rev. D, 79:043009, Feb 2009. doi: 10.1103/PhysRevD.79.043009.
    .. [Lipari] P. Lipari. Lepton spectra in the earth's atmosphere. Astroparticle Physics, 1 (2):195--227, 1993. ISSN 0927-6505. doi: 10.1016/0927-6505(93)90022-6.
    """

    # Spectral index of the integral nucleon flux at the top of the atmosphere
    GAMMA = 1.7
    # Spectrum-weighted moments for nucleon and meson production
    Z_NN, Z_NPI, Z_NK, Z_PIPI,	Z_KK = 0.298, 0.079, 0.0118,  0.271, 0.223
    # Critical energies for pions and kaons above which re-interaction is more
    # likely than decay in flight
    EPS_PI, EPS_K = 115.,  850.
    # Useful constants
    R_PI, R_K, ALAM_N, ALAM_PI, ALAM_K = .5731, 0.0458,	  120.,	   160.,   180.

    F = (GAMMA + 2.)/(GAMMA + 1.)
    B_PI = F * (ALAM_PI - ALAM_N) / (ALAM_PI *
                                     numpy.log(ALAM_PI/ALAM_N)*(1.-R_PI))
    B_K = F * (ALAM_K - ALAM_N) / (ALAM_K *
                                   numpy.log(ALAM_K / ALAM_N)*(1.-R_K))

    if emu is not None:
        z = 1 + emu/enu
        zpimin = 1./(1.-R_PI)
        zkmin = 1./(1.-R_K)
        zzpi = numpy.where(z >= zpimin, z, zpimin)
        zzk = numpy.where(z >= zkmin, z, zkmin)
        A_PI = Z_NPI/((1.-R_PI)*(GAMMA+1)*zzpi**(GAMMA+1.))
        A_K = Z_NK/((1.-R_K)*(GAMMA+1)*zzk**(GAMMA+1.))

        B_PI = zzpi*B_PI*(1.-R_PI)
        B_K = zzk*B_K*(1.-R_K)
    else:
        A_PI = (Z_NPI*((1.-R_PI)**(GAMMA+1.))/(GAMMA+1.))/(1.-R_PI)
        A_K = (Z_NK * ((1.-R_K)**(GAMMA+1.))/(GAMMA+1.))/(1.-R_K)

    CS = effective_costheta(cos_theta)
    return (A_PI / (1. + B_PI*CS*enu/EPS_PI) + 0.635 * A_K / (1. + B_K*CS*enu/EPS_K))


def correlated_passing_rate(enu, emu, cos_theta):
    """
    Calculate the probability that muon neutrinos of the given energy will be
    accompanied by a muon from the same decay vertex.

    :param enu:       neutrino energy
    :param emu:       minimum muon energy needed for veto [GeV at surface]
    :param cos_theta: cosine of zenith angle
    """
    flux = analytic_numu_flux(enu, cos_theta, None)
    sflux = analytic_numu_flux(enu, cos_theta, emu)
    return (flux-sflux)/flux


# Cynthia Brewer's human-friendly colors http://colorbrewer2.org
colors = [(0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
          (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
          (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
          (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
          (1.0, 0.4980392156862745, 0.0),
          (1.0, 1.0, 0.2),
          (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
          (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
          (0.6, 0.6, 0.6)]


def plot_passing_rate(depth):
    import pylab
    ax = pylab.gca()
    enu = numpy.logspace(3, 7, 101)

    for zenith, color in zip((0, 70, 80, 85), colors):
        cos_theta = numpy.cos(numpy.radians(zenith))
        emu = minimum_muon_energy(overburden(cos_theta, depth))
        correlated = correlated_passing_rate(enu, emu, cos_theta)
        uncorr_numu = uncorrelated_passing_rate(
            enu, emu, cos_theta, kind='numu')
        uncorr_nue = uncorrelated_passing_rate(enu, emu, cos_theta, kind='nue')

        pylab.plot(enu, correlated, color=color, label='%d' % zenith)
        pylab.plot(enu, correlated*uncorr_numu, color=color, ls=':')
        pylab.plot(enu, uncorr_nue, color=color, ls='--')

    pylab.semilogx()
    pylab.ylim((0, 1))
    pylab.ylabel('Passing fraction')
    pylab.xlabel('Neutrino energy [GeV]')
    olegend = pylab.legend(loc='lower left', title='Zenith [deg]')
    pylab.legend(ax.lines[:3], (r'$\nu_{\mu}$ (Correlated)', r'$\nu_{\mu}$ (Total)', r'$\nu_{e}$'),
                 loc='center right', prop=dict(size='small'))
    ax.add_artist(olegend)
    pylab.title('Atmospheric neutrino self-veto at %d m depth' % (opts.depth))

    pylab.show()


def format_energy(fmt, energy):
    places = int(numpy.log10(energy)/3)*3
    if places == 1:
        unit = 'GeV'
    elif places == 3:
        unit = 'TeV'
    elif places == 6:
        unit = 'PeV'
    elif places == 9:
        unit = 'EeV'
    return (fmt % (energy/10**(places))) + ' ' + unit


def plot_response_function(enu, depth, cos_theta, kind):

    emu = minimum_muon_energy(overburden(cos_theta, depth))
    response, muyield, energy_per_nucleon = response_function(
        enu, emu, cos_theta, kind)

    import pylab
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import NullFormatter

    fig = pylab.figure()
    grid = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0)

    ax = pylab.subplot(grid[0])
    elements = ['H', 'He', 'C, N, O', 'Mg, Si, Al', 'Fe']
    for i, color, label in zip(range(5), colors, elements):
        if i > 0 or i < 4:
            continue
        pylab.plot(energy_per_nucleon,
                   response[i, :], color=color, lw=1, label=label)
        pylab.plot(energy_per_nucleon,
                   numpy.exp(-muyield[i, :])*response[i, :], color=color, ls='--', lw=1)

    ax.plot(energy_per_nucleon, response.sum(
        axis=0), color='k', lw=2, label='Total')
    ax.plot(energy_per_nucleon, (numpy.exp(-muyield)*response).sum(axis=-2),
            color='k', ls='--', lw=2, label='(no muons)')

    pylab.loglog()
    ax.set_ylabel('Partial flux $[GeV^{-1}\, m^{-2}\, sr^{-1}\, s^{-1}]$')
    #ax.legend(loc='best', prop=dict(size='small'), title='Flux contributions')
    ax.legend(loc='lower left', prop=dict(size='small'),
              title='Flux contributions', ncol=2)
    passrate = (numpy.exp(-muyield)*response).sum(axis=(-2, -1)) / \
        response.sum(axis=(-2, -1))
    if kind == 'charm':
        kindlabel = r'$\nu_{e/\mu}$ (charm)'
    else:
        kindlabel = r'$\nu_'
        if len(kind) > 3:
            kindlabel += '{\\'
        kindlabel += kind[2:] + '}$'
    ax.set_title('%s %s from %d deg at %d m depth: %d%% passing rate' %
                 (format_energy('%d', enu), kindlabel, numpy.round(
                     numpy.degrees(numpy.arccos(cos_theta))), depth, passrate*100),
                 size='medium')

    logmax = numpy.ceil(numpy.log10(response.max()))
    ax.set_ylim(10**(logmax-8), 10**(logmax))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_ticks(ax.yaxis.get_ticklocs()[2:-1])

    ax = pylab.subplot(grid[1])
    for i, color, label in zip(range(5), colors, elements):
        pylab.plot(energy_per_nucleon,
                   muyield[i, :], color=color, lw=1, label=label)
    pylab.loglog()
    logmax = numpy.ceil(numpy.log10(muyield.max()))
    ax.set_ylim(10**(logmax-4), 10**(logmax))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_ylabel(r'$\left<N_{\mu} > 1\, \mathrm{TeV} \right>$')
    ax.yaxis.set_ticks(ax.yaxis.get_ticklocs()[2:-1])

    ax = pylab.subplot(grid[2])
    for i, color, label in zip(range(5), colors, elements):
        pylab.plot(energy_per_nucleon,
                   numpy.exp(-muyield[i, :]), color=color, lw=1, label=label)
    pylab.semilogx()
    ax.set_ylabel(r'$P(N_{\mu} = 0)$')

    pylab.xlabel('Energy per nucleon [GeV]')

    for ax in fig.axes:
        ax.grid(color='lightgrey', ls='-', lw=0.1)
        ax.yaxis.set_label_coords(-.07, .5)

    help = """\
	The solid lines in the upper panel of the displayed plot show
	the contributions of cosmic ray primaries to the differential neutrino flux at
	%s and %d degrees. The center panel shows the mean number of muons from each
	shower that reach a vertical depth of %d meters with more than 1 TeV of kinetic energy.
	The bottom panel shows the Poisson probability of observing 0 muons at depth
	given the mean in the center panel. These probabilities are multiplied with the flux
	contributions to obtain the effective contributions that can be observed if
	accompanying muons are rejected, shown as dashed lines in the upper panel. The
	passing rate is the ratio of observable to the total flux, which is the ratio
	of the integrals of the solid and dashed black curves (%d%%).
	""" % (format_energy('%d', enu), numpy.round(numpy.degrees(numpy.arccos(cos_theta))), depth, passrate*100)

    import textwrap
    for line in textwrap.wrap(textwrap.dedent(help), 78):
        print(line)

    pylab.tight_layout()
    pylab.savefig('response_function.pdf', transparent=True)

    pylab.show()


if __name__ == "__main__":

    from optparse import OptionParser, OptionGroup
    import sys
    parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
    parser.add_option("--flavor", choices=('numu', 'nue'), default='numu',
                      help="[default: %default]")
    parser.add_option("--charm", action="store_true", default=False,
                      help="Evaluate self-veto probability for neutrinos from charm rather than conventional decay")
    parser.add_option("--depth", type=float, default=1950.,
                      help="Vertical depth at which to calculate veto probability [default: %default]")
    group = OptionGroup(parser, "Interactive plotting options",
                        "The following options can be used to display an interactive plot instead of writing a table. This requires matplotlib.")
    group.add_option("--plot", choices=('None', 'passingrate', 'response'),
                     help="Choices: 'passingrate' for a plot of veto passing rates, 'response' for a plot of the neutrino response function and associated muon yields")
    group.add_option("--energy", type=float, default=1e4,
                     help="Evaluate response function for this neutrino energy [%default GeV]")
    group.add_option("--zenith", type=float, default=60,
                     help="Evaluate response for this zenith angle [%default degrees]")
    parser.add_option_group(group)
    opts, args = parser.parse_args()

    if opts.plot == 'passingrate':
        plot_passing_rate(opts.depth)
        sys.exit(0)
    elif opts.plot == 'response':

        plot_response_function(opts.energy, opts.depth,
                               numpy.cos(numpy.radians(opts.zenith)), [opts.flavor, 'charm'][opts.charm])
        sys.exit(0)

    # Calculate passing rate on a grid of energies and zenith angles
    enu = numpy.logspace(3, 7, 101)
    cos_theta = numpy.arange(0, 1, .05)+.05
    enu, cos_theta = numpy.meshgrid(enu, cos_theta)
    emu = minimum_muon_energy(overburden(cos_theta, opts.depth))

    if opts.flavor == 'numu':
        passrate = correlated_passing_rate(enu, emu, cos_theta)
    else:
        passrate = numpy.ones(enu.shape)

    if opts.charm:
        kind = 'charm'
    else:
        kind = opts.flavor
    passrate *= uncorrelated_passing_rate(enu, emu, cos_theta, kind=kind)

    data = numpy.vstack(list(map(numpy.ndarray.flatten, (cos_theta, overburden(
        cos_theta, opts.depth), emu, enu, passrate)))).T
    fields = ['cos_theta', 'overburden', 'emu_min', 'energy', 'passrate']

    numpy.savetxt(sys.stdout, data, fmt='%12.3e', header='\t'.join(
        ['%12s' % s for s in fields])[1:])
