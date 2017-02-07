
from scipy import interpolate
import pickle, os, numpy
import surfaces
from util import *

# hobo-costing!
def surface_area(theta_max, volume):
    """
    Surface coverage area required so that a track that 
    """
    d = 1950 - volume._z_range[0] # depth of the bottom of the detector
    return numpy.pi*(d*numpy.tan(theta_max) + numpy.sqrt(volume.get_cap_area()/numpy.pi))**2

def array_cost(area, fill_factor):
    """
    :param area: area of array, in km^2
    :param fill_factor: fraction of surface that is active
    :returns: cost, in dollars, to build the array
    """
    return 225e3 + 7.5e8*area*fill_factor

def fill_factor_for_threshold(emu_min, passing_rate=1e-4):
    """
    Use the fact that the passing rate of 100 TeV protons for a 1e-3-packed
    scintillator array is roughly 1e-3, and muons come from primaries of ~20x higher energy.
    
    :returns: fill factor necessary for *passing_rate* rejection of muons of
    energy *emu_min*
    """
    return (1e5/(20*emu_min))*(1e-3/passing_rate)*1e-3

def veto_cost_for_angle(theta_max, emu_min, surface):
    """
    cost, in megabucks, of an array that rejects all atmospheric backgrounds
    above emu_min out to theta_max
    """
    fill = fill_factor_for_threshold(emu_min)
    area = surface_area(numpy.radians(theta_max), surface)
    return array_cost(area/1e6, fill)/1e6

def veto_cost(area, emu_min, surface):
    """
    cost, in megabucks, of an array that rejects all atmospheric backgrounds
    above emu_min and covers an area *area* km^2
    """
    fill = fill_factor_for_threshold(emu_min)
    return array_cost(area, fill)/1e6

from scipy.optimize import fsolve
def margin_for_area(base_surface, area):
    """
    :param area: desired area, in km^2
    """
    def area_diff(margin):
        return base_surface.expand(margin).get_cap_area()/1e6 - area
    return fsolve(area_diff, 0)[0]

def get_geometric_coverage_for_area(geometry, spacing, area, ct_bins=numpy.linspace(0, 1, 11), nsamples=int(1e4)):
    """
    Calculate the geometric coverage of a surface veto by Monte Carlo
    :param gcdfile: path to a GCD file defining the geometry of the in-ice detector
    :param area: area of the surface veto, in km^2
    :param ct_bins: bins in cos(zenith) over which to average the coverage
    :param nsamples: number of trials in each bin
    :returns: an array of length len(ct_bins)-1 containing the coverage fraction
              in each bin
    """
    # SamplingSurfaces from MuonGun
    from icecube import MuonGun, phys_services
    rng = phys_services.I3GSLRandomService(numpy.random.randint(numpy.iinfo(numpy.uint32).max))
    
    ref_surface = surfaces.get_fiducial_surface(geometry, spacing)
    margin = margin_for_area(ref_surface, area)
    veto_surface = ref_surface.expand(margin)
    
    if geometry == 'IceCube':
        deep_surface = MuonGun.Cylinder(ref_surface.length, ref_surface.radius)
    else:
        gcdfile = surfaces.get_gcd(geometry, spacing)
        deep_surface = MuonGun.ExtrudedPolygon.from_file(gcdfile, 60.)
    
    coverage = numpy.zeros(ct_bins.size-1)
    
    if not area > 0:
        return coverage
    
    for i, (ct_lo, ct_hi) in enumerate(zip(ct_bins[:-1], ct_bins[1:])):
        # upgoing events can never be vetoed, no matter what
        if ct_hi < 0:
            continue
        inside = 0
        for dummy in xrange(nsamples):
            pos, direction = deep_surface.sample_impact_ray(rng, ct_lo, ct_hi)
            # project up the to the surface
            pos += ((1950-pos.z)/direction.z)*direction
            # catch stupid sign errors
            assert abs(pos.z - 1950) < 1
            # did the shower cross the surface array?
            if veto_surface.point_in_footprint(tuple(pos)):
                inside += 1
        coverage[i] = inside/float(nsamples)
    return coverage

class GeometricVetoCoverage(object):
    cache_file = os.path.join(data_dir, 'veto', 'geometric_veto_coverage.pickle')
    def __init__(self, geometry='Sunflower', spacing=240, area=10.):
        self.geometry = geometry
        self.spacing = spacing
        self.area = area
        if os.path.exists(self.cache_file):
            self.cache = pickle.load(open(self.cache_file))
        else:
            self.cache = dict()
    def __call__(self, ct_bins=numpy.linspace(-1, 1, 11)):
        key = (self.geometry, self.spacing, self.area, (ct_bins[0], ct_bins[-1], len(ct_bins)))
        if key in self.cache:
            return self.cache[key]
        else:
            coverage = get_geometric_coverage_for_area(self.geometry, self.spacing, self.area, ct_bins, 100000)
            self.cache[key] = coverage
            pickle.dump(self.cache, open(self.cache_file, 'w'))
            return coverage

class EulerVetoProbability(object):
    def __init__(self, fname=os.path.join(data_dir, 'veto', 'vetoeffs.pickle')):
        vetoeffs = pickle.load(open(fname))
        x, y, v_mu, v_e = vetoeffs['logE'], vetoeffs['cosZen'], vetoeffs['vetoEff_mu'], vetoeffs['vetoEff_e']
        # assuming that the muoninc and EM components are independent
        z = 1 - (1-v_mu)*(1-v_e)
        self.spline = interpolate.RectBivariateSpline(x, y, z)
        # don't trust table beyond its statistical limits
        self.pmax = numpy.nanmax(v_mu[v_mu<1])
        self.log_emin = x[0]
        self.log_emax = x[-1]
        self.ct_min = y[0]
        self.ct_max = y[-2]
    def __call__(self, energy, cos_theta):
        logE, cos_theta = numpy.broadcast_arrays(numpy.log10(energy/1e3), numpy.clip(cos_theta, -1, self.ct_max))
        p = numpy.clip(self.spline(logE, cos_theta, grid=False), 0, self.pmax)
        p[logE > self.log_emax] = self.pmax
        p[logE < self.log_emin] = 0
        p[cos_theta < self.ct_min] = 0
        return p