from gen2_analysis import effective_areas, diffuse, pointsource, angular_resolution, grb, surface_veto, multillh, plotting
from gen2_analysis import factory, figures_of_merit, util, figures
from gen2_analysis.util import data_dir, center
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import json

# this file is basically a copy/strip down of the figures.pointsource.flare.sensitivity workbook

# quick utility function (because somehow I can't import the figures.cli class...)
def jsonify(obj):
	"""
	Recursively cast to JSON native types
	"""
	if hasattr(obj, 'tolist'):
		return obj.tolist()
	elif hasattr(obj, 'keys'):
		return {jsonify(k): jsonify(obj[k]) for k in obj.keys()}
	elif hasattr(obj, '__len__') and not isinstance(obj, str) and not isinstance(obj, unicode):
		return [jsonify(v) for v in obj]
	else:
		return obj

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("-n",type=str,help="geom name, e.g. hcr", required=False, dest='geom', default='IceCube')
parser.add_argument("-g",type=str,help="gamma", required=False, dest='gamma', default="-2.0")
args = parser.parse_args()

the_geom = args.geom
the_gamma = args.gamma
gamma_num = float(the_gamma)
component_name = 'Gen2'
if 'Sunflower' not in the_geom:
	the_geom = 'Sunflower_' + the_geom
	component_name = 'Gen2-'+the_geom

livetime = 1
exposures = []
exposures.append((the_geom, livetime))
print('The exposures is {}'.format(exposures))

# what figures do we need
figures = OrderedDict([
	('differential_upper_limit', figures_of_merit.DIFF.ul),
	('differential_discovery_potential', figures_of_merit.DIFF.dp),
	('upper_limit', figures_of_merit.TOT.ul),
	('discovery_potential', figures_of_merit.TOT.dp),
])
decades = 1
gamma = gamma_num
emin = 0.

# calculate the geometric area of the detector footprint
surface = effective_areas.get_fiducial_surface(the_geom, spacing=240, padding=0)
area = surface.azimuth_averaged_area(-1.)/1e6 # what is the footprint size for straight downgoing events
print("Surface Area is {}".format(area))

cos_theta = factory.default_cos_theta_bins
psi_bins = dict(factory.default_psi_bins)
# artificially fix the veto area to that of IceTop
opts = dict(geometry=the_geom, spacing=240, veto_area=area, veto_threshold=1e5)
kwargs = {
	'cos_theta': cos_theta,
	'psi_bins':  psi_bins
}
# factory.add_configuration('IceCube', factory.make_options(geometry='IceCube', spacing=125.), **kwargs)
# factory.add_configuration('Gen2', factory.make_options(**opts.__dict__), **kwargs)
# factory.add_configuration('IceCube', factory.make_options(geometry=detector, spacing=125.), **kwargs)
factory.add_configuration(component_name, factory.make_options(**opts), **kwargs)

meta_major = {'detectors' : exposures}

meta = {'cos_zenith': factory.default_cos_theta_bins}
dlabel = the_geom
for zi in tqdm(range(20), desc=dlabel):
	fom = figures_of_merit.PointSource({'IceCube-TracksOnly': 0, 'Gen2-InIce-TracksOnly': 0, component_name : 1}, zi)
	for flabel, q in figures.items():
		kwargs = {'gamma': gamma, 'decades': decades}
		if not flabel.startswith('differential'):
			kwargs['emin'] = emin
		val = fom.benchmark(q, **kwargs)
		if not flabel in meta:
			meta[flabel] = {}
		if not dlabel in meta[flabel]:
			meta[flabel][dlabel] = {}
		if flabel.startswith('differential'):
			val = OrderedDict(zip(('e_center', 'flux', 'n_signal', 'n_background'), val))
		else:
			val = OrderedDict(zip(('flux', 'n_signal', 'n_background'), val))
		meta[flabel][dlabel][str(zi)] = val

meta_major['data'] = jsonify(meta)
outfile = 'sensitivity_' + the_geom + '_' + the_gamma + '.json'
with open(outfile, 'w') as f:
	json.dump(meta_major, f, indent=2)



