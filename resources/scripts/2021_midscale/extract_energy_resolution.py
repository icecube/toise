import matplotlib.pyplot as plt
import tables
import pandas as pd
import numpy as np
from toolz import memoize
from gen2_analysis import plotting, surfaces

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("-n",type=str,help="geom name, e.g. hcr", required=False, dest='geom', default='standard')
parser.add_argument("-p",type=bool,help="make plots?", required=False, dest='do_plots', default=False)
args = parser.parse_args()
the_geom = args.geom
the_file = 'data_sunflower_{}.hdf5'.format(the_geom)
print(("the file {}".format(the_file)))


@memoize
def load_dataset(hdf_fname):
	dfs = {}
	# read each table into a DataFrame
	with tables.open_file(hdf_fname) as hdf:
		for tab in 'MCMuon', 'MuonEffectiveArea', 'SplineMPEMuEXDifferential', 'LineFit', 'SplineMPE_recommendedFitParams', 'SplineMPE_recommendedDirectHitsC':
			dfs[tab] = pd.DataFrame(hdf.get_node('/'+tab).read_where('exists==1')).set_index(['Run', 'Event', 'SubEvent', 'SubEventStream'])
	# apply event selection
	mask = (dfs['LineFit']['speed'] < 0.6)&\
		(dfs['SplineMPE_recommendedFitParams']['rlogl'] < 8.5)&\
		(dfs['SplineMPE_recommendedDirectHitsC']['n_dir_doms'] > 6)&\
		(dfs['SplineMPE_recommendedDirectHitsC']['dir_track_length'] > 120)
	# angular reconstruction error
	def cartesian_components(df):
		theta = df['zenith']
		phi = df['azimuth']
		return -np.sin(theta)*np.cos(phi), -np.sin(theta)*np.sin(phi), -np.cos(theta)
	def opening_angle(df1, df2):
		x1, y1, z1 = cartesian_components(df1)
		x2, y2, z2 = cartesian_components(df2)
		return np.arccos(x1*x2+y1*y2+z1*z2)
	# pack relevant quantities into a single DataFrame
	dats = pd.DataFrame(
		{
			'opening_angle':
				np.degrees(
					  opening_angle(
						  dfs['MCMuon'].loc[mask],
						  dfs['SplineMPEMuEXDifferential'].loc[mask]
					  )
				),
			'energy_reco': dfs['SplineMPEMuEXDifferential']['energy'].loc[mask],
			'aeff':  dfs['MuonEffectiveArea']['value'].loc[mask],
			'energy': dfs['MCMuon']['energy'].loc[mask],
			'cos_zenith': np.cos(dfs['MCMuon']['zenith'].loc[mask]),
		}
	)
	return dats


df = load_dataset(the_file)

# just see what we're working with in terms of statistics
print((df.describe()))

bins = np.linspace(2, 12, 51)

dats = (
    (np.log10(df['energy_reco'])-np.log10(df['energy']))
    .groupby(pd.cut(np.log10(df['energy']), np.linspace(2, 12, 51)))
    .agg([np.mean, np.std])
)
# use the midpoint of the energy interval as the x axis
dats.index = dats.index.map(lambda interval: interval.mid).astype(float)
# assume constant bias and variance above simulated energy range
dats.fillna(method='ffill', inplace=True)
# assume contant variance below simulated energy range
dats['std'].fillna(method='bfill', inplace=True)
# assume bias such that mean reconstructed energy is constant below simulated range
nonnull = dats['mean'][~dats['mean'].isnull()]
constant_offset = pd.Series(nonnull.iloc[0] + (nonnull.index[0]-dats.index), index=dats.index)
dats['mean'].fillna(constant_offset, inplace=True)

if args.do_plots:
	fig, ax = plt.subplots(1,1, figsize=(8,5))
	ax.fill_between(dats.index, dats.index + dats['mean'] - dats['std'], dats.index + dats['mean'] + dats['std'])
	ax.plot(dats.index, dats.index + dats['mean'], color='w', ls=':')
	ax.set_xlabel('log_10(muon energy/GeV)')
	ax.set_ylabel('log_10(reco energy/GeV)')
	fig.savefig('energy_resolution_{}.png'.format(the_geom))
	plt.close(fig)
	del fig, ax

np.savez("energy_resolution_{}.npz".format(the_geom), loge=dats.index, mean=dats.index+dats['mean'], std=dats['std'], smoothing=1e-2)


