
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from icecube.gen2_analysis import plotting, surfaces, effective_areas
from icecube import icetray, dataclasses, dataio

def string_heads(gcdfile):
    from icecube import icetray, dataio, dataclasses
    f = dataio.I3File(gcdfile)
    omgeo = f.pop_frame(icetray.I3Frame.Geometry)['I3Geometry'].omgeo
    f.close()
    strings = dict()
    for omkey, geo in omgeo.items():
        if omkey.om == 1:
            strings[omkey.string] = geo.position
    return np.array([(strings[k][0], strings[k][1]) for k in sorted(strings.keys())])

configs = [
    ('IceCube', 125.),
    ('EdgeWeighted', 240),
    ('Sunflower', 200),
    ('Sunflower', 300),
]

def polygon_outline(surface):
    x = np.concatenate((surface._x[:,0], surface._nx[-1:,0]))/1e3
    y = np.concatenate((surface._x[:,1], surface._nx[-1:,1]))/1e3
    return x, y

with plotting.pretty():
    fig = plt.figure(figsize=(8,8))
    
    griddy = GridSpec(2,2)
    axes = list()
    for i, (geo, spacing) in enumerate(configs):
    
        ax = plt.subplot(griddy[i])
        axes.append(ax)
        heads = string_heads(surfaces.get_gcd(geo, spacing))/1e3

        ax.scatter(heads[:,0], heads[:,1], s=2, color='k')
        surface = surfaces.get_fiducial_surface(geo, spacing)
        if isinstance(surface, surfaces.Cylinder):
            from matplotlib.patches import Circle
            color = ax._get_lines.prop_cycler.next()['color']
            ax.add_artist(Circle((0,0), radius=surface.radius/1e3, edgecolor=color, facecolor='None'))
        else:
            x, y = polygon_outline(surface)
            ax.plot(x, y)
        
        # cascade_volume = effective_areas.HESEishSelectionEfficiency(geo, spacing)
        # x, y = polygon_outline(cascade_volume._fiducial_surface)
        # ax.plot(x, y, ls=':')
        
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')

    plt.tight_layout()
    
    # manually set aspect ratios
    for ax in fig.axes:
        ax.set_xlim((-3.5, 1))
        span = np.diff(ax.get_xlim())
        aspect = ax.bbox.height/ax.bbox.width/2.
        ax.set_ylim((-span*aspect, span*aspect))
    
    for i, (geo, spacing) in enumerate(configs):
        if spacing is None:
            label = geo
        else:
            label = '%s %dm' % (geo, spacing)
        axes[i].add_artist(AnchoredText(label, loc=4, frameon=False))
    plt.show()
    

