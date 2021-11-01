
from gen2_analysis.figures import figure

from gen2_analysis import surfaces
import gzip
import numpy as np
import itertools

def get_string_heads(geometry, spacing, **kwargs):
    with gzip.GzipFile(surfaces.get_geometry_file(geometry, spacing)) as f:
        geo = np.loadtxt(f, dtype=np.dtype(
            [('string', int), ('om', int)] + [(c, float) for c in 'xyz']))
        pos = geo[geo['om'] == 1]
        return pos[list('xy')]

def half_fantasy_radio_geometry(sector, spacing=1.5e3, nstations=200):
    def ray(points):
        x0 = points[0]
        dx = np.diff(points, axis=0)[0, :]
        return lambda x: x0[1] + dx[1]/dx[0]*(x-x0[0])
    edge = sector[1:3]
    top = ray(sector[2:4])
    bottom = ray(sector[0:2])

    # basis vectors along the skiway edge of the dark sector
    dx = np.diff(edge, axis=0)[0, :]
    dx /= np.hypot(*dx)
    perpdx = -np.asarray([dx[1], -dx[0]])
    x0 = edge[0]

    locations = []

    stations = nstations*50
    for row in itertools.count():
        corner = x0 + row*perpdx*spacing
        for col in itertools.count():
            pos = corner + col*dx*spacing
            if pos[1] > top(pos[0]):
                break
            locations.append(pos)
            if len(locations) >= stations:
                break
        for col in itertools.count(1):
            pos = corner - col*dx*spacing
            if pos[1] < bottom(pos[0]):
                break
            locations.append(pos)
            if len(locations) >= stations:
                break
        if len(locations) >= stations:
            break

    locs = np.vstack(locations)

    order = np.hypot(*locs.T).argsort()
    return locs[order][:nstations]

def get_surface_geometry():
    import pandas as pd
    upgrade = np.asarray([[18.289999999999935, -51.05374999999998],  # 87
                          [47.289999999999964, -57.01249999999996],  # 88
                          [14.289999999999935, -80.56374999999997],  # 89
                          [57.28999999999999, -83.68499999999997],  # 90
                          [89.29000000000005, -58.99875],  # 91
                          [62.62333333333336, -35.16374999999999],  # 92
                          [26.95666666666662, -31.191249999999968],  # 93
                          ])
    sectors = {
        "Dark Sector": np.array([
            [-6745.5344331313745, -6666.666666666667],
            [360.24866210820073, -704.0174330560367],
            [929.3377603199988,  732.0277663199995],
            [-1230.6941085521246,  6666.666666666667],
            [-1230.6941085521246,  6666.666666666667],
            [-11166.666666666666,  6666.666666666667],
            [-12500.0,  5333.333333333334],
            [-12500.0, -5333.333333333334],
            [-11166.666666666666, -6666.666666666667],
            [-6745.5344331313745, -6666.666666666667],
        ])
    }
    radio = half_fantasy_radio_geometry(sectors['Dark Sector'])
    pos = get_string_heads('Sunflower', 240)
    x, y = pos['x'], pos['y']
    labels = ['IceCube']*x[:86].shape[0] + ['Upgrade']*upgrade.shape[0] + ['Gen2']*x[86:].shape[0] + ['Radio']*radio.shape[0]
    return pd.DataFrame(dict(subdetector=labels,
        x=np.concatenate([x[:86],upgrade[:,0],x[86:],radio[:,0]]),
        y=np.concatenate([y[:86],upgrade[:,1],y[86:],radio[:,1]]))
    )

@figure
def surface_geometry():
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.axes_grid1 import inset_locator
    import matplotlib.patheffects as path_effects
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))



    pos = get_string_heads('Sunflower', 240)
    upgrade = np.asarray([[18.289999999999935, -51.05374999999998],  # 87
                          [47.289999999999964, -57.01249999999996],  # 88
                          [14.289999999999935, -80.56374999999997],  # 89
                          [57.28999999999999, -83.68499999999997],  # 90
                          [89.29000000000005, -58.99875],  # 91
                          [62.62333333333336, -35.16374999999999],  # 92
                          [26.95666666666662, -31.191249999999968],  # 93
                          ])
    sectors = {
        "Dark Sector": np.array([
            [-6745.5344331313745, -6666.666666666667],
            [360.24866210820073, -704.0174330560367],
            [929.3377603199988,  732.0277663199995],
            [-1230.6941085521246,  6666.666666666667],
            [-1230.6941085521246,  6666.666666666667],
            [-11166.666666666666,  6666.666666666667],
            [-12500.0,  5333.333333333334],
            [-12500.0, -5333.333333333334],
            [-11166.666666666666, -6666.666666666667],
            [-6745.5344331313745, -6666.666666666667],
        ])
    }

    def add_scalebar(ax, size, label):
        scalebar = AnchoredSizeBar(ax.transData,
                                   size, label, 'lower left',
                                   pad=0.5,
                                   frameon=False,
                                   sep=4,
                                   )
        ax.add_artist(scalebar)

    def scatter_noscale(ax, *args, **kwargs):
        """
        scatter without updating axis limits
        """
        toggle = ax.get_autoscale_on()
        ax.autoscale(False)
        points = ax.scatter(*args, **kwargs)
        del ax.collections[-1]
        ax.add_collection(points, autolim=False)
        ax.set_autoscale_on(toggle)
        return points

    def ray(points):
        x0 = points[0]
        dx = np.diff(points, axis=0)[0, :]
        return lambda x: x0[1] + dx[1]/dx[0]*(x-x0[0])

    def half_fantasy_radio_geometry(sector, spacing=1.5e3, nstations=200):
        edge = sector[1:3]
        top = ray(sector[2:4])
        bottom = ray(sector[0:2])

        # basis vectors along the skiway edge of the dark sector
        dx = np.diff(edge, axis=0)[0, :]
        dx /= np.hypot(*dx)
        perpdx = -np.asarray([dx[1], -dx[0]])
        x0 = edge[0]

        locations = []

        stations = nstations*50
        for row in itertools.count():
            corner = x0 + row*perpdx*spacing
            for col in itertools.count():
                pos = corner + col*dx*spacing
                if pos[1] > top(pos[0]):
                    break
                locations.append(pos)
                if len(locations) >= stations:
                    break
            for col in itertools.count(1):
                pos = corner - col*dx*spacing
                if pos[1] < bottom(pos[0]):
                    break
                locations.append(pos)
                if len(locations) >= stations:
                    break
            if len(locations) >= stations:
                break

        locs = np.vstack(locations)

        order = np.hypot(*locs.T).argsort()
        return locs[order][:nstations]

    radio = half_fantasy_radio_geometry(sectors['Dark Sector'])

    axes[0].scatter(upgrade[:, 0], upgrade[:, 1], s=1, color='C3')
    axes[0].scatter(pos[:86]['x'], pos[:86]['y'], s=1, color='C0', marker='H')
    axes[0].scatter(pos[86:]['x'], pos[86:]['y'], s=1, color='C1', marker='o')
    axes[0].scatter(radio[:, 0], radio[:, 1], s=5, color='C4',
                    label='IceCube Gen2-Radio', marker='1')

    axes[1].scatter(upgrade[:, 0], upgrade[:, 1], s=1, color='C3')
    axes[1].scatter(pos[:86]['x'], pos[:86]['y'], s=1, color='C0', marker='H')
    # scatter_noscale(axes[1], radio[:,0], radio[:,1], s=5, color='C4', marker='1')
    axes[1].scatter(pos[86:]['x'], pos[86:]['y'], s=5,
                    color='C1', label='IceCube Gen2-Optical', marker='o')

    axes[2].scatter(pos[:86]['x'], pos[:86]['y'], s=5,
                    color='C0', label='IceCube', marker='H')

    axes[3].scatter(upgrade[:, 0], upgrade[:, 1], s=10,
                    color='C3', label='IceCube Upgrade', marker='P')
    scatter_noscale(axes[3], pos[:86]['x'], pos[:86]
                    ['y'], s=5, marker='H', color='C0')
    axes[3].set_ylim(bottom=-115)

    for ax in axes:
        ax.set_aspect('equal', 'datalim')
        for label in ax.yaxis.get_ticklabels():
            label.set_path_effects([
                path_effects.Stroke(linewidth=5, foreground='white'),
                path_effects.Normal()
            ])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
                  frameon=False, handletextpad=0.1, markerscale=3)

    add_scalebar(axes[0], 5000, '5 km')
    add_scalebar(axes[1], 1000, '1 km')
    add_scalebar(axes[2], 250, '250 m')
    add_scalebar(axes[3], 25, '25 m')

    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_alpha(0.3)

    for spine in axes[0].spines.values():
        spine.set_visible(False)

    for parent, child in reversed(zip(axes[:-1], axes[1:])):
        inset_locator.mark_inset(
            parent, child, 2, 3, edgecolor='black', alpha=0.3, ls='-')

    axes[0].yaxis.pan(-0.32)

    axes[1].margins(0.3)
    axes[1].yaxis.pan(-0.32)

    plt.tight_layout()

    return fig

@figure
def radio_surface_geometry():
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.axes_grid1 import inset_locator
    import matplotlib.patheffects as path_effects
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    fig, axes = plt.subplots(1, 1, figsize=(2*2.5, 2*2.5))

    pos = get_string_heads('Sunflower', 240)
    upgrade = np.asarray([[18.289999999999935, -51.05374999999998],  # 87
                          [47.289999999999964, -57.01249999999996],  # 88
                          [14.289999999999935, -80.56374999999997],  # 89
                          [57.28999999999999, -83.68499999999997],  # 90
                          [89.29000000000005, -58.99875],  # 91
                          [62.62333333333336, -35.16374999999999],  # 92
                          [26.95666666666662, -31.191249999999968],  # 93
                          ])

    ARA = np.asarray([[-2346.96, -345.6432],  # A2
                        [-3344.2656, -2078.1264],  # A2
                        [-4344.6192, -345.948],  # A3
                        [-3388.7664, 1437.4368],  # A4
                        [-4342.7904, -3804.5136],  # A5
                        ])


    sectors = {
        "Dark Sector": np.array([
            [-6745.5344331313745, -6666.666666666667],
            [360.24866210820073, -704.0174330560367],
            [929.3377603199988,  732.0277663199995],
            [-1230.6941085521246,  6666.666666666667],
            [-1230.6941085521246,  6666.666666666667],
            [-11166.666666666666,  6666.666666666667],
            [-12500.0,  5333.333333333334],
            [-12500.0, -5333.333333333334],
            [-11166.666666666666, -6666.666666666667],
            [-6745.5344331313745, -6666.666666666667],
        ])
    }


    def add_scalebar(ax, size, label):
        scalebar = AnchoredSizeBar(ax.transData,
                                   size, label, 'lower left',
                                   pad=0.5,
                                   frameon=False,
                                   sep=4,
                                   )
        ax.add_artist(scalebar)
    
    # function to rotate a point (point_x, point_y) about an origin by degrees
    def rotate_point(point_x, point_y, origin_x, origin_y, degrees):
        radians = np.deg2rad(degrees)
        x,y = point_x, point_y
        offset_x, offset_y = origin_x, origin_y
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = np.cos(radians)
        sin_rad = np.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy

    def review_2021_array_geometry():

        spacing = 2.0
        xx_deep = np.arange(-11, 11.1, spacing)*1000.
        yy_deep = np.arange(-11, 11.1, spacing)*1000.
        d = np.meshgrid(xx_deep, yy_deep)
        xx_shallow = np.arange(-11 - 0.5 * spacing, 11.1 + 0.5 * spacing, spacing)*1000.
        yy_shallow = np.arange(-11 - 0.5 * spacing, 11.1 + 0.5 * spacing, spacing)*1000.
        s = np.meshgrid(xx_shallow, yy_shallow)

        shift = -12000

        xx_deep = d[0].flatten()+shift
        yy_deep = d[1].flatten()-shift
        xx_shallow = s[0].flatten()+shift
        yy_shallow = s[1].flatten()-shift

        for i, _ in enumerate(xx_deep):
            rot_x, rot_y = rotate_point(xx_deep[int(i)], yy_deep[int(i)], 0,0, degrees=-30)
            xx_deep[i] = rot_x
            yy_deep[i] = rot_y
        for i, _ in enumerate(xx_shallow):
            rot_x, rot_y = rotate_point(xx_shallow[int(i)], yy_shallow[int(i)], 0,0, degrees=-30)
            xx_shallow[i] = rot_x
            yy_shallow[i] = rot_y            

        return [xx_deep, yy_deep, xx_shallow, yy_shallow]

    radio_review  = review_2021_array_geometry()
    
    # plot IC68 and Gen2
    axes.scatter(pos[:86]['x'], pos[:86]['y'], s=1, color='C7', marker='H')
    axes.scatter(pos[86:]['x'], pos[86:]['y'], s=1, color='C2', marker='o')

    axes.scatter(radio_review[0], radio_review[1], s=10, color='C0',
                    label='Deep+Shallow Stations', marker='o')
    axes.scatter(radio_review[2], radio_review[3], s=7, color='C1',
                    label='Shallow-only Stations', marker='D')


    # make them equal, add a scalebar
    axes.set_aspect('equal', 'datalim')
    for label in axes.yaxis.get_ticklabels():
        label.set_path_effects([
            path_effects.Stroke(linewidth=5, foreground='white'),
            path_effects.Normal()
        ])
    axes.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
              frameon=False, handletextpad=0.1, markerscale=3)

    add_scalebar(axes, 5000, '5 km')

    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    for spine in axes.spines.values():
        spine.set_color('black')
        spine.set_alpha(0.3)

    for spine in axes.spines.values():
        spine.set_visible(False)

    axes.yaxis.pan(-0.15)

    plt.tight_layout()

    return fig

