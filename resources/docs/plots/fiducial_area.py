import matplotlib.pyplot as plt
import numpy as np
from icecube.gen2_analysis import plotting, surfaces

configs = [
    ("IceCube", None),
    ("EdgeWeighted", 240),
    ("Sunflower", 200),
    ("Sunflower", 240),
    ("Sunflower", 300),
]

with plotting.pretty():
    fig = plt.figure(figsize=(4, 4))

    ct = np.linspace(-1, 0, 30)
    ax = plt.gca()
    for i, (geo, spacing) in enumerate(configs):
        if spacing is None:
            kwargs = dict(color="k")
        else:
            kwargs = dict(label="%s %dm" % (geo, spacing))
        surface = surfaces.get_fiducial_surface(geo, spacing)
        area = np.array(
            [surface.average_area(lo, hi) for lo, hi in zip(ct[:-1], ct[1:])]
        )
        ax.plot(*plotting.stepped_path(ct, area / 1e6), **kwargs)
    ax.legend(loc="best")
    ax.set_ylim((0, 15))
    ax.set_ylabel("Average area [km$^2$]")
    ax.set_xlabel(r"$\cos\theta$")

    plt.tight_layout()
    plt.show()
