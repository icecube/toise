import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from gen2_analysis import plotting, effective_areas

configs = [
    ("IceCube", 125.0),
    # ('EdgeWeighted', 240),
    ("Sunflower", 200),
    ("Sunflower", 240),
    ("Sunflower", 300),
]

with plotting.pretty(tex=False):
    e = np.logspace(2, 9, 101)
    ax = plt.gca()
    for i, (geo, spacing) in enumerate(configs):
        seleff = effective_areas.HESEishSelectionEfficiency(geo, spacing, 1e5)

        if geo == "IceCube":
            kwargs = dict(color="k", label=geo)
        else:
            kwargs = dict(label="%s %dm" % (geo, spacing))

        fiducial_volumes = (
            seleff(
                e,
                0.0,
            )
            * seleff._outer_volume
            / 1e9
        )
        saturated_fiducial_volume = np.max(fiducial_volumes)
        print(
            (
                "Geom {}_{} volumes. saturated fiducial: {:.2f} km^3, max geometric: {:.2f} km^3".format(
                    geo, spacing, saturated_fiducial_volume, seleff._outer_volume / 1e9
                )
            )
        )

        ax.semilogx(e, fiducial_volumes, **kwargs)
        ax.legend(
            frameon=True, framealpha=0.8, loc="upper left"
        ).get_frame().set_linewidth(0)
    ax.set_ylabel("Fiducial volume [km$^3$]")
    for ax in [ax]:
        # ax.set_ylim((0, 1))
        ax.set_xlabel(r"$E_{\rm visible}$ [GeV]")
        ax.grid()
    plt.tight_layout()
    plt.show()
