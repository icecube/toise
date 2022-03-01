import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from icecube.toise import plotting, effective_areas

configs = [
    ("IceCube", 125.0),
    # ('EdgeWeighted', 240),
    ("Sunflower", 200),
    ("Sunflower", 240),
    ("Sunflower", 300),
]

with plotting.pretty():
    fig = plt.figure(figsize=(8, 4))
    griddy = GridSpec(1, 2)
    emu = np.logspace(2, 6, 101)
    ax1, ax2 = (plt.subplot(griddy[i]) for i in range(2))
    for i, (geo, spacing) in enumerate(configs):
        mueff = effective_areas.get_muon_selection_efficiency(geo, spacing)

        if geo == "IceCube":
            kwargs = dict(color="k", label=geo)
        else:
            kwargs = dict(label="%s %dm" % (geo, spacing))

        ax1.semilogx(emu, mueff(emu, 0), **kwargs)
        ax2.semilogx(emu, mueff(emu, -0.5), **kwargs)

    ax1.add_artist(AnchoredText(r"$\cos\theta=0$", loc=4, frameon=False))
    ax2.add_artist(AnchoredText(r"$\cos\theta=-0.5$", loc=4, frameon=False))

    ax1.legend(
        frameon=True, framealpha=0.8, loc="upper left"
    ).get_frame().set_linewidth(0)
    ax1.set_ylabel("Selection efficiency")
    for ax in fig.axes:
        ax.set_ylim((0, 1))
        ax.set_xlabel(r"$E_{\mu}$ [GeV]")
        ax.grid()
    plt.tight_layout()
    plt.show()
