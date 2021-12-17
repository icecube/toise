from toise.figures import figure, figure_data
from toise.figures.diffuse.flavor import psi_binning

import dashi
import numpy as np


def sparse_hese_veto_plot(ax, f):
    npe = 10 ** f["arr_0"]

    def fit(y, yerr, range=None):
        mask = yerr != 0
        xmask = np.ones(npe.size, dtype=bool)
        if range is not None:
            xmask &= npe > range[0]
            xmask &= npe < range[1]
        mask &= xmask
        model = dashi.fitting.powerlaw()
        model.limits["index"] = (-10, -2)
        fit = dashi.fitting.leastsq(npe[mask], y[mask], error=yerr[mask], model=model)
        return npe[xmask], fit(npe[xmask])

    style = dict(
        ls="None", markeredgecolor="None", markersize=5, capsize=0.0, linewidth=1
    )
    ax.axhline(1, color="grey", lw=0.5)
    ax.errorbar(
        npe, f["arr_1"], f["arr_4"], label="No veto", marker="o", color="k", **style
    )

    line = ax.errorbar(
        npe, f["arr_3"], f["arr_6"], label="125 m veto", marker="^", **style
    )[0]
    ax.plot(
        *fit(f["arr_3"], f["arr_6"], (10 ** 3.48, 1e4)), color=line.get_color(), ls="--"
    )

    line = ax.errorbar(
        npe, f["arr_2"], f["arr_5"], label="250 m veto", marker="s", **style
    )[0]
    ax.plot(
        *fit(f["arr_2"], f["arr_5"], (10 ** 3.5, 10 ** 4.7)),
        color=line.get_color(),
        ls="--"
    )

    ax.loglog(nonposy="clip")
    ax.set_ylim((0.5, 1e6))
    ax.set_ylabel("Atmospheric muon events per year")
    ax.set_xlabel("Number of collected photons in IceCube")

    ax.legend(numpoints=1)


@figure
def volume():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from toise import factory, effective_areas, surfaces, plotting

    edep = np.logspace(4, 8, 101)
    fig = plt.figure(figsize=(3.375, 3.375))
    ax = plt.gca()

    pretty_labels = {"Gen2-InIce": "Gen2-Optical"}

    for k in "IceCube", "Gen2-InIce":
        config = factory.default_configs[k]
        efficiency = effective_areas.HECascadeSelectionEfficiency(
            config["geometry"], config["spacing"], config["cascade_energy_threshold"]
        )
        surface = effective_areas.get_fiducial_surface(
            config["geometry"], config["spacing"]
        )
        ax.plot(
            edep,
            efficiency(edep, 0) * surface.volume() / 1e9,
            label=pretty_labels.get(k, k),
        )
        print(
            (
                "Saturated veff is {}".format(
                    np.max(efficiency(edep, 0) * surface.volume() / 1e9)
                )
            )
        )

    ax.semilogx()
    ax.set_xlabel("Deposited energy (GeV)")
    ax.set_ylabel("Effective volume (km$^3$)")
    ax.legend()
    plt.tight_layout()
    return fig


@figure_data(setup=psi_binning)
def effective_area(exposures):
    from toise import factory

    assert len(exposures) == 1
    aeff = factory.get(exposures[0][0])["cascades"][0]
    area = aeff.values.sum(axis=(-2, -1))
    meta = {
        "energy": aeff.get_bin_edges("true_energy").tolist(),
        "cos_zenith": aeff.get_bin_edges("true_zenith_band").tolist(),
        "area": area.tolist(),
    }
    return meta


@figure
def effective_area(datasets):
    import matplotlib.pyplot as plt
    import os
    from toise import plotting

    fig = plt.figure(figsize=(3.375, 3.375))
    ax = plt.gca()

    pretty_labels = {"Gen2-InIce": "Gen2-Optical"}

    for dataset in datasets:
        assert (
            dataset["source"]
            == "toise.figures.detector.cascades.effective_area"
        )
        assert len(dataset["detectors"]) == 1
        energy = np.asarray(dataset["data"]["energy"])
        area = np.asarray(dataset["data"]["area"]).mean(axis=(0, 2))
        k = dataset["detectors"][0][0]
        ax.loglog(*plotting.stepped_path(energy, area), label=pretty_labels.get(k, k))

    ax.set_xlabel("Neutrino energy (GeV)")
    ax.set_ylabel("Neutrino effective area (m$^2$)")
    ax.set_xlim(1e4, 5e8)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


@figure
def sparse_veto():
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig = plt.figure(figsize=(3.375, 3.375))
    ax = plt.gca()

    f = np.load(os.path.join(os.path.dirname(__file__), "OfficialPlot_Veto_Data.npz"))
    npe = 10 ** f["arr_0"]

    def fit(y, yerr, range=None):
        mask = yerr != 0
        xmask = np.ones(npe.size, dtype=bool)
        if range is not None:
            xmask &= npe > range[0]
            xmask &= npe < range[1]
        mask &= xmask
        model = dashi.fitting.powerlaw()
        model.limits["index"] = (-10, -2)
        fit = dashi.fitting.leastsq(npe[mask], y[mask], error=yerr[mask], model=model)
        return npe[xmask], fit(npe[xmask])

    style = dict(
        ls="None", markeredgecolor="None", markersize=5, capsize=0.0, linewidth=1
    )
    ax.axhline(1, color="grey", lw=0.5)
    ax.errorbar(
        npe, f["arr_1"], f["arr_4"], label="No veto", marker="o", color="k", **style
    )

    line = ax.errorbar(
        npe, f["arr_3"], f["arr_6"], label="Single-spaced veto", marker="^", **style
    )[0]
    ax.plot(
        *fit(f["arr_3"], f["arr_6"], (10 ** 3.48, 1e4)), color=line.get_color(), ls="--"
    )

    line = ax.errorbar(
        npe, f["arr_2"], f["arr_5"], label="Double-spaced veto", marker="s", **style
    )[0]
    x, y = fit(f["arr_2"], f["arr_5"], (10 ** 3.5, 10 ** 4.7))
    ax.plot(x, y, color=line.get_color(), ls="--")
    # ax.plot(x/3, y, color=line.get_color(), ls='-')

    ax.loglog(nonposy="clip")
    ax.set_ylim((0.5, 1e6))
    ax.set_ylabel("Atmospheric muon events per year")
    ax.set_xlabel("Number of collected photons in IceCube")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(numpoints=1, frameon=True)
    plt.tight_layout()

    return fig
