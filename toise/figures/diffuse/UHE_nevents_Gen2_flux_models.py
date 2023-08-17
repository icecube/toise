#!/usr/bin/env python
# coding: utf-8

# # Flux model plot for Gen2

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from toise.diffuse import DiffuseModel
from toise import factory
from toise.util import center
from toise import diffuse

# get enums for models ...
from toise.diffuse import DIFFUSE_MODELS
from toise.figures import figure_data, figure

from enum import Enum

IC_DIFFUSE = Enum("IC_DIFFUSE", ["HESE", "NUMU"])


livetime = 10  # *5/4. scaling for 5 vs 4 bins per decade when comparing to Bustamante/Valera plot


# color definition by Markus

blue = "#2078b4"  #'#009fdf'
orange = "#ff7f0e"  #'#f18f1f'
gray = "#566573"  #'#888888'
lightgray = "#AAAABB"
green = "#2ca02c"  #'#009900'
darkblue = "#000060"
purple = "#9467bd"


@figure
def nevents_uhe_model():
    # get detector configurations from factory
    optical = factory.get("Gen2-InIce")
    radio = factory.get("Gen2-Radio")

    eventclasses = {}
    for det in [optical, radio]:
        for key in det.keys():
            eventclasses[key] = det[key][0]

    # calculate  DiffuseModel numbers

    optical_edges = optical["shadowed_tracks"][0].bin_edges[0]
    radio_edges = radio["radio_events"][0].bin_edges[0]
    edges = {}
    for ev in eventclasses:
        if ev == "radio_events":
            edges[ev] = radio_edges
        else:
            edges[ev] = optical[ev][0].bin_edges[0]
    np.log10(optical["shadowed_tracks"][0].bin_edges[0]), np.log10(
        radio["radio_events"][0].bin_edges[0]
    )
    energies = np.logspace(1, 12, 111)

    def get_bins(energies, key, edges=edges):
        return np.isin(
            np.round(np.log10(energies[:-1]), 2), np.round(np.log10(edges[key][:-1]), 2)
        )

    # get all flux model event numbers as function of energy
    plot_models = {}
    plot_models_radio = {}
    plot_models_optical = {}

    for fluxmodel in DIFFUSE_MODELS:
        print(fluxmodel.__doc__)
        print("---------")
        nev = np.zeros_like(energies[:-1])
        nev_radio = np.zeros_like(energies[:-1])
        nev_optical = np.zeros_like(energies[:-1])
        for ev_class in eventclasses:  # ["radio_events"]: #eventclasses:
            model = DiffuseModel(fluxmodel, eventclasses[ev_class], livetime)
            print(ev_class, model.expectations().sum())
            nev[get_bins(energies, ev_class)] += model.expectations().sum(0)
            if ev_class == "radio_events":
                nev_radio[get_bins(energies, ev_class)] += model.expectations().sum(0)
            else:
                nev_optical[get_bins(energies, ev_class)] += model.expectations().sum(0)
        plot_models[fluxmodel] = nev
        plot_models_radio[fluxmodel] = nev_radio
        plot_models_optical[fluxmodel] = nev_optical
        print("---------")
        print("---------\n")

    # atm muon background (only in radio, optical is background free at these energies)
    from toise.radio_aeff_generation import MuonBackground

    radio_background = MuonBackground(
        radio["radio_events"][1], livetime
    ).expectations.sum(0)

    # # calculate IceCube diffuse extrapolation numbers

    hese = np.zeros_like(energies[:-1])
    hese_radio = np.zeros_like(energies[:-1])
    hese_optical = np.zeros_like(energies[:-1])
    for ev_class in eventclasses:
        diff = diffuse.DiffuseAstro(eventclasses[ev_class], livetime)
        # HESE 7.5y
        norm = 6.37 / 3
        events_i = norm * diff.expectations(gamma=-2.87).sum(0)
        hese[get_bins(energies, ev_class)] += events_i
        if ev_class == "radio_events":
            hese_radio[get_bins(energies, ev_class)] += events_i
        else:
            hese_optical[get_bins(energies, ev_class)] += events_i

    plot_models[IC_DIFFUSE.HESE] = hese
    plot_models_optical[IC_DIFFUSE.HESE] = hese_optical
    plot_models_radio[IC_DIFFUSE.HESE] = hese_radio

    numu = np.zeros_like(energies[:-1])
    numu_radio = np.zeros_like(energies[:-1])
    numu_optical = np.zeros_like(energies[:-1])
    for ev_class in eventclasses:
        diff = diffuse.DiffuseAstro(eventclasses[ev_class], livetime)
        # NuMu 9.5y
        norm = 1.44
        events_i = norm * diff.expectations(gamma=-2.37).sum(0)
        numu[get_bins(energies, ev_class)] += events_i
        if ev_class == "radio_events":
            numu_radio[get_bins(energies, ev_class)] += events_i
        else:
            numu_optical[get_bins(energies, ev_class)] += events_i

    plot_models[IC_DIFFUSE.NUMU] = numu
    plot_models_optical[IC_DIFFUSE.NUMU] = numu_optical
    plot_models_radio[IC_DIFFUSE.NUMU] = numu_radio

    # get plot titles, add them to the enum for convenience

    DIFFUSE_MODELS.heinze_zmax_1.plot_title = (
        r"Heinze et al.," "\n" "fit to Auger UHECRs"
    )
    DIFFUSE_MODELS.van_vliet_ta.plot_title = "Bergman & van Vliet, fit to\nTA UHECRs"
    DIFFUSE_MODELS.rodrigues_bench_cosmo.plot_title = (
        r"Rodrigues et al.," "\n" r"all AGN (cosmogenic $\nu$)"
    )
    DIFFUSE_MODELS.rodrigues_bench_source.plot_title = (
        r"Rodrigues et al.," "\n" r"all AGN (source $\nu$)"
    )
    DIFFUSE_MODELS.rodrigues_hlbllacs_cosmo.plot_title = (
        r"Rodrigues et al.," "\n" "HL BL Lacs"
    )
    DIFFUSE_MODELS.fang_murase.plot_title = "Fang & Murase,\ncosmic-ray reservoirs"
    DIFFUSE_MODELS.fang_pulsar.plot_title = "Fang et al.,\nnewborn pulsars SFR"
    DIFFUSE_MODELS.padovani_2015.plot_title = "Padovani et al.," "\n" r"BL Lacs"
    DIFFUSE_MODELS.muzio_2019.plot_title = "Muzio et al.,\nmaximum extra $p$ component"
    DIFFUSE_MODELS.muzio_2021.plot_title = "Muzio et al.,\nCR-gas interactions"

    IC_DIFFUSE.HESE.plot_title = "IceCube cascades\n(7.5 yr) extrapolated"
    IC_DIFFUSE.NUMU.plot_title = r"IceCube tracks" "\n" "(9.5 yr) extrapolated"

    # define order in which to fill panels

    model_order = [
        IC_DIFFUSE.HESE,
        IC_DIFFUSE.NUMU,
        DIFFUSE_MODELS.heinze_zmax_1,
        DIFFUSE_MODELS.van_vliet_ta,
        DIFFUSE_MODELS.rodrigues_bench_cosmo,
        DIFFUSE_MODELS.rodrigues_bench_source,
        DIFFUSE_MODELS.rodrigues_hlbllacs_cosmo,
        DIFFUSE_MODELS.fang_murase,
        DIFFUSE_MODELS.fang_pulsar,
        DIFFUSE_MODELS.padovani_2015,
        DIFFUSE_MODELS.muzio_2019,
        DIFFUSE_MODELS.muzio_2021,
    ]

    # # Plot result plot (fig 25 in TDR)

    ax_list = [*product([*range(3)], [*range(4)])]
    fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 9))
    nbins_per_decade = 2
    pltE = np.logspace(1, 12, 11 * nbins_per_decade + 1)

    """
    for i, m in enumerate(model_order):
        print(m)
        label_c, label_o, label_r, label_1ev = None, None, None, None
        if i == 2:
            label_c = "IceCube-Gen2"
            label_o = "optical only"
            label_r = "radio only"
            label_1ev = "1 event per bin"
        ax[ax_list[i]].hist(10**center(np.log10(energies)), weights=plot_models[m], bins=pltE, alpha=0.4, histtype="stepfilled", linestyle="-", linewidth=2, label=label_c)
        ax[ax_list[i]].hist(10**center(np.log10(energies)), weights=plot_models[m], bins=pltE, color="black", linestyle="-", linewidth=1, histtype="step")
        ax[ax_list[i]].hist(10**center(np.log10(energies)), weights=plot_models_radio[m], bins=pltE, color="tab:orange", linestyle=":", linewidth=2, histtype="step", label=label_r)
        ax[ax_list[i]].hist(10**center(np.log10(energies)), weights=plot_models_optical[m], bins=pltE, color="tab:blue", linestyle="--", linewidth=2, histtype="step", label=label_o)
        ax[ax_list[i]].loglog()
        ax[ax_list[i]].set_ylim(1e-2, 1e2)
        ax[ax_list[i]].set_xlim(1e4, 1e10)
        ax[ax_list[i]].grid()
        ax[ax_list[i]].text(2e4,2e-2, m.plot_title, ha='left', va='bottom')
        
        ax[ax_list[i]].axhline(1, color="red", linestyle="-", label=label_1ev)
        #ax[ax_list[i]].text(10**(0.5*(max(np.log10(energies))+min(np.log10(energies))/2)),1, ha="center", va="center")
        fig.legend(loc=9, ncol=4)
        #ax[ax_list[i]].legend()
    fig.supxlabel(r"Reconstructed shower energy, $E_\mathrm{sh}^\mathrm{rec}$ [GeV]")
    fig.supylabel(r"Number of $\nu$ events in IceCube-Gen2 ("+str(livetime)+" yr)")
    fig.suptitle(f'')
    fig.tight_layout()
    plt.savefig("UHE_model_Nevents_Gen2.png")
    plt.savefig("UHE_model_Nevents_Gen2.pdf")
    """

    # # Plot same, but focus on radio-energy range

    # matplotlib style params by Markus
    params = {
        #'legend.fontsize': 15,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.weight": "regular",
        "figure.figsize": (12, 12),
        "axes.labelsize": 10,
        "axes.titlesize": 30,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "mathtext.default": "regular",
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.formatter.use_mathtext": True,
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 9))
    nbins_per_decade = 5
    pltE = np.logspace(1, 12, 11 * nbins_per_decade + 1)
    plt.subplots_adjust(
        left=0.1, bottom=0.12, right=0.999, top=0.93, wspace=0.04, hspace=0.04
    )

    for i, m in enumerate(model_order):
        # print(m)
        label_c, label_o, label_r, label_1ev = None, None, None, None
        if i == 2:
            label_c = "IceCube-Gen2"
            label_o = "optical only"
            label_r = "radio only"
            label_1ev = "1 event per bin"
        ax[ax_list[i]].hist(
            10 ** center(np.log10(energies)),
            weights=plot_models[m],
            bins=pltE,
            alpha=0.4,
            histtype="stepfilled",
            linestyle="-",
            color=blue,
            linewidth=2,
            label=label_c,
        )
        ax[ax_list[i]].hist(
            10 ** center(np.log10(energies)),
            weights=plot_models[m],
            bins=pltE,
            color="black",
            linestyle="-",
            linewidth=1,
            histtype="step",
        )
        ax[ax_list[i]].hist(
            10 ** center(np.log10(energies)),
            weights=plot_models_radio[m],
            bins=pltE,
            color=orange,
            linestyle=":",
            linewidth=2,
            histtype="step",
            label=label_r,
        )
        ax[ax_list[i]].hist(
            10 ** center(np.log10(energies)),
            weights=plot_models_optical[m],
            bins=pltE,
            color=blue,
            linestyle="--",
            linewidth=2,
            histtype="step",
            label=label_o,
        )
        ax[ax_list[i]].loglog()
        ax[ax_list[i]].set_ylim(1e-2, 1e2)
        ax[ax_list[i]].set_xlim(1e7, 1e10)
        ax[ax_list[i]].grid()
        ax[ax_list[i]].set_yticks([1e-2, 1e-1, 1e0, 1e1])
        if i == 11:
            ax[ax_list[i]].set_xticks([1e7, 1e8, 1e9])
        else:
            ax[ax_list[i]].set_xticks([1e7, 1e8, 1e9])
        ax[ax_list[i]].axhline(1, color="red", linestyle="-", label=label_1ev)
        ax[ax_list[i]].text(
            1.2e7, 0.8e2, m.plot_title, ha="left", va="top", fontsize=14
        )
        ax[ax_list[i]].tick_params(axis="x", pad=8)
        # ax[ax_list[i]].legend()
    fig.legend(loc=9, ncol=4, fontsize=16)
    fig.supxlabel(
        r"Reconstructed shower energy, $E_\mathrm{sh}^\mathrm{rec}$ [GeV]", fontsize=22
    )
    fig.supylabel(
        r"Number of $\nu$ events in IceCube-Gen2 (" + str(livetime) + " yr)",
        fontsize=22,
    )
    fig.suptitle(f"")
    # fig.tight_layout()
    #    plt.savefig("UHE_model_Nevents_Gen2_radioE.png")
    #    plt.savefig("UHE_model_Nevents_Gen2_radioE.pdf")
    return fig


########################
# Time to 5 sigma plot #
########################


@figure
def t5sigma_uhe_models():
    def llh(n_s, n_b, zerosignal=False):
        # no idea how to get the asimov LLH without the point-source backgrounds for diffuse,
        # this should be a straigt copy of what toise.multillh.asimov_llh.llh does
        data = n_s + n_b
        if zerosignal:
            lamb = n_b
        else:
            lamb = n_b + n_s
        with np.errstate(divide="ignore"):
            log_lamb = np.log(lamb)
            log_data = np.log(data)
        log_lamb = np.nan_to_num(log_lamb, neginf=0, posinf=0)
        log_data = np.nan_to_num(log_data, neginf=0, posinf=0)
        # print(log_lamb, log_data, lamb, data, np.sum(data*(log_lamb-log_data)) - np.sum(lamb - data))
        return np.sum(data * (log_lamb - log_data)) - np.sum(lamb - data)

    def n_sigma(n_s, n_b):
        # n sigma = (-2 Delta LLH)**0.5
        return (-2 * (llh(n_s, n_b, zerosignal=True) - llh(n_s, n_b))) ** 0.5

    def T_5sigma(nsignal, nbackground, livetime=10):
        #
        nyears = np.linspace(0, 100, 1000)
        sigmas = []
        for ny in nyears:
            sigmas.append(n_sigma(nsignal * ny / livetime, nbackground * ny / livetime))
        return np.interp(5, sigmas, nyears)

    plt.figure()

    for i, m in enumerate(model_order[::-1]):
        n_signal = plot_models[m][10 ** center(np.log10(energies)) > 1e7]
        # print(m, nsignal, T_5sigma(n_signal, 0))
        ny = float(
            T_5sigma(np.array([n_signal]), np.array(np.full(np.shape(n_signal), 0.3)))
        )
        hatch = ""
        color = blue
        if ny > 10:
            hatch = "/"
            color = "lightgrey"
        label = m.plot_title
        # for this plot, redefine the plot title to break after authors
        if m == DIFFUSE_MODELS.van_vliet_ta:
            label = "Bergman & van Vliet,\nfit to TA UHECRs"
        plt.barh(str(m.plot_title), ny, color=color, hatch=hatch)

    # Add padding between axes and labels
    ax = plt.gca()
    ax.xaxis.set_tick_params(pad=8)
    ax.yaxis.set_tick_params(pad=10)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Add x, y gridlines
    ax.grid(axis="x", color="grey", linestyle="-.", linewidth=1, alpha=0.5)

    # Remove axes splines
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    plt.xlim(0, 10)
    plt.xlabel("years to $5\sigma_{50\%}$ discovery", fontsize=24)
    plt.tight_layout()
    #    plt.savefig("UHE_model_time_to_5_sigma.pdf")
    #    plt.savefig("UHE_model_time_to_5_sigma.png")
    return ax.figure
