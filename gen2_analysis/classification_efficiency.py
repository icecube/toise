"""
Energy-dependent event classification efficiencies, e.g. for separating
contained-vertex events into single cascades, starting tracks, and double cascades.
"""

import numpy as np
from .util import data_dir
import json
import os


def get_classification_efficiency(geometry="IceCube", spacing=125):
    if geometry == "Potemkin":
        return ClassificationEfficiency(
            {
                "nue": {
                    "cascades": ["logpoly", [97.61798482, -0.92491068]],
                    "starting_tracks": [
                        "logpoly",
                        [12.45575941, -4.52083962, 0.89284789, -0.05660718],
                    ],
                    "double_cascades": ["powerlaw", [0.29980095, 0.36109473]],
                },
                "numu": {
                    "cascades": ["logpoly", [89.37025688, -17.82727776, 1.44100813]],
                    "starting_tracks": [
                        "logpoly",
                        [-12.02248755, 29.98400471, -3.72778076, 0.13724512],
                    ],
                    "double_cascades": ["powerlaw", [0.86190289, 0.14669257]],
                },
                "nutau": {
                    "cascades": [
                        "sigmoid",
                        [91.90034143, 0.79907467, 4.76995429, 36.0535759, 1.54379368],
                    ],
                    "starting_tracks": [
                        "logpoly",
                        [
                            -353.501682,
                            251.751001,
                            -64.2850564,
                            7.16802798,
                            -0.292552097,
                        ],
                    ],
                    "double_cascades": [
                        "sigmoid",
                        [0.0662404, 0.86871992, 5.03507898, 42.57592219, 1.75695386],
                    ],
                },
            },
            (6e4, 1e7),
        )
    elif geometry == "IceCube":
        return ClassificationEfficiency.load(
            "icecube_doublecascade_efficiency.json", (6e4, 1e7)
        )
    elif 240 <= spacing <= 260:
        return ClassificationEfficiency.load(
            "sparsecube_doublecascade_efficiency.json", (6e4, 1e7)
        )


class ClassificationEfficiency(object):
    @staticmethod
    def powerlaw(x, a, b):
        """
        :param x: energy in GeV
        :returns: classification efficiency in percent
        """
        return a * (x / 1e3) ** b

    @staticmethod
    def logpoly(x, *params):
        """
        :param x: energy in GeV
        :returns: classification efficiency in percent
        """
        return np.polyval(params[::-1], np.log(x / 1e3))

    @staticmethod
    def sigmoid(x, a, b, c, d, m):
        """
        General sigmoid function
        a adjusts amplitude
        b adjusts y offset
        c adjusts x offset
        d adjusts slope

        :param x: energy in GeV
        :returns: classification efficiency in percent
        """
        return ((a - d) / (1 + (np.exp(b * np.log(x / 1e3) - c)) ** m)) + d

    @classmethod
    def load(cls, filename, energy_range=(0, np.inf)):
        if not filename.startswith("/"):
            filename = os.path.join(data_dir, "selection_efficiency", filename)
        with open(filename) as f:
            return cls(params, energy_range)

    def __init__(self, params, energy_range=(0, np.inf)):
        self._params = params
        classes = set()
        for item in self._params.values():
            for channel, values in item.items():
                funcname = values[0]
                values[0] = getattr(self, funcname)
                classes.add(channel)
        self._energy_range = energy_range
        self.classes = classes

    def __call__(self, neutrino_flavor, event_class, deposited_energy):
        """
        :param neutrino_flavor: flavor index (0-5)
        :param event_class: final event signature
        :param deposited_energy: deposited energy in GeV
        """
        if not isinstance(neutrino_flavor, int):
            raise TypeError("neutrino_flavor must be an integer")
        elif neutrino_flavor < 0 or neutrino_flavor > 5:
            raise ValueError("neutrino_flavor must be 0 <= nu < 6")
        if not event_class in self.classes:
            raise ValueError("Unknown event class {}".format(event_class))
        flavor = ["nue", "numu", "nutau"][neutrino_flavor // 2]
        func, params = self._params[flavor][event_class]
        x = np.clip(deposited_energy, *self._energy_range)
        return np.clip(func(x, *params) / 100, 0, 1)
