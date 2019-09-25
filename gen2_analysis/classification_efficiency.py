
"""
Energy-dependent event classification efficiencies, e.g. for separating
contained-vertex events into single cascades, starting tracks, and double cascades.
"""

import numpy as np
from . util import data_dir
import json
import os


def get_classification_efficiency(geometry='IceCube', spacing=125):
    if geometry == 'IceCube':
        return ClassificationEfficiency('icecube_doublecascade_efficiency.json', (6e4, 1e7))
    elif 240 <= spacing <= 260:
        return ClassificationEfficiency('sparsecube_doublecascade_efficiency.json', (6e4, 1e7))


class ClassificationEfficiency(object):
    @staticmethod
    def powerlaw(x, a, b):
        """
        :param x: energy in GeV
        :returns: classification efficiency in percent
        """
        return a*(x/1e3)**b

    @staticmethod
    def logpoly(x, *params):
        """
        :param x: energy in GeV
        :returns: classification efficiency in percent
        """
        return np.polyval(params[::-1], np.log(x/1e3))

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
        return ((a-d) / (1 + (np.exp(b*np.log(x/1e3)-c))**m)) + d

    def __init__(self, filename, energy_range=(0, np.inf)):
        if not filename.startswith('/'):
            filename = os.path.join(data_dir, 'selection_efficiency', filename)
        with open(filename) as f:
            self._params = json.load(f)
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
        flavor = ['nue', 'numu', 'nutau'][neutrino_flavor/2]
        func, params = self._params[flavor][event_class]
        x = np.clip(deposited_energy, *self._energy_range)
        return np.clip(func(x, *params)/100, 0, 1)
