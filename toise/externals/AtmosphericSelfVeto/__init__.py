import numpy as np

from . import selfveto
from ...util import PDGCode, data_dir


class AnalyticPassingFraction(object):
    """
    A combination of the Schoenert et al calculation and an approximate treatment of uncorrelated muons from the rest of the shower.
    """

    _cache = {}

    def __init__(self, kind="conventional", veto_threshold=1e3, floor=1e-4):
        """
        :param kind: either 'conventional' for neutrinos from pion/kaon decay
                     or 'charm' for neutrinos from charmed meson decay
        :param veto_threshold: energy at depth where a single muon is guaranteed
                               to be rejetected by veto cuts [GeV]
        :param floor: minimum passing fraction (helpful for avoiding numerical
                      problems in likelihood functions)
        """
        self.kind = kind
        self.veto_threshold = veto_threshold
        self.floor = floor
        self.ct_min = 0.05

        self._splines = dict()
        if kind == "conventional":
            numu = self._get_spline("numu", veto_threshold)
            nue = self._get_spline("nue", veto_threshold)
        elif kind == "charm":
            numu = self._get_spline("charm", veto_threshold)
            nue = numu

        self._eval = dict()
        self._eval[PDGCode.NuMu] = np.vectorize(
            lambda enu, ct, depth: numu.evaluate_simple([enu, ct, depth])
        )
        self._eval[PDGCode.NuE] = np.vectorize(
            lambda enu, ct, depth: nue.evaluate_simple([enu, ct, depth])
        )

    def _eval_grid(self, kind, veto_threshold):
        def pad_knots(knots, order=2):
            """
            Pad knots out for full support at the boundaries
            """
            pre = knots[0] - (knots[1] - knots[0]) * np.arange(order, 0, -1)
            post = knots[-1] + (knots[-1] - knots[-2]) * np.arange(1, order + 1)
            return np.concatenate((pre, knots, post))

        def edges(centers):
            dx = np.diff(centers)[0] / 2.0
            return np.concatenate((centers - dx, [centers[-1] + dx]))

        log_enu, ct = np.linspace(1, 9, 51), np.linspace(self.ct_min, 1, 21)
        depth = np.linspace(1e3, 3e3, 11)
        depth_g = depth[None, None, :]
        log_enu_g, ct_g = list(map(np.transpose, np.meshgrid(log_enu, ct)))

        pr = np.zeros(ct_g.shape + (depth.size,))
        for i, d in enumerate(depth):
            slant = selfveto.overburden(ct_g, d)
            emu = selfveto.minimum_muon_energy(slant, veto_threshold)
            pr[..., i] = selfveto.uncorrelated_passing_rate(
                10**log_enu_g, emu, ct_g, kind=kind
            )

        centers = [log_enu, ct, depth]
        knots = list(map(pad_knots, list(map(edges, centers))))

        ndim = pr.ndim
        return pr, centers, knots, [2] * ndim, [1e-16] * ndim, [2] * ndim

    def _create_spline(self, kind, veto_threshold):
        import photospline

        pr, centers, knots, order, penalty, penorder = self._eval_grid(
            kind, veto_threshold
        )
        z, w = photospline.ndsparse.from_data(pr, np.ones(pr.shape))
        spline = photospline.glam_fit(z, w, centers, knots, order, penalty, penorder)
        return spline

    def _get_spline(self, kind, veto_threshold):
        """
        Parameterize the uncorrelated veto probability as a function of
        neutrino energy, zenith angle, and vertical depth, and cache the result.
        """
        key = (kind, veto_threshold)
        if not key in self._cache:
            self._cache[key] = self._create_spline(kind, veto_threshold)
        return self._cache[key]

    def __call__(self, particleType, enu, ct, depth, spline=True):
        """
        Estimate the fraction of atmospheric neutrinos that will arrive without
        accompanying muons from the same air shower.

        :param particleType: neutrino type for which to evaluate the veto
        :type particleType: PDGCode
        :param enu: neutrino energy [GeV]
        :param ct: cosine of the zenith angle
        :param depth: vertical depth [m]
        :param spline: if False, evaluate the uncorrelated veto probability
                       directly. Otherwise, use the much faster B-spline
                       representation.
        """
        if np.isscalar(ct) and not ct > self.ct_min:
            return np.array(1.0)
        emu = selfveto.minimum_muon_energy(
            selfveto.overburden(ct, depth), self.veto_threshold
        )

        # Verify that we're using a sane encoding scheme
        assert abs(PDGCode.NuMuBar) == PDGCode.NuMu
        particleType = abs(np.asarray(particleType))
        if spline:
            pr = np.where(
                particleType == PDGCode.NuMu,
                self._eval[PDGCode.NuMu](np.log10(enu), ct, depth),
                np.where(
                    particleType == PDGCode.NuE,
                    self._eval[PDGCode.NuE](np.log10(enu), ct, depth),
                    1,
                ),
            )
        else:
            enu, ct, depth = np.broadcast_arrays(enu, ct, depth)
            if self.kind == "conventional":
                pr = np.where(
                    particleType == PDGCode.NuMu,
                    selfveto.uncorrelated_passing_rate(enu, emu, ct, kind="numu"),
                    np.where(
                        particleType == PDGCode.NuE,
                        selfveto.uncorrelated_passing_rate(enu, emu, ct, kind="nue"),
                        1,
                    ),
                )
            elif self.kind == "charm":
                pr = selfveto.uncorrelated_passing_rate(enu, emu, ct, kind=self.kind)

        # For NuMu specifically there is a guaranteed accompanying muon.
        # Estimate the passing fraction from the fraction of the decay phase
        # space where the muon has too little energy to make it to depth.
        # NB: strictly speaking this calculation applies only to 2-body
        # decays of pions and kaons, but is at least a conservative estimate
        # for 3-body decays of D mesons.
        direct = selfveto.correlated_passing_rate(enu, emu, ct)
        pr *= np.where(particleType == PDGCode.NuMu, direct, 1)

        return np.where(
            ct > self.ct_min,
            np.where(pr <= 1, np.where(pr >= self.floor, pr, self.floor), 1),
            1,
        )
