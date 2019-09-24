
import numpy as np
from toolz import memoize
from itertools import product
from .crosssections import DISCrossSection, GlashowResonanceCrossSection
from .earth import get_t_earth

class NeutrinoCascade(object):
    """
    Propagate a neutrino flux through the Earth using the method described in
    the nuFATE_ paper.

    .. _nuFATE: https://arxiv.org/pdf/1706.09895.pdf
    """
    def __init__(self, energy_nodes):
        """
        :param energy_nodes: logarithmically spaced grid of energies (in GeV)
            where the neutrino flux will be evaluated
        """
        assert energy_nodes.ndim == 1
        self.energy_nodes = energy_nodes
        # find logarithmic distance between nodes
        dloge = (np.log(self.energy_nodes[-1])-np.log(self.energy_nodes[0]))/(len(self.energy_nodes)-1)
        # ratio between the interval centered on the node energy and the node energy itself
        self._width = 2*np.sinh(dloge/2.)
        # Comparing with NuFate paper: multiply by E_j (= E_in) to account
        # for log scale, then by E_i^2/E_j^2 to account for variable change
        # phi -> E^2*phi
        ei, ej = np.meshgrid(energy_nodes, energy_nodes)
        self.differential_element = 2*dloge*(ei**2/ej)

    def transfer_matrix_element(self, i, flavor, out_flavor, column_density):
        """
        Calculate an element of the transfer matrix, i.e. the fraction of
        neutrions of type `flavor` at energy node `i` that is transferred to
        each other energy node after propagating through `column_density` of
        scattering centers.
        
        :param i: index of energy node for which to calculate transmission probability
        :param flavor: index of neutrino type
        :param out_flavor: index of outgoing neutrino type
        :param column_density: number density of scattering centers in cm^2
        :returns: an array of shape (column_density,energy_nodes)
        """
        # construct a differential flux that is nonzero in only 1 energy bin
        # and integrates to 1
        flux0 = 1./(self._width*self.energy_nodes[i])
        flux = np.where(np.arange(self.energy_nodes.size)==i, flux0, 0)
        if out_flavor != flavor:
            # initial flux of nue/numu is zero
            flux = np.concatenate([np.zeros_like(flux), flux])

        # decompose flux in the eigenbasis of cascade equation solution
        w,v,ci = self.decompose_in_eigenbasis(flux, flavor, out_flavor)

        # attenuate components
        wf = np.exp(w[...,None]*np.asarray(column_density)[None,...])
        # transform back to energy basis and pseudo-integrate to obtain a
        # survival probability
        return np.dot(v,wf*ci[...,None]).T/flux0

    def transfer_matrix(self, cos_zenith, depth=0.5):
        """
        Calculate a transfer matrix that can be used to convert a neutrino flux
        at the surface of the Earth to the one observed at a detector under
        `depth` km of ice.

        :param cos_zenith: cosine of angle between neutrino arrival direction and local zenith
        :param depth: depth below the Earth's surface, in km
        :returns: an array of shape (6,6,T,N,N), where T is the broadcast shape
            of `cos_zenith` and `depth`, and N is the number of energy nodes.
            In other words, the array contains a transfer matrix for each
            combination of initial neutrino type, final neutrino type, and
            trajectory.
        """
        # find [number] column density of nucleons along the trajectory in cm^-2
        Na = 6.0221415e23
        t = np.atleast_1d(np.vectorize(get_t_earth)(np.arccos(cos_zenith), depth)*Na)

        num = self.energy_nodes.size
        transfer_matrix = np.zeros((6, 6) + t.shape + (num, num))
        for i in range(self.energy_nodes.size):
            # nu_e, nu_mu: CC absorption and NC downscattering
            for flavor in range(4):
                transfer_matrix[flavor,flavor,:,i,:] = self.transfer_matrix_element(i,flavor,flavor,t)

            # nu_tau: CC absorption and NC downscattering, plus neutrinos
            # from tau decay
            for flavor in range(4,6):
                for out_flavor in range(flavor % 2, flavor, 2):
                    secondary, tau = np.hsplit(self.transfer_matrix_element(i,flavor,out_flavor,t), 2)
                    transfer_matrix[flavor,flavor,:,i,:] = tau
                    transfer_matrix[flavor,out_flavor,:,i,:] = secondary

        return transfer_matrix

    @staticmethod
    @memoize
    def _get_cross_section(flavor, target, channel, secondary_flavor=None):
        if secondary_flavor:
            return DISCrossSection.create_secondary(flavor, secondary_flavor, target, channel)
        else:
            return DISCrossSection.create(flavor, target, channel)

    @memoize
    def total_cross_section(self, flavor):
        """
        Total interaction cross-section for neutrinos of of type `flavor`
        
        :returns: an array of length N of cross-sections in cm^2
        """
        assert isinstance(flavor,int) and 0 <= flavor < 6
        total = sum((self._get_cross_section(flavor+1, target, channel).total(self.energy_nodes) for (target,channel) in product(['n', 'p'], ['CC', 'NC'])), np.zeros_like(self.energy_nodes))/2.
        if flavor == 1:
            # for nuebar, add Glashow resonance cross-section
            # divide by 2 to account for the average number of electrons per
            # nucleon in an isoscalar medium
            total += GlashowResonanceCrossSection().total(self.energy_nodes)/2.
        return total

    @memoize
    def differential_cross_section(self, flavor, out_flavor):
        """
        Differential cross-section for neutrinos of of type `flavor` to produce
        secondary neutrinos of type `out_flavor`
        
        :returns: an array of shape (N,N) of differential cross-sections in cm^2 GeV^-1
        """
        assert isinstance(flavor,int) and 0 <= flavor < 6
        assert isinstance(out_flavor,int) and 0 <= out_flavor < 6
        assert (flavor % 2) == (out_flavor % 2), "no lepton-number-violating interactions"
        e_nu, e_sec = np.meshgrid(self.energy_nodes, self.energy_nodes, indexing='ij')
        total = np.zeros_like(e_nu)
        if out_flavor == flavor:
            total += sum((self._get_cross_section(flavor+1, target, channel).differential(e_nu, e_sec) for (target,channel) in product(['n', 'p'], ['NC'])), total)/2.
        if flavor == 1:
            # for nuebar, add Glashow resonance cross-section, assuming equal
            # branching ratios to e/mu/tau
            # divide by 2 to account for the average number of electrons per
            # nucleon in an isoscalar medium
            total += GlashowResonanceCrossSection().differential(e_nu, e_sec)/2.
        if flavor in (4,5):
            # for nutau(bar), add regeneration cross-section
            total += sum((self._get_cross_section(flavor+1, target, channel, out_flavor+1).differential(e_nu, e_sec) for (target,channel) in product(['n', 'p'], ['CC'])), np.zeros_like(e_nu))/2.
        return total

    def decompose_in_eigenbasis(self, flux, flavor, out_flavor):
        """
        Decompose `flux` in the eigenbasis of the cascade-equation solution
        
        :returns: (w,v,ci), the eigenvalues, eigenvectors, and coefficients of `flux` in the basis `v`
        """
        w, v = self.get_eigenbasis(flavor, out_flavor)
        ci = np.linalg.solve(v, flux)
        return w, v, ci

    def _sink_matrix(self, flavor):
        """
        Return a matrix with total interaction cross-section on the diagonal
        
        :param flavor: neutrino type (0-6)
        """
        return np.diag(self.total_cross_section(flavor))

    def _source_matrix(self, flavor, out_flavor):
        """
        Return a matrix with E^2-weighted differential neutrino cross-sections below the diagonal
        
        :param flavor: incoming neutrino type (0-6)
        :param out_flavor: outgoing neutrino type (0-6)
        """
        return (self.differential_cross_section(flavor, out_flavor)*self.differential_element).T

    @memoize
    def get_eigenbasis(self, flavor, out_flavor):
        """
        Construct and diagonalize the multiplier on the right-hand side of the
        cascade equation (M in Eq. 6 of the nuFATE paper).

        This is a more concise version of cascade.get_RHS_matrices() from the
        original nuFATE implementation.

        :param flavor: incoming neutrino flavor
        :param out_flavor: outgoing neutrino flavor
        :returns: (eigenvalues,eigenvectors) of M
        """
        downscattering = self._source_matrix(flavor,flavor) - self._sink_matrix(flavor)
        if out_flavor == flavor:
            RHSMatrix = downscattering
        else:
            secondary_production = self._source_matrix(flavor,out_flavor)
            secondary_downscattering = self._source_matrix(out_flavor,out_flavor) - self._sink_matrix(out_flavor)
            # in the flavor-mixing case, the right-hand side is:
            # nue/mu NC   nue/mu production
            # 0           nutau NC + regeneration
            RHSMatrix = np.vstack([
                np.hstack([secondary_downscattering,      secondary_production]),
                np.hstack([np.zeros_like(downscattering), downscattering      ]),
            ])
        w, v = np.linalg.eig(RHSMatrix)
        return w, v