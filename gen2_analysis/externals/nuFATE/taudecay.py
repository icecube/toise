
import numpy as np
import scipy.integrate

class TauDecay:
    """See Appendix A, Table 1 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.62.123001"""
    R_PION  = 0.07856**2
    R_RHO   = 0.43335**2
    R_A1    = 0.70913**2
    R_OTHER = 0.7
    # points where to_all() is not C^1 continuous
    breakpoints = np.array([R_PION, R_RHO, R_A1, R_OTHER])
    @classmethod
    def to_lepton(cls, z, polarization):
        g0 = 5.0/3.0-3.0*np.power(z,2.0)+(4.0/3.0)*np.power(z,3.0)
        g1 = 1.0/3.0-3.0*np.power(z,2.0)+(8.0/3.0)*np.power(z,3.0)
        return g0 + polarization*g1
    @classmethod
    def to_pion(cls, z, polarization):
        R = cls.R_PION
        g0 = np.where(1-R-z > 0, 1./(1.-R), 0)
        g1 = np.where(1-R-z > 0, -(2.0*z-1.0+R)/np.power(1.0-R,2.0), 0)
        return g0 + polarization*g1
    @classmethod
    def to_rho(cls, z, polarization):
        R = cls.R_RHO
        g0 = np.where(1-R-z > 0, 1./(1.-R), 0)
        g1 = np.where(1-R-z > 0, -((2.0*z-1.0+R)/(1.0-R))*((1.0-2.0*R)/(1.0+2.0*R)), 0)
        return g0 + polarization*g1
    @classmethod
    def to_a1(cls, z, polarization):
        R = cls.R_A1
        g0 = np.where(1-R-z > 0, 1./(1.-R), 0)
        g1 = np.where(1-R-z > 0, -((2.0*z-1.0+R)/(1.0-R))*((1.0-2.0*R)/(1.0+2.0*R)), 0)
        return g0 + polarization*g1
    @classmethod
    def to_other_hadrons(cls, z, polarization):
        R = cls.R_OTHER
        return np.where(1-R-z > 0, 1./.3, 0)
    @classmethod
    def to_any(cls, z, polarization):
        return np.where(z < 1,
            2*0.18*cls.to_lepton(z,polarization) \
            +0.12*cls.to_pion(z,polarization) \
            +0.26*cls.to_rho(z,polarization) \
            +0.13*cls.to_a1(z,polarization) \
            +0.13*cls.to_other_hadrons(z,polarization) \
            ,0)
    @classmethod
    def to_hadrons(cls, z, polarization):
        return np.where(z < 1, np.where(z > 0, 
            0.12*cls.to_pion(z,polarization) \
            +0.26*cls.to_rho(z,polarization) \
            +0.13*cls.to_a1(z,polarization) \
            +0.13*cls.to_other_hadrons(z,polarization) \
            ,0),0)

@np.vectorize
def tau_regen_crossdiff(nucrossdiff, e1, e2, polarization):
    """
    Calculate differential cross-section for nu_tau -> nu_tau, assuming
    that the intermediate tau lepton decays instantly.
    
    :param nucrossdiff: a callable with the same signature as nusigma.nucrossdiff
    """
    assert polarization in (-1,1)
    if e2 > e1:
        return 0.
    # cf inner integrand of eq 3 from nuFATE paper
    # substituted e_tau for y and transforming to log space
    def term(loge):
        e_tau = np.exp(loge)
        if e_tau < e2:
            return 0.
        else:
            # nb: d(e_tau)/d(log(e_tau)) cancels d(z)/d(e2)
            return TauDecay.to_any(e2/e_tau, polarization)*nucrossdiff(e1,e_tau)
    hi = np.log(e1)
    lo = hi - 10
    # lo = np.log(energy_nodes[0])
    # note points where TauDecay.to_any() has a 1st-order discontinuity
    points = np.log(e2/(1-TauDecay.breakpoints))
    # clip these to the integration range to keep quadpack happy
    points = points[(points<hi)&(points>lo)]
    return scipy.integrate.quad(term, lo, hi, points=points)[0]

@np.vectorize
def second_bang_crossdiff(nucrossdiff, e1, e2, polarization):
    """
    Calculate differential cross-section for nu_tau -> hadron cascade, assuming
    that the intermediate tau lepton decays instantly.
    
    :param nucrossdiff: a callable with the same signature as nusigma.nucrossdiff
    """
    assert polarization in (-1,1)
    if e2 > e1:
        return 0.
    # cf inner integrand of eq 3 from nuFATE paper
    # substituted e_tau for y and transforming to log space
    def term(loge):
        e_tau = np.exp(loge)
        if e_tau < e2:
            return 0.
        else:
            # nb: d(e_tau)/d(log(e_tau)) cancels d(z)/d(e2)
            return TauDecay.to_hadrons(1-e2/e_tau, polarization)*nucrossdiff(e1,e_tau)
    hi = np.log(e1)
    lo = hi - 10
    # lo = np.log(energy_nodes[0])
    # note points where TauDecay.to_any() has a 1st-order discontinuity
    points = np.log(e2/(TauDecay.breakpoints))
    # clip these to the integration range to keep quadpack happy
    points = points[(points<hi)&(points>lo)]
    return scipy.integrate.quad(term, lo, hi, points=points)[0]

@np.vectorize
def tau_secondary_crossdiff(nucrossdiff, e1, e2, polarization):
    """
    Calculate differential cross-section for nu_tau -> nu_(e|mu), assuming
    that the intermediate tau lepton decays instantly.
    
    :param nucrossdiff: a callable with the same signature as nusigma.nucrossdiff
    """
    assert polarization in (-1,1)
    if e2 > e1:
        return 0.
    # cf inner integrand of eq 3 from nuFATE paper ()
    # substituted e_tau for y and transforming to log space
    def term(loge):
        e_tau = np.exp(loge)
        if e_tau < e2:
            return 0.
        else:
            # nb: d(e_tau)/d(log(e_tau)) cancels d(z)/d(e2)
            z = e2/e_tau
            dndz = 0.18*(4 - 12*z + 12*z**2 - 4*z**3)
            return dndz*nucrossdiff(e1,e_tau)
    hi = np.log(e1)
    lo = hi-10
    return scipy.integrate.quad(term, lo, hi)[0]
