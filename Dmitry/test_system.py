from __future__ import division
import kwant
import numpy as np
from kwant.digest import uniform

from current_matsubara_1 import mount_virtual_leads

class SimpleNamespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __copy__(self):
        out = SimpleNamespace()
        out.__dict__.update(self.__dict__)
        return out

def _disorder(site, salt):
    return 2 * uniform(repr(site), salt) - 1

def _bulk_e_onsite(site, p):
    return -p.mu + p.dis*_disorder(site, p.salt)

def _bulk_h_onsite(site, p):
    return p.mu - p.dis*_disorder(site, p.salt)

def _peierls_phase_func(pos1, pos2, p):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.exp(-1j * np.pi * p.flux * (x1 - x2) * (y1 + y2))

def _peierls_phase(site1, site2, p):
    return _peierls_phase_func(site1.pos, site2.pos, p)

def _bulk_e_hopping(site1, site2, p):
    return -p.t*_peierls_phase(site1, site2, p)

def _bulk_h_hopping(site1, site2, p):
    return p.t*_peierls_phase(site1, site2, p).conjugate()

def _leads_e_onsite(site, p):
    return -p.mu_leads

def _leads_h_onsite(site, p):
    return p.mu_leads

def _leads_e_hopping(site1, site2, p):
    return -p.t_leads    #*_peierls_phase(site1, site2, p)

def _leads_h_hopping(site1, site2, p):
    return p.t_leads     #*_peierls_phase(site1, site2, p).conjugate()

def _leads_delta(site1, site2, p):
    return p.delta

#def _phase_dependent_hopping(site1, site2, p):
    #return -p.t*np.exp(1j*p.phase)*_peierls_phase(site1, site2, p)

def make_system_params(W=1, L=2):
    return SimpleNamespace(W=W, L=L)

def params(mu=0., mu_leads=0., t=1., t_leads=1., flux=0,\
        delta=0, phase=0., dis=0., salt=""):
    """Constructs a parameter storage for Hamiltonian of a system.

    :mu: float
        Chemical potential of scattering region.
    :mu_leads: float
        Chemical potential of leads.
    :t: float
        Hopping between neighboring cells in scattering region, for holes
        sublattice (typically equals 1).
    :t_leads: float
        Hopping between neighboring cells in leads, for holes sublattice.
    :flux: float
        Magnetic flux, in units of flux quanta per unit cell.
    :delta: float
        Superconducting gap value.
    :phase: float
        Superconducting phase difference between leads.
    :dis: float
        Onsite energy disorder amplitude
    :returns: SimpleNamespace
        SimpleNamespace object with all parameters, with default ones
        for unspecifies.
    """
    return SimpleNamespace(mu=mu, mu_leads=mu_leads, t=t, t_leads=t_leads,\
                           flux=flux, delta=delta, phase=phase, dis=dis,\
                           salt=salt)

def make_system(params, vleads=True):
    """Creates a system with rectangular scattering region, two superconducting
    leads and two virtual leads with zero self-energy for obtaining Green's
    function on two neighboring cell columns.

    :params: object
        Parameter object, that will be used while constructing system (i.e.
        SimpleNamespace object). Must contain integer fields W and L, width and
        length of a system respectively.
    :finalized: boolean
        Whether finalize system before returning
    :prepare_vlead_interfaces: boolean
        If true, in addition to system also a tuple of lists of sites will be returned,
        that can be used for constructing virtual leads for current probing in electron
        lattice.
    :returns: kwant.builder.Builder or kwant.system.FiniteSystem and may be tuple
        System (finalized or not)
        If prepare_vlead_interfaces=True, in addition a tuple of lists of sites will be
        returned, that can be used for constructing virtual leads for current probing in
        electron lattice.
    """
    W = params.W
    L = params.L

    lat_e = kwant.lattice.square(name='e')
    lat_h = kwant.lattice.square(name='h')

    rect = lambda pos: 0 <= pos[0] < L and 0 <= pos[1] < W
    stripe = lambda pos: 0 <= pos[1] < W

    # Constructing bulk
    sys = kwant.Builder()
    sys[lat_e.shape(rect, (0, 0))] = _bulk_e_onsite
    sys[lat_h.shape(rect, (0, 0))] = _bulk_h_onsite
    sys[lat_e.neighbors()] = _bulk_e_hopping
    sys[lat_h.neighbors()] = _bulk_h_hopping

    # Constructing virtual leads with zero self energy -- for obtaining Green's function of
    # their interface without modifying system
    #zero_array = np.zeros((W, W), dtype=float)
    #def selfenergy_func(energy, args=()):
        #return zero_array
    #rvlead_pos = int(L//2)
    #lvlead_pos = rvlead_pos - 1
    #sys[((lat_e(rvlead_pos, y), lat_e(lvlead_pos, y)) for y in range(W))] = phase_dependent_hopping


    #levlead = kwant.builder.SelfEnergyLead(selfenergy_func, levlead_iface)
    #revlead = kwant.builder.SelfEnergyLead(selfenergy_func, revlead_iface)
    #sys.leads.append(levlead)
    #sys.leads.append(revlead)

    # Constructing leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[lat_e.shape(stripe, (0, 0))] = _leads_e_onsite
    lead[lat_h.shape(stripe, (0, 0))] = _leads_h_onsite
    lead[lat_e.neighbors()] = _leads_e_hopping
    lead[lat_h.neighbors()] = _leads_h_hopping
    lead[kwant.builder.HoppingKind((0, 0), lat_e, lat_h)] = _leads_delta
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())

    if vleads:
        rvlead_pos = int(L//2)
        lvlead_pos = rvlead_pos - 1
        levlead_iface = [lat_e(lvlead_pos, y) for y in range(W)]
        revlead_iface = [lat_e(rvlead_pos, y) for y in range(W)]
        sys, precalc_hopping_func = mount_virtual_leads(sys, levlead_iface, revlead_iface, 1)
        sys = sys.finalized()
        return sys, precalc_hopping_func
    else:
        return sys.finalized()
