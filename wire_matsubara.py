import kwant

# For plotting

import sys

import numpy as np

import tinyarray as ta

import numpy.random as rnd

import scipy as sp

from numpy.random import seed as sd

import matplotlib.pyplot as pyplot

import scipy.sparse.linalg as sla
import scipy.linalg as la

from current_matsubara_1 import mount_virtual_leads
import current_matsubara_1 as cm
import test_system as test

import matplotlib.pyplot as plt

#definition of the Pauli matrices
sigma0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)

# the structure of the Kronnecker product
#          spin    ph
# np.kron(sigma0, sigmax)

def plot_lead_bandstructure(lead, momenta, E_min, E_max):
    lead = lead.finalized()
    kwant.plotter.bands(lead, momenta=momenta, show=False)
    pyplot.xlabel("momentum [(lattice constant)^-1]")
    pyplot.ylabel("energy [t]")
    pyplot.ylim(E_min, E_max) 
    pyplot.show()

def onsite(a, mu, m, Hx, Hy, Hz, Delta):
    return (3 / m / a**2 * np.kron(sigma0, sigmaz)
            + mu * np.kron(sigma0, sigmaz)
            + Hx * np.kron(sigmax, sigma0)
            + Hy * np.kron(sigmay, sigma0)
            + Hz * np.kron(sigmaz, sigma0)
            + np.real(Delta) * np.kron(sigma0, sigmax)
            + np.imag(Delta) * np.kron(sigma0, sigmay))

def hopping_x(a, m, alpha, L, L1, x, y, z, fluxY, fluxZ): 
    if x <= L1:
        peierls_phase1 = 0.
        peierls_phase2 = 0.
    elif (x < L - L1):
        peierls_phase1 = y
        peierls_phase2 = z
    else:
        peierls_phase1 = 0.
        peierls_phase2 = 0.
    return (np.asmatrix(1.j * alpha / 2. / a * np.kron(sigmay, sigmaz)
                       - 1 / 2. / m / a**2 * np.kron(sigma0, sigmaz)) 
            * la.expm(2.j * np.pi * 
                      (peierls_phase1 * fluxZ + peierls_phase2 * fluxY) * np.kron(sigma0, sigmaz)))

def hopping_y(a, m, alpha):
    return np.asmatrix(- 1.j * alpha / 2. / a * np.kron(sigmax, sigmaz)
                        - 1 / 2. / m / a**2 * np.kron(sigma0, sigmaz)) 

def hopping_z(a, m, alpha):
    return np.asmatrix(- 1 / 2. / m / a**2 * np.kron(sigma0, sigmaz)) 

def onsite_ee(a, mu, m, Hx, Hy, Hz):
    return (3 / m / a**2 * sigma0
            + mu * sigma0
            + Hx * sigmax
            + Hy * sigmay
            + Hz * sigmaz)

def onsite_hh(a, mu, m, Hx, Hy, Hz):
    return (- 3 / m / a**2 * sigma0
            - mu * sigma0
            + Hx * sigmax
            + Hy * sigmay
            + Hz * sigmaz)

def onsite_eh(Delta):
    return Delta * sigma0

def hopping_x_ee(a, m, alpha, L, L1, x, y, z, fluxY, fluxZ):
    if x <= L1:
        peierls_phase1 = 0.
        peierls_phase2 = 0.
    elif (x < L - L1):
        peierls_phase1 = y
        peierls_phase2 = z
    else:
        peierls_phase1 = 0.
        peierls_phase2 = 0.
    return np.asmatrix(1.j * alpha / 2. / a * sigmay
                       - 1 / 2. / m / a**2 * sigma0) * sp.exp(2.j * np.pi * 
                      (peierls_phase1 * fluxZ + peierls_phase2 * fluxY))

def hopping_x_hh(a, m, alpha, L, L1, x, y, z, fluxY, fluxZ):
    if x <= L1:
        peierls_phase1 = 0.
        peierls_phase2 = 0.
    elif (x < L - L1):
        peierls_phase1 = y
        peierls_phase2 = z
    else:
        peierls_phase1 = 0.
        peierls_phase2 = 0.
    return np.asmatrix(- 1.j * alpha / 2. / a * sigmay
                       + 1 / 2. / m / a**2 * sigma0) * sp.exp(- 2.j * np.pi * 
                      (peierls_phase1 * fluxZ + peierls_phase2 * fluxY))

def hopping_y_ee(a, m, alpha):
    return np.asmatrix(- 1.j * alpha / 2. / a *sigmax
                       - 1 / 2. / m / a**2 * sigma0)

def hopping_y_hh(a, m, alpha):
    return np.asmatrix(1.j * alpha / 2. / a * sigmax
                       + 1 / 2. / m / a**2 * sigma0)

def hopping_z_ee(a, m, alpha):
    return np.asmatrix(- 1 / 2. / m / a**2 * sigma0)

def hopping_z_hh(a, m, alpha):
    return np.asmatrix(1 / 2. / m / a**2 * sigma0)

def shape(R, L):
    '''
    Returns the 3D array of points which fit inside the wire of radius R 
    and length L
    
    Parameters:
    R: float
        the radius of the wire in lattice constants
    L: integer
        the length of the wire
    
    Returns:
    shape_xyz: array of triples
        array of the triples containing the site's coordinates
        which fit into the wire
    '''
    
    shape_xyz = []
    #itteration over the possible positions of the points
    for x in range(L):
        for y in range(-R, R + 1):
            for z in range(-R, R + 1):
                #checks if the position is inside the wire
                if ((y**2 + z**2) <= R**2):
                    shape_xyz.append((x, y, z))
                    
    return shape_xyz

def shape_lead(R):
    '''
    Returns the 2D array of points which fit between R and R1 on a
    angular sector of size angle
    
    Parameters:
    R: float
        the radius of the wire in lattice constants
    
    Returns:
    shape_yz: array of tuples
        array of the tuples containing the site's coordinates
        which fit into the wire
    '''
    
    shape_yz = []
    #itteration over the possible positions of the points
    for y in range(-R, R + 1):
        for z in range(-R, R + 1):
            #checks if the position is inside the wire
            if ((y**2 + z**2) < R**2):
                shape_yz.append((y, z))
                
    return shape_yz

def shape_SC(R, R1, L1, L0, angle):
    '''
    Returns the 3D array of points which fit between R and R1 on a
    angular sector of size angle with x coordinates between L0 and L1
    
    Parameters:
    R: float
        the radius of the wire the sc coats in lattice constants
    R1: float
        the thickness of the coating in lattice constants
    L1: integer
        the length of the superconducting coating
    L0: integer
        the position of the left side of the coating
    angle: float
        the angle of the superconducting coating of the wire
    
    Returns:
    shape_xyz: array of triples
        array of the triples containing the site's coordinates
        which fit into the desired shape
    '''
    
    shape_xyz = []
    #itteration over the possible positions of the points
    for x in range(L0, L0 + L1):
        for y in range(- R - R1, R + R1 + 1):
            for z in range(- R - R1, R + R1 + 1):
                #checks if the position is inside the desired shape
                if (((y**2 + z**2) >= R**2)
                    and ((y**2 + z**2) < (R + R1)**2)
                    and ((np.arccos(y / np.sqrt(y**2 + z**2))
                          + (1 - np.sign(z)) * np.pi / 2.) <= angle)):
                    shape_xyz.append((x, y, z))
                    
    return shape_xyz

def shape_lead_SC(R, R1, angle):
    '''
    Returns the 2D array of points which fit between R and R1 on a
    angular sector of size angle
    
    Parameters:
    R: float
        the radius of the wire the sc coats in lattice constants
    R1: float
        the thickness of the coating in lattice constants
    angle: float
        the angle of the superconducting coating of the wire
    
    Returns:
    shape_yz: array of tuples
        array of the tuples containing the site's coordinates
        which fit into the sector
    '''
    
    shape_yz = []
    #itteration over the possible positions of the points
    for y in range(- R - R1, R + R1 + 1):
        for z in range(- R - R1, R + R1 + 1):
            #checks if the position is inside the desired sector
            if (((y**2 + z**2) >= R**2)
                and ((y**2 + z**2) < (R + R1)**2)
                and ((np.arccos(y / np.sqrt(y**2 + z**2))
                      + (1 - np.sign(z)) * np.pi / 2.) <= angle)):
                shape_yz.append((y, z))
                
    return shape_yz

def make_lead(R=20, a=1, #basic parameters of the system
              mu=0., m=1., alpha=0., #dispersion
              Bx=0., By=0., Bz=0., #magnetic field
              Delta=0.001, angle=np.pi, R1=2, #superconducting gap
              sc_lead=False, plot_lead=False, plot_dispersion=True,
              s_zeeman=1., s_orbital=1.):
    '''
    The function makes a lead and plots the bandstructure of it. 
    Returns the lead and the lattice it is defined on
    
    Parameters:
    R: float 
        the radius of the wire in lattice constants
    a: float 
        the lattice constant in nm
    mu: float 
        chemical potential in eV
    m: float
        effective mass in nm^2 / eV
    alpha: float
        spin-orbit coupling in eV * nm
    Bx: float 
        magnetic field in x direction
    By: float
        magnetic field in y direction
    Bz: float
        magnetic field in z direction
    Delta: complex
        superconducting gap in the lead
    angle: float
        the angle of the superconducting coating of the wire
    R1: float
        the thickness of the coating in lattice constants
    sc_lead: bool
        if True, then the lead has superconducting coating,
        if False, there is no superconducting part and no coating
    plot_lead: bool
        if True the function plots the 3d image of the lead
    plot_dispersion: bool
        if True the function plots the dispersion of the lead

    Returns:
    lead: kwant.Builder
        the lead
    lat: kwant.lattice
        the lattice on which the lead is defined
    '''

    #transforms the magnetic fields into phases of Peierls substitution
    fluxX = 0.00024179892623048702 * a**2 * Bx * s_orbital
    fluxY = 0.00024179892623048702 * a**2 * By * s_orbital
    fluxZ = 0.00024179892623048702 * a**2 * Bz * s_orbital

    #transforms the magnetic fields into Zeeman terms
    Hx = 1.4470954502770883 * 10.**(-3) * Bx * s_zeeman
    Hy = 1.4470954502770883 * 10.**(-3) * By * s_zeeman
    Hz = 1.4470954502770883 * 10.**(-3) * Bz * s_zeeman
    
    #defines the 3d lattice
    lat = kwant.lattice.general([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    #creates the lead
    lead = kwant.Builder(kwant.TranslationalSymmetry([1, 0, 0]))

    #finds the shape the lead is defined on
    shape2D = shape_lead(R)

    if sc_lead:
        shape2Dsc = shape_lead_SC(R, R1, angle)
    else:
        shape2Dsc = []

    shape2D += shape2Dsc

    #adds on-site terms
    for (y, z) in shape2D:
        lead[lat(0, y, z)] = onsite(a, mu, m, Hx, Hy, Hz, 0.)
    for (y, z) in shape2Dsc:
        if (y**2 + z**2 == R**2):
            lead[lat(0, y, z)] = onsite(a, mu, m, Hx, Hy, Hz, 0.)
        else:
            lead[lat(0, y, z)] = onsite(a, mu, m, Hx, Hy, Hz, Delta)

    #adds hopping
    for (y, z) in shape2D:
        lead[(lat(0, y, z), lat(1, y, z))] = hopping_x(a, m, alpha, 100, 5, 20, y, z, fluxY, fluxZ)
        if (y + 1, z) in shape2D:
            lead[(lat(0, y, z), lat(0, y + 1, z))] = hopping_y(a, m, alpha)
        if (y, z + 1) in shape2D:
            lead[(lat(0, y, z), lat(0, y, z + 1))] = (la.expm(2.j * np.pi * y * fluxX * np.kron(sigma0, sigmaz)) * 
                                                      np.asmatrix(hopping_z(a, m, alpha)) * la.expm(0.j * np.pi * y * 
                                                                                      fluxX * np.kron(sigma0, sigmaz)))
    #plots the lead
    if plot_lead:
        kwant.plot(lead)

    #plots the dispersion
    if plot_dispersion:
        plot_lead_bandstructure(lead, np.linspace(-np.pi, np.pi, 500), -0.01, 0.01)
    
    return lead, lat

def make_system_eh(R=20, L=20, a=1, #basic parameters of the system
                   mu=0., m=1., alpha=0., #dispersion
                   Bx=0., By=0., Bz=0., #magnetic field
                   Delta1=0., Delta2=0., angle=np.pi, R1=0, L1=1, no_B_SC=False, #superconducting gap
                   U=0., SEED=1, #disorder, not realized yet
                   Lgate=2, mugate=0., #gate in the middle of the wire
                   vleads=False, #presence or absence of virtual leads
                   s_orbital=1., s_zeeman=1.):
    
    '''
    The function makes the wire. 
    Returns the system and the lattice it is defined on
    
    Parameters:
    R: float 
        the radius of the wire in lattice constants
    L: integer
        the length of the wire in lattice constants
    a: float 
        the lattice constant in nm
    mu: float 
        chemical potential in eV
    m: float
        effective mass in nm^2 / eV
    alpha: float
        spin-orbit coupling in eV * nm
    Bx: float 
        magnetic field in x direction
    By: float
        magnetic field in y direction
    Bz: float
        magnetic field in z direction
    Delta1: complex
        superconducting gap in the left coating
    Delta2: complex
        superconducting gap in the right coating
    angle: float
        the angle of the superconducting coating of the wire
    R1: float
        the thickness of the coating in lattice constants
    L1: integer
        the length of the superconducting coating
    sc_lead: bool
        if True, then the lead has superconducting coating,
        if False, there is no superconducting part and no coating
    
    Returns:
    sys: kwant.Builder
        the system
    lat: kwant.lattice
        the lattice on which the system is defined
    '''
    
    #fixes the seed
    sd(SEED)
    
    #defines the 3d lattice
    lat_e = kwant.lattice.general([(1, 0, 0), (0, 1, 0), (0, 0, 1)], name='e')
    lat_h = kwant.lattice.general([(1, 0, 0), (0, 1, 0), (0, 0, 1)], name='h')
    
    #creates the system
    sys = kwant.Builder()
    
    #transforms the magnetic fields into phases of Peierls substitution
    fluxX = 0.00024179892623048702 * a**2 * Bx * s_orbital
    fluxY = 0.00024179892623048702 * a**2 * By * s_orbital
    fluxZ = 0.00024179892623048702 * a**2 * Bz * s_orbital
    
    #transforms the magnetic fields into Zeeman terms
    Hx = 1.4470954502770883 * 10.**(-3) * Bx * s_zeeman
    Hy = 1.4470954502770883 * 10.**(-3) * By * s_zeeman
    Hz = 1.4470954502770883 * 10.**(-3) * Bz * s_zeeman
    
    #fixes the shapes of the parts of the system 
    #not to call the functions every time
    shape3D0 = shape(R, L)
    
    shape3DSCLeft = shape_SC(R, R1, L1, 0, angle)
    
    shape3DSCRight = shape_SC(R, R1, L1, L - L1, angle)
    
    shape3D = shape3D0 + shape3DSCLeft + shape3DSCRight
    
    #puts on-site terms in place
    vllead = []
    vrlead = []
    for (x, y, z) in shape3D:
        if (U>0):
            deltamu = rnd.normal(0., U)
        else:
            deltamu = 0.
        #create on-site Hamiltonian
        if (x >= L1) and (x < L - L1) or (not no_B_SC):
            #print x, y, z
            sys[lat_e(x, y, z)] = onsite_ee(a, mu + deltamu, m, Hx, Hy, Hz)
            sys[lat_h(x, y, z)] = onsite_hh(a, mu + deltamu, m, Hx, Hy, Hz)
            #print onsite_hh(a, mu + deltamu, m, Hx, Hy, Hz)
        else:
            #no Hx and Hy terms under the superconductor
            sys[lat_e(x, y, z)] = onsite_ee(a, mu + deltamu, m, Hx, 0., 0.)
            sys[lat_h(x, y, z)] = onsite_hh(a, mu + deltamu, m, Hx, 0., 0.)
        if (x >= L/2 - Lgate/2) and (x < L/2 - Lgate/2 + Lgate):
            #creates a gate in the middle of the device
            sys[lat_e(x, y, z)] = mugate * sigma0 + sys[lat_e(x, y, z)]
            sys[lat_h(x, y, z)] = - mugate * sigma0 + sys[lat_h(x, y, z)]
        if vleads and (x==L//2):
            vllead.append(lat_e(x, y, z))
        elif vleads and (x==L//2 + 1):
            vrlead.append(lat_e(x, y, z))
    #adds the electron-hole hoppings into the superconducting part of the device
    for (x, y, z) in shape3DSCLeft:
        sys[(lat_e(x, y, z), lat_h(x, y, z))] = onsite_eh(Delta1)
    for (x, y, z) in shape3DSCRight:
        sys[(lat_e(x, y, z), lat_h(x, y, z))] = onsite_eh(Delta2)
    
    #creates hoppings
    for (x, y, z) in shape3D:
        if (x + 1, y, z) in shape3D:
            sys[(lat_e(x, y, z), lat_e(x + 1, y, z))] = hopping_x_ee(a, m, alpha, L, L1, x, y, z, fluxY, fluxZ)
            sys[(lat_h(x, y, z), lat_h(x + 1, y, z))] = hopping_x_hh(a, m, alpha, L, L1, x, y, z, fluxY, fluxZ)
        if (x, y + 1, z) in shape3D:
            sys[(lat_e(x, y, z), lat_e(x, y + 1, z))] = hopping_y_ee(a, m, alpha)
            sys[(lat_h(x, y, z), lat_h(x, y + 1, z))] = hopping_y_hh(a, m, alpha)
        if (x, y, z + 1) in shape3D:
            sys[(lat_e(x, y, z), lat_e(x, y, z + 1))] = hopping_z_ee(a, m, alpha)
            sys[(lat_h(x, y, z), lat_h(x, y, z + 1))] = hopping_z_hh(a, m, alpha)
    
    #changing the hoppings to take into account flux in x direction
    for (x, y, z) in shape3D:
        if ((x, y, z) in shape3D0) and ((x, y + 1, z) in shape3D0):
            for j in range(2 * R + 1):
                if ((x, y, z - j - 1) in shape3D0):
                    sys[(lat_e(x, y, z), lat_e(x, y + 1, z))] = (np.asmatrix(sys[(lat_e(x, y, z), lat_e(x, y + 1, z))])
                                                             * la.expm(1.j * np.pi * fluxX * sigma0))
                    sys[(lat_h(x, y, z), lat_h(x, y + 1, z))] = (np.asmatrix(sys[(lat_h(x, y, z), lat_h(x, y + 1, z))])
                                                             * la.expm(- 1.j * np.pi * fluxX * sigma0))
                if ((x, y + 1, z - j - 1) in shape3D0):
                    sys[(lat_e(x, y, z), lat_e(x, y + 1, z))] = (np.asmatrix(sys[(lat_e(x, y, z), lat_e(x, y + 1, z))])
                                                             * la.expm(1.j * np.pi * fluxX * sigma0))
                    sys[(lat_h(x, y, z), lat_h(x, y + 1, z))] = (np.asmatrix(sys[(lat_h(x, y, z), lat_h(x, y + 1, z))])
                                                             * la.expm(- 1.j * np.pi * fluxX * sigma0)) 
    if vleads:
        sys, precalc_hopping_func = mount_virtual_leads(sys, vllead, vrlead, 2)
        return sys, precalc_hopping_func
    return sys

def compute_Ic_matsubara(R=3, L=66, R1=2, L1=16, angle=3.*np.pi/4., a=16.6, mu=-0.015,
               m=0.20, alpha=0.02, DELTA=0.002, Nphi=100, Bx=0., By=0., Bz=0.,
               no_B_SC=False, Lgate=2, mugate=0., U=0., SEED=1,
               s_orbital=1., s_zeeman=1., plot_current_phase=False):
    sys, hf = make_system_eh(R=R, L=L, a=a, mu=mu, m=m, alpha=alpha, 
                               Bx=Bx, By=By, Bz=Bz, 
                               Delta1=DELTA, 
                               Delta2=DELTA,
                               R1=R1, L1=L1, angle=angle,
                               no_B_SC=no_B_SC, 
                               Lgate=Lgate, mugate=mugate,
                               U=U, SEED=SEED, vleads=True,
                               s_orbital=s_orbital, s_zeeman=s_zeeman)
    phases = np.linspace(-np.pi, np.pi, Nphi)
    params = test.params()
    T = 0.0001
    phases, currents = cm.dep_current_phase(sys.finalized(), hf, params, T, phases,
    rel_error=0.001, max_nmatsfreq=20000, matsfreq_set=[0])
    print(Bx, By, Bz, max(abs(currents)), max(currents), min(currents), R, L, R1, L1, a, mu, m, alpha, DELTA, Nphi, s_orbital, s_zeeman, U)
    
    if plot_current_phase:
        plt.figure()
        plt.plot(phases, currents)
        plt.show()

def main():
    ### Nice parameters, tested by the dispersion of the wire
    #make_lead(R=4, a=12.0,
              #mu=-0.02, m=0.2, alpha=0.02,
              #Bx=0., By=0., Bz=0.,
              #Delta=0.002, angle=np.pi, R1=3,
              #sc_lead=True, plot_lead=False, plot_dispersion=True)
    #make_lead(a=10, R=5, R1=2, alpha=0.0, mu=-0.015, m=0.19685131956193805, plot_dispersion=True,
              #Bx=0., s_zeeman=1., s_orbital=0., sc_lead=True, angle=np.pi, Delta=0.005)
    compute_Ic_matsubara(a=10, L=60, L1=15, R=5, R1=2, alpha=0.02, mu=-0.01, m=0.2, DELTA=0.02, Bx=0.4)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
