from collections import namedtuple
import deepdish
import discretizer
from functools import lru_cache
from itertools import product, starmap
import kwant
from kwant.digest import uniform
import numpy as np
import os
import pandas as pd
import subprocess
from scipy.constants import hbar, m_e, eV, physical_constants
import scipy.optimize
import sympy
import sympy.physics
from sympy.physics.quantum import TensorProduct as kr
import types
import warnings


class SimpleNamespace(types.SimpleNamespace):

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


def named_product(**items):
    Product = namedtuple('Product', items.keys())
    return starmap(Product, product(*items.values()))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_git_revision_hash():
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").replace('\n', '')


def find_nearest(array, value):
    idx = np.abs(np.array(array) - value).argmin()
    return array[idx]


def gate(syst, V, gate_size):
    x_positions = sorted(set(i.pos[0] for i in syst.sites))
    x_mid = (max(x_positions) - min(x_positions)) / 2
    x_L = find_nearest(x_positions, x_mid - gate_size / 2)
    x_R = find_nearest(x_positions, x_mid + gate_size / 2)
    return lambda x: V if x > x_L and x <= x_R else 0


k_B = physical_constants['Boltzmann constant in eV/K'][0] * 1000
sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
s0 = sympy.eye(2)
s0sz = np.kron(s0, sz)
s0s0 = np.kron(s0, s0)

# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental constants, in a SimpleNamespace.
constants = SimpleNamespace(
    m=0.015 * m_e,  # effective mass in kg
    hbar=hbar,
    m_e=m_e,
    e=eV,
    eV=eV,
    meV=eV * 1e-3)

constants.t = (constants.hbar**2 / (2 * constants.m)) * \
    (1e18 / constants.meV)  # meV * nm^2
constants.mu_B = physical_constants['Bohr magneton'][0] / constants.meV


def make_params(alpha=20,
                B_x=0,
                B_y=0,
                B_z=0,
                Delta=0.25,
                mu=0,
                orbital=True,
                t=constants.t,
                g=50,
                mu_B=constants.mu_B,
                V=lambda x: 0,
                **kwargs):
    """Function that creates a namespace with parameters.

    Parameters:
    -----------
    alpha : float
        Spin-orbit coupling strength in units of meV*nm.
    B_x, B_y, B_z : float
        The magnetic field strength in the x, y and z direction in units of Tesla.
    Delta : float
        The superconducting gap in units of meV.
    mu : float
        The chemical potential in units of meV.
    orbital : bool
        Switches the orbital effects on and off.
    t : float
        Hopping parameter in meV * nm^2.
    g : float
        Lande g factor.
    mu_B : float
        Bohr magneton in meV/K.
    V : function
        Potential as function of x.
    Returns:
    --------
    p : SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    """
    p = SimpleNamespace(alpha=alpha,
                        B_x=B_x,
                        B_y=B_y,
                        B_z=B_z,
                        Delta=Delta,
                        mu=mu,
                        orbital=orbital,
                        t=t,
                        g=g,
                        mu_B=mu_B,
                        V=V,
                        **kwargs)
    return p


def discretized_hamiltonian(a=10, spin=True, holes=True):
    k_x, k_y, k_z = discretizer.momentum_operators
    x, y, z = discretizer.coordinates
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g, V = sympy.symbols(
        't B_x B_y B_z mu_B Delta mu alpha g V', real=True)
    t_interface = sympy.symbols('t_interface', real=True)
    k = sympy.sqrt(k_x**2+k_y**2+k_z**2)
    if spin and holes:
        ham = ((t * k**2 - mu + V(x)) * kr(s0, sz) +
               alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
               0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
               Delta * kr(s0, sx))
    elif not spin and holes:
        ham = ((t * k**2 - mu + V(x)) * s0 +
               alpha * (k_y * sx - k_x * sy) +
               0.5 * g * mu_B * (B_x * sx + B_y * sy + B_z * sz) +
               Delta * s0)
    elif spin and not holes:
        ham = ((t * k**2 - mu + V(x)) * sz +
               alpha * (k_y * sz - k_x * sz) +
               0.5 * g * mu_B * (B_x * s0 + B_y * s0 + B_z * s0) +
               Delta * sx)
    args = dict(lattice_constant=a, discrete_coordinates={'x', 'y', 'z'})
    tb_normal = discretizer.Discretizer(ham.subs(Delta, 0), **args)
    tb_sc = discretizer.Discretizer(ham.subs([(g, 0), (alpha, 0)]), **args)
    tb_interface = discretizer.Discretizer(ham.subs(t, t_interface), **args)
    return tb_normal, tb_sc, tb_interface


def hoppingkind_in_shape(hop, shape, syst):
    """Returns an HoppingKind iterator for hoppings in shape."""
    def in_shape(site1, site2, shape):
        return shape[0](site1.pos) and shape[0](site2.pos)
    hoppingkind = kwant.HoppingKind(hop.delta, hop.family_a)(syst)
    return ((i, j) for (i, j) in hoppingkind if in_shape(i, j, shape))


def hoppingkind_at_interface(hop, shape1, shape2, syst):
    """Returns an HoppingKind iterator for hoppings at an interface between
       shape1 and shape2."""
    def at_interface(site1, site2, shape1, shape2):
        return ((shape1[0](site1.pos) and shape2[0](site2.pos)) or
                (shape2[0](site1.pos) and shape1[0](site2.pos)))
    hoppingkind = kwant.HoppingKind(hop.delta, hop.family_a)(syst)
    return ((i, j) for (i, j) in hoppingkind if at_interface(i, j, shape1, shape2))


def peierls(func, ind, a, c=constants):
    """Applies Peierls phase to the hoppings functions.
    Note that this function only works if spin is present.
    Parameters:
    -----------
    func : function
        Hopping function in certain direction.
    ind : int
        Index of xyz direction, corresponding to 0, 1, 2.
    a : int
        Lattice constant in nm.
    c : SimpleNamespace object
        Namespace object that contains fundamental constants.
    Returns:
    --------
    with_phase : function
        Hopping function that contains the Peierls phase if p.orbital
        is True.
    """
    def phase(s1, s2, p):
        x, y, z = s1.pos
        A_site = [p.B_y * z - p.B_z * y, 0, p.B_x * y][ind]
        A_site *= a * 1e-18 * c.eV / c.hbar
        return np.exp(-1j * A_site)
    def with_phase(s1, s2, p):
        hop = func(s1, s2, p).astype('complex128')
        phi = phase(s1, s2, p)
        if p.orbital:
            if hop.shape[0] == 2:
                hop *= phi
            elif hop.shape[0] == 4:
                hop *= np.array([phi, phi.conj(), phi,
                                 phi.conj()], dtype='complex128')
        return hop
    return with_phase


def matsubara_frequency(T, n):
    return (2*n + 1) * np.pi * k_B * T * 1j


def null_H(syst, p, T, n):
    en = matsubara_frequency(T, n)
    gf = kwant.greens_function(
        syst, en, [p], [0], [0], check_hermiticity=False)
    return np.linalg.inv(gf.data[::2, ::2])


def gf_from_H_0(H_0, t):
    H = np.copy(H_0)
    dim = t.shape[0]
    H[:dim, dim:] -= t.T.conj()
    H[dim:, :dim] -= t
    return np.linalg.inv(H)


def current_from_H_0(T, H_0_cache, H12, phase):
    t = H12 * np.exp(1j * phase)
    I = 0
    for H_0 in H_0_cache:
        gf = gf_from_H_0(H_0, t - H12)
        dim = t.shape[0]
        H12G21 = t.T.conj() @ gf[dim:, :dim]
        H21G12 = t @ gf[:dim, dim:]
        I += -4 * T * (np.trace(H21G12) - np.trace(H12G21)).imag
    return I


def I_c_fixed_n(syst, hopping, p, T, matsfreqs=5):
    H_0_cache = [null_H(syst, p, T, n) for n in range(matsfreqs)]
    H12 = hopping(syst, [p])
    fun = lambda phase: -current_from_H_0(T, H_0_cache, H12, phase)
    opt = scipy.optimize.brute(
        fun, ranges=((-np.pi, np.pi),), Ns=30, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid, currents=-Jout)


def current_contrib_from_H_0(T, H_0, H12, phase):
    # Maybe take this line outside of the function?
    t = H12 * np.exp(1j * phase)
    gf = gf_from_H_0(H_0, t - H12)
    dim = t.shape[0]
    H12G21 = t.T.conj() @ gf[dim:, :dim]
    H21G12 = t @ gf[:dim, dim:]
    return -4 * T * (np.trace(H21G12) - np.trace(H12G21)).imag


def current_at_phase(syst, hopping, p, T, H_0_cache, phase, tol=1e-2, max_frequencies=200):
    H12 = hopping(syst, [p])
    I = 0
    for n in range(max_frequencies):
        if len(H_0_cache) <= n:
            H_0_cache.append(null_H(syst, p, T, n))
        I_contrib = current_contrib_from_H_0(T, H_0_cache[n], H12, phase)
        I += I_contrib
        if I_contrib == 0 or tol is not None and abs(I_contrib / I) < tol:
            return I
    # Did not converge within tol using max_frequencies Matsubara frequencies.
    if tol is not None:
        return np.nan
    # if tol is None, return the value after max_frequencies is reached.
    else:
        return I


def I_c(syst, hopping, p, T, tol=1e-2, max_frequencies=200):
    H_0_cache = []
    fun = lambda phase: -current_at_phase(syst, hopping, p, T, H_0_cache,
                                          phase, tol, max_frequencies)
    opt = scipy.optimize.brute(
        fun, ranges=((-np.pi, np.pi),), Ns=30, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid, currents=-Jout)


def to_df(fname_start, vals_columns, remove_keys, save, fname):
    # Load all data from deepdish and convert it to pd.DataFrame
    files = [f for f in os.listdir('./') if f.startswith(fname_start) and f.endswith('.h5')]
    print(files)
    df = pd.DataFrame()
    for f in files:
        x = deepdish.io.load(f)
        df1 = pd.DataFrame(x.pop('vals'), columns=vals_columns)
        df2 = pd.DataFrame(x.pop('current_phase'))
        df_new = pd.concat([df1, df2], axis=1)

        for key in remove_keys:
            x.pop(key)

        for dim in x.keys():
            if isinstance(x[dim], dict):
                dic = {key: val for key, val in x[dim].items() if val is not None}
                df_new = df_new.assign(**dic)
            else:
                df_new[dim] = x[dim]

        df = df.append(df_new, ignore_index=True)

    if save:
        df.reindex().to_hdf(fname, 'all_data', mode='w')
    return df


def cylinder_sector(r_out, r_in=0, L=1, L0=0, phi=360, angle=0, a=10):
    """Returns the shape function and start coords.

    Parameters:
    -----------
    r_out : int
        Outer radius in nm.
    r_in : int
        Inner radius in nm.
    L : int
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int
        Start position in x.
    phi : int
        Coverage angle in degrees.
    angle : int
        Angle of tilting from top in degrees.
    a : int
        Discretization constant in nm.

    Returns:
    --------
    (shape_func, *(start_coords))
    """
    phi *= np.pi / 360
    angle *= np.pi / 180
    r1sq, r2sq = r_out**2, r_in**2

    def sector(pos):
        x, y, z = pos
        n = (y + 1j * z) * np.exp(1j * angle)
        y, z = n.real, n.imag
        rsq = y**2 + z**2

        shape_yz = r2sq <= rsq < r1sq and z >= np.cos(phi) * np.sqrt(rsq)
        return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
    r_mid = (r_out + r_in) / 2
    return sector, (L - a, r_mid * np.sin(angle), r_mid * np.cos(angle))


def square_sector(r_out, r_in=0, L=1, L0=0, phi=360, angle=0, a=10):
    """Returns the shape function and start coords.

    Parameters:
    -----------
    r_out : int
        Outer radius in nm.
    r_in : int
        Inner radius in nm.
    L : int
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int
        Start position in x.
    phi : int
        Coverage angle in degrees.
    angle : int
        Angle of tilting from top in degrees.
    a : int
        Discretization constant in nm.

    Returns:
    --------
    (shape_func, *(start_coords))
    """
    if r_in > 0:
        def sector(pos):
            x, y, z = pos
            shape_yz = -r_in <= y < r_in and r_in <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return sector, (L - a, 0, r_in + a)
    else:
        def sector(pos):
            x, y, z = pos
            shape_yz = -r_out <= y < r_out and -r_out <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return sector, (L - a, 0, 0)


@lru_cache()
def make_3d_wire(a, L, r1, r2, phi, angle, L_sc, disorder, with_vlead,
                 with_leads, with_shell, spin, holes, shape, A_in_SC):
    """Creates a cylindrical 3D wire partially covered with a superconducting (SC) shell, 
    but without superconductor in the scattering region of length L.

    Example arguments:
    ------------------
    (A_in_SC=True, a=10, angle=0, disorder=False, holes=True, L=30, L_sc=10,
     phi=185, r1=50, r2=70, shape='square', spin=True, with_leads=True,
     with_shell=True, with_vlead=True)    

    Note: we are not using default parameters because I want to save them to file,
    so I create a dictionary that is passed to the function.
    save to file.

    Parameters:
    -----------
    a : int
        Discretization constant in nm.
    L : int
        Length of wire (the scattering part without SC shell.) Should be bigger
        than 4 unit cells (4*a) to have the vleads in a region without a SC shell.
    r1 : int
        Radius of normal part of wire in nm.
    r2 : int
        Radius of superconductor in nm.
    phi : int
        Coverage angle of superconductor in degrees, if bigger than 180 degrees,
        the Peierls substitution fails.
    angle : int
        Angle of tilting of superconductor from top in degrees.
    disorder : bool
        When True, syst requires 'disorder' and 'salt' aguments.
    with_vlead : bool
        If True a SelfEnergyLead with zero energy is added to a slice of the system.
    with_leads : bool
        If True it appends infinite leads with superconducting shell.
    L_sc : int
        Number of unit cells that has a superconducting shell. If the system has
        infinite leads, set L_sc=a.
    with_shell : bool
        Adds shell the the correct areas. If False no SC shell is added and only
        a cylindrical wire will be created.
    shape : str
        Either `circle` or `square` shaped cross-section.
    A_in_SC : bool
        Performs the Peierls substitution in the superconductor. Can be True and False
        only when shape='square', and only True when shape='circle'.

    Returns:
    --------
    syst : kwant.builder.FiniteSystem
        The finilized system.
    hopping : function
        Function that returns the hopping matrix between the two cross sections
        of where the SelfEnergyLead is attached.
    """
    assert L_sc % a == 0
    assert L % a == 0
    if not A_in_SC and shape == 'circle':
        warnings.warn("Using shape='circle' and A_in_SC=False will result " + 
            "in incorrect fluxes at the interface of the SC and SM, however " +
            "the results will not differ qualitatively.")

    tb_normal, tb_sc, tb_interface = discretized_hamiltonian(a, spin, holes)
    lat = tb_normal.lattice
    syst = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))

    # The parts with a SC shell are not counted in the length L, so it's
    # modified as:
    L += 2*L_sc

    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise(NotImplementedError('Only square or circle wire cross-section allowed'))

    # Wire scattering region shapes
    shape_normal = shape_function(r_out=r1, angle=angle, L=L, a=a)
    # Superconductor slice in the beginning of the scattering region of L_sc
    # unit cells
    shape_sc_start = shape_function(
        r_out=r2, r_in=r1, phi=phi, angle=angle, L=L_sc, a=a)
    # Superconductor slice in the end of the scattering region of L_sc unit
    # cells
    shape_sc_end = shape_function(
        r_out=r2, r_in=r1, phi=phi, angle=angle, L0=L-L_sc, L=L, a=a)

    # Lead shapes
    shape_sc_lead = shape_function(
        r_out=r2, r_in=r1, phi=phi, angle=angle, L=-1, a=a)
    shape_normal_lead = shape_function(r_out=r1, angle=angle, L=-1, a=a)

    def onsite_dis(site, p):
        identity = np.eye(4) if spin and holes else np.eye(2)
        return p.disorder * (uniform(repr(site), repr(p.salt)) - 0.5) * identity

    # Add onsite terms in the scattering region
    syst[lat.shape(*shape_normal)] = lambda s, p: tb_normal.onsite(s, p) + (onsite_dis(s, p) if disorder else 0)

    if with_shell:
        syst[lat.shape(*shape_sc_start)] = tb_sc.onsite
        syst[lat.shape(*shape_sc_end)] = tb_sc.onsite

    # Add onsite terms in the infinite lead
    lead[lat.shape(*shape_normal_lead)] = tb_normal.onsite
    if with_shell:
        lead[lat.shape(*shape_sc_lead)] = tb_sc.onsite

    for hop, func in tb_normal.hoppings.items():
        # Add hoppings in normal parts of wire and lead with Peierls
        # substitution
        ind = np.argmax(hop.delta)  # Index of direction of hopping
        syst[hoppingkind_in_shape(hop, shape_normal, syst)] = peierls(func, ind, a)
        lead[hoppingkind_in_shape(hop, shape_normal_lead, lead)] = peierls(func, ind, a)

    if with_shell:
        for hop, func in tb_sc.hoppings.items():
            ind = np.argmax(hop.delta)  # Index of direction of hopping
            # Add hoppings in superconducting parts of wire and lead
            if A_in_SC:
                tmp_peierls = peierls
            else:
                tmp_peierls = lambda func, ind, a: func
            syst[hoppingkind_in_shape(hop, shape_sc_start, syst)] = tmp_peierls(func, ind, a)
            syst[hoppingkind_in_shape(hop, shape_sc_end, syst)] = tmp_peierls(func, ind, a)
            lead[hoppingkind_in_shape(hop, shape_sc_lead, lead)] = tmp_peierls(func, ind, a)

        for hop, func in tb_interface.hoppings.items():
            # Add hoppings at the interface of superconducting parts and normal
            # parts of wire and lead
            ind = np.argmax(hop.delta)  # Index of direction of hopping
            syst[hoppingkind_at_interface(
                hop, shape_sc_start, shape_normal, syst)] = peierls(func, ind, a)
            syst[hoppingkind_at_interface(
                hop, shape_sc_end, shape_normal, syst)] = peierls(func, ind, a)
            lead[hoppingkind_at_interface(
                hop, shape_sc_lead, shape_normal_lead, lead)] = peierls(func, ind, a)

    def cut(x_cut):
        """Return the sites at a cross section at x_cut."""
        sites = [lat(x, y, z)
                 for x, y, z in (i.tag for i in syst.sites()) if x == x_cut]
        return sorted(sites, key=lambda s: s.pos[2] * 10000 + s.pos[1])

    # Define left and right cut in wire in the middle of the wire, a region
    # without superconducting shell.
    l_cut = cut(L // (2*a) - 1)
    r_cut = cut(L // (2*a))
    num_orbs = 4
    dim = num_orbs * (len(l_cut) + len(r_cut))
    vlead = kwant.builder.SelfEnergyLead(
        lambda energy, args: np.zeros((dim, dim)), r_cut + l_cut)

    if with_vlead:
        syst.leads.append(vlead)
    if with_leads:
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    syst = syst.finalized()

    r_cut_sites = [syst.sites.index(site) for site in r_cut]
    l_cut_sites = [syst.sites.index(site) for site in l_cut]

    def hopping(syst, args=()):
        """Function that returns the hopping matrix of the electrons
        between the two cross sections."""
        return syst.hamiltonian_submatrix(args=args,
                                          to_sites=l_cut_sites,
                                          from_sites=r_cut_sites)[::2, ::2]
    return syst, hopping
