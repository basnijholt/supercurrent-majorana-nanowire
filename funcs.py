# Standard library imports
from collections import namedtuple
from functools import lru_cache
from itertools import product, starmap
import subprocess
import types
import warnings

# Related third party imports
import discretizer
import kwant
from kwant.digest import uniform
import numpy as np
import scipy.constants
import scipy.optimize
import sympy
import sympy.physics
from sympy.physics.quantum import TensorProduct as kr


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_git_revision_hash():
    """Get the git hash to save with data to ensure reproducibility."""
    git_output = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return git_output.decode("utf-8").replace('\n', '')


def find_nearest(array, value):
    """Find the nearest value in an array to a specified `value`."""
    idx = np.abs(np.array(array) - value).argmin()
    return array[idx]


def drop_constant_columns(df):
    """Taken from http://stackoverflow.com/a/20210048/3447047"""
    return df.loc[:, (df != df.ix[0]).any()]


def gate(syst, V, gate_size):
    x_positions = sorted(set(i.pos[0] for i in syst.sites))
    x_mid = (max(x_positions) - min(x_positions)) / 2
    x_L = find_nearest(x_positions, x_mid - gate_size / 2)
    x_R = find_nearest(x_positions, x_mid + gate_size / 2)
    return lambda x: V if x > x_L and x <= x_R else 0



sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
s0 = sympy.eye(2)
s0sz = np.kron(s0, sz)
s0s0 = np.kron(s0, s0)

# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental
# constants, in a types.SimpleNamespace.
constants = types.SimpleNamespace(
    m_eff=0.015 * scipy.constants.m_e,  # effective mass in kg
    hbar=scipy.constants.hbar,
    m_e=scipy.constants.m_e,
    eV=scipy.constants.eV,
    meV=scipy.constants.eV * 1e-3,
    k=scipy.constants.k / (scipy.constants.eV * 1e-3),
    current_unit=scipy.constants.k * scipy.constants.e / scipy.constants.hbar * 1e9,  # to get nA
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (scipy.constants.eV * 1e-3),
    t=scipy.constants.hbar**2 / (2 * 0.015 * scipy.constants.m_e) / (scipy.constants.eV * 1e-3 * 1e-18),
    c=1e18 / (scipy.constants.eV * 1e-3))


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

    Parameters
    ----------
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

    Returns
    -------
    params : dict
        A simple container that is used to store Hamiltonian parameters.
    """
    p = types.SimpleNamespace(alpha=alpha,
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
    return p.__dict__


@lru_cache()
def discretized_hamiltonian(a=10, spin=True, holes=True, dim=3):
    """Discretize the the BdG Hamiltonian and returns
    functions used to construct a kwant system.

    Parameters
    ----------
    a : int, optional
        Lattice constant in nm.
    spin : bool, optional
        Add spin-space operators in the Hamiltonian.
    holes : bool, optional
        Add particle-hole operators in the Hamiltonian.
    dim : int, optional
        Spatial dimension of the system.

    Returns
    -------
    tb_normal, tb_sc, tb_interface : discretizer.Discretizer ojects
        Discretized Hamilonian functions of the semiconducting part,
        superconducting part, and for the interface, respectively.
    """
    k_x, k_y, k_z = discretizer.momentum_operators
    x, y, z = discretizer.coordinates
    t, B_x, B_y, B_z, mu_B, Delta, mu, alpha, g, V = sympy.symbols(
        't B_x B_y B_z mu_B Delta mu alpha g V', real=True)
    t_interface = sympy.symbols('t_interface', real=True)

    if dim == 1:
        k_y = k_z = 0
    if dim == 2:
        k_z = 0

    k = sympy.sqrt(k_x**2 + k_y**2 + k_z**2)

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

    args = dict(lattice_constant=a, discrete_coordinates=set('xyz'[:dim]))
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
    return ((i, j) for (i, j) in hoppingkind
            if at_interface(i, j, shape1, shape2))


def matsubara_frequency(n, params):
    """n-th fermionic Matsubara frequency at temperature T.

    Parameters
    ----------
    n : int
        n-th Matsubara frequency

    Returns
    -------
    float
        Imaginary energy.
    """
    return (2*n + 1) * np.pi * params['k'] * params['T'] * 1j


def null_H(syst, params, n):
    """Return the Hamiltonian (inverse of the Green's function) of
    the electron part at zero phase.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    params : dict
        A container that is used to store Hamiltonian parameters.
    n : int
        n-th Matsubara frequency

    Returns
    -------
    numpy.array
        The Hamiltonian at zero energy and zero phase."""
    en = matsubara_frequency(n, params)
    gf = kwant.greens_function(syst, en, out_leads=[0], in_leads=[0],
                               check_hermiticity=False, params=params)
    return np.linalg.inv(gf.data[::2, ::2])


def gf_from_H_0(H_0, t):
    """Returns the Green's function at a phase that is defined inside `t`.
    See doc-string of `current_from_H_0`.
    """
    H = np.copy(H_0)
    dim = t.shape[0]
    H[:dim, dim:] -= t.T.conj()
    H[dim:, :dim] -= t
    return np.linalg.inv(H)


def current_from_H_0(H_0_cache, H12, phase, params):
    """Uses Dyson’s equation to obtain the Hamiltonian for other
    values of `phase` without further inversions (calling `null_H`).

    Parameters
    ----------
    H_0_cache : list
        Hamiltonians at different imaginary energies.
    H12 : numpy array
        The hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    phase : float
        Phase at which the supercurrent is calculated.

    Returns
    -------
    float
        Total current of all terms in `H_0_list`.
    """
    I = sum(current_contrib_from_H_0(H_0, H12, phase, params) for H_0 in H_0_cache)
    return I


def I_c_fixed_n(syst, hopping, params, matsfreqs=5, N_brute=30):
    H_0_cache = [null_H(syst, params, n) for n in range(matsfreqs)]
    H12 = hopping(syst, params)
    fun = lambda phase: -current_from_H_0(H_0_cache, H12, phase, params)
    opt = scipy.optimize.brute(
        fun, ranges=[(-np.pi, np.pi)], Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid, currents=-Jout)


def current_contrib_from_H_0(H_0, H12, phase, params):
    """Uses Dyson’s equation to obtain the Hamiltonian for other
    values of `phase` without further inversions (calling `null_H`).

    Parameters
    ----------
    H_0 : list
        Hamiltonian at a certain imaginary energy.
    H12 : numpy array
        The hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    phase : float
        Phase at which the supercurrent is calculated.
    unit : float
        Constant that sets the unit of the current,
        use k*e/hbar to get in A.

    Returns
    -------
    float
        Current contribution of `H_0`.
    """
    t = H12 * np.exp(1j * phase)
    gf = gf_from_H_0(H_0, t - H12)
    dim = t.shape[0]
    H12G21 = t.T.conj() @ gf[dim:, :dim]
    H21G12 = t @ gf[:dim, dim:]
    return -4 * params['T'] * params['current_unit'] * (
        np.trace(H21G12) - np.trace(H12G21)).imag


def current_at_phase(syst, hopping, params, H_0_cache, phase,
                     tol=1e-2, max_frequencies=200):
    """Find the supercurrent at a phase using a list of Hamiltonians at
    different imaginary energies (Matsubara frequencies). If this list
    does not contain enough Hamiltonians to converge, it automatically
    appends them at higher Matsubara frequencies untill the contribution
    is lower than `tol`, however, it cannot exceed `max_frequencies`.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross sections
        of where the SelfEnergyLead is attached.
    params : dict
        A container that is used to store Hamiltonian parameters.
    H_0_cache : list
        Hamiltonians at different imaginary energies.
    phase : float
        Phase at which the supercurrent is calculated.
    tol : float
        Tolerance of the `current_at_phase` function.
    max_frequencies : int
        Maximum number of Matsubara frequencies.

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""

    H12 = hopping(syst, params)
    I = 0
    for n in range(max_frequencies):
        if len(H_0_cache) <= n:
            H_0_cache.append(null_H(syst, params, n))
        I_contrib = current_contrib_from_H_0(H_0_cache[n], H12, phase, params)
        I += I_contrib
        if I_contrib == 0 or tol is not None and abs(I_contrib / I) < tol:
            return I
    # Did not converge within tol using max_frequencies Matsubara frequencies.
    if tol is not None:
        return np.nan
    # if tol is None, return the value after max_frequencies is reached.
    else:
        return I


def I_c(syst, hopping, params, tol=1e-2, max_frequencies=200, N_brute=30):
    """Find the critical current by optimizing the current-phase
    relation.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    params : dict
        A container that is used to store Hamiltonian parameters.
    tol : float
        Tolerance of the `current_at_phase` function.
    max_frequencies : int
        Maximum number of Matsubara frequencies.
    N_brute : int
        Number of points at which the CPR is evaluated in the brute
        force part of the algorithm,

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""
    H_0_cache = []
    fun = lambda phase: -current_at_phase(syst, hopping, params, H_0_cache,
                                          phase, tol, max_frequencies)
    opt = scipy.optimize.brute(
        fun, ranges=((-np.pi, np.pi),), Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid, currents=-Jout)


def peierls(func, ind, a, c=constants):
    """Applies Peierls phase to the hoppings functions. This function only
    works if spin is present.

    Parameters
    ----------
    func : function
        Hopping function in certain direction.
    ind : int
        Index of xyz direction, corresponding to 0, 1, 2.
    a : int
        Lattice constant in nm.
    c : types.SimpleNamespace object, optional
        Namespace object that contains fundamental constants.

    Returns
    -------
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


def cylinder_sector(r_out, r_in=0, L=1, L0=0, phi=360, angle=0, a=10):
    """Returns the shape function and start coords for a wire with
    as cylindrical cross section.

    Parameters
    ----------
    r_out : int
        Outer radius in nm.
    r_in : int, optional
        Inner radius in nm.
    L : int, optional
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int, optional
        Start position in x.
    phi : int, optional
        Coverage angle in degrees.
    angle : int, optional
        Angle of tilting from top in degrees.
    a : int, optional
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    phi *= np.pi / 360
    angle *= np.pi / 180
    r1sq, r2sq = r_out**2, r_in**2

    def sector(site):
        x, y, z = site.pos
        n = (y + 1j * z) * np.exp(1j * angle)
        y, z = n.real, n.imag
        rsq = y**2 + z**2

        shape_yz = r2sq <= rsq < r1sq and z >= np.cos(phi) * np.sqrt(rsq)
        return (shape_yz and L0 <= x < L) if L > 0 else shape_yz

    r_mid = (r_out + r_in) / 2
    start_coords = np.array([L - a,
                             r_mid * np.sin(angle),
                             r_mid * np.cos(angle)])

    return sector, np.round(start_coords / a).astype(int)


def square_sector(r_out, r_in=0, L=1, L0=0, phi=360, angle=0, a=10):
    """Returns the shape function and start coords of a wire
    with a square cross section.

    Parameters
    ----------
    r_out : int
        Outer radius in nm.
    r_in : int
        Inner radius in nm.
    L : int
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int
        Start position in x.
    phi : ignored
        Ignored variable, to have same arguments as cylinder_sector.
    angle : ignored
        Ignored variable, to have same arguments as cylinder_sector.
    a : int
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    if r_in > 0:
        def sector(site):
            x, y, z = site.pos
            shape_yz = -r_in <= y < r_in and r_in <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return sector, (L - a, 0, r_in + a)
    else:
        def sector(site):
            x, y, z = site.pos
            shape_yz = -r_out <= y < r_out and -r_out <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return sector, (L - a, 0, 0)


@lru_cache()
def make_1d_wire(a, L, L_sc):
    """Create a 1D semiconducting wire of length `L` with superconductors
    of length `L_sc` on its ends.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    L : int
        Length of wire (the scattering semi-conducting part) in nm.
    L_sc : int
        Length of superconducting ends in nm.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross sections
        of where the SelfEnergyLead is attached.
    """
    tb_normal, tb_sc, _ = discretized_hamiltonian(a, dim=1)
    lat = tb_normal.lattice
    syst = kwant.Builder()
    syst[(lat(x) for x in range(-L_sc, 0))] = tb_sc.onsite
    syst[(lat(x) for x in range(0, L))] = tb_normal.onsite
    syst[(lat(x) for x in range(L, L+L_sc))] = tb_sc.onsite

    for hop, val in tb_normal.hoppings.items():
        syst[hop] = val

    l_cut = [lat(x) for x in np.squeeze([i.tag for i in syst.sites()]) if x == L//2]
    r_cut = [lat(x) for x in np.squeeze([i.tag for i in syst.sites()]) if x == L//2 + 1]
    num_orbs = 4
    dim = num_orbs * (len(l_cut) + len(r_cut))
    vlead = kwant.builder.SelfEnergyLead(lambda energy, args: np.zeros((dim, dim)), r_cut + l_cut)
    syst.leads.append(vlead)

    lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
    lead[lat(0)] = tb_sc.onsite

    for hop, val in tb_sc.hoppings.items():
        lead[hop] = val

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()

    r_cut_sites = [syst.sites.index(site) for site in r_cut]
    l_cut_sites = [syst.sites.index(site) for site in l_cut]

    def hopping(syst, params):
        return syst.hamiltonian_submatrix(params=params,
                                          to_sites=l_cut_sites,
                                          from_sites=r_cut_sites)[::2, ::2]
    return syst, hopping


@lru_cache()
def make_3d_wire(a, L, r1, r2, phi, angle, L_sc, site_disorder, with_vlead,
                 with_leads, with_shell, spin, holes, shape, A_in_SC):
    """Create a cylindrical 3D wire partially covered with a
    superconducting (SC) shell, but without superconductor in the
    scattering region of length L.

    Parameters
    ----------
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
        Coverage angle of superconductor in degrees.
    angle : int
        Angle of tilting of superconductor from top in degrees.
    site_disorder : bool
        When True, syst requires `disorder` and `salt` aguments.
    with_vlead : bool
        If True a SelfEnergyLead with zero energy is added to a slice of the system.
    with_leads : bool
        If True it appends infinite leads with superconducting shell.
    L_sc : int
        Number of unit cells that has a superconducting shell. If the system
        has infinite leads, set L_sc=a.
    with_shell : bool
        Adds shell to the correct areas. If False no SC shell is added and
        only a cylindrical wire will be created.
    shape : str
        Either `circle` or `square` shaped cross section.
    A_in_SC : bool
        Performs the Peierls substitution in the superconductor. Can be True
        and False only when shape='square', and only True when shape='circle'.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross sections
        of where the SelfEnergyLead is attached.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(A_in_SC=True, a=10, angle=0, site_disorder=False,
    ...                    holes=True, L=30, L_sc=10, phi=185, r1=50, r2=70,
    ...                    shape='square', spin=True, with_leads=True,
    ...                    with_shell=True, with_vlead=True)
    >>> syst, hopping = make_3d_wire(**syst_params)

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
        raise(NotImplementedError('Only square or circle wire cross section allowed'))

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
        mat = s0sz if spin and holes else s0 if spin else sz
        mat = np.array(mat).astype(complex)
        return p.disorder * (uniform(repr(site), repr(p.salt)) - 0.5) * mat

    # Add onsite terms in the scattering region
    syst[lat.shape(*shape_normal)] = (lambda s, p: tb_normal.onsite(s, p) +
                                      (onsite_dis(s, p) if site_disorder else 0))

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
        """Return the sites at a cross section at `x_cut`."""
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
