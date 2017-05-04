# Standard library imports
from functools import lru_cache
import operator
import subprocess
import types

# Related third party imports
import kwant
from kwant.continuum.discretizer import discretize
from kwant.digest import uniform
import numpy as np
import scipy.constants
import scipy.optimize
import sympy
import sympy.physics
from sympy.physics.quantum import TensorProduct as kr

# 3. Internal imports
from combine import combine

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
    e=scipy.constants.e,
    meV=scipy.constants.eV * 1e-3,
    k=scipy.constants.k / (scipy.constants.eV * 1e-3),
    current_unit=scipy.constants.k * scipy.constants.e / scipy.constants.hbar * 1e9,  # to get nA
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (scipy.constants.eV * 1e-3),
    t=scipy.constants.hbar**2 / (2 * 0.015 * scipy.constants.m_e) / (scipy.constants.eV * 1e-3 * 1e-18),
    c=1e18 / (scipy.constants.eV * 1e-3))


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


def make_params(alpha=20,
                B_x=0,
                B_y=0,
                B_z=0,
                Delta=0.25,
                mu=0,
                orbital=True,
                g=50,
                V=lambda x: 0,
                **kwargs):
    """Function that creates a namespace with parameters.

    Parameters
    ----------
    alpha : float
        Spin-orbit coupling strength in units of meV*nm.
    B_x, B_y, B_z : float
        The magnetic field strength in the x, y and z direction
        in units of Tesla.
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
    p = dict(alpha=alpha, B_x=B_x, B_y=B_y, B_z=B_z, Delta=Delta, mu=mu,
             orbital=orbital, t=t, g=g, mu_B=mu_B, V=V, **kwargs)
    return p


@lru_cache()
def discretized_hamiltonian(a, holes=True, dim=3):
    """Discretize the the BdG Hamiltonian and returns
    functions used to construct a kwant system.

    Parameters
    ----------
    a : int
        Lattice constant in nm.
    holes : bool, optional
        Add particle-hole operators in the Hamiltonian.
    dim : int, optional
        Spatial dimension of the system.

    Returns
    -------
    templ_normal, templ_sc, templ_interface : kwant.Builder ojects
        Discretized Hamilonian functions of the semiconducting part,
        superconducting part, and for the interface, respectively.
    """
    sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
    s0 = sympy.eye(2)
    k_x, k_y, k_z = kwant.continuum.momentum_operators
    x, y, z = kwant.continuum.position_operators
    B_x, B_y, B_z, Delta, mu, alpha, g, mu_B, hbar, V = sympy.symbols(
        'B_x B_y B_z Delta mu alpha g mu_B hbar V', real=True)
    m_eff = sympy.symbols('m_eff', commutative=False)

    # c should be (1e18 / constants.meV) if in nm and meV and c_tunnel
    # is a constant between 0 and 1 to reduce the hopping between the
    # interface of the SM and SC.
    c, c_tunnel = sympy.symbols('c, c_tunnel')

    if dim == 1:
        k_y = k_z = 0
    elif dim == 2:
        k_z = 0

    k = sympy.sqrt(k_x**2 + k_y**2 + k_z**2)
    kin = (1 / 2) * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff * c

    if holes:
        ham = ((kin - mu + V(x)) * kr(s0, sz) +
               alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
               0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0) + B_z * kr(sz, s0)) +
               Delta * kr(s0, sx))
    else:
        ham = ((kin - mu + V(x)) * sz +
               alpha * (k_y * sz - k_x * sz) +
               0.5 * g * mu_B * (B_x * s0 + B_y * s0 + B_z * s0) +
               Delta * sx)

    subs_sm = [(Delta, 0)]
    subs_sc = [(g, 0), (alpha, 0)]
    subs_interface = [(c, c * c_tunnel), (alpha, 0)]

    templ_sm = discretize(ham.subs(subs_sm), grid_spacing=a)
    templ_sc = discretize(ham.subs(subs_sc), grid_spacing=a)
    templ_interface = discretize(ham.subs(subs_interface), grid_spacing=a)

    return templ_sm, templ_sc, templ_interface


def get_cuts(syst, lat, x_left=0, x_right=1):
    """Get the sites at two postions of the specified cut coordinates.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    lat : dict
        A container that is used to store Hamiltonian parameters.
    """
    l_cut = [lat(*tag) for tag in [s.tag for s in syst.sites()] if tag[0] == x_left]
    r_cut = [lat(*tag) for tag in [s.tag for s in syst.sites()] if tag[0] == x_right]
    return l_cut, r_cut


def add_vlead(syst, lat, l_cut, r_cut):
    dim = lat.norbs * (len(l_cut) + len(r_cut))
    vlead = kwant.builder.SelfEnergyLead(lambda energy, args: np.zeros((dim, dim)), r_cut + l_cut)
    syst.leads.append(vlead)
    return syst


def hopping_between_cuts(syst, r_cut, l_cut):
    r_cut_sites = [syst.sites.index(site) for site in r_cut]
    l_cut_sites = [syst.sites.index(site) for site in l_cut]

    def hopping(syst, params):
        return syst.hamiltonian_submatrix(params=params,
                                          to_sites=l_cut_sites,
                                          from_sites=r_cut_sites)[::2, ::2]
    return hopping


def lat_from_temp(template):
    return next(iter(template.sites())).family


def add_disorder_to_template(template):
    # Only works with particle-hole + spin DOF or only spin.
    norbs = lat_from_temp(template).norbs
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    s0sz = np.kron(s0, sz)
    mat = s0sz if norbs == 4 else s0

    def onsite_dis(site, disorder, salt):
        return disorder * (uniform(repr(site), repr(salt)) - .5) * mat

    for site, onsite in template.site_value_pairs():
        onsite = template[site]
        template[site] = combine(onsite, onsite_dis, operator.add, 1)

    return template


def phase(site1, site2, B_x, B_y, B_z, orbital, e, hbar):
    x, y, z = site1.tag
    vec = site2.tag - site1.tag
    lat = site1.family
    a = np.max(lat.prim_vecs)  # lattice_contant
    A = [B_y * z - B_z * y, 0, B_x * y]
    A = np.dot(A, vec) * a**2 * 1e-18 * e / hbar
    phi = np.exp(-1j * A)
    if orbital:
        if lat.norbs == 2:  # No PH degrees of freedom
            return phi
        elif lat.norbs == 4:
            return np.array([phi, phi.conj(), phi, phi.conj()],
                            dtype='complex128')
    else:  # No orbital phase
        return 1


def apply_peierls_to_template(template):
    """Adds params['orbital'] argument to the hopping functions."""
    for (site1, site2), hop in template.hopping_value_pairs():
        lat = site1[0]
        a = np.max(lat.prim_vecs)
        template[site1, site2] = combine(hop, phase, operator.mul, 2)
    return template


def at_interface(site1, site2, shape1, shape2):
    return ((shape1[0](site1) and shape2[0](site2)) or
            (shape2[0](site1) and shape1[0](site2)))


def change_hopping_at_interface(syst, template, shape1, shape2):
    for (site1, site2), hop in syst.hopping_value_pairs():
        if at_interface(site1, site2, shape1, shape2):
            syst[site1, site2] = template[site1, site2]
    return syst


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
    params : dict
        A container that is used to store Hamiltonian parameters.

    Returns
    -------
    float
        Total current of all terms in `H_0_list`.
    """
    I = sum(current_contrib_from_H_0(H_0, H12, phase, params)
            for H_0 in H_0_cache)
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
    params : dict
        A container that is used to store Hamiltonian parameters.
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
                     tol=1e-3, max_frequencies=500):
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
    phase : float, optional
        Phase at which the supercurrent is calculated.
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
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


def I_c(syst, hopping, params, tol=1e-3, max_frequencies=500, N_brute=30):
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
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
        Maximum number of Matsubara frequencies.
    N_brute : int, optional
        Number of points at which the CPR is evaluated in the brute
        force part of the algorithm,

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""
    H_0_cache = []
    func = lambda phase: -current_at_phase(syst, hopping, params, H_0_cache,
                                           phase, tol, max_frequencies)
    opt = scipy.optimize.brute(
        func, ranges=((-np.pi, np.pi),), Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid,
                currents=-Jout, N_freqs=len(H_0_cache))


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
def make_1d_wire(a=10, L=400, L_sc=400):
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
    templ_normal, templ_sc, _ = discretized_hamiltonian(a, dim=1)
    lat = lat_from_temp(templ_normal)
    syst = kwant.Builder()

    def shape(x_left, x_right):
        return lambda s: x_left <= s.pos[0] < x_right, (x_left//a,)

    syst.fill(templ_sc, *shape(-L_sc, 0))
    syst.fill(templ_normal, *shape(0, L))
    syst.fill(templ_sc, *shape(L, L+L_sc))

    cuts = get_cuts(syst, lat, L//(2*a), (L//(2*a)+1))
    syst = add_vlead(syst, lat, *cuts)

    lead = kwant.Builder(kwant.TranslationalSymmetry([a]))
    lead.fill(templ_sc, lambda x: True, (0,))
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()

    hopping = hopping_between_cuts(syst, *cuts)
    return syst, hopping


@lru_cache()
def make_2d_test_system(X=2, Y=2):
    ham = "(hbar^2 * (k_x^2 + k_y^2) / (2 * m) * c - mu) * sigma_z + Delta * sigma_x"
    template_lead = kwant.continuum.discretize(ham)
    template = kwant.continuum.discretize(ham.replace('Delta', '0'))
    syst = kwant.Builder()
    syst.fill(template, lambda s: 0 <= s.tag[0] < X and 0 <= s.tag[1] < Y, (0, 0))
    lat = lat_from_temp(template)

    # Add 0 self energy lead
    cuts = get_cuts(syst, lat)
    syst = add_vlead(syst, lat, *cuts)

    # Leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    lead.fill(template_lead, lambda s: 0 <= s.pos[1] < Y, (0, 0))
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    syst = syst.finalized()

    hopping = hopping_between_cuts(syst, *cuts)
    return syst, hopping


@lru_cache()
def make_3d_test_system(X, Y, Z, a=10, test_hamiltonian=True):
    if test_hamiltonian:
        ham = '(t * (k_x**2 + k_y**2 + k_z**2) - mu) * sigma_z + Delta * sigma_x'
        templ_normal = kwant.continuum.discretize(ham.replace('Delta', '0'))
        templ_sc = kwant.continuum.discretize(ham)
    else:
        templ_normal, templ_sc, *_ = discretized_hamiltonian(a)

    lat = lat_from_temp(templ_normal)
    syst = kwant.Builder()
    syst.fill(templ_normal, lambda s: (0 <= s.pos[0] < X and 0 <= s.pos[1] < Y and
                                       0 <= s.pos[2] < Z), (0, 0, 0))

    cuts = get_cuts(syst, lat, 0, a)
    syst = add_vlead(syst, lat, *cuts)

    lead = kwant.Builder(kwant.TranslationalSymmetry((a, 0, 0)))
    lead.fill(templ_sc, lambda s: 0 <= s.pos[1] < Y and 0 <= s.pos[2] < Z, (0, 0, 0))

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    hopping = hopping_between_cuts(syst, *cuts)

    return syst, hopping


@lru_cache()
def make_3d_wire(a, L, r1, r2, phi, angle, L_sc, site_disorder, with_vlead,
                 with_leads, with_shell, shape, holes):
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
    L_sc : int
        Number of unit cells that has a superconducting shell. If the system
        has infinite leads, set L_sc=a.
    site_disorder : bool
        When True, syst requires `disorder` and `salt` aguments.
    with_vlead : bool
        If True a SelfEnergyLead with zero energy is added to a slice of the system.
    with_leads : bool
        If True it appends infinite leads with superconducting shell.
    with_shell : bool
        Adds shell to the correct areas. If False no SC shell is added and
        only a cylindrical wire will be created.
    shape : str
        Either `circle` or `square` shaped cross section.
    holes : bool
        Add particle-hole operators in the Hamiltonian. Turn off when calculating
        the mean-free path.

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
    ...                    L=30, L_sc=10, phi=185, r1=50, r2=70,
    ...                    shape='square', with_leads=True,
    ...                    with_shell=True, with_vlead=True, holes=True)
    >>> syst, hopping = make_3d_wire(**syst_params)

    """
    assert L_sc % a == 0
    assert L % a == 0

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

    # Create the system and the lead Builders
    syst = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0, 0)))

    # Create the templates with Hamiltonian and apply the Peierls subst. to it.
    templ_normal, templ_sc, templ_interface = map(
        apply_peierls_to_template, discretized_hamiltonian(a, holes=holes))
    lat = lat_from_temp(templ_normal)

    # Fill the normal part in the scattering region
    if site_disorder:
        syst.fill(add_disorder_to_template(templ_normal), *shape_normal)
    else:
        syst.fill(templ_normal, *shape_normal)

    # Fill in the infinite lead
    lead.fill(templ_normal, *shape_normal_lead)

    if with_shell:
        # Add the SC shell to the beginning and end slice of the scattering
        # region and to the lead.
        sites_sc = []
        sites_sc += syst.fill(templ_sc, *shape_sc_start)
        sites_sc += syst.fill(templ_sc, *shape_sc_end)
        sites_sc += lead.fill(templ_sc, *shape_sc_lead)

    # Define left and right cut in wire in the middle of the wire, a region
    # without superconducting shell.
    cuts = get_cuts(syst, lat, L // (2*a) - 1, L // (2*a))

    # Sort the sites in both lists
    cuts = [sorted(cut, key=lambda s: s.pos[1] + s.pos[2]*1e6) for cut in cuts]

    if with_vlead:
        syst = add_vlead(syst, lat, *cuts)

    if with_shell:
        # Adding a tunnel barrier between SM and SC
        syst = change_hopping_at_interface(syst, templ_interface,
                                           shape_normal, shape_sc_start)
        syst = change_hopping_at_interface(syst, templ_interface,
                                           shape_normal, shape_sc_end)
        lead = change_hopping_at_interface(lead, templ_interface,
                                           shape_normal_lead, shape_sc_lead)

    if with_leads:
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

    syst = syst.finalized()
    hopping = hopping_between_cuts(syst, *cuts)
    return syst, hopping
