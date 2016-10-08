from types import SimpleNamespace

import discretizer
import kwant
import numpy as np
import scipy.optimize
from scipy.constants import hbar, m_e, eV, physical_constants
import sympy
import sympy.physics
from sympy.physics.quantum import TensorProduct as kr

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

constants.t = (constants.hbar ** 2 / (2 * constants.m)) * \
    (1e18 / constants.meV)  # meV * nm^2
constants.mu_B = physical_constants['Bohr magneton'][0] / constants.meV


class SimpleNamespace(SimpleNamespace):

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


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


def peierls(func, ind, a, z_interface, c=constants):
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
    def with_phase(s1, s2, p):
        hop = func(s1, s2, p).astype('complex128')
        y1, y2, z1 = s1.tag[1], s2.tag[1], s1.tag[2]
        z0 = (z_interface[y1] + z_interface[y2]) / 2 - z1
        
        if p.orbital:
            phase = [0, z0 * p.B_x * a**2, 0][ind]
            phi = np.exp(-1j * 1e-18 * c.eV / c.hbar * phase)
            if hop.shape[0] == 2:
                hop *= phi
            elif hop.shape[0] == 4:
                hop *= np.array([phi, phi.conj(), phi, phi.conj()],
                                dtype='complex128')
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
        if tol is not None and abs(I_contrib / I) < tol:
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