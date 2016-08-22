"""
-----------------------------------------------------------------------
    Supercurrent in Josephson junction of rectangular form
-----------------------------------------------------------------------

Examples of usage:
-----------------

Construct a system, compute phase-current dependency and plot it.

>>> import matplotlib.pyplot as plt
>>> import current_matsubara as ic
>>> import test_system as ts
>>> params = ts.params(delta=0.01)
>>> sys = ts.make_system(W=10, L=10)
>>> phases, currents = ic.dep_current_phase(sys, params, T=0.001,\
...        npoints=101, nmatsfreq=1000)
>>> plt.figure()
>>> plt.plot(phases, currents)
>>> plt.show()

To visualize the system, do

>>> kwant.plot(sys)
"""

from __future__ import division
import kwant
import numpy as np
from sys import stdout

#Temporary
import sys as pysys

logging = False
def LOG(string):
    if logging:
        stdout.write("\r"+string)
        stdout.flush()

def _all_possible_replacements(params, **kwargs):
    """If params is a SimpleNamespace object, and other arguments are represented as iterables
    this function will return a generator of other params with all possible combinations of
    parameters from presented iterables. params shoud have defined __copy__ method. For example:

    >>> params = SimpleNamespace(a=0., b=0., c=0., d=0.)
    >>> gen = _all_possible_replacements(params, a=[1., 2.], b=[3., 4.])
    >>> for elem in gen:
    ...     print elem.__dict__

    will print following:

    {'a': 1.0, 'c': 0.0, 'b': 3.0, 'd': 0.0}
    {'a': 1.0, 'c': 0.0, 'b': 4.0, 'd': 0.0}
    {'a': 2.0, 'c': 0.0, 'b': 3.0, 'd': 0.0}
    {'a': 2.0, 'c': 0.0, 'b': 4.0, 'd': 0.0}

    params will stay unchanged.
    """
    if len(kwargs) == 0:
        yield params.__copy__()
        raise StopIteration
    key = kwargs.keys()[0]
    values = kwargs.pop(key)
    for value in values:
        nparams = params.__copy__()
        nparams.__dict__[key] = value
        # Recursion is a tool to explain recursion to bachelor sudents
        for p in _all_possible_replacements(nparams, **dict(kwargs)):
            yield p

def _phase_dependent_hopping(original_hopping):
    return lambda site1, site2, p: original_hopping*np.exp(1j*p.phase)

def _phase_dependent_hopping_inv(original_hopping):
    return lambda site1, site2, p: original_hopping*np.exp(-1j*p.phase)

def _ham_norb(sys, params):
    """Helper: calculates number of orbitals in system hamiltonian"""
    if hasattr(sys, "onsite_hamiltonians"):
        ham = sys.onsite_hamiltonians[0](sys.sites[0], params)
    elif hasattr(sys, "site_value_pairs"):
        site, func = next(iter(sys.site_value_pairs()))
        ham = func(site, params)
    else:
        raise ValueError("sys is not a proper Kwant system")
    try:
        norb = len(ham)
    except TypeError: #It is a number, not array
        norb = 1
    return norb

def mount_virtual_leads(sys, vlead1_interface, vlead2_interface, norb):
    """Mounts virtual leads to interfaces provided, and changes hopping between them
    to phase dependent one (introducing additional phase shift).
    TODO: better reorganize it to sequence of hoppings instead of two ingerfaces.

    :sys: kwant.builder.Builder
        An unfinalized system to mount (CAREFUL: it will change after using of this function.)
    :vlead1_interface: sequence of kwant.builder.Site
        Interface of first lead
    :vlead2_interface: sequence of kwant.builder.Site
        Interface of second lead
    :norb: integer
        Number of orbitals in system electrons (or holes) hamiltonian.
    :returns: tuple(kwant.builder.FiniteSystem, function)
        Returns a finalized system with virtual leads and function to precalculate hoppings
        between leads, if paramers (SimpleNamespace or similar) and phases list are provided.
    """

    dim1 = len(vlead1_interface)*norb
    dim2 = len(vlead2_interface)*norb
    zero_array_1 = np.zeros((dim1, dim1), dtype=float)
    zero_array_2 = np.zeros((dim2, dim2), dtype=float)
    def selfenergy_func_1(energy, args=()):
        return zero_array_1
    def selfenergy_func_2(energy, args=()):
        return zero_array_2

    vlead1 = kwant.builder.SelfEnergyLead(selfenergy_func_1, vlead1_interface)
    vlead2 = kwant.builder.SelfEnergyLead(selfenergy_func_2, vlead2_interface)
    sys.leads.append(vlead1)
    sys.leads.append(vlead2)

    # Replacing hoppings between virtual leads to phase dependent ones.
    hoppings_to_replace_12 = {}
    hoppings_to_replace_21 = {}
    for hop, val in sys.hopping_value_pairs():
        try:
            i = vlead2_interface.index(hop[0])
            j = vlead1_interface.index(hop[1])
            hoppings_to_replace_21[(i, j)] = _phase_dependent_hopping(val)
        except ValueError:
            try:
                i = vlead1_interface.index(hop[0])
                j = vlead2_interface.index(hop[1])
                hoppings_to_replace_12[(i, j)] = _phase_dependent_hopping_inv(val)
            except ValueError:
                pass

    for (i, j), val in hoppings_to_replace_21.iteritems():
        sys[vlead2_interface[i], vlead1_interface[j]] = val
    for (i, j), val in hoppings_to_replace_12.iteritems():
        sys[vlead1_interface[i], vlead2_interface[j]] = val

    def hopping_func(params):
        H21 = np.zeros((dim1, dim2), dtype=complex)
        H12 = np.zeros((dim2, dim1), dtype=complex)
        for (i, j), val in hoppings_to_replace_21.iteritems():
            hop = val(vlead2_interface[i], vlead1_interface[j], params)
            H21[i*norb:(i+1)*norb, j*norb:(j+1)*norb] = hop
            H12[j*norb:(j+1)*norb, i*norb:(i+1)*norb] = hop.conjugate().T
        for (i, j), val in hoppings_to_replace_12.iteritems():
            #TODO: test this case carefuly, typically only upper cycle works.
            hop = val(vlead1_interface[i], vlead2_interface[j], params)
            H12[i*norb:(i+1)*norb, j*norb:(j+1)*norb] = hop
            H21[j*norb:(j+1)*norb, i*norb:(i+1)*norb] = hop.conjugate().T
        return H21, H12

    return sys, hopping_func

def _matsubara_contribution(sys, hopping_func, params, T, k, phases):
    """ Calculates contribution of k-th Matsubara frequency to total supercurrent.

    :sys: kwant.system.FiniteSystem
        Finalized system with virtual leads attached by mount_virtual_leads.
    :hopping_func: function
        Function, returned by mount_virtual_leads as a second element in tuple.
        It calculates hopping arrays, getting params as an argument
    :params: SimpleNamespace or similar
        Parameter storage for system Hamiltonian.
    :T: float
        Temperature of system in energy units of Hamiltonian.
    :k: integer
        Number of Matsubara frequency.
    :phases: iterable of float
        Set of values of phase to calculate current in.
    :returns: numpy.ndarray
        Currents, generated by k-th Matsubara frequency at each value of phase.
    """
    en = (2*k+1)*np.pi*T*1j
    nleads = len(sys.leads)
    vl1 = nleads-2
    vl2 = nleads-1
    p = params.__copy__()
    currents = np.zeros_like(phases)
    for i, phase in enumerate(phases):
        p.phase = phase
        H21, H12 = hopping_func(p)
        if i == 0:
            G = kwant.greens_function(sys, en, args=(p,), in_leads=[vl1, vl2], out_leads=[vl1, vl2], check_hermiticity=False)
            dim1, dim2 = G.submatrix(vl1, vl2).shape
            I = np.identity(G.data.shape[0], dtype=float)
            G = G.data
        else:
            V = np.zeros_like(G)
            V[:dim1,dim1:] = H12 - H12_old
            V[dim1:,:dim1] = H21 - H21_old
            G = np.linalg.solve(I - np.dot(G, V), G)
        H12G21 = np.dot(H12, G[dim1:,:dim1])
        H21G12 = np.dot(H21, G[:dim1,dim1:])
        currents[i] = -4*T*np.imag(np.trace(H21G12-H12G21))
        H12_old = H12; H21_old = H21
    return currents

def _matsubara_contribution_exact(sys, hopping_func, params, T, k, phases):
    """ Calculates contribution of k-th Matsubara frequency to total supercurrent.
    Recalculates Green's function in every value of phase directly, provided mostly
    for referense. Slow as a snail.

    :sys: kwant.system.FiniteSystem
        Finalized system with virtual leads attached by mount_virtual_leads.
    :hopping_func: function
        Function, returned by mount_virtual_leads as a second element in tuple.
        It calculates hopping arrays, getting params as an argument.
    :params: SimpleNamespace or similar
        Parameter storage for system Hamiltonian.
    :T: float
        Temperature of system in energy units of Hamiltonian.
    :k: integer
        Number of Matsubara frequency.
    :phases: iterable of float
        Set of values of phase to calculate current in.
    :returns: numpy.ndarray
        Currents, generated by k-th Matsubara frequency at each value of phase.
    """
    en = (2*k+1)*np.pi*T*1j
    nleads = len(sys.leads)
    vl1 = nleads-2
    vl2 = nleads-1
    p = params.__copy__()
    currents = np.zeros_like(phases)
    for i, phase in enumerate(phases):
        p.phase = phase
        H21, H12 = hopping_func(p)
        G = kwant.greens_function(sys, en, args=(p,), in_leads=[vl1, vl2], out_leads=[vl1, vl2], check_hermiticity=False)
        dim1, dim2 = G.submatrix(vl1, vl2).shape
        G = G.data
        H12G21 = np.dot(H12, G[dim1:,:dim1])
        H21G12 = np.dot(H21, G[:dim1,dim1:])
        currents[i] = -4*T*np.imag(np.trace(H21G12-H12G21))
    return currents

def _matsubara_contribution_map(sys, hopping_func, T, k, params, phases, **kwargs):
    """Maps _matsubara_contribution function to all possible combination of named arguments,
    provided in kwargs as lists, putting them to params. Leads are precalculated once."""
    en = (2*k+1)*np.pi*T*1j
    psys = sys.precalculate(en, args=[params], what='selfenergy')

    out_dict = {}
    for p in _all_possible_replacements(params, **kwargs):
        out_dict[p] = _matsubara_contribution(psys, hopping_func, T, k, p, phases)

    return out_dict

def _matsubara_contribution_map_fluxes(sys, hopping_func, params, T, k, phases, fluxes):
    """Cleverly maps _matsubara_contribution function for all values of fluxes (no recalculation of
    leads each time. Returns numpy ndarray current_k[flux, phase]"""
    en = (2*k+1)*np.pi*T*1j
    p = params.__copy__()
    p.flux = 0.
    psys = sys.precalculate(energy=en, args=(p,), leads=range(len(sys.leads)-2), what='selfenergy')
    out = np.zeros((len(fluxes), len(phases)), dtype=float)
    for j, flux in enumerate(fluxes):
        p.flux = flux
        out[j] = _matsubara_contribution(psys, hopping_func, p, T, k, phases)
    return out


def _expected_current_correction(k, current_k):
    """Expected correction to current value of currents, if we assume, that sum over Matsubara frequencies
    converges as $N^{-1}$"""
    return (k-1)*current_k

def _abs_error(k, current_k):
    return np.max(np.abs(_expected_current_correction(k, current_k)))

def _rel_error(currents, k, current_k):
    return _abs_error(k, current_k) / np.max(np.abs(currents))

def dep_current_phase(sys, hopping_func, params, T,\
        phases=None, matsfreq_set=None, abs_error=None, rel_error=None,\
        max_nmatsfreq=10000):
    """Calculates current-phase dependensy for junction.
    It is assumed, that leads nomber 0 and 1 are virtual leads with zero self-energy,
    therefore Green's function is calculated on them.

    :sys: kwant.builder.FiniteSystem
        A finalized system with virtual leads mounted on an interface, at which we measure current.
    :hopping_func: function
        Function, returned by mount_virtual_leads as a second element in tuple.
        It calculates hopping arrays, getting params as an argument.
    :params: object
        Parameter object storage for system Hamiltonian (i.e. SimpleNamespace object).
    :T: float
        Temperature of system in energy units of Hamiltonian.
    :phases: iterable of floats or None
        List of points to calculate phases. If None, defaults to 101 homogeniously distributed points
        between -pi and pi
    :matsfreq_set: iterable or generator of integers or None
        Set of Matsubara frequencies to perform calculation in. If None, function will try
        to track convergense, according to provided abs_error or rel_error fields.
    :abs_error: float or None
        Absolute error, that is allowed for calculation. If None, relative error is used.
        Not used, if matsfreq_set is provided explicitly.
    :rel_error: float
        Relative error, that is allowed for calculation. Not used, if matsfreq_set or abs_error
        are provided.
    :max_nmatsfreq: integer
        Maximum Matsubara frequencies, allowed for calculation. If calculation doesn't
        converge up to it, RuntimeError is raised"
    :returns: tuple(numpy.ndarray, numpy.ndarray)
        Tuple with phases values in first elment and supercurrent values in second.
        Dirac constant and electron charge are assumed to be 1, energy unit equals to
        Hamiltonian units.
    """
    if phases is None:
        phases = np.linspace(-np.pi, np.pi, 101)
    _contrib = lambda k: _matsubara_contribution(sys, hopping_func, params, T, k, phases)
    if matsfreq_set is not None:
        LOG("Initializing")
        currents = np.zeros_like(phases)
        for k in matsfreq_set:
            currents = currents + _contrib(k)
            LOG("k={}".format(k))
        LOG("Finished\n")
        return phases, currents


    if abs_error is not None:
        LOG("Initializing")
        currents = _contrib(0)
        LOG("k=0")
        currents = currents + _contrib(1)
        LOG("k=1")
        k=1
        while True:
            k = k+1
            if k > max_nmatsfreq:
                raise RuntimeError("No convergense for supercurrent calculation. "
                        "{} frequensies were used, absolute error is {}, maximal allowed error {}."\
                        .format(k-1, error, abs_error))
            contrib = _contrib(k)
            currents = currents + contrib
            error = _abs_error(k, contrib)
            LOG("k={}: absolute error {}".format(k, error))
            if error < abs_error:
                # Add expected correction
                currents = currents + _expected_current_correction(k, contrib)
                LOG("Finished in {} steps: absolute error {}\n".format(k, error))
                return phases, currents

    if rel_error is not None:
        LOG("Initializing")
        currents = _contrib(0)
        LOG("k=0")
        currents = currents + _contrib(1)
        LOG("k=1")
        k=1
        while True:
            k = k+1
            if k > max_nmatsfreq:
                raise RuntimeError("No convergense for supercurrent calculation. "
                        "{} frequensies were used, relative error is {}, maximal allowed error {}."\
                        .format(k-1, error, rel_error))
            contrib = _contrib(k)
            currents = currents + contrib
            error = _rel_error(currents, k, contrib)
            LOG("k={}: relative error {}".format(k, error))
            if error < rel_error:
                # Add expected correction
                currents = currents + _expected_current_correction(k, contrib)
                LOG("Finished in {} steps: relative error {}\n".format(k, error))
                return phases, currents

    raise RuntimeError("You should provide matsfreq_set, abs_error or rel_error")

def ccurrent(sys, hopping_func, params, T, phases=None, matsfreq_set=None, abs_error=None,\
        rel_error=None, max_nmatsfreq=10000):
    """Calculates critical current of a junction.
    It is assumed, that leads nomber 0 and 1 are virtual leads with zero self-energy,
    therefore Green's function is calculated on them.

    :sys: kwant.builder.FiniteSystem
        A finalized system with virtual leads mounted on an interface, at which we measure current.
    :hopping_func: function
        Function, returned by mount_virtual_leads as a second element in tuple.
        It calculates hopping arrays, getting params as an argument.
    :params: object
        Parameter object storage for system Hamiltonian (i.e. SimpleNamespace object).
    :T: float
        Temperature of system in energy units of Hamiltonian.
    :phases: iterable of floats or None
        List of points to calculate phases. If None, defaults to 101 homogeniously distributed points
        between -pi and pi
    :matsfreq_set: iterable or generator of integers or None
        Set of Matsubara frequencies to perform calculation in. If None, function will try
        to track convergense, according to provided abs_error or rel_error fields.
    :abs_error: float or None
        Absolute error, that is allowed for calculation. If None, relative error is used.
        Not used, if matsfreq_set is provided explicitly.
    :rel_error: float
        Relative error, that is allowed for calculation. Not used, if matsfreq_set or abs_error
        are provided.
    :max_nmatsfreq: integer
        Maximum Matsubara frequencies, allowed for calculation. If calculation doesn't
        converge up to it, RuntimeError is raised"
    :returns: numpy.float64
        A value of critical current, assuming Dirac constant and electron charge equal 1.
        Energy unit equals to Hamiltonian units.
    """
    phases, currents = dep_current_phase(sys, hopping_func, params, T,\
        phases, matsfreq_set, abs_error, rel_error, max_nmatsfreq=10000)
    return np.max(np.abs(currents))

def _dep_current_flux_ccur(cur):
    """Helper, converts currents[flux, phase] to critical_currents[flux]"""
    return np.max(np.abs(cur), axis=1)

def dep_ccurrent_flux(sys, hopping_func, params, T, fluxes, phases=None, matsfreq_set=None,\
        abs_error=None, rel_error=None, max_nmatsfreq=10000):
    """Calculates critical current(flux) dependensy for junction.

    :sys: kwant.builder.FiniteSystem
        A finalized system with virtual leads mounted on an interface, at which we measure current.
    :hopping_func: function
        Function, returned by mount_virtual_leads as a second element in tuple.
        It calculates hopping arrays, getting params as an argument.
    :params: object
        Parameter object storage for system Hamiltonian (i.e. SimpleNamespace object).
    :T: float
        Temperature of system in energy units of Hamiltonian.
    :fluxes: iterable of floats
        List of flux values to calculate depencency in, in units of flux quanta per unit cell.
    :phases: iterable of floats or None
        List of points to calculate phases. If None, defaults to 101 homogeniously distributed points
        between -pi and pi
    :matsfreq_set: iterable or generator of integers or None
        Set of Matsubara frequencies to perform calculation in. If None, function will try
        to track convergense, according to provided abs_error or rel_error fields.
    :abs_error: float or None
        Absolute error, that is allowed for calculation. If None, relative error is used.
        Not used, if matsfreq_set is provided explicitly.
    :rel_error: float
        Relative error, that is allowed for calculation. Not used, if matsfreq_set or abs_error
        are provided.
    :max_nmatsfreq: integer
        Maximum Matsubara frequencies, allowed for calculation. If calculation doesn't
        converge up to it, RuntimeError is raised"
    :returns: numpy.ndarray
        List of critical current values for correspondent values of fluxes provided.
        Dirac constant and electron charge are assumed to be 1, energy unit equals to
        Hamiltonian units.
    """
    if phases is None:
        phases = np.linspace(-np.pi, np.pi, 101)
    _contrib = lambda k: _matsubara_contribution_map_fluxes(sys, hopping_func, params, T, k, phases, fluxes)

    if matsfreq_set is not None:
        LOG("Initializing")
        currents = 0.
        for k in matsfreq_set:
            currents = currents + _contrib(k)
            LOG("k={}".format(k))
        LOG("Finished\n")
        return _dep_current_flux_ccur(currents)

    if abs_error is not None:
        LOG("Initializing")
        currents = _contrib(0)
        LOG("k=0")
        currents = currents + _contrib(1)
        LOG("k=1")
        k=1
        while True:
            k = k+1
            if k > max_nmatsfreq:
                raise RuntimeError("No convergense for supercurrent calculation. "
                        "{} frequensies were used, absolute error is {}, maximal allowed error {}."\
                        .format(k-1, error, abs_error))
            contrib = _contrib(k)
            currents = currents + contrib
            error = _abs_error(k, contrib)
            LOG("k={}: absolute error {}".format(k, error))
            if error < abs_error:
                # Add expected correction
                currents = currents + _expected_current_correction(k, contrib)
                LOG("Finished in {} steps: absolute error {}".format(k, error))
                return _dep_current_flux_ccur(currents)

    if rel_error is not None:
        LOG("Initializing")
        currents = _contrib(0)
        LOG("k=0")
        currents = currents + _contrib(1)
        LOG("k=1")
        k=1

        while True:
            k = k+1
            if k > max_nmatsfreq:
                raise RuntimeError("No convergense for supercurrent calculation. "
                        "{} frequensies were used, absolute error is {}, maximal allowed error {}."\
                        .format(k-1, error, abs_error))
            contrib = _contrib(k)
            currents = currents + contrib
            error = _rel_error(currents, k, contrib)
            LOG("k={}: relative error {}".format(k, error))
            if error < rel_error:
                # Add expected correction
                currents = currents + _expected_current_correction(k, contrib)
                LOG("Finished in {} steps: relative error {}\n".format(k, error))
                return _dep_current_flux_ccur(currents)

    raise RuntimeError("You should provide matsfreq_set, abs_error or rel_error")
