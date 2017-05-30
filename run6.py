import hpc05
from time import sleep
client = hpc05.Client(profile='pbs6', timeout=60)
print("Connected to hpc05"); sleep(2)
dview = client[:]

print('Connected to {} engines.'.format(len(dview)))

while len(dview) < 50:
    sleep(2)
    dview = client[:]
    print(len(dview))

dview.use_dill()
lview = client.load_balanced_view()
get_ipython().magic("px import sys, os; sys.path.append(os.path.expanduser('~/Work/nanowire_current/'))")

# Related third party imports
import kwant
import numpy as np
import pandas as pd

# Local imports
import funcs


syst_pars = dict(a=8, angle=0, site_disorder=True, holes=True,
                 L=640, L_sc=8, phi=135, r1=50, r2=70, shape='circle',
                 with_leads=True, with_shell=True, with_vlead=True)

params = dict(alpha=20, B_y=0, B_z=0, Delta=60, g=50, mu=20, orbital=True,
              c_tunnel=5/8, salt=7, T=100e-3, **funcs.constants.__dict__)

Bs = np.linspace(0, 0.5, 51)
Vs = np.linspace(0, 10, 51)
gate_sizes = [160]
disorders = [0, 75]

vals = funcs.named_product(gate_size=gate_sizes, disorder=disorders, V=Vs, B_x=Bs)


def func(val, syst_pars=syst_pars, params=params):
    import funcs
    syst, hopping = funcs.make_3d_wire(**syst_pars)
    params = funcs.parse_params(dict(**params, **val))
    params['V'] = funcs.gate(syst, params['V'], params['gate_size'])
    return dict(**funcs.I_c(syst, hopping, params), **val)

fname = "tmp/I_c(B_x,_V)_gate160nm_mu20meV_disorder0,75meV_T0.1K_c5over8_salt7_{}.h5"
funcs.run_simulation(lview, func, vals, dict(**params, **syst_pars), fname, 2500)
