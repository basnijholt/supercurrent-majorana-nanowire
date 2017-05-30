import hpc05
from time import sleep
client = hpc05.Client(profile='pbs2', timeout=60)
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

syst_pars = dict(a=8, angle=0, site_disorder=False, holes=True, phi=135,
                 r1=50, r2=70, shape='circle', with_shell=True, with_vlead=True)

params = dict(B_y=0, B_z=0, Delta=60, c_tunnel=5/8, V=lambda x: 0, **funcs.constants.__dict__)

Ts = [0.1, 0.5, 1]
orbital_bools = [False, True]
gs = [0, 50]
alphas = [0, 20]
mus = [10, 15, 20, 25, 30]
Ls = [80, 160, 320, 640]
leads = [(True, 8), (False, 400)]
Bs = np.linspace(0, 2, 101)

vals = funcs.named_product(T=Ts, L=Ls, orbital=orbital_bools,
                           g=gs, alpha=alphas, mu=mus, leads=leads, B_x=Bs)

def func(val, syst_pars=syst_pars, params=params):
    import funcs
    val['with_leads'], val['L_sc'] = val['leads']
    params = funcs.parse_params(dict(**params, **val))

    for x in ['with_leads', 'L_sc', 'L']:
        syst_pars[x] = params.pop(x)

    syst, hopping = funcs.make_3d_wire(**syst_pars)
    return dict(**funcs.I_c(syst, hopping, params), **val)

fname = "tmp/I_c(B_x)_no_disorder_combinations_of_effects_and_geometries_{}.hdf"
funcs.run_simulation(lview, func, vals, dict(**params, **syst_pars), fname, 2000)
