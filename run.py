import hpc05
from time import sleep
client = hpc05.Client(profile='pbs', timeout=60)
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

syst_pars = dict(a=8, angle=0, site_disorder=True, holes=False, L_sc=8,
                 phi=135, r1=50, r2=70, shape='circle', with_leads=True,
                 with_shell=False, with_vlead=False)

params = dict(alpha=20, B_x=0, B_y=0, B_z=0, Delta=0, g=50,
              orbital=True, V='lambda x: 0', **funcs.constants.__dict__)

Ls = np.arange(80, 2000, 80)
salts = np.arange(0, 10)
disorders = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110]
mus = np.arange(10, 31)

vals = funcs.named_product(salt=salts, disorder=disorders, L=Ls, mu=mus)

def func(val, syst_pars=syst_pars, params=params):
    import kwant, funcs
    params = funcs.parse_params(dict(**params, **val))

    for x in ['L']:
        syst_pars[x] = params.pop(x)

    syst, hopping = funcs.make_3d_wire(**syst_pars)
    smatrix = kwant.smatrix(syst, params=params)

    return dict(transmission=smatrix.transmission(0, 1),
                num_propagating=smatrix.num_propagating(0),
                **val)

fname = 'data/mean_free_path.hdf'
funcs.run_simulation(lview, func, vals, dict(**params, **syst_pars), fname, overwrite=True)