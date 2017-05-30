import hpc05
from time import sleep
client = hpc05.Client(profile='pbs5', timeout=60)
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


syst_pars = dict(a=10, L=500, L_sc=100)

params = dict(B_y=0, B_z=0, Delta=0.25, g=50, V='lambda x: 0', T=0.1,
              **funcs.constants.__dict__)

alphas = np.linspace(0, 100, 101)
B_xs = np.linspace(0, 1.5, 101)
mus = [0.1, 0.3, 1, 3, 10, 30]

vals = funcs.named_product(alpha=alphas, B_x=B_xs, mu=mus)

def func(val, syst_pars=syst_pars, params=params):
    import funcs
    syst, hopping = funcs.make_1d_wire(**syst_pars)
    params = funcs.parse_params(dict(**params, **val))
    return dict(**funcs.I_c(syst, hopping, params), **val)

funcs.run_simulation(lview, func, vals, dict(**params, **syst_pars),
                     'tmp/1d_alpha_vs_B_x_{}.hdf', N=20000, overwrite=True)
