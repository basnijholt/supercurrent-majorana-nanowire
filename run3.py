import hpc05
from time import sleep
client = hpc05.Client(profile='pbs3', timeout=60)
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

ONLY_PAPER_COMBINATIONS = False

syst_pars = dict(a=8, angle=0, site_disorder=True, holes=True, L=640,
                 L_sc=8, phi=135, r1=50, r2=70, shape='circle',
                 with_leads=True, with_shell=True, with_vlead=True)

params = dict(B_y=0, B_z=0, Delta=60, c_tunnel=5/8, V='lambda x: 0', T=100e-3, **funcs.constants.__dict__)

Bs = np.linspace(0, 0.5, 201)

if ONLY_PAPER_COMBINATIONS:
    # These are all the combinations of effects that are showed in Fig. 4 of the main paper.
    vals = [[(0, 0, True, 50, 20, 10, B),
             (0, 0, False, 50, 20, 10, B),
             (0, 0, True, 50, 0, 10, B),
             (0, 0, False, 50, 0, 10, B),
             (0, 0, True, 50, 20, 20, B),
             (0, 0, False, 50, 20, 20, B),
             (0, 0, True, 50, 0, 20, B),
             (0, 0, False, 50, 0, 20, B),
             (7, 75, True, 50, 20, 20, B)]
            for B in Bs]
    vals = sum(vals, [])
    names = ['salt', 'disorder', 'orbital', 'g', 'alphas', 'mu', 'B_x']
    vals = [dict(zip(names, val)) for val in vals]

else:
    vals = funcs.named_product(salt=np.arange(0, 8, 1), disorder=[0, 75], 
                               orbital=[False, True], g=[0, 50], alpha=[0, 20],
                               mu=[10, 20], B_x=Bs)
    # Filter out different salts when there is no disorder
    vals = [val for val in vals if not (val['salt'] > 0 and val['disorder'] == 0)]

print(len(vals))

def func(val, syst_pars=syst_pars, params=params):
    import funcs
    syst, hopping = funcs.make_3d_wire(**syst_pars)
    params = funcs.parse_params(dict(**params, **val))
    return dict(**funcs.I_c(syst, hopping, params), **val)

fname_i = "tmp/I_c(B_x)_mu10,20meV_disorder0,75meV_T0.1K_all_combinations_{}.hdf"
funcs.run_simulation(lview, func, vals, dict(**params, **syst_pars), fname_i, 2500)
