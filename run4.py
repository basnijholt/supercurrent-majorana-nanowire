import hpc05
from time import sleep
client = hpc05.Client(profile='pbs4', timeout=60)
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

syst_pars = dict(a=8, angle=0, site_disorder=True, holes=True, L=160, L_sc=8,
                 phi=135, r1=50, r2=70, shape='circle',
                 with_leads=True, with_shell=True, with_vlead=True)

params = dict(alpha=20, B_z=0, Delta=60, g=50, mu=20, salt=7,
              orbital=True, c_tunnel=5/8, V='lambda x: 0', T=100e-3, **funcs.constants.__dict__)

Bs = np.linspace(0, 2, 101)
thetas = [0, 45, 90]
vals = funcs.named_product(B=Bs, theta=thetas, disorder=[0, 75])

def func(val, syst_pars=syst_pars, params=params):
    import funcs
    import numpy as np
    angle = np.deg2rad(val['theta'])
    val['B_x'] = val['B'] * np.cos(angle).round(15)
    val['B_y'] = val['B'] * np.sin(angle).round(15)
    params = funcs.parse_params(dict(**params, **val))
    syst, hopping = funcs.make_3d_wire(**syst_pars)
    return dict(**funcs.I_c(syst, hopping, params), **val)

fname = 'data/I_c(B_x)_mu20meV_rotation_of_field_in_xy_plane.hdf'
funcs.run_simulation(lview, func, vals, dict(**syst_pars, **params), fname, None, overwrite=True)
