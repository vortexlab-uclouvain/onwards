import os

import logging as lg
lg.basicConfig(level=lg.WARNING, ) # can be set to ERROR, WARNING, INFO ... 

import matplotlib.pyplot as plt

from onwards import Farm

# Definition of the Sensors
# (refer to :class:`.SensorsDecoy`)
snrs_args = {
    'type': 'SensorsDecoy',
    'fs': 1,
    'time_bnds': [0, 10]
}

# Estimator
# User defined Turbine's state (refer to :class:`.Est_fld_debug`)
est_args = {
    'n_substeps': 1,
    'estimator0': {'type': 'fld_controller'},
    'ini_states': {'ct': 0}
}

# Lagrangian Model Parameters
model_args = {
    'n_substeps': 2,
    'n_fm': 20,
    'n_wm': 40,
    'n_shed_fm': 8,
    'n_shed_wm': 8,
    'sigma_xi_r': 0.5,
    'sigma_r_f': 1,
    'sigma_t_r': 16/8,
    'sigma_xi_f': 4,
    'sigma_r_r': 1,
    'sigma_t_f': 16/8,
    'cw': 0.541,
    'c0': 0.729,
    'ak': 0.021,
    'bk': 0.039,
    'ceps': 0.201,
    'tau_r': 16, 
} 

grid_args = {'dx': 7.875, 'dz': 7.875, 'margin': [[2*126, 10*126],[2*126,2*126]]}

WF_DIR = f'{os.environ["ONWARDS_PATH"]}/templates/data/controller'

with Farm(WF_DIR, 'NREL', snrs_args, est_args, model_args, grid_args=grid_args, out_dir='/Users/lejeunemax/Desktop/onwards') as wf:

    D = wf.af.D

    # Animation showing the position of the ambient and wake particles
    wf.viz_add('power')

    # 2D hub-height velocity slice animation
    wf.viz_add('velfield', [3,10], 0)#, mp4_export=False)

    # Plot position of the wake centerline 3D, 6D and 9D behind the wind
    # turbine hub
    wf.viz_add('centerline_xloc', [3*126,6*126,9*126], u_norm=8.0)

    # Computes the Rotor Effective wind speed 3D, 6D and 9D behind the wind
    # turbine hub
    # for x in [3*D,6*D,9*D]:
    #     wf.viz_add('rews', [x, 0, 0], wt = wf.wts[0], u_norm=8)

    # Runs the simulation
    import time
    
    # start = time.time()
    # for i in range(100):
        # wf.reset(model_args)
    for t in wf:
        print(f'time = {t} [sec]')
    # print((time.time()-start)/100)

    # Display the plots
    for v in wf.viz:
        v.plot()

    plt.draw()
    plt.pause(0.1)
    input("Press any key to exit")
    plt.close()

    print(f'Data was exported to {wf.out_dir}.')