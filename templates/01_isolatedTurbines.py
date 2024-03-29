# Copyright (C) <2022> <Université catholique de Louvain (UCLouvain), Belgique>

# List of the contributors to the development of OnWaRDS: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OnWaRDS) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.
 
import os

import logging as lg
lg.basicConfig(level=lg.WARNING, ) # can be set to ERROR, WARNING, INFO ... 

import matplotlib.pyplot as plt

from onwards import Farm

from uBEM3D  import BEMSolverFast
bem_solver = BEMSolverFast( 'updatedBEMwithYaw_NREL_n64type2D' )

"""
01_isolatedTurbines.py
----------------------

A small wind farm: coupling sensing and flow state estimation.
8 NREL 5MW Turbines operating in (close-to) freestream conditions.

The Turbine's state is estimated from the Turbine's measurements (Actuator Disk
data extracted from LES simulations). 

.. math::
    \mathbf{\hat{s}}_{wt} = f_{WT}(t) 

Data is compared against the LES data in terms of:
    - Wind turbine state estimation accuracy;
    - Hub height velocity field;
    - Wake centerline position;
    - Rotor Effective Wind Speed;

DATA: The reference data is provided at from https://doi.org/10.14428/DVN/AUVUI6
      and should be stored under the DATA path.

WARNING: This script is only provided as an example.
         The uBEM3D library is not provided.
"""

DATA = f'{os.environ["ONWARDS_PATH"]}/templates/data/isolatedTurbines'
if not os.path.exists(f'{DATA}/geo.npy' ):
    raise Exception(f'Reference data not found at {DATA}. Please first download' + 
                    f'the reference data from https://doi.org/10.14428/DVN/AUVUI6')

# Sensors initialization
snrs_args = {
    'type': 'SensorsPy',
    'fs': 1,
}

# Definition of the Turbine's states estimators
est_args = {
    'n_substeps': 1,

    # Initial state
    'ini_states' : {'ct': 0, 'ti': 1, 'u_inc': 8,  'u_fs': 8},

    # Exports the Estimator data
    'export': {'name': 'snrs_buffer',
               'overwrite': True,
               'user_field': ['ux_d_p3D', 'ux_d_p6D', 'ux_d_p9D', 'ux_d_p12D', 
                                      'ux_d_m2D', 'uz_d_m2D', 'uz_d_r', 'ti_m2D']},

    # Esimator 0  Retrieves the transverse (z) velocity component, ``w_inc``, and 
    # ----------  the yaw angle, ``yaw``, directly from the Turbine's measurements
    'estimator0': {'type': 'fld_fromdata',
                   'meas_in':   ['uz_d_m2D', 'yawA'],
                   'state_out': ['w_inc', 'psi']},

    # Esimator 1  Estimates the rotor normal (x) velocity component, ``u_inc``, 
    # ----------  and Turbulence Intensity, ``ti``, from the blade loads. 2
    'estimator1': {'type': 'uincti_kfbem',
                   'n_sec': 8, 
                   'w_sec': 240,
                   'bem_args': {'type':
                                'fast',
                                'solver': bem_solver}},

    # Esimator 2  Estimates the thrust coefcient, ``ct`` from the incident wind  
    # ----------  speed and measured thrust 
    'estimator2': {'type': 'ct_fromthrust'},


    # Esimator 3  Updates the hub-height velocity estimates for waked wind turbines.
    # ----------  
    'estimator3': {'type': 'ufswfs_waked'},
}

# Lagrangian Model Parameters
model_args = {
    'n_substeps': 2,
    'n_fm': 30,
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

with Farm(DATA, 'NREL', snrs_args, est_args, model_args,
                        wt_cherry_picking=None, enable_logger=True) as wf:

    D = wf.af.D

    # Animation showing the position of the ambient and wake particles
    wf.viz_add('estimators', 
                ['u_inc',   'w_inc',   'ti'    ], 
                ['ux_d_m2D', 'uz_d_m2D', 'ti_m2D'], 
                ['u_{WT}',  'w_{WT}',  'TI'    ], 
                ['ms^{-1}', 'ms^{-1}', '\%'    ], 
                offset=[31.5, 0, 31.5], 
                ylim=[[6, 10],[-1, 1], [0,0.2]])

    # 2D hub-height velocity slice animation
    wf.viz_add('velfield', [3, 10], 0, data_fid=f'velocity/plane00_Vel_WF_01.npz',show=False)

    # Plot position of the wake centerline 3D, 6D and 9D behind the wind
    # turbine hub
    wf.viz_add('centerline_xloc', [3*D, 6*D, 9*D], bf_dir=DATA, u_norm=8.0)

    # Computes the Rotor Effective wind speed 3D, 6D and 9D behind the wind
    # turbine hub
    for wt in wf.wts:
        for xod in [3, 6, 9]:
            key = f'ux_d_p{xod}D'
            if key in wt.snrs:
                wf.viz_add('rews', [xod*D, 0, 0], wt=wt, u_norm=8,
                           t_ref=wt.snrs['time'], u_ref=wt.snrs[key])

    # Runs the simulation
    for t in wf: print(t)

    # Display the plots
    for v in wf.viz:
        v.plot()

    plt.draw()
    plt.pause(0.1)
    input("Press any key to exit")
    plt.close()

    print(f'Data was exported to {wf.out_dir}.')
