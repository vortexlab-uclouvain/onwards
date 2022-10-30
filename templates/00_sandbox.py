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

"""
00_sandbox.py
-------------

A simple OnWaRDS configuration.
The Farm configuration is described in sandbox/geo.txt

The Turbine's Sensors are overridden. SensorDecoy only provide the time discretization.

The Turbine's state is driven by the user defined function defined in 
Est_fld_debug.

.. math::
    \mathbf{\hat{s}}_{wt} = f_{user}(t) 

"""

# Definition of the Sensors
# (refer to :class:`.SensorsDecoy`)
snrs_args = {
    'type': 'SensorsDecoy',
    'fs': 1,
    'time_bnds': [0, 500]
}

# Estimator
# User defined Turbine's state (cfr: onwards_estimator_00_sandbox)
est_args = {
    'n_substeps': 1,
    'estimator0': {'type': '00_sandbox'},
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

WF_DIR = f'{os.environ["ONWARDS_PATH"]}/templates/data/sandbox'

with Farm(WF_DIR, 'NREL', snrs_args, est_args, model_args) as wf:

    D = wf.af.D

    # Animation showing the position of the ambient and wake particles
    # wf.viz_add('particles')

    # 2D hub-height velocity slice animation
    wf.viz_add('velfield', [3,10], 0)

    # Plot position of the wake centerline 3D, 6D and 9D behind the wind
    # turbine hub
    wf.viz_add('centerline_xloc', [3*126,6*126,9*126], u_norm=8.0)

    # Computes the Rotor Effective wind speed 3D, 6D and 9D behind the wind
    # turbine hub
    for x in [3*D,6*D,9*D]:
        wf.viz_add('rews', [x, 0, 0], wt = wf.wts[0], u_norm=8)

    # Runs the simulation
    for t in wf:
        print(f'time = {t} [sec]')

    # Display the plots
    for v in wf.viz:
        v.plot()

    plt.draw()
    plt.pause(0.1)
    input("Press any key to exit")
    plt.close()

    print(f'Data was exported to {wf.out_dir}.')