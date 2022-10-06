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
 
from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import os
import numpy as np
 
if TYPE_CHECKING:
    from typing    import List
    from ..turbine import Turbine

class StateExportBuffer():
    def __init__(self, wt: Turbine, export_args: dict):
        """ Saves the wind turbine estimated states fo future simulations. 

        Along with the :class:`.SensorsPreprocessed` class, it allows direct 
        feed through of the sensors measurements to the ``Farm.lag_solver``.

        This allows for fast computations when evaluating the performances 
        of the Lagrangian (ie: skips the turbine states estimation step).

        Parameters
        ----------
        wt : Turbine
            Parent Turbine object.
        export_args: dict
            Dictionary containing the Estimators's export parameters.
            
            Available options:
            
            :name:         *(str)*                  - 
                Name of the subdirectory (this subdirectory is created inside
                ``Farm.data_dir``).
            :overwrite:    *(bool, optional)*       - 
                Overwrite export data if set to True, by default False.
            :user_field:   *(List[str], optional)*  - 
                List of the measurements, m_wt, not part of the turbine state, 
                s_wt, that should be appended to the exported state vector. By
                default, empty list.
                
        Raises
        ------
        ValueError
            If no subdirectory ``name`` is provided.
        OSError
            If the export directory already exist.
        ValueError
            If a conflict is detected between ``user_field`` and ``Turbine.states``.

        Example
        -------
        Once an Estimator has been exported exported:
            >>> est_args =  { 
            >>>     'export' = { 'name' : 'my_dir',
            >>>                  'overwrite' : True,
            >>>                  'export_user_field': ['myField']
            >>>     ...
            >>>     }

        One may load it using:
            >>> snrs_args = {
            >>>     'type': 'SensorsPreprocessed',
            >>>     'name': 'my_dir'
            >>>     }  
        """

        lg.info('Initializing the Turbine\'s state export.')

        if 'name' not in export_args:
            raise ValueError('No output directory specified.')

        self.export_dir = f'{wt.farm.data_dir}/{export_args["name"]}/'
        if wt.i==0:
            if os.path.exists(self.export_dir):
                if not export_args.get('overwrite', False):
                    raise OSError(f'Directory {self.export_dir} already exist. '
                                + f'Operation terminated to avoid data loss'
                                + f'(switch overwrite to True).')
            else:
                os.mkdir(self.export_dir)

        self.export_name =  f'wt_{wt.i_bf:02d}.npy'

        self.n_time = int(len(wt.snrs)/wt.n_substeps_est)
        self._idx   = -1
        
        self.states      = wt.states
        self.states_user = []

        for s in export_args.get('user_field', []): 
            if s in self.states:
                raise ValueError('Conflicting export field ({s}) in states_user.')
            if s in wt.snrs:
                self.states_user.append(s)
            else:
                lg.warning(f'Field {s} not available in sensors for wt{wt.i_bf}.')
        self.wt = wt

        self.data = {s: np.empty(self.n_time) for s in 
                                    list(self.states.keys()) + self.states_user}
        self.fs   = wt.fs
        
        self.t0 =  getattr(wt.snrs, 't0', 0.0)
        # -------------------------------------------------------------------- #

    def update(self):
        """
        Updates the StateExportBuffer object.
        """
        for s in self.states:
            self.data[s][self._idx] = self.states[s]
        for s in self.states_user:
            self.data[s][self._idx] = self.wt.snrs.get_buffer_data(s)
        self._idx += 1
        # -------------------------------------------------------------------- #

    def save(self):
        """
        Exports the StateExportBuffer object.
        """
        self.data['fs'] = self.fs
        self.data['t0'] = self.t0
        np.save( f'{self.export_dir}/{self.export_name}', self.data )
        # -------------------------------------------------------------------- #

