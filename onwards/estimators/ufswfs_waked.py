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

import numpy as np

from .estimator import Estimator
if TYPE_CHECKING:
    from turbine import Turbine

CT_LIM = 24/25

class Est_ufswfs_waked(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        """
        Corrects the ambient streamwise velocity if the wind turbine is waked. 

        :Input(s) states:       * Incident streamwise velocity component (``u_inc``) [ms-1] 
                                * Incident spanwise velocity component (``w_inc``) [ms-1] 
                                
        :Input(s) measurements: * None
        
        :State(s) computed:     * Estimated ambient streamwise velocity component (``u_fs``) [ms-1] 
                                * Estimated ambient spanwise velocity component (``w_fs``) [ms-1] 
        """

        meas   = []
        states = ['u_fs', 'w_fs']
        req_states = ['u_inc', 'w_inc'] 
        super().__init__(wt, meas, states, req_states, avail_states)

    def update(self):
        if self.wt.is_freestream(): 
            self.wt.states['u_fs'] = self.wt.states['u_inc']
            self.wt.states['w_fs'] = self.wt.states['w_inc']
        else:
            self.wt.states['u_fs'] = self.wt.farm.lag_solver.interp_FlowModel( 
                                                          np.array(self.wt.x[0]), 
                                                          np.array(self.wt.x[2]), filt='rotor', i_wt_exclude=self.wt.i)[0]
            self.wt.states['w_fs'] = self.wt.states['w_inc']




