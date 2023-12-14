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

class Est_ct_fromthrust(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        """
        Computes the thrust coefficient, ``ct``, from the measured thrust, ``T``, 
        and from the estimated incident velocity, ``u_inc``, and yaw angle, 
        ``yaw``. 

        :Input(s) states:       * Rotor incident velocity (``u_inc``) [ms-1]
                                * Yaw (``yaw``) [rad]

        :Input(s) measurements: * Thrust (``T``) [N]

        :State(s) computed:     * Thrust coefficient (``ct``) [-]
        
        """
        meas   = ['T']
        states = ['ct']
        req_states = ['u_inc'] 
        super().__init__(wt, meas, states, req_states, avail_states)
        # -------------------------------------------------------------------- #

    def update(self):
        ct_loc = self.wt.snrs.get_buffer_data('T') \
                        / (self.wt.states['u_inc']**2 * self.wt.af.cCTfac)
                        
        if ct_loc > CT_LIM: 
            # limit as defined by Moriarty PJ, Hansen AC. Aerodyn theory manual. tech. rep., Golden, CO (US), National Renewable Energy Lab.; 2005.
            lg.warning(f'Inconsistent CT = {ct_loc:2.3f} [-] value was encountered at t = {self.wt.t:2.0f} for WT{self.wt.i}')
            ct_loc = CT_LIM

        self.wt.states['ct'] = ct_loc
        # -------------------------------------------------------------------- #
        


