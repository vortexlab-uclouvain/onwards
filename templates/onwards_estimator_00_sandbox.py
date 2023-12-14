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
import numpy as np

lg = logging.getLogger(__name__)

from onwards import Estimator
if TYPE_CHECKING:
    from onwards import Turbine

class Estimator_00_sandbox(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        """
        Set the turbine state to user defined values.

        :Input(s) states:       * User defined (cfr :meth:`.Est_fld_debug.update`)

        :Input(s) measurements: * User defined (cfr :meth:`.Est_fld_debug.update`)
        
        :State(s) computed:     * User defined (cfr :meth:`.Est_fld_debug.update`)
        
        """
        meas   = []
        from onwards import MINIMAL_STATES
        states = MINIMAL_STATES
        req_states = [] 
        super().__init__(wt, meas, states, req_states, avail_states)
        # -------------------------------------------------------------------- #

    def update(self):

        self.wt.states['w_inc'] = 0.5 * np.sin(self.wt.t/32)
        self.wt.states['w_fs']  = 0.5 * np.sin(self.wt.t/32)
        self.wt.states['u_inc'] = 8 + .7 * np.sin(self.wt.t/12)  + 0.1 * np.sin(self.wt.t/5)
        self.wt.states['u_fs']  = 8 + .7 * np.sin(self.wt.t/12) + 0.1 * np.sin(self.wt.t/5)
        self.wt.states['ct']    = 0.8
        self.wt.states['ti']    = 0.1 + np.sin(self.wt.t/10)/10
        self.wt.states['psi']   = (np.arctan((self.wt.t-250)/100)*2/np.pi + 1)/2. * np.deg2rad(-45)
        
        # -------------------------------------------------------------------- #

