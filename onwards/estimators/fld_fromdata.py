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

from .estimator import Estimator
if TYPE_CHECKING:
    from turbine import Turbine

class Est_fld_fromdata(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        """
        Extract the turbine state directly from the Sensor data.

        :Input(s) states:       * None
        
        :Input(s) measurements: * User defined (cfr :meth:`.Est_fld_fromdata.update`)
        
        :State(s) computed:     * User defined (cfr :meth:`.Est_fld_fromdata.update`)
        """
        meas   = est_args['meas_in']
        states = est_args['state_out']
        req_states = [] 
        super().__init__(wt, meas, states, req_states, avail_states)

        self.probe = wt.snrs.get_buffer_data

    def update(self):
        for (s, m) in zip(self.states, self.meas):
            self.wt.states[s] = self.probe(m)
