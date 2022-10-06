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

if TYPE_CHECKING:
    from turbine import Turbine

class Estimator():
    def __init__(self, wt: Turbine, meas: list, states: list, req_state: list,
                 avail_states: list):
        r""" Inits a Estimator object
        
        Prototype class for user defined Estimator objects. 
        Estimator converts th Turbine's measurements, m_wt, to the Turbine's
        state, s_wt.

        .. math::
            \mathbf{m}_{wt} \rightarrow \mathbf{\hat{s}}_{wt} 

        The ith state, s_wt^i, is computed from the sensors measurements possibly
        along with previously computed states, s_wt^J, with J : j < i

        .. math::
            \mathbf{\hat{s}}_{wt}^i
                = \mathbf{\hat{s}}_{wt}^i(\mathbf{m}_{wt}, \mathbf{\hat{s}}_{wt}^J)

        Parameters
        ----------
        wt : Turbine
            Parent :class:`.Turbine` object.
        meas : list
            List of the measurements meas required by the Estimator.
        states : list
            List of the Turbine's states computed by the Estimator.
        req_state : list
            List of the Turbine's states required by the Estimator.
        avail_states : list
            List of the Turbine's states available.

        Raises
        ------
        Exception
            If the requested Turbine's state is not (yet) available.
        Exception
            If the requested Turbine's measurement is not available.
        """        

        self.wt     = wt
        self.meas   = meas
        self.states = states

        for s in req_state:
            if s not in avail_states:
                raise Exception(f'State `{s}` not currently available in {avail_states}.')

        for m in meas:
            if m not in wt.snrs:
                raise Exception(f'Measurement `{m}` not currently available.')
        # -------------------------------------------------------------------- #

    def update(self):
        """
        Computes the current value of the Estimator's state(s).
        
        The corresponding state(s) should be updated in the parent Turbine object:
        ``self.wt.states['myState'] = myValue`` 
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def reset(self):
        """
        Resets the Estimator to its original state.
        """
        pass
        # -------------------------------------------------------------------- #
        