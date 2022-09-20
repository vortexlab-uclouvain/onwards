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
