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
        meas   = ['T']
        states = ['ct']
        req_states = ['u_inc', 'yaw'] 
        super().__init__(wt, meas, states, req_states, avail_states)

    def update(self):
        ct_loc = self.wt.snrs.get_buffer_data('T') \
                        / (self.wt.states['u_inc']**2 * self.wt.af.cCTfac) * np.cos(self.wt.states['yaw'])
                        
        if ct_loc > CT_LIM: 
            # limit as defined by Moriarty PJ, Hansen AC. Aerodyn theory manual. tech. rep., Golden, CO (US), National Renewable Energy Lab.; 2005.
            lg.warning(f'Inconsistent CT = {ct_loc:2.3f} [-] value was encountered at t = {self.wt.t:2.0f} for WT{self.wt.i}')
            ct_loc = CT_LIM

        self.wt.states['ct'] = ct_loc

