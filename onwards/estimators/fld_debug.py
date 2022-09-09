from __future__ import annotations
from typing import TYPE_CHECKING

import logging
import numpy as np

lg = logging.getLogger(__name__)

from .estimator import Estimator
if TYPE_CHECKING:
    from turbine import Turbine

class Est_fld_debug(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        """
        Set the turbine state to user defined values.
        """
        meas   = []
        from ..turbine import MINIMAL_STATES
        states = MINIMAL_STATES
        req_states = [] 
        super().__init__(wt, meas, states, req_states, avail_states)
        # -------------------------------------------------------------------- #

    def update(self):
        self.wt.states['w_inc'] = np.sin(self.wt.t/32)
        self.wt.states['w_fs']  = np.sin(self.wt.t/32)
        self.wt.states['u_inc'] = 8
        self.wt.states['u_fs']  = 8
        self.wt.states['ct']    = 0.8
        self.wt.states['ti']    = 0.1
        self.wt.states['yaw']   = 0
        # -------------------------------------------------------------------- #

