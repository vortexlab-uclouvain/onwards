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




