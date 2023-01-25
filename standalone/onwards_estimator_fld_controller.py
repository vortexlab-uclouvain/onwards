from __future__ import annotations
from typing import TYPE_CHECKING

import logging
import numpy as np

lg = logging.getLogger(__name__)
from standalone.controller import TurbineDynamics 
from onwards.estimators.estimator import Estimator
if TYPE_CHECKING:
    from onwards.turbine import Turbine

RHO = 1.225

class Estimator_fld_controller(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        meas   = []
        from onwards.turbine import MINIMAL_STATES
        states = MINIMAL_STATES
        req_states = [] 
        super().__init__(wt, meas, states, req_states, avail_states)

        self.turbine_dynamics = TurbineDynamics(10, 0, wt.t-1/wt.fs, 0)
        self.ct_fac = wt.af.cCTfac

        self.seed = np.random.rand() * (2*np.pi)
        # -------------------------------------------------------------------- #

    def update(self):
        self.wt.states['u_inc'] = 8 + (self.wt.x[0]/(126*5))  + (self.wt.x[2]/(126*5)) + .1 * np.sin(self.wt.t/12 + self.seed) + 0.1 * np.sin(self.wt.t/5 + self.seed)
        self.wt.states['w_inc'] =     (self.wt.x[0]/(126*10)) + (self.wt.x[2]/(126*10)) +0.25 * np.sin(self.wt.t/32 + self.seed) + 0.1 * np.sin(self.wt.t/8 + self.seed)

        self.wt.states['ti'] = 0.1 + np.sin(self.wt.t/10)/10

        if self.wt.is_freestream():
            self.wt.states['u_fs'] = self.wt.states['u_inc']
            self.wt.states['w_fs'] = self.wt.states['w_inc']
        else:
            self.wt.states['u_fs'] = self.wt.farm.lag_solver.interp_FlowModel(
                np.array(self.wt.x[0]),
                np.array(self.wt.x[2]), filt='rotor', i_wt_exclude=self.wt.i)[0]
            self.wt.states['w_fs'] = self.wt.states['w_inc']

        self.turbine_dynamics.update(self.wt.t, self.wt.states['u_fs'])
        self.wt.states['ct'] = self.turbine_dynamics.drivetrain.T_aero / \
            (self.wt.states['u_fs']**2 * self.ct_fac)

        self.wt.states['psi'] = (np.arctan(self.wt.t-150)*2/np.pi + 1)/2. * np.deg2rad(20)

        # print(f'{self.wt.states["ct"]=} and {self.wt.states["yaw"]=}')
        # -------------------------------------------------------------------- #

