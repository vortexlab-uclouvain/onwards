from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import numpy as np

from .estimator import Estimator
if TYPE_CHECKING:
    from turbine import Turbine

class Est_winc_nn(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, **kwargs):
        meas   = ['T']
        states = ['ct']
        req_states = ['u_inc', 'yaw'] 
        super().__init__(wt, meas, states, req_states, avail_states)

        # Retrieve the neural net from the farm : no NN history
        self.wt       = wt
        self.w_nn_net = self.wt.farm._w_nn
        self.states    = {'w': 0}

        # Initialize nn communicator    
        from .nnComm import NNCommunicator
        self.w_nn_comm = NNCommunicator(self.wt, self.wt.t, 1./self.wt.farm.dt_snrs, self.wt.farm.nn_set)
    
    def update(self):
        if not np.abs(self.w_nn_comm.t - self.wt.t)<1e-6:
            raise Exception(   'Time is not consistent across all submodules:\n'
                             + '   velEstimator root : {:2.2f} [s]\n'.format(self.wt.t)
                             + '   w_nn_comm         : {:2.2f} [s]'  .format(self.w_nn_comm.t) )
        buffer          = self.w_nn_comm.feed2NN()
        self.states['w'] = float(self.w_nn_net( buffer ))

    def update_snrs(self):
        self.w_nn_comm.iterate()