from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

if TYPE_CHECKING:
    from turbine import Turbine

class Estimator():
    def __init__(self, wt: Turbine, meas: list, states: list, req_state: list,
                 avail_states: list):
        """ Inits a Estimator object

        Prototype class for user defined Estimator objects. 
        Estimator converts the wind turbine measurements, m_wt, to the turbine 
        state, s_wt.

        Parameters
        ----------
        wt : Turbine
            Parent Turbine object.
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
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def reset(self):
        """
        Resets the Estimator to its original state.
        """
        pass
        # -------------------------------------------------------------------- #
        