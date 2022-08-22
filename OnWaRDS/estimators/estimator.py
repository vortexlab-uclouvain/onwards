from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

if TYPE_CHECKING:
    from turbine import Turbine

class Estimator():
    def __init__(self, wt: Turbine, meas: list, states: list, req_state: list, avail_states: list):
        self.wt     = wt
        self.meas   = meas
        self.states = states

        for s in req_state:
            if s not in avail_states:
                raise Exception(f'State `{s}` not currently available in {avail_states}.')

        for m in meas:
            if m not in wt.snrs:
                raise Exception(f'Measurement `{m}` not currently available.')

    def update(self):
        raise NotImplementedError