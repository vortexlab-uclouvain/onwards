from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import os
import numpy as np
 
if TYPE_CHECKING:
    from ..turbine import Turbine

class StateExportBuffer():
    def __init__(self, wt: Turbine, export: str, export_dir=False, export_overwrite=False):

        self.export_dir  = export_dir if export_dir else \
                                                  f'{wt.farm.data_dir}/{export}/'
        if wt.i==0:
            if os.path.exists(self.export_dir):
                if not export_overwrite:
                    raise OSError(   f'Directory {self.export_dir} already exist. '
                                + f'Operation terminated to avoid data loss.')
            else:
                os.mkdir(self.export_dir)

        self.export_name =  f'wt_{wt.i_bf:02d}.npy'

        self.n_time = int(len(wt.snrs)/wt.n_substeps_est)
        self._idx   = -1
        
        self.states = wt.states

        self.data = {s: np.empty(self.n_time) for s in self.states}
        self.fs   = wt.fs
        
        self.t0 =  getattr(wt.snrs, 't0', 0.0)
        # -------------------------------------------------------------------- #

    def update(self):
        for s in self.data:
            self.data[s][self._idx] = self.states[s]
        self._idx += 1
        # -------------------------------------------------------------------- #

    def save(self):
        self.data['fs'] = self.fs
        self.data['t0'] = self.t0
        np.save( f'{self.export_dir}/{self.export_name}', self.data )
        # -------------------------------------------------------------------- #

