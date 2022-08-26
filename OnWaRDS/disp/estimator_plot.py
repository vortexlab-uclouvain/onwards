from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from OnWaRDS import farm

from OnWaRDS.turbine import MINIMAL_STATES 
lg = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

from .viz import Viz
from . import linespecs
if TYPE_CHECKING:
    from typing import List
    from ..farm import Farm

class Estimator_plot(Viz):
    def __init__(self, farm: Farm, states: List[str], measurements: List[str], 
                 labels: List[str], units: List[str], offset: List[float]=None, 
                 ylim: List[List[float, float]]=None, 
                 xlim: List[float, float]=None):
        super().__init__(farm)

        self.s_map    = states
        self.m_map    = measurements
        self.l_map    = [f'${l}\; [{u}]$' for l, u in zip(labels, units)]
        self.o_map    = offset or np.zeros(len(states))
        self.ylim_map = ylim   or [None]*len(states)
        self.map      = [ self.s_map, self.m_map, self.l_map, self.o_map, self.ylim_map ]

        self.xlim  = xlim 

        if not all(len(e)==len(states) for e in self.map):
            raise ValueError('All inputs should be the same length.')

        self.time = np.ones( len(farm) ) * np.nan
        ini = lambda s, m: np.ones( (2,len(farm)) ) * np.nan
        self.data = [ {s: ini(s,m) for s, m, *_ in zip(*self.map)} 
                                            for i_wt in range(self.farm.n_wts) ]
        
        for i_wt in range(self.farm.n_wts):
            for s, m, *_ in zip(*self.map):
                if s not in self.farm.wts[i_wt].states:
                    raise ValueError(f'Wind turbine state {s} not available.')
                if m and m not in self.farm.wts[i_wt].snrs:
                    raise ValueError(f'Sensor measurement {m} not available.')

        self._it = 0
        # -------------------------------------------------------------------- #

    def update(self):
        for i_wt in range(self.farm.n_wts):
            for s, m, *_ in zip(*self.map):
                self.data[i_wt][s][0, self._it] \
                             = self.farm.wts[i_wt].states[s]
                if m:
                    self.data[i_wt][s][1, self._it] \
                                = self.farm.wts[i_wt].snrs.get_buffer_data(m)
        self.time[self._it] = self.farm.t
        self._it += 1
        # -------------------------------------------------------------------- #

    def plot(self):
        self.data = [ {s: d[s][:2,:self._it-1] for s in d} for d in self.data ]
        self.time = self.time[:self._it-1]

        np.save(f'{self.farm.out_dir}/estimator_data.npy', self.data)
        np.save(f'{self.farm.out_dir}/estimator_time.npy', self.time)
        np.save(f'{self.farm.out_dir}/estimator_label.npy', self.l_map)

        for i_wt, d_wt in enumerate(self.data):
            _, axs = plt.subplots(len(d_wt), 1, squeeze=False)
            for ax, s, _, l, o, ylim in zip(axs[:,0], *self.map):
                plt.sca(ax)
                
                plt.plot(self.time,     d_wt[s][0], **linespecs.MOD)
                plt.plot(self.time + o, d_wt[s][1], **linespecs.REF)
                
                plt.xlim(self.xlim or self.time[[0,-1]])
                if ylim: plt.ylim(ylim)

                plt.ylabel(l)
            plt.xlabel('t [s]')
        # -------------------------------------------------------------------- #
