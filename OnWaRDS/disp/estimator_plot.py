from __future__ import annotations
from typing import TYPE_CHECKING

import logging

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
                 xlim: List[float, float]=None,
                 diag: bool=True):
        super().__init__(farm)

        self.s_map    = states
        self.m_map    = measurements
        self.l_map    = [f'${l}\; [{u}]$' for l, u in zip(labels, units)]
        self.o_map    = offset or np.zeros(len(states))
        self.ylim_map = ylim   or [None]*len(states)
        self.map      = [ self.s_map, self.m_map, self.l_map, self.o_map, self.ylim_map ]

        self.xlim = xlim 
        self.diag = diag
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

    def reset(self):
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

    def __clean_data__(self):
        self.data = [ {s: d[s][:2,:self._it-1] for s in d} for d in self.data ]
        self.time = self.time[:self._it-1]
        # -------------------------------------------------------------------- #

    def export(self):
        self.__clean_data__()

        out = {'time':self.time,'label':self.l_map,'data':self.data}
        np.save(f'{self.farm.out_dir}/estimator_data.npy', out, allow_pickle=True)
        # -------------------------------------------------------------------- #

    def plot(self):
        if not self._was_exported: self.__clean_data__()

        for i_wt, d_wt in enumerate(self.data):
            _, axs = plt.subplots(len(d_wt), 1, squeeze=False)
            for ax, s, _, l, o, ylim in zip(axs[:,0], *self.map):
                plt.sca(ax)
                
                plt.plot(self.time,     d_wt[s][0], **linespecs.MOD)
                plt.plot(self.time + o, d_wt[s][1], **linespecs.REF)
                
                plt.xlim(self.xlim or self.time[[0,-1]])
                if ylim: plt.ylim(ylim)

                if any(~np.isnan(d_wt[s][1])) and self.diag:
                    idx_t0 = np.argmin(np.abs(self.time-self.time[0]-150))
                    v0 = d_wt[s][0][idx_t0:]
                    v1 = np.interp(self.time, self.time + o, d_wt[s][1])[idx_t0:]

                    norm = (np.mean(v1**2))**.5   

                    rho   = (np.mean((v0-np.mean(v0))*(v1-np.mean(v1))) \
                                                    /(np.std(v0)*np.std(v1)))
                    bias  = np.mean(v0-v1)/norm
                    err   = np.mean(np.abs(v0-v1))/norm
                    mape  = np.mean(np.abs((v0-v1)/v0))

                    buffer  = r'$\rho =' + f'{rho:.2f}'  + '$  '
                    buffer += r'$b ='    + f'{bias:.2f}' + '$  '
                    buffer += r'$e ='    + f'{err:.2f} ({v0.mean():.2g} / {v1.mean():.2g})'  + '$  '
                    buffer += r'$MAPE =' + f'{mape:.2f}' + '$'
                    plt.text( 0.975, 0.95, buffer,
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform = ax.transAxes )

                plt.ylabel(l)

            plt.xlabel('t [s]')
            self.savefig(f'estimator_wt{self.farm.wts[i_wt].i_bf}.pdf')
        # -------------------------------------------------------------------- #
