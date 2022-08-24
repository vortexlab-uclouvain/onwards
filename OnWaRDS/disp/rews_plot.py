from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

from .viz import Viz
from . import linespecs as ls
if TYPE_CHECKING:
    from ..farm           import Farm
    from ..lagSolver.grid import Grid

class REWS_plot(Viz):
    farm: Farm
    grid: Grid

    def __init__(self, farm, *args, **kwargs):
        super().__init__(farm)
        self.t_f3 = np.ones( len( farm ) ) *np.nan
        self.u_f3 = np.ones( len( farm ) ) *np.nan

        self.fs_flag = kwargs.get('fs',False)
        self.exp = kwargs.get('exp',1)
        if self.fs_flag:
            self.u_f3_fs = np.ones(len(farm))*np.nan

        self.ylabel = r'$\frac{u_{RE}}{u_H}$'
        if len(args)==1:
            self.wt   = farm.wts[farm.i2ibf(args[0])]
            self.t_bf = self.t_f3
            self.u_bf = np.ones(len(farm))*np.nan
            self.x    = self.wt.x
            self.x[0] -= 1
            self.title  = f'$WT${args[0]}'
            if self.exp==1: raise ValueError('exp not compatible with WT format')
        elif len(args)==3:
            self.wt = False
            self.t_bf, self.u_bf, self.x = args
            self.ylabel = r'$\frac{u_{RE}}{u_H}$'
            self.title  = f'[ $x = {self.x[0]}D$;  $z = {self.x[2]}D$ ]'
        else:
            raise TypeError('Inconsistent number of arguments')

        if 'filt' in kwargs: self.filt = 1/kwargs['filt']
        else:                self.filt = False
        self._it = 0

        # -------------------------------------------------------------------- #
    def update(self):
        if self.farm.update_LagSolver_flag:
            self.t_f3[self._it] = self.farm.t
            self.u_f3[self._it] = self.farm.lag_solver.rews_compute(self.x, self.farm.af.R)
            if self.fs_flag:
                self.u_f3_fs[self._it] = self.farm.lag_solver.interp_FlowModel(
                                                                    np.array([self.x[0]]), 
                                                                    np.array([self.x[2]]), 
                                                                    filt='flow')[0]
            if self.wt:
                self.u_bf[self._it] = self.wt.flow_est.inst_state['u']
            self._it += 1
        # -------------------------------------------------------------------- #

    def plot(self):
        plt.figure(figsize=(9,3))

        if self.filt:
            fs = 1/(self.t_f3[1]-self.t_f3[0])
            filt_f3 = signal.butter(5, self.filt / (fs / 2), 'low')

            fs = 1/(self.t_bf[1]-self.t_bf[0])
            filt_bf = signal.butter(5, self.filt / (fs / 2), 'low')

            self.u_f3    = signal.filtfilt(*filt_f3, self.u_f3)
            self.u_bf    = signal.filtfilt(*filt_bf, self.u_bf)
            if self.fs_flag:
                self.u_f3_fs = signal.filtfilt(*filt_f3, self.u_f3_fs)

        plt.plot(self.t_f3, self.u_f3, **ls.MOD) 
        if self.fs_flag:
            plt.plot(self.t_f3, self.u_f3_fs, **ls.MOD | {'linestyle':'--'}) 
        plt.plot(self.t_bf, self.u_bf, **ls.REF) 

        # plt.ylim([0.4, 1.2])
        plt.xlim([self.t_f3[0], self.t_f3[np.isnan(self.t_f3)==False][-1]])
        plt.xlabel(r'$\frac{t}{T_{\mathrm{conv}}}$')
        plt.ylabel(self.ylabel, rotation=0)
        plt.title(self.title)

        plt.tight_layout()
        buffer = f'wt{self.wt.i_bf}' if self.wt else ''

        # fid_tmpl = f'{self.farm.glob_set["log_dir"]}/rews_u{self.exp}_{buffer}x{(self.x[0]+self.farm.zero_origin[0]):0.0f}z{(self.x[2]+self.farm.zero_origin[2]):0.0f}'

        # np.save(f'{fid_tmpl}_u_f3.npy',   self.u_f3   )
        # np.save(f'{fid_tmpl}_u_bf.npy',   self.u_bf   )
        # np.save(f'{fid_tmpl}_t_f3.npy',   self.t_f3   )
        # np.save(f'{fid_tmpl}_t_bf.npy',   self.t_bf   )
        # np.save(f'{fid_tmpl}_u_f3_fs.npy',self.u_f3_fs)        
        
        # plt.savefig(f'{fid_tmpl}.eps')
        # -------------------------------------------------------------------- #