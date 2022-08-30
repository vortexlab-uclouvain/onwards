from __future__ import annotations
from typing import TYPE_CHECKING, List

import logging

from OnWaRDS import farm
lg = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

from .viz import Viz
from . import linespecs as ls
if TYPE_CHECKING:
    from typing           import List
    from ..farm           import Farm
    from ..turbine        import Turbine

class REWS_plot(Viz):
    farm: Farm

    def __init__(self, farm: Farm, t_ref:List(float), u_ref:List(float), 
                x_interp:List(float), fs_flag:bool=False, wt:Turbine=None, 
                xlim:List[float]=None, ylim:List[float]=None, 
                u_norm:float=None):
        super().__init__(farm)

        # Data initialisation
        self.t_mod = np.ones( len( farm ) ) *np.nan
        self.u_mod = np.ones( len( farm ) ) *np.nan

        self.fs_flag=fs_flag
        if self.fs_flag:
            self.u_mod_fs = np.ones(len(farm))*np.nan

        self.t_ref = t_ref
        self.u_ref = u_ref

        # Mask center computation
        self.x_interp = x_interp
        self.wt       = wt
        if wt:
            self.x_interp += wt.x
            
        self.xlim   = xlim
        self.ylim   = ylim
        self.u_norm = u_norm or farm.af.D 

        self._it = 0

        # -------------------------------------------------------------------- #

    def reset(self):
        self._it = 0
        # -------------------------------------------------------------------- #
        
    def update(self):
        if self.farm.update_LagSolver_flag:
            self.t_mod[self._it] = self.farm.t
            self.u_mod[self._it] = self.farm.lag_solver.rews_compute(self.x_interp, self.farm.af.R)
            if self.fs_flag:
                self.u_mod_fs[self._it] = self.farm.lag_solver.interp_FlowModel(
                                                                    np.array([self.x_interp[0]]), 
                                                                    np.array([self.x_interp[2]]), 
                                                                    filt='flow')[0]
            self._it += 1
        # -------------------------------------------------------------------- #
    
    def __lt__(self, other):
        return self.x_interp[0] < other.x_interp[0]
        # -------------------------------------------------------------------- #
    
    def __gt__(self, other):
        return other.x_interp[0] < self.x_interp[0]
        # -------------------------------------------------------------------- #

    def export(self):
        if self.wt:
            dx = self.x_interp[0] - self.wt.x[0] 
            str_id = f'wt{self.wt.i_bf}_{dx}'
        else:
            str_id = f'{self.x_interp[0]:.0f}_{self.x_interp[1]:.0f}_{self.x_interp[2]:.0f}'

        np.save(f'{self.farm.out_dir}/rews_{str_id}.npy', 
                { 't_mod':self.t_mod, 'u_mod':self.u_mod,
                  't_ref':self.t_ref, 'u_ref':self.u_ref },
                allow_pickle=True)
        # -------------------------------------------------------------------- #

    def plot(self):
        if self._it == -1: return

        normx = lambda _x: (_x)/(self.farm.af.D/self.u_norm)
        normy = lambda _y: (_y)/(self.u_norm)

        # Gather all REWS_plot objects
        viz_rews_all = [v for v in self.farm.viz if isinstance(v, REWS_plot)]

        # Gather and sort REWS_plot linked to a WT
        viz_rews_wt = [[] for _ in range(self.farm.n_wts)]
        for v in list(viz_rews_all):
            for i_wt, wt in enumerate(self.farm.wts):
                if wt==v.wt: 
                    viz_rews_wt[i_wt].append(v)
                    viz_rews_all.remove(v)

        for i in range(self.farm.n_wts):
            viz_rews_wt[i].sort()   
        
        # Plotting REWS_plot linked to a WT
        for i_wt, v_wt in enumerate(viz_rews_wt):
            i_wt_bf = self.farm.wts[i_wt].i_bf
            _, axs = plt.subplots(len(v_wt), 1, sharex=True, figsize=(8,8))

            for ax, v in zip(axs, v_wt):
                plt.sca(ax)
                plt.plot( normx(v.t_ref),
                          normy(v.u_ref),
                          **ls.REF)
                plt.plot( normx(v.t_mod[:v._it]),
                          normy(v.u_mod[:v._it]), 
                          **ls.MOD)
                if v.fs_flag:
                    plt.plot( normx(v.t_mod[:v._it]) ,
                              normy(v.u_mod_fs[:v._it]),
                               **ls.MOD | {'linestyle':'--'}) 

                dx = v.x_interp[0] - v.wt.x[0] 
                plt.ylabel(r'$\frac{1}{U_{ABL}}u_{RE}(t,'+f'{dx/v.farm.af.D:.1f}'+r'D)$')

                _ylim = [ np.floor(min(normx(v_wt[0].u_ref))),
                          max(np.ceil(max(normx(v_wt[0].u_ref))), 1.1) ]
                plt.xlim(v.xlim or normx(v.t_mod[:v._it][[0,-1]]))
                plt.ylim(v.ylim or _ylim)

                ax_last = ax

            plt.suptitle(f'WT{i_wt}')

            ax_last.set_xlabel(r't [s]' if self.u_norm is None else r'$\frac{t}{T_C}$') 
            ax_last.xaxis.set_tick_params(labelbottom=True)

            plt.tight_layout()
            plt.subplots_adjust(left=0.12, right=0.99, hspace=0.1)

            self.savefig(f'rews_wt{i_wt_bf}.pdf')

        for v_wt in viz_rews_all:
            raise NotImplementedError('plot not implement if no parent turbine selected.')

        for v in self.farm.viz:
            if isinstance(v, REWS_plot): v._it = -1
        # -------------------------------------------------------------------- #