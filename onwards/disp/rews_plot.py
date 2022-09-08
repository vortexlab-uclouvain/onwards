from __future__ import annotations
from typing import TYPE_CHECKING, List

import logging

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

    def __init__(self, farm: Farm, t_ref: List(float), u_ref: List(float),
                 x_interp: float, fs_flag: bool = False, wt: Turbine = None,
                 xlim: List[float] = None, ylim: List[float] = None,
                 u_norm: float = None, diag: bool = True):
        """ Extracts the Rotor Effective Wind Speed (REWS) at the x_interp 
        streamwise location and compares it to the LES reference data.

        Parameters
        ----------
        farm : Farm
            Parent Farm Object
        t_ref : List
            Time vector of the reference LES data in [s]
        u_ref : List
            Time series of the reference REWS velocity from computed from the 
            LES data in [ms-1]        
        x_interp : List[float]
            Downstream streamwise (x) positions, the REWS should be evaluated at. 
        fs_flag : bool, optional
            True if the ambient (ie: no wake) REWS should also be computed, by 
            default False
        wt : Turbine, optional
            Index of the reference turbine, if a turbine is provided, the 
            position the REWS is evaluated at will be wt+x_interp, by default 
            None.        
        xlim : List[float], optional
            User defined time bounds for plotting, by default None.
        ylim : List[float], optional
            User defined REWS bounds for plotting, by default None.
        u_norm : float, optional
            Velocity used for data normalization with u_C=u_norm and T_C = D/u_norm, 
            if no u_norm is provided, no normalization is applied, by default None.
        diag : bool, optional
            If True, the correlation, error and MAPE are evaluated, by default True.

        """

        super().__init__(farm)

        # Data initialisation
        self.data['t_mod']     = np.ones( len( farm ) ) * np.nan
        self.data['u_mod'] = np.ones( len( farm ) ) * np.nan

        self.fs_flag=fs_flag
        if self.fs_flag:
            self.data['u_fs_mod'] = np.ones(len(farm)) * np.nan

        self.data['t_ref'] = t_ref
        self.data['u_ref'] = u_ref

        # Mask center computation
        self.x_interp = x_interp
        self.wt       = wt
        if wt:
            self.x_interp += wt.x
            
        self.xlim   = xlim
        self.ylim   = ylim
        self.u_norm = u_norm or farm.af.D 
        self.diag   = diag

        self._it = 0
        self._was_plot = False
        # -------------------------------------------------------------------- #

    def reset(self):
        self.data['t_mod'][:] = np.nan
        self.data['u_mod'][:] = np.nan
        self._it = 0
        # -------------------------------------------------------------------- #
        
    def update(self):
        if self.farm.update_LagSolver_flag:
            self.data['t_mod'][self._it] = self.farm.t
            self.data['u_mod'][self._it] = self.farm.lag_solver.rews_compute(self.x_interp, self.farm.af.R)
            if self.fs_flag:
                self.data['u_fs_mod'][self._it] = self.farm.lag_solver.interp_FlowModel(
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

    def _data_clean(self):
        self.data['t_mod']    = self.data['t_mod'][:self._it]
        self.data['u_mod']    = self.data['u_mod'][:self._it]
        self.data['u_fs_mod'] = self.data['u_fs_mod'][:self._it]
        # -------------------------------------------------------------------- #
    
    def _export(self):
        if self.wt:
            dx = self.x_interp[0] - self.wt.x[0] 
            str_id = f'wt{self.wt.i_bf}_{dx}'
        else:
            str_id = f'{self.x_interp[0]:.0f}_{self.x_interp[1]:.0f}_{self.x_interp[2]:.0f}'

        np.save(f'{self.farm.out_dir}/rews_{str_id}.npy', self.data, allow_pickle=True)
        # -------------------------------------------------------------------- #

    def plot(self):
        # Gather all REWS_plot objects
        if  self._was_plot: return
        
        viz_rews_all = [v for v in self.farm.viz if isinstance(v, REWS_plot)]
        for v in viz_rews_all:
            v.data_clean()

        normx = lambda _x: (_x)/(self.farm.af.D/self.u_norm)
        normy = lambda _y: (_y)/(self.u_norm)

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
            _, axs = plt.subplots(len(v_wt), 1, sharex=True, figsize=(8,8), squeeze=False)

            for ax, v in zip(axs[:,0], v_wt):
                plt.sca(ax)
                plt.plot( normx(v.data['t_ref']),
                          normy(v.data['u_ref']),
                          **ls.REF)
                plt.plot( normx(v.data['t_mod']),
                          normy(v.data['u_mod']), 
                          **ls.MOD)
                if v.fs_flag:
                    plt.plot( normx(v.data['t_mod']) ,
                              normy(v.data['u_fs_mod']),
                               **ls.MOD | {'linestyle':'--'}) 

                dx = v.x_interp[0] - v.wt.x[0] 
                plt.ylabel(r'$\frac{1}{U_{ABL}}u_{RE}(t,'+f'{dx/v.farm.af.D:.1f}'+r'D)$')

                _ylim = [ np.floor(min(normx(v_wt[0].data['u_ref']))),
                          max(np.ceil(max(normx(v_wt[0].data['u_ref']))), 1.1) ]
                plt.xlim(v.xlim or normx(v.data['t_mod'][[0,-1]]))
                plt.ylim(v.ylim or _ylim)

                if self.diag:
                    v0 = v.data_get( 'u', 'mod')
                    v1 = v.data_get( 'u', 'ref', t_interp=v.data['t_mod'])

                    norm = (np.mean(v1**2))**.5   

                    rho   = (np.mean((v0-np.mean(v0))*(v1-np.mean(v1))) \
                                                    /(np.std(v0)*np.std(v1)))
                    bias  = np.mean(v0-v1)/norm
                    err   = np.mean(np.abs(v0-v1))/norm
                    mape  = np.mean(np.abs((v0-v1)/v1))

                    buffer  = r'$\rho =' + f'{rho:.2f}'  + '$  '
                    buffer += r'$b ='    + f'{bias:.2f}' + '$  '
                    buffer += r'$e ='    + f'{err:.2f} ({((v0**2).mean())**.5:.2g} / {((v1**2).mean())**.5:.2g})'  + '$  '
                    buffer += r'$MAPE =' + f'{mape:.2f}' + '$'
                    plt.text( 0.975, 0.95, buffer,
                                horizontalalignment='right',
                                verticalalignment='top',
                                transform = ax.transAxes )

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
            if isinstance(v, REWS_plot): v._was_plot = True
        # -------------------------------------------------------------------- #

    def _data_get(self, field: str, source:str=None, t_interp:List[float]=None):
        if t_interp is None:
            return self.data[f'{field}_{source}']
        else:
            return np.interp(t_interp, self.data[f't_{source}'], self.data[f'{field}_{source}'])
        # -------------------------------------------------------------------- #