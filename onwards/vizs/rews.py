# Copyright (C) <2022> <Université catholique de Louvain (UCLouvain), Belgique>

# List of the contributors to the development of OnWaRDS: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OnWaRDS) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.
 
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

class Viz_rews(Viz):
    farm: Farm
    viz_type = 'rews'

    def __init__(self, farm: Farm, x_interp: float, t_ref: List(float) = None,
                 u_ref: List(float) = None, fs_flag: bool = False, wt: Turbine = None,
                 xlim: List[float] = None, ylim: List[float] = None,
                 u_norm: float = None, diag: bool = True):
        """ Extracts the Rotor Effective Wind Speed (REWS) at the ``x_interp`` 
        streamwise location and compares it to the LES reference data.

        Parameters
        ----------
        farm : Farm
            Parent :class:`.Farm` Object
        t_ref : List
            Time vector of the reference LES data in [s]
        u_ref : List
            Time series of the reference REWS velocity extracted from the LES 
            data in [ms-1]        
        x_interp : float
            Downstream streamwise (x) position where the REWS should be evaluated. 
        fs_flag : bool, optional
            True if the ambient (ie: no wake) REWS should also be computed, by 
            default False
        wt : Turbine, optional
            Index of the reference turbine: if a turbine is provided, the 
            position the REWS is evaluated at will be ``wt.x + x_interp``, 
            by default Done.        
        xlim : List[float], optional
            User defined time bounds for plotting, by default None.
        ylim : List[float], optional
            User defined REWS bounds for plotting, by default None.
        u_norm : float, optional
            Velocity used for data normalization T_C = D/u_norm, if no u_norm 
            (by default) is provided, no normalization is applied.
        diag : bool, optional
            If True (by default), the correlation, error and MAPE are evaluated.
        
        See also
        --------
        :meth:`.LagSolver.rews_compute`
        """

        super().__init__(farm)

        # Data initialisation
        self.data['t_mod'] = np.ones( len( farm ) ) * np.nan
        self.data['u_mod'] = np.ones( len( farm ) ) * np.nan

        self.fs_flag=fs_flag
        if self.fs_flag:
            self.data['u_fs_mod'] = np.ones(len(farm)) * np.nan

        self.ref_flag = t_ref is not None and u_ref is not None
        if self.ref_flag:
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
        self.diag   = diag and self.ref_flag

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
        if self.fs_flag:
            self.data['u_fs_mod'] = self.data['u_fs_mod'][:self._it]
        # -------------------------------------------------------------------- #
    
    def _export(self):
        if self.wt:
            dx = self.x_interp[0] - self.wt.x[0] 
            str_id = f'wt{self.wt.i_bf}_{dx}'
        else:
            str_id = f'{self.x_interp[0]:.0f}_{self.x_interp[1]:.0f}_{self.x_interp[2]:.0f}'

        self.__savenpy__(f'rews_{str_id}.npy', self.data, allow_pickle=True)
        # -------------------------------------------------------------------- #

    def plot(self):
        # Gather all Viz_rews objects
        if  self._was_plot: return
        
        viz_rews_all = [v for v in self.farm.viz if isinstance(v, Viz_rews)]
        for v in viz_rews_all:
            v.__data_clean__()

        normx = lambda _x: (_x)/(self.farm.af.D/self.u_norm)
        normy = lambda _y: (_y)/(self.u_norm)

        # Gather and sort Viz_rews linked to a WT
        viz_rews_wt = [[] for _ in range(self.farm.n_wts)]
        for v in list(viz_rews_all):
            for i_wt, wt in enumerate(self.farm.wts):
                if wt==v.wt: 
                    viz_rews_wt[i_wt].append(v)
                    viz_rews_all.remove(v)

        for i in range(self.farm.n_wts):
            viz_rews_wt[i].sort()   
        
        viz_rews_wt_map = [(i_wt, v_wt) for i_wt, v_wt in enumerate(viz_rews_wt) if v_wt]
        # Plotting Viz_rews linked to a WT
        for i_wt, v_wt in viz_rews_wt_map:
            i_wt_bf = self.farm.wts[i_wt].i_bf
            figsize = (8, len(v_wt)*2)
            _, axs = plt.subplots(len(v_wt), 1, sharex=True, figsize=figsize, squeeze=False)

            for ax, v in zip(axs[:,0], v_wt):
                plt.sca(ax)
                plt.plot( normx(v.data['t_mod']),
                          normy(v.data['u_mod']), 
                          **ls.MOD)
                if v.fs_flag:
                    plt.plot( normx(v.data['t_mod']) ,
                              normy(v.data['u_fs_mod']),
                               **ls.MOD | {'linestyle':'--'}) 
                if v.ref_flag:
                    plt.plot( normx(v.data['t_ref']),
                            normy(v.data['u_ref']),
                            **ls.REF)

                dx = v.x_interp[0] - v.wt.x[0] 
                plt.ylabel(r'$\frac{1}{U_{ABL}}u_{RE}(t,'+f'{dx/v.farm.af.D:.1f}'+r'D)$')

                _ylim = [ np.floor(min(normx(v_wt[0].data['u_mod']))),
                          max(np.ceil(max(normx(v_wt[0].data['u_mod']))), 1.1) ]
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

            self.__savefig__(f'rews_wt{i_wt_bf}.pdf')

        for v_wt in viz_rews_all:
            raise NotImplementedError('plot not implement if no parent turbine selected.')

        for v in self.farm.viz:
            if isinstance(v, Viz_rews): v._was_plot = True
        # -------------------------------------------------------------------- #

    def _data_get(self, field: str, source:str=None, t_interp:List[float]=None):
        if t_interp is None:
            return self.data[f'{field}_{source}']
        else:
            return np.interp(t_interp, self.data[f't_{source}'], self.data[f'{field}_{source}'])
        # -------------------------------------------------------------------- #