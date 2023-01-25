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
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import os

import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches

from scipy import interpolate

from .viz import Viz
from . import linespecs as ls

if TYPE_CHECKING:
    from typing import List
    from ..farm import Farm

from .centerline import Viz_centerline, I_MASK

class Viz_centerline_xloc(Viz_centerline):
    def __init__(self, farm: Farm, x_loc: List[float], bf_dir: str = None, 
                 i_mask: int = None, xlim: List[float] = None, ylim: List[float] = None,
                 u_norm: float = None, diag: bool = True):
        """ Extracts the position of the wake centerline at the ``x_loc`` location 
        and compares it to the LES reference data.

        Parameters
        ----------
        farm : Farm
            Parent :class:`.Farm` Object
        bf_dir : str
            Path to the reference LES data.
        i_mask : int, optional
            Index of the mask used for the wake centerline tracking for the LES 
            reference data, if None (by default), all available masks are imported.
        xlim : List[float], optional
            User defined time bounds for plotting, by default None.
        ylim : List[float], optional
            User defined streamwise position bounds for plotting, by default None.
        u_norm : float, optional
            Velocity used for data normalization T_C = D/u_norm, if no u_norm 
            (by default) is provided, no normalization is applied.
        diag : bool, optional
            If True (by default), the correlation, error and MAPE are evaluated.

        See also
        --------
        :class:`.Viz_centerline`

        """      
        super().__init__(farm, bf_dir, i_mask)
        self.x_loc  = x_loc
        self.x_lim  = xlim
        self.ylim   = ylim
        self.u_norm = u_norm or farm.af.D 
        self.diag   = diag and bf_dir is not  None
        # -------------------------------------------------------------------- #

    def _plot(self):
        t_mod = self.data_get('t', source='mod') 
        t_ref = self.data_get('t', source='ref') 
        
        for i_wt, wt in enumerate(self.farm.wts):
            
            x      = self.data_get('x', i_wt=i_wt) 
            zc_mod = self.data_get('zc', source='mod', i_wt=i_wt)
            zc_ref = [None]*self.n_masks

            for i_mask in range(self.n_masks):
                zc_ref[i_mask] = self.data_get( 'zc', 'ref', i_wt=i_wt, i_mask=i_mask)
            
            figsize = (8, len(self.x_loc)*2)
            fig, axs = plt.subplots(len(self.x_loc), 1, sharex=True,
                                      sharey=True, figsize=figsize, squeeze=False)

            for ax, x_ax in zip(axs, self.x_loc):
                ax = ax[0]

                # Checking if x_loc is valid for meandering
                x_wt_all = self.farm._load_geo(self.farm.data_dir)
                i_wt_row = np.where(np.abs(x_wt_all[:,2]-wt.x[2])<1E-3)
                x_wt_row = np.sort(x_wt_all[i_wt_row,0])
                row_idx  = np.argmin(np.abs(x_wt_row-wt.x[0]))+1                
                x_wt_next = x_wt_all[i_wt_row,0][0][row_idx] \
                                           if row_idx<len(x_wt_row[0]) else 10E9

                is_valid_wm = (x[0]<x_ax+wt.x[0]<x[-1]) and x_ax+wt.x[0]<x_wt_next

                plt.sca(ax)
                if is_valid_wm:# plotting valid data
                    idx_x     = np.argmin(np.abs(x-wt.x[0]-x_ax))

                    normx = lambda _x: (_x)/(self.farm.af.D/self.u_norm)
                    normy = lambda _y: (_y-wt.x[2])/self.farm.af.D

                    # plotting data
                    if self.bf_dir:
                        plt.plot(normx(t_ref), 
                                normy(zc_ref[I_MASK][:,idx_x]), 
                                **ls.REF)                    
                    plt.plot(normx(t_mod), 
                             normy(zc_mod[:,idx_x]), 
                             **ls.MOD)

                    plt.ylabel(r'$\frac{1}{D}z_C(t,'+f'{x_ax/self.farm.af.D:.1f}'+r'D)$')
                    
                    plt.xlim(self.x_lim or normx(self.data['t_mod'][[0,-1]]))
                    plt.ylim(self.ylim or [-1,1])

                    # wake envelope
                    _zc_ref = [normy(zc[:,idx_x]) for zc in zc_ref]
                    zc_ref_up  = np.max(_zc_ref, axis=0)
                    zc_ref_low = np.min(_zc_ref, axis=0)

                    ax.fill_between( normx(t_ref),
                                     zc_ref_low,
                                     zc_ref_up, 
                                     color=[0.7,0.7,0.7] )

                    if self.diag:
                        T0 = 250
                        idx_t_mod_diag = np.where(t_mod>T0)[0][0]

                        v0 = normy(self.data_get( 'zc', 'mod', i_wt=i_wt,                              ))[idx_t_mod_diag:,idx_x]
                        v1 = normy(self.data_get( 'zc', 'ref', i_wt=i_wt, t_interp=t_mod, i_mask=I_MASK))[idx_t_mod_diag:,idx_x]

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
                        plt.grid()

                    ax_last = ax
                else:
                    plt.axis('off')

            ax_last.set_xlabel(r't [s]' if self.u_norm is None else r'$\frac{t}{T_C}$') 
            ax_last.xaxis.set_tick_params(labelbottom=True)

            plt.tight_layout()
            plt.subplots_adjust(left=0.12, right=0.99, hspace=0.1)

            self.__savefig__(f'centerline_xloc_wt{wt.i_bf:02d}.pdf')
        # -------------------------------------------------------------------- #