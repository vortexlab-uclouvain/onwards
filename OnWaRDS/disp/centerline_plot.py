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

from readCenterlineForWM import Centerline

if TYPE_CHECKING:
    from typing import List
    from ..farm import Farm

I_MASK = 0

class _WakeCenterline(Viz):
    def __init__(self, farm: Farm, bf_dir: str, wm_str_id: str):
        super().__init__(farm)

        # check for previous _WakeCenterline initialization
        
        data = next((v for v in self.farm.viz if isinstance(v, _WakeCenterline)), None)

        if data: # data was already imported
            self.x       = data.x

            self.t_ref   = data.t_ref
            self.t_mod   = data.t_mod

            self.zc_ref  = data.zc_ref
            self.zc_mod  = data.zc_mod

            self.n_masks = data.n_masks

            self._it = None

        else :   # data needs to be imported
            wm_dir = f'{bf_dir}/WM_centerline/'

            wm_fid = [fid for fid in os.listdir(wm_dir) if fid.startswith(wm_str_id)]
            it0_str = wm_fid[0].rsplit('_',1)[1]

            gaussian_flag = 'gaussian' if 'gaussian' in wm_str_id else False
            self.n_masks  = len([fid for fid in wm_fid if fid.rsplit('_',2)[-2]=='w00'])
        
            self.x      = np.empty(farm.n_wts, dtype=object)

            self.zc_ref = np.empty(farm.n_wts, dtype=object)
            self.zc_mod = np.empty(farm.n_wts, dtype=object)

            for i_wt, wt in enumerate(self.farm.wts):
                wm_path = lambda i_mask: f'{wm_dir}{wm_str_id}{i_mask:02d}_w{wt.i_bf:02d}_{it0_str}'

                # n_mask masks are available for each wake tracked
                wms = [Centerline(wm_path(i), mask_type=gaussian_flag) for i in range(self.n_masks)]
                
                # initializing data
                n_t = len(farm)
                n_x = len(wms[0].x)

                self.t_ref  = wms[0].time
                self.t_mod  = np.ones(n_t) * np.nan
                
                self.x[i_wt] = wms[0].x

                self.zc_ref[i_wt] = [wm.z for wm in wms]
                self.zc_mod[i_wt] = np.zeros( (n_t, n_x) )

                if np.sqrt( (  np.sqrt((self.x[i_wt][0]-wt.x[0])**2 
                                     + (self.zc_ref[i_wt][0][-1,0]-wt.x[2])**2 )) > farm.af.D ):
                    raise Exception(  f'Wake tracking initial position and wind'
                                    + f'turbine location do not match.')
            
            self._it = 0
        
        # -------------------------------------------------------------------- #
    def update(self):
        if self._it is None: return 

        for i_wt, wt in enumerate(self.farm.wts):
            self.zc_mod[i_wt][self._it,:] = np.interp( self.x[i_wt], 
                                                self.farm.lag_solver.get('W', 'x_p', 0, i_wt=i_wt),
                                                self.farm.lag_solver.get('W', 'x_p', 1, i_wt=i_wt) ) 
        self.t_mod[self._it] = self.farm.t
        self._it += 1
        # -------------------------------------------------------------------- #

    def export(self):
        super().export()

        # data should only be exported once
        if self._it is not None: 
            out = { 'x':self.x,
                    't_mod':self.t_mod,
                    't_ref':self.t_ref,
                    'zc_mod':self.zc_mod,
                    'zc_ref':self.zc_ref }
            np.save(f'{self.farm.out_dir}/wcl_data.npy', out, allow_pickle=True)
        # -------------------------------------------------------------------- #
    
    def plot(self):
        return super().plot()
        # -------------------------------------------------------------------- #

class WakeCenterlineXloc(_WakeCenterline):
    def __init__(self, farm: Farm, bf_dir:str , wm_str_id:str, x_loc:List[float], 
                 xlim:List[float]=None, ylim:List[float]=None, 
                 u_norm:float=None, diag:bool=True):
        super().__init__(farm, bf_dir, wm_str_id)
        self.x_loc  = x_loc
        self.xlim   = xlim
        self.ylim   = ylim
        self.u_norm = u_norm or farm.af.D 
        self.diag   = diag
        # -------------------------------------------------------------------- #

    def update(self):
        super().update()
        # -------------------------------------------------------------------- #
        
    def plot(self):
        super().plot()

        for i_wt, (wt, x, zc_mod, zc_ref) \
             in enumerate(zip(self.farm.wts, self.x, self.zc_mod, self.zc_ref)):

            # Interpolating reference data
            zc_ref_interp = [None]*self.n_masks
            for i_mask, zc_mask in enumerate(zc_ref):
                ref_interp = interpolate.interp2d(x, self.t_ref, zc_mask, kind='linear')
                zc_ref_interp[i_mask] = ref_interp(x, self.t_ref)
            
            fig, axs = plt.subplots(len(self.x_loc), 1, 
                                        sharex=True, sharey=True, figsize=(6,8))
            for ax, x_ax in zip(axs, self.x_loc):

                # Checking if x_loc is valid for meandering
                x_wt_all = np.load(f'{self.farm.data_dir}/geo.npy')
                i_wt_row = np.where(np.abs(x_wt_all[:,2]-wt.x[2])<1E-3)
                x_wt_row = np.sort(x_wt_all[i_wt_row,0])
                row_idx  = np.argmin(np.abs(x_wt_row-wt.x[0]))+1                
                x_wt_next = x_wt_all[i_wt_row,0][0][row_idx] \
                                           if row_idx<len(x_wt_row[0]) else 10E9

                is_valid_wm = (x[0]<x_ax+wt.x[0]<x[-1]) and x_ax+wt.x[0]<x_wt_next

                plt.sca(ax)
                if is_valid_wm:# plotting valid data
                    idx_t_ref = ~np.isnan(self.t_ref)
                    idx_t_mod = ~np.isnan(self.t_mod)
                    idx_x     = np.argmin(np.abs(self.x[i_wt]-wt.x[0]-x_ax))

                    normx = lambda _x: (_x)/(self.farm.af.D/self.u_norm)
                    normy = lambda _y: (_y-wt.x[2])/self.farm.af.D

                    # plotting data
                    plt.plot(normx(self.t_ref[idx_t_ref]), 
                             normy(zc_ref_interp[I_MASK][idx_t_ref,idx_x]), 
                             **ls.REF)                    
                    plt.plot(normx(self.t_mod[idx_t_mod]), 
                             normy(zc_mod[idx_t_mod,idx_x]), 
                             **ls.MOD)

                    plt.ylabel(r'$\frac{1}{D}z_C(t,'+f'{x_ax/self.farm.af.D:.1f}'+r'D)$')
                    
                    plt.xlim(self.xlim or normx(self.t_mod[idx_t_mod][[0,-1]]))
                    plt.ylim(self.ylim or [-1,1])

                    # wake envelope
                    _zc_ref = [normy(zc[idx_t_ref,idx_x]) for zc in zc_ref]
                    zc_ref_up  = np.max(_zc_ref, axis=0)
                    zc_ref_low = np.min(_zc_ref, axis=0)

                    ax.fill_between( normx(self.t_ref),
                                     zc_ref_low,
                                     zc_ref_up, 
                                     color=[0.7,0.7,0.7] )


                    idx_t0_ref = np.argmin(np.abs(self.t_ref-self.t_ref[0]-150))
                    idx_t0_mod = np.argmin(np.abs(self.t_mod-self.t_mod[0]-150))

                    v0 = normy(zc_mod[idx_t_ref,idx_x])
                    v1 = np.interp( self.t_ref[idx_t_ref], 
                                    self.t_mod, 
                                    normy(zc_ref_interp[I_MASK][:,idx_x]))

                    norm = (np.mean(v1**2))**.5   

                    rho   = (np.mean((v0-np.mean(v0))*(v1-np.mean(v1))) \
                                                    /(np.std(v0)*np.std(v1)))
                    bias  = np.mean(v0-v1)/norm
                    err   = np.mean(np.abs(v0-v1))/norm
                    mape  = np.mean(np.abs((v0-v1)/v0))

                    buffer  = r'$\rho =' + f'{rho:.2f}'  + '$  '
                    buffer += r'$b ='    + f'{bias:.2f}' + '$  '
                    buffer += r'$e ='    + f'{err:.2f} ({((v0**2).mean())**.5:.2g} / {((v1**2).mean())**.5:.2g})'  + '$  '
                    buffer += r'$MAPE =' + f'{mape:.2f}' + '$'
                    plt.text( 0.975, 0.95, buffer,
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform = ax.transAxes )

                    ax_last = ax
                else:
                    plt.axis('off')

            ax_last.set_xlabel(r't [s]' if self.u_norm is None else r'$\frac{t}{T_C}$') 
            ax_last.xaxis.set_tick_params(labelbottom=True)

            plt.tight_layout()
            plt.subplots_adjust(left=0.12, right=0.99, hspace=0.1)

            self.savefig(f'wcl_xloc_wt{wt.i_bf:02d}.pdf')
        # -------------------------------------------------------------------- #

    def export(self):
        return super().export()