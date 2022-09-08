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

class WakeCenterline(Viz):
    def __init__(self, farm: Farm, bf_dir: str, wm_str_id: str, i_mask: int=None):
        """ Extracts the position of the wake centerline from the Lagrangian flow 
        model and from the LES reference data.

        Parameters
        ----------
        farm : Farm
            Parent Farm Object
        bf_dir : str
            Path to the reference LES data.
        wm_str_id : str
            Template for the wake centerline filename generation (eg: WMcenterline_gaussianMask)
        i_mask : int, optional
            Index of the mask used for the wake centerline tracking for the LES 
            reference data, if None, all available masks are imported, by default 
            None.

        Raises
        ------
        Exception
            If any of the wake centerline origin does not match the associated 
            turbine location.

        Note
        ----
        This class does not implement any plot method. It only exports the wake
        centerline data so that it can be postprocessed afterwards.

        See also
        --------
        :class:`disp.WakeCenterlineXloc_plot<.disp.centerline_plot.WakeCenterlineXloc_plot>`
        """        
        super().__init__(farm)

        # check for previous WakeCenterline initialization
        
        data = next((v for v in self.farm.viz if isinstance(v, WakeCenterline)), None)

        if data: # data was already imported
            self.data['x']       = data.x

            self.data['t_ref']   = data.t_ref
            self.data['t_mod']   = data.t_mod

            self.data['zc_ref']  = data.zc_ref
            self.data['zc_mod']  = data.zc_mod

            self.n_masks = data.n_masks

            self._was_clean    = True
            self._was_exported = True

            self._it = None

        else :   # data needs to be imported
            wm_dir = f'{bf_dir}/WM_centerline/'

            wm_fid = [fid for fid in os.listdir(wm_dir) if fid.startswith(wm_str_id)]
            it0_str = wm_fid[0].rsplit('_',1)[1]

            gaussian_flag = 'gaussian' if 'gaussian' in wm_str_id else False
            if i_mask:
                self.n_masks = 1
                self.i_masks = [i_mask]
            else:
                self.n_masks  = len([fid for fid in wm_fid if fid.rsplit('_',2)[-2]=='w00'])
                self.i_masks = range(self.n_masks)

            self.data['x']      = np.empty(farm.n_wts, dtype=object)

            self.data['zc_ref'] = np.empty(farm.n_wts, dtype=object)
            self.data['zc_mod'] = np.empty(farm.n_wts, dtype=object)

            for i_wt, wt in enumerate(self.farm.wts):
                wm_path = lambda i_mask: f'{wm_dir}{wm_str_id}{i_mask:02d}_w{wt.i_bf:02d}_{it0_str}'

                # n_mask masks are available for each wake tracked
                wms = [Centerline(wm_path(i), mask_type=gaussian_flag) for i in self.i_masks]
                
                # initializing data
                n_t = len(farm)
                n_x = len(wms[0].x)

                self.data['t_ref'] = wms[0].time
                self.data['t_mod'] = np.ones(n_t) * np.nan
                
                self.data['x'][i_wt] = wms[0].x

                self.data['zc_ref'][i_wt] = [wm.z for wm in wms]
                self.data['zc_mod'][i_wt] = [np.zeros( (n_t, n_x) )]

                if np.sqrt( (  np.sqrt((self.data['x'][i_wt][0]-wt.x[0])**2 
                                     + (self.data['zc_ref'][i_wt][0][-1,0]-wt.x[2])**2 )) > farm.af.D ):
                    raise Exception(  f'Wake tracking initial position and wind'
                                    + f'turbine location do not match.')
            
            self._it = 0
        
        # -------------------------------------------------------------------- #

    def reset(self):
        self._it = 0
        for i_wt, wt in enumerate(self.farm.wts):
            self.data['zc_mod'][i_wt][0][:] = np.nan
        self.data['t_mod'][:] = np.nan
        # -------------------------------------------------------------------- #

    def update(self):
        if self._it is None: return 

        for i_wt, wt in enumerate(self.farm.wts):
            self.data['zc_mod'][i_wt][0][self._it,:] = np.interp( self.data['x'][i_wt], 
                                                self.farm.lag_solver.get('W', 'x_p', 0, i_wt=i_wt),
                                                self.farm.lag_solver.get('W', 'x_p', 1, i_wt=i_wt) ) 
        self.data['t_mod'][self._it] = self.farm.t
        self._it += 1
        # -------------------------------------------------------------------- #
    
    def _data_clean(self):
        self.data['t_mod']  = self.data['t_mod'][:self._it]

        interp = interpolate.interp2d
        self._interp = { 'ref': [None] * self.farm.n_wts, 
                         'mod': [None] * self.farm.n_wts }
    
        for i_wt in range(self.farm.n_wts):
            self.data['zc_mod'][i_wt][0] = self.data['zc_mod'][i_wt][0][:self._it,:] 

            self._interp['mod'][i_wt] = [ interp( self.data['t_mod'],
                                                  self.data['x'][i_wt],  
                                                  self.data['zc_mod'][i_wt][0].T, 
                                                  kind='linear' ) ]

            self._interp['ref'][i_wt] = [ interp( self.data['t_ref'],
                                                  self.data['x'][i_wt],  
                                                  self.data['zc_ref'][i_wt][i_mask].T, 
                                                  kind='linear' )
                                          for i_mask in range(self.n_masks) ]
        # -------------------------------------------------------------------- #
    
    def _export(self):
        # data should only be exported once
        if self._it is not None: 
            np.save(f'{self.farm.out_dir}/wcl_data.npy', self.data, allow_pickle=True)
        # -------------------------------------------------------------------- #
    
    def _data_get(self, field: str, source:str=None, i_mask:int=0, i_wt:int=0,
                          t_interp:List[float]=None, x_interp:List[float]=None):

        if field == 'x':  
            return self.data['x'][i_wt]

        if field == 't':  
            return self.data[f't_{source}']

        if field == 'zc':
            if (t_interp is None) and (x_interp is None):
                return self.data[f'zc_{source}'][i_wt][i_mask]
            else:                     
                x_interp = self.data['x'][i_wt]     \
                                               if x_interp is None else x_interp
                t_interp = self.data[f't_{source}'] \
                                               if t_interp is None else t_interp
                return self._interp[source][i_wt][i_mask](t_interp, x_interp).T
        # -------------------------------------------------------------------- #

class WakeCenterlineXloc_plot(WakeCenterline):
    def __init__(self, farm: Farm, bf_dir:str , wm_str_id:str, x_loc:List[float], 
                 i_mask:int=None, xlim:List[float]=None, ylim:List[float]=None, 
                 u_norm:float=None, diag:bool=True):
        """ Extracts the position of the wake centerline at the x_loc location and 
        compares it to the LES reference data.

        Parameters
        ----------
        farm : Farm
            Parent Farm Object
        bf_dir : str
            Path to the reference LES data.
        wm_str_id : str
            Template for the wake centerline filename generation 
            (eg: WMcenterline_gaussianMask)
        x_loc : List[float]
            List of downstream streamwise (x) positions in [m] where the wake 
            centerline is evaluated.
        i_mask : int, optional
            Index of the mask used for the wake centerline tracking for the LES 
            reference data, if None, all available masks are imported, by default 
            None.
        xlim : List[float], optional
            User defined time bounds for plotting, by default None.
        ylim : List[float], optional
            User defined streamwise position bounds for plotting, by default None.
        u_norm : float, optional
            Velocity used for data normalization T_C = D/u_norm, if no u_norm is 
            provided, no normalization is applied, by default None.
        diag : bool, optional
            If True, the correlation, error and MAPE are evaluated, by default True.

        See also
        --------
        :class:`disp.WakeCenterline<.disp.centerline_plot.WakeCenterline>`
        """      
        super().__init__(farm, bf_dir, wm_str_id, i_mask)
        self.x_loc  = x_loc
        self.x_lim   = xlim
        self.ylim   = ylim
        self.u_norm = u_norm or farm.af.D 
        self.diag   = diag
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
            
            fig, axs = plt.subplots(len(self.x_loc), 1, sharex=True,
                                      sharey=True, figsize=(6,8), squeeze=False)

            for ax, x_ax in zip(axs, self.x_loc):
                ax = ax[0]

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
                    idx_x     = np.argmin(np.abs(x-wt.x[0]-x_ax))

                    normx = lambda _x: (_x)/(self.farm.af.D/self.u_norm)
                    normy = lambda _y: (_y-wt.x[2])/self.farm.af.D

                    # plotting data
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

                    ax_last = ax
                else:
                    plt.axis('off')

            ax_last.set_xlabel(r't [s]' if self.u_norm is None else r'$\frac{t}{T_C}$') 
            ax_last.xaxis.set_tick_params(labelbottom=True)

            plt.tight_layout()
            plt.subplots_adjust(left=0.12, right=0.99, hspace=0.1)

            self.savefig(f'wcl_xloc_wt{wt.i_bf:02d}.pdf')
        # -------------------------------------------------------------------- #