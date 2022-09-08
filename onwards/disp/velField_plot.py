from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import os
import numpy as np
import matplotlib.pyplot as plt

from .viz import Viz
from . import linespecs as ls
if TYPE_CHECKING:
    from ..farm           import Farm
    from ..lagSolver.grid import Grid

PLANE_ID = 'plane00'
PLOTZERO = True
SAVE_DIR = 'velField_slices'

class VelField_plot(Viz):
    farm: Farm
    grid: Grid

    def __init__(self, farm: Farm, vel_bnds: list, comp: int, bf_dir:str=False, 
                 mp4_export:bool=True, skip:int=1, t_start:float=False, 
                 skeleton: bool=False, slice_export: bool=True,
                 enable_anim: bool=True, du_export: bool=False,):
        
        
        super().__init__(farm)
        self.vel_bnds     = vel_bnds
        self.out_dir      = self.farm.out_dir
        self.grid         = self.farm.lag_solver.grid
        self.enable_anim = enable_anim

        if not self.grid: raise Exception('Grid should be enabled for VelField_plot.')

        # -- MP4 export -- #
        self.mp4_export = mp4_export and self.enable_anim
        if self.mp4_export:
            if not self.farm.out_dir:
                raise ValueError('mp4_export can not be set to True if no output directory is specified in farm.')
            self.frame_id = 0
            if not os.path.exists(f'{self.out_dir}/{SAVE_DIR}/'):
                os.makedirs(f'{self.out_dir}/{SAVE_DIR}/')

        # -- Slice export -- #
        self.slice_export = slice_export
        if self.slice_export:
            if not self.farm.out_dir:
                raise ValueError('slice_export can not be set to True if no output directory is specified in farm.')
            self.slice_id = 0
            if not os.path.exists(f'{self.out_dir}/slices/'): 
                os.makedirs(f'{self.out_dir}/slices/')

        self.skeleton = skeleton
        if skeleton and not bf_dir:
            raise ValueError('\'skeleton\' can not be displayed if no bf_dir is provided')
        self.cmap_vel =  plt.get_cmap('viridis') if self.skeleton else ls.CMAP_VEL 

        if comp not in [0,1]: raise ValueError('comp should be 0 or 1')
        self.comp = comp
        
        self.skip = skip
        self.time = self.farm.it
        if   t_start == False: self.t_start = self.time - 1
        else :                 self.t_start = t_start

        self.bf_dir = bf_dir 
        if self.bf_dir: self.n_ax = 2
        else:           self.n_ax = 1

        # -- Initializing plots -- #
        
        if self.enable_anim:
            self.fig, self.axs = plt.subplots(self.n_ax, 1, sharex=True, sharey=True, squeeze=False, figsize=(12,6))

            i_ax = 0; plt.sca(self.axs[i_ax][0])
            self.im_f3 = plt.imshow( self.grid.xx,
                                    vmin=self.vel_bnds[0], vmax=self.vel_bnds[1], 
                                    cmap=self.cmap_vel,
                                    extent=[ *[x/self.farm.af.D for x in self.grid.x_bnds],
                                            *[z/self.farm.af.D for z in self.grid.z_bnds]] )
            self.plt_wt_f3 = self._set_layout(plt.gca(), self.im_f3, skip_x=bool(bf_dir))

        if bf_dir: 
            if self.enable_anim:
                i_ax += 1; plt.sca(self.axs[i_ax][0])      
            self._bf_ini(bf_dir)
        
        self.du_export=du_export
        if self.du_export:
            self._du_ini()

        uu_f3 = self.grid.u_compute()[self.comp]
        if self.skeleton:
            if self.enable_anim:
                self._skeleton_args = {'colors':'w', 'linestyles':'solid', 'linewidths':0.6}
                # self._skeleton_args = {'cmap':self.cmap_vel, 'linestyles':'solid', 'linewidths':0.6}
                self.im_bf_skeleton = plt.contour( self.grid._x/self.farm.af.D, 
                                                self.grid._z/self.farm.af.D, 
                                                uu_f3.T, 
                                                np.linspace(*self.vel_bnds,7),
                                                vmin=self.vel_bnds[0], vmax=self.vel_bnds[1], 
                                                **self._skeleton_args    )

        self.time_txt = plt.suptitle('')
        plt.tight_layout()

        if self.slice_export:
            self.__save_slice__('x_f3.npy', 
                 np.array([self.grid._x, self.grid._z], dtype=object))
            self.slice_acc_f3 = np.zeros_like(uu_f3)

        self.update()
        
        # -------------------------------------------------------------------- #

    def _bf_ini(self, bf_dir):
        from ensightReader import EnsightReader
        from iniReaderv2   import IniReader
        
        bf_ini     = IniReader(self.bf_dir+'bigflowSimu.ini')
        self.wf_sim_name   = bf_ini.iniDict['simName']

        t0    = self.farm.wts[0].snrs.t0
        dt_bf = bf_ini.getValue('time_stepping','dt',float)

        self.time2it  = lambda t: int((t0+t)/dt_bf)
        self.time2fid = lambda sim_name, t: f'{PLANE_ID}_Vel_{sim_name}_{self.time2it(t):05d}.out'

        case_dict = {
            'meshFile': f'/geo/mesh_{PLANE_ID}_Vel_{self.wf_sim_name}.geo',
            'varType' : ['vector'],
            'elmType' : ['node'],
            'varName' : ['Vel'],
            'varFile' : [f'/velocity/{self.time2fid(self.wf_sim_name, self.time)}']
        }

        self.er_wf = EnsightReader(case_dict, rootDir=self.bf_dir)
        
        u_vec_fld, x_vec = self.er_wf.getField('Vel',self.comp)

        x_idx_bnds = ( np.argmin(np.abs( self.grid.x_bnds[0] - x_vec[0] )) ,
                       np.argmin(np.abs( self.grid.x_bnds[1] - x_vec[0] )) )
        x_vec_idx = np.arange( *x_idx_bnds )

        y_idx_bnds = ( np.argmin(np.abs( self.grid.z_bnds[0] - x_vec[2] )) ,
                       np.argmin(np.abs( self.grid.z_bnds[1] - x_vec[2] )) )

        xlim = [self.grid.x_bnds[0], self.grid.x_bnds[-1]]
        zlim = [self.grid.z_bnds[0], self.grid.z_bnds[-1]]

        if self.grid.x_bnds[0] < x_vec[0][0] : xlim[0] = x_vec[0][0]             
        if self.grid.x_bnds[1] > x_vec[0][-1]: xlim[1] = x_vec[0][-1]
        if self.grid.z_bnds[0] < x_vec[2][0] : zlim[0] = x_vec[2][0] 
        if self.grid.z_bnds[1] > x_vec[2][-1]: zlim[1] = x_vec[2][-1]

        y_vec_idx = np.arange( *y_idx_bnds )

        self.bf_map = tuple(np.meshgrid( x_vec_idx, y_vec_idx, indexing='ij'))

        if self.enable_anim:
            self.im_bf = plt.imshow( np.rot90(u_vec_fld.squeeze()[self.bf_map]),
                                    vmin=self.vel_bnds[0], vmax=self.vel_bnds[1], 
                                    cmap=self.cmap_vel,
                                    extent=[ *[x/self.farm.af.D for x in xlim],
                                            *[z/self.farm.af.D for z in zlim]] )
            plt.xlim([x/self.farm.af.D for x in xlim])
            plt.ylim([z/self.farm.af.D for z in zlim])

            self.plt_wt_bf = self._set_layout(plt.gca(), self.im_bf, skip_x=False)

        if self.slice_export:
            self.__save_slice__('x_bf.npy', np.array(x_vec, dtype=object))
            self.slice_acc_bf = np.zeros_like(u_vec_fld.squeeze())
        # -------------------------------------------------------------------- #

    def _du_ini(self):
        if self.bf_dir:
            from pathlib       import Path
            from ensightReader import EnsightReader
            from iniReaderv2   import IniReader

            abl_dir = str(Path(self.bf_dir).parent.absolute())+'/ABL/'
            
            abl_ini     = IniReader(abl_dir+'/bigflowSimu.ini')
            self.abl_sim_name   = abl_ini.iniDict['simName']

            case_dict = {
                'meshFile': f'/geo/mesh_{PLANE_ID}_Vel_{self.abl_sim_name}.geo',
                'varType' : ['vector'],
                'elmType' : ['node'],
                'varName' : ['Vel'],
                'varFile' : [f'/velocity/{self.time2fid(self.abl_sim_name, self.time)}']
            }

            self.er_abl = EnsightReader(case_dict, rootDir=abl_dir)

        # -------------------------------------------------------------------- #

    def _set_layout(self, ax:plt.Axes, im, skip_x:bool=False):
        plt.colorbar(im,label=r'$u\;[\mathrm{ms}^{-1}]$')
        if not skip_x: ax.set_xlabel(r'$\frac{x}{D}$')
        ax.set_ylabel(r'$\frac{z}{D}$',rotation=0)
        plt_wt = [ plt.plot( *[x/self.farm.af.D for x in wt.get_bounds()],
                                              **ls.WT) for wt in self.farm.wts ]
        if PLOTZERO: 
            z_wts = np.unique([wt.x[2] for wt in self.farm.wts])
            for z in z_wts:
                plt.plot(plt.xlim(), [z/self.farm.af.D]*2, 'k--', linewidth=0.8)
        
        return plt_wt
        # -------------------------------------------------------------------- #

    def _layout_update(self, plt_wt):
        for p, wt in zip(plt_wt, self.farm.wts):
            p[0].set_color('k' if wt.is_freestream() else [0.3]*3)
            p[0].set_xdata([x/self.farm.af.D for x in wt.get_bounds()[0]])
            p[0].set_ydata([x/self.farm.af.D for x in wt.get_bounds()[1]])
        # -------------------------------------------------------------------- #

    def update(self):
        if not self.farm.update_LagSolver_flag: return

        self.time = self.farm.t

        if self.enable_anim:
            plt.figure(self.fig.number)
        self.time_txt.set_text(r'$t={'+f'{self.time:2.1f}'+r'}\; [s]$')

        if (self.farm.it%self.skip)>0 or self.time<self.t_start: 
            return 

        uu_f3 = self.grid.u_compute()[self.comp]

        if self.enable_anim:
            self.im_f3.set_data(np.rot90(uu_f3))
            self._layout_update(self.plt_wt_f3)

        if self.bf_dir:
            self.er_wf.dataTimeUpdate(self.time2it(self.farm.lag_solver.get_time()))
            uu_bf = self.er_wf.getField('Vel',self.comp)[0].squeeze()
            if self.enable_anim:
                self.im_bf.set_data(np.rot90(uu_bf[self.bf_map]))
                self._layout_update(self.plt_wt_bf)

        if self.skeleton and self.enable_anim:
            for coll in self.im_bf_skeleton.collections:
                coll.remove()
            self.im_bf_skeleton = plt.contour( self.grid._x/self.farm.af.D, 
                                               self.grid._z/self.farm.af.D, 
                                               uu_f3.T, 
                                               np.linspace(*self.vel_bnds,7),
                                               vmin=self.vel_bnds[0], vmax=self.vel_bnds[1], 
                                               **self._skeleton_args    )

        if self.mp4_export and self.enable_anim:
            self.savefig(f'/{SAVE_DIR}/frame{self.frame_id:03d}.png', dpi=200)
            self.frame_id += 1

        if self.slice_export:
            self.__save_slice__('u_f3_{self.slice_id}.npy', 
                                  np.array([self.time, uu_f3], dtype=object))
            self.slice_acc_f3 += uu_f3 
            if self.bf_dir: 
                self.__save_slice__('u_bf_{self.slice_id}.npy', 
                                     np.array([self.time, uu_bf], dtype=object))
                self.slice_acc_bf += uu_bf

        if self.du_export:
            duu_f3 = self.grid.du_wm_compute()[self.comp]
            self.__save_slice__('du_f3_{self.slice_id}.npy', 
                                np.array([self.time, duu_f3], dtype=object))
            if self.bf_dir: 
                duu_bf = self.er_abl.getField('Vel',self.comp)[0].squeeze()[:-1,:] - uu_bf
                self.__save_slice__('du_bf_{self.slice_id}.npy', 
                                    np.array([self.time, duu_bf], dtype=object))

        if self.du_export or self.slice_export:
            self.slice_id += 1

        plt.draw(); plt.pause(0.1); 
        # -------------------------------------------------------------------- #
        
    def plot(self):
        if self.slice_export:
            self.__save_slice__('u_f3_avg.npy', self.slice_acc_f3/self.slice_id)
            if self.bf_dir: 
                self.__save_slice__('u_bf_avg.npy', self.slice_acc_bf/self.slice_id)
        # -------------------------------------------------------------------- #

    def __save_slice__(self, fid, array, *args, **kwargs):
       np.save(f'{self.out_dir}/{SAVE_DIR}/{fid}', array, *args, **kwargs)