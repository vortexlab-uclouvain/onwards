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
import matplotlib.pyplot as plt

from .viz import Viz
from . import linespecs as ls
if TYPE_CHECKING:
    from ..farm import Farm
    from ..lagSolver.grid import Grid

PLANE_ID = 'plane00'
PLOTZERO = True
SAVE_DIR = 'mp4'

SKELETON_ARGS = {'colors': 'w',
                 'linestyles': 'solid',
                 'linewidths': 0.6}

class Field_bf():
    def __init__(self, data_path:str, farm: Farm, comp: int):
        from ensightReader import EnsightReader
        from iniReaderv2 import IniReader

        self.comp = comp

        bf_ini = IniReader(f'{data_path}/bigflowSimu.ini')
        sim_name = bf_ini.iniDict['simName']

        self.t0 = farm.wts[0].snrs.t0
        self.dt_bf = bf_ini.getValue('time_stepping', 'dt', float)

        case_dict = {
            'meshFile': f'/geo/mesh_{PLANE_ID}_Vel_{sim_name}.geo',
            'varType': ['vector'],
            'elmType': ['node'],
            'varName': ['Vel'],
            'varFile': [f'/velocity/{PLANE_ID}_Vel_{sim_name}_000000.out']
        }

        self.er_wf = EnsightReader(case_dict, rootDir=data_path, it0_override_flag=True)
        self.x = self.er_wf.getField('Vel', self.comp)[1]

        # -------------------------------------------------------------------- #

    def __call__(self, t):
        it = int(int(self.t0+t)/self.dt_bf)
        self.er_wf.dataTimeUpdate(it)
        if self.comp is None:
            return np.sqrt(sum(self.er_wf.getField('Vel',comp)[0].squeeze()**2 for comp in [0,2]))
        else:
            return self.er_wf.getField('Vel',self.comp)[0].squeeze()
        # -------------------------------------------------------------------- #

class Field_npz():
    def __init__(self, data_path:str, farm: Farm, comp: int):
        self.comp = comp
        data = np.load(data_path)

        self.x = [data[comp] for comp in ['x', 'y', 'z']]
        self.uu = data['u']
        self.t = data['t'] - farm.wts[0].snrs.t0                                                   
        # -------------------------------------------------------------------- #

    def __call__(self, t):
        if self.comp is None:
            return np.sqrt(sum(self.uu[np.argmin(np.abs(t-self.t)), self.comp, :, :]**2 for comp in [0,2]))
        else:
            return self.uu[np.argmin(np.abs(t-self.t)), self.comp, :, :]
        # -------------------------------------------------------------------- #

class Viz_velfield(Viz):
    farm: Farm
    grid: Grid
    viz_type = 'velfield'

    def __init__(self, farm: Farm, vel_bnds: list, comp: int=0, data_fid: str = False,
                 mp4_export: bool = True, skip: int = 1, t_start: float = None,
                 skeleton: bool = False, show: bool = True, slice_export: bool= False):
        """ Extracts and plots the 2D farm flow velocity velocity and compares it 
        against the reference data.

        Parameters
        ----------
        farm : Farm
            Parent :class:`.Farm` Object
        vel_bnds : list
            User defined velocity bounds for plotting [ms-1].
        comp : int
            Velocity field component to be plotted.
        data_fid : str, optional
            Path to the LES 2D fields data.
        mp4_export : bool, optional
            If True (by default), one png is exported at every time step.
        skip : int, optional
            Slices are plotted every ``skip`` timesteps, by default 1
        t_start : float, optional
            Data is only plotted after ``t_start`` [s], by default False
        skeleton : bool, optional
            If True, the model isocontours are overlaid to the BF data for easy 
            comparison, by default False.
        show : bool, optional
            If True (by default), the figure GUI is updated at every timestep. 
        slice_export: bool, optional
            If true export hub height slices as .npy files, by default False.
        """
        super().__init__(farm)

        self.vel_bnds = vel_bnds
        self.comp = comp

        self.grid = self.farm.lag_solver.grid

        if not self.grid:
            raise Exception('Grid should be enabled for VelField_plot.')

        # mp4 export
        self.show = show
        self._frame_id = 0

        if mp4_export:
            if not self.farm.out_dir:
                raise ValueError(
                    'mp4_export can not be set to True if no output directory is specified in farm.')

            self.mp4_dir = f'{self.__dirgen__()}/{SAVE_DIR}'
            if not os.path.exists(self.mp4_dir):
                os.makedirs(self.mp4_dir)
        else:
            self.mp4_dir = False

        # slice export
        if slice_export:
            if not self.farm.out_dir:
                raise ValueError(
                    'slice_export can not be set to True if no output directory is specified in farm.')

            self.slice_dir = f'{self.__dirgen__()}/slices/'
            if not os.path.exists(self.slice_dir):
                os.makedirs(self.slice_dir)
        else:
            self.slice_dir = False
             
        # wake skeleton overlay
        self.skeleton = skeleton
        if skeleton and not data_fid:
            raise ValueError(
                '\'skeleton\' can not be displayed if no data_fid is provided')

        self.cmap_vel = ls.CMAP if self.skeleton else ls.CMAP_VEL

        # time iterator
        self.skip = skip
        if t_start is  None: self.t_start = self.farm.t- 1
        else:                self.t_start = t_start

        # initializing plots
        self.n_ax = 2 if data_fid else 1
        
        af_d = self.farm.af.D
        self.xlim = list(self.grid.x_bnds)
        self.zlim = list(self.grid.z_bnds)

        scale_fac = 5

        lx = (self.grid.x_bnds[-1]-self.grid.x_bnds[0])/af_d/scale_fac
        lz = (self.grid.z_bnds[-1]-self.grid.z_bnds[0])/af_d/scale_fac*self.n_ax

        if lx<12 and lz<6:
            if lx <  lz: lx, lz = 6*lx/lz, 6 
            if lx >= lz: lx, lz = 12,      12*lz/lx 

        figsize = (lx, lz)
        
        self.fig, self.axs = plt.subplots(self.n_ax, 1, sharex=True, sharey=True, 
                                                 squeeze=False, figsize=figsize)

        self.im_args = { 'vmin':self.vel_bnds[0], 
                         'vmax':self.vel_bnds[1],
                         'cmap':self.cmap_vel, }
        
        # AX0: model plot
        i_ax = 0
        plt.sca(self.axs[i_ax][0])
        self.im_mod = plt.imshow( self.grid.xx,
                                  extent=[*[x/af_d for x in self.grid.x_bnds],
                                         *[z/af_d for z in self.grid.z_bnds]], 
                                 **self.im_args)
        
        self.plt_bnds_f = [None]*self.farm.n_wts
        self.plt_bnds_w = [None]*self.farm.n_wts
        for i_wt in range(self.farm.n_wts):
            comp_domain = self.farm.lag_solver.get_bounds('F',i_wt, i_sigma=1)
            self.plt_bnds_f[i_wt] = plt.plot( [comp_domain[0][0] for i in range(-1,4)],
                                              [comp_domain[0][1] for i in range(-1,4)], 'k' )
            
            comp_domain = self.farm.lag_solver.get_bounds('W',i_wt)
            self.plt_bnds_w[i_wt] = plt.plot( [comp_domain[i][0] for i in range(-1,4)],
                                              [comp_domain[i][1] for i in range(-1,4)], 'k--' )

        self.plt_wt_mod = self._set_layout(plt.gca(), self.im_mod, skip_x=bool(data_fid))

        # AX1: reference data plot
        self.data_fid = data_fid
        if data_fid:
            i_ax += 1
            plt.sca(self.axs[i_ax][0])
            uu_ref = self.__init_ref__(data_fid)

        if self.comp is None:
            uu_mod =  np.sqrt(sum(self.grid.u_compute()**2))
        else:
            uu_mod = self.grid.u_compute()[self.comp]

        if self.skeleton:
            self.skeleton_args = { 'levels':np.linspace(*self.vel_bnds, 7),
                                    'vmin':self.vel_bnds[0],
                                    'vmax':self.vel_bnds[1] } | SKELETON_ARGS

            self.im_ref_skeleton = plt.contour(self.grid._x/self.farm.af.D,
                                               self.grid._z/self.farm.af.D,
                                               np.rot90(uu_mod),
                                               **self.skeleton_args)
        self.time_txt = plt.suptitle('')

        self.fig.subplots_adjust(right=0.85)
        cbar = self.fig.add_axes([0.85, 0.25, 0.025, 0.5])
        self.fig.colorbar(self.im_mod, cax=cbar, label=r'u(z=90m) [ms$^{-1}$]')

        if self.n_ax==1:
            plt.subplots_adjust(left=0.1, right=0.8, bottom=0.25, top=0.8)
        # Updating data
        self.update()

        if self.slice_dir:
            np.save(f'{self.slice_dir}/x_mod.npy', 
                 np.array([self.grid._x, self.grid._z], dtype=object))
            self.slice_acc_mod = np.zeros_like(uu_mod)
            if self.data_fid:
                np.save(f'{self.slice_dir}/x_ref.npy', 
                    np.array([self.fld_getter.x[0], self.fld_getter.x[2]], dtype=object))
                self.slice_acc_ref = np.zeros_like(uu_ref)

            if 'yawA' in self.farm.wts[0].snrs:
                np.save(f'{self.slice_dir}/psi.npy',[self.farm.wts[0].snrs['time'], [np.deg2rad(wt.snrs['yawA']) for wt in self.farm.wts]]) 

        # -------------------------------------------------------------------- #
    
    def __init_ref__(self, data_fid: str):
        if data_fid.endswith('.npz'):  
            self.fld_getter = Field_npz(f'{self.farm.data_dir}/{data_fid}', self.farm, self.comp)
        else:
            self.fld_getter = Field_bf(data_fid, self.farm, self.comp)
        
        x_ref = self.fld_getter.x

        # Setting bounds
        x_idx_bnds = (np.argmin(np.abs(self.grid.x_bnds[0] - x_ref[0])),
                      np.argmin(np.abs(self.grid.x_bnds[1] - x_ref[0])))
        self.x_ref_idx = np.arange(*x_idx_bnds)

        y_idx_bnds = (np.argmin(np.abs(self.grid.z_bnds[0] - x_ref[2])),
                      np.argmin(np.abs(self.grid.z_bnds[1] - x_ref[2])))

        if self.grid.x_bnds[0] < x_ref[0][0]:
            self.xlim[0] = x_ref[0][0]
        if self.grid.x_bnds[1] > x_ref[0][-1]:
            self.xlim[1] = x_ref[0][-1]
        if self.grid.z_bnds[0] < x_ref[2][0]:
            self.zlim[0] = x_ref[2][0]
        if self.grid.z_bnds[1] > x_ref[2][-1]:
            self.zlim[1] = x_ref[2][-1]

        # Plotting data
        af_d = self.farm.af.D

        y_vec_idx = np.arange(*y_idx_bnds)
        self.ref_map = tuple(np.meshgrid(self.x_ref_idx, y_vec_idx, indexing='ij'))

        uu_ref = self.fld_getter(self.farm.t)
        self.im_ref = plt.imshow( np.rot90(uu_ref[self.ref_map]), 
                                extent=[*[x/af_d for x in self.xlim],
                                        *[z/af_d for z in self.zlim]],
                                **self.im_args )

        self.plt_wt_ref = self._set_layout(plt.gca(), self.im_ref, skip_x=False)

        return uu_ref
        # -------------------------------------------------------------------- #

    def _set_layout(self, ax: plt.Axes, im, skip_x: bool = False):
        
        af_d = self.farm.af.D

        if not skip_x:
            ax.set_xlabel(r'$\frac{x}{D}$')
        ax.set_ylabel(r'$\frac{z}{D}$', rotation=0)
        plt_wt = [plt.plot(*[x/af_d for x in wt.get_bounds()],
                           **ls.WT) for wt in self.farm.wts]
        if PLOTZERO:
            z_wts = np.unique([wt.x[2] for wt in self.farm.wts])
            for z in z_wts:
                plt.plot(plt.xlim(), [z/af_d]*2, 'k--', linewidth=0.8)

        plt.xlim([x/af_d for x in self.xlim])
        plt.ylim([z/af_d for z in self.zlim])

        return plt_wt
        # -------------------------------------------------------------------- #

    def _layout_update(self, plt_wt):
        for p, wt in zip(plt_wt, self.farm.wts):
            p[0].set_color('k' if wt.is_freestream() else [0.3]*3)
            p[0].set_xdata([x/self.farm.af.D for x in wt.get_bounds()[0]])
            p[0].set_ydata([x/self.farm.af.D for x in wt.get_bounds()[1]])
        # -------------------------------------------------------------------- #

    def update(self):
        if not self.farm.update_LagSolver_flag:
            return

        plt.figure(self.fig.number)
        self.time_txt.set_text(r'$t={'+f'{self.farm.t:2.1f}'+r'}\; [s]$')

        if (self.farm.it % self.skip) > 0 or self.farm.t< self.t_start:
            return

        if self.comp is None:
            uu_mod = np.sqrt(sum(self.grid.u_compute()**2))
        else:
            uu_mod = self.grid.u_compute()[self.comp]

        self.im_mod.set_data(np.rot90(uu_mod))
        self._layout_update(self.plt_wt_mod)

        for i_wt, (p_f, p_w) in enumerate(zip(self.plt_bnds_f, self.plt_bnds_w)):
            comp_domain = self.farm.lag_solver.get_bounds('F',i_wt, i_sigma=0)
            p_f[0].set_xdata([comp_domain[1][0]/self.farm.af.D for i in range(-1,4)])
            p_f[0].set_ydata([comp_domain[1][1]/self.farm.af.D for i in range(-1,4)])

            comp_domain = self.farm.lag_solver.get_bounds('W',i_wt)
            p_w[0].set_xdata([comp_domain[i][0]/self.farm.af.D    for i in range(-1,4)])
            p_w[0].set_ydata([comp_domain[i][1]/self.farm.af.D    for i in range(-1,4)])

        if self.data_fid:
            uu_ref = self.fld_getter(self.farm.t)
            self.im_ref.set_data(np.rot90(uu_ref[self.ref_map]))
            self._layout_update(self.plt_wt_ref)

        if self.skeleton:
            plt.sca(self.im_ref.axes)
            for coll in self.im_ref_skeleton.collections:
                coll.remove()
            self.im_ref_skeleton = plt.contour(self.grid._x/self.farm.af.D,
                                              self.grid._z/self.farm.af.D,
                                              uu_mod.T,
                                              **self.skeleton_args)
        if self.show:
            plt.draw()
            plt.pause(0.1)

        if self.mp4_dir:
            self.__savefig__(f'/{SAVE_DIR}/frame{self._frame_id:03d}.png', dpi=200)
        
        if self.slice_dir:
            np.save(f'{self.slice_dir}/u_mod_{self._frame_id}.npy', 
                                  np.array([self.farm.t, uu_mod], dtype=object))
            self.slice_acc_mod += uu_mod 
            if self.data_fid: 
                np.save(f'{self.slice_dir}/u_ref_{self._frame_id}.npy', 
                                     np.array([self.farm.t, uu_ref], dtype=object))
                self.slice_acc_ref += uu_ref

        self._frame_id += 1
        # -------------------------------------------------------------------- #

    def _data_clean(self, *args, **kwargs):
       
        if self.slice_dir:
            np.save(f'{self.slice_dir}/u_mod_avg.npy', 
                                  np.array(self.slice_acc_mod/self._frame_id) ) 
            if self.data_fid: 
                np.save(f'{self.slice_dir}/u_ref_avg.npy', 
                                     np.array(self.slice_acc_ref/self._frame_id) )
            


        if not self.mp4_dir: 
            return

        if os.system('which ffmpeg > /dev/null 2>&1')==0:
            frame_rate = 10 # [Hz]
            out_name = 'velfield_anim'
            
            lg.info(f'Generating {out_name} (logging to ffmpeg.log)')
            os.system( f'ffmpeg -r {frame_rate} -i {self.mp4_dir}/frame%03d.png'
                     + f' -vcodec mpeg4 -y {self.__dirgen__()}/{out_name}.mp4'
                     + f'> {self.__dirgen__()}/ffmpeg.log 2>&1 ')
            
            # os.system( f'convert {self.mp4_dir}/*.png {self.__dirgen__()}/{out_name}.gif' )

            for fid in os.listdir(self.mp4_dir): os.remove(f'{self.mp4_dir}/{fid}')
            os.rmdir(self.mp4_dir)
        else:
            lg.warning('ffmpeg not install, please generate animation manually')
        # -------------------------------------------------------------------- #

    def _export(self):
        pass
        # -------------------------------------------------------------------- #

    def _plot(self):
        pass
        # -------------------------------------------------------------------- #
