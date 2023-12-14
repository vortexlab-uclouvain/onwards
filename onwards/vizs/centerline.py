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

class Centerline:
    def __init__(self, file, mask_type:str = '', time_zero_origin=True):
        """ Centerline object
        
        Parameters
        ----------
            file : str
                Filepath to the .bin centerline file.
            mask_type : (str, optional)
                Mask type (eg: ``gaussian``), by default, None.
            time_zero_origin: (bool, optional)
                If true, time vector is shifted so that ``self.time[0]=0`` by 
                default, False.
        Note
        ----
        This class is only loaded if no local implementation is found.
        """

        self.path = file

        if 'gaussian' in file and 'gaussian' not in mask_type:
            print('WARNING: mask_type turned of while file identifier contains \'gaussian\'')

        with open(file, 'r') as fid:
            # Number of points in the 3 directions
            nTS = np.fromfile(fid, np.int32, 1)[0]
            nx   = np.fromfile(fid, np.int32, 1)[0]

            # coordinates
            x = np.fromfile(fid, np.float64, nx)

            nGlob = 1
            nLoc = 3 if mask_type=='gaussian' else 2

            nSteps = nGlob+nLoc*nx
            nTot   = nTS*nSteps

            data = np.fromfile(fid, np.float64, nTot)
            if data.shape < nTot:
                nTot = data.shape
                nTS  = int(nTot/nSteps)

            rank = 0
            time = data[range(rank,rank+(nTS-1)*nSteps+1,nSteps)]
            time = time-time[0]*time_zero_origin
            rank += 1

            sigma, y, z = (np.zeros((nTS,nx)) for i in range(3))

            if mask_type=='gaussian':
                for i in range(nx):
                    sigma[0:nTS,i] = data[range(rank,rank+(nTS-1)*nSteps+1,nSteps)]
                    rank += 1

            for i in range(nx):
                y[0:nTS,i] = data[range(rank,rank+(nTS-1)*nSteps+1,nSteps)]
                rank += 1

            for i in range(nx):
                z[0:nTS,i] = data[range(rank,rank+(nTS-1)*nSteps+1,nSteps)]
                rank += 1

            self.time, self.x, self.y, self.z = time, x, y, z

            self.nx = nx
            self.nT = nTS
            
            self.x0_vec = [self.x[0], np.mean(self.y[:,0]), np.mean(self.z[:,0]) ]

try:
    from readCenterlineForWM import Centerline
except ImportError: 
    lg.info('ReadCenterlineForWM not available: using local implementation.')

I_MASK = 0

class Viz_centerline(Viz):
    viz_type = 'centerline'

    def __init__(self, farm: Farm, bf_dir: str = None, i_mask: int = None):
        """ Extracts the position of the wake centerline from the Lagrangian flow 
        model and from the LES reference data.

        Parameters
        ----------
        farm : Farm
            Parent :class:`.Farm` Object
        bf_dir : str
            Path to the reference LES data.
        i_mask : int, optional
            Index of the mask used for the wake centerline tracking for the LES 
            reference data, if None (by default), all available masks are imported.

        Raises
        ------
        Exception
            If any of the wake centerline origin does not match the associated 
            Turbine location.
        ValueError
            If bf_dir is not compatible with previous Viz_centerline initialization.

        Note
        ----
        This class does not implement any plot method. It only exports the wake
        centerline data so that it can be postprocessed afterwards.

        See also
        --------
        :class:`.Viz_centerline_xloc`
        """        
        super().__init__(farm)

        # check for previous Viz_centerline initialization
        
        data = next((v for v in self.farm.viz if isinstance(v, Viz_centerline)), None)

        if data: # data was already imported
            if data.bf_dir != bf_dir:
                raise ValueError('All Viz_centerline must share the same bf_dir.')
            else:
                self.bf_dir = data.bf_dir

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
            self.bf_dir = bf_dir

            if self.bf_dir is not None:
                wm_dir = f'{self.bf_dir}/WM_centerline/'

                wm_fid = [fid for fid in os.listdir(wm_dir) if 'w00' in fid]
                it0    = wm_fid[0].rsplit('_', 1)[1]

                fid_masks  = [fid.rsplit('_', 2)[0] for fid in wm_fid]
                fid_paths  = [f'{wm_dir}/{fm}' for fm in fid_masks]
                fid_gmasks = ['gaussian' if 'gaussian' in fm else False
                              for fm in fid_masks]

                if i_mask:
                    self.n_masks = 1
                    self.i_masks = [i_mask]
                else:
                    self.n_masks = len(fid_masks)
                    self.i_masks = range(self.n_masks)

                self.data['x']      = np.empty(farm.n_wts, dtype=object)

                self.data['zc_ref'] = np.empty(farm.n_wts, dtype=object)
                self.data['zc_mod'] = np.empty(farm.n_wts, dtype=object)

                for i_wt, wt in enumerate(self.farm.wts):

                    # n_mask masks are available for each wake tracked
                    wms = [ Centerline( f'{fid_paths[i]}_w{wt.i_bf:02d}_{it0}', 
                                        mask_type= fid_gmasks[i], time_zero_origin= (wt.snrs.t0!=0) ) 
                           for i in self.i_masks ]
                    
                    # initializing data
                    n_t = len(farm)
                    n_x = len(wms[0].x)

                    self.data['t_ref'] = wms[0].time
                    self.data['t_mod'] = np.ones(n_t) * np.nan
                    
                    self.data['x'][i_wt] = wms[0].x

                    self.data['zc_ref'][i_wt] = [wm.z for wm in wms]
                    self.data['zc_mod'][i_wt] = [np.zeros( (n_t, n_x) )]

                    if (  np.sqrt((self.data['x'][i_wt][0]-wt.x[0])**2 
                                        + (self.data['zc_ref'][i_wt][0][-1,0]-wt.x[2])**2 )) > farm.af.D :
                        raise Exception(  f'Wake tracking initial position and wind'
                                        + f'turbine location do not match.')
            
            else: # if no LES data is provided, override the reference data
                dx = self.farm.lag_solver.grid.dx
                x_loc = np.arange(0,self.farm.lag_solver.grid.margin[0][1],dx)

                n_t = len(farm)
                n_x = len(x_loc)

                self.data['x']      = np.empty(farm.n_wts, dtype=object)

                self.data['zc_ref'] = np.empty(farm.n_wts, dtype=object)
                self.data['zc_mod'] = np.empty(farm.n_wts, dtype=object)

                self.data['t_ref'] = np.ones(n_t) * np.nan
                self.data['t_mod'] = self.data['t_ref']
                
                self.n_masks = 1
                for i_wt, wt in enumerate(self.farm.wts):
                    self.data['x'][i_wt] = wt.x[0] + x_loc

                    self.data['zc_ref'][i_wt] = [np.zeros( (n_t, n_x) ) + np.nan]
                    self.data['zc_mod'][i_wt] = [np.zeros( (n_t, n_x) ) + np.nan]

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
    
    def _export(self):
        # data should only be exported once
        if self._it is None: 
            return

        if self.bf_dir is None:
            data = {k: self.data[k] for k in ['x', 't_mod', 'zc_mod']}
        else:
            data = self.data

        self.__savenpy__(f'centerline_data.npy', data, allow_pickle=True)
        # -------------------------------------------------------------------- #

    def _plot_(self):
        pass
        # -------------------------------------------------------------------- #

