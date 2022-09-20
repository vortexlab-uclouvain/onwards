from __future__ import annotations
from typing import TYPE_CHECKING

import logging

lg = logging.getLogger(__name__)

import numpy as np

from .libc.pyCommunicator import Vec

if TYPE_CHECKING:
    from .lagSolver import LagSolver
    from ..farm     import Farm

class Grid():
    set: dict
    farm: Farm
    lag_solver: LagSolver

    def __init__(self, lag_solver: LagSolver, grid_args: dict):
        """ Inits the Grid over which data will be interpolated. 

        Parameters
        ----------
        lag_solver : LagSolver
            Parent LagSolver object
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization.

            grid_args description

            :enable:     *(bool,  optional)* - 
                If False, the grid is turned off, by default True.
            :dx:         *(float, optional)* - 
                x grid spacing in [m], by default 20.
            :dx:         *(float, optional)* - 
                x grid spacing in [m], by default 20.
            :dz:         *(float, optional)* - 
                z grid spacing in [m], by default 20.
            :margin:     *(List[List[float]], optional)* - 
                ``[[xm, xp], [ym, yp]]`` minimal margin of the domain around 
                each Turbine.
        """
        self.farm       = lag_solver.farm
        self.lag_solver = lag_solver
        self.set        = grid_args
        
        lg.info('Generating the global grid')

        # Sub domain bounds
        margin = [ [ 1*self.farm.af.D , 12*self.farm.af.D ],\
                      [ 2*self.farm.af.D ,  2*self.farm.af.D ] ]
        margin = self.set.setdefault('margin', margin)
        self.margin = margin

        # Main domain bounds
        bnds_check = False
        if all (key in self.set for key in ('x_bnds','z_bnds')):
            self.x_bnds = self.set['x_bnds']
            self.z_bnds = self.set['z_bnds']
            
            # Checking margin are consistent
            warn_str = []
            x_lim = min(self.farm.x_wts[:,0]) - margin[0][0]
            if self.x_bnds[0] > x_lim :
                warn_str.append(['Left',  str(self.x_bnds[0]), 'high', '<', str(x_lim)])
            x_lim = max(self.farm.x_wts[:,0]) + margin[0][1]
            if self.x_bnds[1] < x_lim :
                warn_str.append(['Right', str(self.x_bnds[1]), 'low',  '>', str(x_lim)])
            z_lim = min(self.farm.x_wts[:,2]) - margin[1][0]
            if self.z_bnds[0] > z_lim :
                warn_str.append(['Lower',  str(self.z_bnds[0]), 'high', '<', str(z_lim)])
            z_lim = max(self.farm.x_wts[:,2]) + margin[1][1]
            if self.z_bnds[1] < z_lim :
                warn_str.append(['Upper', str(self.z_bnds[1]), 'low',  '>', str(z_lim)])
            if warn_str: 
                bnds_check = False
                for s in warn_str:
                    lg.warning('%s bound specicified (%s [m]) too %s (should be %s %s [m]) ',*s)
            else:
                bnds_check = True

        if not bnds_check:
            self.x_bnds  = ( min(self.farm.x_wts[:,0])-margin[0][0],
                             max(self.farm.x_wts[:,0])+margin[0][1] )
            self.z_bnds  = ( min(self.farm.x_wts[:,2])-margin[1][0],
                             max(self.farm.x_wts[:,2])+margin[1][1] )
        
        self.set['x_bnds'] = self.x_bnds
        self.set['z_bnds'] = self.z_bnds

        # Mesh initialization
        self.dx = self.set.setdefault('dx', 20)
        self.dz = self.set.setdefault('self.dz', 20)

        self._x = np.arange(self.x_bnds[0], self.x_bnds[1]+self.dx, self.dx, dtype=np.float)
        self._z = np.arange(self.z_bnds[0], self.z_bnds[1]+self.dz, self.dz, dtype=np.float)

        self.xx, self.zz = np.meshgrid(self._x, self._z, indexing='ij')
        self.mesh = self.xx, self.zz

        self._mesh_vec      = Vec(self.xx), Vec(self.zz)
        self._u_vec_buffer  = Vec((2, *(self.xx.shape))) 
        self._du_vec_buffer = Vec((2, *(self.xx.shape))) 

        self._u_was_updated  = False
        self._du_was_updated = False

        lg.info('  Grid boundaries : x- {} > x+ {} | z- {} > z+ {} [m]' \
                                                       .format(*self.x_bnds, *self.z_bnds))
        lg.info('  Grid spacing    : dx {} | dz {}  [m] ({} points)' \
                                        .format(self.dx, self.dz, np.product(self.xx.shape)))      
        # -------------------------------------------------------------------- #

    def update(self):
        """
        Updates the grid
        """
        self._u_was_updated  = False
        self._du_was_updated = False
        # -------------------------------------------------------------------- #

    def u_fm_compute(self, filt: str = 'flow') -> np.array:
        """ Interpolates the ambient flow model over the Grid.

        Parameters
        ----------
        filt : str, optional
            ``flow`` or ``rotor`` depending on the width of the filter used for the 
            ambient velocity field computation, by default ``flow``.

        Returns
        -------
        np.array
            The ambient flow field interpolated at the grid locations.

        Raises
        ------
        ValueError    
            If filt is not valid (ie: ``rotor`` or ``flow``).
        """
        if filt not in ['flow', 'rotor']:
            raise ValueError('Filter type not recognized (should be `flow` or `rotor`.')

        if filt == 'rotor':
            return self.lag_solver.interp_FlowModel(*self._mesh_vec)

        if filt == 'flow':
            if not self._u_was_updated:
                self.lag_solver.interp_FlowModel(*self._mesh_vec, buffer=self._u_vec_buffer)
                self._u_was_updated = True

            return self._u_vec_buffer.x

        # -------------------------------------------------------------------- #
    
    def du_wm_compute(self, subGridFlag=False) -> np.array:
        """ 
        Interpolates the wake flow model over the Grid.

        Returns
        -------
        np.array
            The ambient wake field interpolated at the grid locations.
        
        """
        if not self._du_was_updated:
            self.lag_solver.interp_WakeModel(*self._mesh_vec, buffer=self._du_vec_buffer)
            self._du_was_updated = True
        return self._du_vec_buffer.x        
        # -------------------------------------------------------------------- #

    def u_compute(self, filt='flow') -> np.array:
        """ 
        Interpolates the flow model over the Grid.

        Parameters
        ----------
        filt : str, optional
            ``flow`` or ``rotor`` depending on the width of the filter used for
            the ambient velocity field computation, by default ``flow``.

        Returns
        -------
        np.array
            The ambient flow field interpolated at the grid locations.
        """

        return (  self.u_fm_compute(filt=filt)
                - self.du_wm_compute()                              )
        # -------------------------------------------------------------------- #
