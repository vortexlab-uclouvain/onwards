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
        self.farm       = lag_solver.farm
        self.lag_solver = lag_solver
        self.set        = grid_args
        
        lg.info('Generating the global grid')

        # Sub domain bounds
        margin = [ [ 1*self.farm.af.D , 12*self.farm.af.D ],\
                      [ 2*self.farm.af.D ,  2*self.farm.af.D ] ]
        margin = self.set.setdefault('margin', margin)

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
        dx = self.set.setdefault('dx', 20)
        dz = self.set.setdefault('dz', 20)

        self._x = np.arange(self.x_bnds[0], self.x_bnds[1]+dx, dx, dtype=np.float)
        self._z = np.arange(self.z_bnds[0], self.z_bnds[1]+dz, dz, dtype=np.float)

        self.xx, self.zz = np.meshgrid(self._x, self._z, indexing='ij')
        self.mesh = self.xx, self.zz

        self._mesh_vec      = Vec(self.xx), Vec(self.zz)
        self._u_vec_buffer  = Vec((2, *(self.xx.shape))) 
        self._du_vec_buffer = Vec((2, *(self.xx.shape))) 

        lg.info('  Grid boundaries : x- {} > x+ {} | z- {} > z+ {} [m]' \
                                                       .format(*self.x_bnds, *self.z_bnds))
        lg.info('  Grid spacing    : dx {} | dz {}  [m] ({} points)' \
                                        .format(dx, dz, np.product(self.xx.shape)))

    def u_fm_compute(self, **kwargs):
        return self.lag_solver.interp_FlowModel(*self._mesh_vec, buffer=self._u_vec_buffer, **kwargs)
    
    def du_wm_compute(self, subGridFlag=False):
        return self.lag_solver.interp_WakeModel(*self._mesh_vec, buffer=self._du_vec_buffer)

    def u_compute(self, **kwargs):
        return (  self.u_fm_compute(filt=kwargs.get('filt','flow'))
                - self.du_wm_compute()                              )