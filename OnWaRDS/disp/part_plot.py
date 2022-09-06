from __future__ import annotations
from typing import TYPE_CHECKING

import logging 
lg = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

from .viz import Viz
if TYPE_CHECKING:
    from ..farm import Farm

CMAP = 'tab20'

class Part_plot(Viz):

    str_ls = [ {'s':8,'marker':'o','label':'wake', 'lw':0.78}, 
               {'s':10,'marker':'^','label':'flow', 'facecolors':'w', 'lw':0.78},
               {'s':8,'marker':'o', 'lw':1}                 ]
    vel_str_id = ['u', 'w']

    def __init__(self, farm: Farm, *args, **kwargs):
        """ Plots the position of the wake and ambient flow particles

        Parameters
        ----------
        farm : Farm
            Parent Farm Object
        """        

        super().__init__(farm)

        self.fig, self.axs = plt.subplots(3, 1, sharex=True)

        plt.sca(self.axs[0])
        self.part_plt = [None] * 2

        for i, (s, l) in enumerate(zip(['W', 'F'], self.str_ls)):
            c = plt.get_cmap(CMAP)(self.farm.lag_solver.get_part_iwt(s))
            self.part_plt[i] = plt.scatter(self.farm.lag_solver.get(s, 'x_p', comp=0)/self.farm.af.D,
                                           self.farm.lag_solver.get(s, 'x_p', comp=1)/self.farm.af.D,
                                           edgecolors = c,
                                           **({'facecolors':c}|l))
            plt.ylabel('$z/D$')
        plt.legend()

        plt.xlim( x/self.farm.af.D for x in [self.farm.lag_solver.grid.x_bnds[0], self.farm.lag_solver.grid.x_bnds[-1]])
        plt.ylim( x/self.farm.af.D for x in [self.farm.lag_solver.grid.z_bnds[0], self.farm.lag_solver.grid.z_bnds[-1]])

        self.vel_plt = [[None for i in range(2)] for i in range(2)]

        for comp, ax in enumerate(self.axs[1:]):
            plt.sca(ax)
            for i_f, f_str in enumerate(['u_p','uf_p']):
                self.vel_plt[i_f][comp] = plt.scatter(self.farm.lag_solver.get('F', 'x_p',  comp=0   )/self.farm.af.D,
                                                      self.farm.lag_solver.get('F', f_str,  comp=comp),
                                                      c = plt.get_cmap(CMAP)(self.farm.lag_solver.get_part_iwt('F')),
                                                      alpha= 1 - 0.9 * i_f,
                                                      **self.str_ls[0])
            plt.hlines(7,0,30)
            plt.ylabel(['$u$','$w$'][comp]+r' [ms$^{-1}$]')
            plt.xlim([0,30])

        plt.sca(self.axs[1]); plt.ylim((0,12))
        plt.sca(self.axs[2]); plt.ylim((-1,1))

        self.title = plt.suptitle(f'Time: {self.farm.t} s')
        plt.xlabel('$x/D$')
        plt.tight_layout()
        # -------------------------------------------------------------------- #

    def update(self):
        if not self.farm.update_LagSolver_flag: return

        self.title.set_text(f'Time: {self.farm.t} s')

        plt.figure(self.fig.number)
        for s, p in zip(['W', 'F'], self.part_plt):
            p.set_offsets([(x, z) for x, z in
                        zip(self.farm.lag_solver.get(s, 'x_p', comp=0)/self.farm.af.D,
                            self.farm.lag_solver.get(s, 'x_p', comp=1)/self.farm.af.D)])

        for i_f, f_str in enumerate(['u_p','uf_p']):
            for comp, p in enumerate(self.vel_plt[i_f]):
                p.set_offsets([(x, z) for x, z in
                            zip(self.farm.lag_solver.get('F', 'x_p', comp=0)/self.farm.af.D,
                                self.farm.lag_solver.get('F', f_str, comp=comp))])
                                
        plt.draw(); plt.pause(0.000001); 

        # -------------------------------------------------------------------- #

    def plot(self):
        pass

        # -------------------------------------------------------------------- #
