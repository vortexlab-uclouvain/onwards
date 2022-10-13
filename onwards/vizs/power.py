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
from posixpath import lexists
from typing import TYPE_CHECKING

import logging

from ..turbine import MINIMAL_STATES 
lg = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

from .viz import Viz
from . import linespecs
if TYPE_CHECKING:
    from typing import List
    from ..farm import Farm

class Viz_power(Viz):
    viz_type = 'power'

    def __init__(self, farm: Farm, 
                 ylim: List[float, float]=None, 
                 xlim: List[float, float]=None):
     
        super().__init__(farm)

        self.xlim = xlim 
        self.ylim = ylim 

        self.time = np.ones( len(farm) ) * np.nan
        self.data = np.ones( len(farm) ) * np.nan

        self._it = 0
        # -------------------------------------------------------------------- #

    def reset(self):
        self._it = 0
        # -------------------------------------------------------------------- #

    def update(self):
        power = 0.0
        for wt in self.farm.wts:
            power += wt.estimators[0].turbine_dynamics.drivetrain.Q_g
        self.time[self._it] = self.farm.t
        self.data[self._it] = power
        
        self._it += 1
        # -------------------------------------------------------------------- #

    def _data_clean(self):
        self.time = self.time[:self._it-1]
        self.data = self.data[:self._it-1]
        # -------------------------------------------------------------------- #

    def _export(self):
        out = {'time':self.time,'label':self.l_map,'data':self.data}
        self.__savenpy__(f'estimator_data.npy', out, allow_pickle=True)
        # -------------------------------------------------------------------- #

    def _plot(self):
            plt.figure()
            
            plt.plot(self.time, self.data, **linespecs.MOD)

            plt.ylabel('power')

            plt.xlabel('t [s]')
            self.__savefig__(f'power.pdf')
        # -------------------------------------------------------------------- #
