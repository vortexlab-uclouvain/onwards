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
 
MYBLUE = [0.25, 0.25, 0.75]
MYRED  = [0.8, 0.1, 0.3]

REF = {'linewidth':0.7, 'color':'k', 'label': 'BigFlow'}
MOD = {'linewidth':1, 'color':MYBLUE, 'label': 'Flow Model'}
WT = {'linewidth':2.5, 'color':'k'}
REFscat = {'color':'k', 'label': 'BigFlow'}
MODscat = {'color':MYBLUE, 'label': 'Flow Model'}

import matplotlib.pyplot as plt
CMAP =  plt.get_cmap('viridis')
CMAP_VEL =  plt.get_cmap('seismic')