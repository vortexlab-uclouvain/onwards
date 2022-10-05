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
 
from .lagSolver import LagSolver
from .lagSolver import Grid

# from .src import pyCommunicator as pyCom
from .libc.pyCommunicator import c_LagSolver, c_WakeModel, c_FlowModel, c_Turbine
from .libc.pyCommunicator import c_LagSolver_p, c_WakeModel_p, c_FlowModel_p, c_Turbine_p

from .libc.pyCommunicator import add_WindTurbine

