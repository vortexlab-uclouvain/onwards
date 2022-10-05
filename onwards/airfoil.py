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
 
import logging
lg = logging.getLogger(__name__)

import ctypes
import numpy as np

from ctypes import c_int, c_double, c_char_p

RHO  = 1.225
DATA = { 'NREL': {'Rtip':63., 'Rhub':1.5, 'nB':3, 'Rhub_forTL':8.3333 } }

class Airfoil:
    # ----------------------- __init__ ------------------------ #
    def __init__(self, blade_name: str):
        """
        Inits Airfoil

        Parameters
        ----------
        blade_name : str
            name of the blade profile (eg: ``NREL``)  

        Note
        ----
        This is the minimal configuration of the Airfoil class (hence: 
        ``minimal_config=True``) which facilitates the coupling with the full
        Airfoil library. 
        """
        self.minimal_config = True

        if blade_name not in DATA:
            raise ValueError(f'Requested Airfoil {blade_name} is not available.')
        self.R          = DATA[blade_name]['Rtip']
        self.Rhub       = DATA[blade_name]['Rhub']
        self.nB         = DATA[blade_name]['nB'   ]
        self.Rhub_forTL = DATA[blade_name]['Rhub_forTL']

        self.name   = blade_name
        self.D      = 2.*self.R

        self.A      = np.pi * self.R**2
        self.cCTfac = 0.5 * RHO * self.A

        self._af_c_ = _c_Airfoil(self)
        self.p      = ctypes.pointer(self._af_c_)

class _c_Airfoil(ctypes.Structure):
    _fields_ = [ ("nB",          c_int),
                 ("R",           c_double),
                 ("Rhub",        c_double),
                 ("Rhub_forTL",  c_double),
                 ("D",           c_double),
                 ("A",           c_double),
                 ("cCTfac",      c_double), ]
    def __init__(self, af: Airfoil):
        for f in self._fields_: setattr(self, f[0], getattr(af, f[0]))

c_Airfoil_p = ctypes.POINTER(_c_Airfoil) 
""" ctypes Airfoil pointer """