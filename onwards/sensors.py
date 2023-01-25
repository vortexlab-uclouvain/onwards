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

import numpy as np

if TYPE_CHECKING:
    from typing import List

C2TPI = 2*np.pi

class Sensors():
    def __init__(self,  i_bf: int, data_path: str, snrs_args: dict):
        """ Prototype Sensors class.

        Parameters
        ----------
        i_bf : int
            Index of the wind turbine
        farm_data_dir : str
            Path to the data directory path.
        snrs_args : dict
            Dictionary containing the parameters used for the turbines Sensors 
            initialization (refer to :class:`.Sensors`).

        See also
        --------
        :meth:`Turbine.__init_sensors__`
        """
        self.fs = 0
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def reset(self):
        """ 
        Reset the Sensors to its initial state.
        """            
        raise NotImplementedError
        # -------------------------------------------------------------------- #
        
    def iterate(self):
        """        
        Updates the Sensors object.

        Returns
        -------
        float
            The new time in [sec].
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #
    
    def get_buffer_data(self, fld: str, i_b:int=None) -> float:
        """ Retrieves the current value of the measurement ``fld``.

        Parameters
        ----------
        fld : str
            Name of the measurement.
        i_b : int, optional
            Blade index, by default None.

        Returns
        -------
        float
            The current value of measurement ``fld`` (for blade ``i_b``).
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def __contains__(self, fld: str):
        """
        Check if the measurement, ``fld``, is available.
        """        
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def __len__(self):
        """
        Return the number of measurements contained in the data.
        """        
        raise NotImplementedError
        # -------------------------------------------------------------------- #

class SensorsPy(Sensors):
    def __init__(self, data_path: str, fs: float = None, field_names: List[float] = None,
                 time_bnds: list[float] = None, zero_origin: bool = False, **kwargs):
        """ Inits a SensorPy object that extract data from AD output files.

        This class allows to read data extracted by the Actuator Disks used within
        the high fidelity Large Eddy Simulation solver (developed at UCLouvain [1]_).
        The Actuator Disk provides a full description of the wind turbine operating 
        settings (rotation speed, pitch angle, etc.) and loads. The later are 
        obtained by projecting the disk loads over fictive blades. 

        The SensorsPy object serves as an interface between the High Fidelity 
        numerical environment and the OnWaRDS framework. 

        Parameters
        ----------
        data_path : str
            Path to the data directory path.
        fs : float, optional
            Sampling frequency in [Hz]. If None (by default), the original time 
            sampling is preserved.
        field_names : list, optional
            Name of the measurements to be extracted from the Actuator Disk data.
        time_bnds : list[float], optional
            Data range to be extracted: ``time_bnds[0]`` is the simulation start time 
            and ``time_bnds[-1]`` is the end time, by default None.
        zero_origin : bool, optional
            If set to True, the time vector will be updated so that the initial
            time is set to 0, by default False.

        Raises
        ------
        KeyError
            If the data field/measurement requested is not available.

        References
        ----------
            .. [1]  M. Moens, M. Duponcheel, G. Winckelmans, and P. Chatelain. An actuator disk method with tip-loss correction based on local effective upstream velocities. Wind Energy, 21(9):766–782, 2018.
        """        
        lg.info('Initializing the sensors')

        self.data_path = data_path

        data = np.load(data_path, allow_pickle=True).item()

        self.t0     = (not zero_origin) * data['time'][0]
        self.time   = data['time'] - self.t0
        self.n_time = self.time.shape[0]
        self.fs     = fs or 1/(self.time[1]-self.time[0])

        self.r   = data['r']
        self.n_r = self.r.shape[0]
        self.dr  = self.r[1] - self.r[0]

        self.n_b   = int(data['nB'])
        
        self.data   = dict()

        field_names = field_names or list(data['data'].keys())
        if 'theta' not in field_names: field_names.append('theta')
        
        # Initialize data structure
        for fld in field_names:
            if not fld in data['data']:
                raise KeyError(f"Data has no '{fld}' field available.")
            lg.debug( 'Importing field %s', fld)
            self.data[fld] = data['data'][fld]


        # Resample data
        if fs:  self.interp(fs=fs, time_bnds=time_bnds)

        lg.info(  f'{self.n_time} data points available from {self.time[0]:.1f} '
                + f'to {self.time[-1]:.1f}s (sampling frequency: {self.fs:.2f} Hz and '
                + f'offset: {self.t0:.1f} s)' )
        # -------------------------------------------------------------------- #

    def interp(self, fs: float = None, time_bnds: List[float] = None):
        """ Reinterpolate the data within the specified bounds with the specified frequency.

        Parameters
        ----------
        fs : float, optional
            Sampling frequency in [Hz]. If None (by default), the original time 
            sampling is preserved.
        time_bnds : list[float], optional
            Data range to be extracted: ``time_bnds[0]`` is the simulation start time 
            and ``time_bnds[-1]`` is the end time, by default None.

        Raises
        ------
        ValueError
            If the sensors time bounds selected fall outside the available range.
        Exception
            If reference LES data could not be cast to the correct format.
        """        
        if fs or time_bnds:
            time_bnds   = time_bnds or [self.time[0], self.time[-1]]
            time_interp = np.arange(*time_bnds, 1./fs)

            if time_bnds[0]<self.time[0] or self.time[-1]<time_bnds[-1]:
                raise ValueError('Sensors time bounds selected outside available range.')

            for fld in self.data:
                lg.debug( 'Resampling field %s', fld)
                if len(self.data[fld])==self.n_time:
                    self.data[fld] = np.interp(time_interp, self.time, self.data[fld])

                elif len(self.data[fld])==self.n_b and len(self.data[fld][0])==self.n_time:
                    for i_b in range(self.n_b):
                        self.data[fld][i_b] = np.interp(time_interp, self.time, self.data[fld][i_b])
                
                elif len(self.data[fld])==self.n_b and len(self.data[fld][0])==self.n_r and len(self.data[fld][0][0])==self.n_time:
                    pass

                else: raise Exception('Inconsistent field format.')

            self.time   = time_interp

        self.n_time = len(self.time)
        self._buffer_it = 0

        # Fix signal corruption due to interpolation
        if 'theta' in self.data:
            for i_b in range(self.n_b):
                _x = self.time
                _y = self.data['theta'][i_b]
                rot_dir = np.sign(_y[1]-_y[0])
                for i_t in range(2,self.n_time-2):
                    if rot_dir*_y[i_t-1]>rot_dir*_y[i_t]>rot_dir*_y[i_t+1]:
                        _dydx   = .5 * ( (_y[i_t-1] - _y[i_t-2]) / (_x[i_t-1] - _x[i_t-2])
                                    + (_y[i_t+2] - _y[i_t+1]) / (_x[i_t+2] - _x[i_t+1]) )

                        _y[i_t] = (.5 * ( (( _y[i_t-1] ) + _dydx * (_x[i_t] - _x[i_t-1]))%C2TPI
                                        + (( _y[i_t+1] ) - _dydx * (_x[i_t+1] - _x[i_t]))%C2TPI ))
                _y %= C2TPI

    def reset(self):
        self._buffer_it = 0
        # -------------------------------------------------------------------- #

    def iterate(self):
        if self._buffer_it < self.n_time - 1:
            self._buffer_it = self._buffer_it + 1
            return self.get_buffer_data('time')
        else:
            raise StopIteration()
        # -------------------------------------------------------------------- #
    
    def get_buffer_data(self, fld, i_b=None):
        return (self[(fld,i_b)] if i_b is not None else self[fld])[self._buffer_it]
        # -------------------------------------------------------------------- #

    def __getitem__(self, key):
        if key=='time':  
            return self.time
        elif isinstance(key, str):
            return self.data[key]
        elif len(key)==2:
            key, i_b = key   
            return self.data[key][i_b] 
        else:
            raise Exception('Data type not supported')
        # -------------------------------------------------------------------- #

    def __contains__(self, key):
        if key=='time':     return True
        else:               return key in self.data 
        # -------------------------------------------------------------------- #

    def __len__(self):
        return self.n_time
        # -------------------------------------------------------------------- #

    def get_sensor_data(self, fld, t, i_b=None):
        return np.interp( t, self.time, self[fld] if i_b is None else self[fld,i_b])
        # -------------------------------------------------------------------- #

class SensorsPreprocessed(SensorsPy):
    def __init__(self, i_bf: int, farm_data_dir: str, name: str, fs: float, **kwargs):
        """ Imports the wind turbine estimated states from past simulations. 

        Along with the :class:`.StateExport` class, it allows direct 
        feed through of the sensors measurements to the ``Farm.lag_solver``.

        This allows for fast computations when evaluating the performances 
        of the Lagrangian (ie: skips the turbine states estimation step).
        
        Parameters
        ----------
        i_bf : int
            Index of the wind turbine
        farm_data_dir : str
            Path to the data directory path.
        name : str
            Name of the subdirectory where the preprocessed sensor file where 
            saved (ie: export_args['name'])
        fs : float
            Sampling frequency in [Hz]. If None (by default), the original time 
            sampling is preserved.


        See also
        --------
        :class:`.Estimator`, 
        :class:`.StateExportBuffer`
        """                                               

        lg.info('Initializing the sensors')

        self.data_path = f'{farm_data_dir}/{name}/wt_{i_bf:02d}.npy'

        self.data   = np.load(self.data_path, allow_pickle=True)[()]

        self.t0     = self.data.pop('t0')
        self.fs     = self.data.pop('fs')

        self.n_time = len(self.data[next(iter(self.data))])
        self.time   = np.arange(self.n_time)/self.fs
        self.n_b    = 3

        self.interp(fs=fs)
        self.fs = fs
        self._buffer_it = 0
        # -------------------------------------------------------------------- #

C2TPI = 2*np.pi

class SensorsDecoy(Sensors):
    def __init__(self, fs: float, time_bnds: List[float], **kwargs):
        self.fs     = fs
        self.time   = np.arange(*time_bnds, 1/fs) 
        self.n_time = len(self.time)
        self._buffer_it = 0
        # -------------------------------------------------------------------- #

    def reset(self):
        self._buffer_it = 0
        # -------------------------------------------------------------------- #
        
    def iterate(self):
        if self._buffer_it < self.n_time - 1:
            self._buffer_it = self._buffer_it + 1
            return self.get_buffer_data('time')
        else:
            raise StopIteration()
        # -------------------------------------------------------------------- #

    def get_buffer_data(self, fld: str, i_b:int=None) -> float:
        if fld == 'time':
            return self.time[self._buffer_it]
        else:
            raise ValueError(f'{fld} not available')
        # -------------------------------------------------------------------- #

    def __contains__(self, fld: str):
        return fld=='time'
        # -------------------------------------------------------------------- #

    def __len__(self):
        return self.n_time
        # -------------------------------------------------------------------- #