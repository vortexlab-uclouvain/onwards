from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import List

C2TPI = 2*np.pi

class Sensors():
    def __init__(self, data_path: str, fs: float = None):
        """ Prototype Sensors class.

        Parameters
        ----------
        data_path: str
            Path to the data file.
        fs: float, optional
            Sampling frequency in [Hz]. If None, the original time sampling
            is preserved, by default None.

        See also
        --------
        :meth:`Turbines.__init_sensors__<.turbine.Turbine.__init_sensors__>`
        """
        self.fs = fs
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
        """ Retrieves the current value of the measurement, fld.

        Parameters
        ----------
        fld : str
            Name of the field/measurement.
        i_b : int, optional
            Blade index, by default None.

        Returns
        -------
        float
            The current value of field `m (for blade `i_b`).s
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def __contains__(self, fld: str):
        """
        Check if the field/measurement is available.
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
    def __init__(self, data_path:str, fs:float=None, field_names:List[float]=None, 
                        time_bnds:list=None, zero_origin:bool=False, **kwargs):
        """ Inits a SensorPy object that extract data from LES output files.

        This class allows to read data from the wind turbine data extracted from
        the high fidelity Large Eddy Simulation solver BigFlow (developed at UCLouvain).
        This solver models turbine as actuator disks and obtains the blades loads 
        by projecting the disk loads over fictive blades. 

        The wind turbine feeds its measurement to the OnWaRDS frameworks which 
        allows to compare its output to the reference LES data.
        
        Parameters
        ----------
        data_path : str
            Path to the BigFlow data file.
        fs : float, optional
            Sampling frequency in [Hz]. If None, the original time sampling.s
            is preserved, by default None.
        field_names : list, optional
            Name of the fields that should be extracted from the LES data.
        time_bnds : list, optional
            Data range to be extracted: time_bnds[0] is the simulation start time 
            and time_bnds[-1] is the end time, by default None.
        zero_origin : bool, optional
            If set to true, the time vector will me updated to that the initial
            time is set to 0, by default False.

        Raises
        ------
        KeyError
            If the data field/measurement requested is not available.
        ValueError
            If the sensors time bounds selected fall outside the available range.
        Exception
            If reference LES data could not be cast to the correct format.
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
        if fs or time_bnds:
            lg.debug( 'Resampling field %s', fld)
            time_bnds   = time_bnds or [self.time[0], self.time[-1]]
            time_interp = np.arange(*time_bnds, 1./fs)

            if time_bnds[0]<self.time[0] or self.time[-1]<time_bnds[-1]:
                raise ValueError('Sensors time bounds selected outside available range.')

            for fld in field_names:
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
        
        lg.info(  f'{self.n_time} data points available from {self.time[0]:.1f} '
                + f'to {self.time[-1]:.1f}s (sampling frequency: {fs:.2f} Hz and '
                + f'offset: {self.t0:.1f} s)' )
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
    def __init__(self, i_bf: int, farm_data_dir: str, export: str, **kwargs):
        """ Sensor class that allows to read data from preprocessed estimated 
        turbine states in which case m_wt = s_wt

        This class allows for fast computations when evaluating the performances 
        of the Lagrangian (ie: skips the turbine states estimation step).

        The user can generate its own preprocessed turbine files by setting 
        est_args['export'] to the desired output location.

        Parameters
        ----------
        i_bf : int
            Index of the wind turbine
        farm_data_dir : str
            Path to the BigFlow data file.
        export : str
            Name of the subdirectory where the preprocessed sensor file where 
            saved (ie: est_args['export'])

        See also
        --------
        :class:`Estimator<.estimator.Estimator>`, 
        :class:`SensorsPreprocessed<.sensors.SensorsPreprocessed>`
        
        """                                               

        lg.info('Initializing the sensors')

        self.data_path = f'{farm_data_dir}/{export}/wt_{i_bf:02d}.npy'

        self.data   = np.load(self.data_path, allow_pickle=True)[()]

        self.t0     = self.data.pop('t0')
        self.fs     = self.data.pop('fs')

        self.n_time = len(self.data[next(iter(self.data))])
        self.time   = np.arange(self.n_time)/self.fs

        self._buffer_it = 0
        # -------------------------------------------------------------------- #

