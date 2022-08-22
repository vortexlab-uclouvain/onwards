import logging
lg = logging.getLogger(__name__)

import numpy as np

C2TPI = 2*np.pi

class Sensors():

    def __init__(self, data_path:str, fs:float=None):
        """
        Inits Sensors

        Parameters
        ----------
        data_path : str
            path of the data file
        fs : float
            sampling frequency in [Hz]. Default: None (The original time sampling
             is preserved)
        """
        self.fs = fs
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def __getitem__(self, key):
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def __contains__(self, key):
        raise NotImplementedError
        # -------------------------------------------------------------------- #

    def __len__(self):
        raise NotImplementedError
        # -------------------------------------------------------------------- #
        
    def iterate(self):
        """        
        Updates the Sensors object (moves to the next timestep) 

        Returns
        -------
        t: float
            The update time in [sec]

        Raises
        ------
        StopIteration if the last time step available is reached
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #
    
    def get_buffer_data(self, fld, i_b:int=None):
        """       
       Retrieves the current value of the selected field

        Parameters
        ----------
        fld : str
            field type
        i_b : int
            blade index
        
        Returns
        -------
        out: float
            The current value of field `fld` (for blade `i_b`)
        """
        raise NotImplementedError
        # -------------------------------------------------------------------- #

class SensorsPy(Sensors):
    def __init__(self, data_path:str, fs:float=None, field_names:list=None, time_bnds:list=None, zero_origin:bool=False, **kwargs):

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
                raise ValueError('Sensors time bounds selected outside available range')

            for fld in field_names:
                if not fld in self.data.keys():
                    raise KeyError('Sensors has no data {} available'.format(fld))

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

    def iterate(self):
        if self._buffer_it < self.n_time - 1:
            self._buffer_it = self._buffer_it + 1
            return self.get_buffer_data('time')
        else:
            raise StopIteration()
        # -------------------------------------------------------------------- #

    def get_sensor_data(self, fld, t, i_b=None):
        return np.interp( t, self.time, self[fld] if i_b is None else self[fld,i_b])
        # -------------------------------------------------------------------- #
    
    def get_buffer_data(self, fld, i_b=None):
        return (self[(fld,i_b)] if i_b is not None else self[fld])[self._buffer_it]
        # -------------------------------------------------------------------- #

class SensorsPreprocessed(SensorsPy):
    def __init__(self, i_bf: int, farm_data_dir: str, export: str, 
                                               export_dir: str=False, **kwargs):

        lg.info('Initializing the sensors')

        self.data_path = f'{export_dir or farm_data_dir}/{export}/wt_{i_bf:02d}.npy'

        self.data   = np.load(self.data_path, allow_pickle=True)[()]

        self.t0     = self.data.pop('t0')
        self.fs     = self.data.pop('fs')

        self.n_time = len(self.data[next(iter(self.data))])
        self.time   = np.arange(self.n_time)/self.fs

        self._buffer_it = 0
        # -------------------------------------------------------------------- #

