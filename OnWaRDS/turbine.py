from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import numpy as np

from .      import estimators 
from .utils import LoggingDict
 
if TYPE_CHECKING:
    from typing  import List
    from .farm   import Farm
    from .sensor import Sensors

# MINIMAL_STATES contains the estimated wind turbine states s_WT required by the 
# LagSolver 
MINIMAL_STATES = [ 'u_inc', 'u_fs', 'w_inc', 'w_fs', 'ct', 'ti', 'yaw' ]

class Turbine:
    farm : Farm
    snrs : Sensors
    estimators : List[estimators.Estimator]
    est_export : estimators.StateExportBuffer

    def __init__(self, farm: Farm, i_wt: int, snrs_args: dict, est_args:list):
        """ Inits Turbine

        Parameters
        ----------
        farm : Farm
            parent farm
        i_wt : int
            index of the wind turbine (in the `farm` object)
        snrs_args : dict
            dict containing the sensor parameters (cfr: sensor). `Sensors` 
            type retrieved from snrs_args['type']
        est_args : dict
            dict containing the estimators parameters (cfr: estimators/est_myEst). 
           `Estimator` type retrieved from dict['type']
        """
        lg.info(f'Initializing WT {i_wt+1}/{farm.n_wts}')
        self.farm     = farm
        self.af       = farm.af

        self.i        = i_wt
        self.i_bf     = int(self.farm.wt_map[self.i])
        
        self.x        = farm.x_wts[i_wt,:]

        self.__init_sensors__(snrs_args)
        self.__init_states__(est_args)

        # -------------------------------------------------------------------- #

    def __init_sensors__(self, snrs_args: dict):
        """ Inits the Turbine's Sensors

        Sensors allow to access some measurement m at the current time t
        using the get_buffer_data('m') method

        Parameters
        ----------
        snrs_args : dict
            dict contains the sensor parameters (cfr: sensor/Sensors). `Sensors` 
            type retrieved from snrs_args['type']

        Returns
        -------
        fs : float 
            sampling frequency in [Hz]
        t : float
            time [s] 
        snrs: list of Sensors
            list of n_wt sensors (one sensor object per WT)

        Note
        ----
        Custom Sensors can be defined by the user following the example provided 
        hereunder for MySensors
        """
        if   snrs_args['type']=='SensorsPy':
            from .sensor import SensorsPy
            snrs_fid = f'{self.farm.data_dir}/sensorData_{self.i_bf}.npy' 
            self.snrs = SensorsPy(snrs_fid, **snrs_args)

        elif snrs_args['type']=='SensorsPreprocessed':
            from .sensor import SensorsPreprocessed
            self.snrs = SensorsPreprocessed(self.i_bf, self.farm.data_dir, **snrs_args)

        # elif snrs_type=='MySensors': # custom sensor class example
        #     from sensor import MySensors
        #     self.snrs = MySensors()
        
        else:
            raise ValueError('Sensor type not recognized.')

        self.fs   = self.snrs.fs
        self.t    = self.snrs.get_buffer_data('time')
        # -------------------------------------------------------------------- #

    def update_sensors(self):
        """
        Update the Turbine's sensor
        """
        self.t = self.snrs.iterate()
        return self.t
        # -------------------------------------------------------------------- #

    def __init_states__(self, est_args: dict):
        """ Inits the Turbine's Estimators
        
        Estimators allows to translate the wind measurements into the actual 
        wind turbine states (used as part of the wake model computation).

        Parameters
        ----------
        est_args : dict
            dict containing the description of the estimators used 
        est_args['n_substeps'] : int, default 1
            number of (sensor) time steps between two successive states update. 
        est_args['estimatiori'] : dict 
            Estimator parameters (cfr: Sensors) where `i` is the estimator index. 
            Estimator type retrieved from dict['type']

        Note
        ----
        Custom Estimators can be defined by the user following the example 
        provided hereunder and following the `Estimator` class specifications. 
        Estimators are applied recursively starting from the first `estimator0` 
        to `estimatorn`. The states are updated accordingly as some Estimator 
        might rely on previous state estimation to compute their output state. 
        eg: ct estimations are likely to depend on estimation of the rotor 
        effective wind speed.   

        Raises
        ------
        Exception if the minimal state, `MINIMAL_STATES`, is not computed.
        """
        avail_states    = []
        self.estimators = []

        n_estimators = len([e for e in est_args if e.startswith('estimator')])
        for i_e in range(n_estimators):
            key = f'estimator{i_e}'
            e_type = est_args[key]['type']
            lg.debug(f'Initializing {key} of type {e_type}.')

            if   e_type == 'fld_fromdata':  
                Estimator = estimators.Est_fld_fromdata
            elif e_type == 'uinc:ti_kfbem': 
                Estimator = estimators.Est_uincti_kfbem
            elif e_type == 'winc_nn':       
                Estimator = estimators.Est_winc_nn
            elif e_type == 'ct_fromthrust': 
                Estimator = estimators.Est_ct_fromthrust
            elif e_type == 'ufs:wfs_waked': 
                Estimator = estimators.Est_ufswfs_waked
            elif e_type == 'fld_debug': 
                Estimator = estimators.Est_fld_debug
            else:
                raise ValueError(f'Estimator type `{e_type}` not recognized.')
            
            est_args[key] = LoggingDict(est_args[key])
            self.estimators.append(Estimator(self, avail_states, est_args[key]))
            avail_states += [s for s in self.estimators[-1].states]
        
        for s in set(avail_states):
            if avail_states.count(s)>1:
                lg.warning(f'Some states {avail_states} are re-computed several times.')

        for ms in MINIMAL_STATES:
            if ms not in avail_states:
                raise Exception(f'No estimator associated to state {ms}.')

        self.states         = {a: 0.0 for a in avail_states}
        self.n_substeps_est = est_args.setdefault('n_substeps', 1)

        est_args['export_dir'] = est_args.get('export_dir', False)
        if 'export' in est_args:
            self.est_export = estimators.StateExportBuffer(self, 
                                              est_args['export'], 
                                              export_dir=est_args['export_dir'],
                                              export_overwrite=est_args.get('export_overwrite',False),
                                              states_user=est_args.get('export_user_field',[]))
        else:
            self.est_export = False

        self.update_states()
        if 'ini_states' in est_args:
            for s in est_args['ini_states']:
                self.states[s] = est_args['ini_states'][s]

        lg.info(  f'Estimator are updated every {self.n_substeps_est/self.fs}s.'
                + f' (n_substeps_estimators = {self.n_substeps_est})')
        # -------------------------------------------------------------------- #

    def update_states(self):
        """
        Update the Turbine's state
        """
        for e in self.estimators: 
            e.update()
        if self.est_export: self.est_export.update()

        lg.info( f'Updating states: new state:'
                + "".join([f" {k}: {e:.2f}" for k, e in self.states.items()]) )
        # -------------------------------------------------------------------- #

    def init_LagSolver(self):
        if self.farm.lag_solver:
            from .lagSolver import add_WindTurbine, c_Turbine
            self.c_wt = c_Turbine( self, **self.states ) 
            add_WindTurbine(self.farm.lag_solver.p, self.c_wt.p)
        # -------------------------------------------------------------------- #

    def update_LagSolver(self):
        self.c_wt.update(**self.states)
        # -------------------------------------------------------------------- #
        
    def get_bounds(self):
        theta = np.deg2rad(self.states['yaw'])
        tmp = np.array([-1,1]) * self.af.R
        return [ self.x[0] + tmp*np.sin(theta), 
                 self.x[2] + tmp*np.cos(theta) ]
        # -------------------------------------------------------------------- #

    def is_freestream(self):
        if getattr(self.farm,'lag_solver',False): return bool(self.c_wt.is_fs)
        else:                                     return True
        # -------------------------------------------------------------------- #

    def __exit__(self):
        if self.est_export: self.est_export.save()
        # -------------------------------------------------------------------- #
