from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import numpy as np

from .      import estimators 
from .utils import LoggingDict
 
if TYPE_CHECKING:
    from typing   import List
    from .farm    import Farm
    from .sensors import Sensors

# MINIMAL_STATES contains the estimated wind turbine states s_wt required by the 
# LagSolver 
MINIMAL_STATES = [ 'u_inc', 'u_fs', 'w_inc', 'w_fs', 'ct', 'ti', 'yaw' ]

class Turbine:
    farm : Farm
    snrs : Sensors
    estimators : List[estimators.Estimator]
    est_export : estimators.StateExportBuffer


    def __init__(self, farm: Farm, i_wt: int, snrs_args: dict, est_args:list):
        """ Inits Turbine

        The Turbine object gathers the wind turbines measurements, m_wt,  and translates 
        them into the estimated wind turbine state, s_wt.

        Parameters
        ----------
        farm : Farm
            Parent farm object.
        i_wt : int
            Index of the wind turbine (in the `farm` object).
        snrs_args : dict
            Dictionary containing the parameters used for the turbines Sensors 
            initialization (refer to :class:`.Sensors`).
        est_args : dict
            Dictionary containing the parameters used for the turbines Estimators 
            initialization (refer to :class:`Estimator<.estimators.estimator.Estimator>`).
        
        See also
        --------
        :meth:`<.Turbine.__init_sensors__>`,
        :meth:`<.Turbine.__init_states__>`,
        :meth:`<.Turbine.init_LagSolver>`
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

    def reset(self, ini_states: dict[str, float]={}):       
        """ Reset the wind turbines.

        Parameters
        ----------
        ini_states : dict[str, float], optional
            {'s_wt_i': v} maps the wind turbine state, s_wt_i, to its initial 
            value, v.
        """        

        # Reset sensors
        self.snrs.reset()
        self.t = self.snrs.get_buffer_data('time')

        # Reset estimators and associated states
        for e in self.estimators:
            e.reset()

        self.update_states()
        for s in ini_states:
            self.states[s] = ini_states[s]

        # Update communicator
        self.update_LagSolver()
        # -------------------------------------------------------------------- #

    def __init_sensors__(self, snrs_args: dict):
        """ Inits the Turbine's Sensors

        :class:`.Sensors` provide easy interfacing between OnWaRDS 
        and the data source allow to access some measurement, m_wt, at the current 
        time, t, ``using the get_buffer_data('m_wt')`` method.

        Parameters
        ----------
        snrs_args : dict
            Dictionary containing the parameters used for the turbines Sensors 
            initialization (refer to :class:`.Sensors`).

        Returns
        -------
        fs : float 
            Sampling frequency in [Hz].
        t : float
            Time [s]. 
        snrs: list of Sensors
            List of n_wt sensors (one sensors object per WT)

        Note
        ----
        Custom Sensors can be defined by the user following the example provided 
        hereunder for MySensors::
        >>> if snrs_type=='MySensors':
        >>>    from sensors import MySensors
        >>>    self.snrs = MySensors()

        See also
        --------
        :class:`.Sensors`
        """

        if   snrs_args['type']=='SensorsPy':
            from .sensors import SensorsPy
            snrs_fid = f'{self.farm.data_dir}/sensorsData_{self.i_bf}.npy' 
            self.snrs = SensorsPy(snrs_fid, **snrs_args)

        elif snrs_args['type']=='SensorsPreprocessed':
            from .sensors import SensorsPreprocessed
            self.snrs = SensorsPreprocessed(self.i_bf, self.farm.data_dir, **snrs_args)

        # elif snrs_type=='MySensors': # custom sensors class example
        #     from sensors import MySensors
        #     self.snrs = MySensors()
        
        else:
            raise ValueError('Sensors type not recognized.')

        self.fs   = self.snrs.fs
        self.t    = self.snrs.get_buffer_data('time')
        # -------------------------------------------------------------------- #

    def update_sensors(self):
        """
        Update the Turbine's sensors
        """
        self.t = self.snrs.iterate()
        return self.t
        # -------------------------------------------------------------------- #

    def __init_states__(self, est_args: dict):
        """ Inits the Turbine's Estimators
        
        :class:`Estimators<.estimator.Estimator>` allow to translate the wind 
        measurements, m_wt, into the estimated wind turbine states, s_wt.

        Parameters
        ----------
        est_args : dict
            Dictionary containing the parameters used for the turbines Estimators 
            initialization.

            Available fields:
            
            :estimatiori:       *(dict)* - 
                Estimator parameters (cfr: Sensors) where i is the estimator index. 
                Estimator type retrieved from dict['type']. At least one estimator 
                is required
            :n_substeps:        *(int, optional)* - 
                Number of (sensors) time steps between two successive states 
                update, by default 1.
            :export:            *(str, optional)* - 
                Export flag, by default False. If set to a string, the wind 
                turbine's estimated state, s_wt, is exported to the export_dir
                subdirectory (refer to :class:`SensorsPreprocessed<.sensors.SensorsPreprocessed>`).
            :export_overwrite:  *(bool, optional)* - 
                overwrite data if set to True, by default False.
            :export_user_field: *(List[str], optional)* - 
                List of the measurements, not part of the wind turbine state, s_wt, 
                that should be appended to the exported state vector.

        Raises
        ------
        Exception 
            If some of the state defined by MINIMAL_STATES are not computed.

        Note
        ----
        Custom Estimators can be defined by the user following the example 
        provided hereunder and following the `Estimator` class specifications. 

        Estimators are applied recursively starting from the first `estimator0` 
        to `estimatorn`. The states are updated accordingly as some Estimator 
        might rely on previous state estimation to compute their output state. 
        eg: ct estimations are likely to depend on estimation of the rotor 
        effective wind speed.   

        See also
        --------
        :class:`.estimator.Estimator`, 
        :class:`.sensors.SensorsPreprocessed`, 
        :class:`.estimators.stateExport.StateExportBuffer`
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

        if 'export' in est_args:
            self.est_export = estimators.StateExportBuffer(self, 
                                              est_args['export'], 
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
        Updates the Turbine's state, s_wt.
        """
        for e in self.estimators: 
            e.update()
        if self.est_export: self.est_export.update()

        lg.info( f'Updating states: new state:'
                + "".join([f" {k}: {e:.2f}" for k, e in self.states.items()]) )
        # -------------------------------------------------------------------- #

    def init_LagSolver(self):
        """
        Inits the LagSolver communicator that allows ctypes to access s_wt for 
        the current turbine.
        """        
        if self.farm.lag_solver:
            from .lagSolver import add_WindTurbine, c_Turbine
            self.c_wt = c_Turbine( self, **self.states ) 
            add_WindTurbine(self.farm.lag_solver.p, self.c_wt.p)
        # -------------------------------------------------------------------- #

    def update_LagSolver(self):
        """
        Updates the LagSolver communicator.
        """        
        self.c_wt.update(**self.states)
        # -------------------------------------------------------------------- #
        
    def get_bounds(self) -> List[float]:
        """
        Return the position (x, z) of the tip of the turbines blades.

        Returns
        -------
        List[float]
            The position (x, z) of the tip of the turbines blades given the 
            current rotor orientation.
        """        
        theta = np.deg2rad(self.states['yaw'])
        tmp = np.array([-1,1]) * self.af.R
        return [ self.x[0] + tmp*np.sin(theta), 
                 self.x[2] + tmp*np.cos(theta) ]
        # -------------------------------------------------------------------- #

    def is_freestream(self) -> bool:
        """ Check if the wind turbine is waked or not

        Returns
        -------
        bool
            False if an impinging wake was detected else True.
        """
        if getattr(self.farm,'lag_solver',False): return bool(self.c_wt.is_fs)
        else:                                     return True
        # -------------------------------------------------------------------- #

    def __exit__(self):
        if self.est_export: self.est_export.save()
        # -------------------------------------------------------------------- #
