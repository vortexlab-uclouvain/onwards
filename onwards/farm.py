from __future__ import annotations
from typing import TYPE_CHECKING

import logging

lg = logging.getLogger(__name__)

import os
import numpy as np

from .turbine   import Turbine
from .airfoil   import Airfoil
from .lagSolver import LagSolver 
from .vizs       import Viz
from .utils     import LoggingDict, dict2txt

if TYPE_CHECKING:
    from typing import List

class Farm:
    af: Airfoil             
    lag_solver: LagSolver   
    wts: List[Turbine] 
    viz: List[Viz]

    def __init__(self, data_dir: str, af_name: str, snrs_args: dict,
                 est_args: dict, model_args: dict, grid_args: dict = {},
                 wt_cherry_picking: List[int] = None, out_dir: str = None,
                 enable_logger: bool = True):    
        r"""Inits Farm.

        The Farm object interfaces together the different modules of the OnWaRDS
        toolbox (eg: turbine state estimation, flow modeling, data i/o, plotting)
        [1]_ [2]_ [3]_.

        A Farm consists of a series of :class:`.Turbine` (``self.wts``). 
        Each wind turbine is associated to a :class:`.Sensors` object 
        that gathers its measurements, m_wt, at a prescribed time, t. These 
        measurements are then translated into the estimated turbine state, s_wt, 
        using its built-in :class:`.Estimator`.

        The wind turbine states, s_wt, are eventually fed to ``self.lag_solver``, 
        the :class:`.LagSolver` (written in c and interfaced with main 
        python code using ctypes) that uses them to estimate the flow state, s_flow.
        
        .. math::
                \mathbf{m}_{wt} 
                        \rightarrow \mathbf{\hat{s}}_{wt} 
                                \rightarrow \mathbf{\hat{s}}_{flow} 
        
        Once initialized, a Farm object can be temporally iterated over using the 
        built-in ``for`` loop. 
        Memory cleaning is automatically handled using the ``with`` statement.
        
        Plotting is handled by ``self.viz`` and custom user plot can added using
        :meth:`.Farm.viz_add`.

        Parameters
        ----------
        data_dir : str
            Path to the wind farm data.
        af_name : str
            Name of the airfoil used (eg: ``NREL``).
        snrs_args : dict
            Dictionary containing the parameters used for the turbines Sensors 
            initialization (refer to :class:`.Sensors`).
        est_args : dict
            Dictionary containing the parameters used for the turbines Estimators 
            initialization (refer to :class:`.Estimator`).
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization (refer to :class:`.LagSolver`).
        grid_args : dict, optional
            Dictionary containing the parameters used for the Grid's initialization 
            (refer to :class:`.Grid`).
        wt_cherry_picking : List[int], optional
            Each element corresponds to a wind turbine index (allows
            to pick only the desired turbines), if None (by default), all turbines 
            are modeled..
        out_dir : str, optional
            Export directory name where figures and data are saved. If ``''``,
            all exports are disabled; if None (by default), default export 
            directory name ``onwards_run_id``.
        enable_logger : bool, optional
            If True (by default), logs are saved to the export directory.

        Raises
        ------
        TypeError
            If ``wt_cherry_picking`` is not a list of valid turbines indices.

        Example
        -------
        Typical OnWaRDS initialization:

            >>> from onwards import Farm
            >>> with Farm() as f:
            >>>    v.viz_add('MyViz') 
            >>>    for t in f:
            >>>        print(f'Current time is {t}.')
            >>>    f.plot()

        References
        ----------
        .. [1] M. Lejeune, M. Moens, and P. Chatelain. A meandering-capturing wake model coupled to rotor-based flow-sensing for operational wind farm flow prediction. Frontiers in Energy Research, 10, jul 2022.
        .. [2] M. Lejeune, M. Moens, and P. Chatelain. Extension and validation of an operational dynamic wake model to yawed configurations. Journal of Physics: Conference Series, 2265(2):022018, may 2022.
        .. [3] M. Lejeune, M. Moens, M. Coquelet, N. Coudou, and P. Chatelain. Data assimilation for the prediction of wake trajectories within wind farms. Journal of Physics: Conference Series, 1618:062055, sep 2020.
    
        """
        
        self.data_dir = data_dir
        self.__init_exports__(data_dir, out_dir, enable_logger)

        # Casting all input dictionaries
        snrs_args  = LoggingDict(snrs_args)
        est_args   = LoggingDict(est_args)
        model_args = LoggingDict(model_args)
        grid_args  = LoggingDict(grid_args)

        # Farm geometry
        lg.info(f'Loading data from {data_dir}')
        _x_wt_load = np.load(f'{data_dir}/geo.npy')

        if wt_cherry_picking is None:
            self.x_wts = _x_wt_load
            self.n_wts = self.x_wts.shape[0]
            self.wt_map = list(range(self.n_wts))
        else:
            if not isinstance(wt_cherry_picking, list):
                raise TypeError('wt_cherry_picking should be a list of WT indices')
            self.x_wts = _x_wt_load[(wt_cherry_picking)]
            self.n_wts = self.x_wts.shape[0]
            self.wt_map = wt_cherry_picking

        self.af = Airfoil(af_name)

        # Initializing turbines sensors and states estimators
        self.__init_turbines__(snrs_args, est_args)
        self.it = 0
        self.t  = self.wts[0].t
        self.dt = 1./self.wts[0].fs

        # Initializing Lagrangian flow model
        self.__init_LagSolver__(model_args, grid_args)

        self.update_states_flag    = False
        self.update_LagSolver_flag = False

        # Exporting settings
        if self.out_dir: 
            buffer = {'snrs_args':  snrs_args,
                      'est_args':   est_args,
                      'model_args': model_args,
                      'grid_args':  grid_args}
            dict2txt(buffer, f'{self.out_dir}/settings.txt')
         # -------------------------------------------------------------------- #

    def __init_turbines__(self, snrs_args: dict, est_args: dict):
        """ Inits the wind turbines array.

        Parameters
        ----------
        snrs_args : dict
            Dictionary containing the parameters used for the turbines Sensors 
            initialization (refer to :class:`.Sensors`).
        est_args : dict
            Dictionary containing the parameters used for the turbines Estimators 
            initialization (refer to :class:`.Estimator`).
        """        
        self.wts = [ Turbine(self, i_wt, snrs_args, est_args) for i_wt in range(self.n_wts) ]
        # -------------------------------------------------------------------- #

    def __init_LagSolver__(self, model_args: dict, grid_args: dict):
        """ Inits the Lagrangian flow solver.

        Parameters
        ----------
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization (refer to :class:`.LagSolver`).
        grid_args : dict, optional
            Dictionary containing the parameters used for the grid initialization 
            (refer to :class:`.Grid`).
        """        
        self.lag_solver = LagSolver(self, model_args, grid_args)
        for wt in self.wts: wt.init_LagSolver()
        self.lag_solver._ini_data()
        # -------------------------------------------------------------------- #

    def __get_runid__(self) -> int:        
        """ Extract and increment the run id

        Returns
        -------
        int
            Current run id.
        """        
        run_id_path = f'{os.environ["ONWARDS_PATH"]}/onwards/.runid'

        try:                        fid = open(run_id_path)
        except FileNotFoundError:   run_id = 0
        else: 
            with fid:               run_id = int(fid.readline())
                
        with open(run_id_path, 'w') as fid:
            fid.write('{}'.format(run_id+1))

        return run_id
        # -------------------------------------------------------------------- #

    def __init_exports__(self, data_dir:str, out_dir:str, enable_logger:bool):
        """ Initializes the outputs.

        Parameters
        ----------
        data_dir : str
            path to the wind farm data
        out_dir  : str
            Export directory name where all figures and data are saved. 
        enable_logger : bool
            If True, logs are saved to the export directory.
        """        
        self.out_dir = f'{data_dir}/onwards_run_{self.__get_runid__()}' \
                                                 if out_dir is None else out_dir
        if self.out_dir and not os.path.exists(self.out_dir):
            lg.info(f'Data exported to {self.out_dir}.')
            os.makedirs(self.out_dir)

            if enable_logger: # adding a log file handler
                fh = logging.FileHandler(f'{self.out_dir}/onwards.log')
                fh.setLevel(lg.parent.level)
                fh_formatter = logging.Formatter('%(levelname)s : %(filename)s, line %(lineno)d in %(funcName)s :  %(message)s')
                fh.setFormatter(fh_formatter)
                lg.parent.addHandler(fh)

        else:
            lg.info('out_dir set to \'\': data exports disabled.')

        self.viz = []
        # -------------------------------------------------------------------- #

    def viz_add(self, viz_type: str, *args, **kwargs):
        """ Adds a :class:`Viz` object to the farm

        Parameters
        ----------
        viz_type : str
            Name of the Viz object

        Raises
        ------
        Exception
            If viz_type is not recognized.

        See also
        --------
        :class:`.Viz`,
        :meth:`Farm.viz_plot`

        """        
        _viz_type = viz_type.lower()
        if   _viz_type == 'particles':
            from .vizs.particles  import Viz_particles       as Viz
        elif _viz_type == 'velfield':
            from .vizs.velfield   import Viz_velfield        as Viz
        elif _viz_type == 'centerline':
            from .vizs.centerline import Viz_centerline      as Viz
        elif _viz_type == 'centerline_xloc':
            from .vizs.centerline import Viz_centerline_xloc as Viz
        elif _viz_type == 'rews':
            from .vizs.rews       import Viz_rews            as Viz
        elif _viz_type == 'estimators':
            from .vizs.estimators import Viz_estimators      as Viz
        else:
            raise Exception(f'viz_type {viz_type} not recognized.')

        self.viz.append(Viz(self, *args, **kwargs))
        # -------------------------------------------------------------------- #

    def __viz_update__(self):
        """
        Updates the :class:`.Viz` objects
        """        
        for viz in self.viz: viz.update()
        # -------------------------------------------------------------------- #

    def viz_plot(self):
        """
        Iteratively calls the plot methods of the :class:`.Viz` objects added to 
        the Farm.
        """
        for viz in self.viz: viz.plot()

    def iterate(self):
        """ Updates the Farm

        Successively updates the sensor states, m_wt, wind turbine states, s_wt,
        and flow model states, s_flow.

        If an Estimator/LagSolver was performed as part of the current timestep, 
        the corresponding update_states_flag/update_LagSolver_flag is set to True.

        Raises
        ------
        Exception
            If the time is not consistent across the different turbines.
        """        
        self.update_states_flag    = (self.it)%self.wts[0].n_substeps_est==0
        self.update_LagSolver_flag = (self.it)%self.lag_solver.set['n_substeps']==0

        # Updating the WT state
        if self.update_states_flag:
            lg.info(f'Updating turbine state at t={self.t} s.')
            for wt in self.wts: 
                wt.update_states()

        # Updating the Lagrangian Model
        if self.update_LagSolver_flag:
            lg.info(f'Updating Lagrangian Model at t={self.t} s.')
            for wt in self.wts: 
                wt.update_LagSolver()
            self.lag_solver.update()

        # Updating the WT sensors
        t = [wt.update_sensors() for wt in self.wts]
        if 1 < len(set(t)): 
            raise Exception('Time mismatch between the different wind turbines sensors')
        self.t = t[0]

        # Updating the plots
        self.__viz_update__()

        self.it += 1
        # -------------------------------------------------------------------- #

    def reset(self, model_args: dict, ini_states: dict[str, float]={}):
        """ Resets the wind turbines and flow states to the initial configuration 
        and updates the Lagrangian flow model parameters.

        Parameters
        ----------
        model_args : dict
            Dictionary containing the parameters of the Lagrangian flow model 
            that needs to be updated (refer to :class:`.LagSolver`)
        ini_states : dict[str, float], optional
            ``{'s_wt': v}`` maps the wind turbine state, ``s_wt``, to its initial 
            value, ``v``.
        """        
        model_args   = LoggingDict(model_args)

        # Resetting turbines sensors and states estimators
        for wt in self.wts: wt.reset(ini_states=ini_states)
        self.it = 0
        self.t  = self.wts[0].t

        # Resetting Lagrangian flow model
        self.lag_solver.reset(model_args)

        self.update_states_flag    = False
        self.update_LagSolver_flag = False

        # Resetting  Viz
        for v in self.viz: v.reset()
        # -------------------------------------------------------------------- #

    def __iter__(self):
        return self
        # -------------------------------------------------------------------- #

    def __len__(self):
        return min( len(wt.snrs) for wt in self.wts )
        # -------------------------------------------------------------------- #

    def __next__(self):
        self.iterate()
        return self.t
        # -------------------------------------------------------------------- #
  
    def __enter__(self):
        return self
        # -------------------------------------------------------------------- #

    def __exit__(self, *args, **kwargs):
        for wt in self.wts: wt.__exit__()
        for v in self.viz:  
            if self.out_dir: # export data
                v.export()
        self.lag_solver.free() # free c data
        # -------------------------------------------------------------------- #


