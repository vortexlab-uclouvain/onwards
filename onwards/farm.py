from __future__ import annotations
from typing import TYPE_CHECKING

import logging

lg = logging.getLogger(__name__)

import os
import numpy as np

from .turbine   import Turbine
from .airfoil   import Airfoil
from .lagSolver import LagSolver 
from .disp.viz  import Viz
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
        """Inits Farm.

        The Farm object interfaces together the different modules of the OnWaRDS
        toolbox (eg: state estimation, data i/o, plotting).

        A Farm consists of a series of :class:`Turbines<.Turbine>` (self.wts). 
        Each wind turbine is associated to a :class:`Sensors<.Sensors>` object 
        that gathers its measurements, m_wt, at a prescribed time, t. These 
        measurements are then translated into the estimated turbine state, s_wt, 
        using its built-in :class:`Estimators<.estimators.estimator.Estimator>`.

        The wind turbine states, s_wt, are eventually fed to lag_solver, the 
        :class:`Lagrangian flow model<.lagSolver.LagSolver>` (written in c and 
        interfaced with main python code using ctypes) that uses them to estimate 
        the flow state, s_flow.     

        Once initialized, a Farm object can be temporally iterated over using the 
        built-in for loop.
        Memory cleaning is automatically handled using the with statement.
        
        Plotting is handled by ``self.viz`` and custom user plot can added using
        :obj:`Farm.viz_add<.Farm.viz_add>`.
        
        Parameters
        ----------
        data_dir : str
            Path to the wind farm data.
        af_name : str
            Name of the airfoil used.
        snrs_args : dict
            Dictionary containing the parameters used for the turbines Sensors 
            initialization (see :class:`Sensors<.Sensors>`).
        est_args : dict
            Dictionary containing the parameters used for the turbines Estimators 
            initialization (see :class:`Estimator<.estimators.estimator.Estimator>`).
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization (see :class:`LagSolver<.lagSolver.lagSolver.LagSolver>`).
        grid_args : dict, optional
            Dictionary containing the parameters used for the grid's initialization 
            (see :class:`Grid<.lagSolver.grid.Grid>`).
        wt_cherry_picking : List[int], optional
            List where each element corresponds to a wind turbine index: allows
            to pick only the desired wind turbine. If None, all wind turbines are 
            modeled, by default None.
        out_dir : str, optional
            Export directory name where figures and data are saved. 
            If '': all exports are disabled, if None: default export directory 
            name (onwards_run_id), by default None.
        enable_logger : bool, optional
            If true, logs are saved to the export directory, by default True.

        Raises
        ------
        TypeError
            if wt_cherry_picking is not a list of turbines indices.

        Example
        -------
        Typical OnWaRDS initialization:

            >>> from onwards import Farm
            >>> with Farm() as f:
            >>>    v.viz_add('MyViz') 
            >>>    for t in f:
            >>>        print(f'Current time is {t}.')
            >>>    f.plot()
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
            initialization (see :class:`Sensors<.Sensors>`).
        est_args : dict
            Dictionary containing the parameters used for the turbines Estimators 
            initialization (see :class:`Estimator<.estimators.estimator.Estimator>`).
        """        
        self.wts = [ Turbine(self, i_wt, snrs_args, est_args) for i_wt in range(self.n_wts) ]
        # -------------------------------------------------------------------- #

    def __init_LagSolver__(self, model_args: dict, grid_args: dict):
        """ Inits the Lagrangian flow solver.

        Parameters
        ----------
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization (see :class:`LagSolver<.lagSolver.lagSolver.LagSolver>`).
        grid_args : dict, optional
            Dictionary containing the parameters used for the grid initialization 
            (see :class:`Grid<.lagSolver.grid.Grid>`).
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
            If true, logs are saved to the export directory.
        """        
        self.out_dir = f'{data_dir}/onwards_run_{self.__get_runid__()}' \
                                                 if out_dir is None else out_dir
        if self.out_dir and not os.path.exists(self.out_dir):
            lg.info(f'Data exported to {self.out_dir}.')
            os.makedirs(self.out_dir)

            if enable_logger: # adding a log file handler
                fh = logging.FileHandler(f'{self.out_dir}/.log')
                fh.setLevel(lg.parent.level)
                fh_formatter = logging.Formatter('%(levelname)s : %(filename)s, line %(lineno)d in %(funcName)s :  %(message)s')
                fh.setFormatter(fh_formatter)
                lg.parent.addHandler(fh)

        else:
            lg.info('out_dir set to \'\': data exports disabled.')

        self.viz = []
        # -------------------------------------------------------------------- #

    def viz_add(self, viz_type: str, *args, **kwargs):
        """ Adds a Viz object to the farm

        :class:`Viz<.disp.viz.Viz>` objects store, update  and plot the wind farm data.

        Parameters
        ----------
        viz_type : str
            Name of the Viz object

        Raises
        ------
        Exception
            if viz_type is not recognized.

        See also
        --------
        :class:`Viz<.disp.viz.Viz>`,
        :obj:`Farm.viz_plot<Farm.viz_plot>`

        """        
        _viz_type = viz_type.lower()
        if   _viz_type == 'part':
            from .disp.part_plot       import Part_plot               as Viz
        elif _viz_type == 'velfield':
            from .disp.velField_plot   import VelField_plot           as Viz
        elif _viz_type == 'wakecenterline':
            from .disp.centerline_plot import WakeCenterline          as Viz
        elif _viz_type == 'wakecenterline_xloc':
            from .disp.centerline_plot import WakeCenterlineXloc_plot as Viz
        elif _viz_type == 'rews':
            from .disp.rews_plot       import REWS_plot               as Viz
        elif _viz_type == 'estimator':
            from .disp.estimator_plot  import Estimator_plot          as Viz
        else:
            raise Exception(f'viz_type {viz_type} not recognized.')

        self.viz.append(Viz(self, *args, **kwargs))
        # -------------------------------------------------------------------- #

    def __viz_update__(self):
        """
        Updates the Viz objects
        """        
        for viz in self.viz: viz.update()
        # -------------------------------------------------------------------- #

    def viz_plot(self):
        """
        Iteratively calls the plot methods of the Viz objects added to the farm.
        """
        for viz in self.viz: viz.plot()

    def iterate(self):
        """ Updates the wind farm

        Successively updates the sensor states, wind turbine states and flow model
        states.

        Wind turbines states are updated every n_substeps_Estimator
        (see :class:`Estimator<.estimators.estimator.Estimator>`).
        Lagrangian flow model states are updated every n_substeps_LagSolver
        (see :class:`LagSolver<.lagSolver.lagSolver.LagSolver>`).

        If Estimator/LagSolver was updated at the current timestep, the 
        update_states_flag/update_LagSolver_flag is set to True.

        Raises
        ------
        Exception
            If the time is not consistent between the different wind turbines.
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
            that needs to be updated (see :class:`LagSolver<.lagSolver.lagSolver.LagSolver>`)
        ini_states : dict[str, float], optional
            {'s_wt_i': v} maps the wind turbine state, s_wt_i, to its initial 
            value, v.
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


