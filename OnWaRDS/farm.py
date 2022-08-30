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
                 est_args: dict, model_args: dict, grid_args: dict={}, 
                 wt_cherry_picking: List[int]=None, out_dir: str=None,
                 enable_plot: bool=True, enable_logger: bool=True):

        self.data_dir = data_dir
        self.__init_exports__(data_dir, out_dir, enable_plot, enable_logger)

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

    def __init_turbines__(self, snrs_args, est_args) -> None:
        self.wts = [ Turbine(self, i_wt, snrs_args, est_args) for i_wt in range(self.n_wts) ]
        # -------------------------------------------------------------------- #

    def __init_LagSolver__(self, model_args, grid_args={}):
        self.lag_solver = LagSolver(self, model_args, grid_args)
        for wt in self.wts: wt.init_LagSolver()
        self.lag_solver.ini_data()
        # -------------------------------------------------------------------- #

    def __get_runid__(self):
        run_id_path = f'{os.environ["ONWARDS_PATH"]}/.runid'

        try:                        fid = open(run_id_path)
        except FileNotFoundError:   run_id = 0
        else: 
            with fid:               run_id = int(fid.readline())
                
        with open(run_id_path, 'w') as fid:
            fid.write('{}'.format(run_id+1))

        return run_id
        # -------------------------------------------------------------------- #

    def __init_exports__(self, data_dir:str, out_dir:str, enable_plot:bool, enable_logger:bool):
        self.out_dir = f'{data_dir}/OnWaRDS_run_{self.__get_runid__()}' \
                                                 if out_dir is None else out_dir
        if self.out_dir and not os.path.exists(self.out_dir):
            lg.info(f'Data exported to {self.out_dir}.')
            os.makedirs(self.out_dir)

            if enable_logger: # adding a log file handler
                fh = logging.FileHandler(f'{self.out_dir}/OnWaRDS.log')
                fh.setLevel(lg.parent.level)
                fh_formatter = logging.Formatter('%(levelname)s : %(filename)s, line %(lineno)d in %(funcName)s :  %(message)s')
                fh.setFormatter(fh_formatter)
                lg.parent.addHandler(fh)

        else:
            lg.info('out_dir set to \'\': data exports disabled.')
        self.enable_plot = enable_plot

        self.viz = []
        # -------------------------------------------------------------------- #

    def viz_add(self, type: str, *args, **kwargs):
        _type = type.lower()
        if   _type == 'part':
            from .disp.part_plot       import Part_plot               as Viz
        elif _type == 'velfield':
            from .disp.velField_plot   import VelField_plot           as Viz
        elif _type == 'wakecenterline':
            from .disp.centerline_plot import WakeCenterline          as Viz
        elif _type == 'wakecenterline_xloc':
            from .disp.centerline_plot import WakeCenterlineXloc_plot as Viz
        elif _type == 'rews':
            from .disp.rews_plot       import REWS_plot               as Viz
        elif _type == 'estimator':
            from .disp.estimator_plot  import Estimator_plot          as Viz
        else:
            raise Exception(f'Viz type {type} not recognized.')

        self.viz.append(Viz(self, *args, **kwargs))
        # -------------------------------------------------------------------- #

    def __viz_update__(self):
        for viz in self.viz: viz.update()
        # -------------------------------------------------------------------- #

    def iterate(self):
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
            if self.out_dir:
                v.export()
            if self.enable_plot:
                v.plot()
        self.lag_solver.free()
        # -------------------------------------------------------------------- #

    def reset(self, model_args: dict, ini_states: dict[str, float]={}):
        model_args   = LoggingDict(model_args)

        # Resetting turbines sensors and states estimators
        for wt in self.wts: wt.reset(ini_states)
        self.it = 0
        self.t  = self.wts[0].t

        # Resetting Lagrangian flow model
        self.lag_solver.reset(model_args)

        self.update_states_flag    = False
        self.update_LagSolver_flag = False

        # Resetting  Viz
        for v in self.viz: v.reset()
        # -------------------------------------------------------------------- #





