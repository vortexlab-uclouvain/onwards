from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import time
import numpy as np

from .grid import Grid
from .libc import pyCommunicator as py_comm

if TYPE_CHECKING:
    from typing  import List
    from farm    import Farm
    from sensor  import Sensors
    from turbine import Turbine

def IN2VEC(*args):
    is_vec = all(isinstance(i, py_comm.Vec) for i in args)
    if is_vec: return args
    else:      return (py_comm.Vec(i) for i in args)

FILT2INT = {'rotor': 0, 'flow': 1}

class LagSolver():
    farm: Farm
    grid: Grid
    fms:  List(py_comm.c_FlowSolver_p)
    wms:  List(py_comm.c_WakeModel_p)

    def __init__(self, farm, model_args, grid_args = {}):
        self.dwmLocAll = []
        self.farm = farm 

        # raise Exception

        self.set = model_args
        
        self.set['dt'] = float(self.farm.dt * self.set.setdefault( 'n_substeps', 1 ))

        self.set.setdefault( 'n_fm', 80 )
        self.set.setdefault( 'n_shed_fm', 2 )
        self.set.setdefault( 'c0', 1.0 )

        self.set.setdefault( 'n_wm', 80 )
        self.set.setdefault( 'n_shed_wm', 2 )
        self.set.setdefault( 'cw', 1.0 )
    
        self.set.setdefault( 'sigma_xi_f', 0.25  )
        self.set.setdefault( 'sigma_r_f',  0.5   )
        self.set.setdefault( 'sigma_t_f',  0.125 )  

        self.set.setdefault( 'sigma_xi_r', 0.25  )
        self.set.setdefault( 'sigma_r_r',  0.5   )
        self.set.setdefault( 'sigma_t_r',  0.125 )

        self.set.setdefault( 'tau_r',  32 )

        self.set.setdefault( 'sd_type',  0  )
        self.set.setdefault( 'ak',    0.25  )
        self.set.setdefault( 'bk',    0.5   )
        self.set.setdefault( 'ceps',  0.25 )

        self._set_c_ = py_comm.c_Set(self.set)  
        self.p       = py_comm.cLib.init_LagSolver(self.farm.n_wts, self._set_c_)
        
        grid_args['enable'] = grid_args.get('enable', True)
        self.grid = False if not grid_args['enable'] else Grid(self, grid_args)
        # -------------------------------------------------------------------- #

    def update(self):
        start_F3MOdel = time.time()
        py_comm.cLib.update_LagSolver(self.p)
        self._comm_buffer_update_flag = False
        lg.info("lagSolver self time: %1.3e", time.time()-start_F3MOdel)
        if not np.abs(self.farm.t + self.set['dt'] - self.p.contents.t)<1e-6:
            raise Exception(   'Time is not consistent across all submodules:\n'
                             + '   velEstimator root : {:2.2f} [s]\n'.format(self.farm.t + self.set['dt'])
                             + '   lagSolver.c       : {:2.2f} [s]'  .format(self.get_time()) )
        # -------------------------------------------------------------------- #
    
    def ini_data(self):
        self.data_p = { 'F' : [ py_comm.cLib.get_FlowModel(self.p, wt.c_wt) for wt in self.farm.wts ],
                        'W' : [ py_comm.cLib.get_WakeModel(self.p, wt.c_wt) for wt in self.farm.wts ] }
        # -------------------------------------------------------------------- #

    def get(self, model, field, comp=None, i_wt=None):

        if model not in ['W', 'F']:
            raise ValueError(f'Inconsistent model type ({model}).')

        is_vec = any(s in field for s in ['u', 'x'])
        if comp is None and is_vec :
            raise ValueError(f'Invalid component index ({field}).')
        if comp is not None and not is_vec :
            raise ValueError(f'comp = {comp} not available for scalar data.')
        

        i_wt_list = [i_wt] if i_wt is not None else range(self.farm.n_wts)
        if self.farm.n_wts<i_wt_list[0]:
            raise ValueError(f'Invalid turbine index ({i_wt}<{self.farm.n_wts}).')

        i0 = self.data_p[model][0].contents.i0
        n  = self.data_p[model][0].contents.n

        buffer = np.empty(n*len(i_wt_list))

        for i, i_wt in enumerate(i_wt_list):
            data = self.data_p[model][i_wt].contents

            if comp is not None:
                buffer[i*n:(i+1)*n] = ( [getattr(data, field)[i][comp] for i in range(i0,n)]
                                      + [getattr(data, field)[i][comp] for i in range(0,i0)] )
            else:
                buffer[i*n:(i+1)*n] = ( getattr(data, field)[i0:n]
                                      + getattr(data, field)[0:i0] )
        return buffer
        # -------------------------------------------------------------------- #

    def get_part_iwt(self, model):
        if model not in ['W', 'F']:
            raise ValueError(f'Inconsistent model type ({model}).')
        
        return np.linspace( 0, 
                            self.farm.n_wts-1E-10, 
                            self.farm.n_wts*self.set[f'n_{model.lower()}m'], 
                            dtype=int )
        # -------------------------------------------------------------------- #

    def get_time(self):
        return self.p.contents.t
        # -------------------------------------------------------------------- #

    def free(self):
        for wt_i in self.farm.wts:
            wt_i.c_wt.free()
        py_comm.cLib.free_LagSolver(self.p)
        # -------------------------------------------------------------------- #
    
    def interp_FlowModel(self, xv, yv, filt='flow', buffer: py_comm.Vec=None, i_wt_exclude=-1):
        x, y   = IN2VEC(xv,yv)
        u_vec = py_comm.Vec((2,*x.x.shape)) if buffer is None else buffer
        if u_vec.x.shape != (2,*x.x.shape):
            raise ValueError("Inconsistent matrix shape for `u_vec`.")

        if filt not in FILT2INT:
            raise Exception('Filter type not recognized (should be `flow` or `rotor`.')

        py_comm.cLib.interp_vec_FlowModel(self.p, x.p, y.p, x.s, u_vec.p, FILT2INT[filt], i_wt_exclude)
        return u_vec.x
        # -------------------------------------------------------------------- #

    def interp_WakeModel(self, xv, yv, buffer: py_comm.Vec=None):
        x, y   = IN2VEC(xv,yv)
        du_vec = py_comm.Vec((2,*x.x.shape)) if buffer is None else buffer
        if du_vec.x.shape != (2,*x.x.shape):
            raise ValueError("Inconsistent matrix shape for `du_vec`.")

        py_comm.cLib.interp_vec_WakeModel(self.p, x.p, y.p, x.s, du_vec.p)
        return du_vec.x
        # -------------------------------------------------------------------- #

    def rews_compute(self, x_rotor, r_rotor):
        x_cast = np.array([x_rotor[0],x_rotor[1],x_rotor[2]])
        x = py_comm.Vec(x_cast)
        return py_comm.cLib.rews_compute(self.p, x.p, r_rotor)
        # -------------------------------------------------------------------- #
