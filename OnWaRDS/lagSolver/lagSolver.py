from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import time
import numpy as np

from .grid import Grid
from .libc import pyCommunicator as py_comm

if TYPE_CHECKING:
    from typing   import List
    from farm     import Farm
    from sensors  import Sensors
    from turbine  import Turbine

def _IN2VEC(*args):
    is_vec = all(isinstance(i, py_comm.Vec) for i in args)
    if is_vec: return args
    else:      return (py_comm.Vec(i) for i in args)

_FILT2INT = {'rotor': 0, 'flow': 1}

class LagSolver():
    farm: Farm
    grid: Grid
    fms:  List(py_comm.c_FlowSolver_p)
    wms:  List(py_comm.c_WakeModel_p)

    def __init__(self, farm: Farm, model_args: dict, grid_args: dict = {}):
        """ Inits a LagSolver object.

        LagSolver object interfaces the c Lagrangian flow model using ctypes.

        Parameters
        ----------
        farm : Farm
            Parent farm object.
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization 

            Available fields:

            :n_substeps: *(int  , optional)* - 
                Number of Farm timesteps between two successive Lagrangian flow 
                model updates, by default 100.
            :n_fm:       *(int  , optional)* - 
                Maximum number of ambient flow particles used for each turbine, 
                by default 100  
            :n_shed_fm:  *(int  , optional)* - 
                Number of Lagrangian flow model timesteps between the shedding 
                of two successive ambient flow particles by default 1    
            :c0:         *(float, optional)* - 
                Convective ambient particles tuning constant , by default 0.73 
            :n_wm:       *(int  , optional)* - 
                Maximum number of wake particles used for each turbine, by 
                default 80   
            :n_shed_wm:  *(int  , optional)* - 
                Number of Lagrangian flow model timesteps between the shedding 
                of two successive wake particles, by default 2    
            :cw:         *(float, optional)* - 
                Convective wake particles tuning constant, by default 0.54 
            :sigma_xi_f: *(float, optional)* - 
                Streamwise ambient filtering constant, by default 4.0 
            :sigma_r_f:  *(float, optional)* - 
                Spanwise/transverse ambient filtering constant, by default 1.0 
            :sigma_t_f:  *(float, optional)* - 
                Streamwise ambient filtering constant, by default 2.0 
            :sigma_xi_r: *(float, optional)* - 
                Streamwise ambient filtering constant, by default 0.5 
            :sigma_r_r:  *(float, optional)* - 
                Spanwise/transverse ambient filtering constant, by default 1.0 
            :sigma_t_r:  *(float, optional)* - 
                Streamwise ambient filtering constant, by default 2.0 
            :tau_r:      *(float, optional)* - 
                Rotor time filtering constant in [s], by default 16 
            :sd_type:    *(float, optional)* -
                Type of speed deficit used ( 0 : Gaussian speed deficit [1]_ 
                / only option currently available), by default 0     
            :ak:         *(float, optional)* -
                Wake expansion tuning constant (TI scaling), by default 0.021 
            :bk:         *(float, optional)* -
                Wake expansion tuning constant (TI scaling), by default 0.039 
            :ceps:       *(float, optional)* -
                Initial wake width tuning constant, by default 0.2
        grid_args : dict, optional
            Dictionary containing the parameters used for the grid's initialization 
            (see :class:`Grid<.lagSolver.grid.Grid>`).  

        References
        ----------
            .. [1]  M. Bastankhah and F. Porte-Agel. A new analytical model for wind-turbine wakes. Renewable Energy, 70:116–123, 2014.
        """
        self.farm = farm 

        self.set = model_args
        
        # Retrieving the default parameters

        self.set['dt'] = float(self.farm.dt * self.set.setdefault( 'n_substeps', 1 ))

        self.set.setdefault( 'n_fm',      100  )
        self.set.setdefault( 'n_shed_fm', 1    )
        self.set.setdefault( 'c0',        0.73 )

        self.set.setdefault( 'n_wm',      80   )
        self.set.setdefault( 'n_shed_wm', 2    )
        self.set.setdefault( 'cw',        0.54 )
    
        self.set.setdefault( 'sigma_xi_f', 4.0 )
        self.set.setdefault( 'sigma_r_f',  1.0 )
        self.set.setdefault( 'sigma_t_f',  2.0 )  

        self.set.setdefault( 'sigma_xi_r', 0.5 )
        self.set.setdefault( 'sigma_r_r',  1.0 )
        self.set.setdefault( 'sigma_t_r',  2.0 )

        self.set.setdefault( 'tau_r',  16 )

        self.set.setdefault( 'sd_type', 0     )
        self.set.setdefault( 'ak',      0.021 )
        self.set.setdefault( 'bk',      0.039 )
        self.set.setdefault( 'ceps',    0.2   )

        self._set_c_ = py_comm.c_Set(self.set)  
        self.p       = py_comm.init_LagSolver(self.farm.n_wts, self._set_c_)
        
        grid_args['enable'] = grid_args.get('enable', True)
        self.grid = False if not grid_args['enable'] else Grid(self, grid_args)
        # -------------------------------------------------------------------- #

    def reset(self, model_args_new):
        """ Resets the flow states to the initial configuration and updates the 
            Lagrangian flow model parameters.

        Parameters
        ----------
        model_args : dict
            Dictionary containing the parameters of the Lagrangian flow model 
            that needs to be updated.

        Raises
        ------
        ValueError
            If one of the following model parameters is updated: n_fm, n_wm or sd_type.
        """  

        self._set_c_.update(model_args_new)
        py_comm.reset_LagSolver(self.p)
        # -------------------------------------------------------------------- #

    def update(self):
        """ Updates the Lagrangian flow model

        Raises
        ------
        Exception
            If a timing mismatch is detected between the parent Farm object and 
            the Lagrangian flow model.
        """

        start_F3MOdel = time.time()
        py_comm.update_LagSolver(self.p)
        self._comm_buffer_update_flag = False
        lg.info("lagSolver self time: %1.3e", time.time()-start_F3MOdel)
        if not np.abs(self.farm.t + self.set['dt'] - self.p.contents.t)<1e-6:
            raise Exception(   'Time is not consistent across all submodules:\n'
                             + '   velEstimator root : {:2.2f} [s]\n'.format(self.farm.t + self.set['dt'])
                             + '   lagSolver.c       : {:2.2f} [s]'  .format(self.get_time()) )
        if self.grid: self.grid.update()
        # -------------------------------------------------------------------- #
    
    def _ini_data(self):
        """
        Retrieves the c data pointers.
        """
        self.data_p = { 'F' : [ py_comm.get_FlowModel(self.p, wt.c_wt) for wt in self.farm.wts ],
                        'W' : [ py_comm.get_WakeModel(self.p, wt.c_wt) for wt in self.farm.wts ] }
        # -------------------------------------------------------------------- #

    def free(self):
        """
        Free the underlying c model.
        """
        for wt_i in self.farm.wts:
            wt_i.c_wt.free()
        py_comm.free_LagSolver(self.p)
        # -------------------------------------------------------------------- #

    def get(self, model: str, field: str, comp: int = None, i_wt: int = None) -> np.array:
        """_summary_

        Parameters
        ----------
        model : str
            Sub-model from which data should be extracted 'W' for wake or 'F' 
            for ambient flow field.
        field : str
            Name of the field to be extracted.
        comp : int, optional
            Flow component to be extracted (0: x or 1: z), by default None.
        i_wt : int, optional
            Index of the turbine data should be extracted from if None data is 
            extracted from all turbines, by default None.

        Returns
        -------
        np.array
            Array containing the field requested  

        Raises
        ------
        ValueError
            If model is not 'W' or 'F'.
        ValueError
            If no field component, comp, is provided for a vector field. 
        ValueError
            If a field component, comp, is specified for a scalar field. 
        ValueError
            If the wind turbine index request, i_wt, is not valid.
        """

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

    def get_part_iwt(self, model: str) -> np.array:
        """ Computes the mapping between the :obj:`LagSolver.get<.lagSolver.lagSolver.LagSolver.get>`
        outputs for i_wt=None and the turbines.

        Parameters
        ----------
        model : str
            Sub-model from which data should be extracted 'W' for wake or 'F' 
            for ambient flow field.

        Returns
        -------
        np.array
           Mapping between the array index and the turbine index.

        Raises
        ------       
        ValueError
            If model is not 'W' or 'F'.

        See also
        --------
        :obj:`get<.lagSolver.lagSolver.LagSolver.get>`

        """        

        if model not in ['W', 'F']:
            raise ValueError(f'Inconsistent model type ({model}).')
        
        return np.linspace( 0, 
                            self.farm.n_wts-1E-10, 
                            self.farm.n_wts*self.set[f'n_{model.lower()}m'], 
                            dtype=int )
        # -------------------------------------------------------------------- #

    def get_time(self):
        """
        Returns the current Lagrangian model time
        """
        return self.p.contents.t
        # -------------------------------------------------------------------- #
    
    def interp_FlowModel(self, xv:np.array, zv:np.array, filt:str='flow', 
                            buffer: py_comm.Vec=None, i_wt_exclude:int=-1) -> np.array:
        """ Interpolates the ambient flow field at [ xv, zv ]

        Parameters
        ----------
        xv : np.array
            array containing the x locations where the field should be evaluated.
        zv : np.array
            array containing the z locations where the field should be evaluated.
        filt : str, optional
            'flow' or 'rotor' depending on the width of the filter used for the 
            ambient velocity field computation, by default 'flow'.
        buffer : py_comm.Vec, optional
            Vec object allocating the output memory location (allows not to 
            reallocate the wind farm global grid at every time step), buffer 
            shape should be consistent with xv and zv if None a new output 
            vector is allocated, by default None.
        i_wt_exclude : int, optional
            Ignores the selected wind turbine for the ambient velocity 
            computations, by default -1.

        Returns
        -------
        np.array
            Estimated ambient flow field [u, w]

        Raises
        ------
        ValueError
            If the buffer shape provided is not consistent with the shape of x, z.
        Exception
            If filt is not rotor or flow.
        """
        x, y   = _IN2VEC(xv,zv)
        u_vec = py_comm.Vec((2,*x.x.shape)) if buffer is None else buffer
        if u_vec.x.shape != (2,*x.x.shape):
            raise ValueError("Inconsistent matrix shape for `u_vec`.")

        if filt not in _FILT2INT:
            raise Exception('Filter type not recognized (should be `flow` or `rotor`.')

        py_comm.interp_vec_FlowModel(self.p, x.p, y.p, x.s, u_vec.p, _FILT2INT[filt], i_wt_exclude)
        return u_vec.x
        # -------------------------------------------------------------------- #

    def interp_WakeModel(self, xv:np.array, zv:np.array, buffer: py_comm.Vec=None):
        """ Interpolates the wake flow field at [ xv, zv ]

        Parameters
        ----------
        xv : np.array
            array containing the x locations where the field should be evaluated.
        zv : np.array
            array containing the z locations where the field should be evaluated.
        buffer : py_comm.Vec, optional
            Vec object allocating the output memory location (allows not to 
            reallocate the wind farm global grid at every time step), buffer 
            shape should be consistent with xv and zv if None a new output 
            vector is allocated, by default None.

        Returns
        -------
        np.array
            Estimated ambient flow field [u, w]

        Raises
        ------
        ValueError
            If the buffer shape provided is not consistent with the shape of x, z.
        Exception
            If filt is not rotor or flow.
        """
        x, y   = _IN2VEC(xv,zv)
        du_vec = py_comm.Vec((2,*x.x.shape)) if buffer is None else buffer
        if du_vec.x.shape != (2,*x.x.shape):
            raise ValueError("Inconsistent matrix shape for `du_vec`.")

        py_comm.interp_vec_WakeModel(self.p, x.p, y.p, x.s, du_vec.p)
        return du_vec.x
        # -------------------------------------------------------------------- #

    def rews_compute(self, x_rotor:List[float], r_rotor:float) -> float:
        """ Computes the Rotor Effective Wind Speed at x_rotor over a rotor of 
        diameter, r_rotor and oriented along x.

        Parameters
        ----------
        x_rotor : List[float]
            Fictive rotor center location [x,y,z] in [m].
        r_rotor : float
             Fictive rotor diameter in [m].

        Returns
        -------
        float
            The Rotor Effective Wind Speed of diameter r_rotor and located at x_rotor.
        """
        x_cast = np.array([x_rotor[0],x_rotor[1],x_rotor[2]])
        x = py_comm.Vec(x_cast)
        return py_comm.rews_compute(self.p, x.p, r_rotor)
        # -------------------------------------------------------------------- #
