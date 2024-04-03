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

import time
import numpy as np

from .grid import Grid
from .libc import pyCommunicator as py_comm

if TYPE_CHECKING:
    from typing   import List, Union
    from farm     import Farm
    from sensors  import Sensors
    from turbine  import Turbine

def _IN2VEC(*args):
    is_vec = all(isinstance(i, py_comm.Vec) for i in args)
    if is_vec: return args
    else:      return (py_comm.Vec(i) for i in args)

_FILT2INT = {'rotor': 0, 'flow': 1}

_RST_MAP = { 'F' : [('n'     , 0),
                    ('it'    , 0),
                    ('i0'    , 0),
                    ('t_p'   , 1),
                    ('xi_p'  , 1),
                    ('x_p'   , 2),
                    ('u_p'   , 2), ],
             'W' : [('n'     , 0),
                    ('it'    , 0),
                    ('i0'    , 0),
                    ('t_p'   , 1),
                    ('xi_p'  , 1),
                    ('ct_p'  , 1),
                    ('ti_p'  , 1),
                    ('x_p'   , 2),
                    ('uinc_p', 2), ] }

class LagSolver():
    farm: Farm
    grid: Grid
    fms:  List(py_comm.c_FlowSolver_p)
    wms:  List(py_comm.c_WakeModel_p)
    data_p: dict(str, Union[py_comm.c_FlowModel_p, py_comm.c_WakeModel_p])

    def __init__(self, farm: Farm, model_args: dict, grid_args: dict = {}):
        r""" Inits a LagSolver object.

        LagSolver object allows to compute the estimated flow state from the local, estimated
        Turbine states.

        .. math::
            \mathbf{\hat{s}}_{wt} \rightarrow \mathbf{\hat{s}}_{flow} 

        It interfaces the c Lagrangian flow model using ctypes.

        Parameters
        ----------
        farm : Farm
            Parent :class:`.Farm` object.
        model_args : dict
            Dictionary containing the parameters used for the Lagrangian flow 
            model's initialization. 

            Available fields:

            :n_substeps: *(int  , optional)* - 
                Number of Farm timesteps between two successive Lagrangian flow 
                model updates, by default 1.
            :n_fm:       *(int  , optional)* - 
                Maximum number of ambient flow particles used for each turbine, 
                by default 60  
            :n_shed_fm:  *(int  , optional)* - 
                Number of Lagrangian flow model timesteps between the shedding 
                of two successive ambient flow particles by default 2    
            :c0:         *(float, optional)* - 
                Convective ambient particles tuning constant , by default 0.30 
            :n_wm:       *(int  , optional)* - 
                Maximum number of wake particles used for each turbine, by 
                default 60.   
            :n_shed_wm:  *(int  , optional)* - 
                Number of Lagrangian flow model timesteps between the shedding 
                of two successive wake particles, by default 2.    
            :cw_xi:      *(float, optional)* - 
                Convective wake particles tuning constant, by default 0.50 
            :cw_r:       *(float, optional)* - 
                Convective wake particles tuning constant, by default 0.30 
            :sigma_xi_f: *(float, optional)* - 
                Streamwise ambient filtering constant, by default 10.0 
            :sigma_r_f:  *(float, optional)* - 
                Spanwise/transverse ambient filtering constant, by default 5.0 
            :sigma_t_f:  *(float, optional)* - 
                Streamwise ambient filtering constant, by default 2.0 
            :sigma_xi_r: *(float, optional)* - 
                Streamwise ambient filtering constant, by default 1.0 
            :sigma_r_r:  *(float, optional)* - 
                Spanwise/transverse ambient filtering constant, by default 0.5 
            :sigma_t_r:  *(float, optional)* - 
                Streamwise ambient filtering constant, by default 2.0 
            :tau_r:      *(float, optional)* - 
                Rotor time filtering constant in [s], by default 32 
            :sd_type:    *(float, optional)* -
                Type of speed deficit used ( 0 : Gaussian speed deficit [1]_ 
                / only option currently available), by default 0     
            :ak:         *(float, optional)* -
                Wake expansion tuning constant (TI scaling), by default 0.018 
            :bk:         *(float, optional)* -
                Wake expansion tuning constant (TI scaling), by default 0.060 
            :ceps:       *(float, optional)* -
                Initial wake width tuning constant, by default 0.2

        grid_args : dict, optional
            Dictionary containing the parameters used for the Grid's initialization 
            (refer to :class:`.Grid`).  

        See also
        --------
            :class:`.Grid`

        References
        ----------
            .. [1]  M. Bastankhah and F. Porte-Agel. A new analytical model for wind-turbine wakes. Renewable Energy, 70:116–123, 2014.
        """
        self.farm = farm 

        self.set = model_args
        
        # Retrieving the default parameters

        self.set['dt'] = float(self.farm.dt * self.set.setdefault( 'n_substeps', 1 ))

        self.set.setdefault( 'n_fm',      60  )
        self.set.setdefault( 'n_shed_fm', 2   )
        self.set.setdefault( 'c0',        0.3 )

        self.set.setdefault( 'n_wm',      60  )
        self.set.setdefault( 'n_shed_wm', 2   )
        self.set.setdefault( 'cw_xi',     0.5 )
        self.set.setdefault( 'cw_r',      0.3 )
    
        self.set.setdefault( 'sigma_xi_f', 10 )
        self.set.setdefault( 'sigma_r_f',  5  )
        self.set.setdefault( 'sigma_t_f',  2  )  

        self.set.setdefault( 'sigma_xi_r', 1 )
        self.set.setdefault( 'sigma_r_r',  0.5  )
        self.set.setdefault( 'sigma_t_r',  2 )

        self.set.setdefault( 'tau_r',  32 )

        self.set.setdefault( 'sd_type', 0     )
        self.set.setdefault( 'ak',      0.018 )
        self.set.setdefault( 'bk',      0.06 )
        self.set.setdefault( 'ceps',    0.2   )

        self._set_c_ = py_comm.c_Set(self.set)  
        self.p       = py_comm.init_LagSolver(self.farm.n_wts, self._set_c_)
        
        grid_args['enable'] = grid_args.get('enable', True)
        self.grid = False if not grid_args['enable'] else Grid(self, grid_args)
        # -------------------------------------------------------------------- #

    def reset(self, model_args_new:dict, rst:dict = {}):
        """ Resets the flow states to the initial configuration and updates the 
        Lagrangian flow model parameters.

        Parameters
        ----------
        model_args : dict
            Dictionary containing the parameters of the Lagrangian flow model 
            to be updated.

        Raises
        ------
        ValueError
            If one of the following model parameters is updated: ``n_fm``, ``n_wm`` 
            or ``sd_type``.
        """  

        self._set_c_.update(model_args_new, self.set)
        py_comm.reset_LagSolver(self.p)

        self.init_from_restart(rst)
        # -------------------------------------------------------------------- #

    def update(self):
        """ Updates the Lagrangian flow model

        Raises
        ------
        Exception
            If a time mismatch is detected between the parent Farm object and 
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

    def _get_particles(self, model_p: Union[py_comm.c_FlowModel_p, py_comm.c_WakeModel_p], 
                             field: str, i_wt_list: List[int], 
                             comp: int = None, i0_offset: bool = True, split: bool = False) -> np.array:
        
        is_vec = any(s == field for s in ['u_p', 'uf_p', 'uinc_p', 'x_p'])
        if comp is None:
            if is_vec:
                raise ValueError(f'Invalid component index ({field}).')
        else:
            if not is_vec :
                raise ValueError(f'comp = {comp} not available for scalar data.')
            if not comp < 2:
                raise ValueError(f'Invalid component index ({field}, comp = {comp} < 2).')

        buffer = np.empty(sum(model_p[i_wt].contents.n for i_wt in i_wt_list))

        for i, i_wt in enumerate(i_wt_list):
            data = model_p[i_wt].contents

            i0 = data.i0 * i0_offset
            n  = data.n
    
            if comp is not None:
                buffer[i*n:(i+1)*n] = ( [getattr(data, field)[i][comp] for i in range(i0,n)]
                                      + [getattr(data, field)[i][comp] for i in range(0,i0)] )
            else:
                buffer[i*n:(i+1)*n] = ( getattr(data, field)[i0:n]
                                      + getattr(data, field)[0:i0] )
        if split:
            return list(buffer.reshape(len(i_wt_list), -1))
        else:
            return buffer
        # -------------------------------------------------------------------- #

    def _get_params(self, model_p: Union[py_comm.c_FlowModel_p, py_comm.c_WakeModel_p], 
                             field: str, i_wt_list: List[int])  -> List:
        buffer = [None]*len(i_wt_list)
        for i, i_wt in enumerate(i_wt_list):
            data = getattr(model_p[i_wt].contents, field)
            if not isinstance(data, (int, float)):
                raise AttributeError('Field {field} not supported.')
            buffer[i] = data
            
        return buffer
        # -------------------------------------------------------------------- #

    def get(self, model: str, field: str, comp: int = None, i_wt: int = None, 
                    i0_offset: bool = True, split: bool = False) -> np.array:
        """ Extract the model data from the c LagSolver object

        Parameters
        ----------
        model : str
            Sub-model from which data should be extracted ``W`` for wake or ``F`` 
            for ambient flow field.
        field : str
            Name of the field to be extracted.
        comp : int, optional
            Flow component to be extracted (``0``: x or ``1``: z), by default None.
        i_wt : int, optional
            Index of the Turbine data should be extracted from if None (by default
            data is extracted from all turbines.
        i0_offset : bool, optional
            If True (by default), the i0 offset is removed and data is shifted 
            accordingly.
        i0_offset : bool, optional
            If True the output array is splitted into a len(i_wt) list, by default False.
        
        Returns
        -------
        np.array
            Array containing the field requested.  

        Raises
        ------
        ValueError
            If model is not ``W`` or ``F``.
        ValueError
            If no field component, comp, is provided for a vector field. 
        ValueError
            If a field component, comp, is specified for a scalar field. 
        ValueError
            If the wind turbine index request, ``i_wt``, is not valid.
        """

        if model not in ['W', 'F']:
            raise ValueError(f'Inconsistent model type ({model}).')

        i_wt_list = [i_wt] if i_wt is not None else range(self.farm.n_wts)
        if self.farm.n_wts<i_wt_list[0]:
            raise ValueError(f'Invalid turbine index ({i_wt}<{self.farm.n_wts}).')

        model_p = self.data_p[model]

        if field.endswith('_p'):
            return self._get_particles(model_p, field, i_wt_list, comp = comp, i0_offset = i0_offset, split = split)
        else:
            return self._get_params(model_p, field, i_wt_list)

        # -------------------------------------------------------------------- #

    def get_WakeModel(self, *args, **kwargs) -> np.array:
        """ 
        Proxy for :meth:`.LagSolver.get`( 'W', ... ) 
        """
        return self.get('W', *args, **kwargs) 
        # -------------------------------------------------------------------- #

    def get_FlowModel(self, *args, **kwargs) -> np.array:
        """ 
        Proxy for :meth:`.LagSolver.get`( 'F', ... ) 
        """
        return self.get('F', *args, **kwargs) 
        # -------------------------------------------------------------------- #


    def get_part_iwt(self, model: str) -> np.array:
        """ Computes the mapping between the :meth:`.LagSolver.get`
        outputs for ``i_wt=None`` and the Turbines.

        Parameters
        ----------
        model : str
            Sub-model from which data should be extracted ``W`` for wake or ``F`` 
            for ambient flow field.

        Returns
        -------
        np.array
           Mapping between the array index and the turbine index.

        Raises
        ------       
        ValueError
            If model is not ``W`` or ``F``.

        See also
        --------
        :meth:`get<.LagSolver.get>`

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
            ``flow`` or ``rotor`` depending on the width of the filter used for the 
            ambient velocity field computation, by default ``flow``.
        buffer : py_comm.Vec, optional
            Vec object allocating the output memory location (allows not to 
            reallocate the wind farm global grid at every time step), buffer 
            shape should be consistent with ``xv`` and ``zv`` if None (by default)
            a new output vector is allocated.
        i_wt_exclude : int, optional
            Ignores the selected wind turbine for the ambient velocity 
            computations, by default -1 (ie: all turbines are used).

        Returns
        -------
        np.array
            Estimated ambient flow field ``[u, w]``

        Raises
        ------
        ValueError
            If the buffer shape provided is not consistent with the shape of 
            ``xv``, ``zv``.
        Exception
            If ``filt`` is not ``rotor`` or ``flow``.
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
            shape should be consistent with ``xv`` and ``zv`` if None (by default)
            a new output vector is allocated.

        Returns
        -------
        np.array
            Estimated ambient flow field ``[u, w]``

        Raises
        ------
        ValueError
            If the buffer shape provided is not consistent with the shape of 
            ``xv``, ``zv``.
        """
        x, y   = _IN2VEC(xv,zv)
        du_vec = py_comm.Vec((2,*x.x.shape)) if buffer is None else buffer
        if du_vec.x.shape != (2,*x.x.shape):
            raise ValueError("Inconsistent matrix shape for `du_vec`.")

        py_comm.interp_vec_WakeModel(self.p, x.p, y.p, x.s, du_vec.p)
        return du_vec.x
        # -------------------------------------------------------------------- #

    def rews_compute(self, x_rotor: List[float], r_rotor: float, comp: int = 0) -> float:
        """ Computes the Rotor Effective Wind Speed at ``x_rotor`` over a rotor of 
        diameter, ``r_rotor and`` oriented along x.

        Parameters
        ----------
        x_rotor : List[float]
            Fictive rotor center location ``[x,y,z]`` in [m].
        r_rotor : float
            Fictive rotor diameter in [m].
        comp : int
            Flow component to be evaluated (``0``: x or ``1``: z), by default None.

        Returns
        -------
        float
            The Rotor Effective Wind Speed of diameter ``r_rotor`` and located 
            at ``x_rotor``.
        """
        x_cast = np.array([x_rotor[0],x_rotor[1],x_rotor[2]])
        x = py_comm.Vec(x_cast)
        return py_comm.rews_compute(self.p, x.p, r_rotor, comp)
        # -------------------------------------------------------------------- #

    def get_bounds(self, model: str, i_wt: int, i_sigma: int = 0) -> List[List[float]]:
        """ Returns the bounds of the subdomain 

        Parameters
        ----------
        model : str
            Sub-model from which the bounds of the subdomain are extracted ``W`` 
            for wake or ``F`` for ambient flow field.
        i_wt : int, optional
            Index of the Turbine.
        i_sigma : int
            Index of the subdomain (0 -> sigma_f_* and 1 -> sigma_r_*), by default 0.

        Returns
        -------
        List[List[float]]
            List of the coordinates of the vertices of the subdomain. 

        Raises
        ------
        ValueError
            If model is not ``W`` or ``F``.
        """
        if   model == 'F':
            return [self.data_p['F'][i_wt].contents.bounds[i_sigma][i][0:2] for i in range(4)]
        elif model == 'W':
            return [self.data_p['W'][i_wt].contents.bounds[i][0:2] for i in range(4)]
        else:
            raise ValueError(f'Inconsistent model type ({model}).')
        # -------------------------------------------------------------------- #

    def get_restart(self, rst: dict = {}) -> None:
        for model in ['F', 'W']:
            rst[model] = dict()

            for (field, field_type) in _RST_MAP[model]:
                
                if field_type in [0,1]:
                    rst[model][field]  = self.get(model, field, i0_offset=False, split=True)            
                elif field_type in [2]:
                    rst[model][field] = np.swapaxes( np.array( [ self.get(model, field, comp=0, i0_offset=False, split=True), 
                                                                 self.get(model, field, comp=1, i0_offset=False, split=True) ] ), 0, 1 )          
                else:
                    raise Exception('Unkonwn field type')
        return rst
        # -------------------------------------------------------------------- #

    def init_from_restart(self, rst:dict, i_wt:List[int]=None) -> None:
    
        i_wt_list = list(range(self.farm.n_wts)) if i_wt is None else i_wt

        for model, rst_m in rst.items():
            for i, i_wt in enumerate(i_wt_list):
            
                buffer = [None]*len(_RST_MAP[model])
                
                for i_b, (field, _) in enumerate(_RST_MAP[model]):
                    if field.endswith('_p'):
                        buffer[i_b] = py_comm.Vec(rst_m[field][i].flatten()).p
                    else:
                        buffer[i_b] = rst_m[field][i]

                if model == 'F':
                    py_comm.init_FlowModel_states(self.data_p['F'][i_wt], *buffer)

                if model == 'W':
                    py_comm.init_WakeModel_states(self.data_p['W'][i_wt], *buffer)
        # -------------------------------------------------------------------- #
