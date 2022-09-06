import logging
lg = logging.getLogger(__name__)

import os
import ctypes
import numpy as np

from ctypes import POINTER, pointer, c_int, c_double, c_char
c_double_p = POINTER(c_double) 
c_double_pp = POINTER(c_double_p) 

from ...airfoil import c_Airfoil_p
from ...turbine import MINIMAL_STATES, Turbine

_ls_lib = ctypes.CDLL(os.environ['ONWARDS_PATH']+'/OnWaRDS/lagSolver/libc/lagSolver_c.so') 
# ---------------------------------------------------------------------------- #

class Vec():
    def __init__(self, arg):
        if isinstance(arg, (tuple)) or np.issubdtype(type(arg), int): self.x = np.zeros(arg)
        else:                                                         self.x = arg
        self.p = self.x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.s = self.x.size
        # -------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #

class c_Sensors(ctypes.Structure):
    _fields_ = [ (ms, ctypes.c_double) for ms in MINIMAL_STATES ]
    def __init__(self, wt: Turbine):
        super().__init__()
        self.p  = pointer(self)
        self.wt = wt 

        for s in self._fields_:
            if s[0] not in self.wt.states:
                raise Exception( f'Wind turbine operating state {s} not found.')
        
        self.update()
        # -------------------------------------------------------------------- #

    def update(self):
        for f in self.wt.states:
            setattr(self, f, float(self.wt.states[f]) )
        # -------------------------------------------------------------------- #

    def free(self):
        pass
        # -------------------------------------------------------------------- #

c_Sensors_p = POINTER(c_Sensors)
# ---------------------------------------------------------------------------- #

class c_Turbine(ctypes.Structure):
    _fields_ = [ ('i',         c_int      ),
                 ('is_fs',     c_int      ),
                 ('t',         c_double   ),
                 ('c_x_p',     c_double_p ),
                 ('c_af_p',    c_Airfoil_p),
                 ('c_snrs_p',  c_Sensors_p) ]

    def __init__(self, wt: Turbine, **kwargs):
        super().__init__()
        self.p = pointer(self)
        self.wt     = wt

        self.i      = int(self.wt.i)
        self.t      = float(self.wt.t) 
        self.x      = self.wt.x
        self.is_fs  = 1

        self.c_x_p    = self.x.ctypes.data_as(c_double_p) 
        self.c_af_p   = wt.af.p

        self.c_snrs   = c_Sensors(wt)
        self.c_snrs_p = self.c_snrs.p
        # -------------------------------------------------------------------- #

    def update(self, **kwargs):
        self.t = self.wt.t
        self.c_snrs.update()
        # -------------------------------------------------------------------- #

    def free(self):
        pass
        # -------------------------------------------------------------------- #

c_Turbine_p = POINTER(c_Turbine)
# ---------------------------------------------------------------------------- #

# class c_SpeedDeficit(ctypes.Structure):
#     _fields_ = [ ('ct', c_double),
#                  ('ti', c_double),
#                  ('ti', c_double),
#                  ('ti', c_double) ]

#     def __init__(self, wt: Turbine, **kwargs):
#         super().__init__()
#         # -------------------------------------------------------------------- #

#     def update(self, **kwargs):
#         pass
#         # -------------------------------------------------------------------- #

#     def free(self):
#         pass
#         # -------------------------------------------------------------------- #

# c_SpeedDeficit_p = POINTER(c_SpeedDeficit)
# CALLBACK=ctypes.CFUNCTYPE(None, ctypes.POINTER(c_SpeedDeficit_p))
# ---------------------------------------------------------------------------- #


class c_Set(ctypes.Structure):

    _fields_ = [('n_fm',          c_int   ),
                ('n_shed_fm',     c_int   ),
                ('n_wm',          c_int   ),
                ('n_shed_wm',     c_int   ),
                ('sd_type',       c_int   ),
                ('c0',            c_double),
                ('cw',            c_double),
                ('dt',            c_double),
                ('sigma_xi_f',    c_double),
                ('sigma_r_f',     c_double),
                ('sigma_t_f',     c_double),
                ('sigma_xi_r',    c_double),
                ('sigma_r_r',     c_double),
                ('sigma_t_r',     c_double),
                ('ak',            c_double),
                ('bk',            c_double),
                ('ceps',          c_double),
                ('tau_r',         c_double),]

    _fields_types_map_ = {f[0]: f[1] for f in _fields_}

    def __init__(self, model_args):
        super().__init__()
        self.p = pointer(self)
        for f, f_type in self._fields_types_map_.items():
            if f in model_args:
                if f_type==c_double: setattr( self, f, float( model_args[f] ) )
                if f_type==c_int   : setattr( self, f,   int( model_args[f] ) )
            else:
                raise Exception(f'Model Initialization failed: no value was provided for {f}.')
        # -------------------------------------------------------------------- #

    def update(self, model_args_new):
        for f in ['n_fm', 'n_wm', 'sd_type']:
            if f in model_args_new:
                raise ValueError(f'{f} does not support value reallocation.')

        for f in model_args_new:
            if self._fields_types_map_[f]==c_double: 
                setattr( self, f, float( model_args_new[f] ) )
            if self._fields_types_map_[f]==c_int   :
                setattr( self, f,   int( model_args_new[f] ) )
        
            lg.info(f'updating field {f} = {model_args_new[f]}')
        # -------------------------------------------------------------------- #

    def free(self):
        pass
        # -------------------------------------------------------------------- #

c_Set_p = POINTER(c_Set)
# ---------------------------------------------------------------------------- #

class c_LagSolver(ctypes.Structure): 
    # Opaque structure
    _fields_ = [ ('n_wt', c_int   ),
                 ('t',    c_double) ]

c_LagSolver_p = POINTER(c_LagSolver)

# ---------------------------------------------------------------------------- #

class c_FlowModel(ctypes.Structure): 
    # Opaque structure 
    _fields_ = [ ('i0',      c_int   ),
                 ('it',      c_int   ),
                 ('n',       c_int   ),
                 ('n_tshed', c_int   ),
                 ('dt',      c_double),
                 ('t_p',     c_double_p),
                 ('xi_p',    c_double_p),
                 ('x_p',     c_double_pp),
                 ('u_p',     c_double_pp),
                 ('uf_p',    c_double_pp) ]

c_FlowModel_p = POINTER(c_FlowModel)


# ---------------------------------------------------------------------------- #

class c_WakeModel(ctypes.Structure): 
    _fields_ = [ ('i0',      c_int   ),
                 ('it',      c_int   ),
                 ('n',       c_int   ),
                 ('n_tshed', c_int   ),
                 ('dt',      c_double),
                 ('t_p',     c_double_p),
                 ('xi_p',    c_double_p),
                 ('ct_p',    c_double_p),
                 ('ti_p',    c_double_p),
                 ('yaw_p',    c_double_p),
                 ('x_p',     c_double_pp),
                 ('uinc_p',  c_double_pp) ]

c_WakeModel_p = POINTER(c_WakeModel)

# ---------------------------------------------------------------------------- #

# Initialization functions
_ls_lib.init_LagSolver.argtypes = [c_int, c_Set_p]
_ls_lib.init_LagSolver.restype  = c_LagSolver_p
init_LagSolver = _ls_lib.init_LagSolver

_ls_lib.add_WindTurbine.argtypes = [c_LagSolver_p, c_Turbine_p]
_ls_lib.add_WindTurbine.restype  = None
add_WindTurbine = _ls_lib.add_WindTurbine

_ls_lib.get_FlowModel.argtypes = [c_LagSolver_p, c_Turbine_p]
_ls_lib.get_FlowModel.restype  = c_FlowModel_p
get_FlowModel = _ls_lib.get_FlowModel

_ls_lib.get_WakeModel.argtypes = [c_LagSolver_p, c_Turbine_p]
_ls_lib.get_WakeModel.restype  = c_WakeModel_p
get_WakeModel = _ls_lib.get_WakeModel

# Free memory
_ls_lib.free_LagSolver.argtypes = [c_LagSolver_p]
_ls_lib.free_LagSolver.restype  = None
free_LagSolver = _ls_lib.free_LagSolver

# Updating model's state
_ls_lib.update_LagSolver.argtypes = [c_LagSolver_p]
_ls_lib.update_LagSolver.restype  = None
update_LagSolver = _ls_lib.update_LagSolver

_ls_lib.reset_LagSolver.argtypes = [c_LagSolver_p]
_ls_lib.reset_LagSolver.restype  = None
reset_LagSolver = _ls_lib.reset_LagSolver

# Computing flow field
_ls_lib.rews_compute.argtypes = [c_LagSolver_p, c_double_p, c_double]
_ls_lib.rews_compute.restype  = c_double
rews_compute = _ls_lib.rews_compute

_ls_lib.interp_vec_WakeModel.argtypes = [c_LagSolver_p, c_double_p, c_double_p, c_int, c_double_p]
_ls_lib.interp_vec_WakeModel.restype  = None
interp_vec_WakeModel = _ls_lib.interp_vec_WakeModel

_ls_lib.interp_vec_FlowModel.argtypes = [c_LagSolver_p, c_double_p, c_double_p, c_int, c_double_p, c_int, c_int]
_ls_lib.interp_vec_FlowModel.restype  = None
interp_vec_FlowModel = _ls_lib.interp_vec_FlowModel

# ---------------------------------------------------------------------------- #
