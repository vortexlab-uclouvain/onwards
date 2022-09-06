from .lagSolver import LagSolver

# from .src import pyCommunicator as pyCom
from .libc.pyCommunicator import c_LagSolver, c_WakeModel, c_FlowModel, c_Turbine
from .libc.pyCommunicator import c_LagSolver_p, c_WakeModel_p, c_FlowModel_p, c_Turbine_p

from .libc.pyCommunicator import add_WindTurbine

