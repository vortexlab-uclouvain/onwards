from __future__ import annotations
from typing import TYPE_CHECKING

import logging
lg = logging.getLogger(__name__)

import numpy as np

from .estimator import Estimator
if TYPE_CHECKING:
    from turbine import Turbine

C2TPI    = 2. * np.pi
CPIO180  = np.pi / 180.
SPD2OMEG = C2TPI / 60.

def _amod_r(alpha):
    """
    recompute all angles in 0-pi
    """
    if type(alpha) == list:
        return [a%(C2TPI) for a in alpha]
    else:
        return alpha%(C2TPI)
    # ------------------------------------------------------------------------ #

def _angle_between(n, a, b):
    """
    checks if angle n is between angles a and b
    """
    n, a, b = n%C2TPI, a%C2TPI, b%C2TPI
    if (a < b):  return a <= n and n <= b
    else:        return a <= n or  n <= b
    # ------------------------------------------------------------------------ #

class Est_uincti_kfbem(Estimator):
    def __init__(self, wt: Turbine, avail_states: list, est_args: dict):
        """
        An estimator that computes u_inc and ti from the blade loads using a 
        Kalman Filter coupled to a BEM code [1]_.

        :Input(s) states:       * None

        :Input(s) measurements: * Rotation (``rotSpeed``) [rpm]
                                * Blade collective pitch (``pitchA``) [rad]
                                * Yaw angle (``yawA``) [rad]
                                * Blade azimuthal angle (``theta``) [rad]
                                * Edgewise bending moments (``Me``) [NM]

        :State(s) computed:     * Turbulence intensity (``TI``) [%]
                                * Rotor Effective Wind Speed (``u_inc``) [ms-1]

        The input parameters are:

        Parameters
        ----------
        wt : Turbine
            Parent :class:`.Turbine` object.
        avail_states : list
            List of the Turbine's states available.
        est_args : dict
            Dictionary containing the parameters used for the estimator.

            Available fields:
            
            :n_sec:          *(int, optional)*       - 
                Number of sector.
            :offset_sec:     *(float, optional)*     - 
                Azimuthal angle [rad] corresponding to the start of the first 
                sector.
            :w_sec:          *(str, optional)*       - 
                Time filtering constant [s] used for the computation of the time 
                averaged flow statistics.
            :bem_args:       *(dict)*       - 
                Description of the BEM solver used.

        Raises
        ------
        Exception
            _description_
        ValueError
            If the BEM solver arguments are not consistent.

        References
        ----------
        .. [1] C. L. Bottasso, S. Cacciola, and J. Schreiber. Local wind speed estimation, with application to wake impingement detection. Renewable Energy, 116:155–168, 2018.
        
        """

        meas   = ['rotSpeed', 'pitchA', 'yawA', 'theta', 'Me', 'uCx_m2D']
        states = ['u_inc', 'ti']
        req_states = [] 
        super().__init__(wt, meas, states, req_states, avail_states)
        
        self.n_sec      = est_args.setdefault('n_sec',      4       )
        self.offset_sec = est_args.setdefault('offset_sec', np.pi/4.)
        self.w_sec      = est_args.setdefault('w_sec',      600.    )

        dt = 1./wt.fs
        self.n_t = int((self.w_sec)//(dt))

        # Wind turbine initial state
        u0  = 8.
        ti0 = 7.
        
        self.u_ref = u0 
        self.alpha = np.exp( -(dt/self.w_sec) )
        self._state = {'u': u0, 'ubar': u0, 'ti': ti0}

        # Wind turbine state model
        bem_args = est_args['bem_args']
        if bem_args['type'] == 'fast':
            bem_solver =  bem_args['solver']
            
            def feed_bem(t, V0):
                return ( V0, 
                         self.wt.snrs.get_buffer_data('rotSpeed') * SPD2OMEG, 
                         self.wt.snrs.get_buffer_data('pitchA')   * C2TPI,
                         self.wt.snrs.get_buffer_data('yawA')     * C2TPI )

            def f(x, argf): return x[:]
            def h(x, argh): return bem_solver.QInterpolate( feed_bem(argh, x) )

        else:
            raise ValueError(  f'The BEM solver type you specified ({bem_args["type"]})' 
                             + f' does not exist or is not supported currently' )        
        
        # Initialisation of the Kalman filter (one for each blade)
        theta = _amod_r( np.linspace(0, C2TPI , self.n_sec+1) + self.offset_sec )
        self._sectors = [ _Sector(self, theta[i_s:i_s+2], u0, ti0, f, h) 
                                               for i_s in range(self.n_sec)    ]
        # -------------------------------------------------------------------- #

    def reset(self):
        super().reset()
        dt = 1./self.wt.fs

        u0  = self.wt.snrs.get_buffer_data('uCx_m2D')
        ti0 = 7.
        self.u_ref = u0 
        self._state = {'u': u0, 'ubar': u0, 'ti': ti0}

        for s in self._sectors: s.reset()
        # -------------------------------------------------------------------- #

    def update_snrs(self):
        for i_b in range(self.wt.af.nB):
            sec = next(s for s in self._sectors if s.in_bounds(self.wt.snrs.get_buffer_data('theta', i_b=i_b)))
            sec.update_snrs(self.wt.t, self.wt.snrs.get_buffer_data('Me', i_b=i_b)*self.wt.af.nB )
        # -------------------------------------------------------------------- #

    def update(self):
        self.update_snrs()
        for s in self._sectors:
            s.iterate_ekf(self.wt.t)

        self.wt.states['u_inc' ] = sum( s.get_u() for s in self._sectors )/self.n_sec
        self.u_ref =  self.alpha * self.u_ref + (1.-self.alpha) *  self.wt.states['u_inc' ]
        self.wt.states['ti' ] = np.sqrt( sum( s.get_ti(self.u_ref) for s in self._sectors )/(self.n_sec-1) )/self.u_ref
        # -------------------------------------------------------------------- #

class _Sector():
    def __init__(self, u_est, theta, u0, ti0, f, h):
        self.u_est = u_est # parent rotor streamwise veclocity estimator
        self.theta = theta # sector bounds 

        self.ekf = _SectorEKF(u0, f, h)

        self.it     = -1
        self.t_prev = -1e16
        self.tau    = 6
        self.m_flap_avg = 0
        self.was_updated = True

        self.u_se_buffer = np.ones(self.u_est.n_t) * u0
        self.ti_buffer   = np.ones(self.u_est.n_t) * ti0
        # -------------------------------------------------------------------- #
    
    def reset(self, u0, ti0):
        self.ekf.reset()

        self.it     = -1
        self.t_prev = -1e16
        self.tau    = 6
        self.m_flap_avg = 0
        self.was_updated = True

        self.u_se_buffer[:] = u0
        self.ti_buffer[:]   = ti0


    def iterate_ekf(self, t):
        if self.was_updated:        
            self.it = (self.it+1)%(self.u_est.n_t)
            self.u_se_buffer[self.it] = self.ekf.iterate(t, self.m_flap_avg) / 1.058
            self.was_updated = False
        # -------------------------------------------------------------------- #
            
    def update_snrs(self, t, m_flap_inst):
        alpha = np.exp( -(t-self.t_prev)/self.tau )
        self.m_flap_avg = (alpha) * self.m_flap_avg + (1-alpha) * m_flap_inst 
        self.t_prev = t
        self.was_updated = True
        # -------------------------------------------------------------------- #

    def in_bounds(self, theta):
        return _angle_between(theta, *self.theta)
        # -------------------------------------------------------------------- #

    def get_u(self):
        return self.u_se_buffer[self.it]
        # -------------------------------------------------------------------- #

    def get_ubar(self):
        return np.mean( self.u_se_buffer )
        # -------------------------------------------------------------------- #

    def get_ti(self, u_ref, w=1):
        idx = np.linspace(self.it-w+1, self.it+1, 1, dtype=np.int32)
        return np.sum((self.u_se_buffer[idx%self.u_est.n_t] - u_ref)**2)
        # -------------------------------------------------------------------- #
    
class _SectorEKF():
    def __init__(self, u0, f, h):
        P_0  = np.array([[10.]])
        u_0  = np.array([[u0]])
        Q    = np.array([[0.010]])
        R    = np.array([[10E10]])
        dx_0 = np.array([[0.1]])

        self.EKF = _ExtendedKalmanFilter(Q, R, f, h, P_0, u_0, dx_0)
        # -------------------------------------------------------------------- #
    
    def reset(self):
        pass
        # -------------------------------------------------------------------- #
    
    def iterate(self, t, m_flap_avg):
        return self.EKF.iterate(m_flap_avg, argh=t)
        # -------------------------------------------------------------------- #

class _ExtendedKalmanFilter:
    def __init__(self, Q, R, f, h, P_0, x_0, dx_0):
        self.Q, self.R, self.f, self.h  = Q, R, f, h

        self.nS = len(Q) # number of state
        self.nM = len(R) # number of measurements

        self.I = np.eye( self.nS )

        self.P, self.xk, self.dxk =  P_0, x_0, dx_0
        self.P_reboot, self.xk_reboot = P_0, x_0

        self._jac = np.zeros((self.nS,self.nS))
        self._dxj = np.zeros(self.nS)

        self._xkm1  = np.zeros(self.nS)
        self._dxkm1 = np.zeros(self.nS)
        
        self.dkx = np.zeros_like(self._dxkm1)
        # -------------------------------------------------------------------- #

    def jacobian(self, x, dx, argf, f):
        for j in range(self.nS):
            self._dxj.fill(0.0) 
            self._dxj[j]   = dx[j]
            self._jac[:,j] = ( f(x+self._dxj, argf) - f(x-self._dxj, argf) ) / dx[j]
        return self._jac
        # -------------------------------------------------------------------- #

    def iterate(self, y, argf=None, argh=None):
        self._xkm1[:]  = self.xk[:]
        self._dxkm1[:] = self.dxk[:]

        # Prediction step
        self.xk = self.f(self.xk, argf)
        jac_f = self.jacobian(self.xk, self.dxk, argf, self.f)
        jac_h = self.jacobian(self.xk, self.dxk, argh, self.h)
        self.P = np.dot( np.dot( jac_f, self.P ), jac_f.T ) + self.Q

        # Correction step (gain computation)
        K = np.dot( np.dot( self.P, jac_h.T ), \
                np.linalg.inv( np.dot( np.dot( jac_h, self.P ), jac_h.T ) + self.R ) )
        self.xk += np.dot( K, y - self.h(self.xk, argh) )
        self.P   = np.dot( self.I - K.dot(jac_h) , self.P )

        if np.isnan(self.xk).any():  self.xk[:]  = self._xkm1[:]
        if np.isnan(self.P).any() :  self.P[:,:] = self.P_reboot[:,:]

        self.dkx[:] = self._dxkm1[:] if self.xk==self._xkm1 else self.xk-self._xkm1

        return self.xk