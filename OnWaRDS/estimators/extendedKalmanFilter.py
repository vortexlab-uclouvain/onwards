import numpy as np

class ExtendedKalmanFilter:
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

    def jacobian(self, x, dx, argf, f):
        for j in range(self.nS):
            self._dxj.fill(0.0) 
            self._dxj[j]   = dx[j]
            self._jac[:,j] = ( f(x+self._dxj, argf) - f(x-self._dxj, argf) ) / dx[j]
        return self._jac

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