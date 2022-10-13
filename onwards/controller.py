import numpy as np
from airfoilLib import Airfoil
from uBEM3D import BEMSolver, BEMSolverFast

RHO = 1.225

def rpm2omega(x):
    return x/60*(2*np.pi)

def omega2rpm(x):
    return x*60/(2*np.pi)

class TurbineDynamics():
    def __init__(self, omega_r_0, Q_g_0, t_0, pitch_0):
        self.t = t_0
        self.drivetrain = Drivetrain(self, omega_r_0)
        self.controller = Controller(self, Q_g_0, self.drivetrain.omega_g, pitch_0)

    def update(self, t, uinf):
        dt = t - self.t
        self.t = t
        self.controller.update(dt, self.drivetrain.omega_g, self.drivetrain.omega_r)
        self.drivetrain.update(dt, uinf, self.controller.Q_g, self.controller.pitch)

class Drivetrain():
    def __init__(self, td: TurbineDynamics, omega_r: float):
        self.parent = td
        self.omega_r = omega_r

        self.af = Airfoil('NREL')
        # self.aero = BEMSolver(self.af, {})
        self.aero = BEMSolverFast( 'updatedBEMwithYaw_NREL_n64type2D' )

        self.r = 1173.7/12.1 # [-] gearbox ratio (!= 97 (initial gearbox ratio) if scale factor != 1)
        self.omega_g = omega_r * self.r

        # prameters
        Igen = 534.116 # [kg/m2] # generator inertia (relative to the high-speed shaft)
        Iblade = 11776047 # [kg/m2] # Inertia for 1 blade (w.r.t Root) !!!!!!! --> should be computed w.r.t the rotor center ? !!!!!!!!
        Ihub = 115926 # [kg/m2] # hub inertia (relative to the low-speed shaft)
        self.I = 3*Iblade + Ihub + (self.r**2*Igen) # [kg], drivetrain inertia cast to the low-speed shaft ((generator inertia relative to high speed shaft)*r**2 + (rotor inertia))

    def update(self, dt: float, uinf:float, Q_c: float, pitch_c: float):

        T_aero, _, Q_aero = self.aero.bladeFMCompute(self._feed_bem(uinf, self.omega_r, pitch_c))

        omega_r_new_rad = rpm2omega(self.omega_r) + dt*(Q_aero-Q_c*self.r)/self.I
        self.omega_r = omega2rpm(omega_r_new_rad)

        # print(f'{uinf=}, {self.omega_r=}, {pitch_c=} ===> {Q_aero=}  {self.omega_r=}')

        self.omega_g = self.omega_r * self.r
        self.Q_aero = Q_aero
        self.T_aero = T_aero
        self.Q_g    = Q_c

    def _feed_bem(self, uinf, omega, pitch):
        return { 'V0':    uinf,               # [ms-1]   
                 'Omega': rpm2omega(omega),   # [rpm] 
                 'beta':  np.deg2rad(pitch),  # [rad]
                 'theta': {'yaw':0} }         # [rad]

class Controller():
    def __init__(self, td: TurbineDynamics, Q_g_0: float, omega_g_0: float, pitch_0:float):
        self.parent = td

        self.Q_g = Q_g_0
        self.pitch = pitch_0

        self.omega_g_f = omega_g_0
        self.int_term = 0

        # parameters
        self.eta_mec = 0.944 # [-], mechanical efficiency

        self.ratedP = 5*1e6 # [W], rated power [W]
        self.ratedPtot = self.ratedP/self.eta_mec

        self.genSpeedMin = 670 # [RPM] minimal gen speed (no rescaling (assumption !!!))
        self.genSpeedMax = 1173.7 # [RPM] maximal gen speed (no rescaling (assumption !!!))

        # ----------- Filtering -----------------#
        #recursive, single-pole, low-pass filter with exponential smoothing
        self.fc = 0.25 #[Hz], = one-quarter of the blade's first edgewise natural frequency

        # ------- Torque controller  ------------- #

        Cp_opt = 0.482 # [-], optimal power coefficients (for region II (based on Jonkman's report))
        TSR_opt = 7.55 # [-], optimal Tip Speed Ratio
        Rt = 63

        self.Kstar = 0.5*RHO*np.pi*Rt**5*Cp_opt/(TSR_opt**3)/(self.parent.drivetrain.r**3)*(2*np.pi/60)**2  #for torque control (region II) 

        self.max_Qgen = self.ratedP/self.eta_mec/rpm2omega(self.genSpeedMax)
        self.max_Qgen = self.max_Qgen + 0.1*self.max_Qgen 
        self.max_Qgen_rate = 15000 #[Nm/s] 

        # ------- Blade pitch angle controller  ------------- # 
        #---> !!!!!!!!!!!!!!! currently, no rescaling of the pitch actuators and blade pitch control scheme (same for a rescaled NREL)!!!!!!!!!

        self.threshold_theta =  np.rad2deg(0.01745329)  #[deg] minimal blade pitch angle (useful for defining the control region)
        # PID controlled speed error will respond as a second order system with 
        self.zeta = 0.7 #[-] natural frequency 
        self.omega_0 = 0.6 # [rad/s] damping ratio

        theta_k = 6.302336 #[deg] blade pitch at which the pitch sensitivity has doubled from its value at the rated OP
        self.theta_k_rad = np.deg2rad(theta_k) #[rad]
        self.dP_dtheta = -25520000 #[Watt/rad] pitch sensitivity around OP 11.4 m/s and theta = 0 

        self.min_theta = 0 #[deg] min blade pitch setting 
        self.max_theta = 90 #[deg] max blade pitch setting 
        self.max_theta_rate = 8 #[deg/s] maximal pitch angle rate

    def update(self, dt, omega_g, omega_r):
        # self.pitch = 0

        alpha = np.exp(-2*np.pi*self.fc*dt)
        self.omega_g_f = (1-alpha) * omega_g + alpha * self.omega_g_f 

        self.region = self.control_region(self.omega_g_f, self.pitch)
        self.Q_g    = self.torque_control(dt, self.region, self.omega_g_f)
        self.pitch  = self.pitch_control(dt, self.omega_g_f, self.pitch)

        # print(f'{self.region=}  {self.Q_g=}')

    def control_region(self, omega_g: float, pitch: float):
        if   omega_g >= self.genSpeedMax or pitch >= self.threshold_theta:
            return 5 
        elif omega_g < self.genSpeedMin:
            return 1 
        elif omega_g >= self.genSpeedMin \
                          and omega_g < (self.genSpeedMin+0.3*self.genSpeedMin):
            return 2 
        elif omega_g >= self.genSpeedMin+0.3*self.genSpeedMin \
                                            and omega_g < 0.99*self.genSpeedMax:
            return 3 
        else:
            return 4 

    def torque_control(self, dt, region, omega_g):

        if region == 1:
            
            Q_g = 0.0 
            
        elif region == 2: 
            
            # Region I 1/2 : the generator torque varies linearly with the generator
            # speed (to have a lower bound in the generator torque)
            
            gen_speed_in = self.genSpeedMin 
            gen_speed_out = gen_speed_in + 0.3*gen_speed_in 
            slope_next_region = self.Kstar
            gen_torque_in = 0
            gen_torque_out = ((gen_speed_out*gen_speed_out)*slope_next_region)
            
            slope = (gen_torque_out - gen_torque_in)/(gen_speed_out - gen_speed_in) 
            Q_g = (gen_torque_in + (omega_g - gen_speed_in)*slope)
            
        elif region == 3:
            
            # Region II : the generator torque varies with the square of the
            # generator speed (to keep a constant TSR in that region)
            slope = self.Kstar    
            Q_g = ((omega_g*omega_g)*slope)

            
        elif region == 4:
            
            # Region II 1/2 : the generator torque varies linearly with the
            # generator speed (because the rated generator torque is reached before 
            # the rated generator speed)
            
            gen_speed_out = self.genSpeedMax
            gen_speed_in = 0.99*gen_speed_out

            slope_previous_region = self.Kstar   
            gen_torque_in = ((gen_speed_in*gen_speed_in)*slope_previous_region)
            gen_torque_out = self.ratedPtot/rpm2omega(gen_speed_out) 
            
            slope = (gen_torque_out - gen_torque_in)/(gen_speed_out - gen_speed_in)
            
            Q_g = (gen_torque_in + (omega_g - gen_speed_in)*slope)
            
        elif region == 5:
            # Region III : the generator torque and the generator speed are
            # maintained constant (because of the generator constraints and a 
            # maximal speed at the blade tip)
            Q_g = self.ratedPtot/rpm2omega(omega_g) 

        # Saturation blocks
        if Q_g > self.max_Qgen :
            Q_g = self.max_Qgen 
        elif Q_g < 0.0: 
            Q_g = 0.0 

        Q_g_rate = (Q_g - self.Q_g)/dt 

        if(abs(Q_g_rate) > self.max_Qgen_rate) :
            Q_g_rate = np.sign(Q_g_rate)*self.max_Qgen_rate 

        return self.Q_g + Q_g_rate*dt 
        
    def pitch_control(self, dt, omega_gen, pitch):

        I = self.parent.drivetrain.I 
        r = self.parent.drivetrain.r 

        rated_rotor_speed_rad = rpm2omega(self.genSpeedMax)/r

        # Computation of the speed error (compared to the rated speed)
        speed_error = r*(rpm2omega(omega_gen)/r-rated_rotor_speed_rad)
            
        GK = 1/(1+np.deg2rad(pitch)/self.theta_k_rad)

        # Proportional gain
        KP = 2*I*rated_rotor_speed_rad*self.zeta*self.omega_0*GK/(r*-self.dP_dtheta)
            
        # Integral gain 
        KI = I*rated_rotor_speed_rad*self.omega_0**2*GK/(r*-self.dP_dtheta)
        self.int_term += speed_error*dt 
            
        #Saturation of the integral term 
        if self.int_term < np.deg2rad(self.min_theta)/KI:
            self.int_term = np.deg2rad(self.min_theta)/KI 
        elif self.int_term > np.deg2rad(self.max_theta)/KI:
            self.int_term = np.deg2rad(self.max_theta)/KI
            
        # Update of the blade pitch angle
        pitch_new = KP*speed_error + KI*self.int_term
            
        # Saturation of the blade pitch angle 
        if pitch_new < np.deg2rad(self.min_theta):
                pitch_new = np.deg2rad(self.min_theta)
        elif pitch_new > np.deg2rad(self.max_theta):
                pitch_new = np.deg2rad(self.max_theta)
            
        # Saturation of the blade pitch angle rate
        pitch_angle_rate_n = (pitch_new - pitch)/dt 
        if np.sign(pitch_angle_rate_n) > np.deg2rad(self.max_theta_rate) :
            pitch_new = pitch + np.deg2rad(dt*self.max_theta_rate*np.sign(pitch_angle_rate_n))
            
        return np.rad2deg(pitch_new)

if __name__=='__main__':
    td = TurbineDynamics(1,0,-1,0)

    tt= np.linspace(0,200,201)
    # tt= np.linspace(0,10,11)

    qall = np.zeros_like(tt)
    omall = np.zeros_like(tt)

    uinf = 3

    uu = range(3,20)
    quu = np.zeros_like(uu)
    omuu = np.zeros_like(uu)

    t = 0
    for i_u, u in enumerate(uu):
        for i_t, _ in enumerate(tt):
            t += 1
            td.update(t,u)
            qall[i_t] = td.drivetrain.Q_aero
            omall[i_t] = td.drivetrain.omega_g
        quu[i_u] = qall[-1] 
        omuu[i_u] = omall[-1] 
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot(uu,quu)

    plt.subplot(1,2,2)
    plt.plot(uu,omuu)


    plt.draw(); plt.pause(0.1); input("Press any key to exit"); plt.close() 