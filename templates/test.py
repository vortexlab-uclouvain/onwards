import logging as lg
lg.basicConfig(level=lg.WARNING, )

from onwards.turbine import MINIMAL_STATES

import numpy as np

from onwards import Farm
from uBEM3D  import BEMSolverFast

bem_solver = BEMSolverFast( 'updatedBEMwithYaw_NREL_n64type2D' )

snrs_args = {
                'type':        'SensorsPy',
                'fs':          1, 
                'zero_origin': False
            }      

            
est_args =  {    
                'n_substeps' : 1,
                'export'     : 'snrs_buffer',
                'export_overwrite' : True,
                'export_user_field': ['uCx_p3D', 'uCx_p6D', 'uCx_p9D', 'uCx_p12D', 'uCx_m2D', 'uCz_m2D', 'uCz_r', 'ti_m2D'],
                'estimator0' : {'type':'fld_fromdata',  'meas_in':['uCz_m2D','yawA'], 'state_out':['w_inc','yaw']},
                'estimator1' : {'type':'uinc:ti_kfbem', 'n_sec':8, 'w_sec':240, 'bem_args': {'type':'fast', 'solver':bem_solver}},
                'estimator2' : {'type':'ct_fromthrust'},
                'estimator3' : {'type':'ufs:wfs_waked'} 
            }    
            
snrs_args = {
                'type':   'SensorsPreprocessed',
                'export': 'snrs_buffer'
            }      

est_args =  {    
                'n_substeps' : 1,
                'estimator0' : {'type':'fld_fromdata',  'meas_in':MINIMAL_STATES,  'state_out':MINIMAL_STATES},
                # 'estimator1' : {'type':'fld_debug'},
                'ini_states' : {'ct': 0, 'ti': 1, 'u_inc': 8,  'u_fs': 8}
            }
            
# est_args =  {    
#                 'estimator0' : {'type':'fld_debug'},
#                 'n_substeps' : 1,
#             }
            
model_args = {
                'n_substeps': 4,
                'n_fm' : 100,
                'n_wm' : 100,
                'n_shed_fm': 1,
                'n_shed_wm': 2,
                'sigma_xi_r': 0.5,
                'sigma_r_f': 1,
                'sigma_t_r': 16/8,
                'sigma_xi_f': 4,
                'sigma_r_r': 1,
                'sigma_t_f': 16/8,             
                'cw': 	0.541,
                'c0': 	0.729,
                'ak': 	0.021,
                'bk': 	0.039,
                'ceps': 0.201,          
                # 'cw': 0.45 ,           
                # 'c0': 0.7,           
                # 'ak': 0.016,  
                # 'bk': 0.1,
                # 'ceps': 0.2,         
                'tau_r': 16,         }

grid_args = {
                'dx' : 126/9,
                'dz' : 126/9,
                'margin' : [ [ 2.*126. ,  10.*126. ],
                             [ 2.*126. ,  2.*126. ] ]
            }

grid_args = {
                'dx' : 126/9,
                'dz' : 126/9,
                'margin' : [ [ 2.*126. ,  10.*126. ],
                             [ 2.*126. ,  2.*126. ] ]
            }

import matplotlib.pyplot as plt

import time
WF_DIR = '/Users/lejeunemax/BFBulk/WFprocessed/uh8ms_ti6prc/coupling_fs'
BF_DIR = '/Users/lejeunemax/BFBulk/coupling_32x8x16_8ms_6prc/coupling_fs/WF/'
WM_STR_ID = 'WMcenterline_gaussianMask' 

with  Farm( WF_DIR,'NREL', snrs_args, est_args, model_args, 
            grid_args=grid_args, wt_cherry_picking=None    ) as wf: 
    # wf.viz_add('part')
    wf.viz_add('estimator', ['u_inc', 'ti'], ['uCx_m2D', 'ti_m2D'], ['u_{WT}', 'TI'], ['ms^{-1}', '\%'], offset=[31.5, 31.5], ylim=[[6, 10], [0,0.2]])
    # wf.viz_add('velfield', [3,10], 0, skip=1, bf_dir=BF_DIR, skeleton=True )
    # wf.viz_add('wakeCenterline_xloc', BF_DIR, WM_STR_ID, [3*126,6*126,9*126,12*126], u_norm=8.0)

    for wt in wf.wts:
        xod_rews = [3,6,9]
        for xod in xod_rews:
            x_eval = np.array([xod*wf.af.D, 0, 0])
            key = f'uCx_p{xod}D'
            if key in wt.snrs:
                wf.viz_add('rews', wt.snrs['time'], wt.snrs[key], x_eval, fs_flag=True, wt=wt, u_norm=8)

    start_fun = time.time()
    for t in wf:
        pass

    for v in wf.viz: v.plot()
            
    print(time.time()-start_fun)
plt.draw(); plt.pause(0.1); input("Press any key to exit"); plt.close() 
