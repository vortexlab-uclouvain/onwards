import logging as lg

from OnWaRDS.turbine import MINIMAL_STATES
lg.basicConfig(level=lg.ERROR, )

from OnWaRDS import Farm
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
                'estimator0' : {'type':'fld_fromdata',  'meas_in':['uCz_m2D','yawA'], 'state_out':['w_inc','yaw']},
                'estimator1' : {'type':'uinc:ti_kfbem', 'w_sec':6, 'bem_args': {'type':'fast', 'solver':bem_solver}},
                'estimator2' : {'type':'ct_fromthrust'},
                'estimator3' : {'type':'ufs:wfs_waked',} 
            }    
            
snrs_args = {
                'type':   'SensorsPreprocessed',
                'export': 'snrs_buffer'
            }      

est_args =  {    
                'n_substeps' : 1,
                'estimator0' : {'type':'fld_fromdata',  'meas_in':MINIMAL_STATES,  'state_out':MINIMAL_STATES},
                'ini_states' : {'ct': 0, 'ti': 1, 'u_inc': 8,  'u_fs': 8}
            }
            
# est_args =  {    
#                 'n_substeps' : 1,
#                 'estimator0' : {'type':'fld_debug'},
#             }
            
model_args = {
                'n_substeps': 2s,
                'n_fm' : 100,
                'n_wm' : 100,
                'n_shed_fm': 1,
                'n_shed_wm': 4,
                'sigma_xi_r': 0.25,
                'sigma_r_f': 1,
                'sigma_t_r': 2/8,
                'sigma_xi_f': 4,
                'sigma_r_r': 0.6,
                'sigma_t_f': 4/8,             
                'cw': 0.6,           
                'cw': 0.6,           
                'ak': 0.018,           
                'bk': 0.12,
                'tau_r': 8           }

grid_args = {
                'dx' : 126/9,
                'dz' : 126/9,
                'margin' : [ [ 2.*126. ,  10.*126. ],
                             [ 4.*126. ,  4.*126. ] ]
            }


import matplotlib.pyplot as plt

import time
WF_DIR = '/Users/lejeunemax/BFBulk/WFprocessed/uh8ms_ti6prc/coupling_fs'
BF_DIR = '/Users/lejeunemax/BFBulk/coupling_32x8x16_8ms_6prc/coupling_fs/WF/'
WM_STR_ID = 'WMcenterline_gaussianMask' 

with  Farm(WF_DIR,'NREL', snrs_args, est_args, model_args, grid_args=grid_args, wt_cherry_picking=None) as wf: 
    # wf.viz_add('part')
    wf.viz_add('velfield', [3,10], 0, skip=10, bf_dir=BF_DIR )
    wf.viz_add('wakeCenterline_xloc', BF_DIR, WM_STR_ID, [4*126,6*126,9*126,12*126])

    start_fun = time.time()
    for t in wf:
        pass
            
    print(time.time()-start_fun)
plt.draw(); plt.pause(0.1); input("Press any key to exit"); plt.close() 
