import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

from readCenterlineForWM import Centerline
from ..farm import Farm

from .viz import Viz
from . import linespecs as ls

import matplotlib.patches as patches

I_MASK = 0

class WakeCenterlineTmpl(Viz):
    def __init__(self, farm: Farm, bf_dir, wm_str_id):
        super().__init__(farm)

        # ---------- #

        if hasattr(self.farm, '_viz_wm'): 
            self.wm    = self.farm._viz_wm.wm
            self.x     = self.farm._viz_wm.x
            self.t     = self.farm._viz_wm.t
            self.zc_xt = self.farm._viz_wm.zc_xt

            self.i_t   = -1
        
        else: self.farm._viz_wm = self

        # ---------- #

        self.wm, self.x, self.t, self.zc_xt = ([] for i in range(4))

        it0_str = None
        for fid in os.listdir(f'{bf_dir}/WM_centerline/'):
            if fid.startswith(wm_str_id):
                it0_str = fid.rsplit('_',1)[1]
                break
        gaussian_flag = 'gaussian' if 'gaussian' in wm_str_id else False

        self.n_wm_mask = len([fid for fid in os.listdir(f'{bf_dir}/WM_centerline/') if fid.rsplit('_',2)[-2]=='w00'])
        # ---------- #

        # plt.figure()

        # for wt in self.farm.wts:
        #     wm_path = f'{bf_dir}/WM_centerline/{wm_str_id}_w{wt.get_bf_idx():02d}_{it0_str}'
        #     wm = Centerline(wm_path, type=gaussian_flag)
        #     plt.scatter(wm.x[0],np.mean(wm.z[int(wm.nT/10):,0]))
        #     plt.text(wm.x[0],np.mean(wm.z[int(wm.nT/10):,0]),str(wt.i_bf))
        # plt.draw(); plt.pause(0.1); input("Press any key to exit"); plt.close()

        for wt in self.farm.wts:
            wm_path = lambda i: f'{bf_dir}/WM_centerline/{wm_str_id}{i:02d}_w{wt.i_bf:02d}_{it0_str}'

            wm = [Centerline(wm_path(i), mask_type=gaussian_flag) for i in range(self.n_wm_mask)]
            self.wm.append(wm)
            
            n_x = len(wm[0].x)
            n_t = len(wt.snrs)

            if np.sqrt(  (wm[0].x[0]   -wt.x[0])**2 
                       + (np.mean(wm[0].z[int(wm[0].nT/10):,0])-wt.x[2])**2  )>farm.af.D:
                raise Exception(f'File mismatch: x0 for wt{wt.i} [{wt.x[0]:2.0f}, {wt.x[2]:2.0f}] and wm [{wm[0].x[0]:2.0f}, {np.mean(wm[0].z[int(wm[0].nT/10):,0]):2.0f}] do not match') 

            self.x.append(wm[0].x)
            self.t.append(np.zeros(n_t))
            self.zc_xt.append(np.zeros((n_x, n_t)))
        
        self.i_t = 0
        
        # -------------------------------------------------------------------- #
    def update(self):
        if self.i_t==-1: return 

        for i_wt, wt in enumerate(self.farm.wts):
            self.zc_xt[i_wt][:,self.i_t] = np.interp( self.x[i_wt], 
                                                self.farm.lag_solver.get('W', 'x_p', 0, i_wt=i_wt),
                                                self.farm.lag_solver.get('W', 'x_p', 1, i_wt=i_wt) ) 
            self.t[i_wt][self.i_t] = self.farm.lag_solver.get_time()
        self.i_t += 1
        # -------------------------------------------------------------------- #
    def plot(self):
        super().plot()
        if self.farm._viz_wm==self:
            np.save(f'{self.farm.out_dir}/wakeCenterline_zc_xt', 
                                             np.array(self.zc_xt, dtype=object))
            np.save(f'{self.farm.out_dir}/wakeCenterline_wm', 
                                                np.array(self.wm, dtype=object))
            np.save(f'{self.farm.out_dir}/wakeCenterline_x', 
                                                 np.array(self.x, dtype=object))
            np.save(f'{self.farm.out_dir}/wakeCenterline_t', 
                                                 np.array(self.t, dtype=object))
        # -------------------------------------------------------------------- #

    def get_current(self, i_wt):
        wm = self.wm[i_wt][I_MASK]
        bf_interp = interpolate.interp2d(self.x[i_wt], wm.time, wm.z, kind='linear')
        zc_bf = bf_interp(self.x[i_wt], self.t[i_wt][self.i_t-1])

        return self.x[i_wt], zc_bf

class WakeCenterlineMinConfig(WakeCenterlineTmpl):
    def plot(self): # overides the builtin plot method
        pass

class WakeCenterlineXt(WakeCenterlineTmpl):
    def __init__(self, farm, bf_dir, wm_str_id):
        super().__init__(farm, bf_dir, wm_str_id)
        # -------------------------------------------------------------------- #
    def update(self):
        super().iterate()
        # -------------------------------------------------------------------- #
    def plot(self):
        super().plot()
        t_conv = self.farm.af.D/self.farm.u_h
        for i_wt, wt in enumerate(self.farm.wts):
            bf_interp = interpolate.interp2d(self.x[i_wt], self.wm[i_wt][I_MASK].time, self.wm[i_wt][I_MASK].z, kind='linear')
            zc_xt_bf = bf_interp(self.x[i_wt], self.t[i_wt])
            
            plt.figure(figsize=(6,8))
            
            h = [None]*3
            options = {'cmap':'seismic', 'vmin':-1,'vmax':1}
            plt.subplot(3,1,1)
            h[0] = plt.pcolor( (self.x[i_wt]-wt.x[0])/self.farm.af.D,
                               self.t[i_wt]/t_conv,
                               (zc_xt_bf- wt.x[2])/self.farm.af.D, 
                               **options)
            plt.subplot(3,1,2)
            h[1] = plt.pcolor( (self.x[i_wt]-wt.x[0])/self.farm.af.D,
                               self.t[i_wt]/t_conv,
                               (self.zc_xt[i_wt].T- wt.x[2])/self.farm.af.D, 
                               **options)
            plt.subplot(3,1,3)
            h[2] = plt.pcolor( (self.x[i_wt]-wt.x[0])/self.farm.af.D,
                               self.t[i_wt]/t_conv,
                            #    (self.zc_xt[i_wt].T- zc_xt_bf)/np.mean(np.abs(zc_xt_bf-np.mean(zc_xt_bf)),axis=0),# self.farm.af.D, 
                               np.ones_like(zc_xt_bf)*np.mean(np.abs(zc_xt_bf-np.mean(zc_xt_bf)),axis=0),# self.farm.af.D, 
                                cmap='seismic')

            for i in range(3):
                plt.subplot(3,1,i+1)
                plt.xlabel(r'$x/D$ [-]')
                plt.ylabel(r'$\frac{t}{T_{conv}}$')
                plt.colorbar(h[i], ax=plt.gca(), label=r'$z_{C,f3}-z_{C,bf}/D$ [-]')
                plt.xlim(((self.x[i_wt][j]-wt.x[0])/self.farm.af.D for j in [0, -1] ))
                plt.ylim((self.t[i_wt][0]/t_conv,self.t[i_wt][-1]/t_conv))

        plt.savefig(f'{self.farm.glob_set["log_dir"]}/wakeCenterline_xt_w{i_wt:02d}.eps')
        
        # -------------------------------------------------------------------- #

class WakeCenterlineXloc(WakeCenterlineTmpl):
    def __init__(self, farm, bf_dir, wm_str_id, x_loc, xlim=False):
        super().__init__(farm, bf_dir, wm_str_id)
        self.x_loc = x_loc
        self.xlim = xlim
        # -------------------------------------------------------------------- #
    def update(self):
        super().update()
        # -------------------------------------------------------------------- #
    def plot(self):
        super().plot()
        t_conv = 1.# self.farm.af.D/self.farm.u_h
        for i_wt, wt in enumerate(self.farm.wts):
            zc_xt_bf = [None]*len(self.wm[i_wt])
            for i_wm, wm in enumerate(self.wm[i_wt]):
                bf_interp = interpolate.interp2d(self.x[i_wt], wm.time, wm.z, kind='linear')
                zc_xt_bf[i_wm] = bf_interp(self.x[i_wt], self.t[i_wt])
            
            h = [None]*len(self.x_loc)
            fig, axs = plt.subplots(len(self.x_loc), 1, sharex=True, sharey=True, figsize=(6,8))
            for i_h, (h_i, ax) in enumerate(zip(h, axs)):
                x_wt_all = np.load(f'{self.farm.data_dir}/geo.npy')
                i_wt_row = np.where(np.abs(x_wt_all[:,2]-wt.x[2])<1E-3)
                x_wt_row = np.sort(x_wt_all[i_wt_row,0])
                row_idx  = np.argmin(np.abs(x_wt_row-wt.x[0]))+1                
                x_wt_next = x_wt_all[i_wt_row,0][0][row_idx] if row_idx<len(x_wt_row[0]) else 10E9

                is_valid_wm =     (self.wm[i_wt][I_MASK].x[0]<self.x_loc[i_h]+wt.x[0]<self.wm[i_wt][I_MASK].x[-1]) \
                              and self.x_loc[i_h]+wt.x[0]<x_wt_next

                plt.sca(ax)
                if is_valid_wm:
                    idx = np.argmin(np.abs(self.x[i_wt]-wt.x[0]-self.x_loc[i_h]))
                    plt.plot(self.t[i_wt]/t_conv, (zc_xt_bf[I_MASK][:,idx]-wt.x[2])/self.farm.af.D, **ls.REF)
                    plt.plot(self.t[i_wt]/t_conv, (self.zc_xt[i_wt][idx,:]-wt.x[2])/self.farm.af.D, **ls.MOD)

                    plt.ylabel(r'$z_C/D$',rotation=90)
                    # plt.plot([self.x_loc[i_h]/self.farm.u_h/t_conv]*2, (-1,1), 'k', linewidth=1.5)
                    plt.ylim((-1,1))

                    _x_min = int(self.t[i_wt][0])  
                    _x_max = int(self.t[i_wt][-2])
                    plt.xlim((_x_min, _x_max))

                    # ax.add_patch(patches.Rectangle([_x_min,-10],self.x_loc[i_h]/self.farm.u_h/t_conv,20,color=[0.5,0.5,0.5], linewidth=0))

                    plt.text(0.025, 0.975,f'{self.x_loc[i_h]/self.farm.af.D:3.0f}D',
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform = plt.gca().transAxes)

                    z_wm_up  = np.max([(zc[:,idx]-wt.x[2])/self.farm.af.D for zc in zc_xt_bf],axis=0)
                    z_wm_low = np.min([(zc[:,idx]-wt.x[2])/self.farm.af.D for zc in zc_xt_bf],axis=0)

                    ax.fill_between(self.t[i_wt]/t_conv,z_wm_low,z_wm_up, color=[0.7,0.7,0.7] )
                else:
                    plt.axis('off')

            plt.xlabel(r'$\frac{t}{T_{conv}}$') 
            plt.tight_layout()
            plt.savefig(f'{self.farm.out_dir}/wakeCenterline_xlocPlot_w{wt.i_bf:02d}.pdf')

        # -------------------------------------------------------------------- #