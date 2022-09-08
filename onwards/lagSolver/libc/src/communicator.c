#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "airfoilLib.h"
#include "macro.h"
#include "lagSolver.h"

void interp_vec_FlowModel(LagSolver *wf, double *x_vec, double *y_vec, 
                                 int n_tot, double *u_vec, int filt, int i_wt_exclude) {
    double *sigma;

    switch (filt) {
        case 0: {sigma = wf->fms[0]->sigma_r; break;}
        case 1: {sigma = wf->fms[0]->sigma_f; break;}
    }
    
    int i;
    double *x_interp = VEC(2);
    double *u_interp = VEC(2);

    for (i = 0; i < n_tot; i++) { 
        x_interp[0] = x_vec[i];
        x_interp[1] = y_vec[i];
        interp_FlowModel_all(wf, x_interp, wf->t, sigma, u_interp, i_wt_exclude);
        u_vec[i]       = u_interp[0];
        u_vec[i+n_tot] = u_interp[1];
    } 

    free(x_interp);
    free(u_interp);
}

void interp_WakeModel_wrap(LagSolver *wf, double *x, double *work_var) {
    du_pos2d_compute_from_wf(wf, x, work_var);
}
void interp_vec_WakeModel(LagSolver *wf, double *x_vec, double *y_vec, int n_tot, double *u_vec) {
    VECTORIZE(interp_WakeModel_wrap, wf, x_vec, y_vec, n_tot, u_vec);
}