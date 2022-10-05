// Copyright (C) <2022> <Université catholique de Louvain (UCLouvain), Belgique>

// List of the contributors to the development of OnWaRDS: see LICENSE file.
// Description and complete License: see LICENSE file.
	
// This program (OnWaRDS) is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "airfoilLib.h"
#include "macro.h"  
#include "lagSolver.h"

/* ------------------------------------------------ */
/*                    WakeModel                     */
/* ------------------------------------------------ */

WakeModel* init_WakeModel(LagSolver *wf, WindTurbine *wt) {
    WakeModel *wm = ALLOC(WakeModel);

    wm->wf = wf;
    wm->wt = wt;

    init_WakeModel_set(wm, wf->set);

    wm->t_p     = VEC(wm->n);
    wm->xi_p    = VEC(wm->n);
    wm->ct_p    = VEC(wm->n);
    wm->yaw_p   = VEC(wm->n);
    wm->ti_p    = VEC(wm->n);
    wm->x_p     = (double**) malloc( sizeof(double*) * wm->n );
    wm->n_p     = (double**) malloc( sizeof(double*) * wm->n );
    wm->n2_p    = (double**) malloc( sizeof(double*) * wm->n );
    wm->uinc_p  = (double**) malloc( sizeof(double*) * wm->n );
    
    wm->sd = init_SpeedDeficit(wm);

    int i;
    for (i = 0; i < wm->n; i++) {
        wm->x_p[i]    = VEC(2);
        wm->n_p[i]    = VEC(2);
        wm->n2_p[i]   = VEC(2);
        wm->uinc_p[i] = VEC(2);
    }

    init_WakeModel_states(wm);

    // Work variables
    wm->idx_  = VECINT(2); // closest part index (find_wake_part)
    wm->widx_ = VEC(2);    // closest part weight
    wm->side_ = VEC(2);    // distance between part and wake plane proejcted along n_p

    wm->du2_     = VEC(2); // Sum of the squared local speed deficit
    wm->du2_loc_ = VEC(2); // squared local speed deficit
    
    wm->xi_ = VEC(1);      // Streamwise coordinate (project_frame)
    wm->r_  = VEC(1);      // Spanwise coordinate 
    wm->u_  = VEC(2);
    wm->v_  = VEC(2);
    wm->w_  = VEC(2);
    return wm;
}
/* -- end init_WakeModel ---------------------------------------------------- */

void init_WakeModel_set(WakeModel *wm, LagSet *set) {
    wm->set    = set;
    wm->n      = wm->set->n_wm;
    wm->n_shed = wm->set->n_shed_wm;
}
/* -- end init_WakeModel_set ------------------------------------------------ */

void init_WakeModel_states(WakeModel *wm) {
    wm->i0 = 0; // next particle updated
    wm->it = 0;
    wm->dt = wm->set->dt;

    wm->alpha_r = 1.; 

    int i;
    for (i = 0; i < wm->n; i++) {
        shed_wake_particle(wm, i);

        // Avoid particle collision at start
        wm->t_p[i]    += (i*wm->dt);
        wm->x_p[i][0] += (i*wm->dt) * (wm->set->cw+1E-3) * 1E-3;
        wm->x_p[i][1] += (i*wm->dt) * (wm->set->cw+1E-3) * 1E-3;
    }
    
    wm->alpha_r = exp(- wm->set->tau_r / (wm->n_shed * wm->dt) ) ;
}
/* -- end init_WakeModel_states --------------------------------------------- */

void free_WakeModel(WakeModel *wm) {  
    int i;
    for (i = 0; i < wm->n; i++) {
        free(wm->x_p[i]);
        free(wm->n_p[i]);
        free(wm->n2_p[i]);
        free(wm->uinc_p[i]);
    }

    free(wm->t_p);
    free(wm->xi_p);
    free(wm->ct_p);
    free(wm->ti_p);
    free(wm->x_p);
    free(wm->n_p);
    free(wm->n2_p);
    free(wm->uinc_p);
    
    FREE(SpeedDeficit, wm->sd);

    // Work variables
    free(wm->idx_);
    free(wm->widx_);
    free(wm->side_);

    free(wm->du2_);
    free(wm->du2_loc_);
    
    free(wm->xi_);
    free(wm->r_);
    free(wm->u_);
    free(wm->v_);
    free(wm->w_);
}
/* -- end free_WakeModel ---------------------------------------------------- */

void update_WakeModel(WakeModel *wm) {
    int i;
    double *sigma;
    double *u      = VEC(2);
    double *du     = VEC(2);
    double *u_t_dt = VEC(2);

    for (i = 0; i < wm->n; i++) {
        wm->t_p[i] += wm->dt;

        sigma = wm->wf->fms[0]->sigma_r;
        interp_FlowModel_all(wm->wf, wm->x_p[i], wm->t_p[i], sigma, u, -1); 
        du_part_compute_from_wf(wm->wf, wm, i, du);

        u_t_dt[0] = (u[0] - wm->set->cw * du[0]) * wm->dt;
        u_t_dt[1] = (u[1] - wm->set->cw * du[1]) * wm->dt;

        wm->x_p[i][0] += u_t_dt[0];
        wm->x_p[i][1] += u_t_dt[1];

        wm->xi_p[i] += NORM(u_t_dt[0],u_t_dt[1]);
    }

    if (wm->it%wm->n_shed==0){  wm->i0 = IP2I(wm, wm->n-1); }
    shed_wake_particle(wm, wm->i0);
    regularize_WakeModel(wm);

    wm->it++;

    free(u);
    free(du);
    free(u_t_dt);
}
/* -- end update_WakeModel -------------------------------------------------- */

/* ------------------------------------------------ */
/*               WakeModel Particles                */
/* ------------------------------------------------ */

#define FILTER(new_, old_) (1.- wm->alpha_r) * old_ +  wm->alpha_r * new_
void shed_wake_particle(WakeModel *wm, int i) {
    wm->t_p[i]   = 0.0;
    wm->xi_p[i]  = 0.0;
    wm->ct_p[i]  = wm->wt->snrs->ct;
    wm->ti_p[i]  = wm->wt->snrs->ti;
    wm->yaw_p[i] = wm->wt->snrs->yaw;

    wm->x_p[i][0]     = wm->wt->x[0];        wm->x_p[i][1]     = wm->wt->x[2];
    wm->uinc_p[i][0]  = wm->wt->snrs->u_inc; wm->uinc_p[i][1]  = wm->wt->snrs->w_inc;

    // Filtering
    int im1 = IP2I(wm, 1);
    wm->ti_p[i]       = FILTER(wm->ti_p[i]      , wm->ti_p[im1]     );  
    wm->ct_p[i]       = FILTER(wm->ct_p[i]      , wm->ct_p[im1]     );  
    wm->yaw_p[i]      = FILTER(wm->yaw_p[i]     , wm->yaw_p[im1]    );  
    wm->uinc_p[i][0]  = FILTER(wm->uinc_p[i][0] , wm->uinc_p[im1][0]);   
    wm->uinc_p[i][1]  = FILTER(wm->uinc_p[i][1] , wm->uinc_p[im1][1]);  

    // Precomputations
    wm->n_p[i][0]  = cos( wm->yaw_p[i]);       wm->n_p[i][1]  = -sin( wm->yaw_p[i]);
    wm->n2_p[i][0] = POW2WSIGN(wm->n_p[i][0]); wm->n2_p[i][1] = POW2WSIGN(wm->n_p[i][1]);

    wm->sd->update(wm, i);
}
#undef FILTER
/* -- end shed_wake_particle ------------------------------------------------ */

double side_wake_particle(WakeModel *wm, int i_p, double *x) {
    int i = IP2I(wm, i_p);
    return  wm->n_p[i][0] * ( x[0] - wm->x_p[i][0] )
          + wm->n_p[i][1] * ( x[1] - wm->x_p[i][1] ) ;
}
/* -- end side_wake_particle ------------------------------------------------ */

int is_interp_wake_particle(WakeModel *wm, int i_p, double *x, double *side) {
    side[0] = side_wake_particle(wm, i_p,   x);
    side[1] = side_wake_particle(wm, i_p+1, x);
    return ( (0.<side[0] && side[1]<0)  || (fabs(side[0]) < 1E-10) );
}
/* -- end is_interp_wake_particle ------------------------------------------- */

int interp_wake_particle(WakeModel *wm, int i_p, int *idx, double *widx, double *side) {
    idx[0] = i_p;
    idx[1] = i_p + 1;
    widx[0] = side[0]/(side[0]-side[1]);
    widx[1] = 1. - widx[0];
    return 1;
}
/* -- end interp_wake_particle ---------------------------------------------- */

#define IS_IWP(i_) is_interp_wake_particle(wm, i_, x, side)
#define IWP(i_)    interp_wake_particle(wm, i_, idx, w, side)

int find_wake_particle(WakeModel *wm, double *x, int *idx, double *w) {
    int i_up, i_low, i_mid, i_max, it;
    double *side = wm->side_;
    
    i_max = wm->n-1; 
    i_up  = wm->n-1;
    i_low = 0;  
    
    it=0;

    while (it<i_max) {
        i_mid = (i_up + i_low)/2;

        if (i_mid==i_max)  return -1;
        if (IS_IWP(i_mid)) return IWP(i_mid);
        if (i_mid==i_low)  return -1;

        if (0.0<side[0]) i_low = i_mid;
        else             i_up  = i_mid;

        it++;
    }
    return -1;
}

#undef IS_IWP
#undef IWP
/* -- end find_wake_particle ------------------------------------------------ */

void regularize_WakeModel(WakeModel *wm) {
    int i, ip1, i_p;
    double side, x_tmp, z_tmp, ct_tmp, ti_tmp, xi_tmp;
    for ( i_p = wm->i0; i_p < wm->n-1;  i_p++) {
        i   = IP2I(wm, i_p);
        ip1 = IP2I(wm, i_p+1);
        side = side_wake_particle(wm, i_p, wm->x_p[ip1]);

        if (side < 0.0) {
            x_tmp  = (wm->x_p[i][0] + wm->x_p[ip1][0] ) / 2.; 
            z_tmp  = (wm->x_p[i][1] + wm->x_p[ip1][1] ) / 2.; 
            ct_tmp = (wm->ct_p[i]   + wm->ct_p[ip1]   ) / 2.; 
            ti_tmp = (wm->ti_p[i]   + wm->ti_p[ip1]   ) / 2.; 
            xi_tmp = (wm->xi_p[i]   + wm->xi_p[ip1]   ) / 2.; 

            wm->x_p[i][0] = x_tmp;  wm->x_p[ip1][0]  = x_tmp;
            wm->x_p[i][1] = z_tmp;  wm->x_p[ip1][1]  = z_tmp;
            wm->ct_p[i]   = ct_tmp; wm->ct_p[ip1]    = ct_tmp;
            wm->ti_p[i]   = ti_tmp; wm->ti_p[ip1]    = ti_tmp;
            wm->xi_p[i]   = xi_tmp; wm->xi_p[ip1]    = xi_tmp;

            wm->sd->update(wm, i);
            wm->sd->update(wm, ip1);

            wm->x_p[ip1][0] += wm->n_p[ip1][0];
            wm->x_p[ip1][1] += wm->n_p[ip1][1];
        }
    }
}
/* -- end regularize_WakeModel ---------------------------------------------- */

void project_particle_frame_WakeModel(WakeModel *wm, int i_p, double *x, double *xi, double *r, int side) {
    int i, ip;
    double den, wu, wv, uv;
    double *u, *v, *w;

    u = wm->u_;
    v = wm->v_;
    w = wm->w_;

    i  = IP2I(wm, i_p);
    ip = IP2I(wm, i_p+side);

    u[0] = wm->n_p[i][1];                u[1] = -wm->n_p[i][0];
    v[0] = wm->x_p[ip][0]-wm->x_p[i][0]; v[1] = wm->x_p[ip][1]-wm->x_p[i][1];
    w[0] = x[0] - wm->x_p[i][0];         w[1] = x[1] - wm->x_p[i][1];

    den = sqrt(DOT(v,v));
    wv = DOT(w,v)/den;
    uv = DOT(u,v)/den;
    wu = DOT(w,u);

    den = (1. - uv*uv);
    *r  = (wu - wv*uv)/den;
    *xi = (wv - wu*uv)/den;
}
/* -- end project_particle_frame_WakeModel ------------------------------------------- */
