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
#include <time.h>
#include "macro.h"  
#include "lagSolver.h"

/* ------------------------------------------------ */
/*                    WakeModel                     */
/* ------------------------------------------------ */

WakeModel* init_WakeModel(LagSolver *wf, WindTurbine *wt) {
    WakeModel *wm = ALLOC(WakeModel);

    wm->wf = wf;
    wm->wt = wt;

    wm->d_ww = wf->d_ww[wt->i_wf]; 
    wm->d_wf = wf->d_wf[wt->i_wf]; 

    init_WakeModel_set(wm, wf->set);

    wm->t_p     = VEC(wm->n);
    wm->xi_p    = VEC(wm->n);
    wm->ct_p    = VEC(wm->n);
    wm->psi_p   = VEC(wm->n);
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

    wm->bnds = (double**) malloc( sizeof(double*) * 4 ); 
    for (i = 0; i < 4; i++) {
        wm->bnds[i] = VEC(2); 
    }

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
        wm->x_p[i][0] += (i*wm->dt) * wm->n_p[i][0] * 1E-3;
        wm->x_p[i][1] += (i*wm->dt) * wm->n_p[i][1] * 1E-3;
    }
    
    wm->alpha_r = exp(- 2 * PI * (wm->n_shed * wm->dt) /  wm->set->tau_r  ) ;
}
/* -- end init_WakeModel_states --------------------------------------------- */

void init_WakeModel_states_from_restart(WakeModel *wm, int n, int it, int i0, double *t_p, double *xi_p, double *ct_p, double *psi_p, double *x_p, double *uinc_p) {
    if (n!=wm->n)
        printf("ERROR: the number of particle is inconsistent (%i states required but restart contains %i states).\n", wm->n, n);

    wm->i0 = i0; 
    wm->it = it;
    wm->dt = wm->set->dt;

    int i;
    for (i = 0; i < wm->n; i++) {
        wm->t_p[i]  = t_p[i];
        wm->xi_p[i] = xi_p[i];
        wm->ct_p[i] = ct_p[i];
        wm->ti_p[i] = psi_p[i];
        wm->ct_p[i] = ct_p[i];

        
        wm->x_p[i][0] = x_p[i];
        wm->x_p[i][1] = x_p[i+wm->n];
        
        wm->uinc_p[i][0] = uinc_p[i];
        wm->uinc_p[i][1] = uinc_p[i+wm->n];

        wm->n_p[i][0]  = cos( wm->psi_p[i]);       wm->n_p[i][1]  = -sin( wm->psi_p[i]);
        wm->n2_p[i][0] = POW2WSIGN(wm->n_p[i][0]); wm->n2_p[i][1] = POW2WSIGN(wm->n_p[i][1]);

        wm->sd->update(wm, i);
    }
}
/* -- end init_WakeModel_states_from_restart -------------------------------- */

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

    for (i = 0; i < 4; i++) {
        free(wm->bnds[i]); 
    }
    free(wm->bnds); 

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
    double *u      = VEC(2);
    double *du     = VEC(2);
    double *u_t_dt = VEC(2);

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    double *u_f      = VEC(2);
    double *du_xi      = VEC(2);
    double *du_r      = VEC(2);
    double *du_self   = VEC(2);
    double norm, dot, du_self_1D ;

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
    interp_FlowModel_dep(wm->wf, wm->x_p[IP2I(wm,wm->n/2)], wm->t_p[IP2I(wm,wm->n/2)], 1, u_f, wm->d_wf); 
    for (i = 0; i < wm->n; i++) {
        wm->t_p[i] += wm->dt;
        
        // interpolating fields
        interp_FlowModel_dep(wm->wf, wm->x_p[i], wm->t_p[i], 0, u, wm->d_wf); 
        du_part_compute_from_wf_dep(wm->wf, wm, i, du);
        
        du_self_1D = wm->sd->du_xi(wm, i, wm->xi_p[i]);

        // Self induced speed deficit
        du_self[0] = (du_self_1D * wm->n_p[i][0] + du[0]);
        du_self[1] = (du_self_1D * wm->n_p[i][1] + du[1]); 
    
        norm   = NORM(u_f[0],u_f[1]);

        u_f[0] /= norm; 
        u_f[1] /= norm; 

        // project on centerline
        dot = DOT(du_self, u_f);
        du_xi[0] = dot * u_f[0];
        du_xi[1] = dot * u_f[1];
        
        du_r[0] = du_self[0] - du_xi[0];
        du_r[1] = du_self[1] - du_xi[1];

        u_t_dt[0] = (u[0] - wm->set->cw_r * du_r[0] - wm->set->cw_xi * du_xi[0]) * wm->dt;
        u_t_dt[1] = (u[1] - wm->set->cw_r * du_r[1] - wm->set->cw_xi * du_xi[1]) * wm->dt;

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

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    free(u_f);
    free(du_xi);
    free(du_r);
    free(du_self);
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}
/* -- end update_WakeModel -------------------------------------------------- */

/* ------------------------------------------------ */
/*               WakeModel Particles                */
/* ------------------------------------------------ */

#define FILTER(new_, old_) wm->alpha_r * old_ +  (1.- wm->alpha_r) * new_
void shed_wake_particle(WakeModel *wm, int i) {
    wm->t_p[i]   = 0.0;
    wm->xi_p[i]  = 0.0;
    wm->ct_p[i]  = wm->wt->snrs->ct;
    wm->ti_p[i]  = wm->wt->snrs->ti;
    wm->psi_p[i] = wm->wt->snrs->psi;

    wm->x_p[i][0]     = wm->wt->x[0];        wm->x_p[i][1]     = wm->wt->x[2];
    wm->uinc_p[i][0]  = wm->wt->snrs->u_inc; wm->uinc_p[i][1]  = wm->wt->snrs->w_inc;

    // Filtering
    int im1 = IP2I(wm, 1);
    wm->ti_p[i]       = FILTER(wm->ti_p[i]      , wm->ti_p[im1]     );  
    wm->ct_p[i]       = FILTER(wm->ct_p[i]      , wm->ct_p[im1]     );  
    wm->psi_p[i]      = FILTER(wm->psi_p[i]     , wm->psi_p[im1]    );  
    
    wm->uinc_p[i][0]  = FILTER(wm->uinc_p[i][0] , wm->uinc_p[im1][0]);   
    wm->uinc_p[i][1]  = FILTER(wm->uinc_p[i][1] , wm->uinc_p[im1][1]);  

    // Precomputations
    wm->n_p[i][0]  = cos( wm->psi_p[i]);       wm->n_p[i][1]  = -sin( wm->psi_p[i]);
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

void get_WakeModel_part_bnds(WakeModel *wm, int i, double sigma, double *x_edges, double *buffer) {
    buffer[0] = -wm->n_p[i][1] * 2 * sigma;
    buffer[1] =  wm->n_p[i][0] * 2 * sigma;

    x_edges[0] = wm->x_p[i][0] - buffer[0];
    x_edges[1] = wm->x_p[i][1] - buffer[1];

    x_edges[2] = wm->x_p[i][0] + buffer[0];
    x_edges[3] = wm->x_p[i][1] + buffer[1];
}
/* -- end get_part_bnds ----------------------------------------------------- */

void update_WakeModel_bounds(WakeModel *wm) {
    int i, i0, i_p ;
    double m_left, m_right, m_low_loc, m_up_loc, m_low, m_up, p_left, p_right, p_low, p_up;

    double *buffer    = VEC(2);
    double *x_0       = VEC(4);
    double *x         = VEC(4);
    double **bnds_tmp =  (double**) malloc( sizeof(double*) * 4 );

    for (i = 0; i < 4; i++) {
        bnds_tmp[i] = VEC(4);
    }

    i0 = IP2I(wm,1);
    get_WakeModel_part_bnds(wm, i0, wm->wt->af->D, x_0, buffer);
    
    // equation of up and right
    m_low =  1e16; // min 
    m_up  = -1e16; // max

    for (i_p = 1; i_p < wm->n; i_p++) {
        get_WakeModel_part_bnds(wm, IP2I(wm, i_p), wm->wt->af->D, x, buffer);

        m_low_loc = (x[1]-x_0[1])/(x[0]-x_0[0]) + 1e-6;
        m_up_loc  = (x[3]-x_0[3])/(x[2]-x_0[2]) + 1e-6;
        
        m_low = (m_low_loc > m_low) ? m_low : m_low_loc ;  
        m_up  = (m_up_loc  < m_up)  ? m_up  : m_up_loc  ;  
    }

    p_low = x_0[1] - m_low * x_0[0];
    p_up  = x_0[3] - m_up  * x_0[2];

    // equation of left and right
    
    i = IP2I(wm,1);
    m_left = -wm->n_p[i][1]/wm->n_p[i][0] + 1e-6;
    x[0] = wm->wt->x[0] + .01 * wm->wt->af->D * wm->n_p[i][0];
    x[1] = wm->wt->x[2] + .01 * wm->wt->af->D * wm->n_p[i][1];
    p_left = x[0] - x[1] * m_left;
    
    i = IP2I(wm,wm->n-1);
    m_right = -wm->n_p[i][1]/wm->n_p[i][0] + 1e-6;
    x[0] = wm->x_p[i][0] + 4 * wm->wt->af->D * wm->n_p[i][0];
    x[1] = wm->x_p[i][1] + 4 * wm->wt->af->D * wm->n_p[i][1];
    p_right = x[0] - x[1] * m_right;

    line_intersect(m_left,  p_left,  m_up,  p_up,  bnds_tmp[0]);
    line_intersect(m_right, p_right, m_up,  p_up,  bnds_tmp[1]);
    line_intersect(m_right, p_right, m_low, p_low, bnds_tmp[2]);
    line_intersect(m_left,  p_left,  m_low, p_low, bnds_tmp[3]);

    for (i = 0; i < 4; i++) {
        // wm->bnds[i][0] = .1 * bnds_tmp[i][0] + .9 * wm->bnds[i][0]; 
        // wm->bnds[i][1] = .1 * bnds_tmp[i][1] + .9 * wm->bnds[i][1];  
        wm->bnds[i][0] = bnds_tmp[i][0]; 
        wm->bnds[i][1] = bnds_tmp[i][1];             
        }

    for (i = 0; i < 4; i++) {
        free(bnds_tmp[i]); 
    }
    free(bnds_tmp); 

    free(buffer);
    free(x_0);
    free(x);
}
/* -- end update_bounds ----------------------------------------------------- */
