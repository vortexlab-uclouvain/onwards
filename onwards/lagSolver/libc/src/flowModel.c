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
/*                    FlowModel                     */
/* ------------------------------------------------ */

FlowModel* init_FlowModel(LagSolver *wf, WindTurbine *wt) {
    FlowModel *fm = ALLOC(FlowModel);

    fm->wf = wf;
    fm->wt = wt;

    fm->d_ff = wf->d_ff[wt->i_wf]; 
    fm->d_wf = wf->d_wf[wt->i_wf]; 

    fm->sigma_r = VEC(3);
    fm->sigma_f = VEC(3);

    fm->sigma = (double**) malloc( sizeof(double*) * 2 );
    fm->sigma[0] = fm->sigma_r;
    fm->sigma[1] = fm->sigma_f;

    init_FlowModel_set(fm, wf->set);

    fm->t_p  = VEC(fm->n);
    fm->xi_p = VEC(fm->n);
    fm->x_p  = (double**) malloc( sizeof(double*) * fm->n );
    fm->u_p  = (double**) malloc( sizeof(double*) * fm->n );
    fm->uf_p = (double**) malloc( sizeof(double*) * fm->n );
    
    int i, j;
    for (i = 0; i < fm->n; i++) {
        fm->x_p[i]  = VEC(2);
        fm->u_p[i]  = VEC(2);
        fm->uf_p[i] = VEC(2);
    }

    init_FlowModel_states(fm);

    // Work variables

    fm->w_shep   = VEC(fm->n);

    fm->bnds = (double***) malloc( sizeof(double**) * 2 ); 
    for (i = 0; i < 2; i++) {
        fm->bnds[i] = (double**) malloc( sizeof(double*) * 4 ); 
        for (j = 0; j < 4; j++) {
            fm->bnds[i][j] = VEC(2); 
        }
    }

    fm->bnds[0][0][0] = -1e4;  fm->bnds[1][0][0] = -1e4;
    fm->bnds[0][0][1] =  1e4;  fm->bnds[1][0][1] =  1e4;
    fm->bnds[0][1][0] =  1e4;  fm->bnds[1][1][0] =  1e4;
    fm->bnds[0][1][1] =  1e4;  fm->bnds[1][1][1] =  1e4;
    fm->bnds[0][2][0] =  1e4;  fm->bnds[1][2][0] =  1e4;
    fm->bnds[0][2][1] = -1e4;  fm->bnds[1][2][1] = -1e4;
    fm->bnds[0][3][0] = -1e4;  fm->bnds[1][3][0] = -1e4;
    fm->bnds[0][3][1] = -1e4;  fm->bnds[1][3][1] = -1e3;

    fm->x_p_loc_ = VEC(2); 
    fm->xi_      = VEC(1); // Streamwise coordinate (project_frame)
    fm->r_       = VEC(1); // Spanwise coordinate   (project_frame)

    return fm;
}
/* -- end init_FlowModel ---------------------------------------------------- */

void init_FlowModel_set(FlowModel *fm, LagSet *set) {
    fm->set    = set;
    fm->n      = fm->set->n_fm;
    fm->n_shed = fm->set->n_shed_fm;

    // wm->set updated externally
    fm->n_shed = fm->set->n_shed_fm;

    fm->sigma_r[0] = fm->wf->wts[0]->af->D * fm->set->sigma_xi_r;
    fm->sigma_r[1] = fm->wf->wts[0]->af->D * fm->set->sigma_r_r;
    fm->sigma_r[2] = fm->wf->wts[0]->af->D * fm->set->sigma_t_r;

    fm->sigma_f[0] = fm->wf->wts[0]->af->D * fm->set->sigma_xi_f;
    fm->sigma_f[1] = fm->wf->wts[0]->af->D * fm->set->sigma_r_f;
    fm->sigma_f[2] = fm->wf->wts[0]->af->D * fm->set->sigma_t_f;
}
/* -- end init_FlowModel_set ------------------------------------------------ */

void init_FlowModel_states(FlowModel *fm) {
    fm->i0 = 0; // next particle updated
    fm->it = 0;
    fm->dt = fm->set->dt;

    int i;
    for (i = 0; i < fm->n; i++) {
        shed_vel_particle(fm, i);

        // Avoid particle collision at start
        fm->t_p[i]    += (i*fm->dt*fm->n_shed);
        fm->x_p[i][0] += (i*fm->dt*fm->n_shed) * (1E-3) * fm->u_p[i][0];
        fm->x_p[i][1] += (i*fm->dt*fm->n_shed) * (1E-3) * fm->u_p[i][1];
        fm->xi_p[i]    = NORM(fm->x_p[i][0], fm->x_p[i][1]);
    }
}
/* -- end init_FlowModel_states --------------------------------------------- */

void free_FlowModel(FlowModel *fm) {
    int i,j;
    
    for (i = 0; i < fm->n; i++) {
        free(fm->x_p[i]);
        free(fm->u_p[i]);
        free(fm->uf_p[i]);
    }
    free(fm->t_p);
    free(fm->xi_p);
    free(fm->x_p);
    free(fm->u_p);
    free(fm->uf_p);

    free(fm->sigma_r);
    free(fm->sigma_f);
    free(fm->sigma);

    // Work variables
    free(fm->w_shep);

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            free(fm->bnds[i][j]); 
        }
        free(fm->bnds[i]); 
    }
    free(fm->bnds); 

    free(fm->x_p_loc_);
    free(fm->xi_);
    free(fm->r_);
}
/* -- end free_FlowModel ---------------------------------------------------- */

void update_FlowModel(FlowModel *fm) { 
    int i;
    double *du     = VEC(2);
    double *u_t_dt = VEC(2);

    for (i = 0; i < fm->n; i++) {
        du_ravg_posf_compute_from_wf(fm->wf, fm->wt->i_wf, fm->x_p[i], du, fm->wt->af->R);
        
        u_t_dt[0] = (fm->uf_p[i][0] - fm->set->c0 * du[0]) * fm->dt;
        u_t_dt[1] = (fm->uf_p[i][1] - fm->set->c0 * du[1]) * fm->dt;
        // printf("%.2f\n", fm->set->c0);

        fm->t_p[i] += fm->dt;

        fm->x_p[i][0] += u_t_dt[0];
        fm->x_p[i][1] += u_t_dt[1];

        fm->xi_p[i] += NORM(u_t_dt[0], u_t_dt[1]);
    }

    if (fm->it%fm->n_shed==0){  
        // Update filtered field
        for (i = 0; i < fm->n; i++) {
            interp_FlowModel_all(fm->wf, fm->x_p[i], fm->wt->t, 1, fm->uf_p[i], -1);
        }
        fm->i0 = IP2I(fm, fm->n-1); 
    }

    shed_vel_particle(fm, fm->i0);

    fm->it++;

    free(u_t_dt);
}

/* ------------------------------------------------ */
/*                  Interpolation                   */
/* ------------------------------------------------ */

void project_particle_frame_FlowModel(FlowModel *fm, int i, double *x, double *xi, double *r) {
        double norm;
        double *x_p = fm->x_p_loc_;
        double *v   = fm->uf_p[i];

        x_p[0] = fm->x_p[i][0] - x[0];
        x_p[1] = fm->x_p[i][1] - x[1];

        norm = NORM(v[0],v[1]);
        norm = norm < 1E-3 ? 1E-3 : norm;
        xi[0] = (x_p[0]*v[0]+x_p[1]*v[1])/norm;
        r[0]  = (x_p[1]*v[0]-x_p[0]*v[1])/norm;
}
/* -- end project_particle_frame_FlowModel ---------------------------------- */

int in_bnds_FlowModel(FlowModel *fm, double *x, int i_sigma) {
    return in_quad(x, fm->bnds[i_sigma][0], fm->bnds[i_sigma][1], fm->bnds[i_sigma][2], fm->bnds[i_sigma][3]);
}
/* -- end in_bnds_FlowModel ------------------------------------------------- */

double compute_weight_FlowModel(FlowModel *fm, double *x, int skip, double *sigma) {
    int i;
    double w_skip, w_acc;
    w_acc = 0;

    w_skip = fm->n/floor(fm->n/skip);
    
    for ( i = 0; i < fm->n; i+=skip) { 
        project_particle_frame_FlowModel(fm, i, x, fm->xi_, fm->r_);

        fm->w_shep[i] =  ( exp( (- pow(fm->xi_[0]/sigma[0], 2)
                                 - pow( fm->r_[0]/sigma[1], 2) 
                                 - pow(fm->t_p[i]/sigma[2], 2))/2.  ) ) ;
        fm->w_shep[i] += 1e-16;
        fm->w_shep[i] *= w_skip;
        w_acc += fm->w_shep[i];
    } 

    return w_acc;
}
/* -- end compute_weight ---------------------------------------------------- */

void interp_FlowModel(FlowModel *fm, double *u_interp, int skip) {
    int i;
    for ( i = 0; i < fm->n; i+=skip) { 
        u_interp[0] += fm->w_shep[i] * fm->u_p[i][0];
        u_interp[1] += fm->w_shep[i] * fm->u_p[i][1];
    } 
}
/* -- end interp_FlowModel -------------------------------------------------- */

int DEP_SKIP = 5;

void interp_FlowModel_all(LagSolver *wf, double *x, double t, int i_sigma, double *u_interp, int i_wt_exclude) {
    int i_wt, skip;
    double w_acc;
    FlowModel *fm;
    w_acc = 0;

    u_interp[0] = 0.0;
    u_interp[1] = 0.0;

    for (i_wt = 0;                i_wt < i_wt_exclude; i_wt++) { 
        fm = wf->fms[i_wt];
        skip   = in_bnds_FlowModel(fm, x, i_sigma) ? 1 :DEP_SKIP;
        w_acc += compute_weight_FlowModel(fm, x, skip, fm->sigma[i_sigma]) ;
        interp_FlowModel(fm, u_interp, skip);
    }
    for (i_wt = i_wt_exclude + 1; i_wt < wf->n_wt;     i_wt++) { 
        fm = wf->fms[i_wt];
        skip   = in_bnds_FlowModel(fm, x, i_sigma) ? 1 :DEP_SKIP;
        w_acc += compute_weight_FlowModel(fm, x, skip, fm->sigma[i_sigma]) ;
        interp_FlowModel(fm, u_interp, skip);
    }

    u_interp[0] /= w_acc;
    u_interp[1] /= w_acc;
}
/* -- end interp_FlowModel_all ---------------------------------------------- */

void interp_FlowModel_dep(LagSolver *wf, double *x, double t, int i_sigma, double *u_interp, int *d) {
    int i_wt;
    double w_acc, skip;
    FlowModel *fm;
    w_acc = 0;

    u_interp[0] = 0.0;
    u_interp[1] = 0.0;

    for (i_wt = 0; i_wt < wf->n_wt; i_wt++) { 
        fm = wf->fms[i_wt];
        skip   = d[i_wt] ? 1 : DEP_SKIP;
        w_acc += compute_weight_FlowModel(fm, x, skip, fm->sigma[i_sigma]);
        interp_FlowModel(wf->fms[i_wt], u_interp, skip);
    }

    u_interp[0] /= w_acc;
    u_interp[1] /= w_acc;
}
/* -- end interp_FlowModel_all ---------------------------------------------- */

void get_FloWModel_part_bnds(FlowModel *fm, int i, double *sigma, double *x_edges, double *buffer) {
    double norm = NORM(fm->uf_p[i][0],fm->uf_p[i][1]);
    buffer[0] = -fm->uf_p[i][1]/norm * 3 * sigma[1];
    buffer[1] =  fm->uf_p[i][0]/norm * 3 * sigma[1];

    x_edges[0] = fm->x_p[i][0] - buffer[0];
    x_edges[1] = fm->x_p[i][1] - buffer[1];

    x_edges[2] = fm->x_p[i][0] + buffer[0];
    x_edges[3] = fm->x_p[i][1] + buffer[1];
}
/* -- end get_part_bnds ----------------------------------------------------- */

void update_FlowModel_bounds(FlowModel *fm) {
    int i, i0, i_p, i_sigma;
    double norm, m_left, m_right, m_low_loc, m_up_loc, m_low, m_up, p_left, p_right, p_low, p_up;
    
    double *sigma;

    double *buffer    = VEC(2);
    double *x_0       = VEC(4);
    double *x         = VEC(4);
    double **bnds_tmp =  (double**) malloc( sizeof(double*) * 4 );

    for (i = 0; i < 4; i++) {
        bnds_tmp[i] = VEC(4);
    }

    for (i_sigma = 0; i_sigma < 2; i_sigma++) {

        sigma = i_sigma==0 ? fm->sigma_r : fm->sigma_f; 
        
        i0 = IP2I(fm,1);
        get_FloWModel_part_bnds(fm, i0, sigma, x_0, buffer);
        
        // equation of up and right
        m_low =  1e16; // min 
        m_up  = -1e16; // max

        for (i_p = 1; i_p < fm->n; i_p++) {
            get_FloWModel_part_bnds(fm, IP2I(fm, i_p), sigma, x, buffer);

            m_low_loc = (x[1]-x_0[1])/(x[0]-x_0[0]) + 1e-6;
            m_up_loc  = (x[3]-x_0[3])/(x[2]-x_0[2]) + 1e-6;
            
            m_low = (m_low_loc > m_low) ? m_low : m_low_loc ;  
            m_up  = (m_up_loc  < m_up)  ? m_up  : m_up_loc  ;  
        }

        p_low = x_0[1] - m_low * x_0[0];
        p_up  = x_0[3] - m_low * x_0[2];

        // equation of left and right
        
        i = IP2I(fm,1);
        m_left = -fm->uf_p[i][1]/fm->uf_p[i][0] + 1e-6;
        norm = NORM(fm->uf_p[i][0],fm->uf_p[i][1]);
        x[0] = fm->wt->x[0] - 3 * sigma[0] * fm->uf_p[i][0]/norm;
        x[1] = fm->wt->x[2] - 3 * sigma[0] * fm->uf_p[i][1]/norm;
        p_left = x[0] - x[1] * m_left;
        
        i = IP2I(fm,fm->n-1);
        m_right = -fm->uf_p[i][1]/fm->uf_p[i][0] + 1e-6;
        norm = NORM(fm->uf_p[i][0],fm->uf_p[i][1]);
        x[0] = fm->x_p[i][0] + 3 * sigma[0] * fm->uf_p[i][0]/norm;
        x[1] = fm->x_p[i][1] + 3 * sigma[0] * fm->uf_p[i][1]/norm;
        p_right = x[0] - x[1] * m_right;

        line_intersect(m_left,  p_left,  m_up,  p_up,  bnds_tmp[0]);
        line_intersect(m_right, p_right, m_up,  p_up,  bnds_tmp[1]);
        line_intersect(m_right, p_right, m_low, p_low, bnds_tmp[2]);
        line_intersect(m_left,  p_left,  m_low, p_low, bnds_tmp[3]);

        for (i = 0; i < 4; i++) {
            fm->bnds[i_sigma][i][0] = .1 * bnds_tmp[i][0] + .9 * fm->bnds[i_sigma][i][0]; 
            fm->bnds[i_sigma][i][1] = .1 * bnds_tmp[i][1] + .9 * fm->bnds[i_sigma][i][1];             
            }
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

/* ------------------------------------------------ */
/*               FlowModel Particles                */
/* ------------------------------------------------ */

void shed_vel_particle(FlowModel *fm, int i) {
    fm->xi_p[i] = 0.0;
    fm->t_p[i]  = 0.0;
    
    fm->x_p[i][0]  = fm->wt->x[0];        
    fm->x_p[i][1]  = fm->wt->x[2];  
    
    fm->u_p[i][0]  = fm->wt->snrs->u_fs * cos(fm->wt->snrs->psi)
                   + fm->wt->snrs->w_fs * sin(fm->wt->snrs->psi);
    fm->u_p[i][1]  = fm->wt->snrs->w_fs * cos(fm->wt->snrs->psi)
                   - fm->wt->snrs->u_fs * sin(fm->wt->snrs->psi);
    
    fm->uf_p[i][0]  = fm->u_p[i][0] ;
    fm->uf_p[i][1]  = fm->u_p[i][1];
}

/* -- end shed_vel_particle ------------------------------------------------- */
