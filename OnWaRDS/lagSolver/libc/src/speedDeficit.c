#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "airfoilLib.h"
#include "macro.h"  
#include "lagSolver.h"

/* ------------------------------------------------ */
/*                   SpeedDeficit                   */
/* ------------------------------------------------ */

SpeedDeficit* init_SpeedDeficit(WakeModel *wm) {
    int i;

    SpeedDeficit *sd = ALLOC(SpeedDeficit);

    switch (wm->set->sd_type)
    {
    case 0: // BPA model
        sd->n_wv_      = 4;
        sd->update     = update_BPA;
        sd->du_xi      = du_xi_BPA;
        sd->du_xi_r    = du_xi_r_BPA;
        sd->du_xi_ravg = du_xi_ravg_BPA;

    // case 1: // User Define Model
        // sd->n_wv_      = 42;
        // sd->update     = update_BPA;
        // sd->du_xi      = du_xi_BPA;
        // sd->du_xi_r    = du_xi_r_BPA;
        // sd->du_xi_ravg = du_xi_ravg_BPA;
    }

    sd->wv_   = ((double**) malloc(sizeof(double*)*sd->n_wv_));
    for (i = 0; i < sd->n_wv_; i++)
    {
        sd->wv_[i] = VEC(wm->n);
    }

    return sd;
}
/* -- end init_SpeedDeficit ------------------------------------------------- */

#define KSTR wm->sd->wv_[0]
#define NW   wm->sd->wv_[1]
#define EPS  wm->sd->wv_[2]
#define X0   wm->sd->wv_[3]

void update_BPA(WakeModel *wm, int i) {
    KSTR[i] = (wm->set->ak + wm->set->bk * wm->ti_p[i]);
    NW[i]   = sqrt( 1.- wm->ct_p[i] );
    EPS[i]  = wm->set->ceps * sqrt(.5  * ( 1. + NW[i] ) / NW[i]);
    X0[i]   = (0.3535533906 - EPS[i])/KSTR[i] * wm->wt->af->D;
}
/* -- end update_BPA -------------------------------------------------------- */

double du_xi_BPA(WakeModel *wm, int i, double xi) {

    double sig_o_d_sqr = pow( KSTR[i] * xi/wm->wt->af->D + EPS[i], 2);
    double rad = 1 - wm->ct_p[i] / (8.*sig_o_d_sqr);

    return wm->uinc_p[i][0] * ( 1. - 
                                (xi< X0[i]) * NW[i] -             // potential core
                                (xi>=X0[i]) * sqrt(rad * (rad>0)) // far wake  
                            ); 
}
/* -- end du_xi_BPA --------------------------------------------------------- */

double du_xi_r_BPA(WakeModel *wm, int i, double xi, double r) {
    if (fabs(r) > wm->wt->af->R*(1. - xi/X0[i]) * (xi < X0[i])) {    
        
        double sig_o_d_sqr = pow( KSTR[i] * xi/wm->wt->af->D + EPS[i], 2);
        double rad = 1 - wm->ct_p[i] / (8.*sig_o_d_sqr);

        double fac = (1 - sqrt(rad * (rad>0)))
                         * exp( -1./(2*sig_o_d_sqr) * pow(r/wm->wt->af->D,2.) );

        return wm->uinc_p[i][0] * (    (fac <(1 - NW[i])) * fac 
                                    +  (fac>=(1 - NW[i])) * (1 - NW[i]) ); 
    }
    else {
        return wm->uinc_p[i][0] * (1 - NW[i]);
    }
}
/* -- end du_xi_r_BPA ------------------------------------------------------- */

double du_xi_ravg_BPA(WakeModel *wm, int i, double xi, double ravg) {

    double sig_o_d_sqr = KSTR[i] * xi/wm->wt->af->D + EPS[i];
    ravg = ravg > 0 ? (1.3863 * sig_o_d_sqr * wm->wt->af->D * ravg) : -ravg*63. ;

    sig_o_d_sqr = pow( sig_o_d_sqr, 2);

    double r0   = wm->wt->af->R*(1. - xi/X0[i]) * (xi < X0[i]);

    if (r0<ravg) {
        double rad = 1 - wm->ct_p[i] / (8.*sig_o_d_sqr);

        double u_ring   = (1. - sqrt(rad * (rad>0)) ) * (
                                     exp( -1./(2*sig_o_d_sqr) * pow(r0  /wm->wt->af->D,2.) )
                                   - exp( -1./(2*sig_o_d_sqr) * pow(ravg/wm->wt->af->D,2.) ) 
                              ) * sig_o_d_sqr  * wm->wt->af->D * wm->wt->af->D; 

        double u_center = (1. - NW[i]) * r0 * r0 ;
        return wm->uinc_p[i][0] * (u_ring + u_center)/ravg/ravg ; 
    }
    return wm->uinc_p[i][0]*(1. - NW[i]);
}
/* -- end du_xi_ravg_BPA ---------------------------------------------------- */

#undef KSTR
#undef NW
#undef EPS
#undef X0

/* ------------------------------------------------ */
/*     Speed Deficit Evaluation with Projection     */
/* ------------------------------------------------ */

void du2_part_compute_from_wm(WakeModel *wm, double *x, double *du_interp, double ravg) {
    int *idx_p = wm->idx_;
    double *w_idx_p = wm->widx_;

    if (0 < side_wake_particle(wm, wm->n, x)) { 
        int in_wake = find_wake_particle(wm, x, idx_p, w_idx_p);
        if (in_wake != -1) {
            int i, side;

            double * xi  = wm->xi_;
            double * r   = wm->r_;
            double du2 = 0.0, du = 0.0;

            for (int i_w = 0; i_w < 2; i_w++) {
                i = IP2I(wm,idx_p[i_w]);
                side = (i_w*2-1)*-1;
                project_particle_frame(wm, idx_p[i_w], x, xi, r, side);

                du  = wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r         )
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r+.25*ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r-.25*ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r+.50*ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r-.50*ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r+.75*ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r-.75*ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r+    ravg)
                    + wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r-    ravg);
                
                du2 += du * (1.-w_idx_p[i_w])/9.;
            }
            du2 = POW2WSIGN(du2);

            du_interp[0] += du2 * wm->n2_p[i][0];
            du_interp[1] += du2 * wm->n2_p[i][1];
        }
    }
}
/* -- end du_part_compute_from_wm ------------------------------------------s- */

void du2_pos_compute_from_wm(WakeModel *wm, double *x, double *du_interp, 
                                    double (*proj)(WakeModel*, double, double*)) {
    int *idx_p = wm->idx_;
    double *w_idx_p = wm->widx_;

    int in_wake = find_wake_particle(wm, x, idx_p, w_idx_p);
    if (in_wake != -1) {
        int i, side;

        double   du2 = 0.0;
        double * xi  = wm->xi_;
        double * r   = wm->r_;

        for (int i_w = 0; i_w < 2; i_w++) {
            i = IP2I(wm,idx_p[i_w]);
            side = (i_w*2-1)*-1;
            project_particle_frame(wm, idx_p[i_w], x, xi, r, side);
            *r = (*proj)(wm,*r,x);
            du2 += wm->sd->du_xi_r(wm, i, wm->xi_p[i] + *xi*side, *r) * (1.-w_idx_p[i_w]);
        }
        
        du2 = POW2WSIGN(du2);
        du_interp[0] = du2 * wm->n2_p[i][0];
        du_interp[1] = du2 * wm->n2_p[i][1];
    }
    else {
        du_interp[0] = 0.0;
        du_interp[1] = 0.0;
    }
}
/* -- end du_part_compute_from_wm ------------------------------------------s- */

void du_part_compute_from_wf(LagSolver *wf, WakeModel *wm_p, int i_p, double *du_interp) {
    
    double ravg = wm_p->wt->af->R;

    double du2_norm = pow( wm_p->sd->du_xi(wm_p, i_p, 0.0) , 2);

    // Self induced speed deficit
    du_interp[0] = (du2_norm * wm_p->n2_p[i_p][0]);
    du_interp[1] = (du2_norm * wm_p->n2_p[i_p][1]); 

    // Deficit induced by other wakes 
    int i_wm, i_wm_p;
    i_wm_p = wm_p->wt->i_wf;

    for ( i_wm = 0; i_wm < i_wm_p; i_wm++) {
        du2_part_compute_from_wm(wf->wms[i_wm], wm_p->x_p[i_p], du_interp, ravg);
    }
    for ( i_wm = i_wm_p+1; i_wm < wf->n_wt ; i_wm++) {
        du2_part_compute_from_wm(wf->wms[i_wm], wm_p->x_p[i_p], du_interp, ravg);     
    }
    
    du_interp[0] = SQRTWSIGN(du_interp[0]);
    du_interp[1] = SQRTWSIGN(du_interp[1]);    
}
/* -- end du_pos_compute_from_wf -------------------------------------------- */

void du_pos_compute_from_wf(LagSolver *wf, double *x, double *du_interp, 
                                    double (*proj)(WakeModel*, double, double*)) {
    double* du2     = wf->wms[0]->du2_;
    double* du2_loc = wf->wms[0]->du2_loc_;

    du2[0] = 0.0;
    du2[1] = 0.0;

    int i_wm; 
    for ( i_wm = 0; i_wm < wf->n_wt ; i_wm++) {
        du2_pos_compute_from_wm(wf->wms[i_wm], x, du2_loc, proj);
        du2[0] += du2_loc[0]; 
        du2[1] += du2_loc[1];
    }

    du_interp[0] = SQRTWSIGN(du2[0]);
    du_interp[1] = SQRTWSIGN(du2[1]);
}
/* -- end du_pos_compute_from_wf -------------------------------------------- */

void du_ravg_pos_compute_from_wf(LagSolver *wf, double *x, double *du_interp, double ravg) {

    du_interp[0] = 0.0;
    du_interp[1] = 0.0; 

    // Deficit induced by other wakes 
    int i_wm;
    for ( i_wm = 0; i_wm < wf->n_wt; i_wm++) {
        du2_part_compute_from_wm(wf->wms[i_wm], x, du_interp, ravg);
    }

    du_interp[0] = SQRTWSIGN(du_interp[0]);
    du_interp[1] = SQRTWSIGN(du_interp[1]);    
}
/* -- end du_ravg_pos_compute_from_wf -------------------------------------------- */

double proj_2d(WakeModel *wm, double r_z, double *x) {
    return r_z; 
}

void du_pos2d_compute_from_wf(LagSolver *wf, double *x_vec, double *du_interp) {
    du_pos_compute_from_wf(wf, x_vec, du_interp, proj_2d);
}
/* -- end proj_2d ----------------------------------------------------------- */

double proj_3d(WakeModel *wm, double r_z, double *x) {
    double r_y = x[2] - wm->wt->x[1]; 
    return NORM(r_z, r_y); 
}

void du_pos3d_compute_from_wf(LagSolver *wf, double *x_vec, double *du_interp) {
    du_pos_compute_from_wf(wf, x_vec, du_interp, proj_3d);
}
/* -- end proj_3d ----------------------------------------------------------- */


void ueff_xyz(LagSolver *wf, double *x_vec, double *u, double *du) {
    interp_FlowModel_all(wf, x_vec, wf->wts[0]->t, wf->fms[0]->sigma_f, u, -1);
    du_pos3d_compute_from_wf(wf, x_vec, du);
    
    u[0] -= du[0];
    u[1] -= du[1];
}
/* -- end ueff_xyz ---------------------------------------------------------- */

int    N_INT     =  16;
double R_INT[16] = {0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151, 0.4597008433809831, 0.8880738339771151};
double T_INT[16] = {0.0000000000000000, 0.3926990816987241, 0.7853981633974483, 1.1780972450961724, 1.5707963267948966, 1.9634954084936207, 2.356194490192345, 2.748893571891069, 3.141592653589793, 3.5342917352885173, 3.9269908169872414, 4.319689898685965, 4.71238898038469, 5.105088062083414, 5.497787143782138, 5.890486225480862};
double W_INT[16] = {0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625};

double rews_compute(LagSolver *wf, double *x_c_vec_rotor, double r_rotor) {

    int i;

    double *u      = VEC(2);
    double *du     = VEC(2);
    double *x_eval = VEC(3);
    
    double sum = 0.0;
    double r;

    for (i = 0; i < N_INT; i++) {
        r = R_INT[i]*r_rotor;

        x_eval[0] = x_c_vec_rotor[0] ;
        x_eval[1] = x_c_vec_rotor[2] + r*cos(T_INT[i]);
        x_eval[2] = x_c_vec_rotor[1] + r*sin(T_INT[i]);
        
        ueff_xyz(wf, x_eval, u, du);

        sum +=  u[0] * W_INT[i];
    }

    free(u);
    free(du);
    free(x_eval);

    return sum;

}
/* -- end rews_compute ------------------------------------------------------ */

int is_waked_by(WakeModel *wm, WindTurbine *wt) {
    // double *side = wm->side_;

    double *du = wm->du2_;

    double *x_wt = VEC(3);
    x_wt[0] = wt->x[0];
    x_wt[1] = wt->x[2];
    x_wt[2] = wt->x[1];

    du2_pos_compute_from_wm( wm, x_wt, du, proj_3d);
    double du_loc = NORM(du[0], du[1]);

    return ( du_loc/wt->snrs->u_inc > 0.001 );
}

/* -- end is_waked_by ------------------------------------------- */