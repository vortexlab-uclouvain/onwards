#ifndef _LagSolver_H_
#define _LagSolver_H_

/* -------------------------------------------------------------------------- */

/* ------------------------------------------------ */
/*              STRUCTURES  DEFINITION              */
/* ------------------------------------------------ */

typedef struct LagSolver LagSolver;
typedef struct WakeModel WakeModel;
typedef struct FlowModel FlowModel;
/* -------------------------------------------------------------------------- */

typedef struct {
    int nB;
    double R;
    double Rhub;
    double Rhub_forTL;
    double D;
    double A;
    double cCTfac;
} Airfoil;
/* -------------------------------------------------------------------------- */

typedef struct {
    double u_inc;
    double u_fs;
    double w_inc;
    double w_fs;
    double ct;
    double ti;
    double yaw;
} SnsrsData;
/* -------------------------------------------------------------------------- */

typedef struct {
    int i_wf;
    int is_fs;
    double t;
    double *x;
    Airfoil *af;  
    SnsrsData *snrs;  
} WindTurbine;
/* -------------------------------------------------------------------------- */

typedef struct {
    int n_fm;
    int n_shed_fm;
    int n_wm;
    int n_shed_wm;
    int sd_type;
    double c0;
    double cw;
    double dt;
    double sigma_xi_f;    
    double sigma_r_f;
    double sigma_t_f;    
    double sigma_xi_r;    
    double sigma_r_r;
    double sigma_t_r;    
    double ak;    
    double bk;    
    double ceps;    
    double tau_r;
} LagSet;
/* -------------------------------------------------------------------------- */

typedef struct {
    int type;

    // Work variables
    int n_wv_; 
    double **wv_; 

    // Function
    void   (*update)      (WakeModel *wm, int i);
    double (*du_xi)       (WakeModel *wm, int i ,double xi);
    double (*du_xi_r)     (WakeModel *wm, int i ,double xi, double r);
    double (*du_xi_ravg)  (WakeModel *wm, int i ,double xi, double ravg);

    // Others
    Airfoil *af;
} SpeedDeficit;
/* -------------------------------------------------------------------------- */

struct FlowModel {
    int i0; // next particle updated
    int it;

    int n;
    int n_shed;

    double dt;

    double *t_p;
    double *xi_p;
    double **x_p;
    double **u_p;
    double **uf_p;

    // Opaque part of the structure
    double *sigma_r;
    double *sigma_f;

    double *w_shep;
    double *bnds;

    double *x_p_loc_;
    double *xi_;
    double *r_;

    LagSolver *wf;
    WindTurbine *wt;

    LagSet *set;
};
/* -------------------------------------------------------------------------- */

struct WakeModel {
    int i0; // next particle updated
    int it;

    int n;
    int n_shed;

    double dt;

    double *t_p;
    double *xi_p;
    double *ct_p;
    double *ti_p;
    double *yaw_p;
    double **x_p;
    double **uinc_p;

    // Opaque part of the structure

    double alpha_r;

    double **n_p;
    double **n2_p;

    int *idx_;
    double *widx_;
    double *side_;

    double *du2_;
    double *du2_loc_;

    double *xi_;
    double *r_;
    double *u_;
    double *v_;
    double *w_;

    LagSolver *wf;
    WindTurbine *wt;
    SpeedDeficit *sd;

    LagSet *set;
};
/* -------------------------------------------------------------------------- */

struct LagSolver {
    int n_wt;
    double t;

    LagSet *set;
    
    // Wind Turbines mapping
    WindTurbine  **wts;
    FlowModel    **fms;
    WakeModel    **wms;
};
/* -------------------------------------------------------------------------- */

/* ------------------------------------------------ */
/*               FUNCTIONS DEFINITION               */
/* ------------------------------------------------ */

// Memory Allocation and Structure Initialization       
LagSolver* init_LagSolver(int n_wt, LagSet *set);
void reset_LagSolver(LagSolver *wf);
void free_LagSolver(LagSolver *wf);

FlowModel* init_FlowModel(LagSolver *wf, WindTurbine *wt);
void init_FlowModel_set(FlowModel *fm, LagSet *set);
void init_FlowModel_states(FlowModel *fm);
void free_FlowModel(FlowModel *fm);

WakeModel* init_WakeModel(LagSolver *wf, WindTurbine *wt);
void init_WakeModel_set(WakeModel *wf, LagSet *set);
void init_WakeModel_states(WakeModel *wf);
void free_WakeModel(WakeModel *fm);

SpeedDeficit* init_SpeedDeficit(WakeModel *wm);
void free_SpeedDeficit(SpeedDeficit *sd);

// Flow and Wake Model getters
FlowModel* get_FlowModel(LagSolver *wf, WindTurbine *wt);
WakeModel* get_WakeModel(LagSolver *wf, WindTurbine *wt);

// Wind turbines function
void add_WindTurbine(LagSolver *wf, WindTurbine *wt);
void is_freestream(LagSolver *wf,  WindTurbine *wt);
int is_waked_by(WakeModel *wm, WindTurbine *wt);

// State update
void update_FlowModel(FlowModel *fm);
void update_WakeModel(WakeModel *wm);

// Particle shedding
void shed_vel_particle(FlowModel *fm, int i);
void shed_wake_particle(WakeModel *wm, int i);

// Flow particle interpolation
void project_particle_frame_FlowModel(FlowModel *fm, int i, double *x, double *xi, double *r);
double compute_weight_FlowModel_all(LagSolver *wf, double *x, double* sigma, int i_wt_exclude);
double compute_weight_FlowModel(FlowModel *fm, double *x, double* sigma);
void interp_FlowModel_all(LagSolver *fm, double *x, double t, double* sigma, double *u_interp, int i_wt_exclude);

// Wake particle interpolation
void project_particle_frame_WakeModel(WakeModel *wm, int i_p, double *x, double *xi, double *r, int side);
double side_wake_particle(WakeModel *wm, int i_p, double *x);
int is_interp_wake_particle(WakeModel *wm, int i_p, double *x, double *side);
int interp_wake_particle(WakeModel *wm, int i_p, int *idx, double *widx, double *side);
int find_wake_particle(WakeModel *wm, double *x, int *idx, double *w);

// Speed Deficit interpolation
void regularize_WakeModel(WakeModel *wm);

void du_pos_compute_from_wf(LagSolver *wf, double *x, double *du_interp, 
                                   double (*proj)(WakeModel*, double, double*));
void du2_pos_compute_from_wm(WakeModel *wm, double *x, double *du_interp, 
                                   double (*proj)(WakeModel*, double, double*));

void du_pos2d_compute_from_wf(LagSolver *wf, double *x, double *du_interp);
void du_pos3d_compute_from_wf(LagSolver *wf, double *x, double *du_interp);

void du_part_compute_from_wf(LagSolver *wf, WakeModel *wm_p, int i_p, double *du_interp);
void du_part_compute_from_wm(WakeModel *wm, double *x, double *du_interp, double ravg);

void du_ravg_pos_compute_from_wf(LagSolver *wf, double *x, double *du_interp, double ravg);

double rews_compute(LagSolver *wf, double *x_c_vec_rotor, double r_rotor);

// # TYPE 1 :  Bastankhah 
// M. Bastankhah and F. Port ́e-Agel. A new analytical model for wind-turbine wakes.
// Renewable Energy, 70:116–123, 2014.
double du_xi_BPA(WakeModel *wm, int i, double xi);
double du_xi_ravg_BPA(WakeModel *wm, int i, double xi, double ravg);
double du_xi_r_BPA(WakeModel *wm, int i, double xi, double r);
void update_BPA(WakeModel *wm, int i);

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

#endif // _LagSolver_H_