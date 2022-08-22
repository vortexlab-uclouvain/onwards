#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "airfoilLib.h"
#include "macro.h"  
#include "lagSolver.h"

/* ------------------------------------------------ */
/*           LagSolver (FastFarmFlow model)          */
/* ------------------------------------------------ */

LagSolver* init_LagSolver(int n_wt, LagSet *set) {
    LagSolver *wf = ALLOC(LagSolver);

    wf->n_wt = n_wt;

    wf->set  = set;

    wf->wts = (WindTurbine**)  malloc( sizeof(WindTurbine*) * wf->n_wt ) ;
    wf->fms = (FlowModel**)    malloc( sizeof(FlowModel*)   * wf->n_wt ) ;
    wf->wms = (WakeModel**)    malloc( sizeof(WakeModel*)   * wf->n_wt ) ;
    
    return wf;
};
/* -- end init_LagSolver ----------------------------------------------------- */

void free_LagSolver(LagSolver *wf) { 
    int i_wt;
    for (i_wt = 0; i_wt < wf->n_wt; i_wt++) {
        FREE(FlowModel, wf->fms[i_wt]);
        FREE(WakeModel, wf->wms[i_wt]);
    }
    free(wf->wts);
    free(wf->fms);
    free(wf->wms);
    free(wf);
};
/* -- end free_LagSolver ----------------------------------------------------- */

void update_LagSolver(LagSolver *wf) {
    int i_wt;
    for (i_wt = 0; i_wt < wf->n_wt; i_wt++) {
        is_freestream(wf, wf->wts[i_wt]);
        update_FlowModel(wf->fms[i_wt]);
        update_WakeModel(wf->wms[i_wt]);
    }
    wf->t += wf->set->dt; 
};
/* -- end update_LagSolver --------------------------------------------------- */

void add_WindTurbine(LagSolver *wf, WindTurbine *wt) {
    wf->wts[wt->i_wf] = wt ;
    wf->wms[wt->i_wf] = init_WakeModel(wf, wt);
    wf->fms[wt->i_wf] = init_FlowModel(wf, wt);
    wf->t = wt->t;
};
/* -- end add_WindTurbine --------------------------------------------------- */

FlowModel* get_FlowModel(LagSolver *wf, WindTurbine *wt) {
   return wf->fms[wt->i_wf];
};
/* -- end get_FlowSolver ---------------------------------------------------- */

WakeModel* get_WakeModel(LagSolver *wf, WindTurbine *wt) {
   return wf->wms[wt->i_wf];
};
/* -- end get_WakeSolver ---------------------------------------------------- */

void is_freestream(LagSolver *wf,  WindTurbine *wt) { 
    int i_wm;

    wt->is_fs = 0;
    
    for (i_wm = 0;          i_wm < wt->i_wf; i_wm++) {
        wt->is_fs += is_waked_by(wf->wms[i_wm], wt);
    }
    
    for (i_wm = wt->i_wf+1; i_wm < wf->n_wt; i_wm++) {
        wt->is_fs += is_waked_by(wf->wms[i_wm], wt);
    }

    wt->is_fs = (wt->is_fs==0);
}
