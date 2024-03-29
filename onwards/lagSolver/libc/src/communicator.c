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

void interp_vec_FlowModel(LagSolver *wf, double *x_vec, double *y_vec, 
                                 int n_tot, double *u_vec, int filt, int i_wt_exclude) {
    int i;
    double *x_interp = VEC(2);
    double *u_interp = VEC(2);

    for (i = 0; i < n_tot; i++) { 
        x_interp[0] = x_vec[i];
        x_interp[1] = y_vec[i];
        interp_FlowModel_all(wf, x_interp, wf->t, filt, u_interp, i_wt_exclude);
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
