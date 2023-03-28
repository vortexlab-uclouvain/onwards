#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "macro.h"  

void project(double *axis, double **polygon, double *buffer) {

    int i;
    buffer[0] =  1e16;
    buffer[1] = -1e16;
    double dot;

    for (i = 0; i < 4; i++) {
        dot = DOT(axis, polygon[i]);
        buffer[0] = fmin(buffer[0], dot);
        buffer[1] = fmax(buffer[1], dot);
    }
}

#define FREE_INTERSECT() free(axis); free(proj_a); free(proj_b);

#define LOOP_INTERSECT(a_) \
    for (i = 0; i < 4; i++) {\
        axis[0] = - a_[(i+1)%4][1] + a_[(i)%4][1];\
        axis[1] =   a_[(i+1)%4][0] - a_[(i)%4][0]; \
        norm = NORM(axis[0], axis[1]);\
        axis[0] /= norm;\
        axis[1] /= norm;\
        project(axis, a, proj_a);\
        project(axis, b, proj_b);\
        if (!((proj_a[0] <= proj_b[1]) && (proj_b[0] <= proj_a[1]))) {\
            FREE_INTERSECT();\
            return 0;\
        }\
    }

int intersect(double **a, double **b) {
    int i;
    double norm;
    double *axis, *proj_a, *proj_b;

    axis   = VEC(2);
    proj_a = VEC(2);
    proj_b = VEC(2);

    LOOP_INTERSECT(a);
    LOOP_INTERSECT(b);

    FREE_INTERSECT();

    return 1;
}

#undef LOOP_INTERSECT
#undef FREE_INTERSECT


int in_triangle(double *x, double *a, double *b, double *c){
	double s1 = c[1] - a[1];
	double s2 = c[0] - a[0];
	double s3 = b[1] - a[1];
	double s4 = x[1] - a[1];

	double w1 = (a[0] * s1 + s4 * s2 - x[0] * s1) / (s3 * s2 - (b[0]-a[0]) * s1);
	double w2 = (s4- w1 * s3) / s1;
	return ( (w1 >= 0) & (w2 >= 0) & ((w1 + w2) <= 1) );
}

int in_quad(double *x, double *a, double *b, double *c, double *d) {
    if (in_triangle(x, a, b, c))  {return 1;}
    else                          {return in_triangle(x, c, d, a);}
}

void line_intersect(double mx, double px, double my, double py, double *x) {
    x[0] = -(py+px/mx)/(my-1/mx);
    x[1] = my *  x[0] + py;
}